"""基于预训练 LLM + PEFT (LoRA) 的 SART 训练器。

本模块将 trainer.py 中的 SART 训练逻辑适配到预训练 LLM + LoRA 场景。

与原始 trainer.py 的核心差异：
  1. 模型接口：SmallGPT(x) → HuggingFace CausalLM(input_ids, attention_mask).logits
  2. 梯度向量：全量参数 (~7B) → 仅 LoRA 参数 (~8M)
  3. 优化器：对所有参数使用 AdamW → 只对 requires_grad=True 的 LoRA 参数优化
  4. 梯度累积：batch_size=2, grad_accum_steps=8 → 等效批次大小=16
  5. 精度：float32 → bfloat16（基础模型），float32（梯度运算）
  6. warmup：随机初始化需要 5+ epochs 预热 → 预训练模型 1 epoch 即可

训练流程（三阶段，对应 train_sart_peft）：
  Phase 1 - 标准预热：用加权的标准 cross-entropy 训练，使模型适应任务格式
  Phase 2 - ShortcutScore 计算：逐样本计算梯度，得到每个训练样本的捷径倾向分数
  Phase 3 - SART 训练：带重加权 + 梯度手术的主训练阶段

ShortcutScore 在预训练 LLM 场景下的语义：
  - g_V（验证梯度）：代表"强化预训练推理能力、改善真实推理泛化"的方向
  - A(s) = cos(g_s, g_V)：低值表示样本 s 的梯度与推理泛化方向不一致（捷径信号）
  - R(s) = ||g_ans|| / (||g_ans|| + ||g_reason||)：高值表示模型只关注"答什么"而非"怎么推理"
  这与论文的理论分析在预训练 LLM 场景下具有明确的物理意义。
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.config import LLMConfig as LC
from src.data import ReasoningDataset, get_dataloader, _make_collate
from src.methods import (masked_ce_loss, compute_shortcut_score,
                         compute_shortcut_scores_batched,
                         compute_sample_weight, apply_gradient_surgery)
from src.model_peft import (get_peft_grad_vector, set_peft_grad_vector,
                             peft_forward)


# ============================================================================
# 基础工具：PEFT 场景下的梯度计算
# ============================================================================

def compute_validation_gradient_peft(model, val_loader, device, pad_id):
    """计算验证集上的平均梯度，仅在 LoRA 参数上累积。

    这是 methods.py compute_validation_gradient() 的 PEFT 适配版本。
    区别：
      - 使用 peft_forward() 替代 SmallGPT 的 model(x) 调用
      - 梯度向量 g_V 仅包含 LoRA 参数梯度（~8M 维），而非全量参数

    g_V 的语义（预训练 LLM 场景）：
      验证集始终使用真实推理标签（无捷径注入），因此 g_V 代表
      "引导预训练模型在真实推理任务上泛化"的参数更新方向。
      训练样本梯度与 g_V 的余弦相似度低，说明该样本在推动模型
      偏离真实推理能力的方向——即捷径信号。

    Args:
        model:      PEFT 模型（基础模型冻结，LoRA 可训练）
        val_loader: 验证集 DataLoader（批次包含 input_ids, target_ids, loss_mask）
        device:     计算设备
        pad_id:     填充 token 的 id（用于构造 attention_mask）

    Returns:
        g_V: (D,) float32 平均验证梯度向量，D = LoRA 参数总数
    """
    model.train()  # 需要 train 模式以使 grad 可用（即使在验证时）
    g_V = None
    n_batches = 0

    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        loss_mask = batch['loss_mask'].to(device)

        model.zero_grad()

        # 使用 peft_forward 代替 SmallGPT 的直接调用
        logits = peft_forward(model, input_ids, pad_id)   # (B, T, V)

        # 仅在 loss_mask=1 的位置（推理+答案 token）计算损失
        loss = masked_ce_loss(logits, target_ids, loss_mask)
        loss.backward()

        # 只收集 LoRA 参数的梯度（基础模型参数梯度为 None，被自动跳过）
        grad = get_peft_grad_vector(model)   # (D,) float32

        if g_V is None:
            g_V = grad.clone()
        else:
            g_V = g_V + grad   # 使用加法避免 in-place 操作

        n_batches += 1

    model.zero_grad()  # 清理验证梯度，避免污染后续训练

    if g_V is None:
        raise RuntimeError("[PEFT] 验证集为空，无法计算 g_V。")

    return g_V / max(n_batches, 1)


def compute_sample_gradients_peft(model, input_ids, target_ids,
                                   loss_mask, answer_mask, reasoning_mask,
                                   device, pad_id):
    """计算单个样本的全量、答案、推理梯度（仅 LoRA 参数）。

    这是 methods.py compute_sample_gradients() 的 PEFT 适配版本。
    对每个样本执行 3 次 backward，分别得到：
      - g_full:   完整损失（loss_mask 范围）的梯度
      - g_ans:    仅答案 token 损失的梯度
      - g_reason: 仅推理 token 损失的梯度

    关键实现细节：
      - retain_graph=True 允许多次对同一 logits backward，避免重复前向传播
      - 梯度以 float32 存储，确保 ShortcutScore 计算的数值稳定性
      - 每次 backward 前调用 model.zero_grad() 清除累积梯度

    Args:
        model:          PEFT 模型
        input_ids:      (T,) 单样本输入 token ids
        target_ids:     (T,) 单样本目标 token ids（input_ids 左移 1）
        loss_mask:      (T,) 完整损失掩码（推理+答案位置为 1）
        answer_mask:    (T,) 答案 token 掩码
        reasoning_mask: (T,) 推理 token 掩码
        device:         计算设备
        pad_id:         填充 token id

    Returns:
        g_full:   (D,) 完整梯度向量
        g_ans:    (D,) 答案梯度向量
        g_reason: (D,) 推理梯度向量
    """
    # 扩展为批次维度 (1, T)
    inp = input_ids.unsqueeze(0).to(device)
    tgt = target_ids.unsqueeze(0).to(device)
    lm = loss_mask.unsqueeze(0).to(device)
    am = answer_mask.unsqueeze(0).to(device)
    rm = reasoning_mask.unsqueeze(0).to(device)

    # 前向传播一次，retain_graph=True 以支持后续多次 backward
    model.zero_grad()
    logits = peft_forward(model, inp, pad_id)   # (1, T, V)

    # ---- 1. 完整损失梯度 g_full ----
    full_loss = masked_ce_loss(logits, tgt, lm)
    full_loss.backward(retain_graph=True)
    g_full = get_peft_grad_vector(model).clone()   # 克隆避免后续覆盖

    # ---- 2. 答案 token 梯度 g_ans ----
    model.zero_grad()
    if am.sum() > 0:
        # 有答案 token：计算仅答案位置的损失
        ans_loss = masked_ce_loss(logits, tgt, am)
        ans_loss.backward(retain_graph=True)
        g_ans = get_peft_grad_vector(model).clone()
    else:
        # 没有答案 token 的样本（异常情况）：答案梯度设为零
        g_ans = torch.zeros_like(g_full)

    # ---- 3. 推理 token 梯度 g_reason ----
    model.zero_grad()
    if rm.sum() > 0:
        # 最后一次 backward 不需要 retain_graph（释放计算图节省显存）
        reason_loss = masked_ce_loss(logits, tgt, rm)
        reason_loss.backward()  # 不保留计算图
        g_reason = get_peft_grad_vector(model).clone()
    else:
        g_reason = torch.zeros_like(g_full)

    model.zero_grad()   # 清理，避免梯度泄漏到下一步

    return g_full, g_ans, g_reason


# ============================================================================
# 阶段一：标准预热微调（Standard Fine-Tuning with PEFT）
# ============================================================================

def train_standard_peft(model, dataset, tokenizer, epochs=None, device=None,
                        verbose=True, cfg=None):
    """标准监督微调（作为 SART 的预热阶段）。

    使用普通 cross-entropy 损失对 LoRA 参数进行微调，不做任何捷径识别。
    此阶段目的是让模型适应任务的输入输出格式，建立基础的梯度对齐参考点。

    预训练模型只需较少 epochs（默认 LC.warmup_epochs=1）即可达到合理的任务性能，
    因为预训练阶段已经学到了推理的基本模式。

    训练使用梯度累积（grad_accum_steps）以在小 batch_size 下等效大批次训练，
    避免单步 batch 过小导致梯度噪声过大。

    Args:
        model:     PEFT 模型
        dataset:   数据集字典，包含 'train', 'val' 等键（ReasoningDataset 实例）
        tokenizer: 分词器（用于获取 pad_id）
        epochs:    训练 epochs 数，默认 LC.warmup_epochs
        device:    计算设备，默认 LC.device
        verbose:   是否打印训练进度
        cfg:       可选的超参数字典，用于覆盖 LLMConfig 默认值
    """
    _c = cfg or {}
    _epochs = epochs if epochs is not None else _c.get('warmup_epochs', LC.warmup_epochs)
    _lr = _c.get('lr', LC.lr)
    _wd = _c.get('weight_decay', LC.weight_decay)
    _bs = _c.get('batch_size', LC.batch_size)
    _grad_accum = _c.get('grad_accum_steps', LC.grad_accum_steps)
    _device = device or LC.device

    pad_id = tokenizer.pad_token_id

    # 只优化 requires_grad=True 的 LoRA 参数（基础模型参数已冻结）
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=_lr, weight_decay=_wd
    )

    # 余弦退火调度器：在所有 optimizer.step() 调用上（非每个 batch）
    total_optimizer_steps = _epochs * (len(dataset['train']) // (_bs * _grad_accum) + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_optimizer_steps, 1), eta_min=1e-6
    )

    train_loader = get_dataloader(dataset['train'], batch_size=_bs, shuffle=True)

    model.train()
    for epoch in range(_epochs):
        total_loss = 0.0
        n_tokens = 0
        optimizer.zero_grad()  # 初始清零（梯度累积起点）

        for step_idx, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(_device)
            target_ids = batch['target_ids'].to(_device)
            loss_mask = batch['loss_mask'].to(_device)

            # 前向传播，获取 logits
            logits = peft_forward(model, input_ids, pad_id)   # (B, T, V)

            # 计算 token 级别的损失均值（在 loss_mask=1 的位置）
            loss = masked_ce_loss(logits, target_ids, loss_mask)

            # 梯度累积：将损失除以累积步数，使梯度量级等效于 batch_size*grad_accum 的批次
            loss_scaled = loss / _grad_accum
            loss_scaled.backward()

            total_loss += loss.item()
            n_tokens += loss_mask.sum().item()

            # 每 grad_accum_steps 步执行一次参数更新
            if (step_idx + 1) % _grad_accum == 0:
                # 梯度裁剪：防止 LoRA 参数梯度爆炸（特别是训练初期）
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()  # 重置梯度，开始下一个累积周期

        # 处理最后一个不完整的累积步（若 len(train_loader) 不是 grad_accum 的整数倍）
        if (len(train_loader) % _grad_accum) != 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if verbose:
            avg_loss = total_loss / max(len(train_loader), 1)
            print(f"    Warmup Epoch {epoch+1}/{_epochs} | loss={avg_loss:.4f}")


# ============================================================================
# 阶段二：ShortcutScore 批量计算
# ============================================================================

def _compute_sample_scores_peft(model, dataset, device, pad_id, cfg=None):
    """为训练集样本批量计算 ShortcutScore。

    为每个训练样本计算三种梯度（g_full, g_ans, g_reason），
    再与验证梯度 g_V 对比，得到 ShortcutScore。

    计算开销：
      - 每个样本需要 1 次前向 + 3 次反向传播
      - LoRA 梯度向量约 32MB（rank=16），完全在显存内
      - 500 个样本约需 5-15 分钟（取决于序列长度）

    为控制时间开销，最多处理 LC.score_max_samples 个样本；
    其余样本使用平均分代替（近似但节省大量时间）。

    Args:
        model:   PEFT 模型（处于 train 模式）
        dataset: 包含 'train'、'val' 的数据集字典
        device:  计算设备
        pad_id:  填充 token id
        cfg:     可选超参数覆盖

    Returns:
        all_scores:     所有训练样本的 ShortcutScore 列表（长度 = len(train)）
        collected_data: 用于可视化和调试的统计数据字典
        g_V:            验证梯度向量（用于后续 Phase 3 的梯度手术）
    """
    _c = cfg or {}
    _bs = _c.get('batch_size', LC.batch_size)
    _max_samples = _c.get('score_max_samples', LC.score_max_samples)

    # 超参数（可通过 cfg 覆盖）
    hp_alpha = _c.get('alpha', LC.alpha)
    hp_beta = _c.get('beta', LC.beta)
    hp_tau_A = _c.get('tau_A', LC.tau_A)
    hp_tau_R = _c.get('tau_R', LC.tau_R)

    # ---- 计算验证梯度 g_V ----
    val_loader = get_dataloader(dataset['val'], batch_size=_bs, shuffle=False)
    print("  [Score] 计算验证梯度 g_V...")
    g_V = compute_validation_gradient_peft(model, val_loader, device, pad_id)
    print(f"  [Score] g_V 范数: {g_V.norm().item():.4f}, 维度: {g_V.shape[0]:,}")

    # ---- 逐样本计算 ShortcutScore ----
    # 使用 batch_size=1 的 DataLoader 逐样本迭代（per-sample 梯度必须单独计算）
    score_loader = get_dataloader(dataset['train'], batch_size=1, shuffle=False)

    scores_all = []
    collected_data = {
        'scores': [],
        'is_shortcut': [],
        'alignments': [],
        'concentrations': [],
    }

    n_to_score = min(_max_samples, len(dataset['train']))
    print(f"  [Score] 对 {n_to_score}/{len(dataset['train'])} 个训练样本计分...")

    model.train()  # 确保在 train 模式（梯度可用）
    for i, batch in enumerate(score_loader):
        if i >= n_to_score:
            break  # 达到最大计分数量，停止

        if i % 50 == 0:
            print(f"    进度: {i}/{n_to_score}")

        # 取出单个样本（batch_size=1，去掉批次维度）
        inp = batch['input_ids'][0]
        tgt = batch['target_ids'][0]
        lm = batch['loss_mask'][0]
        am = batch['answer_mask'][0]
        rm = batch['reasoning_mask'][0]

        # 计算三种梯度（仅 LoRA 参数）
        g_full, g_ans, g_reason = compute_sample_gradients_peft(
            model, inp, tgt, lm, am, rm, device, pad_id
        )

        # 计算 ShortcutScore 及各分量
        S, B_val, C_val, A_val, R_val = compute_shortcut_score(
            g_full, g_ans, g_reason, g_V,
            alpha=hp_alpha, beta=hp_beta, tau_A=hp_tau_A, tau_R=hp_tau_R
        )

        scores_all.append(S)

        # 收集调试数据
        collected_data['scores'].append(S)
        collected_data['is_shortcut'].append(batch['is_shortcut'][0].item())
        collected_data['alignments'].append(A_val)
        collected_data['concentrations'].append(R_val)

    # 未计分样本使用平均分代替（合理近似）
    avg_score = sum(scores_all) / max(len(scores_all), 1)
    n_remaining = len(dataset['train']) - len(scores_all)
    all_scores = scores_all + [avg_score] * n_remaining

    print(f"  [Score] 计分完成。平均 ShortcutScore: {avg_score:.4f}")
    if collected_data['is_shortcut']:
        # 验证 ShortcutScore 的区分能力
        sc_scores = [s for s, is_sc in zip(collected_data['scores'],
                                            collected_data['is_shortcut']) if is_sc > 0.5]
        non_sc_scores = [s for s, is_sc in zip(collected_data['scores'],
                                                collected_data['is_shortcut']) if is_sc <= 0.5]
        if sc_scores:
            print(f"  [Score] 捷径样本平均分: {sum(sc_scores)/len(sc_scores):.4f} "
                  f"({len(sc_scores)} 个)")
        if non_sc_scores:
            print(f"  [Score] 非捷径样本平均分: {sum(non_sc_scores)/len(non_sc_scores):.4f} "
                  f"({len(non_sc_scores)} 个)")

    return all_scores, collected_data, g_V


# ============================================================================
# 阶段三：SART 主训练（重加权 + 梯度手术）
# ============================================================================

def _apply_batch_gradient_surgery_peft(model, batch, weights, g_V, device, pad_id,
                                        use_gradient_surgery, cfg=None):
    """对一个 mini-batch 应用 SART 梯度手术并返回（修改后的）损失。

    本函数完成一次 SART 训练步的核心操作：
      1. 前向传播，计算加权损失
      2. backward 得到当前 LoRA 梯度
      3. （可选）对梯度做投影和抑制（Gradient Surgery）
      4. 将修改后的梯度写回参数，供 optimizer.step() 使用

    梯度手术的实现方式（与 methods.py 一致）：
      对整个 batch 的聚合梯度做全局投影，而非逐样本：
        g_mod = g_batch - γ * (g_batch·g_V / ||g_V||²) * g_V   （若对齐度低）
        g_mod = g_mod - ρ * g_ans_avg                            （若答案梯度集中）

    注意：本函数不调用 optimizer.step()，只修改 .grad；
    调用方负责梯度累积控制和 optimizer.step()。

    Args:
        model:                PEFT 模型
        batch:                DataLoader 输出的批次字典
        weights:              (B,) 每个样本的重加权系数 w(s)
        g_V:                  (D,) 当前验证梯度向量
        device:               计算设备
        pad_id:               填充 token id
        use_gradient_surgery: 是否应用梯度手术
        cfg:                  可选超参数覆盖

    Returns:
        loss_val: 标量，未缩放的加权损失值（用于日志记录）
    """
    _c = cfg or {}
    hp_gamma = _c.get('gamma', LC.gamma)
    hp_rho = _c.get('rho', LC.rho)
    hp_tau_A = _c.get('tau_A', LC.tau_A)

    input_ids = batch['input_ids'].to(device)
    target_ids = batch['target_ids'].to(device)
    loss_mask = batch['loss_mask'].to(device)
    answer_mask = batch['answer_mask'].to(device)

    # ---- 前向传播 ----
    logits = peft_forward(model, input_ids, pad_id)   # (B, T, V)

    B, T, V = logits.shape

    # ---- 计算加权损失 ----
    # 按 token 计算 cross-entropy，再按样本聚合后加权
    loss_per_token = F.cross_entropy(
        logits.reshape(-1, V),
        target_ids.reshape(-1),
        reduction='none'
    ).reshape(B, T)

    masked_loss = loss_per_token * loss_mask
    # 每个样本的平均损失（在有效 token 上）
    per_sample_loss = masked_loss.sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1.0)
    # 用重加权系数 w(s) 加权后取均值
    weighted_loss = (per_sample_loss * weights.to(device)).mean()

    loss_val = weighted_loss.item()

    # ---- 反向传播，获取 LoRA 梯度 ----
    weighted_loss.backward(retain_graph=True)   # retain_graph 以便后续计算答案梯度
    g_batch = get_peft_grad_vector(model).clone()  # (D,) 当前批次梯度

    if not use_gradient_surgery:
        # 不做手术：梯度已在 .grad 中，直接返回
        # 但仍需释放计算图
        del logits
        return loss_val

    # ---- 计算答案梯度（用于答案梯度抑制）----
    # 对整个 batch 的答案 token 位置计算平均梯度
    model.zero_grad()
    ans_masked_loss = loss_per_token * answer_mask
    ans_per_sample = ans_masked_loss.sum(dim=1) / answer_mask.sum(dim=1).clamp(min=1.0)
    ans_weighted = (ans_per_sample * weights.to(device)).mean()

    if answer_mask.sum() > 0:
        ans_weighted.backward()
        g_ans_batch = get_peft_grad_vector(model).clone()
    else:
        g_ans_batch = torch.zeros_like(g_batch)

    # ---- 计算全局 ShortcutScore 指标（用于判断是否需要手术）----
    # 使用批次聚合梯度而非逐样本梯度（效率更高）
    norm_full = g_batch.norm()
    norm_V = g_V.norm()
    if norm_full > 1e-10 and norm_V > 1e-10:
        A_batch = (g_batch @ g_V / (norm_full * norm_V)).item()
    else:
        A_batch = 0.0
    B_val = max(0.0, hp_tau_A - A_batch)

    norm_ans = g_ans_batch.norm().item()
    norm_full_val = g_batch.norm().item()
    # 批次级别的答案梯度集中度（近似）
    R_batch = norm_ans / (norm_ans + max(norm_full_val - norm_ans, 1e-10))
    C_val = max(0.0, R_batch - _c.get('tau_R', LC.tau_R))

    # ---- 梯度手术：修改 g_batch ----
    g_modified = apply_gradient_surgery(
        g_batch, g_ans_batch, g_V, B_val, C_val,
        gamma=hp_gamma, rho=hp_rho
    )

    # ---- 将修改后的梯度写回 LoRA 参数的 .grad ----
    model.zero_grad()
    set_peft_grad_vector(model, g_modified)

    return loss_val


# ============================================================================
# 主函数：SART 完整训练流程
# ============================================================================

def train_sart_peft(model, dataset, tokenizer,
                    use_reweighting=True, use_gradient_surgery=True,
                    device=None, verbose=True, cfg=None):
    """使用预训练 LLM + PEFT 进行完整的 SART 训练。

    三阶段训练流程：
      Phase 1: 标准预热 (warmup_epochs)
        - 使用普通 cross-entropy 微调 LoRA 参数
        - 使预训练模型适应任务格式
        - 建立梯度对齐的基准参考点

      Phase 2: ShortcutScore 计算
        - 计算验证集梯度 g_V（代表"真实推理泛化"方向）
        - 对每个训练样本逐一计算梯度，得到 ShortcutScore S(s)
        - 根据 S(s) 计算重加权系数 w(s) = exp(-λ * S(s))

      Phase 3: SART 主训练 (epochs - warmup_epochs)
        - 使用 w(s) 加权的损失进行训练
        - 对高捷径分数样本的梯度做投影（去除不可迁移分量）
        - 定期刷新 g_V（每 val_grad_interval 个 epoch）

    与 trainer.py train_our_method() 的对应关系：
      train_standard() → train_standard_peft()
      _compute_sample_scores() → _compute_sample_scores_peft()
      Phase 3 训练循环 → 本函数 Phase 3 循环（PEFT 适配版）

    Args:
        model:                PEFT 封装的预训练 LLM
        dataset:              数据集字典，包含 'train', 'val' 等键
        tokenizer:            分词器（提供 pad_token_id）
        use_reweighting:      是否启用 ShortcutScore 重加权
        use_gradient_surgery: 是否启用梯度手术
        device:               计算设备
        verbose:              是否打印详细进度
        cfg:                  超参数字典（覆盖 LLMConfig 默认值）

    Returns:
        collected_data: 包含 scores, alignments, concentrations 等调试信息的字典
    """
    _c = cfg or {}
    _epochs = _c.get('epochs', LC.epochs)
    _warmup = _c.get('warmup_epochs', LC.warmup_epochs)
    _bs = _c.get('batch_size', LC.batch_size)
    _grad_accum = _c.get('grad_accum_steps', LC.grad_accum_steps)
    _lr = _c.get('lr', LC.lr)
    _wd = _c.get('weight_decay', LC.weight_decay)
    _val_grad_interval = _c.get('val_grad_interval', LC.val_grad_interval)
    _device = device or LC.device

    pad_id = tokenizer.pad_token_id
    main_epochs = _epochs - _warmup

    # =====================================================================
    # Phase 1: 标准预热微调
    # =====================================================================
    if verbose:
        print(f"\n  [SART-PEFT] Phase 1: 预热微调 ({_warmup} epochs)")
    train_standard_peft(model, dataset, tokenizer,
                        epochs=_warmup, device=_device, verbose=verbose, cfg=cfg)

    # =====================================================================
    # Phase 2: 计算 ShortcutScore
    # =====================================================================
    if verbose:
        print(f"\n  [SART-PEFT] Phase 2: 计算 ShortcutScore...")
    sample_scores, collected_data, g_V = _compute_sample_scores_peft(
        model, dataset, _device, pad_id, cfg=cfg
    )

    # 根据 ShortcutScore 计算重加权系数
    hp_lambda = _c.get('lambda_', LC.lambda_)
    sample_weights = []
    for S in sample_scores:
        if use_reweighting:
            w = compute_sample_weight(S, lambda_=hp_lambda).item()
        else:
            w = 1.0   # 不重加权：所有样本等权
        sample_weights.append(w)

    if verbose:
        n_scored = min(_c.get('score_max_samples', LC.score_max_samples), len(sample_scores))
        avg_s = sum(sample_scores[:n_scored]) / max(n_scored, 1)
        avg_w = sum(sample_weights) / max(len(sample_weights), 1)
        n_downweighted = sum(1 for w in sample_weights if w < 0.5)
        print(f"    均值 ShortcutScore: {avg_s:.4f}")
        print(f"    均值重加权系数: {avg_w:.4f}")
        print(f"    被显著压制的样本数 (w < 0.5): {n_downweighted}/{len(sample_weights)}")

    # =====================================================================
    # Phase 3: SART 主训练（重加权 + 梯度手术）
    # =====================================================================
    if verbose:
        print(f"\n  [SART-PEFT] Phase 3: SART 主训练 ({main_epochs} epochs)")
        print(f"    重加权: {use_reweighting}, 梯度手术: {use_gradient_surgery}")
        print(f"    梯度累积步数: {_grad_accum} (等效批次大小 = {_bs * _grad_accum})")

    # 将重加权系数附加到训练样本（避免修改原始数据集）
    weighted_samples = []
    for i, s in enumerate(dataset['train'].samples):
        ws = dict(s)
        ws['weight'] = sample_weights[i]
        weighted_samples.append(ws)

    weighted_ds = ReasoningDataset(weighted_samples, pad_id=pad_id)
    train_loader = get_dataloader(weighted_ds, batch_size=_bs, shuffle=True)
    val_loader = get_dataloader(dataset['val'], batch_size=_bs, shuffle=False)

    # Phase 3 使用略低的学习率（微调时期，避免破坏预热阶段学到的推理模式）
    phase3_lr = _lr * _c.get('phase3_lr_factor', 0.5)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=phase3_lr, weight_decay=_wd
    )
    total_opt_steps = main_epochs * (len(train_loader) // _grad_accum + 1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_opt_steps, 1), eta_min=1e-7
    )

    model.train()
    for epoch in range(main_epochs):
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        # 定期刷新验证梯度 g_V（保持对齐信号的时效性）
        if use_gradient_surgery and epoch % _val_grad_interval == 0:
            if verbose:
                print(f"    Epoch {epoch+1}: 刷新验证梯度 g_V...")
            g_V = compute_validation_gradient_peft(model, val_loader, _device, pad_id)
            model.train()  # 切回 train 模式

        for step_idx, batch in enumerate(train_loader):
            weights = batch.get('weight', torch.ones(batch['input_ids'].size(0)))

            # 核心：计算加权损失 + 梯度手术
            # 函数内部已完成 backward 和 set_peft_grad_vector
            loss_val = _apply_batch_gradient_surgery_peft(
                model, batch, weights, g_V, _device, pad_id,
                use_gradient_surgery, cfg=cfg
            )

            total_loss += loss_val
            n_batches += 1

            # 梯度累积控制：每 grad_accum 步做一次参数更新
            if (step_idx + 1) % _grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        # 处理最后不完整的累积步
        if (len(train_loader) % _grad_accum) != 0:
            torch.nn.utils.clip_grad_norm_(
                filter(lambda p: p.requires_grad, model.parameters()),
                max_norm=1.0
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if verbose:
            avg_loss = total_loss / max(n_batches, 1)
            print(f"    SART Epoch {epoch+1}/{main_epochs} | "
                  f"加权损失={avg_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    if verbose:
        print(f"\n  [SART-PEFT] 训练完成。")

    return collected_data


# ============================================================================
# 快捷接口：仅重加权（消融实验用）
# ============================================================================

def train_reweighting_only_peft(model, dataset, tokenizer, device=None,
                                 verbose=True, cfg=None):
    """仅启用重加权，不使用梯度手术（消融实验）。"""
    return train_sart_peft(
        model, dataset, tokenizer,
        use_reweighting=True,
        use_gradient_surgery=False,
        device=device, verbose=verbose, cfg=cfg
    )


def train_surgery_only_peft(model, dataset, tokenizer, device=None,
                             verbose=True, cfg=None):
    """仅启用梯度手术，不使用重加权（消融实验）。"""
    return train_sart_peft(
        model, dataset, tokenizer,
        use_reweighting=False,
        use_gradient_surgery=True,
        device=device, verbose=verbose, cfg=cfg
    )
