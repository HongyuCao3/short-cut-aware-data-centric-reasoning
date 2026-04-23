#!/usr/bin/env python3
"""
预训练 LLM + PEFT (LoRA) 的 SART 实验入口。

解决原始实验的核心问题：
  原始 run_all.py 使用随机初始化的 SmallGPT (~277K 参数)，与论文声称的
  "针对预训练 LLM 捷径推理"的场景存在根本性概念不对齐。

  本脚本使用 7B 级别预训练 LLM（Qwen2.5-7B 或 Llama-3.1-8B）+ LoRA 微调：
    1. 预训练知识赋予梯度信号真实的推理语义
    2. LoRA 将可训练参数从 7B 压缩到约 8M，使梯度手术在显存上可行
    3. 4-bit 量化（QLoRA）使整个实验可在单张 24GB GPU 上运行

使用方式：
  # 基础运行（默认 Qwen2.5-7B-Instruct，GSM8K 数据集）
  python run_llm.py

  # 指定模型
  LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct python run_llm.py

  # 指定数据集（gsm8k / math / both）
  DATASET=math python run_llm.py

  # 消融实验（只用重加权，不用梯度手术）
  ABLATION=reweight_only python run_llm.py

  # 不使用量化（需要 ~14GB VRAM）
  NO_QUANT=1 python run_llm.py

显存需求估算（7B 模型）：
  4-bit QLoRA:  ~6-8 GB（训练时 + 梯度）
  8-bit:        ~12-14 GB
  bfloat16:     ~18-20 GB

依赖安装：
  pip install transformers peft bitsandbytes accelerate datasets
"""

import os
import sys
import time
import random
import numpy as np
import torch

# 将项目根目录添加到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import LLMConfig as LC
from src.model_peft import create_peft_model, count_peft_parameters
from src.trainer_peft import (train_sart_peft, train_standard_peft,
                               train_reweighting_only_peft, train_surgery_only_peft)
from src.data import get_dataloader
from src.evaluate import run_full_evaluation_nl


# ============================================================================
# 工具函数
# ============================================================================

def set_seed(seed=LC.seed):
    """设置全局随机种子，确保实验可重复。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset(dataset_name, tokenizer):
    """加载指定的真实世界推理数据集。

    Args:
        dataset_name: 'gsm8k' 或 'math'
        tokenizer:    已加载的 tokenizer（LLM 对应的分词器）

    Returns:
        数据集字典，包含 'train', 'val', 'test_clean', 'test_perturbed'
    """
    from src.data_realworld import generate_gsm8k_dataset, generate_math_dataset_realworld

    if dataset_name.lower() == 'gsm8k':
        print(f"  加载 GSM8K 数据集...")
        return generate_gsm8k_dataset(tokenizer, seed=LC.seed)
    elif dataset_name.lower() in ('math', 'math_lighteval'):
        print(f"  加载 MATH 数据集...")
        return generate_math_dataset_realworld(tokenizer, seed=LC.seed)
    else:
        raise ValueError(f"未知数据集: {dataset_name}。支持: 'gsm8k', 'math'")


def _accuracy_on_split(model, split, tokenizer, max_gen=64,
                        batch_size=4, max_samples=None, verbose=False):
    """Greedy-decode each sample in `split`, parse the predicted answer,
    and compare against the stored `answer_value`. Returns (correct, total).

    PEFT-native eval: uses HuggingFace CausalLM's `.generate()` interface
    (do_sample=False, max_new_tokens=max_gen), not SmallGPT's signature.
    Batches prompts with left-padding so attention masks remain correct
    for mixed-length prefixes.
    """
    import math as _math
    from src.data_realworld import parse_gsm8k_answer

    device = LC.device
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id

    samples = split.samples if hasattr(split, 'samples') else list(split)
    if max_samples is not None and max_samples < len(samples):
        samples = samples[:max_samples]

    # Switch to left-padding for generation so attention_mask lines up
    # with the right-side prefix tokens. Saved and restored.
    _prev_pad = tokenizer.padding_side
    tokenizer.padding_side = "left"

    correct, total, skipped = 0, 0, 0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(samples), batch_size):
            chunk = samples[i:i + batch_size]
            # Build prompt-only sequences using stored prompt_len.
            prompt_token_lists = []
            gt_values = []
            for s in chunk:
                plen = int(s['prompt_len']) if 'prompt_len' in s else len(s['input_ids'])
                prompt_token_lists.append(list(s['input_ids'])[:plen])
                gt = s.get('answer_value', float('nan'))
                if hasattr(gt, 'item'):
                    gt = gt.item()
                gt_values.append(gt)

            max_prompt = max(len(p) for p in prompt_token_lists)
            input_ids = []
            attention_mask = []
            for p in prompt_token_lists:
                pad_n = max_prompt - len(p)
                input_ids.append([pad_id] * pad_n + p)
                attention_mask.append([0] * pad_n + [1] * len(p))
            input_ids = torch.tensor(input_ids, device=device)
            attention_mask = torch.tensor(attention_mask, device=device)

            gen = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_gen,
                do_sample=False,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
            # Only look at the newly generated suffix.
            for b_idx, out in enumerate(gen):
                new_tokens = out[input_ids.shape[1]:].cpu().tolist()
                text = tokenizer.decode(new_tokens, skip_special_tokens=True)
                pred = parse_gsm8k_answer(text)
                gt = gt_values[b_idx]
                if gt is None or (isinstance(gt, float) and _math.isnan(gt)):
                    skipped += 1
                    continue
                if pred is not None and abs(pred - gt) < 0.01:
                    correct += 1
                total += 1
            if verbose and (i // batch_size) % 20 == 0 and i > 0:
                print(f'      decoded {i + len(chunk)}/{len(samples)} '
                      f'(acc so far = {correct / max(total, 1):.3f})',
                      flush=True)

    tokenizer.padding_side = _prev_pad
    model.train()
    return correct, total, skipped


def evaluate_peft_model(model, dataset, tokenizer, verbose=True,
                         max_eval_samples=200, max_gen=64, batch_size=4):
    """Accuracy + robustness on test_clean / test_perturbed.

    Subsamples each split to at most `max_eval_samples` for a fast turnaround
    (PEFT generation at bs=4 on an H100 lands around 2-3 s/batch, so 200
    samples per split ≈ 2 minutes). Robustness is absolute accuracy on the
    perturbed split.
    """
    results = {}

    # Robustness-gap PPL (cheap loss-based signal; still useful when
    # accuracy is low).
    from src.model_peft import peft_forward
    from src.methods import masked_ce_loss
    import math

    def _eval_loss(split_name):
        loader = get_dataloader(dataset[split_name], batch_size=LC.batch_size,
                                shuffle=False)
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                inp = batch['input_ids'].to(LC.device)
                tgt = batch['target_ids'].to(LC.device)
                lm = batch['loss_mask'].to(LC.device)
                logits = peft_forward(model, inp, tokenizer.pad_token_id)
                loss = masked_ce_loss(logits, tgt, lm)
                total_loss += loss.item()
                n += 1
        return total_loss / max(n, 1)

    results['clean_loss']     = _eval_loss('test_clean')
    results['perturbed_loss'] = _eval_loss('test_perturbed')
    results['clean_ppl']      = math.exp(min(results['clean_loss'], 20))
    results['perturbed_ppl']  = math.exp(min(results['perturbed_loss'], 20))
    results['robustness_gap'] = results['perturbed_loss'] - results['clean_loss']

    # Decode-based accuracy: the honest metric that captures shortcut flip.
    print(f'  [eval] decoding test_clean (max_samples={max_eval_samples})...',
          flush=True)
    c_clean, t_clean, s_clean = _accuracy_on_split(
        model, dataset['test_clean'], tokenizer,
        max_gen=max_gen, batch_size=batch_size,
        max_samples=max_eval_samples, verbose=verbose,
    )
    print(f'  [eval] decoding test_perturbed...', flush=True)
    c_pert, t_pert, s_pert = _accuracy_on_split(
        model, dataset['test_perturbed'], tokenizer,
        max_gen=max_gen, batch_size=batch_size,
        max_samples=max_eval_samples, verbose=verbose,
    )

    results['accuracy_clean']     = c_clean / max(t_clean, 1)
    results['accuracy_perturbed'] = c_pert  / max(t_pert, 1)
    results['robustness']         = results['accuracy_perturbed']
    results['n_eval_clean']       = t_clean
    results['n_eval_perturbed']   = t_pert
    results['n_skipped_clean']    = s_clean
    results['n_skipped_perturbed'] = s_pert

    if verbose:
        print(f'    acc_clean     = {results["accuracy_clean"]:.3f} '
              f'({c_clean}/{t_clean}, skipped {s_clean})')
        print(f'    acc_perturbed = {results["accuracy_perturbed"]:.3f} '
              f'({c_pert}/{t_pert}, skipped {s_pert})')
        print(f'    clean PPL / pert PPL = {results["clean_ppl"]:.2f} / '
              f'{results["perturbed_ppl"]:.2f}')
    return results


# ============================================================================
# 主实验流程
# ============================================================================

def main():
    """主实验入口：比较 SFT baseline 和 SART 在预训练 LLM 上的表现。"""

    # ---- 读取命令行/环境变量配置 ----
    dataset_name = os.environ.get('DATASET', 'gsm8k').lower()
    ablation = os.environ.get('ABLATION', 'full').lower()
    # full:           完整 SART（重加权 + 梯度手术）
    # reweight_only:  仅重加权（消融）
    # surgery_only:   仅梯度手术（消融）
    # sft_only:       仅标准微调（baseline）

    no_quant = os.environ.get('NO_QUANT', '0') == '1'
    use_4bit = not no_quant

    print("=" * 70)
    print("SART 实验：预训练 LLM + PEFT (LoRA)")
    print("=" * 70)
    print(f"  模型:       {LC.model_name}")
    print(f"  数据集:     {dataset_name}")
    print(f"  消融模式:   {ablation}")
    print(f"  4-bit 量化: {use_4bit}")
    print(f"  设备:       {LC.device}")
    print(f"  LoRA rank:  {LC.lora_r}, alpha={LC.lora_alpha}")
    print(f"  批次大小:   {LC.batch_size} × {LC.grad_accum_steps} (累积) "
          f"= {LC.batch_size * LC.grad_accum_steps} 等效批次")
    print(f"  学习率:     {LC.lr}")
    print(f"  总 Epochs:  {LC.epochs} (预热 {LC.warmup_epochs} + 主训 "
          f"{LC.epochs - LC.warmup_epochs})")
    print("=" * 70)

    set_seed(LC.seed)

    # ---- 步骤 1: 加载模型和 tokenizer ----
    print("\n[步骤 1] 加载预训练模型和 LoRA 适配器...")
    t0 = time.time()
    model, tokenizer = create_peft_model(load_in_4bit=use_4bit)
    trainable, total = count_peft_parameters(model)
    print(f"  加载完成 ({time.time()-t0:.1f}s)")
    print(f"  可训练参数: {trainable:,} ({100*trainable/total:.3f}%)")
    print(f"  总参数数量: {total:,}")

    # ---- 步骤 2: 加载数据集 ----
    print(f"\n[步骤 2] 加载数据集: {dataset_name}...")
    t0 = time.time()
    dataset = load_dataset(dataset_name, tokenizer)
    print(f"  加载完成 ({time.time()-t0:.1f}s)")

    pad_id = tokenizer.pad_token_id

    # ---- 步骤 3: 运行实验 ----
    print(f"\n[步骤 3] 开始训练 (模式: {ablation})...")

    results = {}

    if ablation in ('full', 'reweight_only', 'surgery_only'):
        # ---- 方案 A: SART 变体 ----
        set_seed(LC.seed)
        t0 = time.time()

        if ablation == 'full':
            print("\n>>> 训练: SART (重加权 + 梯度手术)")
            collected = train_sart_peft(
                model, dataset, tokenizer,
                use_reweighting=True, use_gradient_surgery=True, verbose=True
            )
        elif ablation == 'reweight_only':
            print("\n>>> 训练: SART 消融 - 仅重加权")
            collected = train_reweighting_only_peft(model, dataset, tokenizer, verbose=True)
        else:  # surgery_only
            print("\n>>> 训练: SART 消融 - 仅梯度手术")
            collected = train_surgery_only_peft(model, dataset, tokenizer, verbose=True)

        print(f"\n  训练耗时: {time.time()-t0:.1f}s")

        print("\n[评估] SART 模型...")
        results['sart'] = evaluate_peft_model(model, dataset, tokenizer)

    else:
        # ---- 方案 B: 标准 SFT baseline ----
        print("\n>>> 训练: 标准 SFT (baseline)")
        set_seed(LC.seed)
        t0 = time.time()
        train_standard_peft(
            model, dataset, tokenizer,
            epochs=LC.epochs, verbose=True  # SFT 用全部 epochs
        )
        print(f"\n  训练耗时: {time.time()-t0:.1f}s")

        print("\n[评估] SFT baseline 模型...")
        results['sft'] = evaluate_peft_model(model, dataset, tokenizer)

    # ---- 步骤 4: 汇总结果 ----
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    for method, res in results.items():
        print(f"\n  方法: {method}")
        for k, v in res.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")

    # ---- 步骤 5: 保存模型（可选）----
    save_path = os.environ.get('SAVE_PATH', '')
    if save_path:
        print(f"\n[保存] 将 LoRA 适配器保存到: {save_path}")
        # PEFT 的 save_pretrained 只保存 LoRA 权重（约几十 MB），不保存基础模型
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)
        print(f"  保存完成（仅 LoRA 权重，基础模型不保存）")

    return results


if __name__ == '__main__':
    main()
