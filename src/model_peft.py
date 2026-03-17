"""预训练 LLM + PEFT (LoRA) 模型管理模块。

问题背景（为什么需要这个模块）：
    原始 model.py 中的 SmallGPT 从随机初始化（Xavier uniform）开始训练，这与论文
    "Mitigating Shortcut Reasoning in Language Models" 的核心动机存在概念性错位：

    论文声称 SART 旨在解决"预训练 LLM 在 fine-tuning 阶段遭受捷径推理"的问题。
    ShortcutScore 的两个信号（梯度对齐、答案梯度集中度）只有在以下语境下才有意义：
      - 模型已通过海量预训练数据获得语言推理表示
      - fine-tuning 时捷径样本的梯度方向偏离了这些预训练能力
      - 验证集梯度 g_V 真正代表"强化预训练推理知识"的方向

    随机初始化的模型没有任何先验推理知识，"梯度对齐"仅反映样本难易程度，
    而非论文所定义的"推理能力泛化性"。

解决方案：
    加载预训练因果语言模型（Qwen2.5-7B 或 Llama-3.1-8B），使用 LoRA 进行 PEFT：
      - 基础模型权重冻结（requires_grad=False），保留预训练语言知识
      - LoRA 适配器（约 8-16M 可训练参数）负责任务适应
      - 梯度运算仅在 LoRA 参数上进行，使梯度手术在显存上可行
        （全量 7B 梯度 ~28GB vs LoRA rank-16 梯度 ~32MB）

支持的模型：
    - Qwen/Qwen2.5-7B-Instruct（推荐，无需 HF access token）
    - meta-llama/Llama-3.1-8B-Instruct（需要申请 HF token）

依赖：
    pip install transformers peft bitsandbytes accelerate
"""

import torch
import torch.nn as nn
from src.config import LLMConfig as LC


# ============================================================================
# 模型创建
# ============================================================================

def create_peft_model(model_name=None, device=None, load_in_4bit=None, load_in_8bit=None):
    """加载预训练 LLM 并附加 LoRA 适配器。

    完整流程：
      1. 加载 tokenizer（设置 pad_token = eos_token，右填充）
      2. （可选）配置 bitsandbytes 量化以节省显存
      3. 使用 device_map="auto" 加载基础模型（自动分配多 GPU / CPU offload）
      4. 定义 LoRA 配置，冻结基础模型，插入可训练适配器

    设计要点：
      - 右填充（padding_side="right"）：使 attention_mask = (input_ids != pad_id)
        的推导正确——真实 EOS token 位于序列末端（不在 input_ids 中），
        填充 token 出现在序列右侧，不与真实 EOS 混淆。
      - PEFT 自动将基础模型所有参数设为 requires_grad=False，
        只有 LoRA 的 A/B 矩阵是 requires_grad=True。
      - methods.py 的 get_grad_vector() 已过滤 requires_grad，
        对 PEFT 模型自动只收集 LoRA 梯度，无需修改。

    Args:
        model_name:    HuggingFace 模型 ID 或本地路径。默认使用 LLMConfig.model_name。
        device:        目标设备。若有 CUDA 则使用 device_map="auto"。
        load_in_4bit:  是否使用 4-bit NF4 量化（QLoRA 风格）。默认 LLMConfig.use_4bit。
        load_in_8bit:  是否使用 8-bit LLM.int8 量化。默认 LLMConfig.use_8bit。

    Returns:
        model:      PEFT 封装的模型，只有 LoRA 参数可训练。
        tokenizer:  对应的分词器（pad_token=eos_token, padding_side="right"）。

    显存估算（7B 模型）：
        无量化 (bfloat16): ~14 GB
        8-bit:              ~8-10 GB
        4-bit (QLoRA):      ~4-6 GB  ← 推荐
    """
    # 延迟导入以避免在不需要 PEFT 功能时强制安装依赖
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType

    _model_name = model_name or LC.model_name
    _4bit = load_in_4bit if load_in_4bit is not None else LC.use_4bit
    _8bit = load_in_8bit if load_in_8bit is not None else LC.use_8bit

    # ----------------------------------------------------------------
    # Step 1: 加载 Tokenizer
    # ----------------------------------------------------------------
    print(f"[PEFT] 加载 tokenizer: {_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        _model_name,
        trust_remote_code=True,   # Qwen 需要 trust_remote_code
    )

    # 设置 pad_token：大多数 LLM 没有专用的 pad_token，使用 eos_token 代替
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 右填充：训练时序列右侧填充，使得 attention_mask 计算简单且正确
    # （生成时需要左填充，在 evaluate_peft.py 中切换）
    tokenizer.padding_side = "right"

    # ----------------------------------------------------------------
    # Step 2: 配置量化（可选，节省显存）
    # ----------------------------------------------------------------
    bnb_config = None

    if _4bit:
        # QLoRA 风格：4-bit NF4 量化 + double quantization
        # - NF4 (Normal Float 4): 针对正态分布权重的最优 4-bit 量化格式
        # - double_quant: 对量化常量本身再量化，额外节省约 0.5 bit/param
        # - compute_dtype=bfloat16: 前向传播时将 4-bit 反量化到 bfloat16 计算
        print("[PEFT] 使用 4-bit NF4 量化 (QLoRA)")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif _8bit:
        # LLM.int8：混合精度量化，大矩阵 int8，小矩阵 float16
        print("[PEFT] 使用 8-bit LLM.int8 量化")
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        print("[PEFT] 不使用量化，以 bfloat16 加载")

    # ----------------------------------------------------------------
    # Step 3: 加载基础模型
    # ----------------------------------------------------------------
    print(f"[PEFT] 加载基础模型: {_model_name}")

    # device_map="auto": Accelerate 库自动将模型层分配到可用 GPU/CPU
    # 若无 CUDA，则所有层在 CPU 上运行（速度慢但可调试）
    load_kwargs = {
        "quantization_config": bnb_config,
        "trust_remote_code": True,
        "device_map": "auto" if torch.cuda.is_available() else None,
    }
    # 无量化时显式指定 bfloat16（比 float32 节省一半显存，精度损失极小）
    if not _4bit and not _8bit:
        load_kwargs["torch_dtype"] = torch.bfloat16

    base_model = AutoModelForCausalLM.from_pretrained(_model_name, **load_kwargs)

    # 禁用 KV cache：训练阶段不需要 cache，禁用可节省显存
    base_model.config.use_cache = False

    # ----------------------------------------------------------------
    # Step 4: 配置并应用 LoRA 适配器
    # ----------------------------------------------------------------
    # LoRA 原理：对目标权重矩阵 W ∈ R^{d×k}，插入低秩矩阵对：
    #   h = W_frozen @ x + (B @ A) @ x * (alpha / r)
    # 其中 A ∈ R^{r×k}，B ∈ R^{d×r}，r << min(d,k)
    # 初始化：A 随机正态，B 全零（确保初始化时 LoRA 输出为零）
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LC.lora_r,
        # LoRA 秩：可训练参数 ≈ 2 × r × d_model × n_target_layers
        # rank=16, d_model=4096 (Qwen-7B), 4 layers: 约 8M 参数

        lora_alpha=LC.lora_alpha,
        # 缩放因子：有效缩放 = alpha/r，设为 2*r 使缩放为 2.0
        # 等效于对 LoRA 学习率做 2× 放大

        lora_dropout=LC.lora_dropout,
        # 防止适配器过拟合，fine-tuning 数据较少时尤为重要

        target_modules=LC.lora_target_modules,
        # 注意力层 Q/K/V/O 投影矩阵：这些是捷径推理最容易编码的位置
        # Q/K/V: 决定注意力模式（捷径可能通过关注"简单"特征实现）
        # O: 输出投影，汇聚注意力结果

        bias="none",
        # 不训练 bias 项，节省参数，通常对性能影响可忽略

        inference_mode=False,
        # 训练模式：启用 dropout 和梯度计算
    )

    print(f"[PEFT] 应用 LoRA: rank={LC.lora_r}, alpha={LC.lora_alpha}, "
          f"target_modules={LC.lora_target_modules}")

    model = get_peft_model(base_model, lora_config)

    # 打印可训练参数统计（例如: "trainable params: 8,388,608 || all params: 7.2B"）
    model.print_trainable_parameters()

    return model, tokenizer


# ============================================================================
# 梯度工具函数（PEFT 感知版本）
# ============================================================================

def get_peft_grad_vector(model):
    """将所有可训练（LoRA）参数的梯度拼接为单一向量。

    这是 methods.py 中 get_grad_vector() 的 PEFT 感知版本。
    关键区别：只收集 requires_grad=True 的参数（即 LoRA 的 A/B 矩阵），
    跳过冻结的基础模型参数，使梯度向量维度保持可管理（约 8-32M）。

    注意：methods.py 的 get_grad_vector() 本身已过滤 requires_grad，
    理论上对 PEFT 模型也适用。但本函数额外处理 bfloat16 精度转换：
    将梯度统一转为 float32 以确保梯度运算的数值稳定性。

    Returns:
        grad_vec: (D,) float32 梯度向量，D = LoRA 可训练参数总数
    """
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is not None:
                # 转为 float32 确保梯度运算精度（LoRA 参数可能是 bfloat16）
                grads.append(param.grad.detach().float().flatten())
            else:
                # 未参与当前 forward 的参数，梯度为零
                grads.append(torch.zeros(param.numel(), dtype=torch.float32,
                                         device=param.device))
    if not grads:
        raise RuntimeError(
            "[PEFT] 未找到任何 requires_grad=True 的参数。"
            "请检查 LoRA 是否正确应用（peft.get_peft_model）。"
        )
    return torch.cat(grads)


def set_peft_grad_vector(model, grad_vec):
    """将平铺的梯度向量写回 LoRA 可训练参数的 .grad 属性。

    用于梯度手术（gradient surgery）：
      1. 收集 LoRA 梯度向量 g = get_peft_grad_vector(model)
      2. 对 g 做投影/抑制修改，得到 g_modified
      3. 调用 set_peft_grad_vector(model, g_modified) 写回
      4. optimizer.step() 使用修改后的梯度更新 LoRA 参数

    Args:
        model:    PEFT 封装的模型
        grad_vec: (D,) float32 梯度向量，维度与 get_peft_grad_vector 输出一致
    """
    offset = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            numel = param.numel()
            grad_slice = grad_vec[offset: offset + numel]
            # 转回参数的原始 dtype（通常是 bfloat16 或 float32）
            param.grad = grad_slice.reshape(param.shape).to(param.dtype).clone()
            offset += numel


def count_peft_parameters(model):
    """统计模型的可训练参数和总参数数量。

    Returns:
        (trainable_params, total_params): 整数元组
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def unwrap_peft_model(model):
    """返回底层 PeftModel（去除 DataParallel 包装，如果有的话）。"""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


# ============================================================================
# HuggingFace 前向传播适配器
# ============================================================================

def peft_forward(model, input_ids, pad_id):
    """执行 HuggingFace CausalLM 的前向传播，自动构造 attention_mask。

    SmallGPT 的前向接口：`logits = model(x, pad_mask=pad_mask)`
    HuggingFace CausalLM 的接口：`outputs = model(input_ids=..., attention_mask=...)`

    本函数封装 HuggingFace 接口，使其与 methods.py 中的 masked_ce_loss 兼容：
      - 输入: input_ids (B, T)，来自 ReasoningDataset（已做左移，不含最终 EOS）
      - 输出: logits (B, T, V)，与 SmallGPT 输出格式相同

    attention_mask 推导：
      - 数据使用右填充（tokenizer.padding_side = "right"）
      - 真实 token 在序列左侧，padding token（= pad_id = eos_token_id）在右侧
      - 由于 input_ids 已经是 tokens[:-1]（不含最终 EOS），
        序列中 pad_id 的出现仅来自批次填充，不来自真实序列
      - 因此 attention_mask = (input_ids != pad_id) 正确标识有效 token 位置

    Args:
        model:      PEFT 封装的 CausalLM 模型
        input_ids:  (B, T) 输入 token ids
        pad_id:     填充 token 的 id（通常 = tokenizer.eos_token_id）

    Returns:
        logits: (B, T, V) 未归一化的词汇表分布
    """
    # 构造 attention mask：真实 token 为 1，填充 token 为 0
    attention_mask = (input_ids != pad_id).long()

    # HuggingFace CausalLM 前向传播
    # 注意：不传 labels，避免 HF 内部做 shift 计算 loss（我们自己用 masked_ce_loss）
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,  # 训练时禁用 KV cache 节省显存
    )
    return outputs.logits  # (B, T, V)
