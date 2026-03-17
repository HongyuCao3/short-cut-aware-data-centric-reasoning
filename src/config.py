"""Configuration for all experiments.

Supports two dimensions:
  Scale profile (EXPERIMENT_SCALE):
    - 'local':  Small-scale (277K params, 500 train) for quick local iteration
    - 'server': Large-scale (19M+ params, 10K+ train) for GPU servers (H100 etc.)

  Dataset type (DATASET_TYPE):
    - 'synthetic':  3 synthetic datasets (Math, Financial, Causal) — default
    - 'realworld':  GSM8K + MATH real-world benchmarks
    - 'all':        Both synthetic and real-world

Usage:
  EXPERIMENT_SCALE=server python3 run_all.py                    # synthetic only
  EXPERIMENT_SCALE=server DATASET_TYPE=realworld python3 run_all.py  # GSM8K + MATH
  EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py        # everything
"""
import os
import torch


def _detect_profile():
    env = os.environ.get('EXPERIMENT_SCALE', '').lower()
    if env in ('server', 'large'):
        return 'server'
    elif env in ('local', 'small'):
        return 'local'
    return 'server' if torch.cuda.is_available() else 'local'


PROFILE = _detect_profile()
DATASET_TYPE = os.environ.get('DATASET_TYPE', 'synthetic').lower()


class Config:
    # ================================================================
    # Profile-dependent settings (synthetic datasets)
    # ================================================================
    if PROFILE == 'server':
        vocab_size = 35
        d_model = 512
        nhead = 8
        num_layers = 6
        d_ff = 2048
        max_seq_len = 24
        dropout = 0.0

        batch_size = 128
        lr = 1e-3
        epochs = 50
        seed = 42
        weight_decay = 1e-4

        n_train = 10000
        n_val = 2000
        n_test = 3000
        shortcut_ratio = 0.70

        score_max_samples = 10000   # score all training samples for better weight estimates
        score_batch_size = 64       # larger batches for throughput on H100

        df_warmup_epochs = 5
        df_confidence_threshold = 0.90

        jtt_warmup_epochs = 5
        jtt_upweight_factor = 3
        focal_gamma = 2.0
        gdro_eta = 0.01

        irm_lambda = 1.0
        irm_anneal_epochs = 5
        vrex_beta = 1.0
        fishr_lambda = 1.0
        fishr_ema_decay = 0.95
        lff_q = 0.7
        influence_warmup_epochs = 5
        influence_remove_ratio = 0.3
        meta_reweight_lr = 0.01

    else:
        vocab_size = 35
        d_model = 128
        nhead = 4
        num_layers = 2
        d_ff = 256
        max_seq_len = 24
        dropout = 0.0

        batch_size = 32
        lr = 3e-3
        epochs = 30
        seed = 42
        weight_decay = 1e-5

        n_train = 500
        n_val = 200
        n_test = 300
        shortcut_ratio = 0.70

        score_max_samples = 200
        score_batch_size = 1

        df_warmup_epochs = 3
        df_confidence_threshold = 0.90

        jtt_warmup_epochs = 5
        jtt_upweight_factor = 3
        focal_gamma = 2.0
        gdro_eta = 0.01

        irm_lambda = 1.0
        irm_anneal_epochs = 5
        vrex_beta = 1.0
        fishr_lambda = 1.0
        fishr_ema_decay = 0.95
        lff_q = 0.7
        influence_warmup_epochs = 5
        influence_remove_ratio = 0.3
        meta_reweight_lr = 0.01

    # ================================================================
    # Real-world dataset config (GSM8K / MATH)
    # ================================================================
    class NL:
        """Config for real-world NL reasoning datasets."""
        vocab_size = 50257      # GPT-2 tokenizer
        max_seq_len = 512
        d_model = 768
        nhead = 12
        num_layers = 12
        d_ff = 3072
        dropout = 0.0

        batch_size = 32
        lr = 5e-4
        epochs = 20
        weight_decay = 1e-4

        shortcut_ratio = 0.70

        score_max_samples = 1000
        score_batch_size = 4

        df_warmup_epochs = 3
        df_confidence_threshold = 0.90

        jtt_warmup_epochs = 5
        jtt_upweight_factor = 3
        focal_gamma = 2.0
        gdro_eta = 0.01

        irm_lambda = 1.0
        irm_anneal_epochs = 5
        vrex_beta = 1.0
        fishr_lambda = 1.0
        fishr_ema_decay = 0.95
        lff_q = 0.7
        influence_warmup_epochs = 5
        influence_remove_ratio = 0.3
        meta_reweight_lr = 0.01

        # Special token strings (resolved to IDs at runtime by tokenizer)
        question_sep = "\n\nSolution:\n"   # separates question from reasoning
        answer_sep = "####"                 # separates reasoning from answer (GSM8K style)

    # ================================================================
    # Shared settings (profile-independent)
    # ================================================================

    # ShortcutScore hyperparameters
    alpha = 1.0
    beta = 1.0
    tau_A = 0.3
    tau_R = 0.5
    lambda_ = 2.0
    gamma = 0.8
    rho = 0.7
    val_grad_interval = 5

    # Self-Consistency Decoding
    sc_num_samples = 5
    sc_temperature = 0.8

    # Device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

    # Synthetic token IDs
    PAD = 0
    BOS = 1
    EOS = 2
    SEP = 3
    EQ = 4
    PLUS = 5
    MINUS = 6
    MULT = 7
    COLON = 8
    SAT = 9
    VIO = 10
    CAUS = 11
    NCAUS = 12
    DIGIT_OFFSET = 13
    FEAT_R = 23
    FEAT_C = 24
    FEAT_M = 25
    FEAT_D = 26
    FEAT_X = 27
    FEAT_Y = 28
    FEAT_Z = 29
    FEAT_COR = 30

    @classmethod
    def digit_token(cls, d):
        return d + cls.DIGIT_OFFSET

    @classmethod
    def token_to_digit(cls, t):
        return t - cls.DIGIT_OFFSET


# ============================================================================
# LLM + PEFT 配置
# ----------------------------------------------------------------------------
# 解决核心问题：原始 SmallGPT 从随机初始化训练，与论文动机不符。
# 论文声称 SART 用于解决 *预训练* LLM 在 fine-tuning 过程中产生的捷径推理。
# 捷径样本梯度与验证梯度的"不对齐"信号，只有在模型已具备预训练语言表示时才有
# 物理意义——此时梯度方向的对齐度真正反映了"是否强化了预训练的推理能力"。
#
# 解决方案：
#   - 加载预训练因果语言模型（Qwen2.5-7B 或 Llama-3.1-8B）
#   - 使用 LoRA (Low-Rank Adaptation) 进行参数高效微调 (PEFT)
#   - 梯度运算仅作用于 LoRA 参数（约 8-16M），避免对全量 7B 参数做梯度手术
# ============================================================================

class LLMConfig:
    """预训练 LLM + PEFT (LoRA) 实验配置。

    该配置用于修复原始实现中"随机初始化小模型"与论文动机不符的问题。
    通过加载预训练 7B 级别模型并使用 LoRA 微调，确保：
      1. 模型已有预训练语言理解能力，捷径信号有明确的梯度语义
      2. LoRA 使可训练参数数量从 7B 降至约 8-16M，梯度手术在显存上可行
      3. 4-bit 量化（QLoRA）使 7B 模型可在单张 24GB GPU 上运行

    环境变量：
      LLM_MODEL=Qwen/Qwen2.5-7B-Instruct  （默认，无需 HF token）
      LLM_MODEL=meta-llama/Llama-3.1-8B-Instruct  （需要 HF token）
    """

    # ----------------------------------------------------------------
    # 模型选择
    # ----------------------------------------------------------------
    # Qwen2.5-7B: 推荐，无需 HuggingFace access token，中英文推理能力强
    # Llama-3.1-8B: 需要申请 HF access token
    model_name: str = os.environ.get('LLM_MODEL', 'Qwen/Qwen2.5-7B-Instruct')

    # ----------------------------------------------------------------
    # LoRA (Low-Rank Adaptation) 配置
    # LoRA 在注意力层的权重矩阵旁边插入低秩矩阵对 (A, B)：
    #   W' = W_frozen + (B @ A) * (lora_alpha / lora_r)
    # 其中 W_frozen 完全冻结，只有 A、B 参与梯度计算。
    # ----------------------------------------------------------------
    lora_r: int = 16
    # LoRA 秩（rank）：控制适配器容量。
    # r=16 时对 7B 模型约增加 8M 可训练参数，梯度向量 ~32MB（可行）。
    # r=8 更节省显存，r=32 容量更大但梯度向量 ~64MB。

    lora_alpha: int = 32
    # LoRA 缩放因子。实际缩放 = alpha / r = 2.0（通常设为 2*r）。
    # 等效于对 LoRA 输出做 2× 放大，防止 LoRA 参数初始化接近零时影响过小。

    lora_dropout: float = 0.05
    # LoRA 层的 dropout，防止适配器过拟合。

    lora_target_modules: list = ["q_proj", "k_proj", "v_proj", "o_proj"]
    # 应用 LoRA 的目标模块（注意力层的 Q/K/V/O 投影矩阵）。
    # Qwen2.5 和 Llama-3.1 使用相同的模块命名规范。
    # 可选添加 MLP 层: ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
    # 但这会增加可训练参数和梯度向量维度。

    # ----------------------------------------------------------------
    # 量化配置（显著降低 VRAM 需求）
    # ----------------------------------------------------------------
    use_4bit: bool = True
    # QLoRA (4-bit NF4 量化): 7B 模型约需 4-6 GB VRAM（推荐，单卡可运行）
    # 注意: 4-bit 量化的基础权重仍参与前向传播，但以 bfloat16 计算
    # 梯度仅在 LoRA 参数（bfloat16/float32）上计算，与量化无关

    use_8bit: bool = False
    # LLM.int8 量化: 7B 模型约需 8-10 GB VRAM
    # 与 use_4bit 互斥，use_4bit 优先级更高

    # ----------------------------------------------------------------
    # 训练超参数
    # ----------------------------------------------------------------
    max_seq_len: int = 512
    # 最大序列长度。GSM8K 平均约 200 tokens，MATH 约 400 tokens。

    batch_size: int = 2
    # 每步实际批次大小（受 VRAM 限制，7B + 4-bit 约可支持 batch=2-4）

    grad_accum_steps: int = 8
    # 梯度累积步数：等效批次大小 = batch_size × grad_accum_steps = 16
    # 梯度累积避免了大批次所需的峰值显存

    lr: float = 2e-4
    # LoRA 微调的学习率。比全量微调（~1e-5）高，因为 LoRA 参数较少。

    warmup_epochs: int = 1
    # 标准微调预热阶段（SART 激活前）。
    # 预训练模型比随机初始化模型收敛快，只需 1 个 epoch 预热。

    epochs: int = 3
    # 总微调 epochs（预训练模型 3 epochs 通常已足够）。

    weight_decay: float = 0.01
    # AdamW 权重衰减，防止 LoRA 参数过拟合。

    seed: int = 42

    # ----------------------------------------------------------------
    # SART 超参数（针对预训练 LLM 场景调整）
    # ----------------------------------------------------------------
    # 注意：这些值与合成数据集实验（config.py 主体）不同，原因如下：
    # - 预训练模型的梯度结构更丰富，baseline 对齐度更高
    # - 微调阶段捷径样本的梯度方向偏差相对更小
    # - 过激的梯度手术会破坏预训练获得的语言知识

    alpha: float = 1.0        # ShortcutScore 中对齐项 B(s) 的权重
    beta: float = 1.0         # ShortcutScore 中集中度项 C(s) 的权重

    tau_A: float = 0.1
    # 对齐阈值（低于此值视为捷径）。
    # 比合成实验（0.3）更低：预训练模型的基线对齐度更高，捷径样本
    # 不对齐仍相对明显，但绝对值更接近 0。

    tau_R: float = 0.6
    # 集中度阈值（高于此值视为答案主导）。
    # 比合成实验（0.5）略高：预训练模型在 fine-tuning 时本身倾向于
    # 关注答案 token，需要更高阈值才能判定为异常集中。

    lambda_: float = 1.0
    # 重加权强度。比合成实验（2.0）更温和，避免过度压制含有预训练知识的样本。

    gamma: float = 0.5
    # 梯度投影强度（去除不可迁移梯度分量）。
    # 比合成实验（0.8）更保守，保留预训练语言知识中的有用成分。

    rho: float = 0.3
    # 答案梯度抑制系数。比合成实验（0.7）更保守，
    # 因为答案 token 在真实推理任务中也携带重要语义信息。

    val_grad_interval: int = 1
    # 每隔多少 epochs 重新计算验证梯度 g_V。
    # 比合成实验（5）更频繁，因为总 epochs 少（仅 3 个）。

    # ----------------------------------------------------------------
    # ShortcutScore 计算控制
    # ----------------------------------------------------------------
    score_batch_size: int = 1
    # 每次计算梯度的样本数（逐个处理以获得精确的 per-sample 梯度）

    score_max_samples: int = 500
    # 参与 ShortcutScore 计算的最大样本数（时间/显存预算控制）。
    # 对未计分样本使用平均分作为替代。

    # ----------------------------------------------------------------
    # 数据相关
    # ----------------------------------------------------------------
    shortcut_ratio: float = 0.70
    # 训练集中捷径标签的比例（与合成实验保持一致）

    question_sep: str = "\n\nSolution:\n"
    # 问题与推理之间的分隔符（GSM8K 风格）

    answer_sep: str = "####"
    # 推理与答案之间的分隔符（GSM8K 风格）

    # ----------------------------------------------------------------
    # 生成配置（评估时使用）
    # ----------------------------------------------------------------
    max_new_tokens: int = 50
    # 评估时最大生成 token 数

    # ----------------------------------------------------------------
    # 设备配置
    # ----------------------------------------------------------------
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
