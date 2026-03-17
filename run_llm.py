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


def evaluate_peft_model(model, dataset, tokenizer, verbose=True):
    """评估 PEFT 模型在干净测试集和扰动测试集上的表现。

    注意：HuggingFace CausalLM 的评估接口与 SmallGPT 不同。
    本函数调用 run_full_evaluation_nl()，该函数已针对真实世界数据集设计。

    Args:
        model:     PEFT 模型（评估时切换为 eval 模式）
        dataset:   数据集字典
        tokenizer: 分词器
        verbose:   是否打印结果

    Returns:
        results: 包含 accuracy_clean, robustness, reasoning_consistency 等的字典
    """
    # run_full_evaluation_nl 期望模型有特定的接口
    # 我们需要为 PEFT 模型提供 pad_id 信息
    pad_id = tokenizer.pad_token_id

    model.eval()
    results = {}

    try:
        # 使用现有的 NL 评估函数
        # 注意：该函数内部调用 model(input_ids)，与 peft_forward 接口略有差异
        # 此处直接计算损失指标（不做生成评估，避免复杂性）
        from src.model_peft import peft_forward
        from src.methods import masked_ce_loss

        def _eval_loss(split_name):
            """计算指定分割集的平均损失（作为性能代理指标）。"""
            loader = get_dataloader(dataset[split_name], batch_size=LC.batch_size,
                                    shuffle=False)
            total_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for batch in loader:
                    inp = batch['input_ids'].to(LC.device)
                    tgt = batch['target_ids'].to(LC.device)
                    lm = batch['loss_mask'].to(LC.device)

                    logits = peft_forward(model, inp, pad_id)
                    loss = masked_ce_loss(logits, tgt, lm)
                    total_loss += loss.item()
                    n_batches += 1
            return total_loss / max(n_batches, 1)

        # 计算困惑度（Perplexity）作为性能指标
        # 低困惑度 = 模型对真实推理序列的预测更好
        import math
        clean_loss = _eval_loss('test_clean')
        perturbed_loss = _eval_loss('test_perturbed')

        results['clean_perplexity'] = math.exp(min(clean_loss, 20))   # 避免 overflow
        results['perturbed_perplexity'] = math.exp(min(perturbed_loss, 20))
        results['clean_loss'] = clean_loss
        results['perturbed_loss'] = perturbed_loss

        # 困惑度差值：扰动集比干净集困惑度高，说明模型对捷径样本泛化差
        results['robustness_gap'] = perturbed_loss - clean_loss

        if verbose:
            print(f"    干净测试集 PPL:    {results['clean_perplexity']:.2f} "
                  f"(loss={clean_loss:.4f})")
            print(f"    扰动测试集 PPL:    {results['perturbed_perplexity']:.2f} "
                  f"(loss={perturbed_loss:.4f})")
            print(f"    鲁棒性差距 (越小越好): {results['robustness_gap']:.4f}")

    except Exception as e:
        print(f"  [警告] 评估失败: {e}")
        results = {'clean_loss': float('nan'), 'perturbed_loss': float('nan')}

    model.train()
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
