#!/usr/bin/env python3
"""
Run GPT-2 pretrained experiments on GSM8K (and optionally MATH).

Usage:
  # Debug (1 method, small subset):
  CUDA_VISIBLE_DEVICES=1 python3 run_pretrained.py --debug

  # Full run:
  CUDA_VISIBLE_DEVICES=1 python3 run_pretrained.py

  # GSM8K only:
  CUDA_VISIBLE_DEVICES=1 python3 run_pretrained.py --dataset gsm8k

  # Full run with nohup:
  CUDA_VISIBLE_DEVICES=1 nohup python3 run_pretrained.py > logs/pretrained_exp.log 2>&1 &
"""
import os
import sys
import time
import argparse
import torch
import random
import numpy as np

# Force pretrained mode
os.environ['USE_PRETRAINED'] = '1'
os.environ['EXPERIMENT_SCALE'] = 'server'

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config as C
from src.data import get_dataloader
from src.model import create_model_pretrained, count_parameters
from src.trainer import (train_standard, train_data_filtering, train_our_method,
                         train_jtt, train_focal_loss, train_group_dro,
                         train_irm, train_vrex, train_fishr,
                         train_lff, train_influence_filtering, train_meta_reweight)
from src.evaluate import (run_full_evaluation_nl, evaluate_gradient_alignment)


def set_seed(seed=C.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_nl_cfg():
    return {
        'batch_size': C.NL.batch_size,
        'lr': C.NL.lr,
        'epochs': C.NL.epochs,
        'weight_decay': C.NL.weight_decay,
        'score_max_samples': C.NL.score_max_samples,
        'score_batch_size': C.NL.score_batch_size,
        'df_warmup_epochs': C.NL.df_warmup_epochs,
        'df_confidence_threshold': C.NL.df_confidence_threshold,
        'jtt_warmup_epochs': C.NL.jtt_warmup_epochs,
        'jtt_upweight_factor': C.NL.jtt_upweight_factor,
        'focal_gamma': C.NL.focal_gamma,
        'gdro_eta': C.NL.gdro_eta,
        'irm_lambda': C.NL.irm_lambda,
        'irm_anneal_epochs': C.NL.irm_anneal_epochs,
        'vrex_beta': C.NL.vrex_beta,
        'fishr_lambda': C.NL.fishr_lambda,
        'fishr_ema_decay': C.NL.fishr_ema_decay,
        'lff_q': C.NL.lff_q,
        'influence_warmup_epochs': C.NL.influence_warmup_epochs,
        'influence_remove_ratio': C.NL.influence_remove_ratio,
        'meta_reweight_lr': C.NL.meta_reweight_lr,
    }


def run_method(method_name, train_fn, ds, tokenizer, nl_cfg, device,
               compute_f1=False, compute_alignment=False,
               train_kwargs=None):
    """Train and evaluate a single method."""
    print(f'\n  Training: {method_name}...')
    set_seed()
    model = create_model_pretrained(device)
    t0 = time.time()

    if train_kwargs:
        result_model = train_fn(model, ds, cfg=nl_cfg, **train_kwargs)
    else:
        result_model = train_fn(model, ds, cfg=nl_cfg)

    # train_our_method returns (model, collected)
    collected = None
    if isinstance(result_model, tuple):
        result_model, collected = result_model

    elapsed = time.time() - t0
    print(f'  Training time: {elapsed:.1f}s')

    print('  Evaluating...')
    results = run_full_evaluation_nl(
        result_model, ds, tokenizer, device=device,
        compute_f1=compute_f1, compute_alignment=compute_alignment
    )
    print(f'  Accuracy: {results["accuracy_clean"]:.3f}, '
          f'Robustness: {results["robustness"]:.3f}', end='')
    if results.get('shortcut_f1') is not None:
        print(f', F1: {results["shortcut_f1"]:.3f}', end='')
    if results.get('gradient_alignment') is not None:
        print(f', Alignment: {results["gradient_alignment"]:.3f}', end='')
    print()

    del model, result_model
    torch.cuda.empty_cache()

    return results, collected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true',
                        help='Quick debug: SFT + SART only, 2 epochs')
    parser.add_argument('--dataset', choices=['gsm8k', 'math', 'both'],
                        default='gsm8k', help='Which dataset(s) to run')
    parser.add_argument('--methods', type=str, default=None,
                        help='Comma-separated methods to run (e.g., sft,sart,data_filtering)')
    args = parser.parse_args()

    device = C.device
    print('=' * 70)
    print('GPT-2 Pretrained Experiments')
    print('=' * 70)
    print(f'Device: {device}')
    if device == 'cuda':
        print(f'GPU: {torch.cuda.get_device_name(0)} '
              f'({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)')
    print(f'Model: {C.pretrained_model_name}')
    print(f'Config: bs={C.NL.batch_size}, lr={C.NL.lr}, epochs={C.NL.epochs}')

    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    # Check model size
    tmp = create_model_pretrained('cpu')
    print(f'Model parameters: {count_parameters(tmp):,}')
    del tmp

    # Load datasets
    from src.data_realworld import generate_gsm8k_dataset, generate_math_dataset_realworld

    nl_datasets = {}
    if args.dataset in ('gsm8k', 'both'):
        print('\n--- Loading GSM8K ---')
        nl_datasets['GSM8K'] = generate_gsm8k_dataset(tokenizer, seed=42)
    if args.dataset in ('math', 'both'):
        print('\n--- Loading MATH ---')
        nl_datasets['MATH'] = generate_math_dataset_realworld(tokenizer, seed=43)

    nl_cfg = get_nl_cfg()

    # Debug mode: fewer epochs
    if args.debug:
        nl_cfg['epochs'] = 3
        nl_cfg['score_max_samples'] = 20
        nl_cfg['score_batch_size'] = 1  # Use full gradient path (faster for 124M params)

    # Define methods
    all_methods = {
        'sft': ('Standard Fine-Tuning', train_standard, {}),
        'data_filtering': ('Data Filtering', train_data_filtering, {}),
        'jtt': ('JTT', train_jtt, {}),
        'focal_loss': ('Focal Loss', train_focal_loss, {}),
        'group_dro': ('Group DRO', train_group_dro, {}),
        'irm': ('IRM', train_irm, {}),
        'vrex': ('V-REx', train_vrex, {}),
        'fishr': ('Fishr', train_fishr, {}),
        'lff': ('LfF', train_lff, {}),
        'influence_filtering': ('Influence Filtering', train_influence_filtering, {}),
        'meta_reweight': ('Meta-Reweighting', train_meta_reweight, {}),
        'sart': ('SART (Ours)', train_our_method,
                 {'use_reweighting': True, 'use_gradient_surgery': True,
                  'collect_scores': True}),
    }

    # Select methods
    if args.debug:
        selected = ['sft', 'sart']
    elif args.methods:
        selected = [m.strip() for m in args.methods.split(',')]
    else:
        selected = list(all_methods.keys())

    all_results = {}
    total_start = time.time()

    for ds_name, ds in nl_datasets.items():
        print(f'\n{"=" * 70}')
        print(f'Dataset: {ds_name} (Pretrained GPT-2)')
        print(f'{"=" * 70}')

        for i, method_key in enumerate(selected):
            if method_key not in all_methods:
                print(f'  WARNING: Unknown method {method_key}, skipping')
                continue
            name, train_fn, kwargs = all_methods[method_key]
            compute_f1 = method_key in ('data_filtering', 'sart')
            compute_alignment = method_key == 'sart'

            print(f'\n[{i+1}/{len(selected)}] {name}')
            results, collected = run_method(
                name, train_fn, ds, tokenizer, nl_cfg, device,
                compute_f1=compute_f1, compute_alignment=compute_alignment,
                train_kwargs=kwargs if kwargs else None
            )
            all_results[(ds_name, method_key)] = results

    # Print summary table
    total_time = time.time() - total_start
    print(f'\n{"=" * 70}')
    print(f'RESULTS SUMMARY (Pretrained GPT-2) — {total_time/60:.1f} minutes total')
    print(f'{"=" * 70}')
    print(f'{"Method":<25} {"Clean Acc":>10} {"Robustness":>10} {"F1":>8} {"Align":>8}')
    print('-' * 65)

    for ds_name in nl_datasets:
        print(f'\n  {ds_name}:')
        for method_key in selected:
            key = (ds_name, method_key)
            if key not in all_results:
                continue
            r = all_results[key]
            name = all_methods[method_key][0]
            f1_str = f'{r["shortcut_f1"]:.3f}' if r.get("shortcut_f1") is not None else '  -'
            align_str = f'{r["gradient_alignment"]:.3f}' if r.get("gradient_alignment") is not None else '  -'
            print(f'  {name:<25} {r["accuracy_clean"]:>10.3f} {r["robustness"]:>10.3f} {f1_str:>8} {align_str:>8}')

    print(f'\nTotal time: {total_time/60:.1f} minutes')
    return all_results


if __name__ == '__main__':
    results = main()
