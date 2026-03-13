#!/usr/bin/env python3
"""
Hyperparameter search for SART method.

Searches over (lambda_, gamma, rho) and optionally (tau_A, tau_R, phase3_lr_factor)
using multi-GPU parallel grid search.

Usage:
  # Phase 1: Primary search (60 configs, ~2.25 hours on 4× H100)
  EXPERIMENT_SCALE=server python3 hp_search.py

  # Phase 2: Fine-tune around best config from Phase 1
  EXPERIMENT_SCALE=server python3 hp_search.py --phase 2

  # Quick smoke test (8 configs, local profile)
  python3 hp_search.py --smoke-test

  # Single GPU mode
  CUDA_VISIBLE_DEVICES=0 EXPERIMENT_SCALE=server python3 hp_search.py
"""
import os
import sys
import time
import json
import copy
import itertools
import argparse
import random
import numpy as np
import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset, get_dataloader)
from src.model import create_model
from src.trainer import train_our_method
from src.evaluate import run_full_evaluation


# ========================= Seed Utilities =========================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ========================= Search Space =========================

def get_phase1_configs():
    """Phase 1: Primary parameter grid (lambda_, gamma, rho).

    Total: 5 × 4 × 3 = 60 configs.
    Estimated time: 60 × 9 min / 4 GPUs ≈ 2.25 hours.
    """
    space = {
        'lambda_': [0.2, 0.5, 0.8, 1.0, 1.5],
        'gamma':   [0.1, 0.2, 0.3, 0.5],
        'rho':     [0.1, 0.3, 0.5],
    }
    configs = []
    for lam, gam, rho in itertools.product(
            space['lambda_'], space['gamma'], space['rho']):
        configs.append({
            'lambda_': lam,
            'gamma': gam,
            'rho': rho,
            'score_max_samples': 10000,
            'score_batch_size': 64,
        })
    return configs


def get_phase2_configs(best_config):
    """Phase 2: Fine-tune thresholds and LR around best Phase 1 config.

    Total: 4 × 3 × 3 = 36 configs.
    """
    space = {
        'tau_A':             [0.1, 0.2, 0.3, 0.4],
        'tau_R':             [0.3, 0.5, 0.7],
        'phase3_lr_factor':  [0.3, 0.5, 0.7],
    }
    configs = []
    for tau_a, tau_r, lr_f in itertools.product(
            space['tau_A'], space['tau_R'], space['phase3_lr_factor']):
        cfg = dict(best_config)
        cfg['tau_A'] = tau_a
        cfg['tau_R'] = tau_r
        cfg['phase3_lr_factor'] = lr_f
        configs.append(cfg)
    return configs


def get_smoke_test_configs():
    """Tiny grid for smoke testing (8 configs)."""
    configs = []
    for lam in [0.5, 1.0]:
        for gam in [0.2, 0.5]:
            for rho in [0.1, 0.3]:
                configs.append({
                    'lambda_': lam,
                    'gamma': gam,
                    'rho': rho,
                    # Use default score settings for local profile
                })
    return configs


# ========================= Training & Evaluation =========================

def run_single_config(hp_dict, device, datasets):
    """Train SART with given hyperparams on all datasets, return metrics.

    Args:
        hp_dict: hyperparameter dictionary (passed as cfg to train_our_method)
        device: torch device string (e.g. 'cuda:0')
        datasets: dict of {name: dataset_dict}

    Returns:
        dict with config, per_dataset results, averages, and combined score
    """
    results = {}
    for ds_name, ds in datasets.items():
        set_seed(42)
        model = create_model()
        # Move model to the specific device
        if hasattr(model, 'module'):
            model = model.module
        model = model.to(device)

        cfg = dict(hp_dict)
        model = train_our_method(
            model, ds,
            use_reweighting=True,
            use_gradient_surgery=True,
            device=device,
            verbose=False,
            cfg=cfg,
        )

        eval_result = run_full_evaluation(model, ds, device=device)
        results[ds_name] = {
            'accuracy': round(eval_result['accuracy_clean'], 4),
            'robustness': round(eval_result['robustness'], 4),
            'reasoning': round(eval_result.get('reasoning_consistency', 0.0), 4),
        }

        del model
        if 'cuda' in device:
            torch.cuda.empty_cache()

    n = len(results)
    avg_acc = sum(r['accuracy'] for r in results.values()) / n
    avg_rob = sum(r['robustness'] for r in results.values()) / n
    # Combined score: weight robustness more (it's the harder metric to improve)
    combined = 0.4 * avg_acc + 0.6 * avg_rob

    return {
        'config': hp_dict,
        'per_dataset': results,
        'avg_accuracy': round(avg_acc, 4),
        'avg_robustness': round(avg_rob, 4),
        'combined_score': round(combined, 4),
    }


# ========================= Multi-GPU Worker =========================

def worker(gpu_id, task_queue, result_list, datasets, output_dir):
    """Worker process: pulls configs from queue, trains, stores results."""
    device = f'cuda:{gpu_id}' if torch.cuda.is_available() else C.device

    while True:
        try:
            idx, hp_dict = task_queue.get_nowait()
        except Exception:
            break

        config_str = ', '.join(f'{k}={v}' for k, v in hp_dict.items()
                               if k not in ('score_max_samples', 'score_batch_size'))
        print(f"[GPU {gpu_id}] Config {idx}: {config_str}")
        t0 = time.time()

        try:
            result = run_single_config(hp_dict, device, datasets)
            result['config_idx'] = idx
            result['time_seconds'] = round(time.time() - t0, 1)
            result['status'] = 'success'

            print(f"[GPU {gpu_id}] Config {idx} done in {result['time_seconds']:.0f}s: "
                  f"acc={result['avg_accuracy']:.3f}, rob={result['avg_robustness']:.3f}, "
                  f"combined={result['combined_score']:.3f}")
        except Exception as e:
            result = {
                'config_idx': idx,
                'config': hp_dict,
                'status': 'error',
                'error': str(e),
                'time_seconds': round(time.time() - t0, 1),
            }
            print(f"[GPU {gpu_id}] Config {idx} FAILED: {e}")

        result_list.append(result)

        # Save incrementally after each config
        try:
            all_results = sorted(list(result_list), key=lambda r: r.get('config_idx', 0))
            with open(os.path.join(output_dir, 'results_partial.json'), 'w') as f:
                json.dump(all_results, f, indent=2)
        except Exception:
            pass  # Don't crash on save failure


# ========================= Analysis =========================

def analyze_results(results, output_dir):
    """Analyze search results and print findings."""
    # Filter successful results
    ok_results = [r for r in results if r.get('status') == 'success']
    if not ok_results:
        print("No successful results to analyze!")
        return

    # Sort by combined score
    ok_results.sort(key=lambda r: r['combined_score'], reverse=True)

    # --- Top 10 ---
    print('\n' + '=' * 80)
    print('TOP 10 CONFIGURATIONS (by 0.4*acc + 0.6*rob)')
    print('=' * 80)
    for i, r in enumerate(ok_results[:10]):
        cfg = r['config']
        params = ', '.join(f'{k}={v}' for k, v in cfg.items()
                           if k not in ('score_max_samples', 'score_batch_size'))
        print(f"\n  #{i+1}: {params}")
        print(f"    Avg Acc: {r['avg_accuracy']:.3f}, "
              f"Avg Rob: {r['avg_robustness']:.3f}, "
              f"Combined: {r['combined_score']:.3f}")
        for ds, res in r['per_dataset'].items():
            print(f"      {ds}: acc={res['accuracy']:.3f}, rob={res['robustness']:.3f}")

    # --- Comparison with baselines ---
    print('\n' + '=' * 80)
    print('COMPARISON WITH BASELINES (synthetic-only averages)')
    print('=' * 80)
    best = ok_results[0]
    print(f"  Best SART:    Acc={best['avg_accuracy']:.1%}, Rob={best['avg_robustness']:.1%}")
    print(f"  Current Full: Acc=75.8%, Rob=39.9%  (lambda=2.0, gamma=0.8, rho=0.7)")
    print(f"  GS Only:      Acc=81.0%, Rob=51.3%")
    print(f"  Group DRO:    Acc=78.2%, Rob=48.2%")

    improvement_acc = (best['avg_accuracy'] - 0.758) * 100
    improvement_rob = (best['avg_robustness'] - 0.399) * 100
    print(f"\n  Improvement over current Full: "
          f"{improvement_acc:+.1f}pp acc, {improvement_rob:+.1f}pp rob")

    beats_gs = best['avg_accuracy'] > 0.81 and best['avg_robustness'] > 0.513
    beats_gdro = best['avg_accuracy'] > 0.782 and best['avg_robustness'] > 0.482
    print(f"  Beats GS Only: {'YES' if beats_gs else 'NO'}")
    print(f"  Beats Group DRO: {'YES' if beats_gdro else 'NO'}")

    # --- Sensitivity analysis ---
    print('\n' + '=' * 80)
    print('SENSITIVITY ANALYSIS')
    print('=' * 80)

    best_cfg = best['config']
    for param_name in ['lambda_', 'gamma', 'rho']:
        if param_name not in best_cfg:
            continue
        # Find all results with other params matching best
        other_params = {k: v for k, v in best_cfg.items()
                        if k != param_name and k not in ('score_max_samples', 'score_batch_size')}

        matching = []
        for r in ok_results:
            cfg = r['config']
            if all(cfg.get(k) == v for k, v in other_params.items()):
                matching.append(r)

        if matching:
            matching.sort(key=lambda r: r['config'].get(param_name, 0))
            print(f"\n  {param_name} sensitivity (other params = best):")
            for r in matching:
                val = r['config'].get(param_name, '?')
                marker = ' ← BEST' if r['config_idx'] == best.get('config_idx') else ''
                print(f"    {param_name}={val}: acc={r['avg_accuracy']:.3f}, "
                      f"rob={r['avg_robustness']:.3f}, "
                      f"combined={r['combined_score']:.3f}{marker}")

    # --- Save final results ---
    final_path = os.path.join(output_dir, 'results_final.json')
    with open(final_path, 'w') as f:
        json.dump(ok_results, f, indent=2)
    print(f"\nFull results saved to {final_path}")

    # Save best config separately
    best_path = os.path.join(output_dir, 'best_config.json')
    with open(best_path, 'w') as f:
        json.dump(best, f, indent=2)
    print(f"Best config saved to {best_path}")


# ========================= Main =========================

def main():
    parser = argparse.ArgumentParser(description='SART Hyperparameter Search')
    parser.add_argument('--phase', type=int, default=1, choices=[1, 2],
                        help='Search phase: 1=primary grid, 2=fine-tune')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Quick smoke test with tiny grid')
    parser.add_argument('--best-config', type=str, default=None,
                        help='Path to best_config.json from Phase 1 (for Phase 2)')
    parser.add_argument('--output-dir', type=str, default='hp_results',
                        help='Output directory for results')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Determine number of GPUs
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    print(f"Profile: {PROFILE}, GPUs: {num_gpus}, Device: {C.device}")

    # Generate datasets once
    print('\nGenerating datasets...')
    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }
    print(f"Datasets ready: {list(datasets.keys())}")
    print(f"  Train size: {len(datasets['Math-Arithmetic']['train'])}")

    # Generate search configs
    if args.smoke_test:
        configs = get_smoke_test_configs()
        phase_name = 'smoke_test'
    elif args.phase == 1:
        configs = get_phase1_configs()
        phase_name = 'phase1'
    elif args.phase == 2:
        if args.best_config:
            with open(args.best_config) as f:
                best_data = json.load(f)
            best_cfg = best_data['config']
        else:
            # Try to load from default location
            p1_best = os.path.join(args.output_dir, 'best_config.json')
            if os.path.exists(p1_best):
                with open(p1_best) as f:
                    best_data = json.load(f)
                best_cfg = best_data['config']
            else:
                print(f"ERROR: No best config found. Run Phase 1 first or provide --best-config")
                sys.exit(1)
        configs = get_phase2_configs(best_cfg)
        phase_name = 'phase2'
    else:
        configs = get_phase1_configs()
        phase_name = 'phase1'

    print(f"\nPhase: {phase_name}")
    print(f"Total configs: {len(configs)}")
    est_time = len(configs) * 9 / max(num_gpus, 1)  # ~9 min per config
    print(f"Estimated time: {est_time:.0f} minutes ({est_time/60:.1f} hours)")

    # Save search config
    meta = {
        'phase': phase_name,
        'num_configs': len(configs),
        'num_gpus': num_gpus,
        'profile': PROFILE,
        'configs': configs,
    }
    with open(os.path.join(args.output_dir, f'{phase_name}_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)

    # Run search
    if num_gpus <= 1 or not torch.cuda.is_available():
        # Single-process mode
        print("\nRunning in single-process mode...")
        results = []
        for idx, cfg in enumerate(configs):
            result = run_single_config(cfg, C.device, datasets)
            result['config_idx'] = idx
            result['status'] = 'success'

            config_str = ', '.join(f'{k}={v}' for k, v in cfg.items()
                                   if k not in ('score_max_samples', 'score_batch_size'))
            print(f"[{idx+1}/{len(configs)}] {config_str} → "
                  f"acc={result['avg_accuracy']:.3f}, rob={result['avg_robustness']:.3f}")
            results.append(result)

            # Save incrementally
            with open(os.path.join(args.output_dir, 'results_partial.json'), 'w') as f:
                json.dump(results, f, indent=2)
    else:
        # Multi-GPU mode
        print(f"\nRunning in multi-GPU mode ({num_gpus} GPUs)...")
        mp.set_start_method('spawn', force=True)

        task_queue = mp.Queue()
        for idx, cfg in enumerate(configs):
            task_queue.put((idx, cfg))

        manager = mp.Manager()
        result_list = manager.list()

        processes = []
        for gpu_id in range(num_gpus):
            p = mp.Process(
                target=worker,
                args=(gpu_id, task_queue, result_list, datasets, args.output_dir)
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        results = list(result_list)

    # Analyze and save
    print('\n' + '=' * 80)
    print(f'SEARCH COMPLETE: {len(results)} configs evaluated')
    print('=' * 80)

    analyze_results(results, args.output_dir)


if __name__ == '__main__':
    main()
