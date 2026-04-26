"""Full-SART (Reweight + Surgery) K-sensitivity sweep — Gap 6.

Mirror of scripts/reweight_only_K_sweep.py but with use_gradient_surgery=True.

Question: Reweight-Only's robustness gain over SFT is empirically decoupled
from the score's r calibration (Gap 5; commit 3e27cf9). Does adding
gradient surgery restore K-sensitivity that tracks score quality, or does
full SART inherit the same K-decoupling — meaning surgery is also
score-independent and the entire "detection + correction" framing fails?

Method: same as Gap 5 — vary warmup_epochs K in {3, 8, 30} with total
epochs fixed at 50; run full SART (Reweight + Surgery) end-to-end;
measure accuracy_clean + robustness on test sets; compare to SFT baseline
+ Gap 5's Reweight-Only numbers (commit 3e27cf9, results/reweight_K_sweep.json).

Usage:
    CUDA_VISIBLE_DEVICES=0 EXPERIMENT_SCALE=server PYTHONUNBUFFERED=1 \\
        python -u scripts/full_sart_K_sweep.py

Output:
    results/full_sart_K_sweep.json
    results/full_sart_K_sweep.png
"""
import os
import sys
import time
import json

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset)
from src.model import create_model, count_parameters
from src.trainer import train_standard, train_our_method
from src.evaluate import run_full_evaluation

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    K_values = [3, 8, 30]  # paper baseline = 8; Math-good = 3; Causal-good = 30
    total_epochs = 50

    print(f"PROFILE = {PROFILE}, device = {C.device}", flush=True)
    print(f"K sweep = {K_values}, total epochs = {total_epochs}", flush=True)
    print(f"lambda = {C.lambda_}, tau_A = {C.tau_A}, tau_R = {C.tau_R}, "
          f"gamma defaults via config", flush=True)

    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'results')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'full_sart_K_sweep.json')
    fig_path = os.path.join(out_dir, 'full_sart_K_sweep.png')

    results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                results = json.load(f)
            n_done = sum(len(v) for v in results.values())
            print(f"Resume: {n_done} (ds, config) pairs already done",
                  flush=True)
        except Exception as e:
            print(f"Could not load existing JSON ({e}); starting fresh",
                  flush=True)
            results = {}

    tmp = create_model('cpu')
    print(f"\nSmallGPT params: {count_parameters(tmp):,}", flush=True)
    del tmp

    def save():
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    t_start = time.time()
    for ds_name, ds in datasets.items():
        results.setdefault(ds_name, {})

        # SFT baseline (control)
        if 'SFT' not in results[ds_name]:
            print(f"\n--- [{ds_name}] SFT baseline (epochs=50) ---", flush=True)
            set_seed()
            model = create_model()
            t0 = time.time()
            train_standard(model, ds, epochs=total_epochs, verbose=False)
            t_train = time.time() - t0
            r = run_full_evaluation(model, ds, compute_f1=False)
            results[ds_name]['SFT'] = {
                'accuracy_clean': float(r['accuracy_clean']),
                'robustness': float(r['robustness']),
                'train_sec': t_train,
            }
            print(f"  SFT acc={r['accuracy_clean']:.3f}, "
                  f"rob={r['robustness']:.3f}, time={t_train:.1f}s",
                  flush=True)
            save()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Full SART at each K
        for K in K_values:
            tag = f'SART_K{K}'
            if tag in results[ds_name]:
                print(f"  [{ds_name}] {tag} already done, skipping",
                      flush=True)
                continue
            print(f"\n--- [{ds_name}] Full SART K={K} (Reweight+Surgery, "
                  f"total epochs=50) ---", flush=True)
            set_seed()
            model = create_model()
            t0 = time.time()
            cfg_override = {
                'epochs': total_epochs,
                'warmup_epochs': K,
                'score_max_samples': 10000,
            }
            model, _ = train_our_method(
                model, ds,
                use_reweighting=True,
                use_gradient_surgery=True,  # <-- the only difference vs Gap 5
                collect_scores=True,
                cfg=cfg_override,
            )
            t_train = time.time() - t0
            r = run_full_evaluation(model, ds, compute_f1=False)
            results[ds_name][tag] = {
                'accuracy_clean': float(r['accuracy_clean']),
                'robustness': float(r['robustness']),
                'train_sec': t_train,
                'warmup_epochs': K,
            }
            print(f"  SART K={K} acc={r['accuracy_clean']:.3f}, "
                  f"rob={r['robustness']:.3f}, time={t_train:.1f}s",
                  flush=True)
            save()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save()
    total = time.time() - t_start
    print(f"\n=== Sweep done in {total:.1f}s ({total/60:.1f} min) ===",
          flush=True)

    # ---------------- Plot: full-SART vs Reweight-Only vs SFT ----------------
    # Side-by-side comparison: accuracy and robustness vs K, both methods
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Try to load Reweight-Only sweep results for overlay
    ro_path = os.path.join(out_dir, 'reweight_K_sweep.json')
    ro_data = {}
    if os.path.exists(ro_path):
        with open(ro_path) as f:
            ro_data = json.load(f)

    for ds_name in datasets:
        if ds_name not in results:
            continue
        sft = results[ds_name].get('SFT', {})
        # full SART (this sweep)
        sart_accs = [results[ds_name].get(f'SART_K{K}', {}).get('accuracy_clean')
                     for K in K_values]
        sart_robs = [results[ds_name].get(f'SART_K{K}', {}).get('robustness')
                     for K in K_values]
        # Reweight-Only (from Gap 5)
        ro_accs = [ro_data.get(ds_name, {}).get(f'RO_K{K}', {}).get('accuracy_clean')
                   for K in K_values]
        ro_robs = [ro_data.get(ds_name, {}).get(f'RO_K{K}', {}).get('robustness')
                   for K in K_values]

        axes[0].plot(K_values, sart_accs, marker='o', linewidth=2,
                     label=f'{ds_name} (full SART)')
        if any(a is not None for a in ro_accs):
            axes[0].plot(K_values, ro_accs, marker='x', linestyle='--',
                         alpha=0.6, label=f'{ds_name} (RO, Gap 5)')
        if sft.get('accuracy_clean') is not None:
            axes[0].axhline(y=sft['accuracy_clean'], linestyle=':', alpha=0.4)

        axes[1].plot(K_values, sart_robs, marker='o', linewidth=2,
                     label=f'{ds_name} (full SART)')
        if any(r is not None for r in ro_robs):
            axes[1].plot(K_values, ro_robs, marker='x', linestyle='--',
                         alpha=0.6, label=f'{ds_name} (RO, Gap 5)')
        if sft.get('robustness') is not None:
            axes[1].axhline(y=sft['robustness'], linestyle=':', alpha=0.4)

    axes[0].set_xlabel('Warmup epochs K')
    axes[0].set_ylabel('Accuracy (clean)')
    axes[0].set_title('Full SART vs Reweight-Only: accuracy vs K')
    axes[0].set_xscale('log')
    axes[0].legend(fontsize=7, loc='lower right')

    axes[1].set_xlabel('Warmup epochs K')
    axes[1].set_ylabel('Robustness')
    axes[1].set_title('Full SART vs Reweight-Only: robustness vs K')
    axes[1].set_xscale('log')
    axes[1].legend(fontsize=7, loc='upper right')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == '__main__':
    main()
