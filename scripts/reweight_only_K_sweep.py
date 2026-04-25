"""Reweight-Only K-sensitivity sweep for Gap 5.

Question: Reweight-Only ablation in §5.3 Table 3 reports +20.7 pp robustness
over SFT. The warmup_epoch_sweep.py result shows the score's calibration is
fragile and dataset-specific (Math K=3 only, Financial K=3 only, Causal
peaks at K=2 and K=30). If +20.7 pp depends on the score, accuracy /
robustness should be K-sensitive. If the gain is K-insensitive, the score's
correlation was never load-bearing — the gain comes from elsewhere
(implicit regularization, score-as-difficulty signal, training noise).

Method: vary warmup_epochs K in {3, 8, 30}; keep total epochs fixed at 50;
run Reweight-Only end-to-end (no gradient surgery); evaluate
accuracy_clean + robustness on test sets. Compare to SFT baseline.

Usage:
    CUDA_VISIBLE_DEVICES=2 EXPERIMENT_SCALE=server PYTHONUNBUFFERED=1 \\
        python -u scripts/reweight_only_K_sweep.py

Output:
    results/reweight_K_sweep.json
    results/reweight_K_sweep.png
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
    print(f"lambda = {C.lambda_}, tau_A = {C.tau_A}, tau_R = {C.tau_R}",
          flush=True)

    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'results')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'reweight_K_sweep.json')
    fig_path = os.path.join(out_dir, 'reweight_K_sweep.png')

    # Resume support
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

        # SFT baseline
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

        # Reweight-Only at each K
        for K in K_values:
            tag = f'RO_K{K}'
            if tag in results[ds_name]:
                print(f"  [{ds_name}] {tag} already done, skipping",
                      flush=True)
                continue
            print(f"\n--- [{ds_name}] Reweight-Only K={K} (total epochs=50) ---",
                  flush=True)
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
                use_gradient_surgery=False,
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
            print(f"  RO K={K} acc={r['accuracy_clean']:.3f}, "
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

    # Plot: accuracy + robustness vs K, per dataset, with SFT as horizontal lines.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ds_name in datasets:
        if ds_name not in results:
            continue
        sft = results[ds_name].get('SFT', {})
        accs = [results[ds_name].get(f'RO_K{K}', {}).get('accuracy_clean')
                for K in K_values]
        robs = [results[ds_name].get(f'RO_K{K}', {}).get('robustness')
                for K in K_values]

        axes[0].plot(K_values, accs, marker='o', label=ds_name)
        if sft.get('accuracy_clean') is not None:
            axes[0].axhline(y=sft['accuracy_clean'], linestyle=':', alpha=0.5,
                            label=f'{ds_name} SFT={sft["accuracy_clean"]:.2f}')

        axes[1].plot(K_values, robs, marker='o', label=ds_name)
        if sft.get('robustness') is not None:
            axes[1].axhline(y=sft['robustness'], linestyle=':', alpha=0.5,
                            label=f'{ds_name} SFT={sft["robustness"]:.2f}')

    axes[0].set_xlabel('Warmup epochs K')
    axes[0].set_ylabel('Accuracy (clean)')
    axes[0].set_title('Reweight-Only accuracy vs warmup K')
    axes[0].set_xscale('log')
    axes[0].legend(fontsize=7, loc='lower right')

    axes[1].set_xlabel('Warmup epochs K')
    axes[1].set_ylabel('Robustness')
    axes[1].set_title('Reweight-Only robustness vs warmup K')
    axes[1].set_xscale('log')
    axes[1].legend(fontsize=7, loc='lower right')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")


if __name__ == '__main__':
    main()
