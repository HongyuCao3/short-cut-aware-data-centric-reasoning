"""Multi-seed Surgery-Only verification.

Purpose: estimate true variance of SART (Path A surgery-only) on 3 synthetic
benchmarks at K in {3, 8}, after the May-2 finding that single-seed K-sweep
runs differ by up to 63 pp robustness on the same config (Causal SO K=8 was
0.795 on Apr-26, 0.162 today).

Method:
    seeds = [42, 43, 44, 45, 46]    # 5 seeds for the SART training (model init + RNG)
    datasets = Math, Financial, Causal  (dataset generation seeds stay 42/43/44)
    K in {3, 8}
    SO (use_reweighting=False, use_gradient_surgery=True)
    Plus 1 SFT control per dataset (deterministic, just for diff sanity)

Total: 5*3*2 = 30 SO trainings + 3 SFT = 33 trainings.
Estimate: ~33 * 250s ~ 138 min on H100 NVL clean GPU.

Output:
    results/multi_seed_so.json   keys: results[ds][f"SO_K{K}_s{seed}"] = {acc, rob, train_sec}
                                       results[ds]["SFT"] = {acc, rob, train_sec}
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset)
from src.model import create_model, count_parameters
from src.trainer import train_standard, train_our_method
from src.evaluate import run_full_evaluation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    SEEDS = [42, 43, 44, 45, 46]
    K_VALUES = [3, 8]
    TOTAL_EPOCHS = 50

    print(f"PROFILE = {PROFILE}, device = {C.device}", flush=True)
    print(f"seeds = {SEEDS}, K = {K_VALUES}, epochs = {TOTAL_EPOCHS}", flush=True)
    print(f"Variant: SURGERY-ONLY (use_reweighting=False)", flush=True)

    # Datasets generated with fixed seeds (independent of model-training seed)
    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'multi_seed_so.json')

    # Resume support
    results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                results = json.load(f)
            n_done = sum(len(v) for v in results.values())
            print(f"Resume: {n_done} cells already done", flush=True)
        except Exception as e:
            print(f"Could not load existing JSON ({e}); starting fresh", flush=True)
            results = {}

    def save():
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    tmp = create_model('cpu')
    print(f"\nSmallGPT params: {count_parameters(tmp):,}", flush=True)
    del tmp

    t_start = time.time()

    for ds_name, ds in datasets.items():
        results.setdefault(ds_name, {})

        # SFT control (single seed, deterministic enough)
        if 'SFT' not in results[ds_name]:
            print(f"\n--- [{ds_name}] SFT control ---", flush=True)
            set_seed(42)
            model = create_model()
            t0 = time.time()
            train_standard(model, ds, epochs=TOTAL_EPOCHS, verbose=False)
            t_train = time.time() - t0
            r = run_full_evaluation(model, ds, compute_f1=False)
            results[ds_name]['SFT'] = {
                'accuracy_clean': float(r['accuracy_clean']),
                'robustness': float(r['robustness']),
                'train_sec': t_train,
            }
            print(f"  SFT acc={r['accuracy_clean']:.3f}, rob={r['robustness']:.3f}, "
                  f"time={t_train:.1f}s", flush=True)
            save()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # SO at each (K, seed) combination
        for K in K_VALUES:
            for seed in SEEDS:
                tag = f'SO_K{K}_s{seed}'
                if tag in results[ds_name]:
                    print(f"  [{ds_name}] {tag} already done, skipping", flush=True)
                    continue

                print(f"\n--- [{ds_name}] {tag} (K={K}, seed={seed}, "
                      f"epochs={TOTAL_EPOCHS}) ---", flush=True)
                set_seed(seed)
                model = create_model()
                t0 = time.time()
                cfg_override = {
                    'epochs': TOTAL_EPOCHS,
                    'warmup_epochs': K,
                    'score_max_samples': 10000,
                }
                model, _ = train_our_method(
                    model, ds,
                    use_reweighting=False,
                    use_gradient_surgery=True,
                    collect_scores=False,
                    verbose=False,
                    cfg=cfg_override,
                )
                t_train = time.time() - t0
                r = run_full_evaluation(model, ds, compute_f1=False)
                results[ds_name][tag] = {
                    'accuracy_clean': float(r['accuracy_clean']),
                    'robustness': float(r['robustness']),
                    'train_sec': t_train,
                    'warmup_epochs': K,
                    'seed': seed,
                }
                print(f"  {tag} acc={r['accuracy_clean']:.3f}, "
                      f"rob={r['robustness']:.3f}, time={t_train:.1f}s", flush=True)
                save()
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start
    print(f"\n=== Multi-seed sweep done in {t_elapsed:.1f}s "
          f"({t_elapsed/60:.1f} min) ===", flush=True)

    # Summary stats
    print("\n=== Summary (mean ± std over seeds) ===", flush=True)
    for ds_name in datasets:
        print(f"\n{ds_name}:", flush=True)
        if 'SFT' in results[ds_name]:
            r = results[ds_name]['SFT']
            print(f"  SFT             acc={r['accuracy_clean']:.3f}  "
                  f"rob={r['robustness']:.3f}", flush=True)
        for K in K_VALUES:
            accs = [results[ds_name][f'SO_K{K}_s{s}']['accuracy_clean']
                    for s in SEEDS if f'SO_K{K}_s{s}' in results[ds_name]]
            robs = [results[ds_name][f'SO_K{K}_s{s}']['robustness']
                    for s in SEEDS if f'SO_K{K}_s{s}' in results[ds_name]]
            if accs:
                print(f"  SO K={K:<2d} (n={len(accs)})  "
                      f"acc={np.mean(accs):.3f}±{np.std(accs):.3f}  "
                      f"rob={np.mean(robs):.3f}±{np.std(robs):.3f}  "
                      f"[range rob: {min(robs):.3f} – {max(robs):.3f}]",
                      flush=True)


if __name__ == '__main__':
    main()
