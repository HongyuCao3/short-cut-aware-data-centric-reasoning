"""Warmup-epoch sweep for Gap 4 (score-measurement-time issue).

For each warmup-epoch budget K in a sweep set, trains the model for K epochs
of standard supervised learning, then runs the SART score pass to capture
(S, A, R, is_shortcut) tuples. Records r_bin and r_sample at each K.

Goal: locate the K (if any) where r approaches the paper's reported 0.67,
or confirm that the paper's number is not reproducible from the committed
code at any standard training state.

Usage:
    CUDA_VISIBLE_DEVICES=2 EXPERIMENT_SCALE=server PYTHONUNBUFFERED=1 \\
        python -u scripts/warmup_epoch_sweep.py

Output:
    results/warmup_sweep.pkl              # raw collected_data per (ds, K)
    results/warmup_sweep_r.json           # r_bin / r_sample table
    results/warmup_sweep.png              # r vs warmup-epochs line plot
"""
import os
import sys
import time
import pickle
import json

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset)
from src.model import create_model, count_parameters
from src.trainer import train_standard, _compute_sample_scores

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_r_pair(scores, is_sc, n_bins=20):
    """Return (r_bin, r_sample) on (S, is_shortcut) pairs."""
    scores = np.asarray(scores, dtype=float)
    is_sc = np.asarray(is_sc, dtype=float)
    if len(scores) < 2 or scores.std() == 0:
        return float('nan'), float('nan')
    r_sample, _ = pearsonr(scores, is_sc)
    edges = np.linspace(scores.min(), scores.max(), n_bins + 1)
    centers, rates = [], []
    for i in range(n_bins):
        mask = (scores >= edges[i]) & (scores < edges[i + 1])
        if mask.sum() > 0:
            centers.append((edges[i] + edges[i + 1]) / 2)
            rates.append(is_sc[mask].mean())
    if len(centers) > 1 and np.std(rates) > 0:
        r_bin = float(np.corrcoef(centers, rates)[0, 1])
    else:
        r_bin = float('nan')
    return r_bin, float(r_sample)


def main():
    sweep_K = [1, 2, 3, 5, 8, 12, 20, 30, 50]
    score_max = int(os.environ.get('SWEEP_MAX_SAMPLES', '10000'))
    cfg_override = {'score_max_samples': score_max}

    print(f"PROFILE = {PROFILE}, device = {C.device}", flush=True)
    print(f"Sweep K = {sweep_K}, score_max_samples = {score_max}", flush=True)
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
    pkl_path = os.path.join(out_dir, 'warmup_sweep.pkl')
    json_path = os.path.join(out_dir, 'warmup_sweep_r.json')
    fig_path = os.path.join(out_dir, 'warmup_sweep.png')

    # Resume support: load partial state if available.
    sweep_data = {}
    r_table = {}
    if os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'rb') as f:
                sweep_data = pickle.load(f)
            print(f"Resume: loaded existing sweep with "
                  f"{sum(len(v) for v in sweep_data.values())} (ds, K) pairs",
                  flush=True)
        except Exception as e:
            print(f"Could not load existing sweep ({e}); starting fresh",
                  flush=True)
            sweep_data = {}

    tmp = create_model('cpu')
    print(f"\nSmallGPT params: {count_parameters(tmp):,}", flush=True)
    del tmp

    def save_snapshots():
        with open(pkl_path, 'wb') as f:
            pickle.dump(sweep_data, f)
        with open(json_path, 'w') as f:
            json.dump(r_table, f, indent=2, default=str)

    t_start = time.time()
    for ds_name, ds in datasets.items():
        sweep_data.setdefault(ds_name, {})
        for K in sweep_K:
            if K in sweep_data[ds_name]:
                print(f"  [{ds_name}] K={K} already done, skipping",
                      flush=True)
                continue
            print(f"\n--- [{ds_name}] warmup_epochs = {K} ---", flush=True)
            set_seed()
            model = create_model()
            t0 = time.time()
            train_standard(model, ds, epochs=K, verbose=False)
            t_train = time.time() - t0
            t0 = time.time()
            _, collected, _ = _compute_sample_scores(
                model, ds, cfg=cfg_override)
            t_score = time.time() - t0

            r_bin, r_sample = compute_r_pair(
                collected['scores'], collected['is_shortcut'])
            print(f"  train={t_train:.1f}s, score={t_score:.1f}s, "
                  f"n={len(collected['scores'])}, "
                  f"r_bin={r_bin:+.3f}, r_sample={r_sample:+.3f}",
                  flush=True)

            sweep_data[ds_name][K] = collected
            r_table.setdefault(ds_name, {})[K] = {
                'r_bin': r_bin,
                'r_sample': r_sample,
                'n_samples': len(collected['scores']),
                'avg_score': float(np.mean(collected['scores'])),
                'avg_alignment': float(np.mean(collected['alignments'])),
                'train_sec': t_train,
                'score_sec': t_score,
            }
            save_snapshots()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    save_snapshots()
    total = time.time() - t_start
    print(f"\n=== Sweep done in {total:.1f}s ({total/60:.1f} min) ===",
          flush=True)

    # ---------------- Plot r vs warmup epochs ----------------
    # Aggregate across datasets too (concatenate, then compute)
    agg_r_bin, agg_r_sample = {}, {}
    for K in sweep_K:
        agg_scores, agg_is_sc = [], []
        for ds_name in datasets:
            if K in sweep_data.get(ds_name, {}):
                agg_scores.extend(sweep_data[ds_name][K]['scores'])
                agg_is_sc.extend(sweep_data[ds_name][K]['is_shortcut'])
        if len(agg_scores) > 0:
            rb, rs = compute_r_pair(agg_scores, agg_is_sc)
            agg_r_bin[K] = rb
            agg_r_sample[K] = rs
    r_table['_aggregate_3datasets'] = {
        K: {'r_bin': agg_r_bin.get(K), 'r_sample': agg_r_sample.get(K)}
        for K in sweep_K}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    Ks = np.array(sweep_K)

    # Per-dataset r_bin curves
    for ds_name in datasets:
        if ds_name not in r_table:
            continue
        rb = [r_table[ds_name].get(K, {}).get('r_bin', np.nan) for K in sweep_K]
        axes[0].plot(Ks, rb, marker='o', label=ds_name)
    rb_agg = [agg_r_bin.get(K, np.nan) for K in sweep_K]
    axes[0].plot(Ks, rb_agg, marker='s', linestyle='--', linewidth=2,
                 color='black', label='Aggregate (3 ds)')
    axes[0].axhline(y=0.67, color='red', linestyle=':', alpha=0.6,
                    label='paper: r=0.67')
    axes[0].axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_xlabel('Warmup epochs K')
    axes[0].set_ylabel('r_bin (S vs is_shortcut)')
    axes[0].set_title('r_bin vs warmup epochs')
    axes[0].legend(fontsize=8, loc='lower left')
    axes[0].set_ylim(-1.0, 1.0)

    # r_sample curves
    for ds_name in datasets:
        if ds_name not in r_table:
            continue
        rs = [r_table[ds_name].get(K, {}).get('r_sample', np.nan)
              for K in sweep_K]
        axes[1].plot(Ks, rs, marker='o', label=ds_name)
    rs_agg = [agg_r_sample.get(K, np.nan) for K in sweep_K]
    axes[1].plot(Ks, rs_agg, marker='s', linestyle='--', linewidth=2,
                 color='black', label='Aggregate (3 ds)')
    axes[1].axhline(y=0.0, color='gray', linestyle='-', alpha=0.3)
    axes[1].set_xscale('log')
    axes[1].set_xlabel('Warmup epochs K')
    axes[1].set_ylabel('r_sample')
    axes[1].set_title('r_sample vs warmup epochs')
    axes[1].legend(fontsize=8, loc='lower left')
    axes[1].set_ylim(-0.5, 0.5)

    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {fig_path}")
    print(f"Saved {json_path}")
    print(f"Saved {pkl_path}")


if __name__ == '__main__':
    main()
