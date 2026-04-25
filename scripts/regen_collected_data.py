"""Regenerate collected_data for Figure 2 diagnostics (Gaps 1/2/3a).

Runs SART (Reweighting + Gradient Surgery) on the three synthetic datasets,
captures the per-sample (S, A, R, is_shortcut) tuples in `collected_data`, and
dumps to pickle. Then renders the updated figure3.png with diagnostic numbers.

CPU-friendly defaults (configurable via env vars):
    SART_REGEN_EPOCHS         total training epochs (default 12; paper uses 50)
    SART_REGEN_MAX_SAMPLES    score_max_samples per dataset (default 2000;
                              paper uses 10000)

Usage:
    CUDA_VISIBLE_DEVICES="" EXPERIMENT_SCALE=server \\
        python -u scripts/regen_collected_data.py

Output:
    results/collected_data_synthetic.pkl
    results/figure3.png        (with Gap 1 / 2 / 3a diagnostics)
    results/figure3_diagnostics.json
"""
import os
import sys
import time
import pickle
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset)
from src.model import create_model, count_parameters
from src.trainer import train_our_method
from src.visualize import generate_figure3


def set_seed(seed=42):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    epochs = int(os.environ.get('SART_REGEN_EPOCHS', '8'))
    max_samples = int(os.environ.get('SART_REGEN_MAX_SAMPLES', '1500'))
    cfg_override = {'epochs': epochs, 'score_max_samples': max_samples}

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'collected_data_synthetic.pkl')

    # Resume support: if a partial pickle exists, skip already-completed datasets.
    collected_data_all = {}
    timings = {}
    if os.path.exists(out_path):
        try:
            with open(out_path, 'rb') as f:
                snap = pickle.load(f)
            collected_data_all = snap.get('collected_data_all', {})
            timings = snap.get('timings', {})
            print(f"Resuming: found existing snapshot with "
                  f"{len(collected_data_all)} datasets done", flush=True)
        except Exception as e:
            print(f"Could not load existing snapshot ({e}); starting fresh",
                  flush=True)
            collected_data_all, timings = {}, {}

    print(f"PROFILE = {PROFILE}, device = {C.device}", flush=True)
    print(f"score_max_samples = {max_samples} (config default: {C.score_max_samples}), "
          f"n_train = {C.n_train}, sketch_k = {C.sketch_k}", flush=True)
    print(f"epochs = {epochs} (config default: {C.epochs})", flush=True)
    print(f"lambda = {C.lambda_}, tau_A = {C.tau_A}, tau_R = {C.tau_R}, "
          f"alpha = {C.alpha}, beta = {C.beta}", flush=True)

    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    tmp = create_model('cpu')
    print(f"\nSmallGPT params: {count_parameters(tmp):,}")
    del tmp

    def save_snapshot():
        with open(out_path, 'wb') as f:
            pickle.dump({
                'collected_data_all': collected_data_all,
                'config_snapshot': {
                    'profile': PROFILE,
                    'lambda': C.lambda_,
                    'tau_A': C.tau_A,
                    'tau_R': C.tau_R,
                    'alpha': C.alpha,
                    'beta': C.beta,
                    'score_max_samples': max_samples,
                    'sketch_k': C.sketch_k,
                    'n_train': C.n_train,
                    'epochs': epochs,
                },
                'timings': timings,
            }, f)

    for ds_name, ds in datasets.items():
        if ds_name in collected_data_all and collected_data_all[ds_name].get('scores'):
            print(f"\nSkipping {ds_name} (already in snapshot, "
                  f"n={len(collected_data_all[ds_name]['scores'])})", flush=True)
            continue
        print(f"\n{'=' * 60}\nDataset: {ds_name}", flush=True)
        print(f"  train={len(ds['train'])}, val={len(ds['val'])}", flush=True)
        set_seed()
        model = create_model()
        t0 = time.time()
        model, collected = train_our_method(
            model, ds,
            use_reweighting=True,
            use_gradient_surgery=True,
            collect_scores=True,
            cfg=cfg_override,
        )
        elapsed = time.time() - t0
        timings[ds_name] = elapsed
        print(f"  SART train+score time: {elapsed:.1f}s", flush=True)
        print(f"  collected sizes: scores={len(collected['scores'])}, "
              f"alignments={len(collected['alignments'])}, "
              f"is_shortcut={len(collected['is_shortcut'])}", flush=True)

        collected_data_all[ds_name] = collected
        save_snapshot()
        print(f"  Snapshot updated: {len(collected_data_all)}/3 datasets done",
              flush=True)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_snapshot()
    print(f"\nFinal dump of collected_data to {out_path}", flush=True)
    total = sum(timings.values())
    print(f"Total wall time: {total:.1f}s ({total/60:.1f} min)", flush=True)

    # ------------------------------------------------------------
    # Render figure3.png with the Gap 1/2/3a diagnostics
    # ------------------------------------------------------------
    print("\n--- Rendering figure3.png with diagnostics (Gap 1/2/3a) ---")
    fig_path, summary = generate_figure3(collected_data_all)
    summary_path = os.path.join(out_dir, 'figure3_diagnostics.json')
    import json
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Diagnostic summary dumped to {summary_path}")


if __name__ == '__main__':
    main()
