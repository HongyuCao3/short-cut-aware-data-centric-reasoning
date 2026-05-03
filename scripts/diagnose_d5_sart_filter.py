"""D5: SART-as-filter — use S(s) ranking + threshold filtering, no surgery, SFT.

Tests whether SART's detection (S(s)) signal alone is useful as a filtering
criterion when surgery is removed. If S(s) is a competent ranker for shortcut
samples, dropping top-k% by S(s) and SFT'ing the rest should approach D4's
oracle-drop ceiling of 1.000/1.000.

Method:
  Phase 1: standard 8-epoch warmup
  Phase 2: compute ShortcutScore S(s) for all 10000 training samples
  Filter: drop top 70% by S(s) (matching D4's drop rate of ~70% shortcut)
  Phase 3: SFT on remaining 30% for 50 epochs (matching D4)

Reports filter precision (of dropped samples, how many really shortcut?) and
recall (of all shortcut samples, how many got dropped?). The kept set's clean
fraction tells us how close S(s)-ranking is to oracle is_shortcut.

If acc/rob lifts close to 1.000: SART can be reframed as "ShortcutScore as a
data filter" — surgery is empirically dispensable.
If acc/rob stays low: detection itself is broken at this small-model scale.
"""
from __future__ import annotations

import copy
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
from src.trainer import train_standard, _compute_sample_scores
from src.evaluate import run_full_evaluation


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print(f"PROFILE = {PROFILE}, device = {C.device}", flush=True)
    print(f"D5: SART-as-filter — S(s) ranking + drop top-k%, then SFT",
          flush=True)

    DROP_FRACTION = 0.70
    K_WARMUP = 8
    EPOCHS_FINAL = 50

    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'results')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'diagnose_d5_sart_filter.json')

    results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                results = json.load(f)
        except Exception:
            results = {}

    def save():
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    tmp = create_model('cpu')
    print(f"\nSmallGPT params: {count_parameters(tmp):,}", flush=True)
    del tmp

    t_start = time.time()

    order = ['Causal-Reasoning', 'Math-Arithmetic', 'Financial-Analysis']
    for ds_name in order:
        if ds_name in results and 'D5' in results[ds_name]:
            print(f"\n[{ds_name}] D5 already done, skipping", flush=True)
            continue

        print(f"\n=========================================================",
              flush=True)
        print(f"=== Dataset: {ds_name} ===", flush=True)
        print(f"=========================================================",
              flush=True)

        ds = datasets[ds_name]
        results.setdefault(ds_name, {})

        # Phase 1: warmup
        print(f"  Phase 1: Warmup ({K_WARMUP} epochs)", flush=True)
        set_seed(42)
        model = create_model()
        train_standard(model, ds, epochs=K_WARMUP, verbose=False)

        # Phase 2: compute ShortcutScores
        print(f"  Phase 2: Computing ShortcutScores...", flush=True)
        sample_scores, collected_data, g_V = _compute_sample_scores(model, ds)

        is_shortcut = np.array(
            [bool(s['is_shortcut']) for s in ds['train'].samples])
        scores = np.array(sample_scores)
        n_total = len(scores)

        # Filter: drop top DROP_FRACTION by S(s) (highest scores are most
        # shortcut-like under the ShortcutScore design)
        n_drop = int(n_total * DROP_FRACTION)
        n_keep = n_total - n_drop
        sorted_idx = np.argsort(scores)         # ascending
        keep_idx = sorted_idx[:n_keep]
        drop_idx = sorted_idx[n_keep:]

        # Filter quality vs oracle
        kept_clean = int((~is_shortcut[keep_idx]).sum())
        dropped_short = int(is_shortcut[drop_idx].sum())
        oracle_n_short = int(is_shortcut.sum())
        precision = dropped_short / max(n_drop, 1)
        recall = dropped_short / max(oracle_n_short, 1)
        kept_clean_pct = kept_clean / max(n_keep, 1)

        print(f"  Filter: dropped {n_drop}/{n_total} "
              f"({100*DROP_FRACTION:.0f}% by S(s) ranking)", flush=True)
        print(f"    Precision (dropped & actually shortcut): "
              f"{100*precision:.1f}%", flush=True)
        print(f"    Recall (shortcut samples found by filter): "
              f"{100*recall:.1f}%", flush=True)
        print(f"    Kept set clean fraction: {100*kept_clean_pct:.1f}% "
              f"(oracle drop achieves 100.0%)", flush=True)

        # Build filtered dataset
        ds_filtered = {k: v for k, v in ds.items()}
        train_filt = copy.copy(ds['train'])
        train_filt.samples = [ds['train'].samples[i] for i in keep_idx]
        ds_filtered['train'] = train_filt

        # Phase 3: SFT on filtered set
        print(f"  Phase 3: SFT on {n_keep} kept samples "
              f"({EPOCHS_FINAL} epochs)", flush=True)
        set_seed(42)
        model_filt = create_model()
        t0 = time.time()
        train_standard(model_filt, ds_filtered, epochs=EPOCHS_FINAL,
                       verbose=False)
        t_train = time.time() - t0

        r = run_full_evaluation(model_filt, ds, compute_f1=False)
        results[ds_name]['D5'] = {
            'accuracy_clean': float(r['accuracy_clean']),
            'robustness': float(r['robustness']),
            'train_sec': t_train,
            'drop_fraction': DROP_FRACTION,
            'n_kept': int(n_keep),
            'n_dropped': int(n_drop),
            'filter_precision_pct': float(100 * precision),
            'filter_recall_pct': float(100 * recall),
            'kept_clean_pct': float(100 * kept_clean_pct),
        }
        print(f"\n  Result: acc={r['accuracy_clean']:.3f}, "
              f"rob={r['robustness']:.3f}, time={t_train:.1f}s", flush=True)
        save()
        del model, model_filt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start
    print(f"\n=== Done in {t_elapsed:.1f}s ({t_elapsed/60:.1f} min) ===",
          flush=True)

    print("\n=== Summary ===", flush=True)
    print(f"{'Dataset':<22s}  {'Acc':>6s}  {'Rob':>6s}  "
          f"{'Filter P':>9s}  {'Filter R':>9s}  {'Kept Clean %':>14s}",
          flush=True)
    for ds_name in order:
        if ds_name in results and 'D5' in results[ds_name]:
            m = results[ds_name]['D5']
            print(f"{ds_name:<22s}  {m['accuracy_clean']:>6.3f}  "
                  f"{m['robustness']:>6.3f}  "
                  f"{m['filter_precision_pct']:>8.1f}%  "
                  f"{m['filter_recall_pct']:>8.1f}%  "
                  f"{m['kept_clean_pct']:>13.1f}%", flush=True)


if __name__ == '__main__':
    main()
