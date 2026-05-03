"""SART failure-mode diagnostics — D1 (gate fire rate) + D4 (oracle-drop).

D1: Run Path A surgery-only K=8 on each of 3 datasets with verbose=True.
    Trainer is now instrumented to print per-epoch and total gate-fire rate +
    cos_sim distribution. Tells us whether F1 (batch averaging masks signal)
    is actually happening: a gate that fires <5% of the time means surgery
    is essentially inert; a gate that fires >50% but the model still
    shortcut-locks means F2/F3 (no S* selection / no answer-token suppression)
    are the bottleneck.

D4: For each dataset, drop all ground-truth shortcut samples from training,
    train SFT for 50 epochs, evaluate. Gives the oracle upper bound for any
    SART-like detection-then-correction method on these benchmarks.

Output:
    results/diagnose_sart.log (stdout)
    results/diagnose_sart.json
        {ds: {
            "SO_K8_d1": {acc, rob, gate_fire_rate_pct, cos_sim_mean,
                         cos_sim_min, cos_sim_max, train_sec},
            "OracleDrop": {acc, rob, n_kept, n_dropped, train_sec},
        }}
"""
from __future__ import annotations

import io
import json
import os
import random
import sys
import time
import contextlib
import re

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset)
from src.model import create_model, count_parameters
from src.trainer import train_standard, train_our_method
from src.evaluate import run_full_evaluation


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_gate_diagnostics(captured_stdout):
    """Extract gate-fire stats from the [Gate diagnostics] line printed by trainer.py."""
    # Example: "  [Gate diagnostics] total fires 47/3360 (1.4%); cos_sim range [-0.123, 0.847], mean 0.412"
    m = re.search(
        r'\[Gate diagnostics\] total fires (\d+)/(\d+) \(([\d.]+)%\); '
        r'cos_sim range \[(-?[\d.]+), (-?[\d.]+)\], mean (-?[\d.]+)',
        captured_stdout)
    if not m:
        return None
    return {
        'fires': int(m.group(1)),
        'batches': int(m.group(2)),
        'fire_rate_pct': float(m.group(3)),
        'cos_sim_min': float(m.group(4)),
        'cos_sim_max': float(m.group(5)),
        'cos_sim_mean': float(m.group(6)),
    }


def make_oracle_dropped_dataset(ds):
    """Return a copy of `ds` with all is_shortcut=True samples removed from train split.

    `ds` is a dict like {'train': SyntheticDataset, 'val': ..., 'test_clean': ..., 'test_perturbed': ...}.
    SyntheticDataset.samples is a list of dicts; each has 'is_shortcut' bool.
    """
    import copy
    ds_oracle = {k: v for k, v in ds.items()}
    train_orig = ds['train']
    # Shallow copy the dataset object then filter samples
    train_oracle = copy.copy(train_orig)
    train_oracle.samples = [s for s in train_orig.samples if not s['is_shortcut']]
    ds_oracle['train'] = train_oracle
    return ds_oracle, len(train_orig.samples), len(train_oracle.samples)


def run_diag_for_dataset(ds_name, ds, results, save_fn):
    print(f"\n=========================================================", flush=True)
    print(f"=== Dataset: {ds_name} ===", flush=True)
    print(f"=========================================================", flush=True)

    results.setdefault(ds_name, {})

    # ---- D1: SO K=8, capture gate-fire stats ----
    if 'SO_K8_d1' not in results[ds_name]:
        print(f"\n--- [D1] Surgery-Only K=8 with gate diagnostics ---",
              flush=True)
        set_seed(42)
        model = create_model()
        cfg_override = {
            'epochs': 50,
            'warmup_epochs': 8,
            'score_max_samples': 10000,
        }
        # Capture stdout to parse the gate diagnostics line
        buf = io.StringIO()
        t0 = time.time()
        with contextlib.redirect_stdout(buf):
            model_out = train_our_method(
                model, ds,
                use_reweighting=False,
                use_gradient_surgery=True,
                collect_scores=False,
                verbose=True,
                cfg=cfg_override,
            )
        # train_our_method returns model when collect_scores=False
        model = model_out if not isinstance(model_out, tuple) else model_out[0]
        captured = buf.getvalue()
        print(captured, flush=True)
        t_train = time.time() - t0

        gate_stats = parse_gate_diagnostics(captured)
        r = run_full_evaluation(model, ds, compute_f1=False)
        results[ds_name]['SO_K8_d1'] = {
            'accuracy_clean': float(r['accuracy_clean']),
            'robustness': float(r['robustness']),
            'train_sec': t_train,
            'gate_stats': gate_stats,
        }
        print(f"  D1 acc={r['accuracy_clean']:.3f}, rob={r['robustness']:.3f}, "
              f"time={t_train:.1f}s", flush=True)
        if gate_stats:
            print(f"  D1 gate_fire_rate={gate_stats['fire_rate_pct']:.1f}%, "
                  f"cos_sim mean={gate_stats['cos_sim_mean']:.3f}, "
                  f"range=[{gate_stats['cos_sim_min']:.3f}, "
                  f"{gate_stats['cos_sim_max']:.3f}]", flush=True)
        else:
            print("  D1 (gate stats not parsed — check log)", flush=True)
        save_fn()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- D4: Oracle drop + SFT ----
    if 'OracleDrop' not in results[ds_name]:
        print(f"\n--- [D4] Oracle-drop SFT (drop is_shortcut=True samples) ---",
              flush=True)
        ds_oracle, n_orig, n_kept = make_oracle_dropped_dataset(ds)
        n_dropped = n_orig - n_kept
        print(f"  Train: {n_orig} -> {n_kept} ({n_dropped} dropped, "
              f"{100*n_dropped/n_orig:.1f}%)", flush=True)
        set_seed(42)
        model = create_model()
        t0 = time.time()
        train_standard(model, ds_oracle, epochs=50, verbose=False)
        t_train = time.time() - t0
        r = run_full_evaluation(model, ds, compute_f1=False)
        results[ds_name]['OracleDrop'] = {
            'accuracy_clean': float(r['accuracy_clean']),
            'robustness': float(r['robustness']),
            'n_kept': n_kept,
            'n_dropped': n_dropped,
            'train_sec': t_train,
        }
        print(f"  D4 acc={r['accuracy_clean']:.3f}, rob={r['robustness']:.3f}, "
              f"time={t_train:.1f}s", flush=True)
        save_fn()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    print(f"PROFILE = {PROFILE}, device = {C.device}", flush=True)
    print(f"Diagnostic: D1 (gate fire rate) + D4 (oracle drop)", flush=True)

    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'diagnose_sart.json')

    results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                results = json.load(f)
            print(f"Resume: {sum(len(v) for v in results.values())} cells "
                  f"already done", flush=True)
        except Exception:
            results = {}

    def save():
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    tmp = create_model('cpu')
    print(f"\nSmallGPT params: {count_parameters(tmp):,}", flush=True)
    del tmp

    t_start = time.time()

    # Causal first because it's the failure mode we most need to understand
    order = ['Causal-Reasoning', 'Math-Arithmetic', 'Financial-Analysis']
    for ds_name in order:
        run_diag_for_dataset(ds_name, datasets[ds_name], results, save)

    t_elapsed = time.time() - t_start
    print(f"\n=== Diagnostics done in {t_elapsed:.1f}s "
          f"({t_elapsed/60:.1f} min) ===", flush=True)

    # Summary
    print("\n=== Summary ===", flush=True)
    print(f"{'Dataset':<22s}  {'Method':<14s}  {'Acc':>6s}  {'Rob':>6s}  "
          f"{'Gate Fire %':>12s}  {'cos_sim Mean':>13s}", flush=True)
    for ds_name in order:
        if ds_name in results:
            for variant, m in results[ds_name].items():
                gs = m.get('gate_stats') or {}
                fr = f"{gs.get('fire_rate_pct', 0):.1f}" if gs else '-'
                cm = f"{gs.get('cos_sim_mean', 0):.3f}" if gs else '-'
                print(f"{ds_name:<22s}  {variant:<14s}  "
                      f"{m['accuracy_clean']:>6.3f}  {m['robustness']:>6.3f}  "
                      f"{fr:>12s}  {cm:>13s}", flush=True)


if __name__ == '__main__':
    main()
