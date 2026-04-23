#!/usr/bin/env python3
"""Quick-verify harness: does the PCGrad fix preserve SART's Math result?

Runs only SART-Full and Reweight-Only on Math-Arithmetic at FULL server
scale scoring (10k samples), but skips the expensive F1 eval so each
method lands in ~3 min rather than ~12 min. Enough to know if SART's
+95 pp robustness survived the conflict-only projection rule.

Usage:
  CUDA_VISIBLE_DEVICES=<free-gpu> EXPERIMENT_SCALE=server \
      python3 -u run_verify.py
"""
import os
import sys
import time
import json
import random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(line_buffering=True)

from src.config import Config as C, PROFILE
from src.data import generate_math_dataset, get_dataloader
from src.model import create_model
from src.trainer import train_standard, train_our_method
from src.evaluate import run_full_evaluation, evaluate_gradient_alignment


OUT_DIR = os.environ.get(
    'SART_DATA_ROOT',
    '/data/hongyuca/short-cut-aware-data-centric-reasoning',
) + '/logs/pcgrad-rerun'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, 'verify_results.json')


def set_seed(seed=C.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _fmt(val, spec='.3f'):
    if val is None:
        return '   -  '
    return format(val, spec)


def evaluate_quick(model, ds):
    """Run the cheap parts of evaluation: accuracy + robustness + reasoning
    + gradient alignment on a small subset. Skip per-sample F1."""
    r = run_full_evaluation(
        model, ds, compute_f1=False, compute_alignment=False,
    )
    val_loader = get_dataloader(ds['val'], batch_size=C.batch_size, shuffle=False)
    r['gradient_alignment'] = evaluate_gradient_alignment(
        model, ds['train'], val_loader,
    )
    return r


def main():
    print('=' * 70, flush=True)
    print(f'PCGrad-fix QUICK VERIFY (profile={PROFILE})', flush=True)
    print(f'  Dataset: Math-Arithmetic (full 10k scoring, no F1)', flush=True)
    print(f'  Output:  {OUT_JSON}', flush=True)
    print('=' * 70, flush=True)

    print('\n--- Generating Math-Arithmetic ---', flush=True)
    ds = generate_math_dataset(seed=42)
    print(f'  train={len(ds["train"])} val={len(ds["val"])}', flush=True)

    # First: SFT baseline for reference
    # (cheap, no scoring pass; gives a sanity anchor)
    print('\n[1/3] SFT (baseline anchor)...', flush=True)
    t0 = time.time()
    set_seed(); m = create_model()
    train_standard(m, ds, verbose=False)
    r = evaluate_quick(m, ds)
    r['train_time_sec'] = time.time() - t0
    print(f'  SFT              acc={_fmt(r["accuracy_clean"])} '
          f'rob={_fmt(r["robustness"])} '
          f'align={_fmt(r["gradient_alignment"])} '
          f'time={r["train_time_sec"]:.1f}s', flush=True)
    results = {'sft': r}
    del m; torch.cuda.empty_cache()

    # Phase-2 / 3 methods use FULL 10k scoring (no truncation) so
    # reweighting reaches all training samples. sketch_k kept at server
    # default (128) for fidelity to the published pipeline.
    cfg_override = {'score_max_samples': 10000, 'sketch_k': 128}

    # Reweight-only: does reweighting at full 10k score scale recover rob > 0?
    print('\n[2/3] Reweight-Only (full 10k scoring)...', flush=True)
    t0 = time.time()
    set_seed(); m = create_model()
    m = train_our_method(
        m, ds,
        use_reweighting=True, use_gradient_surgery=False,
        cfg=cfg_override,
    )
    if isinstance(m, tuple):
        m = m[0]
    r = evaluate_quick(m, ds)
    r['train_time_sec'] = time.time() - t0
    print(f'  Reweight-Only    acc={_fmt(r["accuracy_clean"])} '
          f'rob={_fmt(r["robustness"])} '
          f'align={_fmt(r["gradient_alignment"])} '
          f'time={r["train_time_sec"]:.1f}s', flush=True)
    results['reweight_only'] = r
    del m; torch.cuda.empty_cache()

    # SART-Full: the verdict for the PCGrad fix.
    print('\n[3/3] SART-Full (full 10k scoring, PCGrad-gated surgery)...',
          flush=True)
    t0 = time.time()
    set_seed(); m = create_model()
    m = train_our_method(
        m, ds,
        use_reweighting=True, use_gradient_surgery=True,
        cfg=cfg_override,
    )
    if isinstance(m, tuple):
        m = m[0]
    r = evaluate_quick(m, ds)
    r['train_time_sec'] = time.time() - t0
    print(f'  SART-Full        acc={_fmt(r["accuracy_clean"])} '
          f'rob={_fmt(r["robustness"])} '
          f'align={_fmt(r["gradient_alignment"])} '
          f'time={r["train_time_sec"]:.1f}s', flush=True)
    results['sart_full'] = r
    del m; torch.cuda.empty_cache()

    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print('\n' + '=' * 70, flush=True)
    print('VERDICT (Math-Arithmetic, paper Table 2 ref: SFT 77.6/3.2, '
          'SART 98.0/95.8)', flush=True)
    print(f'  SFT            acc={results["sft"]["accuracy_clean"]:.3f} '
          f'rob={results["sft"]["robustness"]:.3f}', flush=True)
    print(f'  Reweight-Only  acc={results["reweight_only"]["accuracy_clean"]:.3f} '
          f'rob={results["reweight_only"]["robustness"]:.3f}', flush=True)
    print(f'  SART-Full      acc={results["sart_full"]["accuracy_clean"]:.3f} '
          f'rob={results["sart_full"]["robustness"]:.3f}', flush=True)
    print('=' * 70, flush=True)


if __name__ == '__main__':
    main()
