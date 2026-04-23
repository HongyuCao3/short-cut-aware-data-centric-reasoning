#!/usr/bin/env python3
"""Targeted re-run for PCGrad-fix validation.

Runs the 6 methods most relevant to the PCGrad-fix sanity check across
all 3 synthetic datasets. Writes per-run metrics to JSON and prints
progress unbuffered.

Usage:
  CUDA_VISIBLE_DEVICES=3 EXPERIMENT_SCALE=server python3 -u run_targeted.py
"""
import os
import sys
import time
import json
import random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Force line-buffered stdout so `tee` sees progress live.
sys.stdout.reconfigure(line_buffering=True)

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset, get_dataloader)
from src.model import create_model
from src.trainer import (train_standard, train_data_filtering,
                         train_group_dro, train_our_method)
from src.evaluate import (run_full_evaluation, evaluate_gradient_alignment)


OUT_DIR = os.environ.get(
    'SART_DATA_ROOT',
    '/data/hongyuca/short-cut-aware-data-centric-reasoning',
) + '/logs/pcgrad-rerun'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, 'targeted_results.json')


def set_seed(seed=C.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, ds, f1=False, alignment=False):
    results = run_full_evaluation(
        model, ds, compute_f1=f1, compute_alignment=alignment,
    )
    if alignment and 'gradient_alignment' not in results:
        val_loader = get_dataloader(
            ds['val'], batch_size=C.batch_size, shuffle=False,
        )
        results['gradient_alignment'] = evaluate_gradient_alignment(
            model, ds['train'], val_loader,
        )
    return results


def run_one(name, train_fn, ds, **train_kwargs):
    print(f'    [{name}] training...', flush=True)
    t0 = time.time()
    set_seed()
    model = create_model()
    out = train_fn(model, ds, **train_kwargs)
    # Some trainers return (model, collected); we only need the model here.
    if isinstance(out, tuple):
        model = out[0]
    elif out is not None:
        model = out
    train_time = time.time() - t0

    print(f'    [{name}] evaluating...', flush=True)
    f1 = name in ('data_filtering', 'sart_full', 'reweight_only', 'gs_only')
    alignment = name in ('sart_full', 'reweight_only', 'gs_only')
    r = evaluate(model, ds, f1=f1, alignment=alignment)
    r['train_time_sec'] = train_time

    def _fmt(val, spec='.3f'):
        """Format a scalar, tolerating None from methods that don't compute it."""
        if val is None:
            return '   -  '
        return format(val, spec)

    print(
        f'    [{name}] '
        f'acc={_fmt(r.get("accuracy_clean"))} '
        f'rob={_fmt(r.get("robustness"))} '
        f'reason={_fmt(r.get("reasoning_consistency"))} '
        f'f1={_fmt(r.get("shortcut_f1"))} '
        f'align={_fmt(r.get("gradient_alignment"))} '
        f'time={train_time:.1f}s',
        flush=True,
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r


def main():
    print('=' * 70, flush=True)
    print(f'Targeted PCGrad-fix re-run (profile={PROFILE})', flush=True)
    print(f'Output JSON: {OUT_JSON}', flush=True)
    print('=' * 70, flush=True)

    print('\n--- Generating synthetic datasets ---', flush=True)
    datasets = {
        'Math-Arithmetic':     generate_math_dataset(seed=42),
        'Financial-Analysis':  generate_financial_dataset(seed=43),
        'Causal-Reasoning':    generate_causal_dataset(seed=44),
    }
    for n, ds in datasets.items():
        print(f'  {n}: train={len(ds["train"])} val={len(ds["val"])}', flush=True)

    methods = [
        ('sft',             train_standard,        {}),
        ('data_filtering',  train_data_filtering,  {}),
        ('group_dro',       train_group_dro,       {}),
        ('reweight_only',   train_our_method,      {'use_reweighting': True,  'use_gradient_surgery': False}),
        ('gs_only',         train_our_method,      {'use_reweighting': False, 'use_gradient_surgery': True}),
        ('sart_full',       train_our_method,      {'use_reweighting': True,  'use_gradient_surgery': True}),
    ]

    results = {}
    total_start = time.time()
    for ds_name, ds in datasets.items():
        print(f'\n=== Dataset: {ds_name} ===', flush=True)
        for m_name, train_fn, kwargs in methods:
            key = f'{ds_name}/{m_name}'
            r = run_one(m_name, train_fn, ds, **kwargs)
            results[key] = r
            # Persist incrementally so partial progress survives a crash.
            with open(OUT_JSON, 'w') as f:
                json.dump(results, f, indent=2)

    total_time = time.time() - total_start
    print(f'\n{"=" * 70}\nTotal: {total_time / 60:.1f} min\n{"=" * 70}',
          flush=True)

    # Compact summary
    print('\nSUMMARY (acc / rob):', flush=True)
    print(f'{"Dataset":<22} {"Method":<16} {"Acc":>6} {"Rob":>6} '
          f'{"Reason":>7} {"F1":>5} {"|Align|":>8}', flush=True)
    for ds_name in datasets:
        for m_name, _, _ in methods:
            r = results[f'{ds_name}/{m_name}']
            align_val = r.get('gradient_alignment')
            align_str = (f'{abs(align_val):.3f}'
                         if align_val is not None else '  -  ')
            f1_val = r.get('shortcut_f1')
            f1_str = f'{f1_val:.2f}' if f1_val is not None else '  -  '
            print(f'{ds_name:<22} {m_name:<16} '
                  f'{r["accuracy_clean"]:.3f} {r["robustness"]:.3f} '
                  f'{r.get("reasoning_consistency", 0):.3f} '
                  f'{f1_str:>5} {align_str:>8}',
                  flush=True)


if __name__ == '__main__':
    main()
