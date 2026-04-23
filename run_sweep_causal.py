#!/usr/bin/env python3
"""Mini-sweep: (gamma, rho) on Causal with PCGrad-gated surgery.

Causal dropped from 70.0/41.3 (pre-PCGrad) to 55.1/28.3 (PCGrad-gated,
default gamma=1.0, rho=0.5). Those defaults were tuned against the old
buggy surgery rule that fired on any cos_sim < tau_A, so they may over-
project under the conflict-only gate. This script asks the minimal
empirical question: does relaxing gamma (less aggressive projection) or
rho (less aggressive answer suppression) recover the lost 15 pp on
Causal? Four configs, ~5 min each, no F1 eval.

Usage:
  CUDA_VISIBLE_DEVICES=<free-gpu> EXPERIMENT_SCALE=server \
      python3 -u run_sweep_causal.py
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
from src.data import generate_causal_dataset, get_dataloader
from src.model import create_model
from src.trainer import train_our_method
from src.evaluate import run_full_evaluation, evaluate_gradient_alignment


OUT_DIR = os.environ.get(
    'SART_DATA_ROOT',
    '/data/hongyuca/short-cut-aware-data-centric-reasoning',
) + '/logs/pcgrad-rerun'
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = os.path.join(OUT_DIR, 'sweep_causal.json')


def set_seed(seed=C.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_config(ds, gamma, rho):
    tag = f'γ={gamma} ρ={rho}'
    print(f'\n--- {tag} ---', flush=True)
    t0 = time.time()
    set_seed()
    m = create_model()
    m = train_our_method(
        m, ds,
        use_reweighting=True, use_gradient_surgery=True,
        cfg={'score_max_samples': 10000, 'sketch_k': 128,
             'gamma': gamma, 'rho': rho},
    )
    if isinstance(m, tuple):
        m = m[0]
    r = run_full_evaluation(m, ds, compute_f1=False, compute_alignment=False)
    val_loader = get_dataloader(ds['val'], batch_size=C.batch_size, shuffle=False)
    r['gradient_alignment'] = evaluate_gradient_alignment(m, ds['train'], val_loader)
    r['train_time_sec'] = time.time() - t0
    r['gamma'] = gamma
    r['rho'] = rho
    print(f'  {tag}  acc={r["accuracy_clean"]:.3f} '
          f'rob={r["robustness"]:.3f} '
          f'align={r["gradient_alignment"]:.3f} '
          f'time={r["train_time_sec"]:.1f}s', flush=True)
    del m
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r


def main():
    print('=' * 70, flush=True)
    print(f'Causal (gamma, rho) mini-sweep (profile={PROFILE})', flush=True)
    print(f'  Baseline at default gamma=1.0, rho=0.5:  55.1 / 28.3', flush=True)
    print(f'  Pre-PCGrad local SART on Causal:         70.0 / 41.3', flush=True)
    print(f'  Output: {OUT_JSON}', flush=True)
    print('=' * 70, flush=True)

    print('\n--- Generating Causal-Reasoning ---', flush=True)
    ds = generate_causal_dataset(seed=44)
    print(f'  train={len(ds["train"])} val={len(ds["val"])}', flush=True)

    configs = [
        (0.3, 0.5),   # less aggressive projection
        (0.5, 0.5),   # moderate projection (paper's non-peak choice)
        (1.0, 0.3),   # keep projection, relax answer suppression
        (1.0, 0.0),   # keep projection, disable answer suppression
    ]

    results = []
    for gamma, rho in configs:
        r = run_config(ds, gamma, rho)
        results.append(r)
        with open(OUT_JSON, 'w') as f:
            json.dump(results, f, indent=2)

    print('\n' + '=' * 70, flush=True)
    print('SWEEP SUMMARY (Causal-Reasoning, PCGrad-gated surgery)', flush=True)
    print(f'{"γ":>5} {"ρ":>5} {"Acc":>6} {"Rob":>6} {"Align":>7}', flush=True)
    for r in results:
        print(f'{r["gamma"]:>5.1f} {r["rho"]:>5.1f} '
              f'{r["accuracy_clean"]:.3f} {r["robustness"]:.3f} '
              f'{r["gradient_alignment"]:+.3f}', flush=True)
    print('Ref baselines:', flush=True)
    print(f'  γ=1.0 ρ=0.5 (current default, PCGrad-gated):  55.1 / 28.3', flush=True)
    print(f'  Pre-PCGrad surgery (old buggy rule), local:    70.0 / 41.3', flush=True)


if __name__ == '__main__':
    main()
