"""D3+D2 hybrid diagnostic: per-shortcut-group surgery with oracle selection.

Tests whether SART's failure (Path A SO K=8 stuck at 0.16-0.40 robustness vs
oracle-drop's 1.000 ceiling) is fixed by:
  1. Using oracle `is_shortcut` to select samples eligible for surgery (D2)
  2. Applying conflict-only PCGrad surgery to the SHORTCUT-GROUP gradient
     specifically, not the full batch gradient (D3 group-level approximation)

Implementation:
  Phase 1: standard 8-epoch warmup
  Phase 2: snapshot g_V on validation set
  Phase 3 (42 epochs):
    Per batch:
      - Single forward pass on full batch
      - Decompose loss into shortcut-group contribution + clean-group contribution
      - Two backward passes (one per group) -> g_short, g_clean
        (g_short + g_clean = standard batch grad)
      - Compute cos_sim(g_short, g_V); if < 0, project g_short out of conflict
      - Combined batch grad = (modified) g_short + g_clean
      - optimizer.step()
    Refresh g_V every 5 epochs.

Output:
  results/diagnose_per_sample_oracle.json
  results/diagnose_per_sample_oracle.log

If acc/rob lifts close to oracle-drop's 1.000, F2 (no S* selection) is the
real failure mode. Fix path: per-sample selection + surgery.
If still stuck, detection is fundamentally broken at per-sample level.
"""
from __future__ import annotations

import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset, get_dataloader)
from src.model import create_model, count_parameters
from src.trainer import train_standard, compute_validation_gradient
from src.utils.gradient_ops import get_grad_vector, set_grad_vector
from src.evaluate import run_full_evaluation


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_oracle_group_surgery(model, dataset, device='cuda', verbose=True,
                                K=8, epochs=50, gamma=1.0):
    bs = C.batch_size
    lr = C.lr
    wd = C.weight_decay

    main_epochs = epochs - K

    if verbose:
        print(f"  Phase 1: Warmup ({K} epochs)", flush=True)
    train_standard(model, dataset, epochs=K, verbose=False)

    val_loader = get_dataloader(dataset['val'], shuffle=False, batch_size=bs)
    g_V = compute_validation_gradient(model, val_loader, device)

    if verbose:
        print(f"  Phase 3: Oracle group-surgery ({main_epochs} epochs)",
              flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = get_dataloader(dataset['train'], shuffle=True, batch_size=bs)
    total_steps = max(1, main_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5)

    gate_fires_total = 0
    surgery_eligible_total = 0
    cs_sum, cs_min, cs_max = 0.0, float('inf'), float('-inf')

    model.train()
    for epoch in range(main_epochs):
        if epoch > 0 and epoch % 5 == 0:
            g_V = compute_validation_gradient(model, val_loader, device)
            model.train()

        epoch_loss, epoch_n = 0.0, 0
        epoch_fires, epoch_eligible = 0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            target_ids = batch['target_ids'].to(device)
            loss_mask = batch['loss_mask'].to(device)
            is_shortcut = batch['is_shortcut'].to(device).bool()

            short_mask = is_shortcut
            clean_mask = ~is_shortcut
            n_short = short_mask.sum().item()
            n_clean = clean_mask.sum().item()
            B = input_ids.shape[0]

            optimizer.zero_grad()
            logits = model(input_ids)
            _, T, V = logits.shape
            loss_per_token = F.cross_entropy(
                logits.reshape(-1, V), target_ids.reshape(-1),
                reduction='none'
            ).reshape(B, T)
            masked_loss = loss_per_token * loss_mask
            total_mask_sum = loss_mask.sum().clamp(min=1)
            per_sample_loss_sum = masked_loss.sum(dim=1)

            loss_short = (per_sample_loss_sum *
                          short_mask.float()).sum() / total_mask_sum
            loss_clean = (per_sample_loss_sum *
                          clean_mask.float()).sum() / total_mask_sum
            batch_loss_val = (loss_short + loss_clean).item()

            # Backward 1: shortcut group
            if n_short > 0:
                loss_short.backward(retain_graph=True)
                g_short = get_grad_vector(model).detach().clone()
                optimizer.zero_grad()
            else:
                g_short = torch.zeros_like(g_V)

            # Backward 2: clean group
            if n_clean > 0:
                loss_clean.backward()
                g_clean = get_grad_vector(model).detach().clone()
            else:
                g_clean = torch.zeros_like(g_V)

            # Surgery on g_short (if there are shortcut samples and g_V is non-zero)
            if n_short > 0:
                norm_short = g_short.norm()
                norm_gv = g_V.norm()
                if norm_short > 1e-10 and norm_gv > 1e-10:
                    cos_sim = (g_short @ g_V) / (norm_short * norm_gv)
                    cs_val = cos_sim.item()
                    surgery_eligible_total += 1
                    epoch_eligible += 1
                    cs_sum += cs_val
                    if cs_val < cs_min:
                        cs_min = cs_val
                    if cs_val > cs_max:
                        cs_max = cs_val
                    if cs_val < 0:
                        gate_fires_total += 1
                        epoch_fires += 1
                        gv_norm_sq = (g_V @ g_V).clamp(min=1e-10)
                        dot = g_short @ g_V
                        g_short = g_short - gamma * (dot / gv_norm_sq) * g_V
                        # Preserve original norm so step magnitude isn't scaled
                        g_short = g_short * (norm_short /
                                             g_short.norm().clamp(min=1e-10))

            g_combined = g_short + g_clean
            set_grad_vector(model, g_combined)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += batch_loss_val
            epoch_n += 1

        if verbose and (epoch + 1) % max(1, main_epochs // 5) == 0:
            fire_pct = 100.0 * epoch_fires / max(epoch_eligible, 1)
            print(f"    Epoch {epoch+1}/{main_epochs}, "
                  f"Loss: {epoch_loss/max(epoch_n,1):.4f}, "
                  f"shortcut-gate fires {epoch_fires}/{epoch_eligible} "
                  f"({fire_pct:.1f}%)", flush=True)

    if verbose and surgery_eligible_total > 0:
        avg_cs = cs_sum / surgery_eligible_total
        total_pct = 100.0 * gate_fires_total / surgery_eligible_total
        print(f"  [Oracle-group gate diagnostics] total fires "
              f"{gate_fires_total}/{surgery_eligible_total} ({total_pct:.1f}%); "
              f"shortcut-group cos_sim range [{cs_min:.3f}, {cs_max:.3f}], "
              f"mean {avg_cs:.3f}", flush=True)

    stats = {
        'gate_fires': gate_fires_total,
        'surgery_eligible': surgery_eligible_total,
        'fire_rate_pct': 100.0 * gate_fires_total / max(surgery_eligible_total, 1),
        'cos_sim_mean': cs_sum / max(surgery_eligible_total, 1),
        'cos_sim_min': cs_min if cs_min != float('inf') else 0.0,
        'cos_sim_max': cs_max if cs_max != float('-inf') else 0.0,
    }
    return model, stats


def main():
    print(f"PROFILE = {PROFILE}, device = {C.device}", flush=True)
    print(f"D3+D2 hybrid: per-shortcut-group surgery with oracle is_shortcut",
          flush=True)

    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'results')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'diagnose_per_sample_oracle.json')

    results = {}
    if os.path.exists(json_path):
        try:
            with open(json_path) as f:
                results = json.load(f)
            n_done = sum(len(v) for v in results.values())
            print(f"Resume: {n_done} cells already done", flush=True)
        except Exception:
            results = {}

    def save():
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

    tmp = create_model('cpu')
    print(f"\nSmallGPT params: {count_parameters(tmp):,}", flush=True)
    del tmp

    t_start = time.time()

    # Causal first (the most extreme failure case)
    order = ['Causal-Reasoning', 'Math-Arithmetic', 'Financial-Analysis']
    for ds_name in order:
        results.setdefault(ds_name, {})
        if 'OracleGroupSurgery' in results[ds_name]:
            print(f"\n[{ds_name}] OracleGroupSurgery already done, skipping",
                  flush=True)
            continue

        print(f"\n=========================================================",
              flush=True)
        print(f"=== Dataset: {ds_name} ===", flush=True)
        print(f"=========================================================",
              flush=True)

        ds = datasets[ds_name]
        set_seed(42)
        model = create_model()
        t0 = time.time()
        model, stats = train_oracle_group_surgery(
            model, ds, K=8, epochs=50, gamma=1.0, verbose=True
        )
        t_train = time.time() - t0
        r = run_full_evaluation(model, ds, compute_f1=False)

        results[ds_name]['OracleGroupSurgery'] = {
            'accuracy_clean': float(r['accuracy_clean']),
            'robustness': float(r['robustness']),
            'train_sec': t_train,
            **stats,
        }
        print(f"\n  Result: acc={r['accuracy_clean']:.3f}, "
              f"rob={r['robustness']:.3f}, time={t_train:.1f}s, "
              f"fire_rate={stats['fire_rate_pct']:.1f}%, "
              f"cos_sim mean={stats['cos_sim_mean']:.3f}", flush=True)
        save()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start
    print(f"\n=== Done in {t_elapsed:.1f}s ({t_elapsed/60:.1f} min) ===",
          flush=True)

    print("\n=== Summary (compare to D1 SO_K8 and D4 OracleDrop) ===",
          flush=True)
    print(f"{'Dataset':<22s}  {'Acc':>6s}  {'Rob':>6s}  "
          f"{'Gate Fire %':>12s}  {'cos_sim Mean':>13s}", flush=True)
    for ds_name in order:
        if ds_name in results and 'OracleGroupSurgery' in results[ds_name]:
            m = results[ds_name]['OracleGroupSurgery']
            print(f"{ds_name:<22s}  {m['accuracy_clean']:>6.3f}  "
                  f"{m['robustness']:>6.3f}  {m['fire_rate_pct']:>12.1f}  "
                  f"{m['cos_sim_mean']:>13.3f}", flush=True)


if __name__ == '__main__':
    main()
