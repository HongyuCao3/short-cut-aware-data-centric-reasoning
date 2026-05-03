"""P1: Self-reference PCGrad — replace g_V with a shortcut-anchor estimated
from high-S(s) samples.

Hypothesis: surgery's failure (D1: cos_sim ≈ 0; D3+D2: shortcut-group cos_sim
also ≈ 0) is partly because g_V isn't the right reference vector —
validation gradient mixes "improve clean" with "improve shortcut-shared
features". A self-reference computed from high-ShortcutScore samples should
point along the shortcut subspace by construction; per-sample / per-group
shortcut gradients should then have meaningfully POSITIVE cos_sim with the
anchor (instead of ≈ 0 with g_V), and surgery can project away the aligning
component.

Method:
  Phase 1: standard 8-epoch warmup
  Phase 2: compute ShortcutScores S(s) for all training samples
  Compute g_anchor: average per-sample gradient over top-K samples by S(s).
  Phase 3 (42 epochs): per-shortcut-group surgery using ORACLE is_shortcut
    as the selection (matches D3+D2 setup for direct comparison) but
    PROJECTING AGAINST g_anchor INSTEAD OF g_V. If cos(g_short, g_anchor) > 0
    (group aligns with shortcut anchor), remove the aligning component.

Also report the per-shortcut-group cos_sim mean / range using g_anchor —
direct comparison to D3+D2's (g_V) numbers shows whether self-reference
recovers a meaningful gate signal.

Output:
  results/diagnose_p1_self_reference.json (per-dataset acc/rob + cos_sim stats)
  results/diagnose_p1_self_reference.log
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
import torch.nn.functional as F

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset, get_dataloader)
from src.model import create_model, count_parameters
from src.trainer import (train_standard, _compute_sample_scores,
                         compute_validation_gradient)
from src.utils.gradient_ops import get_grad_vector, set_grad_vector
from src.evaluate import run_full_evaluation


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_anchor_from_topk(model, dataset, top_idx, device='cuda',
                              batch_size=64):
    """Average gradient over a specified subset of training samples.

    Used to compute a 'shortcut anchor' from high-S(s) samples.
    """
    samples_subset = [dataset['train'].samples[i] for i in top_idx]
    # Construct a temporary loader-like iteration over these samples
    train_orig = dataset['train']
    subset_obj = copy.copy(train_orig)
    subset_obj.samples = samples_subset
    loader = get_dataloader(subset_obj, shuffle=False, batch_size=batch_size)

    n_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    g_anchor = torch.zeros(n_param, device=device)
    n_seen = 0

    model.eval()
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        target_ids = batch['target_ids'].to(device)
        loss_mask = batch['loss_mask'].to(device)
        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()
        logits = model(input_ids)
        B, T, V = logits.shape
        loss_per_token = F.cross_entropy(
            logits.reshape(-1, V), target_ids.reshape(-1),
            reduction='none').reshape(B, T)
        masked_loss = loss_per_token * loss_mask
        loss = masked_loss.sum() / loss_mask.sum().clamp(min=1)
        loss.backward()
        g_batch = get_grad_vector(model)
        g_anchor += g_batch * B
        n_seen += B

    g_anchor = g_anchor / max(n_seen, 1)
    model.train()
    return g_anchor


def train_self_reference_group_surgery(model, dataset, g_anchor, K=8, epochs=50,
                                        gamma=1.0, device='cuda', verbose=True):
    bs = C.batch_size
    lr = C.lr
    wd = C.weight_decay

    main_epochs = epochs - K

    # Note: we ALREADY did warmup before computing g_anchor; this function is
    # called for the 'main' phase only. Caller is responsible for warmup.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    train_loader = get_dataloader(dataset['train'], shuffle=True, batch_size=bs)
    total_steps = max(1, main_epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_steps, eta_min=1e-5)

    gate_fires_total = 0
    surgery_eligible_total = 0
    cs_sum, cs_min, cs_max = 0.0, float('inf'), float('-inf')

    if verbose:
        print(f"  Phase 3: Self-reference group-surgery ({main_epochs} "
              f"epochs); anchor norm = {g_anchor.norm().item():.3e}",
              flush=True)

    model.train()
    for epoch in range(main_epochs):
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
                reduction='none').reshape(B, T)
            masked_loss = loss_per_token * loss_mask
            total_mask_sum = loss_mask.sum().clamp(min=1)
            per_sample_loss_sum = masked_loss.sum(dim=1)

            loss_short = (per_sample_loss_sum *
                          short_mask.float()).sum() / total_mask_sum
            loss_clean = (per_sample_loss_sum *
                          clean_mask.float()).sum() / total_mask_sum
            batch_loss_val = (loss_short + loss_clean).item()

            if n_short > 0:
                loss_short.backward(retain_graph=True)
                g_short = get_grad_vector(model).detach().clone()
                optimizer.zero_grad()
            else:
                g_short = torch.zeros_like(g_anchor)

            if n_clean > 0:
                loss_clean.backward()
                g_clean = get_grad_vector(model).detach().clone()
            else:
                g_clean = torch.zeros_like(g_anchor)

            # Self-reference surgery on g_short:
            # If cos(g_short, g_anchor) > 0 (group aligns WITH shortcut anchor),
            # PROJECT AWAY the shortcut-aligned component.
            if n_short > 0:
                norm_short = g_short.norm()
                norm_anchor = g_anchor.norm()
                if norm_short > 1e-10 and norm_anchor > 1e-10:
                    cos_sim = (g_short @ g_anchor) / (norm_short * norm_anchor)
                    cs_val = cos_sim.item()
                    surgery_eligible_total += 1
                    epoch_eligible += 1
                    cs_sum += cs_val
                    if cs_val < cs_min:
                        cs_min = cs_val
                    if cs_val > cs_max:
                        cs_max = cs_val
                    if cs_val > 0:  # ! gate flips: project when ALIGNED with anchor
                        gate_fires_total += 1
                        epoch_fires += 1
                        ga_norm_sq = (g_anchor @ g_anchor).clamp(min=1e-10)
                        dot = g_short @ g_anchor
                        g_short = g_short - gamma * (dot / ga_norm_sq) * g_anchor
                        # Preserve scale
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
                  f"shortcut-anchor align (cos>0) gate fires "
                  f"{epoch_fires}/{epoch_eligible} ({fire_pct:.1f}%)",
                  flush=True)

    if verbose and surgery_eligible_total > 0:
        avg_cs = cs_sum / surgery_eligible_total
        total_pct = 100.0 * gate_fires_total / surgery_eligible_total
        print(f"  [Self-reference gate diagnostics] cos>0 fires "
              f"{gate_fires_total}/{surgery_eligible_total} ({total_pct:.1f}%); "
              f"shortcut-group cos_sim with g_anchor: range "
              f"[{cs_min:.3f}, {cs_max:.3f}], mean {avg_cs:.3f}", flush=True)

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
    print(f"P1: Self-reference PCGrad (g_anchor = mean grad over top-S(s) "
          f"samples)", flush=True)

    TOP_FRACTION_FOR_ANCHOR = 0.10  # top 10% by S(s) for anchor
    K_WARMUP = 8
    EPOCHS_TOTAL = 50

    datasets = {
        'Math-Arithmetic': generate_math_dataset(seed=42),
        'Financial-Analysis': generate_financial_dataset(seed=43),
        'Causal-Reasoning': generate_causal_dataset(seed=44),
    }

    out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                           'results')
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, 'diagnose_p1_self_reference.json')

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
        if ds_name in results and 'P1' in results[ds_name]:
            print(f"\n[{ds_name}] P1 already done, skipping", flush=True)
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

        # Phase 2: compute S(s)
        print(f"  Phase 2: Computing ShortcutScores...", flush=True)
        sample_scores, _, g_V = _compute_sample_scores(model, ds)
        scores = np.array(sample_scores)

        # Compute anchor over top-fraction by S(s)
        n_total = len(scores)
        n_anchor = max(1, int(n_total * TOP_FRACTION_FOR_ANCHOR))
        top_idx = np.argsort(scores)[-n_anchor:]

        is_shortcut = np.array(
            [bool(s['is_shortcut']) for s in ds['train'].samples])
        anchor_clean_pct = (~is_shortcut[top_idx]).sum() / max(n_anchor, 1)
        anchor_short_pct = is_shortcut[top_idx].sum() / max(n_anchor, 1)
        print(f"  Anchor set: top {TOP_FRACTION_FOR_ANCHOR*100:.0f}% by S(s) "
              f"= {n_anchor} samples", flush=True)
        print(f"    Anchor composition: shortcut {100*anchor_short_pct:.1f}%, "
              f"clean {100*anchor_clean_pct:.1f}%", flush=True)

        print(f"  Computing g_anchor (mean grad over anchor set)...",
              flush=True)
        g_anchor = compute_anchor_from_topk(model, ds, top_idx)
        print(f"    g_anchor norm = {g_anchor.norm().item():.3e}, "
              f"g_V norm = {g_V.norm().item():.3e}", flush=True)
        cos_anchor_v = ((g_anchor @ g_V) / (g_anchor.norm() *
                        g_V.norm()).clamp(min=1e-10)).item()
        print(f"    cos(g_anchor, g_V) = {cos_anchor_v:+.3f} "
              f"(if ≈ +1, anchor ≈ g_V; if ≠ +1, self-reference adds "
              f"new info)", flush=True)

        # Phase 3: self-reference group surgery
        t0 = time.time()
        model, stats = train_self_reference_group_surgery(
            model, ds, g_anchor, K=K_WARMUP, epochs=EPOCHS_TOTAL,
            gamma=1.0, verbose=True
        )
        t_train = time.time() - t0
        r = run_full_evaluation(model, ds, compute_f1=False)

        results[ds_name]['P1'] = {
            'accuracy_clean': float(r['accuracy_clean']),
            'robustness': float(r['robustness']),
            'train_sec': t_train,
            'anchor_top_fraction': TOP_FRACTION_FOR_ANCHOR,
            'anchor_n': int(n_anchor),
            'anchor_short_pct': float(100 * anchor_short_pct),
            'anchor_clean_pct': float(100 * anchor_clean_pct),
            'cos_anchor_g_V': float(cos_anchor_v),
            **stats,
        }
        print(f"\n  Result: acc={r['accuracy_clean']:.3f}, "
              f"rob={r['robustness']:.3f}, "
              f"shortcut-group cos_sim with g_anchor: "
              f"{stats['cos_sim_mean']:+.3f}", flush=True)
        save()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    t_elapsed = time.time() - t_start
    print(f"\n=== Done in {t_elapsed:.1f}s ({t_elapsed/60:.1f} min) ===",
          flush=True)

    print("\n=== Summary (compare to D3+D2 with g_V) ===", flush=True)
    print(f"{'Dataset':<22s}  {'Acc':>6s}  {'Rob':>6s}  "
          f"{'cos(g_short,g_anc)':>18s}  {'cos(g_anc,g_V)':>15s}  "
          f"{'Anchor Short %':>15s}", flush=True)
    for ds_name in order:
        if ds_name in results and 'P1' in results[ds_name]:
            m = results[ds_name]['P1']
            print(f"{ds_name:<22s}  {m['accuracy_clean']:>6.3f}  "
                  f"{m['robustness']:>6.3f}  {m['cos_sim_mean']:>+17.3f}  "
                  f"{m['cos_anchor_g_V']:>+14.3f}  "
                  f"{m['anchor_short_pct']:>14.1f}%", flush=True)


if __name__ == '__main__':
    main()
