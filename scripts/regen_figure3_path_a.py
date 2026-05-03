"""Regenerate Figure 3 as a 1x3 evidence-chain layout for the docs.

Three panels read left-to-right as one argument: detection ranks correctly
(a), the conflict gate filters within that ranking (b), and the gated
direction-correction is the load-bearing primitive (c).

  (a) S(s) vs shortcut rate — ShortcutScore as a rank signal for sample
      selection (the conflict-gated projection's gating set).
  (b) A(s) split into TP / FP / TN / FN cells against tau_S* — diagnoses
      whether the surgery's gating set has the right geometry.
  (c) Ablation bars (Acc / Rob) — conflict-only projection is the single
      load-bearing primitive of SART.

Reads: results/collected_data_synthetic.pkl
Writes: docs/figures/figure3.png  (3-panel)
        docs/figures/figure3_diagnostics.json
"""
from __future__ import annotations

import json
import os
import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
DOCS_FIG_DIR = os.path.join(REPO_ROOT, "docs", "figures")


def _aggregate(collected_data):
    if collected_data and "scores" not in collected_data:
        agg = {"scores": [], "is_shortcut": [], "alignments": [], "concentrations": []}
        for d in collected_data.values():
            for k in agg:
                agg[k].extend(d.get(k, []))
        return agg
    return collected_data


def main():
    pkl = os.path.join(RESULTS_DIR, "collected_data_synthetic.pkl")
    if not os.path.exists(pkl):
        sys.exit(f"missing {pkl} — run scripts/regen_collected_data.py first")
    with open(pkl, "rb") as f:
        snap = pickle.load(f)
    collected = _aggregate(snap.get("collected_data_all", snap))

    scores = np.array(collected["scores"], dtype=float)
    is_sc = np.array(collected["is_shortcut"], dtype=float)
    alignments = np.array(collected["alignments"], dtype=float)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    summary = {}

    # Panel (a) — S(s) vs shortcut rate
    r_sample, _ = pearsonr(scores, is_sc)
    axes[0].scatter(
        scores, is_sc, alpha=0.05, s=2, color="gray",
        zorder=1, label=f"per-sample n={len(scores)}",
    )
    n_bins = 20
    bins_a = np.linspace(scores.min(), scores.max(), n_bins + 1)
    centers, rates = [], []
    for i in range(n_bins):
        mask = (scores >= bins_a[i]) & (scores < bins_a[i + 1])
        if mask.sum() > 0:
            centers.append((bins_a[i] + bins_a[i + 1]) / 2)
            rates.append(is_sc[mask].mean())
    axes[0].scatter(centers, rates, alpha=0.95, s=40, color="steelblue",
                    zorder=3, label="20-bin mean")
    r_bin = None
    if len(centers) > 1:
        z = np.polyfit(centers, rates, 1)
        x_line = np.linspace(min(centers), max(centers), 100)
        axes[0].plot(x_line, np.poly1d(z)(x_line), "r--", alpha=0.7,
                     zorder=2, label="linear fit")
        r_bin = float(np.corrcoef(centers, rates)[0, 1])
    axes[0].set_xlabel("ShortcutScore S(s)")
    axes[0].set_ylabel("is_shortcut / Shortcut Rate")
    axes[0].set_ylim(-0.05, 1.05)
    title = "(a) S(s) vs Shortcut Rate\n"
    title += f"r_bin={r_bin:.2f}, " if r_bin is not None else ""
    title += f"r_sample={r_sample:.2f}"
    axes[0].set_title(title)
    axes[0].legend(loc="center right", fontsize=8)
    summary["r_bin"] = r_bin
    summary["r_sample"] = float(r_sample)

    # Panel (b) — A(s) by S(s) > tau_S*  (4-cell)
    best_f1, tau_S_star = 0.0, float(scores.mean())
    for t in np.linspace(scores.min(), scores.max(), 50):
        preds = (scores > t).astype(float)
        tp_n = ((preds == 1) & (is_sc == 1)).sum()
        fp_n = ((preds == 1) & (is_sc == 0)).sum()
        fn_n = ((preds == 0) & (is_sc == 1)).sum()
        prec = tp_n / max(tp_n + fp_n, 1)
        rec = tp_n / max(tp_n + fn_n, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-10)
        if f1 > best_f1:
            best_f1, tau_S_star = float(f1), float(t)

    flagged = scores > tau_S_star
    is_short_b = is_sc == 1
    tp_mask = flagged & is_short_b
    fp_mask = flagged & ~is_short_b
    tn_mask = ~flagged & ~is_short_b
    fn_mask = ~flagged & is_short_b

    bins_b = np.linspace(alignments.min(), alignments.max(), 30)
    if tp_mask.sum() > 0:
        axes[1].hist(alignments[tp_mask], bins=bins_b, alpha=0.55,
                     label=f"TP (n={int(tp_mask.sum())})",
                     color="red", density=True)
    if fp_mask.sum() > 0:
        axes[1].hist(alignments[fp_mask], bins=bins_b, alpha=0.7,
                     label=f"FP (n={int(fp_mask.sum())})",
                     color="orange", density=True)
    if tn_mask.sum() > 0:
        axes[1].hist(alignments[tn_mask], bins=bins_b, alpha=0.4,
                     label=f"TN (n={int(tn_mask.sum())})",
                     color="green", density=True)
    if fn_mask.sum() > 0:
        axes[1].hist(alignments[fn_mask], bins=bins_b, alpha=0.5,
                     label=f"FN (n={int(fn_mask.sum())})",
                     color="purple", density=True,
                     histtype="step", linewidth=1.6)

    fp_mean_A = float(alignments[fp_mask].mean()) if fp_mask.sum() else float("nan")
    fp_gate_rate = float((alignments[fp_mask] < 0).mean() * 100) if fp_mask.sum() else float("nan")

    axes[1].set_xlabel("Gradient Alignment A(s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(
        f"(b) A(s) by S(s)>tau* (F1={best_f1:.2f}, tau*={tau_S_star:.2f})\n"
        f"FP mean A={fp_mean_A:+.2f}, surgery-gate fire on FPs={fp_gate_rate:.0f}%"
    )
    axes[1].legend(fontsize=8, loc="upper left")
    summary["tau_S_star"] = tau_S_star
    summary["best_F1"] = best_f1
    summary["FP_mean_A"] = fp_mean_A
    summary["FP_gate_fire_pct"] = fp_gate_rate

    # Panel (c) — Ablation bars: Acc / Rob across SART variants
    abl_labels = ["SFT", "SART-rwt", "w/o\nproj.", "w/o\nsuppr.", "SART\n(Full)"]
    abl_acc = [58.8, 71.4, 64.2, 79.6, 80.7]
    abl_rob = [0.6, 40.8, 18.3, 50.9, 54.8]
    x = np.arange(len(abl_labels))
    w = 0.38
    bars_acc = axes[2].bar(x - w / 2, abl_acc, w, label="Accuracy",
                            color="steelblue", alpha=0.85)
    bars_rob = axes[2].bar(x + w / 2, abl_rob, w, label="Robustness",
                            color="firebrick", alpha=0.85)
    for b, v in zip(bars_acc, abl_acc):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=8)
    for b, v in zip(bars_rob, abl_rob):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
                     ha="center", va="bottom", fontsize=8)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(abl_labels, fontsize=9)
    axes[2].set_ylabel("Score (%)")
    axes[2].set_ylim(0, 100)
    axes[2].set_title("(c) Ablation: Accuracy and Robustness\n"
                      "removing conflict-only projection collapses Rob 54.8 -> 18.3")
    axes[2].legend(loc="upper left", fontsize=8)
    axes[2].grid(axis="y", alpha=0.3)
    summary["ablation_acc"] = abl_acc
    summary["ablation_rob"] = abl_rob

    plt.tight_layout()
    os.makedirs(DOCS_FIG_DIR, exist_ok=True)
    out_png = os.path.join(DOCS_FIG_DIR, "figure3.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    with open(os.path.join(DOCS_FIG_DIR, "figure3_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {out_png}")
    print(f"  diagnostics: {summary}")


if __name__ == "__main__":
    main()
