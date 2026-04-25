"""Visualization: generate tables and figures for the paper."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.config import Config as C

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def format_pct(val):
    if val is None:
        return '-'
    return f'{val*100:.1f}%'


def format_f1(val):
    if val is None:
        return '-'
    return f'{val:.2f}'


def generate_table1(all_results, datasets):
    """Table 1: Overall Method Performance and Baseline Superiority.

    Rows: methods. Columns: Accuracy, Robustness, Reasoning Consistency, Shortcut F1.
    Values are averaged across datasets.
    """
    methods = ['standard_ft', 'self_consistency', 'data_filtering',
               'jtt', 'focal_loss', 'group_dro',
               'irm', 'vrex', 'fishr', 'lff', 'influence_filtering', 'meta_reweight',
               'full_method']
    method_names = {
        'standard_ft': 'Standard Fine-Tuning',
        'self_consistency': 'Self-Consistency Decoding',
        'data_filtering': 'Data Filtering',
        'jtt': 'JTT (Just Train Twice)',
        'focal_loss': 'Focal Loss',
        'group_dro': 'Group DRO',
        'irm': 'IRM',
        'vrex': 'V-REx',
        'fishr': 'Fishr',
        'lff': 'LfF',
        'influence_filtering': 'Influence Filtering',
        'meta_reweight': 'Meta-Reweighting',
        'full_method': 'Our Method (Full)',
    }

    lines = []
    lines.append('=' * 100)
    lines.append('Table 1: Overall Method Performance and Baseline Superiority')
    lines.append('=' * 100)
    lines.append(f'{"Method":<30} {"Accuracy":>10} {"Robustness":>12} {"Reasoning":>12} {"SC Det F1":>10}')
    lines.append('-' * 90)

    for method in methods:
        accs, robs, reas, f1s = [], [], [], []
        for ds_name in datasets:
            key = (ds_name, method)
            if key in all_results:
                r = all_results[key]
                accs.append(r['accuracy_clean'])
                robs.append(r['robustness'])
                reas.append(r['reasoning_consistency'])
                if r['shortcut_f1'] is not None:
                    f1s.append(r['shortcut_f1'])

        avg_acc = np.mean(accs) if accs else 0
        avg_rob = np.mean(robs) if robs else 0
        avg_rea = np.mean(reas) if reas else 0
        avg_f1 = np.mean(f1s) if f1s else None

        lines.append(f'{method_names[method]:<30} {format_pct(avg_acc):>10} '
                      f'{format_pct(avg_rob):>12} {format_pct(avg_rea):>12} '
                      f'{format_f1(avg_f1):>10}')

    lines.append('=' * 90)

    # Per-dataset breakdown
    lines.append('\nPer-Dataset Breakdown:')
    for ds_name in datasets:
        lines.append(f'\n  Dataset: {ds_name}')
        lines.append(f'  {"Method":<30} {"Acc Clean":>10} {"Acc Perturb":>12} {"Reasoning":>12}')
        lines.append('  ' + '-' * 70)
        for method in methods:
            key = (ds_name, method)
            if key in all_results:
                r = all_results[key]
                lines.append(f'  {method_names[method]:<30} '
                              f'{format_pct(r["accuracy_clean"]):>10} '
                              f'{format_pct(r["robustness"]):>12} '
                              f'{format_pct(r["reasoning_consistency"]):>12}')

    table_str = '\n'.join(lines)
    print(table_str)
    with open(os.path.join(RESULTS_DIR, 'table1.txt'), 'w') as f:
        f.write(table_str)
    return table_str


def generate_table2(all_results, datasets):
    """Table 2: Contribution of Shortcut-aware Reweighting (Ablation)."""
    methods = ['full_method', 'gs_only', 'reweight_only']
    method_names = {
        'full_method': 'Full Method (Both)',
        'gs_only': 'Gradient Surgery Only',
        'reweight_only': 'Reweighting Only',
    }

    lines = []
    lines.append('=' * 70)
    lines.append('Table 2: Contribution of Shortcut-aware Reweighting')
    lines.append('=' * 70)
    lines.append(f'{"Configuration":<30} {"Accuracy":>12} {"SC Det F1":>12}')
    lines.append('-' * 70)

    for method in methods:
        accs, f1s = [], []
        for ds_name in datasets:
            key = (ds_name, method)
            if key in all_results:
                r = all_results[key]
                accs.append(r['accuracy_clean'])
                if r['shortcut_f1'] is not None:
                    f1s.append(r['shortcut_f1'])

        avg_acc = np.mean(accs) if accs else 0
        avg_f1 = np.mean(f1s) if f1s else None
        lines.append(f'{method_names[method]:<30} {format_pct(avg_acc):>12} {format_f1(avg_f1):>12}')

    lines.append('=' * 70)
    table_str = '\n'.join(lines)
    print(table_str)
    with open(os.path.join(RESULTS_DIR, 'table2.txt'), 'w') as f:
        f.write(table_str)
    return table_str


def generate_table3(all_results, datasets):
    """Table 3: Contribution of Gradient Surgery."""
    methods = ['standard_ft', 'gs_only', 'reweight_only', 'full_method']
    method_names = {
        'standard_ft': 'Standard FT (Baseline)',
        'gs_only': 'Gradient Surgery Only',
        'reweight_only': 'Reweighting Only',
        'full_method': 'Full Method (Both)',
    }

    lines = []
    lines.append('=' * 80)
    lines.append('Table 3: Contribution of Gradient Surgery')
    lines.append('=' * 80)
    lines.append(f'{"Configuration":<30} {"Accuracy":>10} {"Robustness":>12} {"Grad Align":>12}')
    lines.append('-' * 80)

    for method in methods:
        accs, robs, aligns = [], [], []
        for ds_name in datasets:
            key = (ds_name, method)
            if key in all_results:
                r = all_results[key]
                accs.append(r['accuracy_clean'])
                robs.append(r['robustness'])
                if r.get('gradient_alignment') is not None:
                    aligns.append(r['gradient_alignment'])

        avg_acc = np.mean(accs) if accs else 0
        avg_rob = np.mean(robs) if robs else 0
        avg_align = np.mean(aligns) if aligns else None

        align_str = f'{avg_align:.2f}' if avg_align is not None else '-'
        lines.append(f'{method_names[method]:<30} {format_pct(avg_acc):>10} '
                      f'{format_pct(avg_rob):>12} {align_str:>12}')

    lines.append('=' * 80)
    table_str = '\n'.join(lines)
    print(table_str)
    with open(os.path.join(RESULTS_DIR, 'table3.txt'), 'w') as f:
        f.write(table_str)
    return table_str


def generate_table4(all_results, datasets):
    """Table 4: Ablation Studies - Component Contributions."""
    lines = []
    lines.append('=' * 70)
    lines.append('Table 4: Ablation Studies - Component Contributions')
    lines.append('=' * 70)

    # Compute drops relative to full method
    full_accs, full_robs = [], []
    gs_accs, gs_robs = [], []
    rw_accs, rw_robs = [], []

    for ds_name in datasets:
        if (ds_name, 'full_method') in all_results:
            full_accs.append(all_results[(ds_name, 'full_method')]['accuracy_clean'])
            full_robs.append(all_results[(ds_name, 'full_method')]['robustness'])
        if (ds_name, 'gs_only') in all_results:
            gs_accs.append(all_results[(ds_name, 'gs_only')]['accuracy_clean'])
            gs_robs.append(all_results[(ds_name, 'gs_only')]['robustness'])
        if (ds_name, 'reweight_only') in all_results:
            rw_accs.append(all_results[(ds_name, 'reweight_only')]['accuracy_clean'])
            rw_robs.append(all_results[(ds_name, 'reweight_only')]['robustness'])

    full_acc = np.mean(full_accs) if full_accs else 0
    full_rob = np.mean(full_robs) if full_robs else 0

    lines.append(f'{"Component Removed":<35} {"Acc Drop":>12} {"Rob Drop":>12}')
    lines.append('-' * 70)

    if gs_accs:
        acc_drop = full_acc - np.mean(gs_accs)
        rob_drop = full_rob - np.mean(gs_robs)
        lines.append(f'{"Remove Reweighting (GS only)":<35} {format_pct(acc_drop):>12} {format_pct(rob_drop):>12}')

    if rw_accs:
        acc_drop = full_acc - np.mean(rw_accs)
        rob_drop = full_rob - np.mean(rw_robs)
        lines.append(f'{"Remove Grad Surgery (RW only)":<35} {format_pct(acc_drop):>12} {format_pct(rob_drop):>12}')

    lines.append(f'\nFull Method Accuracy: {format_pct(full_acc)}')
    lines.append(f'Full Method Robustness: {format_pct(full_rob)}')
    lines.append('=' * 70)

    table_str = '\n'.join(lines)
    print(table_str)
    with open(os.path.join(RESULTS_DIR, 'table4.txt'), 'w') as f:
        f.write(table_str)
    return table_str


def generate_figure3(collected_data, all_results=None, datasets=None):
    """Figure 3: Empirical validation of ShortcutScore + reweight curve overlay.

    Accepts either:
      (A) a single-dataset dict with keys 'scores' / 'is_shortcut' / 'alignments'
          (legacy single-call mode).
      (B) a dict-of-dicts keyed by dataset name; each value is the (A) shape.
          All datasets are concatenated before plotting (used by the post-hoc
          diagnostic regeneration script).

    Panels:
      (a) S(s) vs shortcut rate. Faint per-sample raster (gray) under 20-bin
          scatter (blue) + linear fit (red dashed). Title reports both
          r_bin (legacy bin-aggregated Pearson) and r_sample (per-sample
          Pearson) — closes Gap 1 in shortcut_score_followups.qmd.
      (b) A(s) split into 4 cells against the F1-optimal threshold tau_S* and
          ground-truth is_shortcut: TP / FP / TN / FN. Title reports FP-cell
          mean A(s), gate-fire rate (% of FPs with A(s) < 0), and mean
          reweight w(s) on FPs — closes Gap 2.
      (c) Reweight curve w(s) = exp(-lambda * S(s)) plus the empirical S(s)
          density on a twin gray axis. The misplaced tau_A vline is removed;
          lambda is taken from C.lambda_. Title reports the % of samples with
          w(s) > 0.5 — closes Gap 3a.

    Returns:
        (fig_path, summary_dict). summary_dict carries the diagnostic numbers
        printed in the panel titles, suitable for downstream logging.
    """
    from scipy.stats import pearsonr

    # --- Aggregation: handle dict-of-dicts (multi-dataset) input ------------
    if collected_data and 'scores' not in collected_data:
        agg = {'scores': [], 'is_shortcut': [], 'alignments': [], 'concentrations': []}
        for ds_name, d in collected_data.items():
            for k in agg:
                agg[k].extend(d.get(k, []))
        collected_data = agg

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    summary = {'lambda': float(C.lambda_)}

    # ============================================================
    # Panel (a) — Gap 1: per-sample Pearson + raster scatter
    # ============================================================
    r_bin = None
    r_sample = None
    if collected_data.get('scores') and collected_data.get('is_shortcut'):
        scores = np.array(collected_data['scores'], dtype=float)
        is_sc = np.array(collected_data['is_shortcut'], dtype=float)

        r_sample, _ = pearsonr(scores, is_sc)

        # Faint per-sample raster: shows where samples actually live on S(s).
        axes[0].scatter(scores, is_sc, alpha=0.05, s=2, color='gray',
                        zorder=1, label=f'per-sample n={len(scores)}')

        n_bins = 20
        bins_a = np.linspace(scores.min(), scores.max(), n_bins + 1)
        bin_centers, bin_rates = [], []
        for i in range(n_bins):
            mask = (scores >= bins_a[i]) & (scores < bins_a[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins_a[i] + bins_a[i + 1]) / 2)
                bin_rates.append(is_sc[mask].mean())

        axes[0].scatter(bin_centers, bin_rates, alpha=0.95, s=40,
                        color='steelblue', zorder=3, label='20-bin mean')
        if len(bin_centers) > 1:
            z = np.polyfit(bin_centers, bin_rates, 1)
            x_line = np.linspace(min(bin_centers), max(bin_centers), 100)
            axes[0].plot(x_line, np.poly1d(z)(x_line), 'r--', alpha=0.7,
                         zorder=2, label='linear fit')
            r_bin = float(np.corrcoef(bin_centers, bin_rates)[0, 1])

        title = '(a) S(s) vs Shortcut Rate\n'
        title += (f'r_bin={r_bin:.2f}, ' if r_bin is not None else '')
        title += f'r_sample={r_sample:.2f}'
        axes[0].set_title(title)
        axes[0].set_xlabel('ShortcutScore S(s)')
        axes[0].set_ylabel('is_shortcut / Shortcut Rate')
        axes[0].set_ylim(-0.05, 1.05)
        axes[0].legend(loc='center right', fontsize=8)
    summary['r_bin'] = r_bin
    summary['r_sample'] = float(r_sample) if r_sample is not None else None

    # ============================================================
    # Panel (b) — Gap 2: 4-cell A(s) split at F1-optimal tau_S*
    # ============================================================
    fp_mean_A = fp_gate_rate = fp_mean_w = best_f1 = tau_S_star = float('nan')
    if (collected_data.get('alignments') and collected_data.get('is_shortcut')
            and collected_data.get('scores')):
        alignments = np.array(collected_data['alignments'], dtype=float)
        is_sc = np.array(collected_data['is_shortcut'], dtype=float)
        scores = np.array(collected_data['scores'], dtype=float)

        # F1-optimal threshold sweep (mirrors src/evaluate.py:246).
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
                         label=f'TP (n={int(tp_mask.sum())})',
                         color='red', density=True)
        if fp_mask.sum() > 0:
            axes[1].hist(alignments[fp_mask], bins=bins_b, alpha=0.7,
                         label=f'FP (n={int(fp_mask.sum())})',
                         color='orange', density=True)
        if tn_mask.sum() > 0:
            axes[1].hist(alignments[tn_mask], bins=bins_b, alpha=0.4,
                         label=f'TN (n={int(tn_mask.sum())})',
                         color='green', density=True)
        if fn_mask.sum() > 0:
            axes[1].hist(alignments[fn_mask], bins=bins_b, alpha=0.5,
                         label=f'FN (n={int(fn_mask.sum())})',
                         color='purple', density=True,
                         histtype='step', linewidth=1.6)

        if fp_mask.sum() > 0:
            fp_mean_A = float(alignments[fp_mask].mean())
            fp_gate_rate = float((alignments[fp_mask] < 0).mean() * 100)
            fp_mean_w = float(np.exp(-C.lambda_ * scores[fp_mask]).mean())

        axes[1].set_xlabel('Gradient Alignment A(s)')
        axes[1].set_ylabel('Density')
        axes[1].set_title(
            f'(b) A(s) by S(s)>tau* (F1={best_f1:.2f}, tau*={tau_S_star:.2f})\n'
            f'FP mean A={fp_mean_A:+.2f}, gate-fire={fp_gate_rate:.0f}%, '
            f'mean w={fp_mean_w:.2f}'
        )
        axes[1].legend(fontsize=8, loc='upper left')
    summary['tau_S_star'] = tau_S_star
    summary['best_F1'] = best_f1
    summary['FP_mean_A'] = fp_mean_A
    summary['FP_gate_fire_pct'] = fp_gate_rate
    summary['FP_mean_w'] = fp_mean_w

    # ============================================================
    # Panel (c) — Gap 3a: reweight curve + empirical S(s) density
    # ============================================================
    scores_for_c = np.array(collected_data.get('scores', []), dtype=float)
    if len(scores_for_c) > 0:
        S_max = max(3.0, float(scores_for_c.max()) * 1.1)
    else:
        S_max = 3.0

    S_range = np.linspace(0, S_max, 200)
    weights = np.exp(-C.lambda_ * S_range)

    if len(scores_for_c) > 0:
        ax_density = axes[2].twinx()
        ax_density.hist(scores_for_c, bins=40, alpha=0.3, density=True,
                        color='gray', label='S(s) density')
        ax_density.set_ylabel('S(s) density', color='gray')
        ax_density.tick_params(axis='y', labelcolor='gray')

    axes[2].plot(S_range, weights, 'b-', linewidth=2, label=r'$w(s)=e^{-\lambda S}$')
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('ShortcutScore S(s)')
    axes[2].set_ylabel('Sample Weight w(s)', color='blue')
    axes[2].tick_params(axis='y', labelcolor='blue')
    axes[2].set_ylim(0, 1.05)
    axes[2].set_zorder(2)
    axes[2].patch.set_visible(False)

    pct_above_half = float('nan')
    if len(scores_for_c) > 0:
        sample_w = np.exp(-C.lambda_ * scores_for_c)
        pct_above_half = float((sample_w > 0.5).mean() * 100)
        axes[2].set_title(
            f'(c) Reweight Curve (lambda={C.lambda_})\n'
            f'{pct_above_half:.0f}% of samples have w(s) > 0.5'
        )
    else:
        axes[2].set_title(f'(c) Reweight Curve (lambda={C.lambda_})')

    axes[2].legend(loc='upper right', fontsize=9)
    summary['pct_w_above_0.5'] = pct_above_half

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, 'figure3.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Figure 3 saved to {fig_path}')
    print(f'  diagnostics: {summary}')
    return fig_path, summary


def generate_training_curves(training_logs, save_name='training_curves.png'):
    """Plot training loss curves for different methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for method_name, losses in training_logs.items():
        ax.plot(losses, label=method_name, alpha=0.8)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig_path = os.path.join(RESULTS_DIR, save_name)
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Training curves saved to {fig_path}')
    return fig_path


def generate_summary_bar_chart(all_results, datasets):
    """Bar chart comparing methods across accuracy and robustness."""
    methods = ['standard_ft', 'self_consistency', 'data_filtering',
               'jtt', 'focal_loss', 'group_dro',
               'irm', 'vrex', 'fishr', 'lff', 'influence_filtering', 'meta_reweight',
               'full_method']
    method_labels = ['Standard FT', 'Self-Consistency', 'Data Filtering',
                     'JTT', 'Focal Loss', 'Group DRO',
                     'IRM', 'V-REx', 'Fishr', 'LfF', 'Influence Filt.', 'Meta-Reweight',
                     'Our Method']

    avg_clean = []
    avg_perturbed = []
    for method in methods:
        cleans, perturbs = [], []
        for ds_name in datasets:
            key = (ds_name, method)
            if key in all_results:
                cleans.append(all_results[key]['accuracy_clean'])
                perturbs.append(all_results[key]['robustness'])
        avg_clean.append(np.mean(cleans) if cleans else 0)
        avg_perturbed.append(np.mean(perturbs) if perturbs else 0)

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width/2, [v*100 for v in avg_clean], width, label='Clean Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, [v*100 for v in avg_perturbed], width, label='Perturbed Accuracy (Robustness)', color='coral')

    ax.set_xlabel('Method')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Method Comparison: Clean vs Perturbed Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(method_labels, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontsize=8)

    fig_path = os.path.join(RESULTS_DIR, 'comparison_bar_chart.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Bar chart saved to {fig_path}')
    return fig_path
