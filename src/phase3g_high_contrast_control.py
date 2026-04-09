"""Phase 3-G: High-Contrast Control Analysis.

Tests the stimulus-statistics confound: does the σ_trial,contrast increase
from C1→C3 survive when C3 is restricted to physical contrasts ≥ 25%?

By removing 0%, 6.25%, and 12.5% trials from C3, we bring the predictor
range closer to C1 (which only has 100% and 50%). If the σ increase were
purely a low-contrast estimation artifact, it should vanish.

Result: +98% increase retained, ~90% of original C1→C3 effect preserved.
Output: Figure 23 (fig_high_contrast_control.pdf)
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (CHUNK_EXPLORATION_FILE, SUMMARY_CSV_FILE, FIGURE_DIR,
                    INTERMEDIATE_DIR, WEIGHT_NAMES, K,
                    SIGMA_INIT, SIGINIT_INIT, SIGDAY_INIT,
                    CHUNK_COLORS)
from src.utils import (setup_plotting, save_figure, build_predictors,
                       paired_wilcoxon_with_effect_size)

try:
    import psytrack as psy
except ImportError:
    psy = None


def run_high_contrast_control():
    """Run high-contrast control: refit C3 on trials with contrast ≥ 25%."""
    print("\n" + "=" * 70)
    print("PHASE 3-G: High-Contrast Control")
    print("=" * 70)

    setup_plotting()

    summary_df = pd.read_csv(SUMMARY_CSV_FILE)
    complete_df = summary_df[summary_df['n_chunks_fit'] == 3].copy()

    with open(CHUNK_EXPLORATION_FILE, 'rb') as f:
        chunk_data = pickle.load(f)

    # ── Check for cached results ──
    refit_csv = os.path.join(INTERMEDIATE_DIR, 'high_contrast_control.csv')
    if os.path.exists(refit_csv):
        print(f"  Loading previous refit results from {refit_csv}")
        print("  (Delete this file to force re-run)")
        refit_df = pd.read_csv(refit_csv)
        subjects = refit_df['subject'].tolist()
        c1 = refit_df['c1_sigma_trial_contrast'].values
        c3h = refit_df['c3_high_sigma_trial_contrast'].values
        c3o = refit_df['c3_orig_sigma_trial_contrast'].values
    else:
        if psy is None:
            print("  psytrack not installed — skipping refit. pip install psytrack")
            return

        c1_values, c3_high_values, c3_orig_values, subjects = [], [], [], []

        for _, row in complete_df.iterrows():
            subject = row['subject']
            mouse_data = chunk_data.get('mouse_data', {}).get(subject, {})
            c3_trials = mouse_data.get('trial_data', {}).get('chunk3')

            if c3_trials is None or len(c3_trials) == 0:
                continue

            # Filter: keep only trials where max physical contrast ≥ 0.25
            max_c = np.maximum(c3_trials['contrast_left'].fillna(0),
                               c3_trials['contrast_right'].fillna(0))
            c3_high = c3_trials[max_c >= 0.25].copy()

            if len(c3_high) < 200:
                continue

            inputs, day_lengths, n_trials, _, _ = build_predictors(c3_high)
            dat = {'y': c3_high['choice'].values.astype(float),
                   'dayLength': day_lengths, 'inputs': inputs}
            weights = {'bias': 1, 'contrast': 1, 'prev_choice': 1, 'wsls': 1}
            n_days = len(day_lengths)

            hyper = {'sigma': [SIGMA_INIT]*K, 'sigInit': [SIGINIT_INIT]*K}
            opt_list = ['sigma']
            if n_days > 1:
                hyper['sigDay'] = [SIGDAY_INIT]*K
                opt_list.append('sigDay')

            try:
                hyp, _, _, _ = psy.hyperOpt(dat, hyper, weights,
                                            optList=opt_list, showOpt=False)
                sigma_vals = hyp.get('sigma', [np.nan]*K)
                if isinstance(sigma_vals, np.ndarray):
                    sigma_vals = sigma_vals.flatten().tolist()
                sigma_c3h = float(sigma_vals[1])

                c1_values.append(row['chunk1_sigma_trial_contrast'])
                c3_high_values.append(sigma_c3h)
                c3_orig_values.append(row['chunk3_sigma_trial_contrast'])
                subjects.append(subject)
                print(f"  {subject}: C1={row['chunk1_sigma_trial_contrast']:.4f} "
                      f"C3h={sigma_c3h:.4f} ({len(c3_high)} trials)")
            except Exception as e:
                print(f"  {subject}: FAILED — {e}")

        c1 = np.array(c1_values)
        c3h = np.array(c3_high_values)
        c3o = np.array(c3_orig_values)

        # Save refit results to CSV
        refit_df = pd.DataFrame({
            'subject': subjects,
            'c1_sigma_trial_contrast': c1,
            'c3_high_sigma_trial_contrast': c3h,
            'c3_orig_sigma_trial_contrast': c3o,
        })
        os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
        refit_df.to_csv(refit_csv, index=False)
        print(f"  Saved: {refit_csv}")

    # ── From here on, runs regardless (cached or fresh) ──
    n = len(c1)
    print(f"\n  Paired fits: {n}")

    if n < 10:
        print("  Too few for statistics.")
        return

    result = paired_wilcoxon_with_effect_size(c1, c3h)
    c1_med, c3h_med, c3o_med = np.median(c1), np.median(c3h), np.median(c3o)
    pct = (c3h_med - c1_med) / c1_med * 100
    retained = (c3h_med - c1_med) / (c3o_med - c1_med) * 100 if c3o_med != c1_med else 0
    n_pos = (c3h > c1).sum()

    print(f"  C1 median={c1_med:.4f}, C3h median={c3h_med:.4f}, C3 median={c3o_med:.4f}")
    print(f"  Increase: +{pct:.0f}%, retained ~{retained:.0f}% of original")
    print(f"  {n_pos}/{n} mice show increase")
    print(f"  W={result['stat']:.0f}, p={result['p_value']:.6f}, r={result['effect_size']:.2f}")

    # ── Figure 16: 3-panel (thesis numbering) ──
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 5))

    # Panel A: Paired comparison with individual dots
    for i in range(n):
        ax1.plot([1, 2], [c1[i], c3h[i]], '-', color='gray', alpha=0.3, linewidth=0.8)
    # Scatter individual data points at C1 and C3-high
    x1_jitter = np.random.normal(1, 0.04, n)
    x2_jitter = np.random.normal(2, 0.04, n)
    ax1.scatter(x1_jitter, c1, color=CHUNK_COLORS[0], s=15, alpha=0.5, zorder=4)
    ax1.scatter(x2_jitter, c3h, color=CHUNK_COLORS[2], s=15, alpha=0.5, zorder=4)
    ax1.plot(1, c1_med, 'D', color=CHUNK_COLORS[0], markersize=10, zorder=5)
    ax1.plot(2, c3h_med, 'D', color=CHUNK_COLORS[2], markersize=10, zorder=5)
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['C1\n(original)', 'C3-high\n(≥25% only)'])
    ax1.set_ylabel(r'$\sigma_{trial,contrast}$')
    ax1.set_title('A. Paired Comparison', fontweight='bold', loc='left')
    # Add W, p, r, n stats
    ax1.text(0.02, 0.98,
             f'W={result["stat"]:.0f}, p={result["p_value"]:.4f}\n'
             f'r={result["effect_size"]:.2f}, n={n}',
             transform=ax1.transAxes, va='top', fontsize=8,
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Panel B: Difference distribution with median line
    diffs = c3h - c1
    med_diff = np.median(diffs)
    ax2.hist(diffs, bins=20, color='gray', edgecolor='white', alpha=0.7)
    ax2.axvline(0, color='black', linestyle='--', linewidth=1)
    ax2.axvline(med_diff, color='red', linestyle='-', linewidth=1.5,
                label=f'Median = {med_diff:.4f}')
    ax2.set_xlabel(r'$\Delta\sigma_{trial,contrast}$ (C3-high $-$ C1)')
    ax2.set_ylabel('Count')
    ax2.set_title(f'B. Differences ({n_pos}/{n} positive)',
                  fontweight='bold', loc='left')
    ax2.legend(fontsize=8)

    # Panel C: Box plots with transparent boxes and visible data points
    bp = ax3.boxplot([c1, c3h, c3o],
                     labels=['C1', 'C3-high\n(≥25%)', 'C3\n(orig)'],
                     patch_artist=True, widths=0.5)
    box_colors = [CHUNK_COLORS[0], '#AA5555', CHUNK_COLORS[2]]
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    # Overlay individual data points
    for i, (vals, color) in enumerate(zip([c1, c3h, c3o], box_colors)):
        x_jitter = np.random.normal(i + 1, 0.06, len(vals))
        ax3.scatter(x_jitter, vals, color=color, s=15, alpha=0.6, zorder=3,
                    edgecolor='white', linewidth=0.3)

    ax3.set_ylabel(r'$\sigma_{trial,contrast}$')
    ax3.set_title('C. Effect Attenuation', fontweight='bold', loc='left')

    fig.suptitle('High-Contrast Control', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_high_contrast_control'))
    print("  PHASE 3-G COMPLETE")


if __name__ == '__main__':
    run_high_contrast_control()
