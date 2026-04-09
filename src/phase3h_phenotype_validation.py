"""Phase 3-H: Phenotype Validation Against Empirical Win-Stay/Lose-Stay Rates.

Validates that model-based phenotype labels (σ_pc/σ_wsls ratio) correspond
to directly observable behavioral signatures.

Win-stay  = P(same choice | previous trial rewarded)
Lose-stay = P(same choice | previous trial unrewarded)

The WS−LS gap should correlate with log10(σ_pc/σ_wsls):
  - Perseverators: repeat choices regardless → gap near zero
  - Reward-sensitive: modulate based on outcome → more negative gap

Result: Spearman ρ = 0.309, p = 0.025
Output: Figure A6 (fig_phenotype_validation.pdf), Table A3
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kruskal, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (CHUNK_EXPLORATION_FILE, SUMMARY_CSV_FILE,
                    FIGURE_DIR, TABLE_DIR, COLORS, LOG_THRESHOLD)
from src.utils import setup_plotting, save_figure, classify_phenotype

SIGMA_FLOOR = 1e-10
MIN_WIN_TRIALS = 20
MIN_LOSS_TRIALS = 20


def compute_ws_ls_rates(trial_df):
    """Compute empirical win-stay and lose-stay rates from trial data.

    Win-stay:  P(repeat choice | previous trial was rewarded)
    Lose-stay: P(repeat choice | previous trial was unrewarded)

    Excludes the first trial of each session (no previous trial available).

    Args:
        trial_df: DataFrame with columns: choice, rewarded, date, session

    Returns:
        dict with ws_rate, ls_rate, ws_ls_gap, n_win, n_loss
        or None if insufficient data
    """
    df = trial_df.sort_values(['date', 'session']).reset_index(drop=True)

    # Identify session boundaries
    if 'chunk_session' in df.columns:
        session_col = 'chunk_session'
    else:
        session_col = 'session'

    session_starts = set()
    prev_session = None
    for i, sess in enumerate(df[session_col]):
        if sess != prev_session:
            session_starts.add(i)
            prev_session = sess

    # Compute stay/switch for each trial (excluding session starts)
    n_win_stay = 0
    n_win_total = 0
    n_loss_stay = 0
    n_loss_total = 0

    for i in range(1, len(df)):
        if i in session_starts:
            continue  # No previous trial at session start

        prev_choice = df.iloc[i-1]['choice']
        curr_choice = df.iloc[i]['choice']
        prev_rewarded = df.iloc[i-1]['rewarded']
        stayed = (curr_choice == prev_choice)

        if prev_rewarded == 1:
            n_win_total += 1
            if stayed:
                n_win_stay += 1
        else:
            n_loss_total += 1
            if stayed:
                n_loss_stay += 1

    if n_win_total < MIN_WIN_TRIALS or n_loss_total < MIN_LOSS_TRIALS:
        return None

    ws_rate = n_win_stay / n_win_total
    ls_rate = n_loss_stay / n_loss_total

    return {
        'ws_rate': ws_rate,
        'ls_rate': ls_rate,
        'ws_ls_gap': ws_rate - ls_rate,
        'n_win': n_win_total,
        'n_loss': n_loss_total,
    }


def run_phenotype_validation():
    """Run phenotype validation analysis."""
    print("\n" + "=" * 70)
    print("PHASE 3-H: Phenotype Validation")
    print("=" * 70)

    setup_plotting()

    summary_df = pd.read_csv(SUMMARY_CSV_FILE)
    complete_df = summary_df[summary_df['n_chunks_fit'] == 3].copy()

    with open(CHUNK_EXPLORATION_FILE, 'rb') as f:
        chunk_data = pickle.load(f)

    # Compute phenotype log-ratios
    pc = np.maximum(complete_df['chunk3_sigma_trial_prev_choice'].values, SIGMA_FLOOR)
    wsls = np.maximum(complete_df['chunk3_sigma_trial_wsls'].values, SIGMA_FLOOR)
    complete_df['log_ratio'] = np.log10(pc) - np.log10(wsls)
    complete_df['phenotype'] = pd.Series(pc / wsls).apply(classify_phenotype).values

    # Compute empirical WS/LS for each mouse from C3 trial data
    results = []
    for _, row in complete_df.iterrows():
        subject = row['subject']
        mouse_data = chunk_data.get('mouse_data', {}).get(subject, {})
        c3_trials = mouse_data.get('trial_data', {}).get('chunk3')

        if c3_trials is None or len(c3_trials) == 0:
            continue

        ws_ls = compute_ws_ls_rates(c3_trials)
        if ws_ls is None:
            continue

        results.append({
            'subject': subject,
            'log_ratio': row['log_ratio'],
            'phenotype': row['phenotype'],
            **ws_ls,
        })

    results_df = pd.DataFrame(results)
    print(f"  Mice with valid WS/LS data: {len(results_df)}")

    if len(results_df) < 10:
        print("  Too few mice for analysis.")
        return

    # ── Spearman correlation ──
    rho, p_spearman = spearmanr(results_df['log_ratio'], results_df['ws_ls_gap'])
    print(f"\n  Spearman correlation (log-ratio vs WS−LS gap):")
    print(f"    ρ = {rho:.3f}, p = {p_spearman:.4f}")

    # ── Kruskal-Wallis by phenotype group ──
    groups = {}
    for pheno in ['perseverator', 'mixed', 'reward_sensitive']:
        vals = results_df[results_df['phenotype'] == pheno]['ws_ls_gap'].values
        if len(vals) > 0:
            groups[pheno] = vals

    if len(groups) >= 2:
        H, p_kw = kruskal(*groups.values())
        print(f"\n  Kruskal-Wallis (WS−LS gap by phenotype):")
        print(f"    H = {H:.2f}, p = {p_kw:.3f}")

    # ── Per-phenotype summary ──
    print(f"\n  Per-phenotype summary:")
    for pheno in ['perseverator', 'mixed', 'reward_sensitive']:
        subset = results_df[results_df['phenotype'] == pheno]
        if len(subset) > 0:
            print(f"    {pheno}: n={len(subset)}, "
                  f"WS={subset['ws_rate'].mean():.3f}, "
                  f"LS={subset['ls_rate'].mean():.3f}, "
                  f"gap={subset['ws_ls_gap'].mean():.3f}")

    # ── Figure A6: 2-panel ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Panel A: Scatter with regression line
    point_colors = [COLORS.get(p, 'gray') for p in results_df['phenotype']]
    ax1.scatter(results_df['log_ratio'], results_df['ws_ls_gap'],
                c=point_colors, s=50, alpha=0.7, edgecolors='white', linewidth=0.5)

    # Add regression line (linear fit for visual reference)
    x_vals = results_df['log_ratio'].values
    y_vals = results_df['ws_ls_gap'].values
    valid = ~(np.isnan(x_vals) | np.isnan(y_vals))
    if valid.sum() > 2:
        z = np.polyfit(x_vals[valid], y_vals[valid], 1)
        x_line = np.linspace(x_vals[valid].min(), x_vals[valid].max(), 100)
        ax1.plot(x_line, np.polyval(z, x_line), '--', color='black', linewidth=1,
                 alpha=0.6, label='Linear fit')

    ax1.axvline(LOG_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(-LOG_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax1.set_xlabel(r'$\log_{10}(\sigma_{pc}/\sigma_{wsls})$')
    ax1.set_ylabel('WS − LS gap')
    ax1.set_title('A. σ-ratio vs Empirical Gap', fontweight='bold', loc='left')
    ax1.text(0.02, 0.02, f'Spearman ρ = {rho:.3f}, p = {p_spearman:.3f}',
             transform=ax1.transAxes, fontsize=9)
    ax1.legend(fontsize=8)

    # Panel B: Grouped box plots — transparent boxes with visible data points
    group_data = []
    group_labels = []
    group_colors_list = []
    for pheno in ['reward_sensitive', 'mixed', 'perseverator']:
        vals = results_df[results_df['phenotype'] == pheno]['ws_ls_gap'].values
        if len(vals) > 0:
            group_data.append(vals)
            group_labels.append(pheno.replace('_', '\n'))
            group_colors_list.append(COLORS.get(pheno, 'gray'))

    if group_data:
        bp = ax2.boxplot(group_data, labels=group_labels,
                         patch_artist=True, widths=0.5)
        for patch, color in zip(bp['boxes'], group_colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.3)

        # Overlay individual data points
        for i, (vals, color) in enumerate(zip(group_data, group_colors_list)):
            x_jitter = np.random.normal(i + 1, 0.06, len(vals))
            ax2.scatter(x_jitter, vals, color=color, s=20, alpha=0.7, zorder=3,
                        edgecolor='white', linewidth=0.3)

    ax2.set_ylabel('WS − LS gap')
    ax2.set_title('B. Gap by Phenotype', fontweight='bold', loc='left')

    # Add KW stats if available
    try:
        ax2.text(0.98, 0.98, f'Kruskal-Wallis H = {H:.2f}, p = {p_kw:.3f}',
                 transform=ax2.transAxes, ha='right', va='top', fontsize=8,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    except NameError:
        pass

    fig.suptitle('Phenotype Validation: Empirical Win-Stay/Lose-Stay Rates',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_phenotype_validation'))

    # ── Table 14: Phenotype Validation (CSV + LaTeX) ──
    # Row order matches thesis: Reward-sensitive, Mixed, Perseverator
    table_rows = []
    for pheno in ['reward_sensitive', 'mixed', 'perseverator']:
        subset = results_df[results_df['phenotype'] == pheno]
        if len(subset) > 0:
            table_rows.append({
                'Phenotype': pheno,
                'N': len(subset),
                'WS_mean': subset["ws_rate"].mean(),
                'WS_sd': subset["ws_rate"].std(),
                'LS_mean': subset["ls_rate"].mean(),
                'LS_sd': subset["ls_rate"].std(),
                'Gap_mean': subset["ws_ls_gap"].mean(),
                'Gap_sd': subset["ws_ls_gap"].std(),
            })

    table_df = pd.DataFrame(table_rows)
    table_df.to_csv(os.path.join(TABLE_DIR, 'table_phenotype_validation.csv'), index=False)

    # LaTeX — matching thesis format exactly
    from config import LATEX_DIR
    os.makedirs(LATEX_DIR, exist_ok=True)

    latex = '\\begin{table}[htbp]\n\\centering\n'
    latex += ('\\caption{\\textbf{Phenotype Validation: Empirical Win-Stay and Lose-Stay Rates '
              'by Model-Based Phenotype (Chunk 3).} '
              'Win-stay (WS) = $P(\\text{stay}\\mid\\text{previous trial rewarded})$; '
              'lose-stay (LS) = $P(\\text{stay}\\mid\\text{previous trial unrewarded})$. '
              'The WS$-$LS gap summarizes outcome modulation in choice repetition.}\n')
    latex += '\\label{tab:phenotype_ws_ls}\n'
    latex += '\\begin{tabular}{lcccc}\n\\toprule\n'
    latex += ('Phenotype & $n$ & Win-stay (mean $\\pm$ SD) '
              '& Lose-stay (mean $\\pm$ SD) & WS$-$LS gap (mean $\\pm$ SD) \\\\\n')
    latex += '\\midrule\n'

    pheno_labels = {
        'reward_sensitive': 'Reward-sensitive',
        'mixed': 'Mixed',
        'perseverator': 'Perseverator',
    }
    for r in table_rows:
        label = pheno_labels.get(r['Phenotype'], r['Phenotype'])
        gap_str = f'$-${abs(r["Gap_mean"]):.3f}' if r['Gap_mean'] < 0 else f'{r["Gap_mean"]:.3f}'
        latex += (f'{label} & {r["N"]} & '
                  f'{r["WS_mean"]:.3f} $\\pm$ {r["WS_sd"]:.3f} & '
                  f'{r["LS_mean"]:.3f} $\\pm$ {r["LS_sd"]:.3f} & '
                  f'{gap_str} $\\pm$ {r["Gap_sd"]:.3f} \\\\\n')

    # Summary statistics rows
    latex += '\\midrule\n'
    latex += (f'\\multicolumn{{5}}{{l}}{{\\textit{{Continuous validation:}} '
              f'Spearman $\\rho = {rho:.3f}$, $p = {p_spearman:.3f}$}} \\\\\n')

    if len(groups) >= 2:
        latex += (f'\\multicolumn{{5}}{{l}}{{\\textit{{Grouped test:}} '
                  f'Kruskal-Wallis $H = {H:.2f}$, $p = {p_kw:.3f}$}} \\\\\n')

    latex += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'

    with open(os.path.join(LATEX_DIR, 'table_phenotype_validation.tex'), 'w') as f:
        f.write(latex)

    print(f"\n  Saved: table_phenotype_validation.csv + .tex (Table 14)")
    print("  PHASE 3-H COMPLETE")


if __name__ == '__main__':
    run_phenotype_validation()
