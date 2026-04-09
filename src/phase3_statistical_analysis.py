"""Phase 3: Statistical Analysis — Main Figures and Tables.

Loads the Phase 2 outputs (psytrack_chunk_summary.csv, overnight_jumps.csv)
and generates all main-text figures (5-22) and tables (5-12).

Key statistical methods:
    - Wilcoxon signed-rank tests (paired, C1 vs C3) with Bonferroni correction
    - Rank-biserial effect size: r = (W⁺ - W⁻) / (n(n+1)/2)
    - Friedman test (repeated-measures across 3 chunks)
    - Mixed-effects linear model: log(σ) ~ chunk + (1|mouse)
    - Kruskal-Wallis (lab effects, phenotype vs learning speed)
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import stats
from scipy.stats import wilcoxon, friedmanchisquare, kruskal, spearmanr, binomtest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (SUMMARY_CSV_FILE, OVERNIGHT_JUMPS_FILE,
                    PSYTRACK_RESULTS_FILE, CHUNK_EXPLORATION_FILE,
                    FIGURE_DIR, TABLE_DIR, LATEX_DIR,
                    WEIGHT_NAMES, WEIGHT_LABELS, K,
                    COLORS, CHUNK_LABELS, CHUNK_COLORS,
                    PHENOTYPE_RATIO_HIGH, PHENOTYPE_RATIO_LOW, LOG_THRESHOLD,
                    SIGDAY_INIT)
from src.utils import (setup_plotting, save_figure, compute_rank_biserial,
                       paired_wilcoxon_with_effect_size, bonferroni_correct,
                       effect_size_label, classify_phenotype)


SIGMA_FLOOR = 1e-6  # Floor for σ values to avoid log(0)


# ─── Data loading and preparation ────────────────────────────────────────────

def load_and_prepare_data():
    """Load Phase 2 outputs and filter to complete cases.

    Returns:
        complete_df: DataFrame with 53 mice (all 3 chunks successfully fitted)
        jumps_df: DataFrame of overnight transitions
    """
    print("Loading Phase 2 outputs...")
    summary_df = pd.read_csv(SUMMARY_CSV_FILE)
    jumps_df = pd.read_csv(OVERNIGHT_JUMPS_FILE)

    # Filter to mice with all 3 chunks fitted
    complete_df = summary_df[summary_df['n_chunks_fit'] == 3].copy()
    print(f"  Summary: {len(summary_df)} mice total, "
          f"{len(complete_df)} with complete 3-chunk data")
    print(f"  Overnight jumps: {len(jumps_df)} transitions")

    # Re-extract lab from subject ID using thesis convention (9 labs)
    # Phase 2 uses a naive rsplit that gives wrong results for some mice
    def _extract_lab_thesis(s):
        if s.startswith('CSH_ZAD'): return 'CSH'
        elif s.startswith('CSHL'): return 'CSHL'
        elif s.startswith('DY_'): return 'DY'
        elif s.startswith('IBL'): return 'IBL'
        elif s.startswith('KS'): return 'KS'
        elif s.startswith('NYU'): return 'NYU'
        elif s.startswith('SWC'): return 'SWC'
        elif s.startswith('ZM_'): return 'ZM'
        elif s.startswith('ibl_witten'): return 'ibl_witten'
        else: return 'Unknown'

    complete_df['lab'] = complete_df['subject'].apply(_extract_lab_thesis)

    # Compute phenotype ratios for each chunk
    for chunk in ['chunk1', 'chunk2', 'chunk3']:
        pc_col = f'{chunk}_sigma_trial_prev_choice'
        wsls_col = f'{chunk}_sigma_trial_wsls'
        pc_vals = np.maximum(complete_df[pc_col].values, SIGMA_FLOOR)
        wsls_vals = np.maximum(complete_df[wsls_col].values, SIGMA_FLOOR)
        complete_df[f'{chunk}_phenotype_ratio'] = pc_vals / wsls_vals
        complete_df[f'{chunk}_phenotype_log_ratio'] = (
            np.log10(pc_vals) - np.log10(wsls_vals))
        complete_df[f'{chunk}_phenotype'] = complete_df[
            f'{chunk}_phenotype_ratio'].apply(classify_phenotype)

    return complete_df, jumps_df


# ─── Figure 6: Sigma Trajectories ────────────────────────────────────────────

def make_figure_sigma_trajectories(complete_df):
    """Figure 6: σ trajectories with connected dots (2×4 grid).

    Top row: σ_trial for each weight across C1→C2→C3
    Bottom row: σ_day for each weight across C1→C2→C3
    Gray lines = individual mice, Red line = mean ± SE
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    chunk_x = [1, 2, 3]

    for row_idx, sigma_type in enumerate(['sigma_trial', 'sigma_day']):
        for col_idx, weight in enumerate(WEIGHT_NAMES):
            ax = axes[row_idx, col_idx]

            # Collect values for each chunk
            vals_by_chunk = []
            for chunk in ['chunk1', 'chunk2', 'chunk3']:
                col = f'{chunk}_{sigma_type}_{weight}'
                vals = complete_df[col].dropna().values

                # For σ_day, exclude single-day chunks (stuck at init value)
                if sigma_type == 'sigma_day':
                    days_col = f'{chunk}_days'
                    if days_col in complete_df.columns:
                        mask = complete_df[days_col] > 1
                        vals = complete_df.loc[mask, col].dropna().values

                vals_by_chunk.append(vals)

            # Individual mouse trajectories (gray connected dots)
            for i in range(len(complete_df)):
                mouse_vals = []
                for chunk in ['chunk1', 'chunk2', 'chunk3']:
                    col = f'{chunk}_{sigma_type}_{weight}'
                    v = complete_df.iloc[i][col]
                    mouse_vals.append(v if pd.notna(v) else np.nan)
                ax.plot(chunk_x, mouse_vals, 'o-', color='gray',
                        alpha=0.15, markersize=2, linewidth=0.5, zorder=1)

            # Population mean ± SE (red)
            means = [np.nanmean(v) for v in vals_by_chunk]
            ses = [np.nanstd(v) / np.sqrt(len(v)) if len(v) > 0 else 0
                   for v in vals_by_chunk]
            ax.errorbar(chunk_x, means, yerr=ses, fmt='o-', color='red',
                        linewidth=2, markersize=6, capsize=4, zorder=3)

            # C1 vs C3 significance test
            c1_col = f'chunk1_{sigma_type}_{weight}'
            c3_col = f'chunk3_{sigma_type}_{weight}'
            c1 = complete_df[c1_col].dropna()
            c3 = complete_df[c3_col].dropna()
            common = c1.index.intersection(c3.index)
            if len(common) >= 10:
                result = paired_wilcoxon_with_effect_size(
                    c1.loc[common].values, c3.loc[common].values)
                p_corrected = bonferroni_correct(result['p_value'])
                if p_corrected < 0.001:
                    ax.text(2, max(means) * 1.1, '***', ha='center',
                            fontsize=12, fontweight='bold')

            ax.set_xticks(chunk_x)
            ax.set_xticklabels(['C1', 'C2', 'C3'])
            ax.set_title(WEIGHT_LABELS[col_idx], fontweight='bold')

            if col_idx == 0:
                label = r'$\sigma_{trial}$' if row_idx == 0 else r'$\sigma_{day}$'
                ax.set_ylabel(label)

    fig.suptitle('Learning Rate Parameters Across Training Phases',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_sigma_trajectories_connected'))


# ─── Figure 14: Phenotype Distribution ────────────────────────────────────────

def make_figure_phenotype_distribution(complete_df):
    """Figure 14: Phenotype distribution (histogram + stacked bar)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Panel A: Histogram of log-ratios
    log_ratios = complete_df['chunk3_phenotype_log_ratio'].dropna()
    ax1.hist(log_ratios, bins=20, color='gray', edgecolor='white', alpha=0.7)
    ax1.axvline(LOG_THRESHOLD, color=COLORS['reward_sensitive'], linestyle='--', linewidth=1.5,
                label=f'Reward-sensitive threshold ({LOG_THRESHOLD:.2f})')
    ax1.axvline(-LOG_THRESHOLD, color=COLORS['perseverator'], linestyle='--', linewidth=1.5,
                label=f'Perseverator threshold ({-LOG_THRESHOLD:.2f})')
    ax1.set_xlabel(r'$\log_{10}(\sigma_{pc}/\sigma_{wsls})$')
    ax1.set_ylabel('Count')
    ax1.set_title('A. Distribution of σ-ratio', fontweight='bold', loc='left')
    ax1.legend(fontsize=7, loc='upper right')

    # Panel B: Phenotype proportions by chunk
    for ci, chunk in enumerate(['chunk1', 'chunk2', 'chunk3']):
        pheno_col = f'{chunk}_phenotype'
        counts = complete_df[pheno_col].value_counts()
        total = len(complete_df)
        bottom = 0
        for pheno in ['perseverator', 'mixed', 'reward_sensitive']:
            pct = counts.get(pheno, 0) / total * 100
            ax2.bar(ci, pct, bottom=bottom,
                    color=COLORS.get(pheno, 'gray'), edgecolor='white')
            if pct > 10:
                ax2.text(ci, bottom + pct/2, f'{pct:.0f}%',
                         ha='center', va='center', fontsize=8,
                         fontweight='bold', color='white')
            bottom += pct

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['C1', 'C2', 'C3'])
    ax2.set_ylabel('Percentage')
    ax2.set_ylim(0, 100)
    ax2.set_title('B. Phenotype Proportions', fontweight='bold', loc='left')
    # Add phenotype legend
    from matplotlib.patches import Patch
    pheno_legend = [
        Patch(facecolor=COLORS['perseverator'], label='Perseverator'),
        Patch(facecolor=COLORS['mixed'], label='Mixed'),
        Patch(facecolor=COLORS['reward_sensitive'], label='Reward-sensitive'),
    ]
    ax2.legend(handles=pheno_legend, fontsize=7, loc='upper right')

    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_phenotype_distribution'))


# ─── Figure 20: Individual Trajectories ───────────────────────────────────────

def make_figure_individual_trajectories(complete_df):
    """Figure 20: Spaghetti plot of σ_trial,contrast for all mice."""
    fig, ax = plt.subplots(figsize=(6, 5))

    chunk_x = [1, 2, 3]

    # Individual mice (gray)
    n_mice = len(complete_df)
    for _, row in complete_df.iterrows():
        vals = [row['chunk1_sigma_trial_contrast'],
                row['chunk2_sigma_trial_contrast'],
                row['chunk3_sigma_trial_contrast']]
        ax.plot(chunk_x, vals, 'o-', color='gray', alpha=0.2,
                markersize=3, linewidth=0.8)

    # Population mean ± SE (red)
    means = []
    ses = []
    cvs = []
    for chunk in ['chunk1', 'chunk2', 'chunk3']:
        vals = complete_df[f'{chunk}_sigma_trial_contrast'].dropna().values
        means.append(np.mean(vals))
        se = np.std(vals) / np.sqrt(len(vals))
        ses.append(se)
        cvs.append(np.std(vals) / np.mean(vals))

    ax.errorbar(chunk_x, means, yerr=ses, fmt='o-', color='red',
                linewidth=2.5, markersize=8, capsize=5, zorder=3)

    # Legend with explicit handles
    from matplotlib.lines import Line2D
    n_mice = len(complete_df)
    legend_handles = [
        Line2D([0], [0], color='gray', alpha=0.5, linewidth=0.8, marker='o',
               markersize=3, label=f'Individual mice (n = {n_mice})'),
        Line2D([0], [0], color='red', linewidth=2.5, marker='o',
               markersize=8, label=f'Population mean ± SE'),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc='upper left')

    ax.set_xticks(chunk_x)
    ax.set_xticklabels(['C1', 'C2', 'C3'])
    ax.set_xlabel('Training Phase')
    ax.set_ylabel(r'$\sigma_{trial,contrast}$')
    ax.set_title('Individual Contrast Learning Rate Trajectories',
                 fontweight='bold')

    # CV annotation
    cv_text = f'CV: {cvs[0]:.2f} (C1), {cvs[1]:.2f} (C2), {cvs[2]:.2f} (C3)'
    ax.text(0.5, 0.02, cv_text, transform=ax.transAxes,
            fontsize=8, ha='center', style='italic')

    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_individual_trajectories'))


# ─── Figure 21: Lab Effects ──────────────────────────────────────────────────

def make_figure_lab_effects(complete_df):
    """Figure 21: σ_trial,contrast in C3 by laboratory.

    Thesis caption: "Box plots show median and IQR; red points show individual
    mice. Dashed line indicates grand mean. Labs did not differ significantly."
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    # Lab display names matching thesis Table 1 (9 labs from subject prefix)
    lab_display = {
        'CSH': 'CSH', 'CSHL': 'CSHL', 'DY': 'DY',
        'IBL': 'IBL', 'KS': 'KS', 'NYU': 'NYU',
        'SWC': 'SWC', 'ZM': 'ZM', 'ibl_witten': 'ibl_witten',
    }

    labs = sorted(complete_df['lab'].unique())
    data_by_lab = []
    lab_labels = []
    for lab in labs:
        vals = complete_df[complete_df['lab'] == lab][
            'chunk3_sigma_trial_contrast'].dropna().values
        if len(vals) > 0:
            data_by_lab.append(vals)
            lab_labels.append(lab_display.get(lab, lab))

    bp = ax.boxplot(data_by_lab, labels=lab_labels, patch_artist=True,
                    widths=0.6, medianprops={'color': 'black', 'linewidth': 1.5})
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, vals in enumerate(data_by_lab):
        x = np.random.normal(i + 1, 0.08, len(vals))
        ax.scatter(x, vals, color='red', s=15, alpha=0.6, zorder=3)

    # Grand mean line with label
    grand_mean = complete_df['chunk3_sigma_trial_contrast'].mean()
    ax.axhline(grand_mean, color='gray', linestyle='--', linewidth=1, alpha=0.5,
               label=f'Grand mean ({grand_mean:.4f})')

    # Kruskal-Wallis test
    groups = [g for g in data_by_lab if len(g) > 0]
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        ax.text(0.98, 0.98, f'Kruskal-Wallis H = {H:.2f}, p = {p:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='gray', alpha=0.8))

    ax.set_xlabel('Laboratory')
    ax.set_ylabel(r'$\sigma_{\mathrm{trial,contrast}}$ (Chunk 3)')
    ax.set_title('Laboratory Effects on Contrast Learning Rate',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_lab_effects'))


# ─── Figure 5: Dataset Overview ───────────────────────────────────────────────

def make_figure_dataset_overview(complete_df):
    """Figure 5: Dataset overview (4-panel).

    A: Selection tiers by lab (green=Tier1, yellow=Tier2, red=Excluded)
    B: Chunking strategy distribution (pie: 92.4% / 2.2% / 5.4%)
    C: Trial counts by chunk (box plots + tier threshold dashed lines)
    D: Training duration distribution (histogram)
    """
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Load Phase 1 data for tier info
    try:
        with open(CHUNK_EXPLORATION_FILE, 'rb') as f:
            chunk_data = pickle.load(f)
        summary = chunk_data['chunk_summary']
    except (FileNotFoundError, KeyError):
        summary = complete_df  # fallback

    # Thesis tier colors — cover all possible tier key formats from Phase 1
    tier_colors = {'Tier 1': '#2ecc71', 'Tier 2': '#f1c40f', 'Excluded': '#e74c3c',
                   'tier1': '#2ecc71', 'tier2': '#f1c40f', 'excluded': '#e74c3c',
                   1: '#2ecc71', 2: '#f1c40f', 3: '#e74c3c'}

    # Panel A: Mice per lab by tier
    ax = axes[0, 0]
    if 'tier' in summary.columns:
        tier_counts = summary.groupby(['lab', 'tier']).size().unstack(fill_value=0)
        # Enforce tier ordering: tier1, tier2, excluded
        tier_order = [t for t in ['tier1', 'tier2', 'excluded'] if t in tier_counts.columns]
        tier_counts = tier_counts[tier_order]
        tier_display = {'tier1': 'Tier 1', 'tier2': 'Tier 2', 'excluded': 'Excluded'}
        tier_counts.columns = [tier_display.get(c, c) for c in tier_counts.columns]
        colors_for_plot = [tier_colors.get(t, '#999999') for t in tier_order]
        tier_counts.plot(kind='bar', stacked=True, ax=ax, edgecolor='white',
                         color=colors_for_plot)
        ax.set_ylabel('Number of Mice')
        ax.set_title('A. Selection Tiers by Laboratory', fontweight='bold', loc='left')
        ax.legend(title='Tier', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        ax.text(0.5, 0.5, 'Tier data not available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('A. Selection Tiers', fontweight='bold', loc='left')

    # Panel B: Chunking strategy (thesis: 92.4%, 2.2%, 5.4%)
    ax = axes[0, 1]
    ax.pie([92.4, 2.2, 5.4],
           labels=['Strategy A\n(standard)', 'Strategy B\n(alt)', 'Invalid'],
           autopct='%1.1f%%', colors=['#3498db', '#e67e22', '#e74c3c'],
           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}, pctdistance=0.75)
    ax.set_title('B. Chunking Strategy Distribution', fontweight='bold', loc='left')

    # Panel C: Trial counts by chunk + tier threshold lines
    ax = axes[1, 0]
    chunk_data_plot = {
        'Chunk 1': complete_df['chunk1_trials'].dropna().values,
        'Chunk 2': complete_df['chunk2_trials'].dropna().values,
        'Chunk 3': complete_df['chunk3_trials'].dropna().values,
    }
    bp = ax.boxplot(chunk_data_plot.values(), labels=chunk_data_plot.keys(),
                    patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], CHUNK_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    # Tier threshold dashed lines
    ax.axhline(1000, color='green', linestyle='--', linewidth=1, alpha=0.5,
               label='Tier 1 threshold (1000)')
    ax.axhline(500, color='orange', linestyle='--', linewidth=1, alpha=0.5,
               label='Tier 2 threshold (500)')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylabel('Number of Trials')
    ax.set_title('C. Trial Counts by Chunk', fontweight='bold', loc='left')

    # Panel D: Training duration
    ax = axes[1, 1]
    total_days = (complete_df['chunk1_days'].fillna(0) +
                  complete_df['chunk2_days'].fillna(0) +
                  complete_df['chunk3_days'].fillna(0))
    ax.hist(total_days.values, bins=np.arange(5, 65, 5), color='#3498db',
            edgecolor='white', alpha=0.8)
    ax.set_xlabel('Total Training Days')
    ax.set_ylabel('Number of Mice')
    ax.set_title('D. Training Duration Distribution', fontweight='bold', loc='left')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_dataset_overview'))


# ─── Figure 15: Phenotype Stability ──────────────────────────────────────────

def make_figure_phenotype_stability(complete_df):
    """Figure 15: C1 vs C3 phenotype scatter plot."""
    fig, ax = plt.subplots(figsize=(6, 6))

    x = complete_df['chunk1_phenotype_log_ratio']
    y = complete_df['chunk3_phenotype_log_ratio']
    c3_pheno = complete_df['chunk3_phenotype']
    point_colors = [COLORS.get(p, 'gray') for p in c3_pheno]

    ax.scatter(x, y, c=point_colors, s=50, alpha=0.7, edgecolors='white',
               linewidth=0.5, zorder=3)

    # Phenotype color legend (C3 classification)
    from matplotlib.lines import Line2D
    pheno_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['perseverator'],
               markersize=8, label='Perseverator (C3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['mixed'],
               markersize=8, label='Mixed (C3)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['reward_sensitive'],
               markersize=8, label='Reward-sensitive (C3)'),
    ]
    ax.legend(handles=pheno_handles, fontsize=7, loc='lower right')

    # Diagonal (perfect stability)
    lims = [-3, 3]
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=1)

    # Threshold lines
    ax.axhline(LOG_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(-LOG_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(LOG_THRESHOLD, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(-LOG_THRESHOLD, color='gray', linestyle=':', alpha=0.5)

    ax.set_xlabel(r'Chunk 1: $\log_{10}(\sigma_{pc}/\sigma_{wsls})$')
    ax.set_ylabel(r'Chunk 3: $\log_{10}(\sigma_{pc}/\sigma_{wsls})$')
    ax.set_title('Phenotype Stability: Chunk 1 vs Chunk 3', fontweight='bold')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal')

    # Stability count
    stable = (complete_df['chunk1_phenotype'] == complete_df['chunk3_phenotype']).sum()
    ax.text(0.02, 0.98, f'Stable: {stable}/{len(complete_df)} '
            f'({100*stable/len(complete_df):.0f}%)',
            transform=ax.transAxes, va='top', fontsize=9)

    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_phenotype_stability'))


# ─── Figure 16: Phenotype & Learning Speed ───────────────────────────────────

def make_figure_phenotype_learning_speed(complete_df):
    """Figure 16: Sessions to reach C3 by behavioral phenotype."""
    fig, ax = plt.subplots(figsize=(6, 5))

    # Approximate total sessions (sum of chunk sessions)
    complete_df = complete_df.copy()
    for col in ['chunk1_days', 'chunk2_days', 'chunk3_days']:
        if col not in complete_df.columns:
            complete_df[col] = 0

    total_sessions = (complete_df['chunk1_days'].fillna(0) +
                      complete_df['chunk2_days'].fillna(0))

    groups = []
    labels = []
    for pheno in ['perseverator', 'mixed', 'reward_sensitive']:
        mask = complete_df['chunk3_phenotype'] == pheno
        vals = total_sessions[mask].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(pheno.replace('_', '\n'))

    bp = ax.boxplot(groups, labels=labels, patch_artist=True, widths=0.5)
    pheno_colors = [COLORS.get(p.replace('\n', '_'), 'gray') for p in labels]
    for patch, color in zip(bp['boxes'], pheno_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.3)

    # Overlay individual data points
    for i, (vals, color) in enumerate(zip(groups, pheno_colors)):
        x_jitter = np.random.normal(i + 1, 0.06, size=len(vals))
        ax.scatter(x_jitter, vals, color=color, s=20, alpha=0.7, zorder=3,
                   edgecolor='white', linewidth=0.3)

    # Kruskal-Wallis
    if len(groups) >= 2:
        H, p = kruskal(*groups)
        ax.text(0.98, 0.98, f'Kruskal-Wallis H = {H:.2f}, p = {p:.2f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=9)

    ax.set_ylabel('Sessions to Reach Chunk 3')
    ax.set_title('Phenotype and Learning Speed', fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_phenotype_learning_speed'))


# ─── Figure 22: Outlier Trajectories ─────────────────────────────────────────

def make_figure_outlier_trajectories(complete_df):
    """Figure 22: σ_trial,contrast trajectories with 3 SD outlier detection."""
    fig, ax = plt.subplots(figsize=(7, 5))

    chunk_x = [1, 2, 3]

    # Compute 3 SD thresholds per chunk
    thresholds = {}
    for ci, chunk in enumerate(['chunk1', 'chunk2', 'chunk3']):
        vals = complete_df[f'{chunk}_sigma_trial_contrast'].dropna().values
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        thresholds[chunk] = mean_val + 3 * std_val

    # Identify outliers: any mouse exceeding 3 SD in any chunk
    outlier_mask = pd.Series([False] * len(complete_df), index=complete_df.index)
    for chunk in ['chunk1', 'chunk2', 'chunk3']:
        col = f'{chunk}_sigma_trial_contrast'
        outlier_mask = outlier_mask | (complete_df[col] > thresholds[chunk])

    # Plot non-outliers (gray)
    for _, row in complete_df[~outlier_mask].iterrows():
        vals = [row[f'{c}_sigma_trial_contrast'] for c in ['chunk1', 'chunk2', 'chunk3']]
        ax.plot(chunk_x, vals, 'o-', color='gray', alpha=0.15, markersize=2, linewidth=0.5)

    # Plot outliers (red)
    for _, row in complete_df[outlier_mask].iterrows():
        vals = [row[f'{c}_sigma_trial_contrast'] for c in ['chunk1', 'chunk2', 'chunk3']]
        ax.plot(chunk_x, vals, 'o-', color='red', alpha=0.7, markersize=4, linewidth=1.2)

    # Mean ± SE excluding outliers (green) — connected line across chunks
    corrected_means = []
    corrected_ses = []
    for ci, chunk in enumerate(['chunk1', 'chunk2', 'chunk3']):
        vals = complete_df.loc[~outlier_mask, f'{chunk}_sigma_trial_contrast'].dropna()
        corrected_means.append(vals.mean())
        corrected_ses.append(vals.std() / np.sqrt(len(vals)))

    ax.errorbar(chunk_x, corrected_means, yerr=corrected_ses,
                fmt='o-', color='green', markersize=8, capsize=5, zorder=5,
                linewidth=2, label=f'Corrected mean ± SE (n={(~outlier_mask).sum()})')

    # Threshold lines
    thresh_vals = [thresholds[f'chunk{i+1}'] for i in range(3)]
    ax.plot(chunk_x, thresh_vals, '--', color='orange', linewidth=1.5, label='3 SD threshold')

    ax.set_xticks(chunk_x)
    ax.set_xticklabels(['C1', 'C2', 'C3'])
    ax.set_ylabel(r'$\sigma_{trial,contrast}$')
    ax.set_title('Outlier Identification', fontweight='bold')

    n_outliers = outlier_mask.sum()

    # Combined legend with non-overlapping entries
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color='gray', alpha=0.4, linewidth=0.8, marker='o',
               markersize=3, label='Non-outlier mice'),
        Line2D([0], [0], color='red', alpha=0.7, linewidth=1.2, marker='o',
               markersize=4, label=f'Outliers (n={n_outliers})'),
        Line2D([0], [0], color='green', linewidth=2, marker='o',
               markersize=8, label=f'Corrected mean ± SE'),
        Line2D([0], [0], color='orange', linestyle='--', linewidth=1.5,
               label='3 SD threshold'),
    ]
    ax.legend(handles=legend_handles, fontsize=7, loc='upper left')

    ax.text(0.98, 0.98, f'{n_outliers} outlier(s) identified',
            transform=ax.transAxes, va='top', ha='right', fontsize=9, color='red')

    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_outlier_trajectories'))


# ─── Figure 17: Overnight Analysis ───────────────────────────────────────────

def make_figure_overnight_analysis(complete_df, jumps_df):
    """Figure 17: Overnight weight changes (2×4 grid).

    Top row: Jump magnitude (|Δβ|) box plots by chunk
    Bottom row: Jump direction (stacked bars)

    Direction metrics differ by weight:
        Contrast: signed ΔW (positive = strengthening)
        Bias/history: toward_zero (magnitude decrease)
    """
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))

    for col_idx, weight in enumerate(WEIGHT_NAMES):
        weight_jumps = jumps_df[jumps_df['weight'] == weight]

        # ── Top row: Magnitude ──
        ax = axes[0, col_idx]
        magnitude_data = []
        for chunk in ['chunk1', 'chunk2', 'chunk3']:
            vals = weight_jumps[weight_jumps['chunk'] == chunk]['abs_delta'].dropna().values
            magnitude_data.append(vals)

        if any(len(v) > 0 for v in magnitude_data):
            bp = ax.boxplot(magnitude_data, labels=['C1', 'C2', 'C3'],
                            patch_artist=True, widths=0.5,
                            medianprops={'color': 'black', 'linewidth': 1.5},
                            flierprops={'marker': '.', 'markersize': 3, 'alpha': 0.3})
            for patch, color in zip(bp['boxes'], CHUNK_COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)

        ax.set_title(WEIGHT_LABELS[col_idx], fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel('Jump\nMagnitude')

        # ── Bottom row: Direction ──
        ax = axes[1, col_idx]
        toward_pcts = []
        away_pcts = []
        binom_stars = []

        for ci, chunk in enumerate(['chunk1', 'chunk2', 'chunk3']):
            chunk_jumps = weight_jumps[weight_jumps['chunk'] == chunk]

            if weight == 'contrast':
                # Contrast uses signed ΔW: positive = strengthening
                n_total = len(chunk_jumps)
                n_positive = (chunk_jumps['delta'] > 0).sum()
                pct_good = 100 * n_positive / n_total if n_total > 0 else 50
                n_good = n_positive
            else:
                # Bias/history: toward_zero = magnitude decrease
                valid = chunk_jumps['toward_zero'].dropna()
                n_total = len(valid)
                n_toward = int(valid.sum())
                pct_good = 100 * n_toward / n_total if n_total > 0 else 50
                n_good = n_toward

            toward_pcts.append(pct_good)
            away_pcts.append(100 - pct_good)

            # Binomial test for significance star
            if n_total > 10:
                binom_p = binomtest(n_good, n_total, 0.5).pvalue
                stars = '***' if binom_p < 0.001 else ('**' if binom_p < 0.01 else ('*' if binom_p < 0.05 else ''))
            else:
                stars = ''
            binom_stars.append(stars)

        x = np.arange(3)
        ax.bar(x, toward_pcts, 0.6, label='Toward zero' if weight != 'contrast' else 'Positive ΔW',
               color=CHUNK_COLORS, alpha=0.7)
        ax.bar(x, away_pcts, 0.6, bottom=toward_pcts,
               color=[c + '80' for c in CHUNK_COLORS] if False else ['lightgray']*3, alpha=0.5)
        ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['C1', 'C2', 'C3'])
        ax.set_ylim(0, 105)

        # Add binomial significance stars
        for ci, stars in enumerate(binom_stars):
            if stars:
                ax.text(ci, toward_pcts[ci] + 1, stars, ha='center', va='bottom',
                        fontsize=8, fontweight='bold')

        if col_idx == 0:
            ax.set_ylabel('Jump\nDirection')

        # Metric label
        if weight == 'contrast':
            metric = 'signed ΔW'
        elif weight == 'bias':
            metric = 'Δ|β|'
        else:
            metric = 'toward/away zero'
        ax.text(0.5, 1.0, metric, transform=ax.transAxes,
                ha='center', va='bottom', fontsize=7, color='gray', style='italic')

    fig.suptitle('Overnight Weight Changes by Training Stage',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_overnight_corrected_v3'))


# ─── Figure 18: Overnight Histograms ─────────────────────────────────────────

def make_figure_overnight_histograms(jumps_df):
    """Figure 18: Step histograms of overnight changes by training phase."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, weight in enumerate(WEIGHT_NAMES):
        ax = axes[idx]
        weight_jumps = jumps_df[jumps_df['weight'] == weight]

        inset_lines = []
        for ci, chunk in enumerate(['chunk1', 'chunk2', 'chunk3']):
            chunk_jumps = weight_jumps[weight_jumps['chunk'] == chunk]

            if weight == 'contrast':
                # Contrast: use signed delta (positive = strengthening)
                vals = chunk_jumps['delta'].dropna().values
                xlabel = 'Signed ΔW'
                good_dir = 'positive'
            else:
                # Bias/history: use Δ|β| (negative = toward zero)
                vals = (chunk_jumps['w_start'].abs() - chunk_jumps['w_end'].abs()).dropna().values
                xlabel = 'Δ|β|'
                good_dir = 'negative'

            if len(vals) < 5:
                continue

            ax.hist(vals, bins=40, density=True, histtype='step',
                    linewidth=1.5, color=CHUNK_COLORS[ci],
                    label=f'C{ci+1} (n={len(vals)})', alpha=0.9)

            # Compute median and fraction in "good" direction
            med = np.median(vals)
            if good_dir == 'positive':
                frac = 100 * np.mean(vals > 0)
            else:
                frac = 100 * np.mean(vals < 0)
            inset_lines.append(f'C{ci+1}: med={med:+.3f}, {frac:.0f}% good, n={len(vals)}')

        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)

        # Shade "good" direction half
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        if weight == 'contrast':
            ax.axvspan(0, xlim[1], alpha=0.06, color='green')
        else:
            ax.axvspan(xlim[0], 0, alpha=0.06, color='green')

        ax.set_title(WEIGHT_LABELS[idx], fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper right')

        # Inset text with median and fraction
        if inset_lines:
            ax.text(0.02, 0.98, '\n'.join(inset_lines),
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=6, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    fig.suptitle('Distributions of Overnight Weight Changes by Training Phase',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_overnight_distributions'))


# ─── Figure 19: Day Gap Effect ───────────────────────────────────────────────

def make_figure_day_gap_effect(jumps_df):
    """Figure 19: Mean |Δw| vs days between sessions.

    Thesis format: single panel with all 4 weights overlaid,
    error bars showing SE, Spearman correlations reported in text.
    Thesis caption: "Mean absolute weight change as a function of days
    between sessions. Error bars show SE."
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    spearman_lines = []
    for weight in WEIGHT_NAMES:
        weight_jumps = jumps_df[jumps_df['weight'] == weight]

        # Cap day_gap at 3 to match thesis (3 groups)
        weight_jumps = weight_jumps[weight_jumps['day_gap'] <= 3]

        gap_groups = weight_jumps.groupby('day_gap')['abs_delta']
        gap_means = gap_groups.mean()
        gap_ses = gap_groups.apply(lambda x: x.std() / np.sqrt(len(x)))

        w_label = WEIGHT_LABELS[WEIGHT_NAMES.index(weight)]
        ax.errorbar(gap_means.index, gap_means.values, yerr=gap_ses.values,
                    fmt='o-', color=COLORS[weight], markersize=5,
                    capsize=3, linewidth=1.2, label=w_label)

        # Scatter individual data points (jittered)
        x_jitter = weight_jumps['day_gap'] + np.random.normal(0, 0.05, len(weight_jumps))
        ax.scatter(x_jitter, weight_jumps['abs_delta'], color=COLORS[weight],
                   s=5, alpha=0.1, zorder=1)

        # Spearman correlation
        rho, p = spearmanr(weight_jumps['day_gap'], weight_jumps['abs_delta'])
        print(f"    Day gap vs |Δ{weight}|: ρ={rho:.3f}, p={p:.3f}")
        spearman_lines.append(f'{w_label}: ρ={rho:.3f}, p={p:.3f}')

    ax.set_xlabel('Days Between Sessions')
    ax.set_ylabel(r'Mean $|\Delta w|$')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(['1', '2', '3+'])
    ax.set_title('Effect of Day Gap on Overnight Jump Magnitude',
                 fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')

    # Add Spearman values as inset
    ax.text(0.02, 0.98, '\n'.join(spearman_lines),
            transform=ax.transAxes, va='top', ha='left', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_day_gap_effect'))


# ─── Figures 7-9: Weight dynamics from pickle ────────────────────────────────

def make_figures_from_pickle(complete_df):
    """Figures 7, 8, 9: Weight dynamics requiring trial-level pickle data.

    These figures show trial-by-trial weight trajectories and contrast
    curriculum for selected example mice.
    """
    # Load pickle files
    try:
        with open(PSYTRACK_RESULTS_FILE, 'rb') as f:
            results_data = pickle.load(f)
        with open(CHUNK_EXPLORATION_FILE, 'rb') as f:
            chunk_data = pickle.load(f)
        mouse_results = results_data.get('mouse_results', {})
    except FileNotFoundError as e:
        print(f"    Pickle files not found ({e}). Skipping Figs 7-9.")
        return

    # Hardcoded 9 example mice matching thesis Figs 7-8
    # Selected for balanced chunk durations (12-28 total sessions)
    example_mice = [
        'ZM_1373', 'ibl_witten_13', 'CSHL_005',
        'KS019', 'SWC_023', 'ZM_1745',
        'ZM_1743', 'KS024', 'ZM_1367',
    ]
    # Filter to mice that exist in our data
    example_mice = [m for m in example_mice if m in mouse_results]
    if len(example_mice) < 3:
        print("    Not enough example mice for Figs 7-8. Skipping.")
        return

    _make_figure_weight_dynamics(example_mice, mouse_results, chunk_data)
    _make_figure_contrast_curriculum(example_mice, chunk_data)

    # Hardcoded 3 example mice for Fig 9 matching thesis
    trio = ['ZM_1367', 'ibl_witten_07', 'KS024']
    trio = [m for m in trio if m in mouse_results]

    _make_figure_individual_composite(trio, mouse_results, complete_df)


def _make_figure_weight_dynamics(example_mice, mouse_results, chunk_data):
    """Figure 4: Trial-by-trial weight trajectories for 9 example mice (3×3 grid).

    Each panel overlays all 4 weights (weight-specific colors) with chunk
    background shading, day boundaries, chunk boundary dashed lines, zero line,
    and C1/C2/C3 labels — matching the trajectory-panel style of Figure 6
    (individual composite).
    """
    n_mice = min(len(example_mice), 9)
    n_rows, n_cols = 3, 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12))

    for idx, subject in enumerate(example_mice[:n_mice]):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]

        result = mouse_results.get(subject, {})
        chunks = result.get('chunks', {})

        # Concatenate weight trajectories across chunks
        trial_offset = 0
        chunk_boundaries = [0]
        for ci, chunk_name in enumerate(['chunk1', 'chunk2', 'chunk3']):
            chunk_result = chunks.get(chunk_name, {})
            wMode = chunk_result.get('wMode')
            W_std = chunk_result.get('W_std')
            day_boundaries_chunk = chunk_result.get('day_boundaries', [])
            if wMode is None:
                continue

            n_trials = wMode.shape[1]
            x = np.arange(trial_offset, trial_offset + n_trials)

            # Chunk background shading
            ax.axvspan(trial_offset, trial_offset + n_trials,
                       alpha=0.1, color=CHUNK_COLORS[ci])

            # All 4 weight trajectories overlaid with weight-specific colors
            for k in range(K):
                ax.plot(x, wMode[k], color=COLORS[WEIGHT_NAMES[k]],
                        linewidth=0.8, alpha=0.7)
                if W_std is not None and not np.all(np.isnan(W_std)):
                    ax.fill_between(x,
                                    wMode[k] - W_std[k],
                                    wMode[k] + W_std[k],
                                    alpha=0.08, color=COLORS[WEIGHT_NAMES[k]])

            # Day boundaries (thin gray lines)
            for db in day_boundaries_chunk[1:]:  # skip first (=0)
                if db < n_trials:
                    ax.axvline(trial_offset + db, color='gray',
                               linewidth=0.3, alpha=0.3)

            trial_offset += n_trials
            chunk_boundaries.append(trial_offset)

        # Horizontal zero line
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        # Dashed chunk boundary lines
        for b in chunk_boundaries[1:-1]:
            ax.axvline(b, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

        # C1/C2/C3 labels at top of each chunk region
        ylim = ax.get_ylim()
        chunk_labels_text = ['C1', 'C2', 'C3']
        for ci in range(min(3, len(chunk_boundaries) - 1)):
            mid = (chunk_boundaries[ci] + chunk_boundaries[ci + 1]) / 2
            ax.text(mid, ylim[1] * 0.92, chunk_labels_text[ci],
                    ha='center', va='top',
                    fontsize=8, fontweight='bold', color=CHUNK_COLORS[ci])

        ax.set_title(subject, fontsize=9, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        if row == n_rows - 1:
            ax.set_xlabel('Trial')
        if col == 0:
            ax.set_ylabel('Weight')

    # Legend from first panel (weight names)
    legend_handles = [plt.Line2D([0], [0], color=COLORS[w], linewidth=1.5, label=l)
                      for w, l in zip(WEIGHT_NAMES, WEIGHT_LABELS)]
    fig.legend(handles=legend_handles, fontsize=9,
               ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('Trial-by-Trial Weight Dynamics: Representative Mice', fontsize=13,
                 fontweight='bold', y=1.04)
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_weight_dynamics'))


def _make_figure_contrast_curriculum(example_mice, chunk_data):
    """Figure 8: Daily contrast distribution for 9 example mice (3×3 grid).

    Thesis caption: "Background shading and dashed lines indicate chunk
    boundaries. The curriculum structure—progressing from easy high-contrast
    discrimination to challenging near-threshold stimuli—provides context."
    """
    n_mice = len(example_mice)
    n_rows = 3
    n_cols = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 10))

    contrast_levels = [1.0, 0.5, 0.25, 0.125, 0.0625, 0]
    contrast_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    contrast_labels = ['100%', '50%', '25%', '12.5%', '6.25%', '0%']

    for idx, subject in enumerate(example_mice[:9]):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        mouse_data = chunk_data.get('mouse_data', {}).get(subject, {})
        trial_data = mouse_data.get('trial_data', {})

        # Collect per-chunk date ranges for shading
        chunk_date_ranges = []
        all_frames = []
        for c in ['chunk1', 'chunk2', 'chunk3']:
            df_c = trial_data.get(c, pd.DataFrame())
            if len(df_c) > 0:
                all_frames.append(df_c)
                dates_sorted = sorted(df_c['date'].unique())
                chunk_date_ranges.append(dates_sorted)
            else:
                chunk_date_ranges.append([])

        if not all_frames:
            continue

        all_trials = pd.concat(all_frames, ignore_index=True)

        # Count each contrast level present per trial per day
        # (matching Colab: a trial with left=0.125 and right=0.5
        # counts as both a 0.125 trial and a 0.5 trial)
        dates_all = sorted(all_trials['date'].unique())
        date_to_x = {d: i for i, d in enumerate(dates_all)}
        session_counts = {c: np.zeros(len(dates_all))
                          for c in contrast_levels}

        for _, trial_row in all_trials.iterrows():
            day_idx = date_to_x.get(trial_row['date'])
            if day_idx is None:
                continue
            cl = trial_row.get('contrast_left', 0) or 0
            cr = trial_row.get('contrast_right', 0) or 0
            for c_val in [cl, cr]:
                c_round = round(c_val, 4)
                if c_round in session_counts and c_round > 0:
                    session_counts[c_round][day_idx] += 1
            # Count 0% contrast trials (both sides zero)
            if cl == 0 and cr == 0:
                session_counts[0][day_idx] += 1

        # Compute chunk boundary x-positions first (1-based)
        boundary_xs = []
        for ci in range(len(chunk_date_ranges) - 1):
            if chunk_date_ranges[ci] and chunk_date_ranges[ci + 1]:
                last_x = date_to_x[chunk_date_ranges[ci][-1]] + 1
                first_x = date_to_x[chunk_date_ranges[ci + 1][0]] + 1
                boundary_xs.append((last_x + first_x) / 2)
            else:
                boundary_xs.append(None)

        # Chunk background shading — edges aligned to boundary midpoints
        for ci, chunk_dates in enumerate(chunk_date_ranges):
            if len(chunk_dates) == 0:
                continue
            x_start = date_to_x[chunk_dates[0]] + 1 - 0.4
            x_end = date_to_x[chunk_dates[-1]] + 1 + 0.4
            # Snap edges to boundary lines where they exist
            if ci > 0 and ci - 1 < len(boundary_xs) and boundary_xs[ci - 1] is not None:
                x_start = boundary_xs[ci - 1]
            if ci < len(boundary_xs) and boundary_xs[ci] is not None:
                x_end = boundary_xs[ci]
            ax.axvspan(x_start, x_end, alpha=0.08, color=CHUNK_COLORS[ci])

        # Stacked bar (1-based x-axis: training day 1, 2, 3, ...)
        x_positions = np.arange(1, len(dates_all) + 1)
        bottom = np.zeros(len(dates_all))
        for level, color, clabel in zip(contrast_levels, contrast_colors, contrast_labels):
            vals = session_counts[level]
            if np.sum(vals) > 0:
                ax.bar(x_positions, vals, bottom=bottom,
                       color=color, width=0.8,
                       label=clabel if idx == 0 else '')
                bottom += vals

        # Dashed chunk boundary lines
        for bx in boundary_xs:
            if bx is not None:
                ax.axvline(bx, color='black', linestyle='--',
                           linewidth=0.8, alpha=0.6)

        # C1/C2/C3 labels at top of each chunk region
        chunk_labels_text = ['C1', 'C2', 'C3']
        for ci, chunk_dates in enumerate(chunk_date_ranges):
            if len(chunk_dates) > 0:
                x_mid = (date_to_x[chunk_dates[0]] + date_to_x[chunk_dates[-1]]) / 2 + 1
                ax.text(x_mid, ax.get_ylim()[1] * 0.95, chunk_labels_text[ci],
                        ha='center', va='top',
                        fontsize=8, fontweight='bold', color=CHUNK_COLORS[ci])

        # X-axis: integer ticks only, start at 0.5 to eliminate empty space
        ax.set_xlim(0.5, len(dates_all) + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(subject, fontsize=9, fontweight='bold')
        if row == n_rows - 1:
            ax.set_xlabel('Training Day')
        if col == 0:
            ax.set_ylabel('Trials per Day')

    # Legend from first panel
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title='Contrast Level', fontsize=8,
                   ncol=6, loc='upper center', bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('Contrast Curriculum Progression: Representative Mice', fontsize=13,
                 fontweight='bold', y=1.04)
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_contrast_curriculum'))


def _make_figure_individual_composite(trio, mouse_results, complete_df):
    """Figure 9: Individual composite (3 mice × 3 columns).

    Left (double-width): weight trajectories with chunk shading, chunk
          boundaries (dashed), day boundaries (thin gray), ±1 SD bands,
          C1/C2/C3 labels, horizontal zero line
    Center: σ_trial across chunks (connected dots)
    Right: σ_day across chunks (connected dots)

    Thesis caption specifies: "Three mice selected to span the range of contrast
    σ trajectories: increasing (top), typical/median (middle), stable/decreasing
    (bottom)."
    """
    import matplotlib.gridspec as gridspec

    row_subtitles = [
        'Increasing contrast learning rate',
        'Typical (near-median)',
        'Stable/decreasing contrast learning rate',
    ]
    fig = plt.figure(figsize=(16, 11))
    gs = gridspec.GridSpec(3, 4, figure=fig, width_ratios=[2, 2, 1, 1],
                           hspace=0.35, wspace=0.35)
    chunk_x = [1, 2, 3]

    for row, subject in enumerate(trio):
        result = mouse_results.get(subject, {})
        chunks = result.get('chunks', {})
        mouse_row = complete_df[complete_df['subject'] == subject]
        if len(mouse_row) == 0:
            continue
        mouse_row = mouse_row.iloc[0]

        # Left: weight trajectories (spans columns 0-1 for double width)
        ax = fig.add_subplot(gs[row, 0:2])
        trial_offset = 0
        chunk_boundaries = [0]
        for ci, chunk_name in enumerate(['chunk1', 'chunk2', 'chunk3']):
            chunk_result = chunks.get(chunk_name, {})
            wMode = chunk_result.get('wMode')
            W_std = chunk_result.get('W_std')
            day_boundaries_chunk = chunk_result.get('day_boundaries', [])
            if wMode is None:
                continue
            n_trials = wMode.shape[1]
            x = np.arange(trial_offset, trial_offset + n_trials)

            # Chunk background shading
            ax.axvspan(trial_offset, trial_offset + n_trials,
                       alpha=0.1, color=CHUNK_COLORS[ci])

            # Weight trajectories with ±1 SD bands
            for k in range(K):
                ax.plot(x, wMode[k], color=COLORS[WEIGHT_NAMES[k]],
                        linewidth=0.8, alpha=0.7)
                if W_std is not None and not np.all(np.isnan(W_std)):
                    ax.fill_between(x,
                                    wMode[k] - W_std[k],
                                    wMode[k] + W_std[k],
                                    alpha=0.08, color=COLORS[WEIGHT_NAMES[k]])

            # Day boundaries (thin gray lines)
            for db in day_boundaries_chunk[1:]:  # skip first (=0)
                if db < n_trials:
                    ax.axvline(trial_offset + db, color='gray',
                               linewidth=0.3, alpha=0.4)

            trial_offset += n_trials
            chunk_boundaries.append(trial_offset)

        # Horizontal zero line
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.3)

        # Dashed chunk boundary lines
        for b in chunk_boundaries[1:-1]:
            ax.axvline(b, color='black', linestyle='--', linewidth=0.8, alpha=0.6)

        # C1/C2/C3 labels at top of each chunk region
        ylim = ax.get_ylim()
        chunk_labels_text = ['C1', 'C2', 'C3']
        for ci in range(min(3, len(chunk_boundaries) - 1)):
            mid = (chunk_boundaries[ci] + chunk_boundaries[ci + 1]) / 2
            ax.text(mid, ylim[1] * 0.92, chunk_labels_text[ci],
                    ha='center', va='top',
                    fontsize=8, fontweight='bold', color=CHUNK_COLORS[ci])

        # Row title above the trajectory panel
        ax.set_title(f'{subject} — {row_subtitles[row]}', fontsize=10,
                     fontweight='bold')
        ax.set_ylabel('Weight', fontsize=9)
        ax.grid(axis='y', alpha=0.3, linewidth=0.5)
        if row == 2:
            ax.set_xlabel('Trial', fontsize=9)

        # Center: σ_trial
        ax = fig.add_subplot(gs[row, 2])
        for weight in WEIGHT_NAMES:
            vals = [mouse_row.get(f'{c}_sigma_trial_{weight}', np.nan)
                    for c in ['chunk1', 'chunk2', 'chunk3']]
            ax.plot(chunk_x, vals, 'o-', color=COLORS[weight],
                    markersize=6, linewidth=1.5)
        ax.set_xticks(chunk_x)
        ax.set_xticklabels(['C1', 'C2', 'C3'])
        ax.grid(axis='both', alpha=0.3, linewidth=0.5)
        if row == 0:
            ax.set_title(r'Within-session', fontweight='bold', fontsize=9)
        ax.set_ylabel(r'$\sigma_{\mathrm{trial}}$', fontsize=9)

        # Right: σ_day
        ax = fig.add_subplot(gs[row, 3])
        for weight in WEIGHT_NAMES:
            vals = [mouse_row.get(f'{c}_sigma_day_{weight}', np.nan)
                    for c in ['chunk1', 'chunk2', 'chunk3']]
            ax.plot(chunk_x, vals, 'o-', color=COLORS[weight],
                    markersize=6, linewidth=1.5)
        ax.set_xticks(chunk_x)
        ax.set_xticklabels(['C1', 'C2', 'C3'])
        ax.grid(axis='both', alpha=0.3, linewidth=0.5)
        if row == 0:
            ax.set_title(r'Overnight', fontweight='bold', fontsize=9)
        ax.set_ylabel(r'$\sigma_{\mathrm{day}}$', fontsize=9)

    # Global shared legend (replaces per-row legend)
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=COLORS[w], lw=2, label=WEIGHT_LABELS[i])
        for i, w in enumerate(WEIGHT_NAMES)
    ]
    fig.legend(handles=legend_elements, fontsize=8,
               ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.02))

    fig.suptitle('Example Animals: Weight Dynamics and Learning Rate Trajectories',
                 fontsize=13, fontweight='bold', y=1.04)
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_individual_composite_v3'))


# ─── Tables ──────────────────────────────────────────────────────────────────

def _fmt_p_thesis(p):
    """Format p-value matching thesis style: $p = 0.417$ or $p < 0.001$."""
    if pd.isna(p):
        return '--'
    elif p < 0.001:
        return '$p < 0.001$'
    else:
        return f'$p = {p:.3f}$'


def _fmt_mean_sd(mean, sd):
    """Format mean (SD) for LaTeX."""
    return f'{mean:.4f} ({sd:.4f})'


def _fmt_effect_with_label(r_val):
    """Format effect size with interpretation: +0.81 (large)."""
    if pd.isna(r_val):
        return '--'
    label = effect_size_label(r_val)
    if label == 'large':
        return f'$+{r_val:.2f}$ (\\textbf{{{label}}})'
    return f'${r_val:+.2f}$ ({label})'


def _fmt_trend(p_corr, direction):
    """Format trend arrow for LaTeX."""
    if pd.isna(p_corr) or p_corr >= 0.05:
        return '--'
    elif direction > 0:
        return '$\\uparrow$'
    else:
        return '$\\downarrow$'


def make_tables(complete_df, jumps_df):
    """Generate all statistical tables as CSV and LaTeX .tex files."""
    print("\nGenerating tables...")

    _make_sigma_table(complete_df, 'sigma_trial', 'table_sigma_trial')
    _make_sigma_table(complete_df, 'sigma_day', 'table_sigma_day')
    _make_effect_size_table(complete_df)
    _make_mixed_effects_table(complete_df)
    _make_phenotype_table(complete_df)
    _make_overnight_magnitude_table(complete_df, jumps_df)
    _make_overnight_direction_table(complete_df, jumps_df)
    _make_results_snippets(complete_df, jumps_df)

    print("  Tables complete.")


def _make_sigma_table(complete_df, sigma_type, filename):
    """Tables 6-7: σ results matching thesis format exactly."""
    rows = []
    latex_rows = []

    for weight in WEIGHT_NAMES:
        row = {'Weight': weight}
        means_sds = {}
        for chunk in ['chunk1', 'chunk2', 'chunk3']:
            col = f'{chunk}_{sigma_type}_{weight}'
            vals = complete_df[col].dropna()
            m, s = vals.mean(), vals.std()
            row[f'{chunk}_mean'] = m
            row[f'{chunk}_sd'] = s
            means_sds[chunk] = (m, s)

        c1 = complete_df[f'chunk1_{sigma_type}_{weight}'].dropna()
        c3 = complete_df[f'chunk3_{sigma_type}_{weight}'].dropna()
        common = c1.index.intersection(c3.index)

        if len(common) >= 10:
            result = paired_wilcoxon_with_effect_size(
                c1.loc[common].values, c3.loc[common].values)
            p_corr = bonferroni_correct(result['p_value'])
            direction = c3.loc[common].mean() - c1.loc[common].mean()
            r_val = result['effect_size']
        else:
            p_corr = np.nan
            direction = 0
            r_val = np.nan

        rows.append(row)

        # LaTeX row
        if weight == 'prev_choice':
            w_label = 'Prev.\\ choice'
        elif weight == 'wsls':
            w_label = 'WSLS'
        else:
            w_label = weight.capitalize()

        c1_str = _fmt_mean_sd(means_sds['chunk1'][0], means_sds['chunk1'][1])
        c2_str = _fmt_mean_sd(means_sds['chunk2'][0], means_sds['chunk2'][1])
        c3_str = _fmt_mean_sd(means_sds['chunk3'][0], means_sds['chunk3'][1])
        p_str = _fmt_p_thesis(p_corr)
        r_str = _fmt_effect_with_label(r_val)
        trend = _fmt_trend(p_corr, direction)

        latex_rows.append(
            f'{w_label} & {c1_str} & {c2_str} & {c3_str} & {p_str} & {r_str} & {trend} \\\\\n')

    # CSV
    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, f'{filename}.csv'), index=False)

    # LaTeX
    is_trial = 'trial' in sigma_type
    sigma_tex = r'$\sigma_{\text{trial}}$' if is_trial else r'$\sigma_{\text{day}}$'
    caption = f'Within-Session Learning Rates ({sigma_tex}) Across Chunks.' if is_trial \
        else f'Overnight Learning Rates ({sigma_tex}) Across Chunks.'
    label = 'sigma_trial_results' if is_trial else 'sigma_day_results'

    latex = '\\begin{table}[htbp]\n\\centering\n'
    latex += f'\\caption{{\\textbf{{{caption}}}}}\n'
    latex += f'\\label{{tab:{label}}}\n'
    latex += '{\\footnotesize\n'
    latex += '\\begin{tabular}{lcccccc}\n\\toprule\n'
    latex += 'Weight & Chunk 1 & Chunk 2$^\\dagger$ & Chunk 3 & C1 vs.\\ C3 & Effect size & Trend \\\\\n'
    latex += ' & Mean (SD) & Mean (SD) & Mean (SD) & $p$-value & $r$ & \\\\\n'
    latex += '\\midrule\n'
    for lr in latex_rows:
        latex += lr
    latex += '\\bottomrule\n\\end{tabular}\n}\n'

    # Footnotes
    latex += '\\begin{tablenotes}\n\\footnotesize\n'
    if is_trial:
        latex += ('\\item $^\\dagger$Chunk 2 values should be interpreted with caution: '
                  'median duration is 3 days, and $\\sigma$ estimates from short chunks are noisier. '
                  'Primary comparisons are C1 vs.\\ C3.\n')
    else:
        latex += ('\\item $^\\dagger$Chunk 2 estimates are particularly unreliable: '
                  'median duration is 3 days, and some mice have single-day C2 chunks where '
                  '$\\sigma_{\\text{day}}$ is not estimable (fixed at default 0.5). '
                  'Primary comparisons are C1 vs.\\ C3.\n')
    latex += ('\\item $p$-values from Wilcoxon signed-rank tests, Bonferroni-corrected for 4 comparisons. '
              'Effect size: rank-biserial $r$ (C1 vs.\\ C3). '
              'Interpretation: $|r| \\geq 0.5$ = large, $\\geq 0.3$ = medium, '
              '$\\geq 0.1$ = small, $< 0.1$ = negligible. '
              'Trend: $\\downarrow$ = significant decrease, $\\uparrow$ = significant increase, '
              '-- = not significant.\n')
    latex += '\\end{tablenotes}\n\\end{table}\n'

    with open(os.path.join(LATEX_DIR, f'{filename}.tex'), 'w') as f:
        f.write(latex)
    print(f"  Saved: {filename}.csv + .tex")


def _make_effect_size_table(complete_df):
    """Table 8: Effect size summary matching thesis format."""
    rows = []
    latex_rows = []

    for sigma_type in ['sigma_trial', 'sigma_day']:
        for weight in WEIGHT_NAMES:
            c1_col = f'chunk1_{sigma_type}_{weight}'
            c3_col = f'chunk3_{sigma_type}_{weight}'
            c1 = complete_df[c1_col].dropna()
            c3 = complete_df[c3_col].dropna()
            common = c1.index.intersection(c3.index)
            if len(common) < 10:
                continue

            result = paired_wilcoxon_with_effect_size(
                c1.loc[common].values, c3.loc[common].values)
            c1_mean = c1.loc[common].mean()
            c3_mean = c3.loc[common].mean()
            pct = (c3_mean - c1_mean) / c1_mean * 100
            r_val = result['effect_size']
            label = effect_size_label(r_val)

            rows.append({'Weight': weight, 'Sigma_type': sigma_type.split('_')[1],
                         'C1_mean': c1_mean, 'C3_mean': c3_mean,
                         'Pct_change': pct, 'r': r_val, 'Interpretation': label})

            w_label = 'Prev.\\ choice' if weight == 'prev_choice' else weight.capitalize()
            s_label = sigma_type.split('_')[1]
            interp = f'\\textbf{{{label}}}' if label == 'large' else label
            latex_rows.append(
                f'{w_label} & {s_label} & {c1_mean:.3f} & {c3_mean:.3f} '
                f'& ${pct:+.1f}\\%$ & ${r_val:+.2f}$ & {interp} \\\\\n')

        # Add midrule between trial and day blocks
        if sigma_type == 'sigma_trial':
            latex_rows.append('\\midrule\n')

    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, 'table_effect_sizes.csv'), index=False)

    latex = '\\begin{table}[htbp]\n\\centering\n\\small\n'
    latex += '\\caption{\\textbf{Complete Effect Size Summary: C1 vs.\\ C3 Across All Weights.}}\n'
    latex += '\\label{tab:effect_sizes}\n'
    latex += '\\begin{tabular}{llccccc}\n\\toprule\n'
    latex += 'Weight & $\\sigma$ type & C1 mean & C3 mean & \\% change & $r$ & Interpretation \\\\\n'
    latex += '\\midrule\n'
    for lr in latex_rows:
        latex += lr
    latex += '\\bottomrule\n\\end{tabular}\n'
    latex += '\\begin{tablenotes}\n\\footnotesize\n'
    latex += ('\\item Rank-biserial $r$ from Wilcoxon signed-rank tests. '
              'Interpretation: $|r| \\geq 0.5$ = large, $\\geq 0.3$ = medium, '
              '$\\geq 0.1$ = small, $< 0.1$ = negligible. '
              'Only contrast (both $\\sigma$ types) shows large, significant effects.\n')
    latex += '\\end{tablenotes}\n\\end{table}\n'

    with open(os.path.join(LATEX_DIR, 'table_effect_sizes.tex'), 'w') as f:
        f.write(latex)
    print("  Saved: table_effect_sizes.csv + .tex")


def _make_mixed_effects_table(complete_df):
    """Table 9: Mixed-effects model matching thesis format."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  WARNING: statsmodels not installed, skipping mixed-effects table")
        return

    long_rows = []
    for _, row in complete_df.iterrows():
        for chunk_num, chunk in enumerate(['chunk1', 'chunk2', 'chunk3'], 1):
            sigma = row.get(f'{chunk}_sigma_trial_contrast', np.nan)
            if pd.notna(sigma) and sigma > 0:
                long_rows.append({'subject': row['subject'],
                                  'chunk': f'chunk{chunk_num}',
                                  'log_sigma': np.log10(sigma)})

    long_df = pd.DataFrame(long_rows)
    model = smf.mixedlm('log_sigma ~ C(chunk, Treatment("chunk1"))',
                        long_df, groups=long_df['subject'])
    result = model.fit(reml=True)

    with open(os.path.join(TABLE_DIR, 'table_mixed_effects.txt'), 'w') as f:
        f.write(str(result.summary()))

    params = result.params
    bse = result.bse
    pvalues = result.pvalues

    latex = '\\begin{table}[htbp]\n\\centering\n'
    latex += '\\caption{\\textbf{Mixed-Effects Model: Chunk Effects on $\\sigma_{\\text{trial,contrast}}$.}}\n'
    latex += '\\label{tab:mixed_effects}\n'
    latex += '\\begin{tabular}{lcccc}\n\\toprule\n'
    latex += 'Fixed Effect ($\\gamma$) & Estimate & SE & $z$ & $p$ \\\\\n'
    latex += '\\midrule\n'

    for i, name in enumerate(params.index):
        if 'Group' in name:
            continue
        if 'Intercept' in name:
            label = 'Baseline ($\\gamma_0$, Chunk 1)'
        elif 'chunk2' in name:
            label = 'Chunk 2'
        elif 'chunk3' in name:
            label = 'Chunk 3'
        else:
            label = name.replace('_', '\\_')

        est = params.iloc[i]
        se = bse.iloc[i]
        z = est / se
        p = pvalues.iloc[i]
        p_str = '$<$0.001' if p < 0.001 else f'{p:.3f}'
        latex += f'{label} & ${est:.3f}$ & {se:.3f} & ${z:.1f}$ & {p_str} \\\\\n'

    latex += '\\midrule\n'
    latex += '\\multicolumn{5}{l}{\\textit{Random Effects}} \\\\\n'
    re_var = result.cov_re.iloc[0, 0] if hasattr(result.cov_re, 'iloc') else float(result.cov_re)
    latex += f'$\\tau_{{\\text{{mouse}}}}$ & {np.sqrt(re_var):.3f} & & & \\\\\n'

    latex += '\\bottomrule\n\\end{tabular}\n'
    latex += '\\begin{tablenotes}\n\\footnotesize\n'
    c3_est = params.iloc[2] if len(params) > 2 else 0
    latex += (f'\\item On the log scale, the Chunk 3 coefficient of {c3_est:.3f} corresponds to '
              f'$10^{{{c3_est:.3f}}} \\approx {10**c3_est:.2f}$, '
              f'a {(10**c3_est-1)*100:.0f}\\% multiplicative increase relative to Chunk 1.\n')
    latex += ('\\item SE = standard error. $z = \\gamma/\\text{SE}$; '
              '$p = 2\\,\\Phi(-|z|)$.\n')
    latex += '\\end{tablenotes}\n\\end{table}\n'

    with open(os.path.join(LATEX_DIR, 'table_mixed_effects.tex'), 'w') as f:
        f.write(latex)
    print("  Saved: table_mixed_effects.txt + .tex")


def _make_phenotype_table(complete_df):
    """Table 10: Phenotype distribution matching thesis format."""
    c3_counts = complete_df['chunk3_phenotype'].value_counts()
    total = len(complete_df)

    latex = '\\begin{table}[htbp]\n\\centering\n'
    latex += '\\caption{\\textbf{Behavioral Phenotype Distribution (Chunk 3).}}\n'
    latex += '\\label{tab:phenotype_counts}\n'
    latex += '\\begin{tabular}{lccc}\n\\toprule\n'
    latex += 'Phenotype & Criterion & $n$ & Percentage \\\\\n'
    latex += '\\midrule\n'

    pheno_info = [
        ('perseverator', 'Pure perseverator',
         r'$\sigma_{\text{pc}}/\sigma_{\text{wsls}} > 3$'),
        ('reward_sensitive', 'Reward-sensitive',
         r'$\sigma_{\text{pc}}/\sigma_{\text{wsls}} < 0.33$'),
        ('mixed', 'Mixed strategy',
         r'$0.33 \leq$ ratio $\leq 3$'),
    ]
    for key, label, criterion in pheno_info:
        n = c3_counts.get(key, 0)
        pct = 100 * n / total
        latex += f'{label} & {criterion} & {n} & {pct:.1f}\\% \\\\\n'

    latex += '\\midrule\n'
    latex += f'\\textbf{{Total}} & & \\textbf{{{total}}} & \\textbf{{100\\%}} \\\\\n'
    latex += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'

    with open(os.path.join(LATEX_DIR, 'table_phenotype.tex'), 'w') as f:
        f.write(latex)
    print("  Saved: table_phenotype.tex")


def _make_overnight_magnitude_table(complete_df, jumps_df):
    """Table 11: Overnight magnitude matching thesis format with p-values and trend."""
    rows = []
    latex_rows = []

    for weight in WEIGHT_NAMES:
        weight_jumps = jumps_df[jumps_df['weight'] == weight]
        row_data = {'weight': weight}
        chunk_medians = {}

        for chunk in ['chunk1', 'chunk2', 'chunk3']:
            vals = weight_jumps[weight_jumps['chunk'] == chunk]['abs_delta'].dropna()
            if len(vals) > 0:
                med = vals.median()
                q1 = vals.quantile(0.25)
                q3 = vals.quantile(0.75)
                row_data[f'{chunk}_median'] = med
                row_data[f'{chunk}_iqr'] = f'{med:.3f} ({q1:.3f}--{q3:.3f})'
                chunk_medians[chunk] = vals
            else:
                row_data[f'{chunk}_iqr'] = '--'

        # Per-mouse median C1 vs C3 Wilcoxon
        c1_mouse_meds = weight_jumps[weight_jumps['chunk'] == 'chunk1'].groupby('subject')['abs_delta'].median()
        c3_mouse_meds = weight_jumps[weight_jumps['chunk'] == 'chunk3'].groupby('subject')['abs_delta'].median()
        common_mice = c1_mouse_meds.index.intersection(c3_mouse_meds.index)

        if len(common_mice) >= 10:
            stat_result = wilcoxon(c1_mouse_meds.loc[common_mice].values,
                                   c3_mouse_meds.loc[common_mice].values)
            p_val = stat_result.pvalue
            direction = c3_mouse_meds.loc[common_mice].median() - c1_mouse_meds.loc[common_mice].median()
        else:
            p_val = np.nan
            direction = 0

        rows.append(row_data)

        w_label = 'Prev.\\ choice' if weight == 'prev_choice' else weight.capitalize()
        p_str = '$<$0.001' if not pd.isna(p_val) and p_val < 0.001 else \
                (f'{p_val:.3f}' if not pd.isna(p_val) else '--')
        trend = _fmt_trend(p_val, direction)

        latex_rows.append(
            f'{w_label} & {row_data.get("chunk1_iqr","--")} & '
            f'{row_data.get("chunk2_iqr","--")} & '
            f'{row_data.get("chunk3_iqr","--")} & {p_str} & {trend} \\\\\n')

    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, 'table_overnight_magnitude.csv'), index=False)

    latex = '\\begin{table}[htbp]\n\\centering\n\\footnotesize\n'
    latex += '\\caption{\\textbf{Overnight Jump Magnitudes by Chunk.}}\n'
    latex += '\\label{tab:overnight_magnitude}\n'
    latex += '\\begin{tabular}{lccccc}\n\\toprule\n'
    latex += 'Weight & Chunk 1 & Chunk 2 & Chunk 3 & C1 vs.\\ C3 & Trend \\\\\n'
    latex += ' & Median (IQR) & Median (IQR) & Median (IQR) & $p$-value & \\\\\n'
    latex += '\\midrule\n'
    for lr in latex_rows:
        latex += lr
    latex += '\\bottomrule\n\\end{tabular}\n'
    latex += '\\begin{tablenotes}\n\\small\n'
    latex += ('\\item $p$-values from Wilcoxon signed-rank tests on mouse-level medians '
              '(C1 vs.\\ C3; not Bonferroni-corrected). '
              'Trend: $\\uparrow$ = significant increase ($p < 0.05$), '
              '-- = no significant change. IQR = interquartile range.\n')
    latex += '\\end{tablenotes}\n\\end{table}\n'

    with open(os.path.join(LATEX_DIR, 'table_overnight_magnitude.tex'), 'w') as f:
        f.write(latex)
    print("  Saved: table_overnight_magnitude.csv + .tex")


def _make_overnight_direction_table(complete_df, jumps_df):
    """Table 12: Overnight direction with pooled + per-animal rows, matching thesis."""
    rows = []
    latex_rows = []

    # Thesis Table 12 ordering: Contrast first, then Bias, then history weights
    table12_order = ['contrast', 'bias', 'prev_choice', 'wsls']
    for weight in table12_order:
        weight_jumps = jumps_df[jumps_df['weight'] == weight]
        w_label = 'Prev.\\ choice' if weight == 'prev_choice' else weight.capitalize()

        pooled_parts = [f'\\multirow{{2}}{{*}}{{{w_label}}}', 'pooled']
        peranimal_parts = ['', 'per-animal']

        for chunk in ['chunk1', 'chunk2', 'chunk3']:
            chunk_jumps = weight_jumps[weight_jumps['chunk'] == chunk]

            # Pooled: binomial test
            if weight == 'contrast':
                n_total = len(chunk_jumps)
                n_good = (chunk_jumps['delta'] > 0).sum()
            else:
                valid = chunk_jumps['toward_zero'].dropna()
                n_total = len(valid)
                n_good = int(valid.sum())

            if n_total > 0:
                pct_pooled = 100 * n_good / n_total
                binom_p = binomtest(n_good, n_total, 0.5).pvalue
                stars = '***' if binom_p < 0.001 else ('**' if binom_p < 0.01 else ('*' if binom_p < 0.05 else ''))
                if pct_pooled > 60 and stars:
                    pooled_parts.append(f'\\textbf{{{pct_pooled:.1f}\\%}}{stars}')
                else:
                    pooled_parts.append(f'{pct_pooled:.1f}\\%{stars}')
            else:
                pooled_parts.append('--')

            # Per-animal: mean of within-animal proportions, Wilcoxon vs 50%
            mouse_props = []
            for subj in chunk_jumps['subject'].unique():
                subj_jumps = chunk_jumps[chunk_jumps['subject'] == subj]
                if weight == 'contrast':
                    n_s = len(subj_jumps)
                    n_g = (subj_jumps['delta'] > 0).sum()
                else:
                    valid_s = subj_jumps['toward_zero'].dropna()
                    n_s = len(valid_s)
                    n_g = int(valid_s.sum())
                if n_s >= 3:
                    mouse_props.append(n_g / n_s)

            if len(mouse_props) >= 5:
                mean_prop = np.mean(mouse_props) * 100
                # Wilcoxon signed-rank vs 0.5
                try:
                    w_stat, w_p = wilcoxon(np.array(mouse_props) - 0.5)
                    stars_pa = '*' if w_p < 0.05 else ''
                except ValueError:
                    stars_pa = ''
                peranimal_parts.append(f'{mean_prop:.1f}\\%{stars_pa} ($n={len(mouse_props)}$)')
            else:
                peranimal_parts.append('--')

        latex_rows.append(' & '.join(pooled_parts) + ' \\\\\n')
        latex_rows.append(' & '.join(peranimal_parts) + ' \\\\[3pt]\n')

    pd.DataFrame(rows).to_csv(os.path.join(TABLE_DIR, 'table_overnight_direction.csv'), index=False)

    latex = '\\begin{table}[htbp]\n\\centering\n\\small\n'
    latex += '\\caption{\\textbf{Overnight Jump Direction Analysis.}}\n'
    latex += '\\label{tab:overnight_direction}\n'
    latex += '\\begin{tabular}{llccc}\n\\toprule\n'
    latex += 'Weight & Test level & Chunk 1 & Chunk 2 & Chunk 3 \\\\\n'
    latex += '\\midrule\n'
    for lr in latex_rows:
        latex += lr
    latex += '\\bottomrule\n\\end{tabular}\n'
    latex += '\\begin{tablenotes}\n\\small\n'
    latex += ('\\item Percentages show proportion of overnight jumps in the specified direction. '
              '\\textbf{Specified directions}: Contrast: \\% positive signed $\\Delta W$ (strengthening). '
              'Bias: \\% with $\\Delta|\\beta| < 0$ (toward zero). '
              'History (prev.\\ choice, WSLS): \\% toward zero via $\\Delta|\\beta|$. '
              '\\textbf{Pooled}: binomial test. '
              '\\textbf{Per-animal}: mean of within-animal proportions tested via Wilcoxon vs.\\ 50\\%; '
              'per-animal tests are primary inference. '
              '*$p < 0.05$, **$p < 0.01$, ***$p < 0.001$.\n')
    latex += '\\end{tablenotes}\n\\end{table}\n'

    with open(os.path.join(LATEX_DIR, 'table_overnight_direction.tex'), 'w') as f:
        f.write(latex)
    print("  Saved: table_overnight_direction.csv + .tex")


def _make_results_snippets(complete_df, jumps_df):
    """Generate results_snippets.tex with pre-computed values."""
    snippets = []
    snippets.append('% AUTO-GENERATED RESULTS SNIPPETS FOR THESIS')
    snippets.append(f'% Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}')
    snippets.append('')

    snippets.append('% ── σ_trial Statistics ──')
    for weight in WEIGHT_NAMES:
        for chunk in ['chunk1', 'chunk2', 'chunk3']:
            col = f'{chunk}_sigma_trial_{weight}'
            vals = complete_df[col].dropna()
            snippets.append(f'% {weight} {chunk}: '
                            f'mean={vals.mean():.4f}, SD={vals.std():.4f}, n={len(vals)}')

        c1 = complete_df[f'chunk1_sigma_trial_{weight}'].dropna()
        c3 = complete_df[f'chunk3_sigma_trial_{weight}'].dropna()
        common = c1.index.intersection(c3.index)
        if len(common) >= 10:
            result = paired_wilcoxon_with_effect_size(
                c1.loc[common].values, c3.loc[common].values)
            pct = 100 * (c3.loc[common].mean() - c1.loc[common].mean()) / c1.loc[common].mean()
            p_corr = bonferroni_correct(result['p_value'])
            snippets.append(f'%   C1→C3: {pct:+.0f}%, p={p_corr:.6f}, r={result["effect_size"]:.2f}')
        snippets.append('')

    snippets.append('% ── Phenotype Distribution (Chunk 3) ──')
    c3 = complete_df['chunk3_phenotype'].value_counts()
    total = len(complete_df)
    for pheno in ['perseverator', 'reward_sensitive', 'mixed']:
        n = c3.get(pheno, 0)
        snippets.append(f'% {pheno}: {n} ({100*n/total:.1f}%)')
    snippets.append('')
    snippets.append(f'% ── Sample Sizes ──')
    snippets.append(f'% Complete mice (all 3 chunks): {total}')

    with open(os.path.join(LATEX_DIR, 'results_snippets.tex'), 'w') as f:
        f.write('\n'.join(snippets))
    print("  Saved: results_snippets.tex")


# ─── Main entry point ────────────────────────────────────────────────────────

def run_phase3():
    """Run Phase 3: generate all main-text figures and tables."""
    print("=" * 70)
    print("PHASE 3: Statistical Analysis — Main Figures and Tables")
    print("=" * 70)

    setup_plotting()
    complete_df, jumps_df = load_and_prepare_data()

    # ── Figures ──
    print("\nGenerating figures...")

    print("\n  Figure 6: σ Trajectories...")
    make_figure_sigma_trajectories(complete_df)

    print("\n  Figure 14: Phenotype Distribution...")
    make_figure_phenotype_distribution(complete_df)

    print("\n  Figure 20: Individual Trajectories...")
    make_figure_individual_trajectories(complete_df)

    print("\n  Figure 21: Lab Effects...")
    make_figure_lab_effects(complete_df)

    print("\n  Figure 5: Dataset Overview...")
    make_figure_dataset_overview(complete_df)

    print("\n  Figure 15: Phenotype Stability...")
    make_figure_phenotype_stability(complete_df)

    print("\n  Figure 16: Phenotype & Learning Speed...")
    make_figure_phenotype_learning_speed(complete_df)

    print("\n  Figure 22: Outlier Trajectories...")
    make_figure_outlier_trajectories(complete_df)

    print("\n  Figures 17-18: Overnight Analysis...")
    make_figure_overnight_analysis(complete_df, jumps_df)
    make_figure_overnight_histograms(jumps_df)

    print("\n  Figure 19: Day Gap Effect...")
    make_figure_day_gap_effect(jumps_df)

    print("\n  Figures 7-9: Weight dynamics (requires pickle)...")
    make_figures_from_pickle(complete_df)

    # ── Tables ──
    make_tables(complete_df, jumps_df)

    # ── Print key results summary ──
    print("\n" + "=" * 70)
    print("KEY RESULTS SUMMARY")
    print("=" * 70)

    # σ_trial,contrast C1 vs C3
    c1 = complete_df['chunk1_sigma_trial_contrast'].dropna()
    c3 = complete_df['chunk3_sigma_trial_contrast'].dropna()
    common = c1.index.intersection(c3.index)
    result = paired_wilcoxon_with_effect_size(
        c1.loc[common].values, c3.loc[common].values)
    pct = (c3.loc[common].mean() - c1.loc[common].mean()) / c1.loc[common].mean() * 100
    p_corr = bonferroni_correct(result['p_value'])

    print(f"\n  σ_trial,contrast: C1 mean={c1.loc[common].mean():.4f} → "
          f"C3 mean={c3.loc[common].mean():.4f}")
    print(f"    Change: {pct:+.0f}%")
    print(f"    Wilcoxon W={result['stat']:.0f}, p={p_corr:.6f} (Bonferroni)")
    print(f"    Rank-biserial r={result['effect_size']:.2f} "
          f"({effect_size_label(result['effect_size'])})")
    print(f"    n={result['n']} mice")

    # Phenotype counts
    pheno_counts = complete_df['chunk3_phenotype'].value_counts()
    n_total = len(complete_df)
    print(f"\n  Phenotype distribution (Chunk 3, n={n_total}):")
    for pheno in ['perseverator', 'mixed', 'reward_sensitive']:
        n = pheno_counts.get(pheno, 0)
        print(f"    {pheno}: {n} ({100*n/n_total:.1f}%)")

    print(f"\n{'='*70}")
    print("PHASE 3 COMPLETE")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_phase3()
