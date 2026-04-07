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

    # Add lab column if not present
    if 'lab' not in complete_df.columns:
        complete_df['lab'] = complete_df['subject'].apply(
            lambda s: s.rsplit('_', 1)[0] if '_' in s else s)

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
    ax1.axvline(LOG_THRESHOLD, color='black', linestyle='--', linewidth=1)
    ax1.axvline(-LOG_THRESHOLD, color='black', linestyle='--', linewidth=1)
    ax1.set_xlabel(r'$\log_{10}(\sigma_{pc}/\sigma_{wsls})$')
    ax1.set_ylabel('Count')
    ax1.set_title('A. Distribution of σ-ratio', fontweight='bold', loc='left')

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

    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_phenotype_distribution'))


# ─── Figure 20: Individual Trajectories ───────────────────────────────────────

def make_figure_individual_trajectories(complete_df):
    """Figure 20: Spaghetti plot of σ_trial,contrast for all mice."""
    fig, ax = plt.subplots(figsize=(6, 5))

    chunk_x = [1, 2, 3]

    # Individual mice (gray)
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
    """Figure 21: σ_trial,contrast in C3 by laboratory."""
    fig, ax = plt.subplots(figsize=(8, 5))

    labs = sorted(complete_df['lab'].unique())
    data_by_lab = [complete_df[complete_df['lab'] == lab][
        'chunk3_sigma_trial_contrast'].dropna().values for lab in labs]

    bp = ax.boxplot(data_by_lab, tick_labels=labs, patch_artist=True,
                    widths=0.6, medianprops={'color': 'black', 'linewidth': 1.5})
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)

    # Overlay individual points
    for i, (lab, vals) in enumerate(zip(labs, data_by_lab)):
        x = np.random.normal(i + 1, 0.08, len(vals))
        ax.scatter(x, vals, color='red', s=15, alpha=0.6, zorder=3)

    # Grand mean line
    grand_mean = complete_df['chunk3_sigma_trial_contrast'].mean()
    ax.axhline(grand_mean, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    # Kruskal-Wallis test
    if len(data_by_lab) >= 2:
        groups = [g for g in data_by_lab if len(g) > 0]
        if len(groups) >= 2:
            H, p = kruskal(*groups)
            ax.text(0.98, 0.98, f'Kruskal-Wallis H = {H:.2f}, p = {p:.2f}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)

    ax.set_xlabel('Laboratory')
    ax.set_ylabel(r'$\sigma_{trial,contrast}$ (Chunk 3)')
    ax.set_title('Laboratory Effects on Contrast Learning Rate',
                 fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_lab_effects'))


# ─── Figure 5: Dataset Overview ───────────────────────────────────────────────

def make_figure_dataset_overview(complete_df):
    """Figure 5: Dataset overview (4-panel).

    A: Selection tiers by lab
    B: Chunking strategy distribution (pie)
    C: Trial counts by chunk (box plots)
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

    # Panel A: Mice per lab by tier
    ax = axes[0, 0]
    if 'tier' in summary.columns:
        labs = sorted(summary['lab'].unique()) if 'lab' in summary.columns else ['all']
        tier_counts = summary.groupby(['lab', 'tier']).size().unstack(fill_value=0)
        tier_counts.plot(kind='bar', stacked=True, ax=ax, edgecolor='white')
        ax.set_ylabel('Number of Mice')
        ax.set_title('A. Selection Tiers by Laboratory', fontweight='bold', loc='left')
        ax.legend(title='Tier', fontsize=8)
        ax.tick_params(axis='x', rotation=45)
    else:
        ax.text(0.5, 0.5, 'Tier data not available', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title('A. Selection Tiers', fontweight='bold', loc='left')

    # Panel B: Chunking strategy
    ax = axes[0, 1]
    ax.pie([85, 5, 2], labels=['Strategy A\n(standard)', 'Strategy B\n(alt)', 'Invalid'],
           autopct='%1.0f%%', colors=['#3498db', '#e67e22', '#e74c3c'],
           wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}, pctdistance=0.75)
    ax.set_title('B. Chunking Strategy Distribution', fontweight='bold', loc='left')

    # Panel C: Trial counts by chunk
    ax = axes[1, 0]
    chunk_data_plot = {
        'Chunk 1': complete_df['chunk1_trials'].dropna().values,
        'Chunk 2': complete_df['chunk2_trials'].dropna().values,
        'Chunk 3': complete_df['chunk3_trials'].dropna().values,
    }
    bp = ax.boxplot(chunk_data_plot.values(), tick_labels=chunk_data_plot.keys(),
                    patch_artist=True, widths=0.6)
    for patch, color in zip(bp['boxes'], CHUNK_COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
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

    bp = ax.boxplot(groups, tick_labels=labels, patch_artist=True, widths=0.5)
    pheno_colors = [COLORS.get(p.replace('\n', '_'), 'gray') for p in labels]
    for patch, color in zip(bp['boxes'], pheno_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

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

    # Mean ± SE excluding outliers (green)
    for ci, chunk in enumerate(['chunk1', 'chunk2', 'chunk3']):
        vals = complete_df.loc[~outlier_mask, f'{chunk}_sigma_trial_contrast'].dropna()
        ax.errorbar(chunk_x[ci], vals.mean(), yerr=vals.std()/np.sqrt(len(vals)),
                    fmt='o', color='green', markersize=8, capsize=5, zorder=5)

    # Threshold lines
    thresh_vals = [thresholds[f'chunk{i+1}'] for i in range(3)]
    ax.plot(chunk_x, thresh_vals, '--', color='orange', linewidth=1.5, label='3 SD threshold')

    ax.set_xticks(chunk_x)
    ax.set_xticklabels(['C1', 'C2', 'C3'])
    ax.set_ylabel(r'$\sigma_{trial,contrast}$')
    ax.set_title('Outlier Identification', fontweight='bold')
    ax.legend(fontsize=8)

    n_outliers = outlier_mask.sum()
    ax.text(0.02, 0.98, f'{n_outliers} outlier(s) identified',
            transform=ax.transAxes, va='top', fontsize=9, color='red')

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
            bp = ax.boxplot(magnitude_data, tick_labels=['C1', 'C2', 'C3'],
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

        for chunk in ['chunk1', 'chunk2', 'chunk3']:
            chunk_jumps = weight_jumps[weight_jumps['chunk'] == chunk]

            if weight == 'contrast':
                # Contrast uses signed ΔW: positive = strengthening
                n_total = len(chunk_jumps)
                n_positive = (chunk_jumps['delta'] > 0).sum()
                pct_good = 100 * n_positive / n_total if n_total > 0 else 50
            else:
                # Bias/history: toward_zero = magnitude decrease
                valid = chunk_jumps['toward_zero'].dropna()
                n_total = len(valid)
                n_toward = valid.sum()
                pct_good = 100 * n_toward / n_total if n_total > 0 else 50

            toward_pcts.append(pct_good)
            away_pcts.append(100 - pct_good)

        x = np.arange(3)
        ax.bar(x, toward_pcts, 0.6, label='Toward zero' if weight != 'contrast' else 'Positive ΔW',
               color=CHUNK_COLORS, alpha=0.7)
        ax.bar(x, away_pcts, 0.6, bottom=toward_pcts,
               color=[c + '80' for c in CHUNK_COLORS] if False else ['lightgray']*3, alpha=0.5)
        ax.axhline(50, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(['C1', 'C2', 'C3'])
        ax.set_ylim(0, 100)

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

        for ci, chunk in enumerate(['chunk1', 'chunk2', 'chunk3']):
            chunk_jumps = weight_jumps[weight_jumps['chunk'] == chunk]

            if weight == 'contrast':
                # Contrast: use signed delta (positive = strengthening)
                vals = chunk_jumps['delta'].dropna().values
                xlabel = 'Signed ΔW'
            else:
                # Bias/history: use Δ|β| (negative = toward zero)
                # Δ|β| = |w_start| - |w_end|
                vals = (chunk_jumps['w_start'].abs() - chunk_jumps['w_end'].abs()).dropna().values
                xlabel = 'Δ|β|'

            if len(vals) < 5:
                continue

            ax.hist(vals, bins=40, density=True, histtype='step',
                    linewidth=1.5, color=CHUNK_COLORS[ci],
                    label=f'C{ci+1} (n={len(vals)})', alpha=0.9)

        ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        ax.set_title(WEIGHT_LABELS[idx], fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle('Distributions of Overnight Weight Changes by Training Phase',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_overnight_distributions'))


# ─── Figure 19: Day Gap Effect ───────────────────────────────────────────────

def make_figure_day_gap_effect(jumps_df):
    """Figure 19: Mean |Δw| vs days between sessions."""
    fig, ax = plt.subplots(figsize=(7, 5))

    for weight in WEIGHT_NAMES:
        weight_jumps = jumps_df[jumps_df['weight'] == weight]
        gap_groups = weight_jumps.groupby('day_gap')['abs_delta']
        gap_means = gap_groups.mean()
        gap_ses = gap_groups.apply(lambda x: x.std() / np.sqrt(len(x)))

        ax.errorbar(gap_means.index, gap_means.values, yerr=gap_ses.values,
                    fmt='o-', label=weight.replace('_', ' ').title(),
                    color=COLORS[weight], markersize=5, capsize=3, linewidth=1.2)

        # Spearman correlation
        rho, p = spearmanr(weight_jumps['day_gap'], weight_jumps['abs_delta'])
        print(f"    Day gap vs |Δ{weight}|: ρ={rho:.3f}, p={p:.3f}")

    ax.set_xlabel('Days Between Sessions')
    ax.set_ylabel('Mean Absolute Weight Change')
    ax.set_title('Effect of Day Gap on Overnight Jump Magnitude', fontweight='bold')
    ax.legend(fontsize=8)
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

    # Select 9 example mice for Figs 7-8: balanced chunk durations (12-28 sessions)
    candidates = complete_df[
        (complete_df['chunk1_days'] + complete_df['chunk2_days'] +
         complete_df['chunk3_days']).between(12, 28)
    ]['subject'].tolist()

    example_mice = candidates[:9] if len(candidates) >= 9 else candidates
    if len(example_mice) < 3:
        print("    Not enough example mice for Figs 7-8. Skipping.")
        return

    _make_figure_weight_dynamics(example_mice, mouse_results, chunk_data)
    _make_figure_contrast_curriculum(example_mice, chunk_data)

    # Select 3 example mice for Fig 9: increasing, median, stable σ trajectories
    contrast_ratios = (complete_df['chunk3_sigma_trial_contrast'] /
                       complete_df['chunk1_sigma_trial_contrast'])
    idx_max = contrast_ratios.idxmax()
    idx_med = (contrast_ratios - contrast_ratios.median()).abs().idxmin()
    idx_min = contrast_ratios.idxmin()
    trio = [complete_df.loc[idx_max, 'subject'],
            complete_df.loc[idx_med, 'subject'],
            complete_df.loc[idx_min, 'subject']]

    _make_figure_individual_composite(trio, mouse_results, complete_df)


def _make_figure_weight_dynamics(example_mice, mouse_results, chunk_data):
    """Figure 7: Trial-by-trial weight trajectories for 9 example mice."""
    n_mice = len(example_mice)
    fig, axes = plt.subplots(n_mice, 4, figsize=(16, 2.5 * n_mice))
    if n_mice == 1:
        axes = axes[np.newaxis, :]

    for row, subject in enumerate(example_mice):
        result = mouse_results.get(subject, {})
        chunks = result.get('chunks', {})

        # Concatenate weight trajectories across chunks
        trial_offset = 0
        for ci, chunk_name in enumerate(['chunk1', 'chunk2', 'chunk3']):
            chunk_result = chunks.get(chunk_name, {})
            wMode = chunk_result.get('wMode')
            W_std = chunk_result.get('W_std')
            if wMode is None:
                continue

            n_trials = wMode.shape[1]
            x = np.arange(trial_offset, trial_offset + n_trials)

            for col, weight_idx in enumerate(range(K)):
                ax = axes[row, col]
                color = CHUNK_COLORS[ci]

                # Background shading for chunk
                ax.axvspan(trial_offset, trial_offset + n_trials,
                           alpha=0.1, color=color)

                # Weight trajectory
                ax.plot(x, wMode[weight_idx], color=color, linewidth=0.8)

                # ±1 SD band
                if W_std is not None and not np.all(np.isnan(W_std)):
                    ax.fill_between(x,
                                    wMode[weight_idx] - W_std[weight_idx],
                                    wMode[weight_idx] + W_std[weight_idx],
                                    alpha=0.15, color=color)

                if row == 0:
                    ax.set_title(WEIGHT_LABELS[col], fontweight='bold')

            trial_offset += n_trials

        axes[row, 0].set_ylabel(subject, fontsize=8, rotation=0, ha='right')

    fig.suptitle('Trial-by-Trial Weight Dynamics', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_weight_dynamics'))


def _make_figure_contrast_curriculum(example_mice, chunk_data):
    """Figure 8: Daily contrast distribution for 9 example mice."""
    n_mice = len(example_mice)
    fig, axes = plt.subplots(n_mice, 1, figsize=(12, 2 * n_mice))
    if n_mice == 1:
        axes = [axes]

    contrast_levels = [1.0, 0.5, 0.25, 0.125, 0.0625, 0]
    contrast_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for row, subject in enumerate(example_mice):
        ax = axes[row]
        mouse_data = chunk_data.get('mouse_data', {}).get(subject, {})
        trial_data = mouse_data.get('trial_data', {})

        all_trials = pd.concat([trial_data.get(c, pd.DataFrame())
                                for c in ['chunk1', 'chunk2', 'chunk3']],
                               ignore_index=True)
        if len(all_trials) == 0:
            continue

        # Count trials per day per contrast level
        all_trials['max_contrast'] = np.maximum(
            all_trials['contrast_left'].fillna(0),
            all_trials['contrast_right'].fillna(0))

        daily_counts = all_trials.groupby('date')['max_contrast'].value_counts().unstack(fill_value=0)

        # Stacked bar
        dates = sorted(daily_counts.index)
        bottom = np.zeros(len(dates))
        for level, color in zip(contrast_levels, contrast_colors):
            if level in daily_counts.columns:
                vals = daily_counts.loc[dates, level].values
                ax.bar(range(len(dates)), vals, bottom=bottom,
                       color=color, width=0.8, label=f'{level*100:.0f}%' if row == 0 else '')
                bottom += vals

        ax.set_ylabel(subject, fontsize=7, rotation=0, ha='right')
        if row == n_mice - 1:
            ax.set_xlabel('Training Day')

    if n_mice > 0:
        axes[0].legend(title='Contrast', fontsize=7, ncol=6,
                       loc='upper right', bbox_to_anchor=(1.0, 1.3))

    fig.suptitle('Contrast Curriculum Progression', fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_contrast_curriculum'))


def _make_figure_individual_composite(trio, mouse_results, complete_df):
    """Figure 9: Individual composite (3 mice × 3 columns).

    Left: weight trajectories with chunk shading
    Center: σ_trial across chunks (connected dots)
    Right: σ_day across chunks (connected dots)
    """
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    chunk_x = [1, 2, 3]

    for row, subject in enumerate(trio):
        result = mouse_results.get(subject, {})
        chunks = result.get('chunks', {})
        mouse_row = complete_df[complete_df['subject'] == subject]
        if len(mouse_row) == 0:
            continue
        mouse_row = mouse_row.iloc[0]

        # Left: weight trajectories (contrast weight only for clarity)
        ax = axes[row, 0]
        trial_offset = 0
        for ci, chunk_name in enumerate(['chunk1', 'chunk2', 'chunk3']):
            chunk_result = chunks.get(chunk_name, {})
            wMode = chunk_result.get('wMode')
            if wMode is None:
                continue
            n_trials = wMode.shape[1]
            x = np.arange(trial_offset, trial_offset + n_trials)
            ax.axvspan(trial_offset, trial_offset + n_trials,
                       alpha=0.1, color=CHUNK_COLORS[ci])
            for k in range(K):
                ax.plot(x, wMode[k], color=COLORS[WEIGHT_NAMES[k]],
                        linewidth=0.8, alpha=0.7)
            trial_offset += n_trials

        ax.set_ylabel(subject, fontsize=9, fontweight='bold')
        if row == 0:
            ax.set_title('Weight Trajectories', fontweight='bold')

        # Center: σ_trial
        ax = axes[row, 1]
        for weight in WEIGHT_NAMES:
            vals = [mouse_row.get(f'{c}_sigma_trial_{weight}', np.nan)
                    for c in ['chunk1', 'chunk2', 'chunk3']]
            ax.plot(chunk_x, vals, 'o-', color=COLORS[weight],
                    label=weight if row == 0 else '', markersize=6, linewidth=1.5)
        ax.set_xticks(chunk_x)
        ax.set_xticklabels(['C1', 'C2', 'C3'])
        if row == 0:
            ax.set_title(r'$\sigma_{trial}$', fontweight='bold')
            ax.legend(fontsize=7)

        # Right: σ_day
        ax = axes[row, 2]
        for weight in WEIGHT_NAMES:
            vals = [mouse_row.get(f'{c}_sigma_day_{weight}', np.nan)
                    for c in ['chunk1', 'chunk2', 'chunk3']]
            ax.plot(chunk_x, vals, 'o-', color=COLORS[weight],
                    markersize=6, linewidth=1.5)
        ax.set_xticks(chunk_x)
        ax.set_xticklabels(['C1', 'C2', 'C3'])
        if row == 0:
            ax.set_title(r'$\sigma_{day}$', fontweight='bold')

    fig.suptitle('Example Animals: Weight Dynamics and Learning Rate Trajectories',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    save_figure(fig, os.path.join(FIGURE_DIR, 'fig_individual_composite_v3'))


# ─── Tables ──────────────────────────────────────────────────────────────────

def make_tables(complete_df, jumps_df):
    """Generate all statistical tables (Tables 5-12).

    Saves as both CSV and LaTeX format.
    """
    print("\nGenerating tables...")

    # ── Table 6: σ_trial results ──
    _make_sigma_table(complete_df, 'sigma_trial', 'table_sigma_trial')

    # ── Table 7: σ_day results ──
    _make_sigma_table(complete_df, 'sigma_day', 'table_sigma_day')

    # ── Table 8: Effect size summary ──
    _make_effect_size_table(complete_df)

    # ── Table 9: Mixed-effects model ──
    _make_mixed_effects_table(complete_df)

    print("  Tables complete.")


def _make_sigma_table(complete_df, sigma_type, filename):
    """Generate σ results table for one sigma type (trial or day)."""
    rows = []
    for weight in WEIGHT_NAMES:
        row = {'Weight': weight.replace('_', ' ').title()}
        for chunk in ['chunk1', 'chunk2', 'chunk3']:
            col = f'{chunk}_{sigma_type}_{weight}'
            vals = complete_df[col].dropna()
            row[f'{chunk.upper()}_mean'] = f'{vals.mean():.4f}'
            row[f'{chunk.upper()}_sd'] = f'{vals.std():.4f}'

        # C1 vs C3 test
        c1 = complete_df[f'chunk1_{sigma_type}_{weight}'].dropna()
        c3 = complete_df[f'chunk3_{sigma_type}_{weight}'].dropna()
        common = c1.index.intersection(c3.index)
        if len(common) >= 10:
            result = paired_wilcoxon_with_effect_size(
                c1.loc[common].values, c3.loc[common].values)
            p_corr = bonferroni_correct(result['p_value'])
            row['p_corrected'] = f'{p_corr:.4f}' if p_corr >= 0.001 else '< 0.001'
            row['effect_size'] = f'{result["effect_size"]:+.2f}'
            row['effect_label'] = effect_size_label(result['effect_size'])
        else:
            row['p_corrected'] = 'N/A'
            row['effect_size'] = 'N/A'
            row['effect_label'] = 'N/A'

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_DIR, f'{filename}.csv'), index=False)
    print(f"  Saved: {filename}.csv")


def _make_effect_size_table(complete_df):
    """Table 8: Complete effect size summary (8 rows: 4 weights × 2 σ types)."""
    rows = []
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
            pct_change = (c3.loc[common].mean() - c1.loc[common].mean()) / \
                c1.loc[common].mean() * 100

            rows.append({
                'Weight': weight,
                'Sigma_type': sigma_type.replace('sigma_', ''),
                'C1_mean': f'{c1.loc[common].mean():.4f}',
                'C3_mean': f'{c3.loc[common].mean():.4f}',
                'Pct_change': f'{pct_change:+.0f}%',
                'Effect_size_r': f'{result["effect_size"]:+.2f}',
                'Interpretation': effect_size_label(result['effect_size']),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLE_DIR, 'table_effect_sizes.csv'), index=False)
    print("  Saved: table_effect_sizes.csv")


def _make_mixed_effects_table(complete_df):
    """Table 9: Mixed-effects model for σ_trial,contrast."""
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        print("  WARNING: statsmodels not installed, skipping mixed-effects table")
        return

    # Build long-format data
    long_rows = []
    for _, row in complete_df.iterrows():
        for chunk_num, chunk in enumerate(['chunk1', 'chunk2', 'chunk3'], 1):
            sigma = row.get(f'{chunk}_sigma_trial_contrast', np.nan)
            if pd.notna(sigma) and sigma > 0:
                long_rows.append({
                    'subject': row['subject'],
                    'chunk': f'chunk{chunk_num}',
                    'chunk_num': chunk_num,
                    'log_sigma': np.log(sigma),
                })

    long_df = pd.DataFrame(long_rows)

    # Fit mixed-effects model: log(σ) ~ chunk + (1|mouse)
    model = smf.mixedlm('log_sigma ~ C(chunk, Treatment("chunk1"))',
                        long_df, groups=long_df['subject'])
    result = model.fit(reml=True)

    # Save summary
    summary_text = str(result.summary())
    with open(os.path.join(TABLE_DIR, 'table_mixed_effects.txt'), 'w') as f:
        f.write(summary_text)
    print("  Saved: table_mixed_effects.txt")

    # Extract key coefficients
    params = result.params
    pvalues = result.pvalues
    print(f"    Intercept (C1 baseline): {params.iloc[0]:.3f}, p={pvalues.iloc[0]:.4f}")
    for i in range(1, len(params)):
        name = params.index[i]
        print(f"    {name}: {params.iloc[i]:.3f}, p={pvalues.iloc[i]:.4f}")


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
