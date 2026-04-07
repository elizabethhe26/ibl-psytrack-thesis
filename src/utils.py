"""Shared utility functions used across all pipeline phases.

This module contains:
- Plotting setup (publication-quality matplotlib defaults)
- Figure saving (dual PDF + PNG export)
- Statistical helpers (rank-biserial effect size, Bonferroni correction)
- PsyTrack data preparation helpers (contrast compression, predictor building)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os


# ─── Plotting setup ──────────────────────────────────────────────────────────

def setup_plotting():
    """Configure matplotlib for publication-quality figures.

    Call once at the start of any script that generates figures.
    Matches the style used in the original Colab notebooks.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 12,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'lines.linewidth': 1.5,
        'patch.linewidth': 0.5,
    })


def save_figure(fig, filepath, dpi=300):
    """Save figure as both PDF and PNG.

    Args:
        fig: matplotlib Figure object
        filepath: path without extension, or with .pdf extension
        dpi: resolution for both formats
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Normalize path
    base = filepath.replace('.pdf', '').replace('.png', '')

    fig.savefig(f"{base}.pdf", bbox_inches='tight', dpi=dpi)
    fig.savefig(f"{base}.png", bbox_inches='tight', dpi=dpi)
    print(f"  Saved: {base}.pdf and .png")
    plt.close(fig)


# ─── Statistical helpers ─────────────────────────────────────────────────────

def compute_rank_biserial(x, y):
    """Rank-biserial correlation for paired Wilcoxon signed-rank test.

    Thesis definition (Section 2.6):
        r = (W⁺ - W⁻) / (n(n+1)/2)
    where W⁺ = sum of ranks for positive differences (y > x),
          W⁻ = sum of ranks for negative differences (y < x),
          n = number of nonzero differences.

    Args:
        x: array-like, first condition (e.g., C1 values)
        y: array-like, second condition (e.g., C3 values)

    Returns:
        r: rank-biserial correlation. Positive means y > x on average.
    """
    diff = np.array(y) - np.array(x)
    diff = diff[diff != 0]  # Remove zeros (ties at zero)
    n = len(diff)
    if n == 0:
        return 0.0

    ranks = stats.rankdata(np.abs(diff))
    W_pos = np.sum(ranks[diff > 0])
    W_neg = np.sum(ranks[diff < 0])
    r = (W_pos - W_neg) / (n * (n + 1) / 2)
    return r


def paired_wilcoxon_with_effect_size(x, y):
    """Run paired Wilcoxon signed-rank test with rank-biserial effect size.

    Args:
        x: array-like, condition 1 values (e.g., C1)
        y: array-like, condition 2 values (e.g., C3)

    Returns:
        dict with keys: stat, p_value, effect_size, n, direction
    """
    x = np.array(x)
    y = np.array(y)

    # Align: only use mice present in both
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)

    if n < 10:
        return {'stat': np.nan, 'p_value': np.nan, 'effect_size': np.nan,
                'n': n, 'direction': 'insufficient data'}

    stat, p_value = stats.wilcoxon(x, y)
    r = compute_rank_biserial(x, y)
    direction = 'increase' if np.mean(y) > np.mean(x) else 'decrease'

    return {
        'stat': stat,
        'p_value': p_value,
        'effect_size': r,
        'n': n,
        'direction': direction,
    }


def bonferroni_correct(p_value, n_comparisons=4):
    """Apply Bonferroni correction, capping at 1.0."""
    return min(p_value * n_comparisons, 1.0)


def effect_size_label(r):
    """Interpret rank-biserial effect size magnitude.

    Thresholds from thesis: |r| ≥ 0.5 = large, ≥ 0.3 = medium,
    ≥ 0.1 = small, < 0.1 = negligible.
    """
    abs_r = abs(r)
    if abs_r >= 0.5:
        return 'large'
    elif abs_r >= 0.3:
        return 'medium'
    elif abs_r >= 0.1:
        return 'small'
    else:
        return 'negligible'


# ─── PsyTrack data preparation helpers ───────────────────────────────────────

def compress_contrast(contrast_left, contrast_right, p=5):
    """Apply tanh compression and compute signed contrast predictor.

    Transform: ĉ = tanh(p·c) / tanh(p)
    This maps [0, 0.0625, 0.125, 0.25, 0.5, 1.0]
           → [0, 0.383,  0.555, 0.848, 0.987, 1.0]

    The signed predictor is: ĉ_right - ĉ_left
    Positive values → right stimulus is stronger.

    Args:
        contrast_left: array of left stimulus contrasts (0 to 1)
        contrast_right: array of right stimulus contrasts (0 to 1)
        p: compression parameter (default 5, matching Roy et al. 2021)

    Returns:
        signed_contrast: array of compressed signed contrast values
    """
    def transform(c):
        return np.tanh(p * np.asarray(c)) / np.tanh(p)

    return transform(contrast_right) - transform(contrast_left)


def build_predictors(chunk_df):
    """Build PsyTrack input predictors from trial data.

    Constructs 4 predictors:
        1. contrast: signed compressed contrast (tanh transformation)
        2. prev_choice: previous choice direction (+1 right, -1 left, 0 at boundaries)
        3. wsls: win-stay/lose-shift = prev_choice × (2·prev_reward - 1)
        4. bias: always 1 (intercept, handled by PsyTrack internally)

    At the first trial of each calendar day, prev_choice and wsls are
    reset to 0 (conservative: no carry-over across overnight gaps).

    Args:
        chunk_df: DataFrame with columns: contrast_left, contrast_right,
                  choice, rewarded, date

    Returns:
        inputs: dict of predictor arrays, each shape (n_trials, 1)
        day_lengths: array of trial counts per calendar day
        n_trials: total number of trials
    """
    # Sort by date then trial order
    df = chunk_df.sort_values(['date', 'session']).reset_index(drop=True)
    n_trials = len(df)

    # 1. Compressed signed contrast
    contrast = compress_contrast(
        df['contrast_left'].fillna(0).values,
        df['contrast_right'].fillna(0).values
    )

    # 2. Previous choice: {-1, +1}, with 0 at day boundaries
    choice_signed = 2 * df['choice'].values - 1  # 0→-1, 1→+1
    prev_choice = np.zeros(n_trials)
    prev_choice[1:] = choice_signed[:-1]

    # 3. WSLS: prev_choice × (2·prev_reward - 1)
    prev_reward = np.zeros(n_trials)
    prev_reward[1:] = df['rewarded'].values[:-1]
    wsls = prev_choice * (2 * prev_reward - 1)

    # Compute day boundaries from calendar dates
    dates = pd.to_datetime(df['date'])
    date_series = dates.values
    day_changes = np.where(date_series[1:] != date_series[:-1])[0] + 1
    day_boundaries = [0] + list(day_changes)

    # Reset history predictors at day boundaries (no carry-over overnight)
    for boundary in day_boundaries[1:]:  # Skip first (0) and last
        prev_choice[boundary] = 0
        wsls[boundary] = 0

    # Compute dayLength array (trials per calendar day)
    day_lengths = np.diff(day_boundaries + [n_trials])
    unique_dates = [date_series[b] for b in day_boundaries]

    inputs = {
        'contrast': contrast[:, None],
        'prev_choice': prev_choice[:, None],
        'wsls': wsls[:, None],
    }

    return inputs, day_lengths, n_trials, day_boundaries, unique_dates


def classify_phenotype(ratio):
    """Classify mouse behavioral phenotype from σ_pc/σ_wsls ratio.

    Thresholds:
        ratio > 3.0   → 'perseverator'   (choice repetition dominates)
        ratio < 0.33  → 'reward_sensitive' (outcome modulation dominates)
        otherwise     → 'mixed'

    Args:
        ratio: σ_pc / σ_wsls (floored to avoid division by zero)

    Returns:
        phenotype label string
    """
    if ratio > 3.0:
        return 'perseverator'
    elif ratio < 0.33:
        return 'reward_sensitive'
    else:
        return 'mixed'
