"""Central configuration for all pipeline parameters.

All paths, constants, and hyperparameters in one place.
Nothing is hard-coded elsewhere in the codebase.
"""
import os

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DOWNLOAD_DIR = os.path.join(DATA_DIR, 'raw')  # Figshare zip lands here
INTERMEDIATE_DIR = os.path.join(BASE_DIR, 'intermediate')
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
TABLE_DIR = os.path.join(OUTPUT_DIR, 'tables')
LATEX_DIR = os.path.join(OUTPUT_DIR, 'latex')

PROCESSED_DATA_FILE = os.path.join(DATA_DIR, 'ibl_processed.csv')  # output of data_download.py
RAW_DATA_FILE = PROCESSED_DATA_FILE  # backward-compat alias (used by Phase 1, run_all)
FIGSHARE_URL = 'https://ndownloader.figshare.com/files/21623715'

CHUNK_EXPLORATION_FILE = os.path.join(INTERMEDIATE_DIR, 'ibl_chunk_exploration.pkl')
PSYTRACK_RESULTS_FILE = os.path.join(INTERMEDIATE_DIR, 'psytrack_chunk_results.pkl')
SUMMARY_CSV_FILE = os.path.join(INTERMEDIATE_DIR, 'psytrack_chunk_summary.csv')
OVERNIGHT_JUMPS_FILE = os.path.join(INTERMEDIATE_DIR, 'overnight_jumps.csv')

# ─── Data quality thresholds ─────────────────────────────────────────────────
MIN_TRIALS_OK = 1000       # Tier 1: ideal for PsyTrack fitting
MIN_TRIALS_WARNING = 500   # Tier 2: marginal — may work
MIN_TRIALS_CRITICAL = 200  # Below this, σ estimates are unreliable
MIN_BLOCK_LENGTH = 30      # Min consecutive trials to detect bias blocks

# ─── Contrast names mapping ──────────────────────────────────────────────────
CONTRAST_NAMES = {
    0: '0%', 0.0625: '6.25%', 0.125: '12.5%',
    0.25: '25%', 0.5: '50%', 1.0: '100%'
}

# ─── PsyTrack model settings ────────────────────────────────────────────────
# Weight ordering matters — index 0=bias, 1=contrast, 2=prev_choice, 3=wsls
WEIGHT_NAMES = ['bias', 'contrast', 'prev_choice', 'wsls']
WEIGHT_LABELS = ['Bias', 'Contrast', 'Prev. Choice', 'WSLS']
K = 4

# Hyperparameter initialization (matching the original Colab notebooks)
SIGMA_INIT = 2**-5     # Initial σ_trial for each weight
SIGINIT_INIT = 2**5    # Initial weight variance (prior width)
SIGDAY_INIT = 2**-1    # Initial σ_day for each weight
USE_SIGDAY = True      # Fit σ_day (overnight smoothness)
TANH_P = 5             # Compression parameter: tanh(p*c)/tanh(p)

# ─── Phenotype thresholds ────────────────────────────────────────────────────
# Based on σ_pc/σ_wsls ratio in Chunk 3
PHENOTYPE_RATIO_HIGH = 3.0    # > 3 → perseverator
PHENOTYPE_RATIO_LOW = 0.33    # < 0.33 → reward-sensitive
# log10(3) ≈ 0.477, log10(0.33) ≈ -0.477
LOG_THRESHOLD = 0.477

# ─── Plotting ────────────────────────────────────────────────────────────────
# Colorblind-friendly palette (Tol's qualitative scheme)
COLORS = {
    'chunk1': '#4477AA',       # Blue
    'chunk2': '#228833',       # Green
    'chunk3': '#EE6677',       # Red/Pink
    'bias': '#4477AA',
    'contrast': '#228833',
    'prev_choice': '#CCBB44',
    'wsls': '#EE6677',
    'perseverator': '#AA3377',      # Purple
    'reward_sensitive': '#66CCEE',  # Cyan
    'mixed': '#BBBBBB',             # Gray
}
CHUNK_LABELS = ['C1\n(Early)', 'C2\n(Expand)', 'C3\n(Full)']
CHUNK_COLORS = [COLORS['chunk1'], COLORS['chunk2'], COLORS['chunk3']]

DPI = 300
FIGURE_FORMAT = 'pdf'
