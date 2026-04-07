"""Phase 2: Model Fitting — PsyTrack Chunk Fitting.

Fits the 4-weight PsyTrack model (bias, contrast, prev_choice, wsls)
independently to each chunk of each mouse. Extracts smoothness
hyperparameters (σ_trial, σ_day) and computes overnight weight jumps.

The model:
    P(choose right) = logit⁻¹(w_bias·1 + w_contrast·x̂_c + w_pc·x_pc + w_wsls·x_wsls)
where x̂_c uses tanh(5c)/tanh(5) compression on signed contrast.

Output files:
    - psytrack_chunk_results.pkl (complete fit results)
    - psytrack_chunk_summary.csv (σ values per mouse per chunk)
    - overnight_jumps.csv (individual overnight transitions)
"""
import os
import sys
import pickle
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (CHUNK_EXPLORATION_FILE, PSYTRACK_RESULTS_FILE,
                    SUMMARY_CSV_FILE, OVERNIGHT_JUMPS_FILE,
                    INTERMEDIATE_DIR, WEIGHT_NAMES, K,
                    SIGMA_INIT, SIGINIT_INIT, SIGDAY_INIT, USE_SIGDAY)
from src.utils import build_predictors

try:
    import psytrack as psy
except ImportError:
    print("ERROR: psytrack not installed. Run: pip install psytrack")
    sys.exit(1)


# ─── PsyTrack fitting ───────────────────────────────────────────────────────

def fit_psytrack_chunk(chunk_df):
    """Fit PsyTrack to a single chunk of trial data.

    Args:
        chunk_df: DataFrame with columns: contrast_left, contrast_right,
                  choice, rewarded, date, session

    Returns:
        dict with fit results, or dict with success=False on failure
    """
    if len(chunk_df) < 50:
        return {'success': False, 'error_msg': f'Too few trials ({len(chunk_df)})'}

    # Build predictors
    inputs, day_lengths, n_trials, day_boundaries, unique_dates = \
        build_predictors(chunk_df)

    n_days = len(day_lengths)

    # Assemble PsyTrack data dictionary
    dat = {
        'y': chunk_df['choice'].values.astype(float),
        'dayLength': day_lengths,
        'inputs': inputs,
    }

    # Hyperparameter initialization
    weights = {'bias': 1, 'contrast': 1, 'prev_choice': 1, 'wsls': 1}

    if USE_SIGDAY and n_days > 1:
        hyper = {
            'sigma': [SIGMA_INIT] * K,
            'sigInit': [SIGINIT_INIT] * K,
            'sigDay': [SIGDAY_INIT] * K,
        }
        opt_list = ['sigma', 'sigDay']
    else:
        hyper = {
            'sigma': [SIGMA_INIT] * K,
            'sigInit': [SIGINIT_INIT] * K,
        }
        opt_list = ['sigma']

    # Fit
    try:
        hyp, evd, wMode, hess_info = psy.hyperOpt(
            dat, hyper, weights, optList=opt_list, showOpt=False)
        success = True
        error_msg = None
    except Exception as e:
        error_msg = str(e)
        success = False
        # Return fallback values
        hyp = hyper
        evd = np.nan
        wMode = np.full((K, n_trials), np.nan)
        hess_info = {'W_std': np.full((K, n_trials), np.nan)}

    # Extract sigma values
    sigma_trial = _extract_sigma(hyp.get('sigma'))
    sigma_day = _extract_sigma(hyp.get('sigDay')) if USE_SIGDAY and n_days > 1 \
        else {name: np.nan for name in WEIGHT_NAMES}

    return {
        'success': success,
        'error_msg': error_msg,
        'hyp': hyp,
        'evidence': float(evd) if not np.isnan(evd) else np.nan,
        'wMode': wMode,
        'W_std': hess_info.get('W_std', np.full((K, n_trials), np.nan)),
        'sigma_trial': sigma_trial,
        'sigma_day': sigma_day,
        'weights': weights,
        'n_trials': n_trials,
        'n_days': n_days,
        'day_boundaries': day_boundaries,
        'unique_dates': unique_dates,
    }


def _extract_sigma(hyp_values):
    """Convert PsyTrack hyperparameter array to named dict.

    Handles the various formats PsyTrack may return (numpy array, list, None).
    """
    if hyp_values is None:
        return {name: np.nan for name in WEIGHT_NAMES}

    if isinstance(hyp_values, np.ndarray):
        hyp_values = hyp_values.flatten().tolist()
    elif not isinstance(hyp_values, list):
        hyp_values = [hyp_values] * K

    # Ensure we have K values
    while len(hyp_values) < K:
        hyp_values.append(np.nan)

    return {name: float(hyp_values[i]) for i, name in enumerate(WEIGHT_NAMES)}


# ─── Overnight jump computation ──────────────────────────────────────────────

def compute_overnight_jumps(wMode, day_boundaries, unique_dates, subject, chunk_name):
    """Compute weight changes at overnight (day) boundaries.

    For each pair of consecutive calendar days, records the weight value
    at the end of day d and the start of day d+1.

    Direction metrics differ by weight type:
        - Contrast: signed ΔW (positive = strengthening)
        - Bias/history: toward_zero = |w_next_start| < |w_prev_end|

    Args:
        wMode: (K, n_trials) weight trajectory array
        day_boundaries: list of trial indices where each day starts
        unique_dates: list of dates for each day
        subject: mouse ID string
        chunk_name: 'chunk1', 'chunk2', or 'chunk3'

    Returns:
        list of dicts, one per overnight transition per weight
    """
    n_days = len(day_boundaries)
    jumps = []

    for d in range(n_days - 1):
        # Last trial of day d
        if d + 1 < len(day_boundaries):
            end_idx = day_boundaries[d + 1] - 1
        else:
            continue

        # First trial of day d+1
        start_idx = day_boundaries[d + 1]
        if start_idx >= wMode.shape[1]:
            continue

        # Day gap in calendar days
        try:
            date_end = pd.to_datetime(unique_dates[d])
            date_start = pd.to_datetime(unique_dates[d + 1])
            day_gap = (date_start - date_end).days
        except (IndexError, TypeError):
            day_gap = 1

        # Only count as "overnight" if actually different calendar days
        if day_gap < 1:
            continue

        for k, name in enumerate(WEIGHT_NAMES):
            w_end = wMode[k, end_idx]
            w_start = wMode[k, start_idx]
            delta = w_start - w_end

            # Direction classification
            if w_end != 0:
                toward_zero = (abs(w_start) < abs(w_end))
            else:
                toward_zero = None

            jumps.append({
                'subject': subject,
                'chunk': chunk_name,
                'weight': name,
                'day_from': d,
                'day_to': d + 1,
                'date_from': str(date_end.date()) if pd.notna(date_end) else '',
                'date_to': str(date_start.date()) if pd.notna(date_start) else '',
                'day_gap': day_gap,
                'w_end': w_end,
                'w_start': w_start,
                'delta': delta,
                'abs_delta': abs(delta),
                'toward_zero': toward_zero,
            })

    return jumps


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_phase2():
    """Run Phase 2: fit PsyTrack to all mice and chunks.

    Processes mice in priority order (Tier 1 first). Saves checkpoint
    after each mouse to prevent data loss during the long fitting process.

    Returns:
        dict: complete results (also saved to pickle and CSV)
    """
    print("=" * 70)
    print("PHASE 2: Model Fitting — PsyTrack Chunk Fitting")
    print("=" * 70)

    # Load Phase 1 output
    print(f"\nLoading Phase 1 data: {CHUNK_EXPLORATION_FILE}")
    with open(CHUNK_EXPLORATION_FILE, 'rb') as f:
        chunk_data = pickle.load(f)

    # Determine which mice to process
    tier1 = chunk_data['selection']['tier1']
    tier2 = chunk_data['selection']['tier2']
    process_order = tier1 + tier2  # Tier 1 first
    print(f"  Tier 1: {len(tier1)} mice")
    print(f"  Tier 2: {len(tier2)} mice")
    print(f"  Total to process: {len(process_order)}")

    # Check for existing checkpoint
    checkpoint_file = PSYTRACK_RESULTS_FILE.replace('.pkl', '_checkpoint.pkl')
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'rb') as f:
            mouse_results = pickle.load(f)
        already_done = set(mouse_results.keys())
        print(f"  Checkpoint loaded: {len(already_done)} mice already fitted")
    else:
        mouse_results = {}
        already_done = set()

    all_jumps = []
    total_start = time.time()

    for i, subject_id in enumerate(process_order):
        if subject_id in already_done:
            # Recover jumps from checkpoint
            for cn in ['chunk1', 'chunk2', 'chunk3']:
                if cn in mouse_results[subject_id]['chunks']:
                    all_jumps.extend(
                        mouse_results[subject_id]['chunks'][cn].get('overnight_jumps', []))
            continue

        print(f"\n  [{i+1}/{len(process_order)}] {subject_id}")
        mouse_info = chunk_data['mouse_data'].get(subject_id)
        if mouse_info is None:
            print(f"    SKIPPED: not found in Phase 1 data")
            continue

        mouse_result = {'chunks': {}}
        mouse_start = time.time()

        for chunk_name in ['chunk1', 'chunk2', 'chunk3']:
            trial_df = mouse_info['trial_data'].get(chunk_name)
            if trial_df is None or len(trial_df) == 0:
                print(f"    {chunk_name}: no data")
                continue

            n_trials = len(trial_df)
            print(f"    {chunk_name}: {n_trials} trials...", end=' ', flush=True)

            # Fit PsyTrack
            fit_result = fit_psytrack_chunk(trial_df)

            if fit_result['success']:
                sigma_c = fit_result['sigma_trial'].get('contrast', np.nan)
                print(f"OK (σ_contrast={sigma_c:.4f}, "
                      f"evidence={fit_result['evidence']:.1f})")

                # Compute overnight jumps
                jumps = compute_overnight_jumps(
                    fit_result['wMode'],
                    fit_result['day_boundaries'],
                    fit_result['unique_dates'],
                    subject_id, chunk_name)
                all_jumps.extend(jumps)
                fit_result['overnight_jumps'] = jumps
            else:
                print(f"FAILED: {fit_result['error_msg']}")
                fit_result['overnight_jumps'] = []

            mouse_result['chunks'][chunk_name] = fit_result

        elapsed = time.time() - mouse_start
        print(f"    Done in {elapsed:.1f}s")

        mouse_results[subject_id] = mouse_result

        # Save checkpoint after each mouse
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(mouse_results, f)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"PHASE 2 COMPLETE — {total_elapsed/60:.1f} minutes")
    print(f"{'='*70}")

    # ── Build summary CSV ──
    summary_rows = []
    for subject_id, result in mouse_results.items():
        row = {'subject': subject_id}
        n_chunks_fit = 0

        for chunk_name in ['chunk1', 'chunk2', 'chunk3']:
            chunk_result = result['chunks'].get(chunk_name, {})

            if chunk_result.get('success', False):
                n_chunks_fit += 1
                for weight in WEIGHT_NAMES:
                    row[f'{chunk_name}_sigma_trial_{weight}'] = \
                        chunk_result['sigma_trial'].get(weight, np.nan)
                    row[f'{chunk_name}_sigma_day_{weight}'] = \
                        chunk_result['sigma_day'].get(weight, np.nan)
                row[f'{chunk_name}_trials'] = chunk_result.get('n_trials', 0)
                row[f'{chunk_name}_days'] = chunk_result.get('n_days', 0)
                row[f'{chunk_name}_evidence'] = chunk_result.get('evidence', np.nan)
            else:
                for weight in WEIGHT_NAMES:
                    row[f'{chunk_name}_sigma_trial_{weight}'] = np.nan
                    row[f'{chunk_name}_sigma_day_{weight}'] = np.nan
                row[f'{chunk_name}_trials'] = 0
                row[f'{chunk_name}_days'] = 0
                row[f'{chunk_name}_evidence'] = np.nan

        row['n_chunks_fit'] = n_chunks_fit
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Add lab column (prefix of subject ID)
    summary_df['lab'] = summary_df['subject'].apply(
        lambda s: s.rsplit('_', 1)[0] if '_' in s else s)

    # ── Save outputs ──
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    # Full results pickle
    output = {'mouse_results': mouse_results}
    with open(PSYTRACK_RESULTS_FILE, 'wb') as f:
        pickle.dump(output, f)
    print(f"\n  Saved: {PSYTRACK_RESULTS_FILE}")

    # Summary CSV
    summary_df.to_csv(SUMMARY_CSV_FILE, index=False)
    print(f"  Saved: {SUMMARY_CSV_FILE}")
    print(f"    {len(summary_df)} mice, "
          f"{(summary_df['n_chunks_fit'] == 3).sum()} with complete 3-chunk data")

    # Overnight jumps CSV
    jumps_df = pd.DataFrame(all_jumps)
    jumps_df.to_csv(OVERNIGHT_JUMPS_FILE, index=False)
    print(f"  Saved: {OVERNIGHT_JUMPS_FILE} ({len(jumps_df)} transitions)")

    # Clean up checkpoint
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)

    return output


if __name__ == '__main__':
    run_phase2()
