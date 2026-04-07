"""Phase 1: Data Preparation — IBL Chunk Exploration.

Processes the raw IBL CSV into a structured pickle file containing
per-mouse chunk boundaries and trial data. This is the data foundation
for all subsequent analysis.

Chunk definitions (basic task only, excludes bias blocks):
    C1 (Early Training):       Session 1 to session before 25% introduction.
                               Contains only 100% and 50% contrast.
    C2 (Curriculum Expansion): 25% introduction session to session before
                               0% or 6.25% introduction.
    C3 (Full Contrast Set):    0%/6.25% introduction session through end
                               of basic task (before bias blocks begin).

Output: ibl_chunk_exploration.pkl
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import (RAW_DATA_FILE, CHUNK_EXPLORATION_FILE, INTERMEDIATE_DIR,
                    MIN_TRIALS_OK, MIN_TRIALS_WARNING, MIN_TRIALS_CRITICAL,
                    MIN_BLOCK_LENGTH, CONTRAST_NAMES)


# ─── Session identification ──────────────────────────────────────────────────

def get_unique_sessions(mouse_df):
    """Get unique sessions as (date, session) pairs, chronologically ordered.

    The IBL data can have multiple sessions per date. We identify each
    unique session by its (date, session_number) pair.

    Returns:
        list of (date, session_num) tuples, sorted chronologically
    """
    session_keys = (mouse_df.groupby(['date', 'session'])
                    .size()
                    .reset_index()[['date', 'session']])
    session_keys = session_keys.sort_values(['date', 'session'])
    return list(zip(session_keys['date'], session_keys['session']))


def get_trials_for_session(mouse_df, date, session_num):
    """Get all trials for a specific (date, session) pair."""
    mask = (mouse_df['date'] == date) & (mouse_df['session'] == session_num)
    return mouse_df[mask]


# ─── Contrast analysis ───────────────────────────────────────────────────────

def get_trial_contrasts(row):
    """Get all contrast values present in a single trial.

    A trial has stimuli on left, right, or both sides.
    Returns the set of contrast levels present.
    """
    contrasts = set()
    if row['contrast_left'] > 0:
        contrasts.add(round(row['contrast_left'], 4))
    if row['contrast_right'] > 0:
        contrasts.add(round(row['contrast_right'], 4))
    if row['contrast_left'] == 0 and row['contrast_right'] == 0:
        contrasts.add(0)  # 0% contrast trial
    return contrasts


def find_contrast_introduction_sessions(mouse_df, sessions):
    """Find the first session where each contrast level appears.

    Returns:
        dict mapping contrast_value → 1-indexed session number
    """
    introduction_sessions = {}
    for session_idx, (date, session_num) in enumerate(sessions):
        session_df = get_trials_for_session(mouse_df, date, session_num)
        for _, row in session_df.iterrows():
            for contrast in get_trial_contrasts(row):
                if contrast not in introduction_sessions:
                    introduction_sessions[contrast] = session_idx + 1  # 1-indexed
    return introduction_sessions


# ─── Bias block detection ────────────────────────────────────────────────────

def find_max_consecutive_run(values, target, tolerance=0.01):
    """Find maximum length of consecutive values matching target."""
    max_run = 0
    current_run = 0
    for v in values:
        if abs(v - target) < tolerance:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def analyze_probability_blocks(prob_values):
    """Analyze the block structure of probabilityLeft values.

    Full task sessions have STRUCTURED BLOCKS of ~90 consecutive trials
    at p=0.2 or p=0.8. Basic task sessions have SCATTERED values from
    trial-by-trial counterbalancing.

    Returns:
        dict with max_run lengths and is_bias_block flag
    """
    if len(prob_values) == 0:
        return {'max_run_02': 0, 'max_run_08': 0, 'max_run_05': 0,
                'unique_values': set(), 'is_bias_block': False}

    max_run_02 = find_max_consecutive_run(prob_values, 0.2)
    max_run_08 = find_max_consecutive_run(prob_values, 0.8)
    max_run_05 = find_max_consecutive_run(prob_values, 0.5)
    unique_values = set(np.round(prob_values, 2))

    # Bias blocks detected if there's a LONG consecutive run at p=0.2 OR p=0.8
    is_bias_block = (max_run_02 >= MIN_BLOCK_LENGTH) or (max_run_08 >= MIN_BLOCK_LENGTH)

    return {
        'max_run_02': max_run_02,
        'max_run_08': max_run_08,
        'max_run_05': max_run_05,
        'unique_values': unique_values,
        'is_bias_block': is_bias_block,
    }


def find_bias_block_introduction(mouse_df, sessions):
    """Find the first session where full-task bias blocks appear.

    Returns:
        1-indexed session number where bias blocks start, or None
    """
    prob_col = 'probabilityLeft' if 'probabilityLeft' in mouse_df.columns else None
    if prob_col is None:
        return None

    for session_idx, (date, session_num) in enumerate(sessions):
        session_df = get_trials_for_session(mouse_df, date, session_num)
        if len(session_df) == 0:
            continue

        prob_values = session_df[prob_col].values
        block_info = analyze_probability_blocks(prob_values)

        if block_info['is_bias_block']:
            return session_idx + 1  # 1-indexed

    return None  # No bias blocks detected — all data is basic task


# ─── Chunk boundary detection ────────────────────────────────────────────────

def get_chunk_stats(mouse_df, sessions, start, end):
    """Compute trial count and session count for a chunk.

    Args:
        start: 1-indexed start session
        end: 1-indexed end session (inclusive)

    Returns:
        (n_trials, n_sessions, contrasts_present, status)
    """
    if not start or not end or end < start:
        return 0, 0, set(), 'MISSING'

    # Collect trials from all sessions in range
    n_trials = 0
    contrasts = set()
    n_sessions = end - start + 1

    for session_idx in range(start - 1, end):  # Convert to 0-indexed
        if session_idx >= len(sessions):
            break
        date, session_num = sessions[session_idx]
        session_df = get_trials_for_session(mouse_df, date, session_num)
        n_trials += len(session_df)
        for _, row in session_df.iterrows():
            contrasts.update(get_trial_contrasts(row))

    # Quality status
    if n_trials >= MIN_TRIALS_OK:
        status = 'OK'
    elif n_trials >= MIN_TRIALS_WARNING:
        status = 'WARNING'
    elif n_trials > 0:
        status = 'CRITICAL'
    else:
        status = 'MISSING'

    return n_trials, n_sessions, contrasts, status


def detect_chunk_boundaries(mouse_df, subject_id):
    """Detect chunk boundaries for a single mouse.

    This is the core algorithm. It identifies transition points based on
    when new contrast levels are introduced, then defines three chunks.

    Returns:
        dict with chunk boundary info, or None if detection fails
    """
    mouse_df = mouse_df.sort_values(['date', 'session']).reset_index(drop=True)
    sessions = get_unique_sessions(mouse_df)
    n_sessions = len(sessions)

    if n_sessions == 0:
        return None

    # Find key transition points
    intro = find_contrast_introduction_sessions(mouse_df, sessions)

    session_25_intro = intro.get(0.25)
    session_0_intro = intro.get(0)
    session_0625_intro = intro.get(0.0625)

    # Find when bias blocks start (end of basic task)
    session_bias_intro = find_bias_block_introduction(mouse_df, sessions)

    # Basic task end: before bias blocks, or end of all data
    if session_bias_intro:
        basic_task_end = session_bias_intro - 1
    else:
        basic_task_end = n_sessions

    # ── Define Chunk Boundaries ──

    # C1: Session 1 to session before 25% introduction
    c1_start = 1
    c1_end = (session_25_intro - 1) if session_25_intro and session_25_intro > 1 else None

    # C2: 25% introduction to session before 0%/6.25% introduction
    c2_start = session_25_intro if session_25_intro else None

    # C3 starts at whichever comes first: 0% or 6.25% introduction
    c3_trigger = None
    if session_0_intro and session_0625_intro:
        c3_trigger = min(session_0_intro, session_0625_intro)
    elif session_0_intro:
        c3_trigger = session_0_intro
    elif session_0625_intro:
        c3_trigger = session_0625_intro

    c2_end = (c3_trigger - 1) if c3_trigger and c2_start and c3_trigger > c2_start else c2_start

    # C3: from 0%/6.25% introduction to basic task end
    c3_start = c3_trigger if c3_trigger else None
    c3_end = basic_task_end if c3_start else None

    # Handle edge cases
    if c3_end and c3_start and c3_end < c3_start:
        c3_end = c3_start  # At least one session

    # Compute chunk statistics
    c1_trials, c1_sess, c1_contrasts, c1_status = get_chunk_stats(
        mouse_df, sessions, c1_start, c1_end)
    c2_trials, c2_sess, c2_contrasts, c2_status = get_chunk_stats(
        mouse_df, sessions, c2_start, c2_end)
    c3_trials, c3_sess, c3_contrasts, c3_status = get_chunk_stats(
        mouse_df, sessions, c3_start, c3_end)

    return {
        'subject': subject_id,
        'total_trials': len(mouse_df),
        'total_sessions': n_sessions,
        'sessions': sessions,

        'contrast_intro': intro,
        'session_bias_intro': session_bias_intro,
        'basic_task_end': basic_task_end,

        'chunk1': {
            'name': 'Early Training',
            'description': '[100%, 50%] only',
            'start_session': c1_start,
            'end_session': c1_end,
            'trials': c1_trials,
            'n_sessions': c1_sess,
            'contrasts': c1_contrasts,
            'status': c1_status,
        },
        'chunk2': {
            'name': 'Curriculum Expansion',
            'description': '+25%, +12.5% (before 0%/6.25%)',
            'start_session': c2_start,
            'end_session': c2_end,
            'trials': c2_trials,
            'n_sessions': c2_sess,
            'contrasts': c2_contrasts,
            'status': c2_status,
        },
        'chunk3': {
            'name': 'Full Contrast Set',
            'description': '+0%, +6.25%, -50% (basic task)',
            'start_session': c3_start,
            'end_session': c3_end,
            'trials': c3_trials,
            'n_sessions': c3_sess,
            'contrasts': c3_contrasts,
            'status': c3_status,
        },
    }


# ─── Trial data extraction ───────────────────────────────────────────────────

def extract_chunk_trials(mouse_df, sessions, start_session, end_session):
    """Extract trial-level data for a specific chunk.

    Args:
        mouse_df: full mouse DataFrame
        sessions: list of (date, session) tuples
        start_session: 1-indexed start
        end_session: 1-indexed end (inclusive)

    Returns:
        DataFrame with chunk trial data, or empty DataFrame
    """
    if not start_session or not end_session or end_session < start_session:
        return pd.DataFrame()

    start_idx = start_session - 1
    end_idx = end_session
    session_range = sessions[start_idx:end_idx]

    mask = pd.Series([False] * len(mouse_df), index=mouse_df.index)
    for date, session_num in session_range:
        session_mask = (mouse_df['date'] == date) & (mouse_df['session'] == session_num)
        mask = mask | session_mask

    subset = mouse_df[mask].copy()

    if len(subset) > 0:
        subset = subset.sort_values(['date', 'session']).reset_index(drop=True)

        # Add within-chunk session numbering
        session_keys = subset.groupby(['date', 'session']).ngroup()
        subset['chunk_session'] = session_keys + 1

        # Add within-chunk trial numbering
        subset['chunk_trial'] = range(1, len(subset) + 1)

    return subset


# ─── Mouse selection / tiering ───────────────────────────────────────────────

def classify_mouse_tier(result):
    """Assign a selection tier based on chunk data quality.

    Tier 1: All 3 chunks have ≥1000 trials (ideal for PsyTrack)
    Tier 2: All chunks have ≥500 trials, some <1000
    Excluded: Any chunk <500 trials or missing
    """
    statuses = [
        result['chunk1']['status'],
        result['chunk2']['status'],
        result['chunk3']['status'],
    ]

    if all(s == 'OK' for s in statuses):
        return 'tier1'
    elif all(s in ('OK', 'WARNING') for s in statuses):
        return 'tier2'
    else:
        return 'excluded'


# ─── Main pipeline ───────────────────────────────────────────────────────────

def run_phase1():
    """Run Phase 1: process all mice, detect chunks, extract trial data.

    Returns:
        dict: the complete exploration results (also saved to pickle)
    """
    print("=" * 70)
    print("PHASE 1: Data Preparation — IBL Chunk Exploration")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from: {RAW_DATA_FILE}")
    df = pd.read_csv(RAW_DATA_FILE)
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    subject_col = 'subject' if 'subject' in df.columns else 'Subject'
    subjects = sorted(df[subject_col].unique())
    print(f"  {len(df):,} trials from {len(subjects)} mice")

    # Process each mouse
    all_results = []
    mouse_data = {}

    for i, subject_id in enumerate(subjects):
        print(f"\n  [{i+1}/{len(subjects)}] {subject_id}...", end=' ')

        mouse_df = df[df[subject_col] == subject_id].copy()
        result = detect_chunk_boundaries(mouse_df, subject_id)

        if result is None:
            print("SKIPPED (no data)")
            continue

        # Extract trial data for each chunk
        trial_data = {}
        for chunk_name in ['chunk1', 'chunk2', 'chunk3']:
            chunk_info = result[chunk_name]
            trial_data[chunk_name] = extract_chunk_trials(
                mouse_df, result['sessions'],
                chunk_info['start_session'],
                chunk_info['end_session']
            )

        c1t = result['chunk1']['trials']
        c2t = result['chunk2']['trials']
        c3t = result['chunk3']['trials']
        print(f"C1={c1t} C2={c2t} C3={c3t} "
              f"[{result['chunk1']['status']}/{result['chunk2']['status']}/{result['chunk3']['status']}]")

        all_results.append(result)
        mouse_data[subject_id] = {
            'info': result,
            'chunks': {k: result[k] for k in ['chunk1', 'chunk2', 'chunk3']},
            'trial_data': trial_data,
        }

    # Tier classification
    tier1 = [r['subject'] for r in all_results if classify_mouse_tier(r) == 'tier1']
    tier2 = [r['subject'] for r in all_results if classify_mouse_tier(r) == 'tier2']
    excluded = [r['subject'] for r in all_results if classify_mouse_tier(r) == 'excluded']

    print(f"\n{'='*70}")
    print(f"PHASE 1 COMPLETE")
    print(f"{'='*70}")
    print(f"  Total mice processed: {len(all_results)}")
    print(f"  Tier 1 (all OK):     {len(tier1)}")
    print(f"  Tier 2 (some WARNING): {len(tier2)}")
    print(f"  Excluded:            {len(excluded)}")

    # Build summary DataFrame
    summary_rows = []
    for r in all_results:
        row = {
            'subject': r['subject'],
            'total_trials': r['total_trials'],
            'total_sessions': r['total_sessions'],
            'basic_task_end': r['basic_task_end'],
            'tier': classify_mouse_tier(r),
        }
        for cn in ['chunk1', 'chunk2', 'chunk3']:
            c = r[cn]
            row[f'{cn}_start'] = c['start_session']
            row[f'{cn}_end'] = c['end_session']
            row[f'{cn}_trials'] = c['trials']
            row[f'{cn}_sessions'] = c['n_sessions']
            row[f'{cn}_status'] = c['status']
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    # Assemble output
    output = {
        'metadata': {
            'n_mice': len(all_results),
            'n_labs': len(set(s.split('_')[0] for s in
                             [r['subject'] for r in all_results] if '_' in s)),
            'analysis_date': datetime.now().isoformat(),
            'source_file': RAW_DATA_FILE,
        },
        'chunk_summary': summary_df,
        'mouse_data': mouse_data,
        'selection': {
            'tier1': tier1,
            'tier2': tier2,
            'excluded': excluded,
            'recommended': tier1 + tier2,
        },
    }

    # Save
    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    with open(CHUNK_EXPLORATION_FILE, 'wb') as f:
        pickle.dump(output, f)
    print(f"\n  Saved: {CHUNK_EXPLORATION_FILE}")

    return output


if __name__ == '__main__':
    run_phase1()
