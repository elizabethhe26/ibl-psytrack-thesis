"""Download and assemble the IBL behavioral dataset.

Replicates the exact data loading and cleaning from Roy et al. (2021),
PsyTrack Manuscript Figures notebook, Steps 2-3.

Key filtering rules (from Roy's code):
    Step 2 — Session inclusion:
        A session is only included if ALL 5 required .npy files exist:
            _ibl_trials.choice.npy
            _ibl_trials.contrastLeft.npy
            _ibl_trials.contrastRight.npy
            _ibl_trials.feedbackType.npy
            _ibl_trials.probabilityLeft.npy
        Sessions missing ANY of these are silently skipped.

    Step 3 — Trial-level corrections:
        1. Drop mistrials: choice == 0 (timeout / no response)
        2. Drop feedbackType == 0 (3 anomalous ZM_1084 trials)
        3. CSHL_002: 81 trials have negative contrastRight → abs()
        4. NaN contrasts → 0
        5. Convert choice {-1,+1} → {0,1}
        6. Convert feedbackType {-1,+1} → {0,1}

Source: International Brain Laboratory (2019).
    https://doi.org/10.6084/m9.figshare.11636748
"""
import os
import sys
import io
import zipfile
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import PROCESSED_DATA_FILE, FIGSHARE_URL, RAW_DOWNLOAD_DIR

# These must ALL exist for a session to be included (Roy Step 2, line 47)
REQUIRED_VARS = [
    '_ibl_trials.choice',
    '_ibl_trials.contrastLeft',
    '_ibl_trials.contrastRight',
    '_ibl_trials.feedbackType',
    '_ibl_trials.probabilityLeft',
]


def download_from_figshare(url, output_dir):
    """Download the IBL dataset zip from Figshare."""
    os.makedirs(output_dir, exist_ok=True)
    zip_path = os.path.join(output_dir, 'ibl_download.zip')

    if os.path.exists(zip_path):
        print(f"  Zip already downloaded: {zip_path}")
        return zip_path

    print(f"Downloading IBL dataset from Figshare...")
    print(f"  URL: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get('content-length', 0))
    downloaded = 0

    with open(zip_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = 100 * downloaded / total_size
                print(f"\r  Progress: {pct:.1f}% "
                      f"({downloaded / 1e6:.1f} / {total_size / 1e6:.1f} MB)",
                      end='', flush=True)
    print()
    return zip_path


def load_npy_from_zip(zf, filepath):
    """Load a .npy file from inside a zip archive."""
    try:
        with zf.open(filepath) as f:
            return np.load(io.BytesIO(f.read()), allow_pickle=True)
    except (KeyError, ValueError, Exception):
        return None


def assemble_csv_from_zip(zip_path):
    """Walk the zip archive and assemble all sessions into a DataFrame.

    Matches Roy et al. (2021) Step 2: only includes sessions where ALL 5
    required .npy files exist. Sessions missing any file are skipped.
    """
    print("Assembling CSV from numpy archive...")
    print("  (Only sessions with all 5 required trial files are included)")

    all_dfs = []
    sessions_included = 0
    sessions_skipped_missing_files = 0
    sessions_skipped_other = 0

    with zipfile.ZipFile(zip_path, 'r') as zf:
        all_names_set = set(zf.namelist())

        # Find candidate sessions by locating _ibl_trials.choice.npy
        choice_files = [n for n in all_names_set
                        if n.endswith('_ibl_trials.choice.npy')]
        print(f"  Found {len(choice_files)} candidate sessions")

        for i, choice_path in enumerate(choice_files):
            if (i + 1) % 500 == 0:
                print(f"  Processing {i+1}/{len(choice_files)}...", flush=True)

            # Parse path
            parts = choice_path.split('/')
            try:
                subj_idx = parts.index('Subjects')
                lab = parts[subj_idx - 1]
                subject = parts[subj_idx + 1]
                date_str = parts[subj_idx + 2]
                session_num = int(parts[subj_idx + 3])
            except (ValueError, IndexError):
                sessions_skipped_other += 1
                continue

            alf_dir = '/'.join(parts[:-1]) + '/'

            # ── KEY FILTER: require ALL 5 .npy files (Roy Step 2, line 47) ──
            all_present = all(
                (alf_dir + var + '.npy') in all_names_set
                for var in REQUIRED_VARS
            )
            if not all_present:
                sessions_skipped_missing_files += 1
                continue

            # Load all required arrays
            choice = load_npy_from_zip(zf, alf_dir + '_ibl_trials.choice.npy')
            contrast_left = load_npy_from_zip(zf, alf_dir + '_ibl_trials.contrastLeft.npy')
            contrast_right = load_npy_from_zip(zf, alf_dir + '_ibl_trials.contrastRight.npy')
            feedback = load_npy_from_zip(zf, alf_dir + '_ibl_trials.feedbackType.npy')
            prob_left = load_npy_from_zip(zf, alf_dir + '_ibl_trials.probabilityLeft.npy')

            if choice is None or len(choice) == 0:
                sessions_skipped_other += 1
                continue

            n_trials = len(choice)
            session_df = pd.DataFrame({
                'subject': subject,
                'lab': lab,
                'date': date_str,
                'session': session_num,
                'contrastLeft': contrast_left.flatten()[:n_trials],
                'contrastRight': contrast_right.flatten()[:n_trials],
                'choice': choice.flatten()[:n_trials],
                'feedbackType': feedback.flatten()[:n_trials],
                'probabilityLeft': prob_left.flatten()[:n_trials],
            })

            all_dfs.append(session_df)
            sessions_included += 1

    print(f"  Sessions included:              {sessions_included}")
    print(f"  Sessions skipped (missing files): {sessions_skipped_missing_files}")
    print(f"  Sessions skipped (other):        {sessions_skipped_other}")

    df = pd.concat(all_dfs, ignore_index=True)
    return df


def apply_corrections(df):
    """Apply Roy et al. (2021) Step 3 data corrections.

    These are known anomalies in the December 2019 IBL data release.
    """
    n_before = len(df)
    mice_before = df['subject'].nunique()
    print("\nApplying data corrections (Roy et al. 2021, Step 3)...")

    # 0. Exclude mice not in the thesis analytic dataset.
    #    The Figshare archive contains 101 mice, but the original analysis
    #    used IBL's ONE Light library to build the session table, which
    #    excluded 9 mice that lacked complete behavioral data or were
    #    ephys-only subjects. The 92-mouse list matches Table 1 of the
    #    thesis and was verified against the original ibl_processed.csv.
    EXCLUDED_MICE = [
        'KS003', 'KS018', 'KS025',                          # 3 cortexlab
        'SWC_001', 'SWC_012', 'SWC_013', 'SWC_014',         # 6 mrsicflogellab
        'SWC_015', 'SWC_021',
    ]
    n_excluded_mice = df['subject'].isin(EXCLUDED_MICE).sum()
    mice_excluded_names = df.loc[df['subject'].isin(EXCLUDED_MICE), 'subject'].unique()
    df = df[~df['subject'].isin(EXCLUDED_MICE)].copy()
    print(f"  Excluded {len(mice_excluded_names)} mice not in thesis dataset "
          f"({n_excluded_mice:,} trials): {', '.join(sorted(mice_excluded_names))}")

    # 1. Drop mistrials: choice == 0 (timeout / no response)
    n_mistrials = (df['choice'] == 0).sum()
    df = df[df['choice'] != 0].copy()
    print(f"  Dropped {n_mistrials:,} mistrials (choice == 0)")

    # 2. Drop feedbackType == 0 (ZM_1084 anomaly + any others)
    n_bad_fb = (df['feedbackType'] == 0).sum()
    if n_bad_fb > 0:
        df = df[df['feedbackType'] != 0].copy()
        print(f"  Dropped {n_bad_fb} trials with feedbackType == 0")

    # 3. Fix negative contrasts (CSHL_002 anomaly)
    #    Roy's code (lines 45-46): negative contrastRight means left contrast
    #    was miscoded as negative right. Move abs value to contrastLeft, zero
    #    contrastRight.
    neg_right = df['contrastRight'] < 0
    if neg_right.sum() > 0:
        df.loc[neg_right, 'contrastLeft'] = df.loc[neg_right, 'contrastRight'].abs()
        df.loc[neg_right, 'contrastRight'] = 0
        print(f"  Corrected {neg_right.sum()} trials with negative contrastRight "
              f"(moved to contrastLeft, zeroed contrastRight)")

    # 4. NaN contrasts → 0
    df['contrastLeft'] = df['contrastLeft'].fillna(0)
    df['contrastRight'] = df['contrastRight'].fillna(0)

    # 5. KS005: ONE Light excluded 32 sessions that are present in the
    #    Figshare archive (habituation sessions, early training before the
    #    standard task, and late probe sessions). The original ibl_processed.csv
    #    has 45 sessions for KS005; the Figshare archive has 77. We restrict
    #    to the date range present in the original dataset.
    ks005_mask = df['subject'] == 'KS005'
    if ks005_mask.sum() > 0:
        ks005_dates_original = [
            '2019-03-18', '2019-03-25', '2019-03-26', '2019-03-30',
            '2019-04-02', '2019-04-03', '2019-04-04', '2019-04-05',
            '2019-04-08', '2019-04-09', '2019-04-11', '2019-04-26',
            '2019-04-30', '2019-05-03', '2019-05-07', '2019-05-08',
            '2019-05-10', '2019-05-13', '2019-05-16', '2019-05-20',
            '2019-05-21', '2019-05-28', '2019-05-29', '2019-05-30',
            '2019-05-31', '2019-06-03', '2019-06-04', '2019-06-05',
            '2019-06-07', '2019-06-10', '2019-06-11', '2019-06-13',
            '2019-06-17', '2019-06-21', '2019-06-26', '2019-06-27',
            '2019-07-23', '2019-07-24', '2019-07-25', '2019-07-26',
            '2019-07-30', '2019-08-06', '2019-08-08', '2019-08-12',
            '2019-08-16',
        ]
        ks005_date_col = pd.to_datetime(df.loc[ks005_mask, 'date'])
        ks005_valid = ks005_date_col.dt.strftime('%Y-%m-%d').isin(ks005_dates_original)
        n_ks005_removed = ks005_mask.sum() - ks005_valid.sum()
        # Build full mask: keep non-KS005 rows + valid KS005 rows
        keep_mask = ~ks005_mask  # all non-KS005
        keep_mask.loc[ks005_valid[ks005_valid].index] = True
        df = df[keep_mask].copy()
        if n_ks005_removed > 0:
            print(f"  KS005: excluded {n_ks005_removed:,} trials from "
                  f"sessions not in original dataset")

    # Remove mice that have zero trials after filtering
    trial_counts = df.groupby('subject').size()
    empty_mice = trial_counts[trial_counts == 0].index.tolist()
    if empty_mice:
        df = df[~df['subject'].isin(empty_mice)]
        print(f"  Removed {len(empty_mice)} mice with 0 trials after filtering")

    mice_after = df['subject'].nunique()
    print(f"  Trials: {n_before:,} → {len(df):,} (removed {n_before - len(df):,})")
    print(f"  Mice:   {mice_before} → {mice_after}")

    return df


def _extract_lab_from_subject(subject_id):
    """Derive lab abbreviation from subject ID.

    The Figshare archive uses internal directory names (e.g., 'zadorlab')
    which group CSH and CSHL together. The thesis uses subject-name
    prefixes, giving 9 distinct labs (Table 1).
    """
    if subject_id.startswith('CSH_ZAD'):
        return 'CSH'
    elif subject_id.startswith('CSHL'):
        return 'CSHL'
    elif subject_id.startswith('DY_'):
        return 'DY'
    elif subject_id.startswith('IBL'):
        return 'IBL'
    elif subject_id.startswith('KS'):
        return 'KS'
    elif subject_id.startswith('NYU'):
        return 'NYU'
    elif subject_id.startswith('SWC'):
        return 'SWC'
    elif subject_id.startswith('ZM_'):
        return 'ZM'
    elif subject_id.startswith('ibl_witten'):
        return 'ibl_witten'
    else:
        return 'Unknown'


def standardize_columns(df):
    """Convert IBL raw encodings to pipeline standard format.

    IBL raw: choice {-1 = left, +1 = right}, feedbackType {-1 = error, +1 = correct}

    Colab convention (matching the thesis notebooks):
        choice:   +1 (right) → 0,  -1 (left) → 1
        rewarded: -1 (error) → 0,  +1 (correct) → 1

    Note on choice encoding: PsyTrack models P(y=1) using a sigmoid,
    internally labeling it "pR" (probability Right). With our encoding
    y=1 = left choice, the model's "P(Right)" actually tracks P(Left).
    This inverts all weight signs relative to the Roy et al. (2021)
    natural convention, but is the encoding used in all thesis Colab
    notebooks and produces the weight trajectories shown in the thesis
    figures. The σ hyperparameters (the thesis's primary finding) are
    unaffected because they measure weight volatility, not direction.

    Also derives lab from subject name prefix (9 labs) rather than
    the Figshare directory name (8 labs), because the thesis treats
    CSH and CSHL as separate lab identifiers.
    """
    # Choice: +1 (right) → 0, -1 (left) → 1  [Colab/thesis convention]
    df['choice'] = df['choice'].map({1: 0, -1: 1}).astype(int)

    # Rewarded: -1 (error) → 0, +1 (correct) → 1
    df['rewarded'] = df['feedbackType'].map({-1: 0, 1: 1}).astype(int)

    # Rename contrast columns
    df = df.rename(columns={
        'contrastLeft': 'contrast_left',
        'contrastRight': 'contrast_right',
    })

    # Derive lab from subject name prefix (matches thesis Table 1)
    df['lab'] = df['subject'].apply(_extract_lab_from_subject)

    # Convert date
    df['date'] = pd.to_datetime(df['date'])

    # Keep pipeline columns
    df = df[['subject', 'lab', 'date', 'session',
             'contrast_left', 'contrast_right', 'choice', 'rewarded',
             'probabilityLeft']].copy()

    return df


def download_and_prepare():
    """Main entry point: download, assemble, correct, standardize, save."""
    if os.path.exists(PROCESSED_DATA_FILE):
        print(f"Pre-processed data already exists: {PROCESSED_DATA_FILE}")
        df = pd.read_csv(PROCESSED_DATA_FILE)
        print(f"  {len(df):,} trials, {df['subject'].nunique()} mice")
        return PROCESSED_DATA_FILE

    # Download
    zip_path = download_from_figshare(FIGSHARE_URL, RAW_DOWNLOAD_DIR)

    # Assemble (only sessions with all 5 required files)
    df = assemble_csv_from_zip(zip_path)

    # Apply Roy et al. corrections
    df = apply_corrections(df)

    # Standardize
    df = standardize_columns(df)

    # Verification
    print(f"\n  Verification:")
    print(f"    Total trials:  {len(df):,}")
    print(f"    Total mice:    {df['subject'].nunique()}")
    print(f"    Total labs:    {df['lab'].nunique()}")
    print(f"    Choice values: {sorted(df['choice'].unique())}")
    print(f"    Reward values: {sorted(df['rewarded'].unique())}")
    print(f"    Date range:    {df['date'].min()} — {df['date'].max()}")

    # Save
    os.makedirs(os.path.dirname(PROCESSED_DATA_FILE), exist_ok=True)
    df.to_csv(PROCESSED_DATA_FILE, index=False)
    print(f"\n  Saved: {PROCESSED_DATA_FILE}")

    return PROCESSED_DATA_FILE


if __name__ == '__main__':
    download_and_prepare()
