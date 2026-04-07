"""Download and standardize the IBL behavioral dataset.

Downloads the December 2019 IBL snapshot from Figshare and standardizes
column names for use by the rest of the pipeline.

Source: International Brain Laboratory (2019).
    Data released for: Standardized and reproducible measurement of
    decision-making in mice. Figshare.
    https://doi.org/10.6084/m9.figshare.11636748
"""
import os
import sys
import requests
import zipfile
import pandas as pd

# Add parent directory to path for config import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import RAW_DATA_FILE, FIGSHARE_URL, DATA_DIR


def download_from_figshare(url, output_dir):
    """Download the IBL dataset from Figshare.

    Handles both direct CSV downloads and zip archives.

    Args:
        url: Figshare download URL
        output_dir: directory to save the downloaded file

    Returns:
        path to the downloaded/extracted CSV file
    """
    os.makedirs(output_dir, exist_ok=True)
    download_path = os.path.join(output_dir, 'figshare_download')

    print(f"Downloading IBL dataset from Figshare...")
    print(f"  URL: {url}")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    # Determine file size for progress reporting
    total_size = int(response.headers.get('content-length', 0))
    total_mb = total_size / (1024 * 1024) if total_size else 0

    # Save to disk
    downloaded = 0
    with open(download_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = 100 * downloaded / total_size
                print(f"\r  Progress: {pct:.1f}% ({downloaded / 1e6:.1f} / {total_mb:.1f} MB)",
                      end='', flush=True)
    print()  # newline after progress

    # Check if it's a zip file
    if zipfile.is_zipfile(download_path):
        print("  Download is a zip archive — extracting...")
        with zipfile.ZipFile(download_path, 'r') as zf:
            csv_files = [f for f in zf.namelist() if f.endswith('.csv')]
            if not csv_files:
                raise RuntimeError(f"No CSV files found in zip archive. Contents: {zf.namelist()}")
            # Extract the largest CSV (likely the main data file)
            csv_file = max(csv_files, key=lambda f: zf.getinfo(f).file_size)
            print(f"  Extracting: {csv_file}")
            zf.extract(csv_file, output_dir)
            csv_path = os.path.join(output_dir, csv_file)
        os.remove(download_path)
        return csv_path
    else:
        # Assume it's a CSV directly
        csv_path = download_path + '.csv'
        os.rename(download_path, csv_path)
        return csv_path


def standardize_columns(df):
    """Standardize IBL dataset column names and values.

    The raw IBL data uses camelCase column names and may encode
    feedbackType as {-1, +1} instead of {0, 1}.

    Args:
        df: raw DataFrame from Figshare download

    Returns:
        DataFrame with standardized columns:
            subject, session, date, contrast_left, contrast_right,
            choice, rewarded, probabilityLeft
    """
    # Column name mapping (IBL camelCase → our snake_case)
    rename_map = {
        'contrastLeft': 'contrast_left',
        'contrastRight': 'contrast_right',
        'feedbackType': 'rewarded',
    }
    df = df.rename(columns=rename_map)

    # Handle 'Subject' vs 'subject'
    if 'Subject' in df.columns and 'subject' not in df.columns:
        df = df.rename(columns={'Subject': 'subject'})

    # Verify required columns exist
    required = ['subject', 'session', 'contrast_left', 'contrast_right',
                'choice', 'rewarded']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # Fill NaN contrasts with 0 (0% contrast trials)
    df['contrast_left'] = df['contrast_left'].fillna(0)
    df['contrast_right'] = df['contrast_right'].fillna(0)

    # Ensure choice is 0/1 (not -1/+1)
    if df['choice'].min() < 0:
        # Map: -1 → 0 (left), +1 → 1 (right)
        df['choice'] = (df['choice'] + 1) // 2

    # Ensure rewarded is 0/1 (not -1/+1)
    if df['rewarded'].min() < 0:
        # Map: -1 → 0 (incorrect), +1 → 1 (correct)
        df['rewarded'] = (df['rewarded'] + 1) // 2

    return df


def download_and_prepare():
    """Main entry point: download data if needed, standardize, and save.

    Returns:
        path to the standardized CSV file
    """
    if os.path.exists(RAW_DATA_FILE):
        print(f"Data file already exists: {RAW_DATA_FILE}")
        df = pd.read_csv(RAW_DATA_FILE)
        print(f"  {len(df):,} trials, {df['subject'].nunique()} mice")
        return RAW_DATA_FILE

    # Download
    csv_path = download_from_figshare(FIGSHARE_URL, DATA_DIR)

    # Load and standardize
    print("Loading and standardizing data...")
    df = pd.read_csv(csv_path)
    df = standardize_columns(df)

    # Verification summary
    n_trials = len(df)
    n_mice = df['subject'].nunique()
    n_labs = len(set(s.split('_')[0] for s in df['subject'].unique()
                     if '_' in s))
    print(f"\n  Verification:")
    print(f"    Total trials: {n_trials:,}")
    print(f"    Total mice:   {n_mice}")
    print(f"    Approx labs:  {n_labs}")
    print(f"    Choice range: {df['choice'].min()} - {df['choice'].max()}")
    print(f"    Reward range: {df['rewarded'].min()} - {df['rewarded'].max()}")

    # Save standardized version
    df.to_csv(RAW_DATA_FILE, index=False)
    print(f"\n  Saved standardized data to: {RAW_DATA_FILE}")

    # Clean up intermediate download if different from final path
    if csv_path != RAW_DATA_FILE and os.path.exists(csv_path):
        os.remove(csv_path)

    return RAW_DATA_FILE


if __name__ == '__main__':
    download_and_prepare()
