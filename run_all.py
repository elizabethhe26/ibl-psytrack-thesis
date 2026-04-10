#!/usr/bin/env python3
"""
Reproduce all figures and tables from:

    He, E. (2026). Learning Rate Dynamics Across Training Phases in
    Mouse Visual Decision-Making: A Population-Level PsyTrack Analysis.
    Princeton University Senior Thesis.

Usage:
    python run_all.py                  # Run full pipeline (skips phases with existing outputs)
    python run_all.py --phase 1        # Phase 1 only: data preparation
    python run_all.py --phase 2        # Phase 2 only: model fitting (~2-4 hours)
    python run_all.py --phase 3        # Phase 3 only: statistics & figures (~5 min)
    python run_all.py --skip-download  # Skip data download step

Skip logic:
    Each phase checks whether its outputs already exist and skips if so.
    To force regeneration, delete the relevant files first:
        rm data/ibl_processed.csv                      # Force re-download + re-process
        rm intermediate/ibl_chunk_exploration.pkl       # Force Phase 1 re-run
        rm intermediate/psytrack_chunk_results.pkl      # Force Phase 2 re-run (~2-4 hours)
        rm intermediate/*                               # Force full re-run from Phase 1

Expected runtime (from scratch):
    Data download: ~5 minutes (228 MB from Figshare)
    Phase 1: ~2 minutes
    Phase 2: ~2-4 hours (PsyTrack fitting for ~54 mice × 3 chunks)
    Phase 3: ~5 minutes
"""
import os
import sys
import argparse
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (DATA_DIR, RAW_DOWNLOAD_DIR, INTERMEDIATE_DIR, FIGURE_DIR,
                    TABLE_DIR, LATEX_DIR, PROCESSED_DATA_FILE, CHUNK_EXPLORATION_FILE,
                    PSYTRACK_RESULTS_FILE, SUMMARY_CSV_FILE, OVERNIGHT_JUMPS_FILE)


def create_directories():
    """Create all output directories if they don't exist."""
    for d in [DATA_DIR, RAW_DOWNLOAD_DIR, INTERMEDIATE_DIR,
              FIGURE_DIR, TABLE_DIR, LATEX_DIR]:
        os.makedirs(d, exist_ok=True)


def run_pipeline(phase=None, skip_download=False):
    """Run the reproduction pipeline.

    Args:
        phase: int or None. If None, run all phases (skipping those with
               existing outputs).
                1 = data preparation only
                2 = model fitting only
                3 = statistical analysis only
        skip_download: if True, skip the data download step
    """
    total_start = time.time()

    print()
    print("=" * 70)
    print("  He (2026) Senior Thesis — Full Reproduction Pipeline")
    print("=" * 70)
    print()

    create_directories()

    # ── Data download and preparation ──
    if phase is None or phase == 1:
        if os.path.exists(PROCESSED_DATA_FILE):
            print(f"  Data: {PROCESSED_DATA_FILE} already exists, skipping download.")
            print(f"    To regenerate: rm {PROCESSED_DATA_FILE}")
        elif skip_download:
            print("ERROR: Data file not found and --skip-download was set:")
            print(f"  {PROCESSED_DATA_FILE}")
            sys.exit(1)
        else:
            from src.data_download import download_and_prepare
            download_and_prepare()

    # ── Phase 1: Data preparation ──
    if phase is None or phase == 1:
        if not os.path.exists(PROCESSED_DATA_FILE):
            print("ERROR: Data file not found. Run without --phase to download.")
            print(f"  Expected: {PROCESSED_DATA_FILE}")
            sys.exit(1)
        if os.path.exists(CHUNK_EXPLORATION_FILE):
            print(f"\n  Phase 1: {CHUNK_EXPLORATION_FILE} already exists, skipping.")
            print(f"    To regenerate: rm {CHUNK_EXPLORATION_FILE}")
        else:
            from src.phase1_data_preparation import run_phase1
            run_phase1()

    # ── Phase 2: Model fitting ──
    if phase is None or phase == 2:
        if not os.path.exists(CHUNK_EXPLORATION_FILE):
            print("ERROR: Phase 1 output not found.")
            print(f"  Expected: {CHUNK_EXPLORATION_FILE}")
            print("  Run Phase 1 first: python run_all.py --phase 1")
            sys.exit(1)
        if (os.path.exists(PSYTRACK_RESULTS_FILE) and
                os.path.exists(SUMMARY_CSV_FILE) and
                os.path.exists(OVERNIGHT_JUMPS_FILE)):
            print(f"\n  Phase 2: Intermediate files already exist, skipping.")
            print(f"    {PSYTRACK_RESULTS_FILE}")
            print(f"    {SUMMARY_CSV_FILE}")
            print(f"    {OVERNIGHT_JUMPS_FILE}")
            print(f"    To regenerate: rm intermediate/psytrack_chunk_results.pkl "
                  f"intermediate/psytrack_chunk_summary.csv intermediate/overnight_jumps.csv")
        else:
            from src.phase2_model_fitting import run_phase2
            run_phase2()

    # ── Phase 3: Statistical analysis and figures ──
    if phase is None or phase == 3:
        if not os.path.exists(SUMMARY_CSV_FILE) or not os.path.exists(OVERNIGHT_JUMPS_FILE):
            print("ERROR: Phase 2 output not found.")
            print(f"  Expected: {SUMMARY_CSV_FILE}")
            print(f"           {OVERNIGHT_JUMPS_FILE}")
            print("  Run Phase 2 first: python run_all.py --phase 2")
            sys.exit(1)

        # Main analysis (Figures 2-15, Tables 6-12)
        from src.phase3_statistical_analysis import run_phase3
        run_phase3()

        # High-contrast control (Figure 16)
        # Internal skip logic handles refitting from cache;
        # figure is always regenerated.
        from src.phase3g_high_contrast_control import run_high_contrast_control
        run_high_contrast_control()

        # Phenotype validation (Figure 25, Table 14)
        from src.phase3h_phenotype_validation import run_phenotype_validation
        run_phenotype_validation()

    total_elapsed = time.time() - total_start
    print()
    print("=" * 70)
    print(f"  PIPELINE COMPLETE — Total time: {total_elapsed/60:.1f} minutes")
    print("=" * 70)
    print(f"\n  Figures: {FIGURE_DIR}/")
    print(f"  Tables:  {TABLE_DIR}/")
    print(f"  LaTeX:   {LATEX_DIR}/")
    print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Reproduce He (2026) senior thesis figures and tables.')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3],
                        help='Run only this phase (1=data, 2=fitting, 3=analysis)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Skip data download step')
    args = parser.parse_args()

    run_pipeline(phase=args.phase, skip_download=args.skip_download)
