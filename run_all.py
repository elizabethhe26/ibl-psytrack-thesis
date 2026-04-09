#!/usr/bin/env python3
"""
Reproduce all figures and tables from:

    He, E. (2026). Learning Rate Dynamics Across Training Phases in
    Mouse Visual Decision-Making: A Population-Level PsyTrack Analysis.
    Princeton University Senior Thesis.

Usage:
    python run_all.py                  # Run full pipeline
    python run_all.py --phase 1        # Phase 1 only: data preparation
    python run_all.py --phase 2        # Phase 2 only: model fitting
    python run_all.py --phase 3        # Phase 3 only: statistics & figures
    python run_all.py --skip-download  # Skip data download step

Expected runtime:
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
                    TABLE_DIR, LATEX_DIR, RAW_DATA_FILE, CHUNK_EXPLORATION_FILE,
                    SUMMARY_CSV_FILE, OVERNIGHT_JUMPS_FILE)


def create_directories():
    """Create all output directories if they don't exist."""
    for d in [DATA_DIR, RAW_DOWNLOAD_DIR, INTERMEDIATE_DIR,
              FIGURE_DIR, TABLE_DIR, LATEX_DIR]:
        os.makedirs(d, exist_ok=True)


def run_pipeline(phase=None, skip_download=False):
    """Run the reproduction pipeline.

    Args:
        phase: int or None. If None, run all phases.
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
        if not skip_download and not os.path.exists(RAW_DATA_FILE):
            from src.data_download import download_and_prepare
            download_and_prepare()
        elif os.path.exists(RAW_DATA_FILE):
            from src.data_download import download_and_prepare
            download_and_prepare()  # Prints "Pre-processed data already exists"
        else:
            print("ERROR: Data file not found at:")
            print(f"  {RAW_DATA_FILE}")
            print("Run without --skip-download to download from Figshare.")
            sys.exit(1)

    # ── Phase 1: Data preparation ──
    if phase is None or phase == 1:
        from src.phase1_data_preparation import run_phase1
        run_phase1()

    # ── Phase 2: Model fitting ──
    if phase is None or phase == 2:
        if not os.path.exists(CHUNK_EXPLORATION_FILE):
            print("ERROR: Phase 1 output not found.")
            print("  Run Phase 1 first: python run_all.py --phase 1")
            sys.exit(1)
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

        # Main analysis (Figures 5-22, Tables 5-12)
        from src.phase3_statistical_analysis import run_phase3
        run_phase3()

        # High-contrast control (Figure 16)
        # Internal skip logic in phase3g handles refitting from cache;
        # the figure is always regenerated.
        from src.phase3g_high_contrast_control import run_high_contrast_control
        run_high_contrast_control()

        # Phenotype validation (Figure A6, Table A3)
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
