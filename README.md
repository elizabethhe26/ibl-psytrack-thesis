# Reproducing: Learning Rate Dynamics in Mouse Visual Decision-Making

Reproduction code for:

> **He, E. (2026).** *Learning Rate Dynamics Across Training Phases in Mouse Visual Decision-Making: A Population-Level PsyTrack Analysis.* Princeton University Senior Thesis. Advisor: Professor Jonathan Pillow.

## Quickstart

```bash
git clone https://github.com/elizabethhe26/ibl-psytrack-thesis.git
cd ibl-psytrack-thesis
pip install -r requirements.txt
python run_all.py
```

## Pipeline Overview

The analysis runs in three phases:

```
Phase 1: Data Preparation
    Downloads IBL dataset → detects chunk boundaries → structures trial data
    Generates: Table 1 (lab summary), Table 5 (chunk characteristics)

Phase 2: Model Fitting (several hours)
    Fits PsyTrack 4-weight model to each chunk of each mouse
    Extracts σ_trial and σ_day hyperparameters
    Computes overnight weight jumps

Phase 3: Statistical Analysis
    Generates all thesis figures and statistical tables
    Includes high-contrast control and phenotype validation
```

Run individual phases with `python run_all.py --phase 1|2|3`.

## Data Source

The raw dataset is the December 2019 IBL behavioral data archive, automatically downloaded from Figshare:

> International Brain Laboratory (2019). *Data released for: Standardized and reproducible measurement of decision-making in mice.* Figshare. https://doi.org/10.6084/m9.figshare.11636748

The archive contains per-session numpy files for 101 mice across 9 laboratories. The pipeline applies the data corrections documented in Roy et al. (2021) — dropping mistrials, fixing anomalous contrast values in CSHL_002 and ZM_1084, excluding 9 mice not present in the original processed dataset, and filtering 32 extra sessions for KS005 that were excluded by IBL's ONE Light data pipeline — to produce the 92-mouse, 3,259,282-trial analytic sample used in the thesis.

## Output: Figures

All figures are saved as PDF and PNG in `output/figures/`.

| Thesis Figure | Description | Output File |
|---|---|---|
| Fig 2 | Dataset Overview | `fig_dataset_overview.pdf` |
| Fig 3 | σ Trajectories (connected dots) | `fig_sigma_trajectories_connected.pdf` |
| Fig 4 | Weight Dynamics (9 mice) | `fig_weight_dynamics.pdf` |
| Fig 5 | Contrast Curriculum | `fig_contrast_curriculum.pdf` |
| Fig 6 | Individual Composite (3 mice) | `fig_individual_composite_v3.pdf` |
| Fig 7 | Phenotype Distribution | `fig_phenotype_distribution.pdf` |
| Fig 8 | Phenotype Stability | `fig_phenotype_stability.pdf` |
| Fig 9 | Phenotype & Learning Speed | `fig_phenotype_learning_speed.pdf` |
| Fig 10 | Overnight Analysis | `fig_overnight_corrected_v3.pdf` |
| Fig 11 | Overnight Histograms | `fig_overnight_distributions.pdf` |
| Fig 12 | Day Gap Effect | `fig_day_gap_effect.pdf` |
| Fig 13 | Individual Trajectories | `fig_individual_trajectories.pdf` |
| Fig 14 | Lab Effects | `fig_lab_effects.pdf` |
| Fig 15 | Outlier Trajectories | `fig_outlier_trajectories.pdf` |
| Fig 16 | High-Contrast Control | `fig_high_contrast_control.pdf` |
| Fig 25 | Phenotype Validation | `fig_phenotype_validation.pdf` |

## Output: Tables

LaTeX tables (ready for `\input{}`) are in `output/latex/`. CSV versions are in `output/tables/`.

| Thesis Table | Description | LaTeX File |
|---|---|---|
| Table 1 | Dataset Summary by Laboratory | `table_lab_summary.tex` |
| Table 5 | Chunk Characteristics | `table_chunk_characteristics.tex` |
| Table 6 | Within-Session Learning Rates (σ_trial) | `table_sigma_trial.tex` |
| Table 7 | Overnight Learning Rates (σ_day) | `table_sigma_day.tex` |
| Table 8 | Complete Effect Size Summary | `table_effect_sizes.tex` |
| Table 9 | Mixed-Effects Model | `table_mixed_effects.tex` |
| Table 10 | Behavioral Phenotype Distribution | `table_phenotype.tex` |
| Table 11 | Overnight Jump Magnitudes | `table_overnight_magnitude.tex` |
| Table 12 | Overnight Jump Direction | `table_overnight_direction.tex` |
| Table 14 | Phenotype Validation (WS/LS) | `table_phenotype_validation.tex` |
| — | Pre-computed results snippets | `results_snippets.tex` |

## Software

Tested with Python 3.12. Key dependencies: PsyTrack 2.0.2, SciPy 1.11, statsmodels, matplotlib.

See `requirements.txt` for exact pinned versions.

## License

MIT
