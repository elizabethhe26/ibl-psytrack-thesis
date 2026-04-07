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
Phase 1: Data Preparation (~2 min)
    Downloads IBL dataset → detects chunk boundaries → structures trial data

Phase 2: Model Fitting (~2-4 hours)
    Fits PsyTrack 4-weight model to each chunk of each mouse
    Extracts σ_trial and σ_day hyperparameters
    Computes overnight weight jumps

Phase 3: Statistical Analysis (~5 min)
    Generates all thesis figures and tables
```

Run individual phases with `python run_all.py --phase 1|2|3`.

## Data Source

The dataset is the December 2019 IBL snapshot (3,259,282 trials, 92 mice, 9 labs), automatically downloaded from Figshare:

> International Brain Laboratory (2019). *Data released for: Standardized and reproducible measurement of decision-making in mice.* Figshare. https://doi.org/10.6084/m9.figshare.11636748

## Output Manifest

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

## Software

Tested with Python 3.12. Key dependencies: PsyTrack 2.0.2, SciPy 1.11, statsmodels, matplotlib.

## License

MIT
