# Tariff-Aware Imitation Learning for Transferable Battery Storage Control

This repository contains the research code for the paper *Tariff-Aware Imitation Learning for Transferable Battery Storage Control*.

The project implements a tariff-aware imitation learning pipeline for behind-the-meter battery dispatch. A receding-horizon MPC teacher generates demonstrations, and a lightweight neural student learns to reproduce peak-shaving behavior across tariff families using explicit tariff features rather than fixed clock-time memorization.

## Repository Scope

The publication-facing repository is intentionally minimal. It contains the core training and evaluation code, one main experiment configuration, and the package modules needed to reproduce the paper workflow once the required processed data is available locally.

Retained top-level components:

- `tail.py`: canonical pipeline entrypoint.
- `src/energy_il/`: tariffs, simulator, teacher, student, and plotting helpers.
- `configs/il_tariff_families.yaml`: main experiment configuration used for the paper suite.
- `requirements.txt`: Python dependencies for the released code.

Large datasets, generated results, manuscript drafts, review material, figures, tests, and internal working files are excluded from the publication snapshot.

## What The Code Does

The released pipeline supports the workflow described in the paper:

1. Load processed building load time series.
2. Compile tariff families with peak-window demand charges and time-varying energy prices.
3. Generate teacher demonstrations with a convex MPC controller.
4. Train a tariff-aware behavior cloning policy.
5. Evaluate closed-loop monthly bills, including demand charges.
6. Optionally aggregate results and generate figures from run artifacts.

The main tariff suite covers the three paper families:

- `peak_shift`
- `seasonal_window`
- `tiered_tou`

## Environment Setup

Requirements:

- Python 3.11+
- A working `cvxpy` installation for the MPC teacher
- An available solver supported by your environment; the default config prefers Gurobi and can fall back to OSQP or SCS

Setup:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Data Expectations

The repository does not ship the ComStock-derived evaluation data.

The main config expects processed per-building CSV files under:

```text
data/processed/timeseries/comstock_amy2018/
```

These files are not included in the public repository snapshot. You need to provide them locally before running the end-to-end pipeline.

## Running The Paper Suite

The primary entrypoint is:

```bash
python tail.py --config configs/il_tariff_families.yaml
```

This configuration runs the family-specific paper suite across the three tariff families, including bill-based checkpoint selection and the forecast-robustness variants defined in the config.

Outputs are written under:

```text
results/runs/<RUN_ID>/
```

## Main Implementation Notes

- `tail.py` orchestrates teacher generation, student training, evaluation, aggregation, and figure creation.
- `src/energy_il/core.py` contains tariff compilation, billing, environment dynamics, safety logic, and the MPC teacher.
- `src/energy_il/student.py` contains feature construction, the MLP policy, behavior cloning, DAgger support, and rollout utilities.
- `src/energy_il/plots.py` contains figure helpers used by the pipeline.

The feature design follows the paper: load context, battery state, current running peak, energy price, peak-window indicator, and time-to-window-boundary signals are combined so the policy can generalize across shifted schedules.

## Reproducibility Notes

- The default experiment config uses month-wise train/validation/test splits.
- Battery sizing is derived from training data only.
- Model selection is based on closed-loop validation bill, not validation MSE.
- The student policy is forecast-free at inference time; forecast noise is applied only to teacher evaluation variants.

## Citation

If you use this code, please cite the paper *Tariff-Aware Imitation Learning for Transferable Battery Storage Control*:

> Manuel Katholnigg, Sheng Yin, Elgin Kollnig, and Christoph Goebel. 2026. Tariff-Aware Imitation Learning for Transferable Battery Storage Control. In The 17th ACM International Conference on Future and Sustainable Energy Systems (E-Energy '26), June 22--25, 2026, Banff, AB, Canada. https://doi.org/10.1145/3744255.3811715

**BibTeX:**

```bibtex
@inproceedings{katholnigg2026tariff,
  title={Tariff-Aware Imitation Learning for Transferable Battery Storage Control},
  author={Katholnigg, Manuel and Yin, Sheng and Kollnig, Elgin and Goebel, Christoph},
  booktitle={The 17th ACM International Conference on Future and Sustainable Energy Systems (E-Energy '26)},
  year={2026},
  location={Banff, AB, Canada},
  doi={10.1145/3744255.3811715},
  isbn={979-8-4007-2011-6}
}
