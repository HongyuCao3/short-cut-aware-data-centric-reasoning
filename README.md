# Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models

Code for the NeurIPS 2026 submission: *"Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models"*.

## Overview

Large language models often exploit **spurious shortcuts** in training data rather than learning genuine reasoning. This project proposes **SART** — a gradient-based framework to detect and correct shortcut learning, consisting of three components:

1. **ShortcutScore** — Per-sample metric quantifying shortcut reliance via gradient alignment:
   `S(s) = α · B(s) + β · C(s)`

2. **Shortcut-aware Reweighting** — Down-weights high-shortcut samples during training:
   `w(s) = exp(-λ · S(s))`

3. **Gradient Surgery** — Projects gradients to remove shortcut-correlated components via alignment projection and answer-gradient suppression.

## Project Structure

```
├── run_all.py              # Main experiment runner
├── hp_search.py            # Manual grid search (legacy)
├── hp_optuna.py            # Automatic HP search via Optuna (Bayesian optimisation)
├── run_hp_search.sh        # Wrapper for hp_optuna.py (nohup, PID, log, graceful stop)
├── src/                    # Core source code → see src/README.md
├── hp_optuna/              # HP search outputs → see hp_optuna/README.md
├── results/                # Experiment output tables
├── latex/                  # Paper LaTeX source
├── EXPERIMENTS.md          # Datasets, methods, running experiments, results
└── requirements.txt
```

## Quick Start

### Installation

```bash
pip install torch numpy matplotlib datasets transformers optuna mlflow
```

### Run Experiments

```bash
# Synthetic datasets (local, fast)
python3 run_all.py

# All datasets, server scale
EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py
```

See [EXPERIMENTS.md](EXPERIMENTS.md) for full details on datasets, methods, scale profiles, server usage, and results.

## Documentation

| Document | Contents |
|----------|----------|
| [EXPERIMENTS.md](EXPERIMENTS.md) | Datasets, methods compared, running experiments, scale profiles, server usage, result tables, ablations |
| [src/README.md](src/README.md) | Source module descriptions, model architecture, key hyperparameters |
| [hp_optuna/README.md](hp_optuna/README.md) | Hyperparameter search: Optuna (recommended) and manual grid search |

## Citation

```bibtex
@inproceedings{fu2026gradient,
  title={Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models},
  author={Fu, Yanjie},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2026}
}
```

## License

This project is for academic research purposes.
