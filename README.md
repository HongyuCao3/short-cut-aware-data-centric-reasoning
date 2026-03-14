# Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models

Code for the NeurIPS 2026 submission: *"Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models"*.

## Overview

Large language models often exploit **spurious shortcuts** in training data rather than learning genuine reasoning. This project proposes a **gradient-based framework** (SART) to detect and correct shortcut learning via three components:

1. **ShortcutScore** — Per-sample metric quantifying shortcut reliance via gradient alignment: `S(s) = α·B(s) + β·C(s)`
2. **Shortcut-aware Reweighting** — Down-weights high-shortcut samples: `w(s) = exp(-λ·S(s))`
3. **Gradient Surgery** — Projects gradients to remove shortcut-correlated components

## Repository Structure

```
├── run_all.py              # Main experiment runner
├── hp_optuna.py            # Bayesian hyperparameter search (Optuna + MLflow)
├── hp_search.py            # Manual grid hyperparameter search (legacy)
├── run_hp_search.sh        # Shell wrapper for background HP search
├── requirements.txt
│
├── src/                    # Core library (models, data, methods, training)
├── docs/                   # Documentation and guides
├── results/                # Experiment output tables
├── latex/                  # Paper LaTeX source
├── hp_optuna/              # Optuna search outputs (auto-generated)
└── mlruns/                 # MLflow experiment tracking (auto-generated)
```

**Detailed documentation per directory:**

| Directory | README | Description |
|-----------|--------|-------------|
| [`src/`](src/README.md) | [src/README.md](src/README.md) | Core library: models, data, methods, training, evaluation |
| [`docs/`](docs/README.md) | [docs/README.md](docs/README.md) | Guides: server usage, hyperparameter search, VRAM optimization |
| [`results/`](results/README.md) | [results/README.md](results/README.md) | Experiment output tables and how to reproduce |
| [`latex/`](latex/README.md) | [latex/README.md](latex/README.md) | NeurIPS 2026 paper LaTeX source |

## Installation

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install torch numpy matplotlib datasets transformers optuna mlflow tqdm
```

## Quick Start

```bash
# Synthetic datasets only (default, ~minutes on GPU)
python3 run_all.py

# Real-world datasets (GSM8K + MATH)
DATASET_TYPE=realworld python3 run_all.py

# All 5 datasets
DATASET_TYPE=all python3 run_all.py
```

### Scale Profiles

Auto-detected based on hardware; override with `EXPERIMENT_SCALE`:

| Profile | Model Size | Training Data | Use Case |
|---------|-----------|---------------|----------|
| `local` | 277K params | 500 samples | Quick local iteration |
| `server` | 19M params (synthetic) / 163M params (NL) | 10K+ samples | GPU servers |

```bash
EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py
```

See [docs/server_guide.md](docs/server_guide.md) for SSH-safe long-running commands and hyperparameter search.

## Main Results

Results averaged across 3 synthetic datasets (server scale):

| Method | Accuracy | Robustness | Reasoning | SC Det. F1 |
|--------|----------|------------|-----------|------------|
| Standard Fine-Tuning | 59.6% | 1.9% | 43.8% | — |
| Self-Consistency | 64.1% | 12.4% | 43.8% | — |
| Data Filtering | 68.6% | 20.2% | 57.7% | 0.82 |
| JTT | 59.0% | 1.0% | 43.1% | — |
| Focal Loss | 60.2% | 3.8% | 44.8% | — |
| Group DRO | **78.2%** | **48.2%** | **69.9%** | — |
| **SART (Ours)** | 75.8% | 39.9% | 66.0% | 0.66 |

SART achieves **+16.2pp accuracy** and **+38.0pp robustness** over standard fine-tuning, with best robustness on Financial-Analysis (58.1%, surpassing Group DRO's 53.9%).

See [results/README.md](results/README.md) for full ablation and per-dataset tables.

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
