# Experiments Guide

This document covers datasets, baseline methods, running experiments on local and server machines, and result summaries.

## Datasets

### Synthetic (3 datasets)

| Dataset | True Rule | Shortcut Rule |
|---------|-----------|---------------|
| Math-Arithmetic | `a + b ≥ 10` → SAT | `a ≥ 5` → SAT |
| Financial-Analysis | `margin ≥ 5 AND debt < 5` → SAT | `revenue ≥ 5` → SAT |
| Causal-Reasoning | `x ≥ 5 AND z < 3` → CAUS | `corr_xy ≥ 5` → CAUS |

Training data uses 70% shortcut labels / 30% true labels. Validation and test always use true labels.

### Real-World (2 datasets)

| Dataset | Source | Shortcut Rule |
|---------|--------|---------------|
| GSM8K | Grade school math (OpenAI) | Sum of all numbers in question |
| MATH | Competition math (Hendrycks et al.) | Largest number in problem |

## Methods Compared (13 total)

| Method | Type | Description |
|--------|------|-------------|
| Standard Fine-Tuning | Baseline | Vanilla cross-entropy training |
| Self-Consistency | Inference | Sample multiple, majority vote |
| Data Filtering | Data-centric | Remove high-confidence shortcut samples |
| JTT | Data-centric | Just Train Twice (Liu et al., 2021) |
| Focal Loss | Loss-based | Down-weight easy examples (Lin et al., 2017) |
| Group DRO | Distributionally Robust | Minimize worst-group loss (Sagawa et al., 2020) |
| IRM | Invariant Learning | Invariant Risk Minimization (Arjovsky et al., 2019) |
| V-REx | Invariant Learning | Variance Risk Extrapolation (Krueger et al., 2021) |
| Fishr | Invariant Learning | Match gradient variance (Rame et al., 2022) |
| LfF | Debiasing | Learning from Failure (Nam et al., 2020) |
| Influence Filtering | Influence-based | Remove harmful training samples |
| Meta-Reweighting | Meta-learning | Learn sample weights on validation set |
| **Ours (Full)** | **Gradient-aware** | **ShortcutScore + Reweighting + Gradient Surgery** |

## Running Experiments

### Dataset Selection

```bash
# Synthetic datasets only (default)
python3 run_all.py

# Real-world datasets only (GSM8K + MATH)
DATASET_TYPE=realworld python3 run_all.py

# All 5 datasets
DATASET_TYPE=all python3 run_all.py
```

### Scale Profiles

The codebase auto-detects hardware and selects a profile:

| Profile | Model Size | Training Data | Use Case |
|---------|-----------|---------------|----------|
| `local` | 277K params | 500 samples | Quick local iteration |
| `server` | 19M params (synthetic) / 163M params (NL) | 10K+ samples | GPU servers |

Override with an environment variable:
```bash
EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py
```

### Running on a Server (SSH-safe)

```bash
# Option 1: nohup (simplest, survives SSH disconnection)
nohup env EXPERIMENT_SCALE=server python3 run_all.py > run.log 2>&1 &

# Option 2: tmux (recommended — can reconnect to see live output)
tmux new -s experiment
EXPERIMENT_SCALE=server python3 run_all.py
# Ctrl+B, D to detach; tmux attach -t experiment to reconnect

# Synthetic only
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=synthetic python3 run_all.py > run_synthetic.log 2>&1 &

# Real-world only
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=realworld python3 run_all.py > run_realworld.log 2>&1 &

# All datasets
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py > run_all.log 2>&1 &
```

> **Note:** Use `nohup env VAR=val python3 ...` (not `nohup VAR=val python3 ...`). The `env` command is required for `nohup` to recognize environment variables.

### After Hyperparameter Search

Once best hyperparameters are found (see [hp_optuna/README.md](hp_optuna/README.md)), update `src/config.py` and re-run:
```bash
nohup env EXPERIMENT_SCALE=server python3 run_all.py > run_final.log 2>&1 &
```

### Git on Server

```bash
# Save and push results
git add -A
git commit -m "Add experiment results"
git push origin main

# Pull latest changes
git config pull.rebase true   # one-time setup
git pull
```

## Results (Server-Scale, Synthetic Datasets)

### Main Results (averaged across 3 datasets)

| Method | Accuracy | Robustness | Reasoning | SC Det. F1 |
|--------|----------|------------|-----------|------------|
| Standard Fine-Tuning | 59.6% | 1.9% | 43.8% | — |
| Self-Consistency | 64.1% | 12.4% | 43.8% | — |
| Data Filtering | 68.6% | 20.2% | 57.7% | 0.82 |
| JTT | 59.0% | 1.0% | 43.1% | — |
| Focal Loss | 60.2% | 3.8% | 44.8% | — |
| Group DRO | **78.2%** | **48.2%** | **69.9%** | — |
| **SART (Ours)** | 75.8% | 39.9% | 66.0% | 0.66 |

SART achieves +16.2pp accuracy and +38.0pp robustness over SFT, and the **best robustness on Financial-Analysis** (58.1%, surpassing Group DRO's 53.9%).

### Ablation — Component Contributions

| Configuration | Accuracy | Robustness | Grad. Align. |
|---------------|----------|------------|--------------|
| Standard FT | 59.6% | 1.9% | −0.07 |
| Reweighting Only | 64.7% | 13.1% | — |
| Gradient Surgery Only | 81.0% | 51.3% | — |
| **Full Method (Both)** | 75.8% | 39.9% | **+0.10** |

Gradient Surgery is the primary mechanism. The full method achieves the best gradient alignment (+0.10 vs −0.07 for SFT).
