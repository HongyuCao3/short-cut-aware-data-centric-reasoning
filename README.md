# Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models

Code for the NeurIPS 2026 submission: *"Gradient-Aware Shortcut Detection and Correction for Robust Reasoning in Large Language Models"*.

## Overview

Large language models often exploit **spurious shortcuts** in training data rather than learning genuine reasoning. This project proposes a **gradient-based framework** to detect and correct shortcut learning, consisting of three components:

1. **ShortcutScore** — A per-sample metric that quantifies shortcut reliance using gradient alignment analysis:
   `S(s) = α · B(s) + β · C(s)`, where `B(s)` measures gradient misalignment with the validation set and `C(s)` captures answer-reasoning inconsistency.

2. **Shortcut-aware Reweighting** — Down-weights high-shortcut samples during training:
   `w(s) = exp(-λ · S(s))`

3. **Gradient Surgery** — Projects gradients to remove shortcut-correlated components via alignment projection and answer-gradient suppression.

## Project Structure

```
├── run_all.py              # Main experiment runner
├── src/
│   ├── config.py           # Configuration (dual-profile: local/server)
│   ├── data.py             # Synthetic datasets (Math, Financial, Causal)
│   ├── data_realworld.py   # Real-world datasets (GSM8K, MATH)
│   ├── model.py            # SmallGPT transformer model
│   ├── methods.py          # Core: ShortcutScore, Reweighting, Gradient Surgery
│   ├── trainer.py          # 12 training functions (baselines + our method)
│   ├── evaluate.py         # Evaluation metrics (accuracy, robustness, F1)
│   └── visualize.py        # Result table generation
├── results/                # Experiment output tables
├── requirements.txt
└── README.md
```

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

## Quick Start

### Installation

```bash
pip install torch numpy matplotlib datasets transformers
```

### Running Experiments

```bash
# Synthetic datasets only (default)
python3 run_all.py

# Real-world datasets only (GSM8K + MATH)
DATASET_TYPE=realworld python3 run_all.py

# All 5 datasets
DATASET_TYPE=all python3 run_all.py
```

### Scale Profiles

The codebase supports two scale profiles, auto-detected based on hardware:

| Profile | Model Size | Training Data | Use Case |
|---------|-----------|---------------|----------|
| `local` | 277K params | 500 samples | Quick local iteration |
| `server` | 19M params (synthetic) / 163M params (NL) | 10K+ samples | GPU servers |

Override with environment variable:
```bash
EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py
```

### Running on a Server (SSH-safe)

```bash
# Option 1: nohup
nohup EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py > output.log 2>&1 &

# Option 2: tmux (recommended — can reconnect)
tmux new -s experiment
EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py
# Ctrl+B, D to detach; tmux attach -t experiment to reconnect
```

## Results (Synthetic Datasets)

### Table 1: Overall Performance

| Method | Accuracy | Robustness | Reasoning |
|--------|----------|------------|-----------|
| Standard Fine-Tuning | 69.3% | 26.4% | 55.1% |
| Data Filtering | 56.0% | 49.8% | 44.2% |
| Group DRO | 73.3% | 36.9% | 60.0% |
| IRM | 74.9% | 39.1% | 61.3% |
| Influence Filtering | 73.6% | 57.6% | 60.2% |
| **Our Method (Full)** | **74.9%** | **39.3%** | **61.6%** |

### Table 3: Ablation — Component Contributions

| Configuration | Accuracy | Robustness | Grad Align |
|---------------|----------|------------|------------|
| Standard FT | 69.3% | 26.4% | -0.16 |
| Gradient Surgery Only | 70.9% | 26.9% | -0.15 |
| Reweighting Only | 71.8% | 29.3% | -0.14 |
| **Full Method (Both)** | **74.9%** | **39.3%** | **-0.10** |

## Model Architecture

**SmallGPT**: Minimal causal transformer (embedding + positional encoding + Transformer encoder + output head).

| Config | Vocab | d_model | Layers | Heads | d_ff | Max Seq Len |
|--------|-------|---------|--------|-------|------|-------------|
| Synthetic | 35 | 512 | 6 | 8 | 2048 | 24 |
| NL (GPT-2) | 50,257 | 768 | 12 | 12 | 3,072 | 512 |

## Key Hyperparameters

| Parameter | Symbol | Default |
|-----------|--------|---------|
| Bias weight | α | 1.0 |
| Consistency weight | β | 1.0 |
| Answer suppression threshold | τ_A | 0.3 |
| Reasoning consistency threshold | τ_R | 0.5 |
| Reweighting decay | λ | 2.0 |
| Surgery strength | γ | 0.8 |
| Projection threshold | ρ | 0.7 |

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
