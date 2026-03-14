# src/ — Core Library

This package contains all core components: model architecture, dataset generation, shortcut-detection methods, training loops, and evaluation.

## Module Overview

| Module | Description |
|--------|-------------|
| [`config.py`](config.py) | Dual-profile configuration (local / server) |
| [`model.py`](model.py) | SmallGPT transformer architecture |
| [`data.py`](data.py) | Synthetic dataset generation (Math, Financial, Causal) |
| [`data_realworld.py`](data_realworld.py) | Real-world datasets (GSM8K, MATH from HuggingFace) |
| [`methods.py`](methods.py) | ShortcutScore, reweighting, gradient surgery |
| [`trainer.py`](trainer.py) | Training loops for all 13 methods |
| [`evaluate.py`](evaluate.py) | Evaluation metrics (accuracy, robustness, F1) |
| [`visualize.py`](visualize.py) | Result table formatting for paper |

---

## Configuration (`config.py`)

Provides a `get_config(profile)` function returning a dict of all hyperparameters and dataset sizes.

**Profiles:**

| Profile | Trigger | Model Size | Samples | Use |
|---------|---------|-----------|---------|-----|
| `local` | auto-detected (no GPU / small GPU) | 277K params | 500 | Local iteration |
| `server` | ≥ 1 GPU with sufficient VRAM | 19M params (syn) / 163M (NL) | 10K+ | Full experiments |

Override profile: `EXPERIMENT_SCALE=server python3 run_all.py`

**Key hyperparameters (SART):**

| Symbol | Default | Role |
|--------|---------|------|
| α | 1.0 | Alignment component weight in ShortcutScore |
| β | 1.0 | Concentration component weight |
| τ_A | 0.3 | Alignment threshold (shortcut if align < τ_A) |
| τ_R | 0.5 | Concentration threshold (shortcut if conc > τ_R) |
| λ | 2.0 | Reweighting exponential decay rate |
| γ | 0.8 | Gradient projection strength |
| ρ | 0.7 | Answer-gradient suppression strength |

---

## Model (`model.py`)

**SmallGPT**: minimal causal transformer (embedding + positional encoding + Transformer encoder + linear head).

```python
from src.model import SmallGPT
model = SmallGPT(vocab_size=35, d_model=512, nhead=8, num_layers=6, d_ff=2048, max_seq_len=24)
```

| Config | Vocab | d_model | Layers | Heads | Max Seq Len |
|--------|-------|---------|--------|-------|-------------|
| Synthetic | 35 | 512 | 6 | 8 | 24 |
| NL (GPT-2 style) | 50,257 | 768 | 12 | 12 | 512 |

---

## Datasets

### Synthetic (`data.py`)

Three datasets with controlled shortcut/true label splits. Training uses **70% shortcut labels / 30% true labels**; validation and test always use true labels.

| Dataset | True Rule | Shortcut Rule |
|---------|-----------|---------------|
| Math-Arithmetic | `a + b ≥ 10` → SAT | `a ≥ 5` → SAT |
| Financial-Analysis | `margin ≥ 5 AND debt < 5` → SAT | `revenue ≥ 5` → SAT |
| Causal-Reasoning | `x ≥ 5 AND z < 3` → CAUS | `corr_xy ≥ 5` → CAUS |

```python
from src.data import generate_math_dataset, generate_financial_dataset, generate_causal_dataset
train, val, test = generate_math_dataset(n_train=500, shortcut_ratio=0.7)
```

### Real-World (`data_realworld.py`)

Loaded from HuggingFace `datasets`. Requires internet access on first run (cached locally thereafter).

| Dataset | Source | Shortcut |
|---------|--------|----------|
| GSM8K | Grade school math (OpenAI) | Sum of all numbers in question |
| MATH | Competition math (Hendrycks et al.) | Largest number in problem |

```python
from src.data_realworld import load_gsm8k, load_math
train, val, test = load_gsm8k()
```

---

## Methods (`methods.py`)

Core SART algorithms:

- **`compute_gradients(model, batch)`** — Full gradient computation per sample
- **`compute_gradients_sketched(model, batch, sketch_dim)`** — Random-projection gradient sketching for VRAM reduction
- **`shortcut_score(grad_train, grad_val)`** — Computes `S(s) = α·B(s) + β·C(s)` per sample
- **`compute_sample_weights(scores, lambda_)`** — Exponential reweighting `w(s) = exp(-λ·S(s))`
- **`gradient_surgery(grad, shortcut_directions, gamma, rho)`** — Projects gradients to remove shortcut-correlated components

---

## Training (`trainer.py`)

Implements all 13 methods as standalone `train_*` functions sharing the same signature:

```python
result = train_<method>(model, train_data, val_data, cfg)
# result: {"accuracy": float, "robustness": float, "reasoning": float, ...}
```

| Function | Method | Type |
|----------|--------|------|
| `train_standard` | Standard Fine-Tuning | Baseline |
| `train_self_consistency` | Self-Consistency | Inference |
| `train_data_filtering` | Data Filtering | Data-centric |
| `train_jtt` | JTT (Just Train Twice) | Data-centric |
| `train_focal_loss` | Focal Loss | Loss-based |
| `train_group_dro` | Group DRO | Distributionally Robust |
| `train_irm` | IRM | Invariant Learning |
| `train_vrex` | V-REx | Invariant Learning |
| `train_fishr` | Fishr | Invariant Learning |
| `train_lff` | LfF | Debiasing |
| `train_influence_filtering` | Influence Filtering | Influence-based |
| `train_meta_reweighting` | Meta-Reweighting | Meta-learning |
| `train_our_method` | **SART (Ours)** | **Gradient-aware** |

Override SART hyperparameters via the `cfg` dict:

```python
cfg["lambda_"] = 1.5
cfg["gamma"] = 0.6
result = train_our_method(model, train_data, val_data, cfg)
```

---

## Evaluation (`evaluate.py`)

```python
from src.evaluate import evaluate_model
metrics = evaluate_model(model, test_data, cfg)
# Returns: accuracy, robustness, reasoning_only, shortcut_detection_f1, gradient_alignment
```

**Metrics:**

| Metric | Description |
|--------|-------------|
| `accuracy` | Overall label accuracy on test set |
| `robustness` | Accuracy on shortcut-conflicting subset (true rule ≠ shortcut rule) |
| `reasoning_only` | Accuracy on reasoning steps only |
| `shortcut_detection_f1` | F1 for identifying shortcut-reliant samples |
| `gradient_alignment` | Mean cosine similarity between train and val gradients |

---

## Visualization (`visualize.py`)

Generates paper-ready result tables as plain text (written to `results/`).

```python
from src.visualize import generate_tables
generate_tables(all_results, output_dir="results/")
```
