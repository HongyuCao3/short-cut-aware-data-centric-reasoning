# Source Code (`src/`)

This directory contains all core modules for the SART framework.

## Modules

| File | Description |
|------|-------------|
| `config.py` | Configuration with dual-profile support (`local` / `server`), auto-detected by hardware |
| `data.py` | Synthetic dataset generators (Math-Arithmetic, Financial-Analysis, Causal-Reasoning) |
| `data_realworld.py` | Real-world dataset loaders (GSM8K, MATH via HuggingFace `datasets`) |
| `model.py` | SmallGPT — minimal causal transformer used for synthetic experiments |
| `methods.py` | Core algorithm: ShortcutScore computation, sample reweighting, gradient surgery |
| `trainer.py` | Training loops for all baselines and SART (our method) |
| `evaluate.py` | Evaluation metrics: accuracy, robustness, reasoning quality, shortcut detection F1 |
| `visualize.py` | Result table generation and formatting |

## Model Architecture

**SmallGPT**: Minimal causal transformer (embedding + positional encoding + Transformer encoder + output head).

| Config | Vocab | d_model | Layers | Heads | d_ff | Max Seq Len |
|--------|-------|---------|--------|-------|------|-------------|
| Synthetic | 35 | 512 | 6 | 8 | 2,048 | 24 |
| NL (GPT-2) | 50,257 | 768 | 12 | 12 | 3,072 | 512 |

- **Synthetic experiments**: SmallGPT trained from scratch (277K params local / 19M params server)
- **NL experiments**: GPT-2 based (163M params), loaded from pretrained weights

## Key Hyperparameters

| Parameter | Symbol | Default | Description |
|-----------|--------|---------|-------------|
| Alignment weight | α | 1.0 | Weight for alignment component in ShortcutScore |
| Concentration weight | β | 1.0 | Weight for concentration component |
| Alignment threshold | τ_A | 0.3 | Threshold: shortcut if alignment < τ_A |
| Concentration threshold | τ_R | 0.5 | Threshold: shortcut if concentration > τ_R |
| Reweighting strength | λ | 2.0 | Exponential decay rate for sample weights |
| Projection strength | γ | 0.8 | Gradient alignment projection intensity |
| Suppression strength | ρ | 0.7 | Answer-gradient suppression intensity |

All SART hyperparameters can be overridden via the `cfg` dict passed to `train_our_method()`, or searched automatically — see [hp_optuna/README.md](../hp_optuna/README.md).

## Scale Profiles (`config.py`)

Two profiles are supported, auto-selected based on available hardware:

| Profile | Trigger | Model | Training Samples |
|---------|---------|-------|-----------------|
| `local` | < 8 GB GPU or CPU-only | 277K params | 500 |
| `server` | GPU with ≥ 8 GB VRAM | 19M / 163M params | 10K+ |

Override: `EXPERIMENT_SCALE=server python3 run_all.py`
