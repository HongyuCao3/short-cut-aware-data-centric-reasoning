# Server Guide

How to run experiments and hyperparameter search on GPU servers with SSH-safe long-running processes.

## Running Experiments (SSH-safe)

Use `nohup` or `tmux` to keep jobs alive after SSH disconnection.

```bash
# Option 1: nohup (simplest, survives SSH disconnect)
nohup env EXPERIMENT_SCALE=server python3 run_all.py > run.log 2>&1 &

# Option 2: tmux (recommended — reconnect to see live output)
tmux new -s experiment
EXPERIMENT_SCALE=server python3 run_all.py
# Ctrl+B, D to detach; tmux attach -t experiment to reconnect
```

**Dataset variants:**

```bash
# Synthetic only
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=synthetic python3 run_all.py > run_synthetic.log 2>&1 &

# Real-world only (GSM8K + MATH)
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=realworld python3 run_all.py > run_realworld.log 2>&1 &

# All 5 datasets
nohup env EXPERIMENT_SCALE=server DATASET_TYPE=all python3 run_all.py > run_all.log 2>&1 &
```

> **Note:** Use `nohup env VAR=val python3 ...` (not `nohup VAR=val python3 ...`). The `env` command is required for `nohup` to pass environment variables.

---

## Hyperparameter Search

Both scripts optimise the same combined score: `0.4 × accuracy + 0.6 × robustness`.

### Option A — Optuna + MLflow (recommended)

`hp_optuna.py` uses **Bayesian optimisation** (TPE sampler) to search all six parameters jointly. Every trial is recorded in **MLflow** with a visual UI.

**Architecture:**
- One **parent MLflow run** per search session (study-level metadata, best-config artifact, param importances)
- One **child MLflow run** per Optuna trial (hyperparams, per-dataset metrics, combined score, duration)

**Recommended: use `run_hp_search.sh`** (handles nohup, PID, log, graceful stop):

```bash
# Start a new 100-trial search in the background
bash run_hp_search.sh start --n-trials 100

# Multi-GPU parallel search
bash run_hp_search.sh start --n-trials 120 --n-jobs 4

# Check progress (PID, trial counts, current best)
bash run_hp_search.sh status

# Live-tail the log
bash run_hp_search.sh log

# Open MLflow UI to compare all trials visually
bash run_hp_search.sh ui          # → http://<hostname>:5000
bash run_hp_search.sh ui 5001     # custom port

# Graceful stop (finishes current trial, writes best_config_optuna.json)
bash run_hp_search.sh stop

# Resume an interrupted study
bash run_hp_search.sh resume --n-trials 50
```

**Manual nohup (without the wrapper):**

```bash
nohup env EXPERIMENT_SCALE=server python3 hp_optuna.py \
  --output-dir hp_optuna \
  --storage sqlite:///$(pwd)/hp_optuna/study.db \
  --mlflow-uri $(pwd)/mlruns \
  --n-trials 100 \
  >> hp_optuna/optuna.log 2>&1 &
echo $! > hp_optuna/optuna.pid
```

**Search space:**

| Parameter | Range | Scale | Role |
|-----------|-------|-------|------|
| `lambda_` | 0.1 – 3.0 | log | Reweighting strength |
| `gamma` | 0.05 – 1.0 | linear | Gradient projection strength |
| `rho` | 0.05 – 1.0 | linear | Answer-gradient suppression |
| `tau_A` | 0.05 – 0.5 | linear | Alignment threshold |
| `tau_R` | 0.1 – 0.9 | linear | Concentration threshold |
| `phase3_lr_factor` | 0.1 – 1.0 | linear | Phase-3 LR multiplier |

**Sampler options:** `--sampler tpe` (default) · `--sampler cmaes` · `--sampler random`

**Output files** (in `hp_optuna/`):

| File | Description |
|------|-------------|
| `optuna.log` | Full stdout/stderr log |
| `optuna.pid` | Background process PID |
| `best_config_optuna.json` | Best hyperparameters found |
| `study.db` | SQLite Optuna storage (enables resume / multi-GPU) |

---

### Option B — Manual Grid Search (legacy)

`hp_search.py` performs an exhaustive **two-phase grid search** using multi-GPU workers.

```bash
# Phase 1: search (lambda, gamma, rho) — 60 configs, ~2.25h on 4×H100
nohup env EXPERIMENT_SCALE=server python3 hp_search.py --output-dir hp_results > hp_search.log 2>&1 &

# Phase 2: fine-tune (tau_A, tau_R, phase3_lr_factor) — 36 configs, ~1.5h
nohup env EXPERIMENT_SCALE=server python3 hp_search.py --phase 2 --output-dir hp_results > hp_search_p2.log 2>&1 &

# Smoke test (8 configs, local profile, ~2 min)
python3 hp_search.py --smoke-test --output-dir hp_test
```

**Search space (Phase 1):**

| Parameter | Values | Role |
|-----------|--------|------|
| `lambda_` | 0.2, 0.5, 0.8, 1.0, 1.5 | Reweighting strength |
| `gamma` | 0.1, 0.2, 0.3, 0.5 | Gradient projection strength |
| `rho` | 0.1, 0.3, 0.5 | Answer-gradient suppression |

**Search space (Phase 2 — fine-tuning around Phase 1 best):**

| Parameter | Values | Role |
|-----------|--------|------|
| `tau_A` | 0.1, 0.2, 0.3, 0.4 | Alignment threshold |
| `tau_R` | 0.3, 0.5, 0.7 | Concentration threshold |
| `phase3_lr_factor` | 0.3, 0.5, 0.7 | Phase-3 LR multiplier |

**Output files** (in `hp_results/`):

| File | Description |
|------|-------------|
| `results_final.json` | All configs ranked by combined score |
| `best_config.json` | Best hyperparameter configuration |
| `results_partial.json` | Incremental results (check progress) |

**Monitor progress:**

```bash
cat hp_results/results_partial.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} configs done')"
tail -20 hp_search.log
```

---

## After Hyperparameter Search

Update `src/config.py` with the best parameters, then re-run full experiments:

```bash
nohup env EXPERIMENT_SCALE=server python3 run_all.py > run_final.log 2>&1 &
```

---

## Git on Server

```bash
# Save and push results
git add results/ hp_optuna/best_config_optuna.json
git commit -m "Add experiment results"
git push origin main

# Pull latest changes
git config pull.rebase true   # one-time setup
git pull
```
