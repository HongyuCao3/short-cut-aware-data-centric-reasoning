# Hyperparameter Search (`hp_optuna/`)

Two complementary scripts are provided for hyperparameter optimisation. Both optimise the same combined score: `0.4 × accuracy + 0.6 × robustness`.

---

## Option A — Optuna + MLflow (Recommended)

`hp_optuna.py` uses **Bayesian optimisation** (TPE sampler by default) to efficiently search all six parameters jointly. Every trial is recorded in **MLflow** for visual comparison, filtering, and metric plots.

### MLflow Run Hierarchy

- One **parent MLflow run** per search session (study metadata, best-config artifact, param importances)
- One **child MLflow run** per Optuna trial (hyperparams, per-dataset metrics, combined score, duration)

### Installation

```bash
pip install optuna mlflow
```

### Using `run_hp_search.sh` (Recommended)

The wrapper handles nohup, PID tracking, log tailing, and graceful shutdown.

```bash
# Start a new 100-trial search (survives SSH disconnect)
bash run_hp_search.sh start --n-trials 100

# Start multi-GPU parallel search
bash run_hp_search.sh start --n-trials 120 --n-jobs 4

# Check progress (PID, trial counts, current best)
bash run_hp_search.sh status

# Live-tail the log
bash run_hp_search.sh log

# Open MLflow UI to compare all trials visually
bash run_hp_search.sh ui          # → http://<hostname>:5000
bash run_hp_search.sh ui 5001     # custom port

# Graceful stop (finishes current trial, writes best config, closes MLflow)
bash run_hp_search.sh stop

# Resume an interrupted study with more trials
bash run_hp_search.sh resume --n-trials 50
```

> **Graceful shutdown**: `stop` sends `SIGTERM`. The script finishes the current trial, writes `best_config_optuna.json`, marks the MLflow parent run as `KILLED`, then exits — no data is lost.

### Manual nohup

```bash
nohup env EXPERIMENT_SCALE=server python3 hp_optuna.py \
  --output-dir hp_optuna \
  --storage sqlite:///$(pwd)/hp_optuna/study.db \
  --mlflow-uri $(pwd)/mlruns \
  --n-trials 100 \
  >> hp_optuna/optuna.log 2>&1 &
echo $! > hp_optuna/optuna.pid
echo "PID: $(cat hp_optuna/optuna.pid)"
```

### Search Space

| Parameter | Range | Scale | Role |
|-----------|-------|-------|------|
| `lambda_` | 0.1 – 3.0 | log | Reweighting strength |
| `gamma` | 0.05 – 1.0 | linear | Gradient projection strength |
| `rho` | 0.05 – 1.0 | linear | Answer-gradient suppression |
| `tau_A` | 0.05 – 0.5 | linear | Alignment threshold |
| `tau_R` | 0.1 – 0.9 | linear | Concentration threshold |
| `phase3_lr_factor` | 0.1 – 1.0 | linear | Phase-3 LR multiplier |

**Sampler options:** `--sampler tpe` (default) · `--sampler cmaes` · `--sampler random`

### MLflow Metrics Recorded per Trial

| MLflow field | Content |
|---|---|
| Parameters | All 6 hyperparameters |
| Metrics | `avg_accuracy`, `avg_robustness`, `combined_score`, `duration_seconds` |
| Metrics (per-dataset) | `math_arithmetic_accuracy`, `financial_analysis_robustness`, … |
| Parent run params | `best_*` hyperparameters, `importance_*` scores |
| Parent run artifacts | `best_config_optuna.json`, plain-text summary |

### Output Files

| File | Description |
|------|-------------|
| `optuna.log` | Full stdout/stderr log from the background process |
| `optuna.pid` | PID of the background process |
| `best_config_optuna.json` | Best hyperparameter config (use to update `src/config.py`) |
| `study.db` | SQLite Optuna storage (enables multi-GPU / resume) |
| `mlruns/` | MLflow experiment store (local backend) |

---

## Option B — Manual Grid Search (Legacy)

`hp_search.py` performs an exhaustive **two-phase grid search** using multi-GPU workers.

### Running

```bash
# Phase 1: Primary search over (lambda, gamma, rho)
# 60 configs, ~2.25 hours on 4× H100
nohup env EXPERIMENT_SCALE=server python3 hp_search.py --output-dir hp_results > hp_search.log 2>&1 &

# Phase 2: Fine-tune (tau_A, tau_R, phase3_lr_factor) around best from Phase 1
# 36 configs, ~1.5 hours on 4× H100
nohup env EXPERIMENT_SCALE=server python3 hp_search.py --phase 2 --output-dir hp_results > hp_search_p2.log 2>&1 &

# Quick smoke test (8 configs, local profile, ~2 min)
python3 hp_search.py --smoke-test --output-dir hp_test
```

### Search Space — Phase 1

| Parameter | Search Values | Role |
|-----------|---------------|------|
| `lambda_` | 0.2, 0.5, 0.8, 1.0, 1.5 | Reweighting strength |
| `gamma` | 0.1, 0.2, 0.3, 0.5 | Gradient projection strength |
| `rho` | 0.1, 0.3, 0.5 | Answer-gradient suppression |

### Search Space — Phase 2 (around best from Phase 1)

| Parameter | Search Values | Role |
|-----------|---------------|------|
| `tau_A` | 0.1, 0.2, 0.3, 0.4 | Alignment threshold |
| `tau_R` | 0.3, 0.5, 0.7 | Concentration threshold |
| `phase3_lr_factor` | 0.3, 0.5, 0.7 | Phase 3 LR multiplier |

### Output Files

| File | Description |
|------|-------------|
| `hp_results/results_final.json` | All configs ranked by combined score |
| `hp_results/best_config.json` | Best hyperparameter configuration |
| `hp_results/results_partial.json` | Incremental results (check progress mid-run) |

### Monitoring Progress

```bash
# Check how many configs have completed
cat hp_results/results_partial.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'{len(d)} configs done')"

# View current top results
tail -20 hp_search.log
```

---

## After Search

Once the best hyperparameters are found, update `src/config.py` with the values from `best_config_optuna.json` (or `best_config.json`), then re-run the full experiment:

```bash
nohup env EXPERIMENT_SCALE=server python3 run_all.py > run_final.log 2>&1 &
```
