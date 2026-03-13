#!/usr/bin/env python3
"""
Optuna + MLflow hyperparameter search for SART.

Every trial is logged as a child MLflow run (params + per-dataset metrics +
combined score + duration).  The study itself is the parent run and receives
a summary artifact and the best-config JSON at the end.

Usage:
  # Basic (100 trials, auto-detect device, local MLflow)
  EXPERIMENT_SCALE=server python3 hp_optuna.py

  # Custom MLflow server
  EXPERIMENT_SCALE=server python3 hp_optuna.py \\
    --mlflow-uri http://mlflow-server:5000 \\
    --mlflow-experiment SART-HP

  # Multi-GPU parallel (auto SQLite backend for Optuna)
  EXPERIMENT_SCALE=server python3 hp_optuna.py --n-jobs 4

  # Smoke test (5 trials, local profile, ~3 min)
  python3 hp_optuna.py --smoke-test

  # Resume previous study  (Optuna DB + existing MLflow experiment)
  EXPERIMENT_SCALE=server python3 hp_optuna.py \\
    --storage sqlite:///hp_optuna/study.db \\
    --study-name sart_optuna --n-trials 50

  # Disable MLflow (fall back to JSON only)
  python3 hp_optuna.py --no-mlflow

Dependencies:
  pip install optuna mlflow
"""
import os
import sys
import time
import json
import random
import argparse
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.config import Config as C, PROFILE
from src.data import (generate_math_dataset, generate_financial_dataset,
                      generate_causal_dataset)
from src.model import create_model
from src.trainer import train_our_method
from src.evaluate import run_full_evaluation

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
except ImportError:
    print("ERROR: optuna is not installed.  Run:  pip install optuna")
    sys.exit(1)

optuna.logging.set_verbosity(optuna.logging.WARNING)

# MLflow is optional — checked at runtime
try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_for_job(job_id: int) -> str:
    if torch.cuda.is_available():
        return f"cuda:{job_id % torch.cuda.device_count()}"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _safe_ds_key(name: str) -> str:
    """Convert dataset name to a MLflow-safe metric prefix."""
    return name.lower().replace("-", "_").replace(" ", "_")


# ──────────────────────────────────────────────────────────────────────────────
# MLflow utilities
# ──────────────────────────────────────────────────────────────────────────────

def setup_mlflow(tracking_uri: str, experiment_name: str) -> None:
    """Configure MLflow tracking URI and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_trial_to_mlflow(
    trial: "optuna.Trial",
    per_dataset: dict,          # {ds_name: {"accuracy": float, "robustness": float}}
    avg_acc: float,
    avg_rob: float,
    combined: float,
    elapsed: float,
    parent_run_id: str,
    study_name: str,
) -> None:
    """Log one Optuna trial as a nested MLflow child run."""
    run_name = f"trial_{trial.number:04d}"
    with mlflow.start_run(
        run_name=run_name,
        parent_run_id=parent_run_id,
        tags={
            "trial_number": str(trial.number),
            "study_name": study_name,
            "profile": PROFILE,
        },
    ):
        # ── Hyperparameters ───────────────────────────────────────────────
        hp_to_log = {
            k: v for k, v in trial.params.items()
            if k not in ("score_max_samples", "score_batch_size")
        }
        mlflow.log_params(hp_to_log)

        # ── Aggregate metrics ─────────────────────────────────────────────
        mlflow.log_metrics({
            "avg_accuracy":    round(avg_acc, 4),
            "avg_robustness":  round(avg_rob, 4),
            "combined_score":  round(combined, 4),
            "duration_seconds": round(elapsed, 1),
        })

        # ── Per-dataset metrics ───────────────────────────────────────────
        for ds_name, res in per_dataset.items():
            prefix = _safe_ds_key(ds_name)
            mlflow.log_metrics({
                f"{prefix}_accuracy":   round(res["accuracy"], 4),
                f"{prefix}_robustness": round(res["robustness"], 4),
            })


def finalize_mlflow_run(
    parent_run_id: str,
    study: "optuna.Study",
    output_dir: str,
    importances: dict | None,
) -> None:
    """Log study-level summary and artifacts to the parent MLflow run."""
    best = study.best_trial
    best_acc = best.user_attrs.get("avg_accuracy", 0.0)
    best_rob = best.user_attrs.get("avg_robustness", 0.0)

    with mlflow.start_run(run_id=parent_run_id):
        # Summary metrics
        mlflow.log_metrics({
            "best_combined_score": round(best.value, 4),
            "best_avg_accuracy":   round(best_acc, 4),
            "best_avg_robustness": round(best_rob, 4),
            "n_trials_completed":  sum(
                1 for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ),
        })

        # Best hyperparameters as params on the parent run
        mlflow.log_params({f"best_{k}": v for k, v in best.params.items()
                           if k not in ("score_max_samples", "score_batch_size")})

        # Importance scores as params (0-1 floats are fine as params)
        if importances:
            mlflow.log_params({f"importance_{k}": round(v, 4)
                               for k, v in importances.items()})

        # Artifact: best config JSON
        best_cfg_path = os.path.join(output_dir, "best_config_optuna.json")
        if os.path.exists(best_cfg_path):
            mlflow.log_artifact(best_cfg_path, artifact_path="results")

        # Artifact: plain-text summary
        summary_lines = [
            f"Study       : {study.study_name}",
            f"Best trial  : #{best.number}",
            f"Combined    : {best.value:.4f}",
            f"Avg acc     : {best_acc:.4f}",
            f"Avg rob     : {best_rob:.4f}",
            "",
            "Best hyperparameters:",
        ] + [f"  {k:25s} = {v}" for k, v in best.params.items()
             if k not in ("score_max_samples", "score_batch_size")]

        if importances:
            summary_lines += ["", "Hyperparameter importances:"]
            for param, imp in importances.items():
                bar = "#" * int(imp * 30)
                summary_lines.append(f"  {param:25s} {imp:.3f}  {bar}")

        summary_lines += [
            "",
            "Baseline comparison:",
            f"  Grid-search best : Acc=75.8%, Rob=39.9%",
            f"  GS Only          : Acc=81.0%, Rob=51.3%",
            f"  Group DRO        : Acc=78.2%, Rob=48.2%",
            f"  Delta vs Grid    : Acc={best_acc-0.758:+.1%}, Rob={best_rob-0.399:+.1%}",
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="sart_summary_"
        ) as f:
            f.write("\n".join(summary_lines) + "\n")
            tmp_path = f.name
        mlflow.log_artifact(tmp_path, artifact_path="results")
        os.unlink(tmp_path)


# ──────────────────────────────────────────────────────────────────────────────
# Objective
# ──────────────────────────────────────────────────────────────────────────────

def make_objective(
    datasets: dict,
    device: str,
    smoke_test: bool = False,
    mlflow_parent_run_id: str | None = None,
    study_name: str = "sart_optuna",
):
    """Return an Optuna objective closed over datasets, device, and MLflow state."""

    def objective(trial: "optuna.Trial") -> float:
        t0 = time.time()

        # ── Suggest hyperparameters ───────────────────────────────────────
        cfg = {
            "lambda_":           trial.suggest_float("lambda_",          0.1,  3.0, log=True),
            "gamma":             trial.suggest_float("gamma",            0.05, 1.0),
            "rho":               trial.suggest_float("rho",              0.05, 1.0),
            "tau_A":             trial.suggest_float("tau_A",            0.05, 0.5),
            "tau_R":             trial.suggest_float("tau_R",            0.1,  0.9),
            "phase3_lr_factor":  trial.suggest_float("phase3_lr_factor", 0.1,  1.0),
            "score_max_samples": 200 if smoke_test else C.score_max_samples,
            "score_batch_size":  1   if smoke_test else C.score_batch_size,
        }

        # ── Train & evaluate ──────────────────────────────────────────────
        acc_list, rob_list = [], []
        per_dataset: dict[str, dict] = {}

        for ds_name, ds in datasets.items():
            set_seed(42)
            model = create_model()
            if hasattr(model, "module"):
                model = model.module
            model = model.to(device)

            try:
                model = train_our_method(
                    model, ds,
                    use_reweighting=True,
                    use_gradient_surgery=True,
                    device=device,
                    verbose=False,
                    cfg=dict(cfg),
                )
                result = run_full_evaluation(model, ds, device=device)
                acc = float(result["accuracy_clean"])
                rob = float(result["robustness"])
                acc_list.append(acc)
                rob_list.append(rob)
                per_dataset[ds_name] = {"accuracy": acc, "robustness": rob}
            finally:
                del model
                if "cuda" in device:
                    torch.cuda.empty_cache()

        avg_acc  = float(np.mean(acc_list))
        avg_rob  = float(np.mean(rob_list))
        combined = 0.4 * avg_acc + 0.6 * avg_rob
        elapsed  = time.time() - t0

        # ── Store in Optuna user attrs ────────────────────────────────────
        trial.set_user_attr("avg_accuracy",  round(avg_acc, 4))
        trial.set_user_attr("avg_robustness", round(avg_rob, 4))
        trial.set_user_attr("per_dataset",   {
            k: {m: round(v, 4) for m, v in res.items()}
            for k, res in per_dataset.items()
        })

        # ── Log to MLflow ─────────────────────────────────────────────────
        if mlflow_parent_run_id is not None and _MLFLOW_AVAILABLE:
            try:
                log_trial_to_mlflow(
                    trial, per_dataset, avg_acc, avg_rob, combined,
                    elapsed, mlflow_parent_run_id, study_name,
                )
            except Exception as exc:
                print(f"  [MLflow] Logging failed for trial {trial.number}: {exc}")

        return combined

    return objective


# ──────────────────────────────────────────────────────────────────────────────
# Terminal report + JSON backup
# ──────────────────────────────────────────────────────────────────────────────

def report_study(study: "optuna.Study", output_dir: str) -> dict | None:
    """Print summary to stdout, save best_config JSON, return importances."""
    best = study.best_trial
    all_ok = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    ranked = sorted(all_ok, key=lambda t: t.value, reverse=True)

    acc = best.user_attrs.get("avg_accuracy", 0.0)
    rob = best.user_attrs.get("avg_robustness", 0.0)

    print("\n" + "=" * 70)
    print(f"OPTUNA SEARCH COMPLETE — {len(all_ok)} successful trials")
    print("=" * 70)
    print(f"\nBest trial  : #{best.number}")
    print(f"Combined    : {best.value:.4f}")
    print(f"Avg acc     : {acc:.4f}")
    print(f"Avg rob     : {rob:.4f}")
    print("\nBest hyperparameters:")
    for k, v in best.params.items():
        if k not in ("score_max_samples", "score_batch_size"):
            print(f"  {k:25s} = {v:.4g}")

    print("\nTop-5 trials:")
    for t in ranked[:5]:
        ps = ", ".join(
            f"{k}={v:.3g}" for k, v in t.params.items()
            if k not in ("score_max_samples", "score_batch_size")
        )
        print(f"  #{t.number:4d}  combined={t.value:.4f}  [{ps}]")

    print("\n" + "=" * 70)
    print("COMPARISON WITH BASELINES")
    print("=" * 70)
    print(f"  Best Optuna SART : Acc={acc:.1%}, Rob={rob:.1%}")
    print(f"  Grid-search best : Acc=75.8%, Rob=39.9%  (λ=2.0, γ=0.8, ρ=0.7)")
    print(f"  GS Only          : Acc=81.0%, Rob=51.3%")
    print(f"  Group DRO        : Acc=78.2%, Rob=48.2%")
    print(f"  Δ vs Grid Best   : Acc={acc-0.758:+.1%}, Rob={rob-0.399:+.1%}")

    # ── Save best-config JSON (small, handy for config.py update) ─────────
    os.makedirs(output_dir, exist_ok=True)
    best_cfg = {
        "trial_number":    best.number,
        "combined_score":  round(best.value, 4),
        "avg_accuracy":    round(acc, 4),
        "avg_robustness":  round(rob, 4),
        "params":          {k: v for k, v in best.params.items()
                            if k not in ("score_max_samples", "score_batch_size")},
    }
    best_path = os.path.join(output_dir, "best_config_optuna.json")
    with open(best_path, "w") as f:
        json.dump(best_cfg, f, indent=2)
    print(f"\nBest config saved to {best_path}")

    # ── Param importance ──────────────────────────────────────────────────
    importances = None
    try:
        importances = optuna.importance.get_param_importances(study)
        print("\nHyperparameter importances:")
        for param, imp in importances.items():
            bar = "#" * int(imp * 40)
            print(f"  {param:25s} {imp:.3f}  {bar}")
    except Exception as e:
        print(f"(Importance analysis skipped: {e})")

    return importances


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna + MLflow hyperparameter search for SART"
    )
    # Optuna
    parser.add_argument("--n-trials",   type=int,   default=100)
    parser.add_argument("--n-jobs",     type=int,   default=1,
                        help="Parallel workers (>1 requires --storage)")
    parser.add_argument("--timeout",    type=float, default=None,
                        help="Wall-clock budget in seconds")
    parser.add_argument("--smoke-test", action="store_true",
                        help="5 trials, local data settings (~3 min)")
    parser.add_argument("--storage",    type=str,   default=None,
                        help="Optuna storage URL (sqlite:///... or postgresql://...)")
    parser.add_argument("--study-name", type=str,   default="sart_optuna")
    parser.add_argument("--sampler",    type=str,   default="tpe",
                        choices=["tpe", "cmaes", "random"])
    parser.add_argument("--output-dir", type=str,   default="hp_optuna",
                        help="Directory for best_config_optuna.json")
    parser.add_argument("--device",     type=str,   default=None)

    # MLflow
    parser.add_argument("--mlflow-uri",        type=str, default="mlruns",
                        help="MLflow tracking URI (default: local ./mlruns)")
    parser.add_argument("--mlflow-experiment", type=str, default="SART-HP-Search",
                        help="MLflow experiment name")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Disable MLflow tracking")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device = args.device or _device_for_job(0)
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    print(f"Profile  : {PROFILE}")
    print(f"Device   : {device}  (GPUs: {num_gpus})")
    print(f"Trials   : {args.n_trials}  |  Jobs: {args.n_jobs}")
    print(f"Sampler  : {args.sampler.upper()}")

    # ── Auto-storage for parallel jobs ────────────────────────────────────
    if args.n_jobs > 1 and args.storage is None:
        args.storage = f"sqlite:///{args.output_dir}/study.db"
        print(f"Auto storage: {args.storage}")

    # ── MLflow setup ──────────────────────────────────────────────────────
    use_mlflow = (not args.no_mlflow) and _MLFLOW_AVAILABLE
    if not args.no_mlflow and not _MLFLOW_AVAILABLE:
        print("WARNING: mlflow not installed — tracking disabled. "
              "Run: pip install mlflow")

    mlflow_parent_run_id: str | None = None
    if use_mlflow:
        setup_mlflow(args.mlflow_uri, args.mlflow_experiment)
        parent_run = mlflow.start_run(
            run_name=args.study_name,
            tags={
                "profile":    PROFILE,
                "sampler":    args.sampler,
                "n_trials":   str(args.n_trials),
                "study_name": args.study_name,
            },
        )
        mlflow_parent_run_id = parent_run.info.run_id
        tracking_uri = mlflow.get_tracking_uri()
        print(f"\nMLflow tracking URI : {tracking_uri}")
        print(f"MLflow experiment   : {args.mlflow_experiment}")
        print(f"Parent run ID       : {mlflow_parent_run_id}")
        if tracking_uri.startswith("mlruns") or tracking_uri == "mlruns":
            print("View UI             : mlflow ui  (then open http://127.0.0.1:5000)")
        else:
            print(f"View UI             : {tracking_uri}")

    # ── Sampler ───────────────────────────────────────────────────────────
    seed = 42
    if args.sampler == "tpe":
        sampler = TPESampler(seed=seed)
    elif args.sampler == "cmaes":
        from optuna.samplers import CmaEsSampler
        sampler = CmaEsSampler(seed=seed)
    else:
        from optuna.samplers import RandomSampler
        sampler = RandomSampler(seed=seed)

    # ── Study ─────────────────────────────────────────────────────────────
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        sampler=sampler,
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=0),
        storage=args.storage,
        load_if_exists=True,
    )

    # ── Datasets ──────────────────────────────────────────────────────────
    print("\nGenerating datasets...")
    datasets = {
        "Math-Arithmetic":    generate_math_dataset(seed=42),
        "Financial-Analysis": generate_financial_dataset(seed=43),
        "Causal-Reasoning":   generate_causal_dataset(seed=44),
    }
    print(f"Datasets  : {list(datasets.keys())}")
    print(f"Train size: {len(datasets['Math-Arithmetic']['train'])}")

    n_trials = 5 if args.smoke_test else args.n_trials

    # ── Objective with console logging ────────────────────────────────────
    _objective = make_objective(
        datasets, device,
        smoke_test=args.smoke_test,
        mlflow_parent_run_id=mlflow_parent_run_id,
        study_name=args.study_name,
    )
    counter = {"n": 0}

    def objective_logged(trial):
        t0 = time.time()
        value = _objective(trial)
        elapsed = time.time() - t0
        counter["n"] += 1
        acc = trial.user_attrs.get("avg_accuracy", 0)
        rob = trial.user_attrs.get("avg_robustness", 0)
        ps = ", ".join(
            f"{k}={v:.3g}" for k, v in trial.params.items()
            if k not in ("score_max_samples", "score_batch_size")
        )
        print(
            f"[{counter['n']:3d}/{n_trials}] Trial #{trial.number:4d}  "
            f"acc={acc:.3f} rob={rob:.3f} combined={value:.4f}  "
            f"({elapsed:.0f}s)  [{ps}]"
        )
        return value

    # ── Run ───────────────────────────────────────────────────────────────
    print(f"\nStarting Optuna optimisation ({n_trials} trials)...\n")
    study.optimize(
        objective_logged,
        n_trials=n_trials,
        n_jobs=args.n_jobs,
        timeout=args.timeout,
        gc_after_trial=True,
        show_progress_bar=False,
    )

    # ── Terminal report + best-config JSON ───────────────────────────────
    importances = report_study(study, args.output_dir)

    # ── Finalize MLflow parent run ────────────────────────────────────────
    if use_mlflow and mlflow_parent_run_id:
        try:
            finalize_mlflow_run(
                mlflow_parent_run_id, study, args.output_dir, importances
            )
        except Exception as e:
            print(f"[MLflow] Failed to finalize parent run: {e}")
        finally:
            mlflow.end_run()
        print(f"\nAll trial results recorded in MLflow experiment '{args.mlflow_experiment}'.")
        print("Launch UI with:  mlflow ui")

    # ── Next steps ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("NEXT STEPS")
    print("=" * 70)
    print("1. Update src/config.py with the best hyperparameters shown above.")
    print("2. Re-run full experiments:")
    print("   nohup env EXPERIMENT_SCALE=server python3 run_all.py > run_final.log 2>&1 &")
    if args.storage:
        print("3. Resume this study with more trials:")
        print(f"   python3 hp_optuna.py --storage {args.storage} \\")
        print(f"     --study-name {args.study_name} --n-trials 50")


if __name__ == "__main__":
    main()
