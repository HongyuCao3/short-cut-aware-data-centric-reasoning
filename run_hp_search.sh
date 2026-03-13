#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_hp_search.sh  —  nohup wrapper for hp_optuna.py
#
# Commands:
#   bash run_hp_search.sh start   [extra hp_optuna.py args]
#   bash run_hp_search.sh resume  [extra hp_optuna.py args]
#   bash run_hp_search.sh status
#   bash run_hp_search.sh stop
#   bash run_hp_search.sh log     [N lines, default 50]
#   bash run_hp_search.sh ui      [port, default 5000]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

# ── Paths (all absolute so nohup never gets confused) ────────────────────────
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${REPO_DIR}/hp_optuna"
LOG_FILE="${OUTPUT_DIR}/optuna.log"
PID_FILE="${OUTPUT_DIR}/optuna.pid"
STORAGE="sqlite:///${OUTPUT_DIR}/study.db"
STUDY_NAME="sart_optuna"
MLFLOW_DIR="${REPO_DIR}/mlruns"
EXPERIMENT_SCALE="${EXPERIMENT_SCALE:-server}"

mkdir -p "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
_is_running() {
    [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

_pid() { cat "${PID_FILE}" 2>/dev/null || echo "none"; }

# ─────────────────────────────────────────────────────────────────────────────
cmd_start() {
    if _is_running; then
        echo "Already running (PID $(_pid)). Use 'status' or 'stop' first."
        exit 1
    fi

    echo "Starting hp_optuna.py in background..."
    echo "  Log     : ${LOG_FILE}"
    echo "  DB      : ${OUTPUT_DIR}/study.db"
    echo "  MLflow  : ${MLFLOW_DIR}"
    echo ""

    nohup env EXPERIMENT_SCALE="${EXPERIMENT_SCALE}" \
        python3 "${REPO_DIR}/hp_optuna.py" \
            --output-dir  "${OUTPUT_DIR}" \
            --storage     "${STORAGE}" \
            --study-name  "${STUDY_NAME}" \
            --mlflow-uri  "${MLFLOW_DIR}" \
            "$@" \
        >> "${LOG_FILE}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "Started — PID $(cat "${PID_FILE}")"
    echo ""
    echo "Monitor:"
    echo "  bash run_hp_search.sh log      # live tail"
    echo "  bash run_hp_search.sh status   # progress summary"
    echo "  bash run_hp_search.sh ui       # open MLflow UI"
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_resume() {
    # Same as start but passes --storage so Optuna resumes the existing study.
    # Extra args are forwarded (e.g. --n-trials 50).
    echo "Resuming study '${STUDY_NAME}'..."
    cmd_start "$@"
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_status() {
    echo "── Process ──────────────────────────────────────────"
    if _is_running; then
        PID=$(_pid)
        echo "  Status  : RUNNING (PID ${PID})"
        echo "  Elapsed : $(ps -o etime= -p "${PID}" 2>/dev/null | tr -d ' ')"
        echo "  CPU/Mem : $(ps -o %cpu,%mem -p "${PID}" 2>/dev/null | tail -1)"
    else
        echo "  Status  : NOT RUNNING"
    fi

    echo ""
    echo "── Optuna trials ────────────────────────────────────"
    if [[ -f "${OUTPUT_DIR}/study.db" ]]; then
        python3 - <<PYEOF
import optuna, sys
optuna.logging.set_verbosity(optuna.logging.WARNING)
try:
    s = optuna.load_study("${STUDY_NAME}", storage="${STORAGE}")
    done = [t for t in s.trials if t.state.name == "COMPLETE"]
    run  = [t for t in s.trials if t.state.name == "RUNNING"]
    fail = [t for t in s.trials if t.state.name == "FAIL"]
    print(f"  Completed : {len(done)}")
    print(f"  Running   : {len(run)}")
    print(f"  Failed    : {len(fail)}")
    if done:
        b = s.best_trial
        print(f"  Best      : trial #{b.number}  combined={b.value:.4f}  "
              f"acc={b.user_attrs.get('avg_accuracy',0):.4f}  "
              f"rob={b.user_attrs.get('avg_robustness',0):.4f}")
        print(f"  Best params:")
        for k, v in b.params.items():
            if k not in ("score_max_samples", "score_batch_size"):
                print(f"    {k:25s} = {v:.4g}")
except Exception as e:
    print(f"  (Could not load study: {e})")
PYEOF
    else
        echo "  No study DB found yet."
    fi

    echo ""
    echo "── Last log lines ───────────────────────────────────"
    if [[ -f "${LOG_FILE}" ]]; then
        tail -5 "${LOG_FILE}"
    else
        echo "  No log file yet."
    fi
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_stop() {
    if ! _is_running; then
        echo "Not running."
        return
    fi
    PID=$(_pid)
    echo "Sending SIGTERM to PID ${PID} (graceful — waits for current trial to finish)..."
    kill -TERM "${PID}"

    # Wait up to 60 s for graceful shutdown
    for i in $(seq 1 60); do
        sleep 1
        if ! _is_running; then
            echo "Process exited cleanly."
            rm -f "${PID_FILE}"
            return
        fi
        (( i % 10 == 0 )) && echo "  Still waiting... (${i}s)"
    done

    echo "Process did not exit in 60 s — sending SIGKILL."
    kill -KILL "${PID}" 2>/dev/null || true
    rm -f "${PID_FILE}"
    echo "Killed."
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_log() {
    N="${1:-50}"
    echo "── ${LOG_FILE} (last ${N} lines, live) ─────────────"
    tail -n "${N}" -f "${LOG_FILE}"
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_ui() {
    PORT="${1:-5000}"
    echo "Starting MLflow UI on port ${PORT}..."
    echo "  Tracking dir : ${MLFLOW_DIR}"
    echo "  URL          : http://$(hostname -f 2>/dev/null || hostname):${PORT}"
    echo "  (Ctrl+C to stop the UI — does NOT affect the running search)"
    echo ""
    mlflow ui --backend-store-uri "${MLFLOW_DIR}" --port "${PORT}"
}

# ─────────────────────────────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true

case "${CMD}" in
    start)   cmd_start  "$@" ;;
    resume)  cmd_resume "$@" ;;
    status)  cmd_status ;;
    stop)    cmd_stop ;;
    log)     cmd_log    "$@" ;;
    ui)      cmd_ui     "$@" ;;
    *)
        echo "Usage: bash run_hp_search.sh <command> [options]"
        echo ""
        echo "Commands:"
        echo "  start  [hp_optuna args]   Start a new search in the background"
        echo "  resume [hp_optuna args]   Resume an existing study (same DB)"
        echo "  status                    Show PID, trial counts, and best result"
        echo "  stop                      Graceful shutdown (waits for current trial)"
        echo "  log    [N=50]             Live tail of the log file"
        echo "  ui     [port=5000]        Launch MLflow UI"
        echo ""
        echo "Examples:"
        echo "  bash run_hp_search.sh start --n-trials 100"
        echo "  bash run_hp_search.sh start --n-trials 120 --n-jobs 4 --sampler cmaes"
        echo "  bash run_hp_search.sh resume --n-trials 50"
        echo "  bash run_hp_search.sh status"
        echo "  bash run_hp_search.sh log 100"
        echo "  bash run_hp_search.sh ui 5001"
        ;;
esac
