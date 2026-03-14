#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_hp_search.sh  —  nohup wrapper for hp_optuna.py
#
# Commands:
#   bash run_hp_search.sh start   [extra hp_optuna.py args]
#   bash run_hp_search.sh resume  [extra hp_optuna.py args]
#   bash run_hp_search.sh status
#   bash run_hp_search.sh gpu
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
# SQLite backend avoids the Feb-2026 file-store deprecation warning
MLFLOW_URI="sqlite:///${REPO_DIR}/mlflow.db"
EXPERIMENT_SCALE="${EXPERIMENT_SCALE:-server}"

# Minimum free VRAM (MiB) required to run one server-scale trial
MIN_FREE_MIB=20000

mkdir -p "${OUTPUT_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
_is_running() {
    [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

_pid() { cat "${PID_FILE}" 2>/dev/null || echo "none"; }

# ── Print GPU summary and return the index of the GPU with most free memory ──
_best_gpu() {
    python3 - <<'PYEOF'
import subprocess, sys
try:
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,memory.free,memory.total,memory.used",
         "--format=csv,noheader,nounits"],
        text=True
    )
    best_idx, best_free = -1, -1
    for line in out.strip().splitlines():
        idx, free, total, used = [x.strip() for x in line.split(",")]
        free, total, used = int(free), int(total), int(used)
        print(f"  GPU {idx}: {free:6d} MiB free / {total} MiB total  ({used} used)")
        if free > best_free:
            best_free, best_idx = free, int(idx)
    print(f"  → Selected GPU {best_idx} ({best_free} MiB free)")
    sys.exit(0 if best_free >= 20000 else 2)
except FileNotFoundError:
    print("  nvidia-smi not found — will use CPU/MPS")
    sys.exit(1)
PYEOF
}

# Return just the index (used inside cmd_start)
_gpu_index() {
    python3 - <<'PYEOF'
import subprocess, sys
try:
    out = subprocess.check_output(
        ["nvidia-smi",
         "--query-gpu=index,memory.free",
         "--format=csv,noheader,nounits"],
        text=True
    )
    best_idx, best_free = 0, -1
    for line in out.strip().splitlines():
        idx, free = [x.strip() for x in line.split(",")]
        if int(free) > best_free:
            best_free, best_idx = int(free), int(idx)
    print(best_idx)
except Exception:
    print(0)
PYEOF
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_gpu() {
    echo "── GPU memory ───────────────────────────────────────"
    _best_gpu || true
    echo ""
    echo "── Active GPU processes ─────────────────────────────"
    nvidia-smi --query-compute-apps=pid,used_memory,name \
               --format=csv,noheader 2>/dev/null \
    | while IFS=, read -r pid mem name; do
        user=$(ps -o user= -p "${pid// /}" 2>/dev/null || echo "?")
        cmd=$(ps -o cmd= -p "${pid// /}" 2>/dev/null | cut -c1-80 || echo "?")
        echo "  PID ${pid// /}  user=${user}  mem=${mem}  ${cmd}"
    done
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_start() {
    if _is_running; then
        echo "Already running (PID $(_pid)). Use 'status' or 'stop' first."
        exit 1
    fi

    echo "── GPU check ────────────────────────────────────────"
    GPU_IDX=$(_gpu_index)
    _best_gpu || {
        EXIT=$?
        if [[ $EXIT -eq 2 ]]; then
            echo ""
            echo "WARNING: No GPU has ${MIN_FREE_MIB} MiB free."
            echo "  Other users may be occupying the GPUs (run: bash run_hp_search.sh gpu)"
            echo "  The search will still start on GPU ${GPU_IDX} but may OOM."
            echo "  Consider waiting for GPU memory to free up, or use --device cpu."
        fi
    }

    echo ""
    echo "Starting hp_optuna.py in background..."
    echo "  Device  : cuda:${GPU_IDX}  (auto-selected — most free memory)"
    echo "  Log     : ${LOG_FILE}"
    echo "  Optuna  : ${OUTPUT_DIR}/study.db"
    echo "  MLflow  : ${MLFLOW_URI}"
    echo ""

    # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True reduces fragmentation
    # and prevents cascade OOM when reserved-but-unallocated memory is large.
    nohup env \
        EXPERIMENT_SCALE="${EXPERIMENT_SCALE}" \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python3 "${REPO_DIR}/hp_optuna.py" \
            --output-dir  "${OUTPUT_DIR}" \
            --storage     "${STORAGE}" \
            --study-name  "${STUDY_NAME}" \
            --mlflow-uri  "${MLFLOW_URI}" \
            --device      "cuda:${GPU_IDX}" \
            --n-jobs      1 \
            "$@" \
        >> "${LOG_FILE}" 2>&1 &

    echo $! > "${PID_FILE}"
    echo "Started — PID $(cat "${PID_FILE}")"
    echo ""
    echo "Monitor:"
    echo "  bash run_hp_search.sh log      # live tail"
    echo "  bash run_hp_search.sh status   # progress + best result"
    echo "  bash run_hp_search.sh ui       # MLflow UI"
}

# ─────────────────────────────────────────────────────────────────────────────
cmd_resume() {
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
        # Show how much VRAM this process is using
        VRAM=$(nvidia-smi --query-compute-apps=pid,used_memory \
                          --format=csv,noheader,nounits 2>/dev/null \
               | awk -F, -v p="${PID}" '$1+0==p+0 {print $2}')
        [[ -n "${VRAM}" ]] && echo "  VRAM    : ${VRAM// /} MiB"
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
    echo "Sending SIGTERM to PID ${PID} (waits for current trial to finish, then closes MLflow)..."
    kill -TERM "${PID}"

    for i in $(seq 1 120); do
        sleep 1
        if ! _is_running; then
            echo "Process exited cleanly after ${i}s."
            rm -f "${PID_FILE}"
            return
        fi
        (( i % 15 == 0 )) && echo "  Still waiting... (${i}s) — finishing current trial"
    done

    echo "Process did not exit in 120 s — sending SIGKILL."
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
    echo "  Backend  : ${MLFLOW_URI}"
    echo "  URL      : http://$(hostname -f 2>/dev/null || hostname):${PORT}"
    echo "  (Ctrl+C stops the UI — does NOT affect the running search)"
    echo ""
    mlflow ui --backend-store-uri "${MLFLOW_URI}" --port "${PORT}"
}

# ─────────────────────────────────────────────────────────────────────────────
CMD="${1:-help}"
shift || true

case "${CMD}" in
    start)   cmd_start  "$@" ;;
    resume)  cmd_resume "$@" ;;
    status)  cmd_status ;;
    gpu)     cmd_gpu ;;
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
        echo "  gpu                       Show GPU memory usage by process"
        echo "  stop                      Graceful shutdown (waits for current trial)"
        echo "  log    [N=50]             Live tail of the log file"
        echo "  ui     [port=5000]        Launch MLflow UI"
        echo ""
        echo "Examples:"
        echo "  bash run_hp_search.sh start --n-trials 100"
        echo "  bash run_hp_search.sh resume --n-trials 50"
        echo "  bash run_hp_search.sh gpu"
        echo "  bash run_hp_search.sh status"
        echo "  bash run_hp_search.sh log 100"
        echo "  bash run_hp_search.sh ui 5001"
        ;;
esac
