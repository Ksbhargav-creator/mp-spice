#!/usr/bin/env bash
# mp-spice -- regression entrypoint run inside the container by either the
# fast Job (push/PR-triggered) or the weekly CronJob (environment-drift
# heartbeat). Fetches whichever matrices are missing, runs
# klu_quire_IR_study against each, then hands every produced log to
# regression_check.py for the invariant + drift checks.
#
# Usage: run_regression.sh <matrix> [<matrix> ...]
#   run_regression.sh rajat11 rajat14                       # fast set
#   run_regression.sh rajat04 rajat05 rajat11 rajat12 \      # full set
#     rajat13 rajat14 rajat15 rajat16 rajat17 rajat19 \
#     add20 add32 rajat30 rajat01 rajat07 rajat08 rajat09 rajat10
#
# Expects to run from inside the container (see Dockerfile: binaries are on
# PATH at /usr/local/bin, scripts/ci live under /opt/mp-spice). HISTORY_DIR
# should be a mounted PVC, not the container's ephemeral filesystem, or every
# run's history is lost when the pod terminates.

set -uo pipefail

BIN="/usr/local/bin/klu_quire_IR_study"
DATA_DIR="${DATA_DIR:-/data}"                 # PVC-backed matrix cache
LOG_DIR="${LOG_DIR:-/tmp/regression_logs}"    # ephemeral -- logs are consumed this run only
HISTORY_DIR="${HISTORY_DIR:-/history}"        # PVC-backed -- must persist across runs
BASELINE="${BASELINE:-/opt/mp-spice/ci/baseline.json}"  # baked into the image, human-updated via PR
FETCH="/opt/mp-spice/scripts/fetch_matrices.sh"
export DATA_DIR   # fetch_matrices.sh reads it from the environment to decide where matrices land

mkdir -p "$LOG_DIR" "$HISTORY_DIR"

if [ "$#" -eq 0 ]; then
    echo "usage: $0 <matrix> [<matrix> ...]" >&2
    exit 2
fi

echo "Fetching any missing matrices into $DATA_DIR ..."
for m in "$@"; do
    if [ ! -f "$DATA_DIR/$m/$m.mtx" ]; then
        "$FETCH" "$m" || echo "warning: fetch failed for $m (will surface as a load failure below)"
    fi
done

echo "Running $# matrices through klu_quire_IR_study ..."
LOGS=()
for m in "$@"; do
    mtx="$DATA_DIR/$m/$m.mtx"
    log="$LOG_DIR/${m}_sweep.log"
    if [ -f "$mtx" ]; then
        "$BIN" "$mtx" > "$log" 2>&1
    else
        echo "Loading data/$m/$m.mtx" > "$log"
        echo "fetch failed -- no matrix file available" >> "$log"
    fi
    LOGS+=("$log")
done

echo "Running regression check ..."
python3 /opt/mp-spice/ci/regression_check.py \
    --logs "${LOGS[@]}" \
    --baseline "$BASELINE" \
    --history "$HISTORY_DIR/regression_history.csv"
STATUS=$?

echo "Exit status: $STATUS"
exit "$STATUS"
