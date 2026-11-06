#!/usr/bin/env bash
# Run klu_quire_IR_study against a list of matrices in sequence, so a batch
# of small/medium circuit matrices can be swept unattended instead of
# launching each one by hand. Every matrix's Table 1 + Table 2 results
# append to the same csv/product_magnitude_histogram.csv and
# csv/dynamic_range_histogram.csv files (that's how the app already writes
# them), plus a full stdout transcript per matrix under logs/ so nothing is
# lost even if you're not watching it live.
#
# Usage (run from the mp-spice repo root):
#   scripts/run_matrix_sweep.sh                     # runs the default Tier 1 list below
#   scripts/run_matrix_sweep.sh rajat07 rajat08      # or pass your own matrix names
#
# Safe to re-run: csv/normalize_dynamic_range.py already drops duplicate
# rows, so re-sweeping a matrix you've already run isn't harmful, just
# wasteful of time.

set -uo pipefail

BIN="build/applications/klu_quire_IR_study/klu_quire_IR_study"
DATA_DIR="data"
LOG_DIR="logs"

if [ ! -x "$BIN" ]; then
    echo "error: $BIN not found or not executable -- build first:"
    echo "  cmake -B build -DCMAKE_BUILD_TYPE=Release -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 -Wno-dev"
    echo "  cmake --build build -j"
    exit 1
fi

# Default: Tier 1 (small/fast, 135-7.6K rows) -- safe to run without
# supervision. Override by passing matrix names as arguments.
DEFAULT_MATRICES=(rajat11 rajat14 rajat05 rajat19 rajat04 rajat12 add20 add32 rajat01 rajat13 rajat03)
MATRICES=("${@:-${DEFAULT_MATRICES[@]}}")

mkdir -p "$LOG_DIR"

echo "Sweeping ${#MATRICES[@]} matrices: ${MATRICES[*]}"
echo

for m in "${MATRICES[@]}"; do
    mtx="$DATA_DIR/$m/$m.mtx"
    log="$LOG_DIR/${m}_sweep.log"

    if [ ! -f "$mtx" ]; then
        echo "[$m] SKIP -- $mtx not found"
        continue
    fi

    echo "[$m] starting... (log: $log)"
    start=$(date +%s)

    if "$BIN" "$mtx" > "$log" 2>&1; then
        end=$(date +%s)
        echo "[$m] done in $((end - start))s"
        # Pull out just the two summary tables so you can see progress
        # without opening every log file.
        grep -A 4 "Native KLU + double-residual" "$log" | tail -3
        grep -A 4 "Working-precision residual" "$log" | tail -3
        echo
    else
        end=$(date +%s)
        echo "[$m] FAILED after $((end - start))s -- see $log"
        echo
    fi
done

echo "Sweep complete. Full per-matrix output is under $LOG_DIR/."
echo "Next: cd csv && python3 normalize_dynamic_range.py && python3 plot_dynamic_range.py"
