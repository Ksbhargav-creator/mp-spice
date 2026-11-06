#!/usr/bin/env bash
# mp-spice -- rerun the experiments behind the README's results tables on a
# compute cluster, then regenerate the tables so they can be diffed against
# the README.
#
#   reproduce_readme.sh list                       # matrices + their array index
#   reproduce_readme.sh run   [opts]               # run solver experiments
#   reproduce_readme.sh merge [opts]               # combine runs -> csv, plots, tables
#   reproduce_readme.sh all   [opts]               # run, then merge (default)
#
# Options (for run/merge/all):
#   --out DIR         results root (default /results -- mount a host/shared dir here)
#   --matrices "a b"  matrices to run (default: the README set, see `list`)
#   --index N         run ONLY the Nth matrix (0-based) of the list. This is the
#                     array-job hook: Slurm `--array=0-13` -> `--index $SLURM_ARRAY_TASK_ID`;
#                     a Kubernetes Indexed Job -> `--index $JOB_COMPLETION_INDEX`.
#                     SHARD_INDEX / SLURM_ARRAY_TASK_ID / JOB_COMPLETION_INDEX in
#                     the environment are honoured if --index isn't given (first set wins).
#   --jobs N          run N matrices concurrently inside this one container (default 1)
#
# Why each matrix gets its own working directory: klu_quire_IR_study APPENDS to
# a relative csv/ directory. Two matrices running concurrently in one cwd would
# interleave writes into the same files (and a rerun would silently double-count
# rows -- the data-hygiene bug this repo has already been bitten by). So every
# matrix runs in $OUT/runs/<matrix>/ from a clean slate, and `merge` stitches the
# per-matrix CSVs together afterwards.
#
# Layout under $OUT after run+merge:
#   runs/<matrix>/{<matrix>.log, <matrix>.status, csv/*.csv}   raw per-matrix output
#   logs/<matrix>_sweep.log                                    same logs, sweep-style names
#   csv/{product_magnitude,dynamic_range}_histogram.csv        merged
#   csv/dynamic_range_{normalized,summary}.csv, csv/*.png      normalize + plot output
#   readme_tables.md                                           regenerated tables + diff vs README
#   regression_check.txt, regression_history.csv               invariant/drift check
#   PROVENANCE.txt                                             what exactly was run

set -uo pipefail

ROOT="${MPSPICE_ROOT:-/opt/mp-spice}"
BIN="${BIN:-/usr/local/bin/klu_quire_IR_study}"
BIN_DIRECT="${BIN_DIRECT:-/usr/local/bin/klu_quire_study}"
DIRECT_MATRICES="${DIRECT_MATRICES:-add32}"   # space separated; matrices in the README's direct-solve table
DATA_DIR="${DATA_DIR:-$ROOT/data}"
export DATA_DIR BIN_DIRECT DIRECT_MATRICES

# The matrices behind the README tables, in a FIXED order (array indices depend on it).
# rajat03 is last: its fixed-code result is still "pending" in the README, and it is
# the slow one that previously timed out in the sandbox.
DEFAULT_MATRICES=(rajat04 rajat05 rajat11 rajat12 rajat13 rajat14 rajat15 rajat16 rajat17 rajat19 add20 add32 rajat30 rajat03)

cmd="${1:-all}"
[ "$#" -gt 0 ] && shift

OUT="/results"
MATRICES=()
INDEX="${SHARD_INDEX:-${SLURM_ARRAY_TASK_ID:-${JOB_COMPLETION_INDEX:-}}}"
JOBS=1

while [ "$#" -gt 0 ]; do
    case "$1" in
        --out)       OUT="$2"; shift 2 ;;
        --matrices)  read -r -a MATRICES <<< "$2"; shift 2 ;;
        --index)     INDEX="$2"; shift 2 ;;
        --jobs)      JOBS="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ "${#MATRICES[@]}" -eq 0 ] && MATRICES=("${DEFAULT_MATRICES[@]}")

# ---------------------------------------------------------------------------
run_one() {
    local m="$1"
    local dir="$OUT/runs/$m"
    local mtx="$DATA_DIR/$m/$m.mtx"

    rm -rf "$dir"            # clean slate: a rerun must REPLACE, never append to, old rows
    mkdir -p "$dir"

    if [ ! -f "$mtx" ]; then
        echo "[$m] matrix file missing, trying to fetch ..."
        "$ROOT/scripts/fetch_matrices.sh" "$m" || true
    fi
    if [ ! -f "$mtx" ]; then
        echo "missing 0 0" > "$dir/$m.status"
        echo "[$m] MISSING -- $mtx not found and could not be fetched (compute nodes without internet need the matrices baked into the image or mounted at \$DATA_DIR)" >&2
        return 1
    fi

    echo "[$m] starting ($(date -u +%H:%M:%SZ))"
    local start end code
    start=$(date +%s)
    ( cd "$dir" && "$BIN" "$mtx" > "$m.log" 2>&1 )
    code=$?
    # README's "Quire in the KLU factorization" direct-solve table comes from the
    # OTHER app (klu_quire_study, no iterative refinement). Run it for the matrices
    # that table covers; its outcome never changes this matrix's ok/fail status.
    if [[ " $DIRECT_MATRICES " == *" $m "* ]] && [ -x "$BIN_DIRECT" ]; then
        ( cd "$dir" && "$BIN_DIRECT" "$mtx" > "$m.direct.log" 2>&1 ) \
            || echo "[$m] note: klu_quire_study exited nonzero (see $dir/$m.direct.log)" >&2
    fi
    end=$(date +%s)
    if [ "$code" -eq 0 ]; then
        echo "ok $code $((end - start))" > "$dir/$m.status"
        echo "[$m] done in $((end - start))s"
    else
        echo "fail $code $((end - start))" > "$dir/$m.status"
        echo "[$m] FAILED (exit $code) after $((end - start))s -- see $dir/$m.log" >&2
    fi
    return "$code"
}

do_run() {
    mkdir -p "$OUT/runs"
    local todo=("${MATRICES[@]}")
    if [ -n "$INDEX" ]; then
        if ! [[ "$INDEX" =~ ^[0-9]+$ ]] || [ "$INDEX" -ge "${#MATRICES[@]}" ]; then
            echo "error: index '$INDEX' out of range for ${#MATRICES[@]} matrices (valid: 0-$(( ${#MATRICES[@]} - 1 )))" >&2
            return 2
        fi
        todo=("${MATRICES[$INDEX]}")
        echo "array shard $INDEX -> ${todo[0]}"
    fi

    local rc=0
    if [ "$JOBS" -le 1 ] || [ "${#todo[@]}" -le 1 ]; then
        for m in "${todo[@]}"; do run_one "$m" || rc=1; done
    else
        export -f run_one
        export OUT DATA_DIR BIN ROOT
        printf '%s\n' "${todo[@]}" | xargs -P "$JOBS" -I{} bash -c 'run_one {}' || rc=1
    fi
    return "$rc"
}

# ---------------------------------------------------------------------------
merge_csv() {   # merge_csv <filename>  -> $OUT/csv/<filename>, header once
    local name="$1" dest="$OUT/csv/$1" header_done=0 f
    : > "$dest"
    for m in "${MATRICES[@]}"; do
        f="$OUT/runs/$m/csv/$name"
        [ -f "$f" ] || continue
        if [ "$header_done" -eq 0 ]; then
            cat "$f" >> "$dest"; header_done=1
        else
            tail -n +2 "$f" >> "$dest"
        fi
    done
}

do_merge() {
    mkdir -p "$OUT/csv" "$OUT/logs"

    local have=() missing=() failed=()
    for m in "${MATRICES[@]}"; do
        local st="$OUT/runs/$m/$m.status"
        if [ ! -f "$st" ]; then
            missing+=("$m")                       # never ran / killed before finishing (e.g. Slurm time limit)
        elif grep -q '^ok' "$st"; then
            have+=("$m")
            cp "$OUT/runs/$m/$m.log" "$OUT/logs/${m}_sweep.log"
            [ -f "$OUT/runs/$m/$m.direct.log" ] && cp "$OUT/runs/$m/$m.direct.log" "$OUT/logs/$m.direct.log"
        else
            failed+=("$m")
            [ -f "$OUT/runs/$m/$m.log" ] && cp "$OUT/runs/$m/$m.log" "$OUT/logs/${m}_sweep.log"
        fi
    done
    echo "merge: ${#have[@]} ok, ${#failed[@]} failed, ${#missing[@]} never finished"
    [ "${#failed[@]}" -gt 0 ]  && echo "  failed:  ${failed[*]}"
    [ "${#missing[@]}" -gt 0 ] && echo "  missing: ${missing[*]}"
    if [ "${#have[@]}" -eq 0 ]; then
        echo "nothing to merge" >&2
        return 1
    fi

    merge_csv product_magnitude_histogram.csv
    merge_csv dynamic_range_histogram.csv

    export MPLBACKEND=Agg
    ( cd "$OUT/csv" && python3 "$ROOT/csv/normalize_dynamic_range.py" > normalize.log 2>&1 ) \
        || echo "warning: normalize_dynamic_range.py failed (see $OUT/csv/normalize.log)" >&2
    ( cd "$OUT/csv" && python3 "$ROOT/csv/plot_dynamic_range.py"      > plot.log      2>&1 ) \
        || echo "warning: plot_dynamic_range.py failed (see $OUT/csv/plot.log)" >&2

    # Invariant + drift check, only over matrices that finished.
    local logs=()
    for m in "${have[@]}"; do logs+=("$OUT/logs/${m}_sweep.log"); done
    python3 "$ROOT/ci/regression_check.py" --logs "${logs[@]}" \
        --baseline "$ROOT/ci/baseline.json" --history "$OUT/regression_history.csv" \
        > "$OUT/regression_check.txt" 2>&1
    echo "regression_check exit status: $? (details: $OUT/regression_check.txt)"

    python3 "$ROOT/scripts/make_readme_tables.py" \
        --logs-dir "$OUT/logs" --csv-dir "$OUT/csv" --readme "$ROOT/README.md" \
        --missing "${missing[*]:-}" --failed "${failed[*]:-}" \
        > "$OUT/readme_tables.md"
    echo "wrote $OUT/readme_tables.md"

    { cat "$ROOT/PROVENANCE.txt" 2>/dev/null || echo "(no PROVENANCE.txt baked into image)"
      echo; echo "merged at: $(date -u +%FT%TZ)"
      echo "matrices ok: ${have[*]}"; } > "$OUT/PROVENANCE.txt"
}

# ---------------------------------------------------------------------------
case "$cmd" in
    list)
        i=0; for m in "${DEFAULT_MATRICES[@]}"; do echo "$i  $m"; i=$((i+1)); done ;;
    run)   do_run ;;
    merge) do_merge ;;
    all)   do_run; run_rc=$?; do_merge; merge_rc=$?
           # nonzero if ANY matrix failed or the merge failed, so a scheduler marks the job failed
           if [ "$run_rc" -ne 0 ] || [ "$merge_rc" -ne 0 ]; then exit 1; fi
           exit 0 ;;
    *) echo "usage: $0 {list|run|merge|all} [--out DIR] [--matrices \"a b\"] [--index N] [--jobs N]" >&2; exit 2 ;;
esac
