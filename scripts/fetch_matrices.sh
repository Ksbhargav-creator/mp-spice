#!/usr/bin/env bash
# Fetch circuit-simulation matrices from the SuiteSparse Matrix Collection
# (https://sparse.tamu.edu/) into ./data. Matrices are NOT committed to the repo.
#
# Usage:
#   scripts/fetch_matrices.sh                    # legacy default: the rajat30 demo matrix
#   scripts/fetch_matrices.sh all                # legacy: rajat30 + circuit5M (~hundreds of MB)
#   scripts/fetch_matrices.sh rajat11 add20      # any named matrices (group looked up below)
#   scripts/fetch_matrices.sh Rajat/rajat11      # or spell out Group/name for anything not in the table
#
# Environment:
#   DATA_DIR            where matrices land (default: <repo>/data). Each matrix
#                       ends up at $DATA_DIR/<name>/<name>.mtx.
#   SUITESPARSE_BASE    download base URL (default: the herokuapp mirror
#                       sparse.tamu.edu links to for Matrix Market tarballs).
#
# Exit status is nonzero if ANY requested matrix failed to download, but the
# remaining matrices are still attempted -- one bad name shouldn't hide the rest.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="${DATA_DIR:-$ROOT/data}"
BASE="${SUITESPARSE_BASE:-https://suitesparse-collection-website.herokuapp.com/MM}"
mkdir -p "$DATA"

# Group for each matrix name this project has used. Anything else: pass Group/name.
group_of() {
    case "$1" in
        rajat*)                                              echo Rajat ;;
        add20|add32|memplus|scircuit|bcircuit|hcircuit)      echo Hamm ;;
        Freescale1|Freescale2|memchip|transient|circuit5M|circuit5M_dc|FullChip|nxp1)
                                                             echo Freescale ;;
        circuit_[1-4])                                       echo Bomhof ;;
        meg1|meg4)                                           echo Grund ;;
        *) return 1 ;;
    esac
}

FAILED=()

fetch() {
    local group="$1" name="$2"
    local url="$BASE/$group/$name.tar.gz"
    local tarball="$DATA/$name.tar.gz"
    if [ -f "$DATA/$name/$name.mtx" ]; then
        echo "already have $name"
        return 0
    fi
    echo "downloading $group/$name ..."
    if ! curl -L --fail --retry 5 --retry-delay 5 -sS -o "$tarball" "$url"; then
        echo "error: download failed for $group/$name ($url)" >&2
        rm -f "$tarball"
        FAILED+=("$group/$name")
        return 1
    fi
    if ! tar -xzf "$tarball" -C "$DATA"; then
        echo "error: could not extract $tarball" >&2
        rm -f "$tarball"
        FAILED+=("$group/$name")
        return 1
    fi
    rm -f "$tarball"
    if [ ! -f "$DATA/$name/$name.mtx" ]; then
        echo "error: expected $DATA/$name/$name.mtx after extraction" >&2
        FAILED+=("$group/$name")
        return 1
    fi
    echo "extracted to $DATA/$name/$name.mtx"
}

fetch_named() {
    local arg="$1" group name
    if [[ "$arg" == */* ]]; then
        group="${arg%%/*}"; name="${arg#*/}"
    else
        name="$arg"
        if ! group="$(group_of "$name")"; then
            echo "error: don't know the SuiteSparse group for '$name' -- pass Group/name" >&2
            FAILED+=("$name")
            return 1
        fi
    fi
    fetch "$group" "$name"
}

if [ "$#" -eq 0 ]; then
    # Legacy behaviour: the small/medium runnable demo.
    fetch Rajat rajat30
    echo
    echo "Run the demo, e.g.:"
    echo "  ./build/applications/klu_mixed_precision/klu_mixed_precision $DATA/rajat30/rajat30.mtx"
else
    for arg in "$@"; do
        if [ "$arg" = "all" ]; then
            fetch Rajat rajat30
            fetch Freescale circuit5M   # very large stress/scaling target (opt-in)
        else
            fetch_named "$arg"
        fi
    done
fi

if [ "${#FAILED[@]}" -gt 0 ]; then
    echo >&2
    echo "FAILED: ${FAILED[*]}" >&2
    exit 1
fi
