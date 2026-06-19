#!/usr/bin/env bash
# Fetch circuit-simulation matrices from the SuiteSparse Matrix Collection
# (https://sparse.tamu.edu/) into ./data. Matrices are NOT committed to the repo.
#
# Usage:
#   scripts/fetch_matrices.sh            # fetch the small demo matrix (rajat30)
#   scripts/fetch_matrices.sh all        # also fetch the large stress matrix (circuit5M, ~hundreds of MB)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="$ROOT/data"
mkdir -p "$DATA"

BASE="https://suitesparse-collection-website.herokuapp.com/MM"

fetch() {
    local group="$1" name="$2"
    local url="$BASE/$group/$name.tar.gz"
    local tarball="$DATA/$name.tar.gz"
    if [ -f "$DATA/$name/$name.mtx" ]; then
        echo "already have $name"
        return
    fi
    echo "downloading $group/$name ..."
    curl -L --fail -o "$tarball" "$url"
    tar -xzf "$tarball" -C "$DATA"
    rm -f "$tarball"
    echo "extracted to $DATA/$name/$name.mtx"
}

# Small/medium runnable demo.
fetch Rajat rajat30

# Very large stress/scaling target (opt-in).
if [ "${1:-}" = "all" ]; then
    fetch Freescale circuit5M
fi

echo
echo "Run the demo, e.g.:"
echo "  ./build/applications/klu_mixed_precision/klu_mixed_precision $DATA/rajat30/rajat30.mtx"
