#!/usr/bin/env bash
# Native (no-container) build + run of the README reproduction on a cluster,
# e.g. a RISC-V machine. Run the BUILD/FETCH steps on a login node (needs internet);
# run the compute steps via your scheduler.
#
#   scripts/cluster_native_build.sh build      # clone sister repos, configure, compile
#   scripts/cluster_native_build.sh fetch      # download matrices (login node)
#   scripts/cluster_native_build.sh run  [reproduce_readme.sh args...]   # e.g. run --index N
#   scripts/cluster_native_build.sh merge
#
# Env overrides:
#   WORK=$HOME/mpspice-work   MTL5_REPO / MTL5_BRANCH   UNIVERSAL_REF (SHA!)   JOBS=2
#
# Notes:
#  * No -march=native; -ffp-contract=off so FMA contraction doesn't differ from the Mac run.
#  * Universal templates are RAM hungry: keep JOBS small (2 GB+/job).
#  * long double is soft binary128 on rv64 -- anything using it will be much slower.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"           # mp-spice checkout
WORK="${WORK:-$HOME/mpspice-work}"
MTL5_REPO="${MTL5_REPO:-https://github.com/Ksbhargav-creator/mtl5.git}"
MTL5_BRANCH="${MTL5_BRANCH:-quire-instrumentation}"
UNIVERSAL_REPO="${UNIVERSAL_REPO:-https://github.com/stillwater-sc/universal.git}"
UNIVERSAL_REF="${UNIVERSAL_REF:-main}"
JOBS="${JOBS:-2}"
export DATA_DIR="${DATA_DIR:-$WORK/data}"

cmd="${1:-}"; shift || true
case "$cmd" in
  build)
    mkdir -p "$WORK"; cd "$WORK"
    [ -d mtl5 ] || git clone --branch "$MTL5_BRANCH" "$MTL5_REPO" mtl5
    if [ ! -d universal ]; then
      git init -q universal && git -C universal remote add origin "$UNIVERSAL_REPO"
    fi
    git -C universal fetch -q --depth 1 origin "$UNIVERSAL_REF" && git -C universal checkout -q FETCH_HEAD
    [ -f mtl5/include/mtl/sparse/instrumentation/kernel_stats.hpp ] || {
      echo "ERROR: mtl5 branch '$MTL5_BRANCH' lacks kernel_stats.hpp -- push your local quire-instrumentation branch first." >&2; exit 1; }
    cmake -B "$WORK/build" -S "$HERE" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-ffp-contract=off" \
      -DMPSPICE_MIXED_PRECISION_KLU=ON -DMPSPICE_BUILD_TESTS=OFF \
      -DFETCHCONTENT_SOURCE_DIR_MTL5="$WORK/mtl5" \
      -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL="$WORK/universal" -Wno-dev
    cmake --build "$WORK/build" -j"$JOBS" --target klu_quire_IR_study klu_quire_study
    {
      echo "built:     $(date -u +%FT%TZ)  host: $(uname -srm)"
      echo "mp-spice:  $(git -C "$HERE" rev-parse HEAD)  dirty files: $(git -C "$HERE" status --porcelain -- include applications | wc -l)"
      echo "mtl5:      $(git -C mtl5 rev-parse HEAD) ($MTL5_BRANCH)"
      echo "universal: $(git -C universal rev-parse HEAD) (requested $UNIVERSAL_REF)"
      echo "compiler:  $(c++ --version | head -1)"
    } | tee "$WORK/PROVENANCE.txt"
    ;;
  fetch)
    mkdir -p "$DATA_DIR"
    "$HERE/scripts/fetch_matrices.sh" rajat04 rajat05 rajat11 rajat12 rajat13 rajat14 rajat15 \
      rajat16 rajat17 rajat19 add20 add32 rajat30 rajat03
    ;;
  run|merge|list|all)
    export MPSPICE_ROOT="$HERE"
    export BIN="$WORK/build/applications/klu_quire_IR_study/klu_quire_IR_study"
    export BIN_DIRECT="$WORK/build/applications/klu_quire_study/klu_quire_study"
    exec "$HERE/scripts/reproduce_readme.sh" "$cmd" --out "${OUT:-$WORK/results}" "$@"
    ;;
  *) echo "usage: $0 build|fetch|run|merge|list|all [args]" >&2; exit 2 ;;
esac
