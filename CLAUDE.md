# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project Overview

mp-spice is the **integration layer** for mixed-precision SPICE modernization.
It composes two header-only sister libraries:

- [MTL5](https://github.com/stillwater-sc/mtl5) — C++20 linear algebra (sparse
  direct solvers, including native KLU).
- [Universal](https://github.com/stillwater-sc/universal) — parameterized number
  systems (`cfloat`, `posit`, ...).

**Architectural rule:** MTL5 is the general linear-algebra layer and MUST NOT
depend on Universal. All MTL5 + Universal coupling lives here in mp-spice.

## Build Commands

```bash
# Dependencies are pulled automatically via FetchContent.
cmake -B build -DCMAKE_BUILD_TYPE=Release -Wno-dev
cmake --build build -j
ctest --test-dir build --output-on-failure

# Use local sister checkouts instead of fetching from GitHub:
cmake -B build -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 \
               -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=../universal

# Enable the cfloat/posit KLU paths (see roadmap: needs the MTL5 ADL-abs fix):
cmake -B build -DMPSPICE_MIXED_PRECISION_KLU=ON
```

## Architecture

- Header-only composition under `include/sw/mp_spice/`. Namespace: `sw::mp_spice`.
- CMake: INTERFACE library `sw::mp_spice` linking MTL5 + Universal. Options:
  `MPSPICE_BUILD_APPLICATIONS`, `MPSPICE_BUILD_TESTS`, `MPSPICE_MIXED_PRECISION_KLU`.
- `applications/` — demonstration programs (each its own CMakeLists).
- `tests/` — lightweight self-checking executables (no external framework);
  register with `mpspice_add_test`.
- `scripts/fetch_matrices.sh` — downloads SuiteSparse matrices into `data/`
  (git-ignored; never commit matrices).
- `docs/roadmap.md` — milestones and known integration work.

## Conventions

- C++20, header-only. Match the sister repos (mtl5, mixed-precision-dsp) for
  style and CMake structure.
- Conventional Commits. Feature branches + PRs to `main`; CI must pass.
- Never commit downloaded matrices or build artifacts.
