# mp-spice

[![CMake](https://github.com/stillwater-sc/mp-spice/actions/workflows/cmake.yml/badge.svg)](https://github.com/stillwater-sc/mp-spice/actions/workflows/cmake.yml)

**Mixed-precision SPICE modernization.** mp-spice composes two header-only
libraries — [MTL5](https://github.com/stillwater-sc/mtl5) for linear algebra and
[Universal](https://github.com/stillwater-sc/universal) for parameterized number
systems — to explore SPICE circuit-simulation solvers under custom arithmetic
(half precision, posits, ...).

MTL5 deliberately has **no dependency on Universal**: it is the general
linear-algebra layer. mp-spice is the integration layer where MTL5's algorithms
meet Universal's number types.

## First milestone: native KLU across precisions

[KLU](https://dl.acm.org/doi/10.1145/1824801.1824814) is the sparse direct
solver of choice for circuit-simulation (Modified Nodal Analysis) matrices.
MTL5 provides a native, value-type-generic KLU
(`sparse::factorization::native_klu`). mp-spice drives it on real circuit
matrices from the [SuiteSparse Matrix Collection](https://sparse.tamu.edu/) and
compares accuracy across:

| Type | Notes |
|------|-------|
| `double` | reference |
| `float` | single precision |
| `cfloat<16,5>` | IEEE-like half (Universal) |
| `posit<16,2>` | 16-bit posit (Universal) |

Target matrices:

- [`Rajat/rajat30`](https://sparse.tamu.edu/Rajat/rajat30) — small/medium demo.
- [`Freescale/circuit5M`](https://sparse.tamu.edu/Freescale/circuit5M) — very
  large stress/scaling target.

> **Status:** all four precisions work (enabled by default;
> `-DMPSPICE_MIXED_PRECISION_KLU=OFF` to build `double`/`float` only). Example
> output on `Rajat/rajat11` (135×135 circuit matrix, exact solution all-ones):
>
> ```
>   type              ||Ax-b||inf     ||x-1||inf
>   ------------------------------------------
>   double              3.553e-15      2.803e-13
>   float               9.505e-07      7.176e-05
>   cfloat<16,5>     solve failed: zero pivot (block singular in half precision)
>   posit<16,2>         3.229e-04      1.819e-01
> ```
>
> Note how `posit<16,2>` completes the solve where `cfloat<16,5>` fails — a
> concrete mixed-precision result. Robustness on stiff circuit matrices is
> expected to improve with scaling and iterative refinement (see
> [docs/roadmap.md](docs/roadmap.md), Milestone 2).

## Build

```bash
# Dependencies (MTL5 + Universal) are pulled automatically via FetchContent.
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run the smoke test
ctest --test-dir build --output-on-failure

# Run the demo on the built-in synthetic circuit matrix
./build/applications/klu_mixed_precision/klu_mixed_precision

# ...or on a real SuiteSparse matrix
scripts/fetch_matrices.sh                  # downloads rajat30 into ./data
./build/applications/klu_mixed_precision/klu_mixed_precision data/rajat30/rajat30.mtx
```

Using local checkouts instead of fetching from GitHub:

```bash
cmake -B build \
  -DFETCHCONTENT_SOURCE_DIR_MTL5=../mtl5 \
  -DFETCHCONTENT_SOURCE_DIR_UNIVERSAL=../universal
```

## Layout

```
applications/klu_mixed_precision/   # the KLU mixed-precision demo
include/sw/mp_spice/                # shared composition-layer headers
scripts/fetch_matrices.sh           # SuiteSparse matrix downloader
tests/                              # smoke tests
docs/roadmap.md                     # milestones and known integration work
```

## License

MIT — see [LICENSE](LICENSE).
