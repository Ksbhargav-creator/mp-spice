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

## Research methodology

### 1. Experiment results

**Quire in the KLU factorization (LU updates + triangular solve)**

Accuracy, direct solve, plain vs. quire accumulator:

| Matrix | n | Type | Forward Error (plain) | Forward Error (quire) |
|--------|---|------|-----------------------|------------------------|
| add32  | 4960 | posit<16,2> | 1.07e-2 | 0.98e-2 |
| add32  | 4960 | posit<32,2> | 2.61e-7 | 2.38e-7 |
|        |   |      |                        |                        |
|        |   |      |                        |                        |

Accumulation length (median number of terms per dot product):

| Matrix | n | Type | Kernel | Median | Mean |
|--------|---|------|--------|--------|------|
| rajat11 | 135 | posit<16,2> | Forward | 1 | 2.652 |
| rajat11 | 135 | posit<16,2> | Backward | 2 | 2.756 |
| rajat11 | 135 | posit<16,2> | Schur | 0 | 0.731 |
| rajat11 | 135 | posit<32,2> | Forward | 1 | 2.652 |
| rajat11 | 135 | posit<32,2> | Backward | 2 | 2.756 |
| rajat11 | 135 | posit<32,2> | Schur | 0 | 0.995 |
| rajat14 | 180 | posit<16,2> | Forward | 1.5 | 3.206 |
| rajat14 | 180 | posit<16,2> | Backward | 3 | 3.15 |
| rajat14 | 180 | posit<16,2> | Schur | 2 | 3.114 |
| rajat14 | 180 | posit<32,2> | Forward | 1.5 | 3.206 |
| rajat14 | 180 | posit<32,2> | Backward | 3 | 3.15 |
| rajat14 | 180 | posit<32,2> | Schur | 2 | 2.738 |
| rajat05 | 301 | posit<16,2> | Forward | 1 | 4.336 |
| rajat05 | 301 | posit<16,2> | Backward | 3 | 3.628 |
| rajat05 | 301 | posit<16,2> | Schur | 0 | 0.728 |
| rajat05 | 301 | posit<32,2> | Forward | 1 | 4.336 |
| rajat05 | 301 | posit<32,2> | Backward | 3 | 3.628 |
| rajat05 | 301 | posit<32,2> | Schur | 1 | 1.609 |
| rajat04 | 1041 | posit<16,2> | Forward | 1 | 5.481 |
| rajat04 | 1041 | posit<16,2> | Backward | 1 | 5.487 |
| rajat04 | 1041 | posit<16,2> | Schur | 1 | 2.944 |
| rajat04 | 1041 | posit<32,2> | Forward | 1 | 5.481 |
| rajat04 | 1041 | posit<32,2> | Backward | 4 | 5.487 |
| rajat04 | 1041 | posit<32,2> | Schur | 1 | 3.283 |
| rajat12 | 1879 | posit<16,2> | Forward | 0 | 1.999 |
| rajat12 | 1879 | posit<16,2> | Backward | 2 | 1.999 |
| rajat12 | 1879 | posit<16,2> | Schur | 1 | 2.003 |
| rajat12 | 1879 | posit<32,2> | Forward | 0 | 1.999 |
| rajat12 | 1879 | posit<32,2> | Backward | 2 | 1.999 |
| rajat12 | 1879 | posit<32,2> | Schur | 1 | 1.848 |
|  |  |  |  |  |  |

**Quire in the residual accumulation (cast to double before accumulating — not genuinely mixed precision)**

Accuracy, "Standard IR" (quire in factorization) vs. "Residual IR" (quire on the double-cast residual):

| Matrix | n | Type | Method | Forward Error | Iterations |
|--------|---|------|--------|----------------|------------|
| rajat11 | 135 | posit16 | Standard IR | 1.528e-10 | 25 |
| rajat11 | 135 | posit16 | Residual IR | 1.528e-10 | 25 |
| rajat11 | 135 | posit32 | Standard IR | 2.050e-13 | 3 |
| rajat11 | 135 | posit32 | Residual IR | 2.180e-13 | 3 |
| rajat14 | 180 | posit16 | Standard IR | 1.819e-11 | 16 |
| rajat14 | 180 | posit16 | Residual IR | 4.366e-11 | 15 |
| rajat14 | 180 | posit32 | Standard IR | 4.580e-12 | 7 |
| rajat14 | 180 | posit32 | Residual IR | 3.352e-12 | 6 |
| rajat05 | 301 | posit16 | Standard IR | 3.94e-09 | 25 |
| rajat05 | 301 | posit16 | Residual IR | 3.94e-09 | 25 |
| rajat05 | 301 | posit32 | Standard IR | 4.27e-13 | 3 |
| rajat05 | 301 | posit32 | Residual IR | 3.84e-13 | 3 |
| rajat04 | 1041 | posit16 | Standard IR | 7.28e-12 | 17 |
| rajat04 | 1041 | posit16 | Residual IR | 7.28e-12 | 17 |
| rajat04 | 1041 | posit32 | Standard IR | 1.29e-13 | 5 |
| rajat04 | 1041 | posit32 | Residual IR | 9.16e-12 | 3 |
| rajat12 | 1879 | posit16 | Standard IR | 4.82e-11 | 16 |
| rajat12 | 1879 | posit16 | Residual IR | 4.82e-11 | 16 |
| rajat12 | 1879 | posit32 | Standard IR | 3.54e-12 | 9 |
| rajat12 | 1879 | posit32 | Residual IR | 3.62e-12 | 7 |
|  |  |  |  |  |  |

Accumulation length (median number of terms per residual dot product — same for
both Standard and Residual IR, since term count doesn't depend on the accumulator):

| Matrix | n | Type | Mean | Median |
|--------|---|------|------|--------|
| rajat11 | 135 | posit<16,2> / posit<32,2> | 6.015 | 6 |
| rajat14 | 180 | posit<16,2> / posit<32,2> | 8.35 | 7 |
| rajat05 | 301 | posit<16,2> / posit<32,2> | 4.598 | 5 |
| rajat04 | 1041 | posit<16,2> / posit<32,2> | 9.262 | 6 |
| rajat12 | 1879 | posit<16,2> / posit<32,2> | 6.879 | 4 |
|  |  |  |  |  |

### 2. Hypothesis

Posits (and other tapered-precision number systems, e.g. LNS) allocate their
bit budget non-uniformly: near magnitude 1.0 the regime field is short and
nearly all remaining bits go to the fraction field (maximum precision); at
the extremes of the representable range the regime field grows and the
fraction field shrinks toward zero. If a dot product's terms swing through
those extreme, fraction-starved regimes, the information that would have
lived in the missing fraction bits is lost — permanently, regardless of how
well-represented later terms near 1.0 are.

The illustrative case: `maxpos × minpos = 1.0`. Both operands carry zero
precision bits and only has regime bits, but the product is filled with
precision bits.

To test this without needing one histogram per dot product, the proposed
metric is **dynamic range**: for each dot product, `log10(max |term|) −
log10(min |term|)` across its nonzero terms. 
### 3. Diagnostic attempts

**Pooled product-magnitude histogram.** Every individual term `a_ik · x_k`
formed while assembling a residual was bucketed by decade of magnitude
(`ProductMagnitudeStats` in `include/sw/mp_spice/klu_study.hpp`), pooled
across all rows and IR iterations, looking for the two-cluster ("bimodal")
signature of catastrophic cancellation. On `rajat11`/`rajat14` the result was
a broad, roughly unimodal distribution — no bimodality. Pooling across every
row smears out whatever structure exists *within* a single dot product,
which is the thing that actually matters for Theo's hypothesis.

**Per-dot-product dynamic-range histogram.** Reduces each dot product (each
row, each IR iteration) to one scalar — its dynamic range in decades — and
histograms those scalars instead of raw magnitudes. Implemented as
`DynamicRangeStats` in `include/sw/mp_spice/klu_study.hpp`, wired into the
same residual loop as the product-magnitude histogram, exported to
`csv/dynamic_range_histogram.csv`, and printed as an ASCII histogram in
`klu_quire_IR_study`. Measured from the true double-precision magnitudes of
each term (not the low-precision-cast values), so it's a property of the
matrix and the `x` trajectory, independent of which number system is being
tested. First smoke test on `rajat11` validated the mechanism but was
inconclusive on the hypothesis itself — the matrix is too small and
well-conditioned to produce a real pass/fail contrast.

### 4. Future direction

Test on bigger, stiffer matrices. `rajat11`/`rajat14` are too small and
well-conditioned to produce dot products with meaningful dynamic range or to
show a real pass/fail contrast for Theo's hypothesis — a genuinely stiff
SPICE Jacobian at scale is needed next.

---

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
