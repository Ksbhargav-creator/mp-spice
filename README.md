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

Accumulation length (median number of terms per dot product) — from an
earlier build, unaffected by the Table 2 fix (this table is about the
factorization path, not the residual). Historical, kept as-is:

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

**Quire in the residual accumulation**.

`posit<32,2>`, plain vs. quire accumulator, sorted worst-to-best by plain
forward error:

| Matrix | n | Forward Error (plain) | Iterations | Forward Error (quire) | Iterations | Quire improvement |
|--------|---|------------------------|------------|-------------------------|------------|--------------------|
| rajat19 | 1157 | 1.047e-01 | 8 | 5.896e-02 | 5 | 1.8x |
| rajat13 | 7598 | 7.833e-02 | 7 | 4.233e-03 | 6 | 18.5x |
| rajat04 | 1041 | 4.387e-03 | 8 | 1.008e-04 | 5 | 43.5x |
| rajat14 | 180  | 9.270e-04 | 7 | 5.272e-05 | 5 | 17.6x |
| rajat12 | 1879 | 3.833e-04 | 7 | 5.446e-06 | 5 | 70.4x |
| rajat05 | 301  | 2.575e-05 | 6 | 1.087e-05 | 5 | 2.4x |
| add20   | 2395 | 1.452e-05 | 5 | 6.363e-06 | 5 | 2.3x |
| rajat11 | 135  | 4.098e-06 | 6 | 2.123e-06 | 6 | 1.9x |
| add32   | 4960 | 5.439e-07 | 9 | 2.794e-07 | 5 | 1.9x |
| rajat03 | 7602 | pending (local re-run) | — | pending | — | — |

**Quire shows a real, consistent forward-error improvement on every
matrix re-run** — 1.8x to 70x.

Accumulation length (median number of terms per residual dot product) — from
an earlier build; `klu_quire_IR_study` no longer prints this directly
(the LU factorization's own accumulation-length tracking was intentionally
stripped, see `docs/roadmap.md`), so these are not re-derivable from the
current sweep without re-adding that instrumentation. Left here as-is,
historical:

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

### 4. Dataset results (Tier 1 sweep)

First real pass/fail dataset: 10 SuiteSparse circuit matrices (135–7,602
rows), run through both algorithms via `scripts/run_matrix_sweep.sh`,
normalized with `csv/normalize_dynamic_range.py`. `rajat01` is excluded — it
crashed on load (`unsupported Matrix Market field 'pattern'`, a reader
limitation, not a result).

Sorted worst-to-best by Table 2 (`posit<32,2>`, plain) forward error, against
that same run's dynamic-range mean/max. Re-run with the Table 2 fix (see
above); `rajat03`'s dynamic range is unaffected by that fix (Table 1-derived)
but its Table 2 forward error is still pending a local re-run:

| Matrix | n | Table 2 fwd err (plain) | Table 2 fwd err (quire) | Dyn. range Mean / Max |
|--------|---|---------------------------|----------------------------|------------------------|
| rajat19 | 1157 | 1.047e-01 | 5.896e-02 | 1.78 / 21 |
| rajat13 | 7598 | 7.833e-02 | 4.233e-03 | 1.32 / 7 |
| rajat04 | 1041 | 4.387e-03 | 1.008e-04 | 2.22 / 8 |
| rajat14 | 180  | 9.270e-04 | 5.272e-05 | 1.81 / 8 |
| rajat12 | 1879 | 3.833e-04 | 5.446e-06 | 2.13 / 7 |
| rajat05 | 301  | 2.575e-05 | 1.087e-05 | 2.83 / 6 |
| add20   | 2395 | 1.452e-05 | 6.363e-06 | 4.98 / 10 |
| rajat11 | 135  | 4.098e-06 | 2.123e-06 | 3.17 / 6 |
| add32   | 4960 | 5.439e-07 | 2.794e-07 | 3.12 / 34 |
| rajat03 | 7602 | pending | pending | 8.93 / 12 |

**The dynamic-range-predicts-accuracy question is still not clean.**
`rajat19` is now unambiguously the worst matrix in the set and also has the
highest dynamic-range max (21) among everything but `add32` — consistent
with the hypothesis. But `rajat13` is nearly as bad with a max of only 7, and
`add32` still has the *widest* dynamic range of the whole set (34) with the
*best* accuracy. One matrix-wide summary number (mean or max) still doesn't
cleanly separate real failures from real passes.

**The plain-vs-quire question, however, is no longer ambiguous.** With the
Table 2 double-cast bug fixed, quire improves forward error on every single
matrix re-run — 1.8x to 70x, see the residual-accumulation table above. The
original "no quire gain" conclusion for this experiment was an artifact of
the bug, not a real finding.

Also worth noting: `add20` fails badly (fwd err ≈ 4.0) but only at
`posit<16,2>` in Table 1 — a different, precision-width-specific failure
mode, not obviously the same mechanism as `rajat13`/`rajat19`.

### 5. Future direction

The dynamic-range-vs-accuracy question argues for a **per-row
diagnostic**, not more matrices at the same aggregate granularity: pair each
row's *own* dynamic range against that row's *own* residual contribution,
rather than summarizing a whole matrix into one mean/max. `rajat13` may have
a small number of genuinely bad rows that a matrix-wide summary washes out
entirely.

`rajat30` (644K rows) already ran once and showed a large quire gain in
Table 2 (residual res 7.57e-01→4.87e-03, ferr 9.92e-02→1.25e-03) — but that
run used the pre-fix double-casting code, so it needs to be re-run with the
fixed code before that number can be trusted. `rajat03` needs the same
treatment (too slow to finish in the sandbox this fix was verified in).
Both are the immediate next step, not new diagnostics.

### 6. Logs and raw data

Everything the tables above are built from is checked into `logs/` and
`csv/`, for anyone who wants to go past the summarized numbers.

**`logs/*.log`** — one full stdout transcript per matrix from
`klu_quire_IR_study` (via `scripts/run_matrix_sweep.sh` or a direct
invocation), e.g. `logs/rajat13_sweep.log`, `logs/rajat30_run.log`. Each
contains both experiments' accuracy tables in full, plus the ASCII
product-magnitude and dynamic-range histograms for every type/variant
combination — the raw evidence behind every summarized number in this doc.

**`csv/product_magnitude_histogram.csv`, `csv/dynamic_range_histogram.csv`**
— the same data the app writes out per run (`Matrix,n,Experiment,Type,Variant,...`),
appended across every run against that matrix. Written in append mode, so
re-running a matrix multiple times duplicates rows — `csv/normalize_dynamic_range.py`
drops exact-duplicate rows on read, but see the caveat below.

**`csv/normalize_dynamic_range.py` → `dynamic_range_normalized.csv`,
`dynamic_range_summary.csv`** — normalizes the raw histogram into
percentages per group, and computes weighted Mean/Median/P90/Max per
`(Matrix, n, Experiment, Type, Variant)` group. `dynamic_range_summary.csv`
is what the Section 4 dataset table above is built from.

**`csv/plot_dynamic_range.py`** — renders the normalized data as grouped bar
charts, one PNG per `(Matrix, Experiment, Type)` (e.g.
`dynamic_range_rajat13_WorkingPrecisionResidualIR_posit32_2.png`).

**`csv/lu_accumulation.csv`, `csv/residual_accumulation.csv`,
`csv/quire_effectiveness.csv`** — earlier accumulation-length and
Standard-IR-vs-Residual-IR data, from before the app was pared down (Section
3's stripped instrumentation) and before the Table 2 double-casting fix.
Source data for the "historical" tables flagged in Section 1 — kept for
reference, not representative of the current (fixed) code.

**Known data-hygiene issue:** the Tier 1 re-run against the fixed code
(Section 1/4 above) appended fresh rows into the *same* `csv/` files
alongside the pre-fix rows, under the same grouping keys. The dedup step
only drops exact-duplicate rows, so it will **not** catch this — pre-fix and
post-fix runs produce genuinely different distributions under the same key.
`csv/dynamic_range_summary.csv` should not be trusted until `csv/`'s two raw
histogram files are cleared and the sweep is re-run fresh end to end.

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
