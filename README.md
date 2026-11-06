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
| rajat17 | 94294 | 3.433e+00 | 5 | 1.940e+01 | 8 | **0.18x — quire is worse** |
| rajat15 | 37261 | 1.603e+00 | 10 | 1.227e-02 | 6 | **130.6x** |
| rajat19 | 1157 | 1.047e-01 | 8 | 5.896e-02 | 5 | 1.8x |
| rajat13 | 7598 | 7.833e-02 | 7 | 4.233e-03 | 6 | 18.5x |
| rajat16 | 94294 | 7.319e-02 | 7 | 5.176e-02 | 14 | 1.4x |
| rajat04 | 1041 | 4.387e-03 | 8 | 1.008e-04 | 5 | 43.5x |
| rajat14 | 180  | 9.270e-04 | 7 | 5.272e-05 | 5 | 17.6x |
| rajat12 | 1879 | 3.833e-04 | 7 | 5.446e-06 | 5 | 70.4x |
| rajat05 | 301  | 2.575e-05 | 6 | 1.087e-05 | 5 | 2.4x |
| add20   | 2395 | 1.452e-05 | 5 | 6.363e-06 | 5 | 2.3x |
| rajat11 | 135  | 4.098e-06 | 6 | 2.123e-06 | 6 | 1.9x |
| add32   | 4960 | 5.439e-07 | 9 | 2.794e-07 | 5 | 1.9x |
| rajat03 | 7602 | pending (local re-run) | — | pending | — | — |

**rajat17 breaks the pattern.** Every matrix up to this point showed quire
helping, even if only marginally (1.8x at worst). rajat17 is the first
genuine counterexample: quire's forward error is *5.7x worse* than plain's,
and it needs more iterations to get there (8 vs. 5). rajat16 — same
dimensions as rajat17 (94,294 rows), from the same follow-up sweep — shows
only a marginal 1.4x gain while taking twice the iterations (14 vs. 7).
rajat15, by contrast, shows the single largest quire improvement in the
whole dataset (130.6x). All three are discussed against their dynamic-range
data in Section 4, since the obvious reading (worse dynamic range → bigger
counterexample) does not hold cleanly here.

`rajat07`, `rajat08`, `rajat09`, and `rajat10` were attempted in the same
sweep and all crashed identically to `rajat01` —
`unsupported Matrix Market field 'pattern'`, the same reader limitation, not
a result. Excluded from every table in this document.

**rajat30 (large-scale, 643,994 rows / 6.2M nonzeros)** — reported separately
since it's two orders of magnitude larger than the rest of the Tier 1 set:

| Matrix | n | Forward Error (plain) | Iterations | Forward Error (quire) | Iterations | Quire improvement |
|--------|---|------------------------|------------|-------------------------|------------|--------------------|
| rajat30 | 643994 | 8.705e-02 | 7 | 1.280e-03 | 6 | 68x |

Confirms the effect holds at scale, not just on the small Tier 1 matrices —
though the gain here (68x) is somewhat below rajat30's original, pre-fix
measurement (~79x), since that earlier number was itself an artifact of the
double-casting bug. One caveat specific to this run: quire's *residual*
actually got worse post-fix (4.9e-03 → 1.6e-02) even though forward error
barely moved, so residual and forward error decoupled slightly once the
computation was made genuinely precision-honest — residual alone isn't a
fully reliable accuracy proxy here.

**Quire shows a real forward-error improvement on most matrices re-run** —
1.8x to 130.6x across Tier 1, and 68x confirmed at rajat30's scale — but this
is no longer a universal claim: rajat17 is a confirmed counterexample where
quire makes forward error worse (0.18x). "Quire consistently helps residual
accumulation" should be read as "usually, substantially" rather than
"always" going forward.

Accumulation length (median number of terms per residual dot product) — from
an earlier build; `klu_quire_IR_study` no longer prints this directly
, so these are not re-derivable from thecurrent sweep without re-adding that instrumentation. 
Left here as-is, historical:

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

First real pass/fail dataset: SuiteSparse circuit matrices, run through both
algorithms via `scripts/run_matrix_sweep.sh`, normalized with
`csv/normalize_dynamic_range.py`. `rajat01`, `rajat07`, `rajat08`, `rajat09`,
and `rajat10` are all excluded — every one of them crashes on load
(`unsupported Matrix Market field 'pattern'`, a reader limitation, not a
result).

Sorted worst-to-best by Table 2 (`posit<32,2>`, plain) forward error, against
that same run's dynamic-range mean/max (`posit<32,2>`, plain variant).
`rajat15`, `rajat16`, and `rajat17` are new additions from a follow-up sweep;
`rajat03`'s dynamic range is unaffected by the Table 2 fix (Table 1-derived)
but its Table 2 forward error is still pending a local re-run:

| Matrix | n | Table 2 fwd err (plain) | Table 2 fwd err (quire) | Dyn. range Mean / Max |
|--------|---|---------------------------|----------------------------|------------------------|
| rajat17 | 94294 | 3.433e+00 | 1.940e+01 | 7.08 / 38 |
| rajat15 | 37261 | 1.603e+00 | 1.227e-02 | 3.20 / 20 |
| rajat19 | 1157 | 1.047e-01 | 5.896e-02 | 1.78 / 21 |
| rajat13 | 7598 | 7.833e-02 | 4.233e-03 | 1.32 / 7 |
| rajat16 | 94294 | 7.319e-02 | 5.176e-02 | 20.64 / 38 |
| rajat04 | 1041 | 4.387e-03 | 1.008e-04 | 2.22 / 8 |
| rajat14 | 180  | 9.270e-04 | 5.272e-05 | 1.81 / 8 |
| rajat12 | 1879 | 3.833e-04 | 5.446e-06 | 2.13 / 7 |
| rajat05 | 301  | 2.575e-05 | 1.087e-05 | 2.83 / 6 |
| add20   | 2395 | 1.452e-05 | 6.363e-06 | 4.98 / 10 |
| rajat11 | 135  | 4.098e-06 | 2.123e-06 | 3.17 / 6 |
| add32   | 4960 | 5.439e-07 | 2.794e-07 | 3.12 / 34 |
| rajat03 | 7602 | pending | pending | 8.93 / 12 |

**rajat30** (643,994 rows — reported separately, ~7x larger than rajat16/17
and two orders of magnitude larger than the rest of this table):

| Matrix | n | Table 2 fwd err (plain) | Table 2 fwd err (quire) | Dyn. range Mean / Max |
|--------|---|---------------------------|----------------------------|------------------------|
| rajat30 | 643994 | 8.705e-02 | 1.280e-03 | 6.41 / 39 |

rajat30 sits close to rajat16/rajat19 on both forward error and dynamic
range, but behaves like the "quire clearly helps" majority (68x) rather than
like rajat16/rajat17 despite comparable or wider dynamic range — one more
data point against dynamic range alone being the deciding factor, and a
reason the per-row diagnostic (Section 5) should probably include rajat30
alongside rajat16/rajat17 once it exists.

**The dynamic-range-predicts-accuracy question is now actively contradicted,
not just unclean.** `rajat16` has by far the widest dynamic range in the
entire dataset (mean 20.64, more than 6x the next-highest mean) and its
`ProductMagnitudeStats` show genuine bimodality — a dominant cluster at 1e-3
and a second real cluster at 1e-32/1e-33 — the exact catastrophic-cancellation
signature the hypothesis predicts and that the earlier pooled rajat11/rajat14
histograms failed to find. Yet quire barely helps it (1.4x, using twice the
iterations). `rajat17` has a similarly wide tail (max 38) and quire actively
*hurts* it. Meanwhile `rajat15`, with a comparatively unremarkable dynamic
range (mean 3.20), gets the single best quire improvement in the whole study
(130.6x). Dynamic range does not cleanly predict whether quire helps a given
matrix — if it predicts anything, it may be more about how hard the
underlying problem is (`rajat16`/`rajat17` also have the worst *plain*
forward errors in the set) than about whether an exact accumulator fixes it.
`add32` remains a standing counterexample too — widest range among the
original Tier 1 set (34) with the best accuracy of that set.

**The plain-vs-quire question is no longer simple either.** With the Table 2
double-cast bug fixed, quire improves forward error on most matrices re-run —
1.8x to 130.6x — but `rajat17` is a confirmed counterexample where quire
makes forward error worse (0.18x, i.e. 5.7x worse). The original "no quire
gain" conclusion for this experiment was still an artifact of the bug on the
original Tier 1 set, but "quire always helps" is not the correct replacement
claim now that a wider matrix sample has been run.

Also worth noting: `add20` fails badly (fwd err ≈ 4.0) but only at
`posit<16,2>` in Table 1 — a different, precision-width-specific failure
mode, not obviously the same mechanism as `rajat13`/`rajat19`/`rajat16`/`rajat17`.

### 5. Future direction

The dynamic-range-vs-accuracy question argues for a **per-row
diagnostic**, not more matrices at the same aggregate granularity: pair each
row's *own* dynamic range against that row's *own* residual contribution,
rather than summarizing a whole matrix into one mean/max. `rajat13` may have
a small number of genuinely bad rows that a matrix-wide summary washes out
entirely.

`rajat30` (644K rows) has now been re-run with the fixed code
(`logs/rajat30_fixed.log`), confirming the quire gain is real and not an
artifact of the double-cast bug: `posit<32,2>` forward error goes from
8.705e-02 (plain) to 1.280e-03 (quire) — **~68x**, versus the pre-fix run's
~79x (9.920e-02 → 1.253e-03). Plain barely moved; quire's forward error
barely moved either. The one number that did shift is quire's *residual*
(4.869e-03 pre-fix → 1.559e-02 post-fix, ~3x worse) despite forward error
staying flat — residual and forward error decoupled slightly once the
computation is genuinely precision-honest, worth keeping in mind as a
caveat rather than a red flag. The dynamic-range distribution is unaffected
by this fix (it's built from true double-precision term magnitudes,
independent of accumulator precision) and reproduces the same bimodal shape
as the original run.

`rajat03` still needs the same re-run treatment (too slow to finish in the
sandbox this fix was originally verified in) — that's the remaining
immediate next step, not a new diagnostic.

A follow-up sweep added `rajat15`, `rajat16`, and `rajat17` (94K–37K rows,
between rajat30 and the rest of Tier 1 in scale). `rajat07`–`rajat10` were
also attempted and all failed to load, the same `pattern`-field limitation as
`rajat01`. The three that did run sharpen, rather than resolve, the
per-row-diagnostic argument above: `rajat16` has the widest dynamic range and
a genuinely bimodal product-magnitude distribution in the whole dataset, yet
only a marginal quire gain; `rajat17`, similar in scale and range, is the
first matrix where quire measurably hurts forward error. A per-row diagnostic
on these two specifically — rather than more matrices at the current
aggregate granularity — is now the most likely way to explain why wide
dynamic range predicted neither's outcome correctly.

### 6. Logs and raw data

Everything the tables above are built from is checked into `logs/` and
`csv/`, for anyone who wants to go past the summarized numbers.

**`logs/*.log`** — one full stdout transcript per matrix from
`klu_quire_IR_study` (via `scripts/run_matrix_sweep.sh` or a direct
invocation), e.g. `logs/rajat13_sweep.log`, `logs/rajat30_run.log` (pre-fix),
`logs/rajat30_fixed.log` (post-fix, current numbers). Each contains both
experiments' accuracy tables in full, plus the ASCII product-magnitude and
dynamic-range histograms for every type/variant combination — the raw
evidence behind every summarized number in this doc.

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

**Data-hygiene status (resolved for the current sweep):** the pre-fix rows
that used to sit alongside post-fix rows in `csv/product_magnitude_histogram.csv`
and `csv/dynamic_range_histogram.csv` under the same grouping keys have been
moved to `csv/archive_prefix/` and are no longer live. The current
`csv/*.csv` files contain only fixed-code data — rajat30 (from
`rajat30_fixed.log`) plus the full follow-up sweep (rajat04/05/11/12/13/14/19,
add20, add32, rajat15/16/17). `csv/archive_prefix/*.csv` is kept for
reference only and predates the double-cast fix — do not use it for current
numbers.

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
