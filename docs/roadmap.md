# mp-spice roadmap

mp-spice is the integration layer that brings together MTL5 (linear algebra)
and Universal (number systems) for SPICE modernization. MTL5 itself must remain
free of any Universal dependency.

## Milestone 1 — Native KLU mixed-precision comparison

Drive MTL5's native KLU (`sparse::factorization::native_klu`) on
SuiteSparse circuit matrices and compare accuracy across `double`, `float`,
`cfloat<16,5>`, and `posit<16,2>`.

- [x] Repo scaffold: CMake composition of MTL5 + Universal via FetchContent,
      CI, smoke test, demo application skeleton.
- [x] `double` / `float` KLU solve on a synthetic block-triangular matrix and on
      loaded Matrix Market files.
- [x] **Unblock low-precision KLU.** Fixed upstream: MTL5's sparse factorizations
      now use ADL-friendly `abs`/`sqrt` (stillwater-sc/mtl5#121), so `native_klu`
      compiles for `cfloat`/`posit`. (`sparse_lu`, `sparse_cholesky`, `sparse_qr`
      all addressed; `sparse_ldlt`/`triangular_solve` were already clean.)
- [x] Enable `cfloat<16,5>` / `posit<16,2>` paths (now ON by default) and produce
      the four-precision accuracy table (residual + forward error). Per-type
      failures (e.g. a block going singular in half precision) are reported, not
      fatal. Validated on `Rajat/rajat11`: posit<16,2> completes where
      cfloat<16,5> fails.
- [x] `mm_read` loads real SuiteSparse `.mtx` (verified on rajat11, `coordinate
      real general`).
- [x] **Mixed-precision study** (`applications/klu_mixed_precision_study`, study
      logic in `include/sw/mp_spice/klu_study.hpp`): per type, compares a direct
      solve against **mixed-precision iterative refinement** — factor once in the
      low precision (native KLU, MTL5 v5.5.0), then refine with a
      **double-precision residual**, reusing the low-precision factorization.
      Reports residual + forward error for both, with `--csv`.

      **Headline finding (add32, 4960×4960):** a direct posit<16,2> solve has
      ~1e-2 forward error, but mixed IR recovers it to **~5e-12 in 30 steps** —
      while cfloat<16,5> (same 16 bits) **does not converge** (correction rounds
      to ~0 when the double residual is cast back to half). float recovers to
      ~5e-15 in 2 steps. Posit's tapered precision near 1.0 makes the low-
      precision correction usable where IEEE half does not — the core
      mixed-precision-for-circuits result, and motivation for keeping the
      residual/correction in higher precision (quire, #3) — see Milestone 2.
- [ ] Matrix loading at scale: rajat30 / circuit5M (gzip / large-file handling).
- [ ] Cross-check against the external SuiteSparse KLU binding where available.
- [ ] Native-KLU robustness on stiff circuit matrices is largely addressed by the
      MTL5 KLU performance epic (scaling + AMD ordering, v5.5.0).

## Milestone 2 — Mixed-precision solver strategies

- [x] Low-precision factorization + `double` iterative refinement (see the
      mixed-precision study under Milestone 1 — the compelling result). The IR
      loop now delegates to MTL5's generic `mtl::sparse::iterative_refine` core
      (stillwater-sc/mtl5#119); `mixed_refine`/`mixed_refine_scaled` are thin
      wrappers that factor in the low precision and call it.
- [x] Row/column scaling (equilibration): native KLU row-equilibrates by default
      (MTL5 v5.5.0).
- [x] **Quire super-accumulator** adapter (`include/sw/mp_spice/quire_accumulator.hpp`)
      fills MTL5's sparse_lu accumulator seam (stillwater-sc/mtl5#122) with a
      Universal posit quire (exact fused dot product); MTL5
      `native_klu_factor` forwards an `Accumulator` (stillwater-sc/mtl5#156) so
      the full BTF-KLU factorization can use it.
- [x] **Comprehensive precision study** (`applications/klu_precision_study`) and
      quire deep-dive (`applications/klu_quire_study`). Full write-up with tables:
      [docs/mixed-precision-klu-study.md](mixed-precision-klu-study.md).

      Headline findings (add32, 4960×4960):
      - **For iterative refinement, dynamic range beats mantissa width.** `float`,
        `bfloat16`, and both posits refine to ~1e-13 or better; IEEE `half`
        **stalls** (~5.7e-5) despite a better *direct* solve and 3× the mantissa —
        its narrow 5-bit exponent underflows the small IR corrections. Choose the
        IR carrier by exponent range, not precision.
      - **Posits are the best 16/32-bit carriers** (best direct accuracy at each
        width).
      - **The quire helps a direct solve (~1.5×) but is washed out by IR** — IR
        already absorbs the factorization's accumulation error. Prefer cheap IR
        over the expensive quire when you can iterate.
- [x] **Scaled iterative refinement** (extended-precision residual magnitude):
      `mixed_refine_scaled` normalizes each residual to O(1) before casting to the
      low-precision type and restores the correction magnitude in double. On
      add32 this **rescues IEEE `half`** (unscaled IR stalls at 5.7e-5 → scaled
      reaches **2.8e-14 in 8 iterations**) and accelerates `posit<16,2>`
      (2.7e-12 → **1.2e-14**); wide-range types are unchanged. Confirms the
      `half` stall was a residual *representation* problem, fixed by a single
      double scale factor per step — making every 16-bit type studied a viable
      carrier. Table 3 in [the study](mixed-precision-klu-study.md).
- [ ] Per-precision conditioning / accuracy study across a matrix suite.

## Milestone 3 — SPICE front-end

- [ ] Netlist parser → Modified Nodal Analysis assembly into MTL5 sparse
      matrices.
- [ ] DC operating point and transient analysis driving native KLU.

## Related

- MTL5 native KLU: stillwater-sc/mtl5 epic #114 (and follow-ups #117 ordering,
  #118 scaling, #119 iterative refinement).
- MTL5 double-only SuiteSparse example: stillwater-sc/mtl5 #120.
