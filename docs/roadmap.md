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

- [ ] Low-precision factorization + `double` iterative refinement (the
      compelling mixed-precision story for stiff circuit matrices).
- [ ] Row/column scaling (equilibration) before factorization.
- [ ] Per-precision conditioning / accuracy study across a matrix suite.

## Milestone 3 — SPICE front-end

- [ ] Netlist parser → Modified Nodal Analysis assembly into MTL5 sparse
      matrices.
- [ ] DC operating point and transient analysis driving native KLU.

## Related

- MTL5 native KLU: stillwater-sc/mtl5 epic #114 (and follow-ups #117 ordering,
  #118 scaling, #119 iterative refinement).
- MTL5 double-only SuiteSparse example: stillwater-sc/mtl5 #120.
