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
- [ ] Matrix loading at scale: rajat30 / circuit5M (gzip / large-file handling).
- [ ] Cross-check against the external SuiteSparse KLU binding where available.
- [ ] Investigate native-KLU robustness on stiff circuit matrices (zero pivots
      in low precision) — relates to Milestone 2 (scaling, iterative refinement)
      and MTL5 follow-ups #117/#118/#119.

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
