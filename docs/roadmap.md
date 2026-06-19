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
- [ ] **Unblock low-precision KLU.** MTL5's `sparse_lu.hpp` calls unqualified
      `std::abs(x)`, which does not find Universal's `abs()` via ADL, so
      `native_klu` does not compile for `cfloat`/`posit`. Fix upstream in MTL5
      with an ADL-friendly form (`using std::abs; ... abs(x)`) — this is
      dependency-free and directly serves MTL5's stated mixed-precision mission.
      Audit `sparse_cholesky`, `sparse_ldlt`, `sparse_qr` for the same pattern.
- [ ] Enable `cfloat<16,5>` / `posit<16,2>` paths in the demo
      (`-DMPSPICE_MIXED_PRECISION_KLU=ON`) and produce the four-precision
      accuracy table.
- [ ] Matrix loading at scale: confirm `mtl::io::mm_read` handles the rajat30
      and circuit5M headers; add gzip / large-file handling as needed.
- [ ] Cross-check against the external SuiteSparse KLU binding where available.

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
