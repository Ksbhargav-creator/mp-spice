# Applications (demonstration programs)
# Each subdirectory contains its own CMakeLists.txt.

# Native KLU mixed-precision comparison on SuiteSparse circuit matrices.
add_subdirectory(klu_mixed_precision)

# Mixed-precision study: direct vs mixed iterative-refinement accuracy per type.
add_subdirectory(klu_mixed_precision_study)

# Quire super-accumulator study: plain vs exact-accumulation posit sparse LU.
add_subdirectory(klu_quire_study)
