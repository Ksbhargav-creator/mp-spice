# Applications (demonstration programs)
# Each subdirectory contains its own CMakeLists.txt.

# Native KLU mixed-precision comparison on SuiteSparse circuit matrices.
add_subdirectory(klu_mixed_precision)

# Mixed-precision study: direct vs mixed iterative-refinement accuracy per type.
add_subdirectory(klu_mixed_precision_study)

# Quire super-accumulator study: plain vs exact-accumulation posit sparse LU
# (one-shot direct solve, no iterative refinement).
add_subdirectory(klu_quire_study)

# Companion to klu_quire_study: native KLU (BTF) + iterative refinement, both
# the factor-accumulator and residual-accumulator quire experiments.
add_subdirectory(klu_quire_IR_study)

# Comprehensive precision study: double/float/bfloat16/half/posit x {direct,IR,quire}.
add_subdirectory(klu_precision_study)
