# Comprehensive precision study: needs Universal + the MTL5 accumulator seam.
if(MPSPICE_MIXED_PRECISION_KLU)
	add_executable(klu_precision_study klu_precision_study.cpp)
	target_link_libraries(klu_precision_study PRIVATE sw::mp_spice)
	target_compile_definitions(klu_precision_study PRIVATE MPSPICE_MIXED_PRECISION_KLU)
endif()
