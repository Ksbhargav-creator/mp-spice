# The quire study requires Universal (quire) + the MTL5 accumulator seam.
if(MPSPICE_MIXED_PRECISION_KLU)
	add_executable(klu_quire_study klu_quire_study.cpp)
	target_link_libraries(klu_quire_study PRIVATE sw::mp_spice)
	target_compile_definitions(klu_quire_study PRIVATE MPSPICE_MIXED_PRECISION_KLU)
endif()
