# The IR quire study requires Universal (quire) + the MTL5 accumulator seam.
if(MPSPICE_MIXED_PRECISION_KLU)
	add_executable(klu_quire_IR_study klu_quire_IR_study.cpp)
	target_link_libraries(klu_quire_IR_study PRIVATE sw::mp_spice)
	target_compile_definitions(klu_quire_IR_study PRIVATE MPSPICE_MIXED_PRECISION_KLU)
endif()
