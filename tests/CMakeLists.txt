include(CTest)

# Helper: each test is a self-checking executable linked against sw::mp_spice.
function(mpspice_add_test test_name)
	add_executable(${test_name} ${test_name}.cpp)
	target_link_libraries(${test_name} PRIVATE sw::mp_spice)
	add_test(NAME ${test_name} COMMAND ${test_name})
endfunction()

# Smoke test: native KLU solves through the MTL5 + Universal composition.
mpspice_add_test(test_klu_smoke)
