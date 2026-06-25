#pragma once

#include <string>
enum class SolveStatus {
    Success,
    ZeroPivot,
    Failure
};

struct BenchmarkResult {
    std::string matrix_name;
    std::string arithmetic_name;

    double factor_ms;
    double solve_ms;

    double residual_inf;
    double forward_error_inf;

    SolveStatus status;

    std::string error_message;
};