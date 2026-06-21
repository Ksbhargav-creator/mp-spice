// mp-spice -- native KLU mixed-precision study (mp-spice #1 / #5)
//
// For a circuit-simulation matrix, compares, per arithmetic type:
//   * direct solve        -- factor and solve entirely in the type
//   * mixed refinement    -- factor in the type, refine with a double residual
// reporting residual ||Ax-b||inf and forward error ||x-1||inf for each.
//
// The headline result: a low-precision direct solve (cfloat<16,5>, posit<16,2>)
// loses accuracy, but the SAME low-precision factorization, reused inside
// iterative refinement with a double-precision residual, recovers accuracy
// toward double -- the core mixed-precision-for-circuits claim.
//
// Usage:
//   klu_mixed_precision_study [matrix.mtx] [--csv out.csv]
// No matrix -> a built-in synthetic block-triangular circuit matrix.
//
// double/float always built; cfloat<16,5>/posit<16,2> require
// -DMPSPICE_MIXED_PRECISION_KLU=ON (default ON; needs Universal).

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/io/matrix_market.hpp>
#include <sw/mp_spice/klu_study.hpp>

#ifdef MPSPICE_MIXED_PRECISION_KLU
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>
#endif

namespace {

using sw::mp_spice::DSparse;
using sw::mp_spice::solve_stats;

DSparse make_synthetic_circuit() {
    DSparse A(6, 6);
    mtl::mat::inserter<DSparse> ins(A);
    ins[0][0] << 3.0;  ins[0][1] << -1.0;
    ins[1][0] << -1.0; ins[1][1] << 3.0;  ins[1][2] << -1.0;
    ins[2][1] << -1.0; ins[2][2] << 3.0;
    ins[3][3] << 4.0;  ins[3][4] << -1.0;
    ins[4][3] << -1.0; ins[4][4] << 4.0;  ins[4][5] << -1.0;
    ins[5][4] << -1.0; ins[5][5] << 4.0;
    ins[0][5] << -2.0;
    return A;
}

struct Row {
    std::string type;
    solve_stats direct, refined;
};

template <typename T>
Row run(const std::string& type, const DSparse& A,
        const std::vector<double>& b, const std::vector<double>& ones) {
    return {type,
            sw::mp_spice::direct_solve<T>(A, b, ones),
            sw::mp_spice::mixed_refine<T>(A, b, ones)};
}

void print_cell(const solve_stats& s) {
    if (s.ok) std::printf(" %11.3e %11.3e", s.residual, s.fwd_error);
    else      std::printf(" %11s %11s", "FAIL", "-");
}

} // namespace

int main(int argc, char** argv) {
    std::string mtx, csv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csv = argv[++i];
        else mtx = a;
    }

    DSparse A;
    if (!mtx.empty()) { std::printf("Loading %s\n", mtx.c_str()); A = mtl::io::mm_read<double>(mtx); }
    else { std::printf("Synthetic block-triangular circuit matrix (pass a .mtx for a real one).\n");
           A = make_synthetic_circuit(); }
    std::printf("Matrix: %zu x %zu, nnz = %zu\n\n",
                (size_t)A.num_rows(), (size_t)A.num_cols(), (size_t)A.nnz());

    std::vector<double> ones(A.num_rows(), 1.0);
    auto b = sw::mp_spice::rhs_from_ones(A);

    std::vector<Row> rows;
    rows.push_back(run<double>("double", A, b, ones));
    rows.push_back(run<float>("float", A, b, ones));
#ifdef MPSPICE_MIXED_PRECISION_KLU
    rows.push_back(run<sw::universal::cfloat<16, 5>>("cfloat<16,5>", A, b, ones));
    rows.push_back(run<sw::universal::posit<16, 2>>("posit<16,2>", A, b, ones));
#endif

    // direct = factor+solve in the type; refined = type factor + double residual IR.
    std::printf("%-14s | %11s %11s | %11s %11s %5s\n",
                "type", "dir resid", "dir ferr", "ref resid", "ref ferr", "iters");
    std::printf("%s\n", std::string(72, '-').c_str());
    for (const auto& r : rows) {
        std::printf("%-14s |", r.type.c_str());
        print_cell(r.direct);
        std::printf(" |");
        print_cell(r.refined);
        if (r.refined.ok) std::printf(" %5d", r.refined.iters); else std::printf(" %5s", "-");
        std::printf("\n");
    }
#ifndef MPSPICE_MIXED_PRECISION_KLU
    std::printf("\n[cfloat<16,5>/posit<16,2> disabled: build -DMPSPICE_MIXED_PRECISION_KLU=ON]\n");
#endif

    if (!csv.empty()) {
        std::ofstream o(csv);
        o << "type,dir_status,dir_residual,dir_fwd_error,ref_status,ref_residual,ref_fwd_error,ref_iters\n";
        for (const auto& r : rows) {
            o << r.type << ',' << (r.direct.ok ? "ok" : "fail") << ',';
            if (r.direct.ok) o << r.direct.residual << ',' << r.direct.fwd_error; else o << ',';
            o << ',' << (r.refined.ok ? "ok" : "fail") << ',';
            if (r.refined.ok) o << r.refined.residual << ',' << r.refined.fwd_error << ',' << r.refined.iters;
            else o << ",,";
            o << '\n';
        }
        std::printf("\nCSV: %s\n", csv.c_str());
    }
    return 0;
}
