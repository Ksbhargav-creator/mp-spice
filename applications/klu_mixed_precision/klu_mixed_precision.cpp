// mp-spice -- Native KLU mixed-precision comparison on circuit-simulation
// matrices from the SuiteSparse Matrix Collection (https://sparse.tamu.edu/).
//
// Solves A*x = b with MTL5's native KLU sparse direct solver across several
// arithmetic types and compares accuracy. This is the headline demonstration
// of the MTL5 (linear algebra) + Universal (number systems) composition:
//
//   double                -- reference
//   float                 -- single precision
//   cfloat<16,5>          -- IEEE-like half precision   (requires Universal)
//   posit<16,2>           -- 16-bit posit               (requires Universal)
//
// Usage:
//   klu_mixed_precision [matrix.mtx]
//
// With no argument, a small synthetic reducible (block-triangular) circuit-like
// matrix is used so the demo runs without any download. Pass a Matrix Market
// file (e.g. Rajat/rajat30 or Freescale/circuit5M, fetched via
// scripts/fetch_matrices.sh) to run on a real circuit matrix.
//
// The cfloat<16,5> / posit<16,2> paths are compiled only when
// MPSPICE_MIXED_PRECISION_KLU is defined (CMake option, OFF by default). They
// are currently blocked by an MTL5 integration issue: sparse_lu.hpp calls
// unqualified std::abs(), which does not find Universal's abs() via ADL. See
// docs/roadmap.md. The double/float comparison works today and is the template
// the low-precision paths will follow once that is resolved.

#include <cstddef>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/io/matrix_market.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>

#ifdef MPSPICE_MIXED_PRECISION_KLU
#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>
#endif

namespace {

using Sparse = mtl::mat::compressed2D<double>;

// A small reducible (block-triangular) matrix that mimics circuit structure:
// two decoupled 3x3 diagonal blocks linked by an upper coupling. Native KLU's
// BTF should find two blocks.
Sparse make_synthetic_circuit() {
    Sparse A(6, 6);
    mtl::mat::inserter<Sparse> ins(A);
    // block {0,1,2}
    ins[0][0] << 3.0;  ins[0][1] << -1.0;
    ins[1][0] << -1.0; ins[1][1] << 3.0;  ins[1][2] << -1.0;
    ins[2][1] << -1.0; ins[2][2] << 3.0;
    // block {3,4,5}
    ins[3][3] << 4.0;  ins[3][4] << -1.0;
    ins[4][3] << -1.0; ins[4][4] << 4.0;  ins[4][5] << -1.0;
    ins[5][4] << -1.0; ins[5][5] << 4.0;
    // upper coupling block {0,1,2} -> {3,4,5}
    ins[0][5] << -2.0;
    return A;
}

// Residual ||A*x - b||_inf, all computed in double.
double residual_inf(const Sparse& A,
                    const std::vector<double>& x,
                    const std::vector<double>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double m = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        double ax = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            ax += dat[k] * x[ci[k]];
        m = std::max(m, std::abs(ax - b[r]));
    }
    return m;
}

// Solve A*x = b in arithmetic type T, return the solution cast back to double.
template <typename T>
std::vector<double> solve_in(const Sparse& A_ref, const std::vector<double>& b_ref) {
    const std::size_t n = A_ref.num_rows();

    // Re-cast the reference system into type T.
    mtl::mat::compressed2D<T> A(n, n);
    {
        mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(A);
        const auto& rp = A_ref.ref_major();
        const auto& ci = A_ref.ref_minor();
        const auto& dat = A_ref.ref_data();
        for (std::size_t r = 0; r < n; ++r)
            for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
                ins[r][ci[k]] << static_cast<T>(dat[k]);
    }
    mtl::vec::dense_vector<T> b(n), x(n, T(0));
    for (std::size_t i = 0; i < n; ++i) b(static_cast<int>(i)) = static_cast<T>(b_ref[i]);

    mtl::sparse::factorization::native_klu_solve(A, x, b);

    std::vector<double> out(n);
    for (std::size_t i = 0; i < n; ++i)
        out[i] = static_cast<double>(x(static_cast<int>(i)));
    return out;
}

// Forward error ||x - exact||_inf (exact solution is known: all ones).
double forward_error_inf(const std::vector<double>& x,
                         const std::vector<double>& exact) {
    double m = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::abs(x[i] - exact[i]));
    return m;
}

// Solve in type T and print a result row. A failed factorization (common for
// low precision on stiff circuit matrices) is reported, not fatal.
template <typename T>
void run_row(const std::string& label, const Sparse& A,
             const std::vector<double>& b, const std::vector<double>& exact) {
    try {
        auto x = solve_in<T>(A, b);
        std::cout << "  " << std::left << std::setw(14) << label << std::right
                  << "   " << std::setw(12) << std::scientific << std::setprecision(3)
                  << residual_inf(A, x, b)
                  << "   " << std::setw(12) << forward_error_inf(x, exact) << '\n';
    } catch (const std::exception& e) {
        std::cout << "  " << std::left << std::setw(14) << label
                  << "   solve failed: " << e.what() << '\n';
    }
}

} // namespace

int main(int argc, char** argv) {
    Sparse A;
    if (argc > 1) {
        std::cout << "Loading Matrix Market file: " << argv[1] << '\n';
        A = mtl::io::mm_read<double>(argv[1]);
    } else {
        std::cout << "No matrix given; using synthetic block-triangular circuit "
                     "matrix.\n(Pass a .mtx file, e.g. via scripts/fetch_matrices.sh, "
                     "to run on a real circuit.)\n";
        A = make_synthetic_circuit();
    }
    const std::size_t n = A.num_rows();
    std::cout << "Matrix: " << n << " x " << A.num_cols()
              << ", nnz = " << A.nnz() << "\n\n";

    // Reproducible RHS: b = A * ones, so the exact solution is all-ones.
    std::vector<double> ones(n, 1.0), b(n, 0.0);
    {
        const auto& rp = A.ref_major();
        const auto& ci = A.ref_minor();
        const auto& dat = A.ref_data();
        for (std::size_t r = 0; r < n; ++r) {
            double s = 0.0;
            for (std::size_t k = rp[r]; k < rp[r + 1]; ++k) s += dat[k] * ones[ci[k]];
            b[r] = s;
        }
    }

    std::cout << "Native KLU solve, exact solution is all-ones:\n\n";
    std::cout << "  " << std::left << std::setw(14) << "type" << std::right
              << "   " << std::setw(12) << "||Ax-b||inf"
              << "   " << std::setw(12) << "||x-1||inf" << '\n';
    std::cout << "  " << std::string(42, '-') << '\n';

    run_row<double>("double", A, b, ones);
    run_row<float>("float", A, b, ones);

#ifdef MPSPICE_MIXED_PRECISION_KLU
    run_row<sw::universal::cfloat<16, 5>>("cfloat<16,5>", A, b, ones);
    run_row<sw::universal::posit<16, 2>>("posit<16,2>", A, b, ones);
#else
    std::cout << "\n  [cfloat<16,5> and posit<16,2> paths disabled: build with "
                 "-DMPSPICE_MIXED_PRECISION_KLU=ON]\n";
#endif

    return 0;
}
