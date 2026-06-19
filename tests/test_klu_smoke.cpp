// mp-spice smoke test: verify the MTL5 + Universal composition builds and that
// MTL5's native KLU solves a small block-triangular system. Returns non-zero on
// failure (no external test framework, matching the repo's lightweight style).
#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>

// Universal number types: native KLU must solve in these through the MTL5 +
// Universal composition (relies on the ADL-abs fix, stillwater-sc/mtl5#121).
#include <universal/number/posit/posit.hpp>
#include <universal/number/cfloat/cfloat.hpp>

namespace {

template <typename T>
bool solve_ok(double tol) {
    using Sparse = mtl::mat::compressed2D<T>;
    Sparse A(4, 4);
    {
        mtl::mat::inserter<Sparse> ins(A);
        // Two 2x2 diagonal blocks with an upper coupling -> reducible.
        ins[0][0] << T(2); ins[0][1] << T(1); ins[0][3] << T(5);
        ins[1][0] << T(1); ins[1][1] << T(3);
        ins[2][2] << T(4); ins[2][3] << T(1);
        ins[3][2] << T(2); ins[3][3] << T(5);
    }
    mtl::vec::dense_vector<T> b = {T(1), T(2), T(3), T(4)};
    mtl::vec::dense_vector<T> x(4, T(0));
    mtl::sparse::factorization::native_klu_solve(A, x, b);

    // residual ||A*x - b||_inf in double
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double r = 0.0;
    for (std::size_t i = 0; i < 4; ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ax += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        r = std::max(r, std::abs(ax - static_cast<double>(b(static_cast<int>(i)))));
    }
    return r < tol;
}

} // namespace

int main() {
    int failures = 0;

    if (!solve_ok<double>(1e-10)) { std::cerr << "double KLU solve failed\n"; ++failures; }
    if (!solve_ok<float>(1e-4))   { std::cerr << "float KLU solve failed\n";  ++failures; }

    // Native KLU through the MTL5 + Universal composition, in low precision.
    if (!solve_ok<sw::universal::cfloat<16, 5>>(1e-1)) {
        std::cerr << "cfloat<16,5> KLU solve failed\n"; ++failures;
    }
    if (!solve_ok<sw::universal::posit<16, 2>>(1e-1)) {
        std::cerr << "posit<16,2> KLU solve failed\n"; ++failures;
    }

    if (failures == 0) std::cout << "mp-spice smoke test passed\n";
    return failures == 0 ? 0 : 1;
}
