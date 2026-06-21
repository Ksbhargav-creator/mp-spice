#pragma once
// mp-spice -- mixed-precision KLU study utilities
//
// Reusable building blocks for the native-KLU mixed-precision study (mp-spice
// #1/#5): recast a double system into an arbitrary arithmetic type, solve it
// with MTL5's native KLU, and -- the headline -- run mixed-precision iterative
// refinement (factor in low precision, refine with a double-precision residual)
// to recover accuracy a low-precision direct solve alone cannot reach.
//
// Composition layer over MTL5 (linear algebra) + Universal (number systems).

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>
#include <mtl/sparse/iterative_refine.hpp>

namespace sw::mp_spice {

using DSparse = mtl::mat::compressed2D<double>;

/// Result of one solve configuration.
struct solve_stats {
    bool        ok = false;
    std::string error;         ///< failure reason if !ok
    double      residual = 0;  ///< ||A x - b||_inf  (computed in double)
    double      fwd_error = 0; ///< ||x - exact||_inf
    int         iters = 0;     ///< refinement steps (0 for direct solve)
};

/// Recast a double CSR matrix into arithmetic type T.
template <typename T>
mtl::mat::compressed2D<T> recast(const DSparse& A) {
    std::size_t n = A.num_rows();
    mtl::mat::compressed2D<T> M(n, A.num_cols());
    mtl::mat::inserter<mtl::mat::compressed2D<T>> ins(M);
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            ins[r][ci[k]] << static_cast<T>(dat[k]);
    return M;
}

/// Residual ||A x - b||_inf, computed in double regardless of the solve type.
inline double residual_inf(const DSparse& A,
                           const std::vector<double>& x,
                           const std::vector<double>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double m = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        double ax = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k) ax += dat[k] * x[ci[k]];
        m = std::max(m, std::abs(ax - b[r]));
    }
    return m;
}

inline double forward_error_inf(const std::vector<double>& x,
                                const std::vector<double>& exact) {
    double m = 0.0;
    for (std::size_t i = 0; i < x.size(); ++i)
        m = std::max(m, std::abs(x[i] - exact[i]));
    return m;
}

inline double norm_inf(const std::vector<double>& v) {
    double m = 0.0;
    for (double e : v) m = std::max(m, std::abs(e));
    return m;
}

/// Direct solve of A x = b entirely in arithmetic type T (native KLU). The
/// optional `Accumulator` selects the per-block accumulator policy (default:
/// ordinary T arithmetic; pass e.g. a posit quire for an exact fused dot product).
template <typename T, typename Accumulator = T>
solve_stats direct_solve(const DSparse& A,
                         const std::vector<double>& b,
                         const std::vector<double>& exact) {
    solve_stats s;
    try {
        std::size_t n = A.num_rows();
        auto AT = recast<T>(A);
        mtl::vec::dense_vector<T> bT(n), xT(n, T(0));
        for (std::size_t i = 0; i < n; ++i) bT(static_cast<int>(i)) = static_cast<T>(b[i]);
        auto fac = mtl::sparse::factorization::native_klu_factor<
            T, mtl::mat::parameters<>, Accumulator>(AT);
        fac.solve(xT, bT);
        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = static_cast<double>(xT(static_cast<int>(i)));
        s.residual = residual_inf(A, x, b);
        s.fwd_error = forward_error_inf(x, exact);
        s.ok = true;
    } catch (const std::exception& e) { s.error = e.what(); }
    return s;
}

/// Mixed-precision iterative refinement: factor A once in type T, then refine
/// with a DOUBLE-precision residual via MTL5's generic `iterative_refine` core
/// (stillwater-sc/mtl5#119). The factorization's solve runs in T; the residual
/// and corrections are carried in double.
///
/// `Accumulator` selects the per-block accumulator policy of the low-precision
/// factorization (default: ordinary T arithmetic; e.g. a posit quire for a fused
/// dot product). `scaled` normalizes each residual to O(1) before the correction
/// solve (carrying the magnitude in double) -- rescues narrow-exponent types.
///
/// Note: `iters` counts every correction step including the initial solve (x
/// starts at zero), and the core returns the best iterate, stopping once the
/// residual stops improving.
template <typename T, typename Accumulator = T>
solve_stats mixed_refine(const DSparse& A,
                         const std::vector<double>& b,
                         const std::vector<double>& exact,
                         int max_iter = 30,
                         double tol = 1e-14,
                         bool scaled = false) {
    solve_stats s;
    try {
        const std::size_t n = A.num_rows();
        auto AT = recast<T>(A);
        auto fac = mtl::sparse::factorization::native_klu_factor<
            T, mtl::mat::parameters<>, Accumulator>(AT);             // factor once in T

        mtl::vec::dense_vector<double> bv(n), xv(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) bv(static_cast<int>(i)) = b[i];

        mtl::sparse::refine_options opt;
        opt.max_iter = max_iter;
        opt.rel_tol  = tol;
        opt.scaled   = scaled;
        auto rr = mtl::sparse::iterative_refine<double>(A, fac, bv, xv, opt);

        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = xv(static_cast<int>(i));
        s.iters = rr.iters;
        s.residual = residual_inf(A, x, b);
        s.fwd_error = forward_error_inf(x, exact);
        s.ok = true;
    } catch (const std::exception& e) { s.error = e.what(); }
    return s;
}

/// Scaled mixed-precision iterative refinement (see `mixed_refine`, `scaled=true`):
/// each residual is normalized to O(1) before the low-precision correction solve
/// and its magnitude restored in double, rescuing narrow-exponent factor types.
template <typename T, typename Accumulator = T>
solve_stats mixed_refine_scaled(const DSparse& A,
                                const std::vector<double>& b,
                                const std::vector<double>& exact,
                                int max_iter = 30,
                                double tol = 1e-14) {
    return mixed_refine<T, Accumulator>(A, b, exact, max_iter, tol, /*scaled=*/true);
}

/// Build a reproducible RHS b = A * ones, so the exact solution is all-ones.
inline std::vector<double> rhs_from_ones(const DSparse& A) {
    std::size_t n = A.num_rows();
    std::vector<double> b(n, 0.0);
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    for (std::size_t r = 0; r < n; ++r) {
        double sum = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k) sum += dat[k];  // * 1
        b[r] = sum;
    }
    return b;
}

} // namespace sw::mp_spice
