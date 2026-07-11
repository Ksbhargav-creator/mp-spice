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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>
#include <mtl/sparse/iterative_refine.hpp>
#include <mtl/sparse/instrumentation/kernel_stats.hpp>

inline mtl::sparse::instrumentation::KernelStatistics residual_stats;

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

/// Scale every entry of A by a scalar mu (Quinlan & Omtzigt, "Iterative
/// Refinement with Low-Precision Posits," Algorithm 4: "Scale matrix entries,
/// then round to low-precision"). mu*A*x = mu*b has the SAME solution x as
/// A*x = b, so this changes nothing mathematically -- what it changes is the
/// ROUNDING ERROR `recast<T>` incurs: posits (like every floating format) carry
/// their best relative precision near magnitude 1 and taper off toward the
/// extremes of their dynamic range, so choosing mu to pull A's entries toward
/// O(1) before casting to T reduces the per-entry error introduced by the cast,
/// without needing a second-order correction anywhere downstream.
inline DSparse scale_matrix(const DSparse& A, double mu) {
    std::size_t n = A.num_rows();
    DSparse S(n, A.num_cols());
    mtl::mat::inserter<DSparse> ins(S);
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            ins[r][ci[k]] << mu * dat[k];
    return S;
}

/// Scale b by the same mu used on A (Algorithm 4 requires both, "to preserve
/// the equality of the system after scaling the matrix").
inline std::vector<double> scale_rhs(const std::vector<double>& b, double mu) {
    std::vector<double> bs(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) bs[i] = mu * b[i];
    return bs;
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

/// ||M||_inf = max row sum of |M(i,j)|, computed directly from CSR storage
/// (O(nnz), not a dense O(n^2) scan over every (i,j) pair) and always returned
/// in double regardless of M's element type -- feeds the textbook convergence
/// bound below.
template <typename MatType>
double matrix_inf_norm(const MatType& M) {
    const auto& rp = M.ref_major();
    const auto& dat = M.ref_data();
    double result = 0.0;
    for (std::size_t r = 0; r < M.num_rows(); ++r) {
        double row_sum = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            row_sum += std::abs(static_cast<double>(dat[k]));
        result = std::max(result, row_sum);
    }
    return result;
}

/// Residual r = b - A*x formed with GENUINELY low-precision products: both the
/// matrix `A` (already recast to `Value`, the same type the factorization runs
/// in) and `x` are read at `Value` precision for every multiply, and only the
/// *accumulation* runs at `ResidualAccumulator` precision. This is what makes
/// the residual mixed precision -- the bulk of the work (one product per
/// nonzero) happens in the low-precision type; only the summation gets the
/// wider/exact treatment. The earlier version of this function took the
/// double-precision matrix and upcast everything to `Value` (or left it at
/// `double`), so the accumulator never saw a genuinely low-precision product to
/// compensate for -- with `ResidualAccumulator = Value` (the pessimistic
/// low-precision-everywhere baseline) or `ResidualAccumulator = quire_acc<Value>`
/// (single-rounding fused dot product), the choice now actually changes what
/// gets computed.
///
/// `Value` names the operand/element type the same way MTL5's own accumulator
/// seam does (`accumulator_traits<Acc, Value>`, and every `Accumulator = Value`
/// default in sparse_lu.hpp/native_klu.hpp/triangular_solve.hpp) -- there's no
/// separate concept here, so it keeps the same name.
template <typename Value, typename ResidualAccumulator = Value>
void residual_with_accumulator(const mtl::mat::compressed2D<Value>& A,
                               const mtl::vec::dense_vector<double>& b,
                               const mtl::vec::dense_vector<double>& x,
                               mtl::vec::dense_vector<double>& r) {
    using AT = mtl::math::accumulator_traits<ResidualAccumulator, Value>;

    const std::size_t n = A.num_rows();
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();  // already Value -- no upcast to double

    for (std::size_t i = 0; i < n; ++i) {
        ResidualAccumulator acc{};
        AT::clear(acc);
        std::size_t accum_len = 0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) {
            AT::add_product(acc, dat[k], static_cast<Value>(x(static_cast<int>(ci[k]))));
            ++accum_len;
        }

        residual_stats.record(accum_len);

        const double ax = AT::template value<double>(acc);
        r(static_cast<int>(i)) = b(static_cast<int>(i)) - ax;
    }
}

/// Iterative refinement with an mp-spice-selected residual accumulator.
///
/// This intentionally lives in mp-spice rather than MTL5 so the sparse LU
/// implementation and the generic MTL5 refinement API remain unchanged. Only
/// the residual formation `r = b - A*x` is replaced by the accumulator policy --
/// and, unlike MTL5's generic (double-only) `iterative_refine` core, `A` here is
/// the matrix already recast to `Value` (the same low precision the
/// factorization itself runs in), so the residual's A*x products are genuinely
/// low precision and the accumulator is doing real compensating work.
template <typename Value,
          typename ResidualAccumulator = Value,
          typename Factorization>
mtl::sparse::refine_result iterative_refine_accumulated_residual(
    const mtl::mat::compressed2D<Value>& A,
    const Factorization& fac,
    const mtl::vec::dense_vector<double>& b,
    mtl::vec::dense_vector<double>& x,
    const mtl::sparse::refine_options& opt = {}) {
    const std::size_t n = A.num_rows();
    if (A.num_cols() != n)
        throw std::invalid_argument("iterative_refine_accumulated_residual: matrix must be square");
    if (static_cast<std::size_t>(b.size()) != n || static_cast<std::size_t>(x.size()) != n)
        throw std::invalid_argument("iterative_refine_accumulated_residual: b/x size does not match A");

    auto norm_inf_vec = [&](const mtl::vec::dense_vector<double>& v) {
        double m = 0.0;
        for (std::size_t i = 0; i < n; ++i)
            m = std::max(m, std::abs(v(static_cast<int>(i))));
        return m;
    };

    const double bnorm = norm_inf_vec(b);
    const double Ainf = matrix_inf_norm(A);  // ||A||_inf, from the SAME (low-precision)
                                              // matrix this function forms its residual from
    mtl::vec::dense_vector<double> r(n), dx(n, 0.0);
    mtl::vec::dense_vector<double> best_x = x;
    double best_rn = std::numeric_limits<double>::infinity();
    const int patience = std::max(1, opt.patience);
    int stalls = 0;

    mtl::sparse::refine_result res;
    for (int it = 0; it < opt.max_iter; ++it) {
        residual_with_accumulator<Value, ResidualAccumulator>(A, b, x, r);

        const double rn = norm_inf_vec(r);
        if (rn < best_rn) { best_rn = rn; best_x = x; stalls = 0; }
        else              { ++stalls; }

        // ||r_i|| <= rel_tol * (||A||_inf*||x_i||_inf + ||b||_inf) -- see
        // mtl::sparse::refine_options::rel_tol's doc comment (MTL5's core uses
        // the identical criterion; this mirrors it here since this function
        // deliberately reimplements the loop rather than calling into MTL5).
        const double xn = norm_inf_vec(x);
        const double bound = Ainf * xn + bnorm;
        if (opt.rel_tol > 0.0 && rn <= opt.rel_tol * bound) { res.converged = true; break; }
        if (stalls >= patience) break;

        if (opt.scaled) {
            const double rho = rn;
            if (rn == 0.0) break;
            for (std::size_t i = 0; i < n; ++i) r(static_cast<int>(i)) /= rho;
            fac.solve(dx, r);
            for (std::size_t i = 0; i < n; ++i)
                x(static_cast<int>(i)) += rho * dx(static_cast<int>(i));
        } else {
            fac.solve(dx, r);
            for (std::size_t i = 0; i < n; ++i)
                x(static_cast<int>(i)) += dx(static_cast<int>(i));
        }
        ++res.iters;
    }

    if (!std::isfinite(best_rn)) {
        residual_with_accumulator<Value, ResidualAccumulator>(A, b, x, r);
        best_rn = norm_inf_vec(r);
    }

    x = best_x;
    res.rel_residual = (bnorm > 0.0) ? best_rn / bnorm : best_rn;
    return res;
}

/// Direct solve of A x = b entirely in arithmetic type T (native KLU). The
/// optional `Accumulator` selects the per-block accumulator policy (default:
/// ordinary T arithmetic; pass e.g. a posit quire for an exact fused dot product).
///
/// `mu` applies Algorithm 4 (Quinlan & Omtzigt): A and b are scaled by `mu`
/// before rounding to T (default mu=1.0 -- no scaling, unchanged behavior).
/// Since mu*A*x = mu*b has the same solution x as A*x = b, the reported
/// residual/forward-error are always computed against the ORIGINAL (unscaled)
/// A and b, so results are comparable across mu.
template <typename T, typename Accumulator = T>
solve_stats direct_solve(const DSparse& A,
                         const std::vector<double>& b,
                         const std::vector<double>& exact,
                         double mu = 1.0) {
    solve_stats s;
    try {
        std::size_t n = A.num_rows();
        const bool scale = (mu != 1.0);
        const DSparse Ascaled = scale ? scale_matrix(A, mu) : DSparse{};
        const std::vector<double> bscaled = scale ? scale_rhs(b, mu) : std::vector<double>{};
        const DSparse& Ause = scale ? Ascaled : A;
        const std::vector<double>& buse = scale ? bscaled : b;

        auto AT = recast<T>(Ause);
        mtl::vec::dense_vector<T> bT(n), xT(n, T(0));
        for (std::size_t i = 0; i < n; ++i) bT(static_cast<int>(i)) = static_cast<T>(buse[i]);
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
///
/// `mu` applies Algorithm 4 (Quinlan & Omtzigt, "Iterative Refinement with
/// Low-Precision Posits"): "scale matrix entries, then round to low-precision."
/// A and b are scaled by `mu` before the factorization's recast to T AND before
/// every double-precision residual evaluation in the refinement loop -- the
/// paper's text is explicit that "[a]fter scaling and rounding the matrix,
/// Algorithm 2 [mixed-precision IR] will be applied," i.e. scaling is a
/// pre-processing step, not a change to the refinement algorithm itself.
/// Default mu=1.0 leaves behavior unchanged. Since mu*A*x = mu*b has the same
/// solution x as A*x = b, the reported residual/forward-error are always
/// computed against the ORIGINAL (unscaled) A and b, so results stay
/// comparable across different mu.
template <typename T, typename Accumulator = T>
solve_stats mixed_refine(const DSparse& A,
                         const std::vector<double>& b,
                         const std::vector<double>& exact,
                         int max_iter = 30,
                         double tol = 1e-14,
                         bool scaled = false,
                         double mu = 1.0) {
    solve_stats s;
    try {
        const std::size_t n = A.num_rows();
        const bool scale = (mu != 1.0);
        const DSparse Ascaled = scale ? scale_matrix(A, mu) : DSparse{};
        const std::vector<double> bscaled = scale ? scale_rhs(b, mu) : std::vector<double>{};
        const DSparse& Ause = scale ? Ascaled : A;
        const std::vector<double>& buse = scale ? bscaled : b;

        auto AT = recast<T>(Ause);
        auto fac = mtl::sparse::factorization::native_klu_factor<
            T, mtl::mat::parameters<>, Accumulator>(AT);             // factor once in T

        mtl::vec::dense_vector<double> bv(n), xv(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) bv(static_cast<int>(i)) = buse[i];

        mtl::sparse::refine_options opt;
        opt.max_iter = max_iter;
        opt.rel_tol  = tol;
        opt.scaled   = scaled;
        auto rr = mtl::sparse::iterative_refine<double>(Ause, fac, bv, xv, opt);

        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = xv(static_cast<int>(i));
        s.iters = rr.iters;
        s.residual = residual_inf(A, x, b);
        s.fwd_error = forward_error_inf(x, exact);
        s.ok = true;
    } catch (const std::exception& e) { s.error = e.what(); }
    return s;
}

/// Mixed-precision iterative refinement with a caller-selected residual
/// accumulator -- the genuinely mixed-precision counterpart to `mixed_refine`.
///
/// `mixed_refine` forms its residual entirely in double via MTL5's generic core:
/// cheap low-precision factor, expensive double residual. This variant instead
/// forms the residual from the SAME low-precision matrix `T` the factorization
/// uses, so the A*x products genuinely run at `T` precision, and relies on
/// `ResidualAccumulator` to recover accuracy: `ResidualAccumulator = T` (the
/// default) is the pessimistic low-precision-everywhere baseline;
/// `ResidualAccumulator = quire_acc<T>` tests whether an exact (single-rounding)
/// accumulation over `T`-precision products can match the double-residual result
/// without ever materializing a double product.
///
/// `FactorAccumulator` independently controls the LU factor/solve accumulator
/// (as in `mixed_refine`); it need not match `ResidualAccumulator`.
///
/// `mu` applies Algorithm 4 (see `mixed_refine`): A and b are scaled by `mu`
/// before rounding to T, for both the factorization and the residual's A*x
/// products (both now run at T precision -- see `residual_with_accumulator`).
/// Default mu=1.0 leaves behavior unchanged; reported residual/forward-error
/// are always against the original (unscaled) A and b.
template <typename T,
          typename FactorAccumulator = T,
          typename ResidualAccumulator = T>
solve_stats mixed_refine_residual_accumulator(const DSparse& A,
                                              const std::vector<double>& b,
                                              const std::vector<double>& exact,
                                              int max_iter = 30,
                                              double tol = 1e-14,
                                              bool scaled = false,
                                              double mu = 1.0) {
    solve_stats s;
    try {
        const std::size_t n = A.num_rows();
        const bool scale = (mu != 1.0);
        const DSparse Ascaled = scale ? scale_matrix(A, mu) : DSparse{};
        const std::vector<double> bscaled = scale ? scale_rhs(b, mu) : std::vector<double>{};
        const DSparse& Ause = scale ? Ascaled : A;
        const std::vector<double>& buse = scale ? bscaled : b;

        auto AT = recast<T>(Ause);  // low-precision matrix -- shared by factorization AND residual
        auto fac = mtl::sparse::factorization::native_klu_factor<
            T, mtl::mat::parameters<>, FactorAccumulator>(AT);

        mtl::vec::dense_vector<double> bv(n), xv(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) bv(static_cast<int>(i)) = buse[i];

        mtl::sparse::refine_options opt;
        opt.max_iter = max_iter;
        opt.rel_tol  = tol;
        opt.scaled   = scaled;
        auto rr = iterative_refine_accumulated_residual<T, ResidualAccumulator>(
            AT, fac, bv, xv, opt);

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
/// Not to be confused with `mu` (Algorithm 4's matrix scaling, passed through
/// unchanged here) -- that scales A/b once before rounding; this scales each
/// residual's magnitude every refinement step. The two are independent and
/// composable.
template <typename T, typename Accumulator = T>
solve_stats mixed_refine_scaled(const DSparse& A,
                                const std::vector<double>& b,
                                const std::vector<double>& exact,
                                int max_iter = 30,
                                double tol = 1e-14,
                                double mu = 1.0) {
    return mixed_refine<T, Accumulator>(A, b, exact, max_iter, tol, /*scaled=*/true, mu);
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
