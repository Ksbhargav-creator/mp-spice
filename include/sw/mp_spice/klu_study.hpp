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
#include <map>
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

/// Bucketed by decade (floor(log10(|value|))) rather than linearly, since
/// product magnitudes in a circuit matrix's residual can span many orders of
/// magnitude in a single row. Exact zeros are tracked separately.
class ProductMagnitudeStats {
public:
    void record(double value) {
        if (value == 0.0) { ++zero_count_; return; }
        int decade = static_cast<int>(std::floor(std::log10(std::abs(value))));
        ++buckets_[decade];
        ++total_;
    }
    void reset() { buckets_.clear(); zero_count_ = 0; total_ = 0; }
    const std::map<int, std::size_t>& buckets() const { return buckets_; }
    std::size_t zero_count() const { return zero_count_; }
    std::size_t total() const { return total_; }

private:
    std::map<int, std::size_t> buckets_;  ///< decade -> count, sorted by decade
    std::size_t zero_count_ = 0;
    std::size_t total_ = 0;
};

inline ProductMagnitudeStats product_magnitude_stats;

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

/// Scale every entry of A by a scalar mu. mu*A*x = mu*b has the same solution x as
/// A*x = b, so this changes nothing mathematically -- what it changes is the
/// rounding error `recast<T>` incurs: posits (like every floating format) carry
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

/// Scale b by the same mu used on A.
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

/// Residual r = b - A*x formed with low-precision products: both the
/// matrix `A` and `x` are read at `Value` precision for every multiply, and only the
/// *accumulation* runs at `ResidualAccumulator` precision.
///
/// Also records every product term `a_ij * x_j` into `product_magnitude_stats`and every 
/// row's term count into `residual_stats` -- callers wanting a clean sample should `reset()`
/// both before the call they care about. This works for ANY `Value`,
/// including plain `double` (the default `accumulator_traits<double,double>`
/// specialization is a zero-overhead identity, so `Value=double,
/// ResidualAccumulator=double` reproduces MTL5's own double-residual
/// arithmetic exactly, with the instrumentation as a side effect).
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
            const Value xv = static_cast<Value>(x(static_cast<int>(ci[k])));
            product_magnitude_stats.record(static_cast<double>(dat[k] * xv));
            AT::add_product(acc, dat[k], xv);
            ++accum_len;
        }

        residual_stats.record(accum_len);

        const double ax = AT::template value<double>(acc);
        r(static_cast<int>(i)) = b(static_cast<int>(i)) - ax;
    }
}

/// Iterative refinement with an mp-spice-selected residual accumulator.
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

/// Same question as `mixed_refine` (factor once in low precision T, refine
/// with a DOUBLE-precision residual) -- but routed through
/// `iterative_refine_accumulated_residual<double, double>` over the ORIGINAL
/// double matrix, instead of MTL5's generic core, so
/// `residual_with_accumulator`'s instrumentation (`product_magnitude_stats`,
/// `residual_stats`) records every term of every residual dot product formed
/// during the run.
///
/// Mathematically identical to `mixed_refine`: `Value=double,
/// ResidualAccumulator=double` is the default `accumulator_traits<double,
/// double>` identity specialization (`a += m*v`, zero overhead) -- the exact
/// same plain-double arithmetic MTL5's own `iterative_refine<double>` loop
/// performs. This function exists purely to expose the histogram hook without
/// adding research-specific instrumentation to MTL5's clean generic core (the
/// same reasoning `iterative_refine_accumulated_residual` itself documents).
///
/// Caller should `product_magnitude_stats.reset()` (and `residual_stats.reset()`
/// if the accumulation-length summary is also wanted) before calling, then read
/// the stats back out afterward -- they accumulate across every iteration of
/// the run, not just one snapshot.
template <typename T, typename FactorAccumulator = T>
solve_stats mixed_refine_with_histogram(const DSparse& A,
                                        const std::vector<double>& b,
                                        const std::vector<double>& exact,
                                        int max_iter = 30,
                                        double tol = 1e-14,
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
            T, mtl::mat::parameters<>, FactorAccumulator>(AT);          // factor once in T

        mtl::vec::dense_vector<double> bv(n), xv(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) bv(static_cast<int>(i)) = buse[i];

        mtl::sparse::refine_options opt;
        opt.max_iter = max_iter;
        opt.rel_tol  = tol;
        // Value=double over the ORIGINAL matrix -- same "double residual"
        // question as mixed_refine, routed through the instrumented path.
        auto rr = iterative_refine_accumulated_residual<double, double>(Ause, fac, bv, xv, opt);

        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = xv(static_cast<int>(i));
        s.iters = rr.iters;
        s.residual = residual_inf(A, x, b);
        s.fwd_error = forward_error_inf(x, exact);
        s.ok = true;
    } catch (const std::exception& e) { s.error = e.what(); }
    return s;
}


template <typename Working, typename ResidualAccumulator = Working>
solve_stats working_precision_refine_with_histogram(const DSparse& A,
                                                     const std::vector<double>& b,
                                                     const std::vector<double>& exact,
                                                     int max_iter = 30,
                                                     double tol = 1e-14,
                                                     double mu = 1.0) {
    solve_stats s;
    try {
        const std::size_t n = A.num_rows();
        const bool scale = (mu != 1.0);
        const DSparse Ascaled = scale ? scale_matrix(A, mu) : DSparse{};
        const std::vector<double> bscaled = scale ? scale_rhs(b, mu) : std::vector<double>{};
        const DSparse& Ause = scale ? Ascaled : A;
        const std::vector<double>& buse = scale ? bscaled : b;

        auto AW = recast<Working>(Ause);
        // Factorization is ALWAYS plain at Working precision -- quire never
        // goes here; this is not the axis under test.
        auto fac = mtl::sparse::factorization::native_klu_factor<
            Working, mtl::mat::parameters<>, Working>(AW);

        mtl::vec::dense_vector<double> bv(n), xv(n, 0.0);
        for (std::size_t i = 0; i < n; ++i) bv(static_cast<int>(i)) = buse[i];

        mtl::sparse::refine_options opt;
        opt.max_iter = max_iter;
        opt.rel_tol  = tol;
        // Residual formed at Working precision -- the SAME precision as the
        // factorization, never downcast further. ResidualAccumulator (plain
        // vs quire_acc<Working>) is the only thing this function varies.
        auto rr = iterative_refine_accumulated_residual<Working, ResidualAccumulator>(
            AW, fac, bv, xv, opt);

        std::vector<double> x(n);
        for (std::size_t i = 0; i < n; ++i) x[i] = xv(static_cast<int>(i));
        s.iters = rr.iters;
        s.residual = residual_inf(A, x, b);
        s.fwd_error = forward_error_inf(x, exact);
        s.ok = true;
    } catch (const std::exception& e) { s.error = e.what(); }
    return s;
}


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
