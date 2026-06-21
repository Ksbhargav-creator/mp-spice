#pragma once
// mp-spice -- Universal quire super-accumulator adapter for MTL5's sparse LU
//
// MTL5's sparse_lu_numeric accumulates its dense workspace through the policy
// point accumulator_traits<Acc, Value> (stillwater-sc/mtl5#122), and stays free
// of any external number library. This header is the mp-spice composition that
// fills that seam with a Universal posit *quire* -- a fixed-point
// super-accumulator wide enough to hold sums of posit products EXACTLY, so the
// inner products of the factorization round only once (a fused dot product).
//
// Inject it by instantiating sparse_lu_numeric with the quire accumulator:
//   using Posit = sw::universal::posit<16,2>;
//   sparse_lu_numeric<Posit, mat::parameters<>,
//                     sw::mp_spice::quire_acc<Posit>>(A, sym);
//
// Requires Universal (build with MPSPICE_MIXED_PRECISION_KLU=ON).

#include <universal/number/posit/posit.hpp>
#include <universal/number/quire/quire.hpp>

#include <mtl/sparse/factorization/sparse_lu.hpp>

namespace sw::mp_spice {

/// Accumulator state: a Universal quire sized for posit type P. Wrapped in a
/// struct so it is a distinct type the accumulator_traits specialization keys on.
template <typename P>
struct quire_acc {
    sw::universal::quire<P> q;
};

} // namespace sw::mp_spice

namespace mtl::sparse::factorization {

/// Specialize MTL5's accumulator seam for the posit quire. Products accumulate
/// exactly into the quire; value() resolves (rounds) to a posit once, at the
/// point the column entry is consumed -- giving single-rounding fused-dot-product
/// semantics for the whole left-looking elimination of each column.
template <typename P>
struct accumulator_traits<sw::mp_spice::quire_acc<P>, P> {
    using QA = sw::mp_spice::quire_acc<P>;

    static void clear(QA& a) { a.q.reset(); }

    static void assign(QA& a, const P& v) {
        a.q.reset();
        a.q += sw::universal::quire_mul(v, P(1));   // exact: v * 1
    }

    static P value(const QA& a) {
        sw::universal::quire<P> tmp(a.q);           // quire_resolve takes a mutable copy
        return sw::universal::quire_resolve(tmp);
    }

    static void add_product(QA& a, const P& m, const P& v) {
        a.q += sw::universal::quire_mul(m, v);      // exact product, no rounding
    }
};

} // namespace mtl::sparse::factorization
