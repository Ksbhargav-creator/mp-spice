// mp-spice -- quire super-accumulator study for sparse LU (mp-spice #3)
//
// Compares posit sparse-LU factorization with and without an exact quire
// accumulator (a fused dot product inside the elimination), across posit widths.
// The quire makes every column's inner products round only once, so a
// low-precision factorization keeps accuracy a naive per-operation-rounded
// accumulation loses.
//
// Uses MTL5's sparse_lu directly (single elimination, no BTF) to isolate the
// accumulator's effect. Requires Universal + the MTL5 accumulator seam (#122);
// build with -DMPSPICE_MIXED_PRECISION_KLU=ON (default ON).
//
// Usage: klu_quire_study [matrix.mtx] [--csv out.csv]
// No matrix -> an ill-conditioned Hilbert-like system where the effect is clear.

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/io/matrix_market.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>

#include <universal/number/posit/posit.hpp>
#include <sw/mp_spice/quire_accumulator.hpp>
#include <sw/mp_spice/klu_study.hpp>

namespace {

using Dbl = mtl::mat::compressed2D<double>;

// Dense, diagonally-dominant (well-conditioned) matrix with deterministic
// mixed-sign off-diagonals. Dense -> each elimination accumulates ~n products
// with cancellation; well-conditioned -> the solution is meaningful, so the quire
// (one rounding over the whole accumulation) shows a clean benefit over per-op
// rounding without conditioning swamping the result.
Dbl dense_mixed(std::size_t n) {
    Dbl A(n, n);
    mtl::mat::inserter<Dbl> ins(A);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            ins[i][j] << (i == j ? static_cast<double>(n)
                                 : std::sin(static_cast<double>(i * 7 + j * 13)));  // in [-1,1]
    return A;
}

// Residual ||A x - b||_inf and forward error ||x - 1||_inf, both in double.
template <typename P>
void errors(const mtl::mat::compressed2D<P>& A,
            const mtl::vec::dense_vector<P>& x,
            const mtl::vec::dense_vector<P>& b,
            double& res, double& ferr) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    res = 0.0; ferr = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        double ax = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            ax += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        res = std::max(res, std::abs(ax - static_cast<double>(b(static_cast<int>(r)))));
        ferr = std::max(ferr, std::abs(static_cast<double>(x(static_cast<int>(r))) - 1.0));
    }
}

struct Result { std::string type; double plain_res, plain_ferr, quire_res, quire_ferr; bool ok; };

// Factor A (double) in posit type P twice -- plain accumulation vs quire -- and
// solve A x = b with b = A*1 so the exact solution is all-ones.
template <typename P>
Result compare(const std::string& type, const Dbl& Ad) {
    using Sparse = mtl::mat::compressed2D<P>;
    Result r{type, 0, 0, 0, 0, false};
    try {
        std::size_t n = Ad.num_rows();
        Sparse A(n, n);
        {
            mtl::mat::inserter<Sparse> ins(A);
            const auto& rp = Ad.ref_major(); const auto& ci = Ad.ref_minor(); const auto& dat = Ad.ref_data();
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) ins[i][ci[k]] << static_cast<P>(dat[k]);
        }
        // b = A * ones (in posit) -> exact solution all-ones.
        mtl::vec::dense_vector<P> ones(n, P(1)), b(n, P(0));
        {
            const auto& rp = A.ref_major(); const auto& ci = A.ref_minor(); const auto& dat = A.ref_data();
            for (std::size_t i = 0; i < n; ++i) {
                P s(0);
                for (std::size_t k = rp[i]; k < rp[i + 1]; ++k) s += dat[k];
                b(static_cast<int>(i)) = s;
            }
        }
        auto sym = mtl::sparse::factorization::sparse_lu_symbolic(A);

        mtl::vec::dense_vector<P> xp(n, P(0));
        auto plain = mtl::sparse::factorization::sparse_lu_numeric(A, sym);
        plain.solve(xp, b);
        errors(A, xp, b, r.plain_res, r.plain_ferr);

        mtl::vec::dense_vector<P> xq(n, P(0));
        auto quire = mtl::sparse::factorization::sparse_lu_numeric<
            P, mtl::mat::parameters<>, sw::mp_spice::quire_acc<P>>(A, sym);
        quire.solve(xq, b);
        errors(A, xq, b, r.quire_res, r.quire_ferr);
        r.ok = true;
    } catch (const std::exception&) {}
    return r;
}

} // namespace

int main(int argc, char** argv) {
    std::string mtx, csv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csv = argv[++i]; else mtx = a;
    }

    Dbl A;
    if (!mtx.empty()) { std::printf("Loading %s\n", mtx.c_str()); A = mtl::io::mm_read<double>(mtx); }
    else { std::printf("Well-conditioned dense mixed-sign(40) system (pass a .mtx to override).\n"); A = dense_mixed(40); }
    std::printf("Matrix: %zu x %zu, nnz = %zu\n\n",
                (size_t)A.num_rows(), (size_t)A.num_cols(), (size_t)A.nnz());

    using namespace sw::universal;

    // Part 1 -- direct sparse_lu (no BTF, single elimination) plain vs quire.
    // This is an O(fill) dense-ish factorization done THREE times in slow posit
    // arithmetic, so restrict it to small matrices; large circuit matrices use
    // Part 2 (BTF native KLU) below.
    constexpr std::size_t kDirectMax = 256;
    if (A.num_rows() <= kDirectMax) {
        std::vector<Result> rows = {
            compare<posit<8, 2>>("posit<8,2>", A),
            compare<posit<16, 2>>("posit<16,2>", A),
            compare<posit<32, 2>>("posit<32,2>", A),
        };
        std::printf("Direct sparse-LU factorization (plain vs quire accumulator):\n");
        std::printf("%-13s | %11s %11s | %11s %11s\n",
                    "type", "plain res", "plain ferr", "quire res", "quire ferr");
        std::printf("%s\n", std::string(66, '-').c_str());
        for (const auto& r : rows) {
            if (r.ok) std::printf("%-13s | %11.3e %11.3e | %11.3e %11.3e\n",
                                  r.type.c_str(), r.plain_res, r.plain_ferr, r.quire_res, r.quire_ferr);
            else std::printf("%-13s | factorization failed\n", r.type.c_str());
        }
        if (!csv.empty()) {
            std::ofstream o(csv);
            o << "type,plain_residual,plain_fwd_error,quire_residual,quire_fwd_error\n";
            for (const auto& r : rows)
                if (r.ok) o << r.type << ',' << r.plain_res << ',' << r.plain_ferr << ','
                            << r.quire_res << ',' << r.quire_ferr << '\n';
            std::printf("CSV: %s\n", csv.c_str());
        }
    } else {
        std::printf("Direct sparse-LU comparison skipped (n=%zu > %zu; use the "
                    "BTF native-KLU + IR section below for large matrices).\n",
                    (size_t)A.num_rows(), kDirectMax);
    }

    // --- Native KLU + mixed-precision iterative refinement: plain vs quire ---
    // Factor in posit (full BTF KLU) and refine with a double residual. Does the
    // exact per-block accumulator improve the IR result/convergence?
    std::printf("\nNative KLU + double-residual iterative refinement (factor in posit):\n");
    std::printf("%-13s | %11s %11s %5s | %11s %11s %5s\n",
                "type", "plain res", "plain ferr", "it", "quire res", "quire ferr", "it");
    std::printf("%s\n", std::string(74, '-').c_str());
    std::vector<double> ones(A.num_rows(), 1.0);
    auto b = sw::mp_spice::rhs_from_ones(A);
    auto ir_row = [&](const std::string& type, auto tag) {
        using P = decltype(tag);
        auto plain = sw::mp_spice::mixed_refine<P>(A, b, ones);
        auto quire = sw::mp_spice::mixed_refine<P, sw::mp_spice::quire_acc<P>>(A, b, ones);
        auto cell = [](const sw::mp_spice::solve_stats& s) {
            if (s.ok) std::printf(" %11.3e %11.3e %5d", s.residual, s.fwd_error, s.iters);
            else      std::printf(" %11s %11s %5s", "FAIL", "-", "-");
        };
        std::printf("%-13s |", type.c_str()); cell(plain); std::printf(" |"); cell(quire); std::printf("\n");
    };
    ir_row("posit<16,2>", posit<16, 2>{});
    ir_row("posit<32,2>", posit<32, 2>{});
    return 0;
}
