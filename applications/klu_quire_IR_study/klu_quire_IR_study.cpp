// mp-spice -- native KLU + iterative refinement quire study (mp-spice #3/#5)
//
// Companion to klu_quire_study, split out because these two questions use a
// different solve path and shouldn't share a binary:
//
//   klu_quire_study     -- one-shot DIRECT solve, plain vs quire accumulator
//                           INSIDE the factorization (MTL5 sparse_lu, no BTF).
//   klu_quire_IR_study   -- native KLU (full BTF) + iterative refinement, two
//                           independent questions:
//
//     1. Factor-accumulator IR: factor A in posit (plain vs quire), then
//        refine with MTL5's generic DOUBLE-precision residual. Does an exact
//        accumulator inside the factorization still matter once IR refines
//        away its accumulation error? (residual/forward-error/iteration count
//        only -- no accumulation-length instrumentation here; the LU
//        factorization's own forward/backward/Schur accumulation lengths are
//        a separate concern this app no longer tracks.)
//     2. Residual-accumulator IR: factor A in posit (always plain), but form
//        the REFINEMENT residual itself at posit precision, with either plain
//        (round-every-add) or quire (exact, single-rounding) accumulation.
//        This is the "make the residual genuinely mixed precision" question --
//        see sw::mp_spice::mixed_refine_residual_accumulator. Accumulation-
//        length instrumentation (count/mean/median/max) is reported here only,
//        via the residual accumulator's own residual_stats counter.
//
// Requires Universal + the MTL5 accumulator seam (#122); build with
// -DMPSPICE_MIXED_PRECISION_KLU=ON (default ON).
//
// Usage: klu_quire_IR_study [matrix.mtx]
// No matrix -> an ill-conditioned Hilbert-like system where the effect is clear.

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/io/matrix_market.hpp>
#include <mtl/sparse/instrumentation/kernel_stats.hpp>

#include <universal/number/posit/posit.hpp>
#include <sw/mp_spice/quire_accumulator.hpp>
#include <sw/mp_spice/klu_study.hpp>

namespace {

using Dbl = mtl::mat::compressed2D<double>;

// Dense, diagonally-dominant (well-conditioned) matrix with deterministic
// mixed-sign off-diagonals -- same default system as klu_quire_study, so the
// two apps are directly comparable when run with no matrix argument.
Dbl dense_mixed(std::size_t n) {
    Dbl A(n, n);
    mtl::mat::inserter<Dbl> ins(A);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            ins[i][j] << (i == j ? static_cast<double>(n)
                                 : std::sin(static_cast<double>(i * 7 + j * 13)));  // in [-1,1]
    return A;
}

struct FactorComparison {
    std::string type;
    sw::mp_spice::solve_stats plain;
    sw::mp_spice::solve_stats quire;
};

struct ResidualComparison {
    std::string type;
    sw::mp_spice::solve_stats standard;
    sw::mp_spice::solve_stats quire;

    mtl::sparse::instrumentation::KernelStatistics::StatisticSummary standard_summary;
    mtl::sparse::instrumentation::KernelStatistics::StatisticSummary quire_summary;
};

} // namespace

int main(int argc, char** argv) {
    std::string mtx;
    if (argc > 1) mtx = argv[1];

    Dbl A;
    if (!mtx.empty()) { std::printf("Loading %s\n", mtx.c_str()); A = mtl::io::mm_read<double>(mtx); }
    else { std::printf("Well-conditioned dense mixed-sign(40) system (pass a .mtx to override).\n"); A = dense_mixed(40); }
    std::printf("Matrix: %zu x %zu, nnz = %zu\n\n",
                (size_t)A.num_rows(), (size_t)A.num_cols(), (size_t)A.nnz());

    using namespace sw::universal;

    std::vector<double> ones(A.num_rows(), 1.0);
    auto b = sw::mp_spice::rhs_from_ones(A);

    // --- 1. Factor-accumulator IR: plain vs quire INSIDE the factorization,
    //        refined with MTL5's generic double-precision residual. ---
    std::printf("Native KLU + double-residual iterative refinement (factor in posit):\n");
    std::printf("%-13s | %11s %11s %5s | %11s %11s %5s\n",
                "type", "plain res", "plain ferr", "it", "quire res", "quire ferr", "it");
    std::printf("%s\n", std::string(74, '-').c_str());
    auto ir_row = [&](const std::string& type, auto tag) -> FactorComparison {
        using P = decltype(tag);
        auto plain = sw::mp_spice::mixed_refine<P>(A, b, ones);
        auto quire = sw::mp_spice::mixed_refine<P, sw::mp_spice::quire_acc<P>>(A, b, ones);
        return {type, plain, quire};
    };
    const std::vector<FactorComparison> ir_rows = {
        ir_row("posit<16,2>", posit<16, 2>{}),
        ir_row("posit<32,2>", posit<32, 2>{})
    };
    for (const auto& row : ir_rows) {
        auto cell = [](const sw::mp_spice::solve_stats& s) {
            if (s.ok) std::printf(" %11.3e %11.3e %5d", s.residual, s.fwd_error, s.iters);
            else      std::printf(" %11s %11s %5s", "FAIL", "-", "-");
        };
        std::printf("%-13s |", row.type.c_str());
        cell(row.plain);
        std::printf(" |");
        cell(row.quire);
        std::printf("\n");
    }

    // --- 2. Residual-accumulator IR: factorization always plain; the residual
    //        r = b - A*x now runs its A*x products at the SAME low precision P
    //        the factorization uses, with plain (round-every-add) vs quire
    //        (exact, single-rounding) accumulation. ---
    std::printf("\nNative KLU IR residual accumulation (plain factorization):\n");
    std::printf("%-13s | %11s %11s %5s | %11s %11s %5s\n",
                "type", "std res", "std ferr", "it", "quire res", "quire ferr", "it");
    std::printf("%s\n", std::string(74, '-').c_str());

    auto residual_row = [&](const std::string& type, auto tag) -> ResidualComparison {
        using P = decltype(tag);

        residual_stats.reset();
        auto standard = sw::mp_spice::mixed_refine_residual_accumulator<P, P, P>(
            A, b, ones);
        auto standard_summary = residual_stats.summary();

        residual_stats.reset();
        auto quire = sw::mp_spice::mixed_refine_residual_accumulator<
            P, P, sw::mp_spice::quire_acc<P>>(A, b, ones);
        auto quire_summary = residual_stats.summary();

        return {type, standard, quire, standard_summary, quire_summary};
    };
    const std::vector<ResidualComparison> residual_rows = {
        residual_row("posit<16,2>", posit<16, 2>{}),
        residual_row("posit<32,2>", posit<32, 2>{})
    };
    for (const auto& row : residual_rows) {
        auto cell = [](const sw::mp_spice::solve_stats& s) {
            if (s.ok) std::printf(" %11.3e %11.3e %5d", s.residual, s.fwd_error, s.iters);
            else      std::printf(" %11s %11s %5s", "FAIL", "-", "-");
        };
        std::printf("%-13s |", row.type.c_str());
        cell(row.standard);
        std::printf(" |");
        cell(row.quire);
        std::printf("\n");

        auto print_summary = [](const char* name,
                                const mtl::sparse::instrumentation::KernelStatistics::StatisticSummary& s)
        {
            std::printf("%s\n", name);
            std::printf("Count:  %zu\n", s.count);
            std::printf("Mean:   %.3f\n", s.mean);
            std::printf("Median: %.3f\n", s.median);
            std::printf("Max:    %.3f\n\n", s.maximum);
        };

        print_summary("Standard residual accumulation", row.standard_summary);
        print_summary("Quire residual accumulation", row.quire_summary);
    }
    return 0;
}
