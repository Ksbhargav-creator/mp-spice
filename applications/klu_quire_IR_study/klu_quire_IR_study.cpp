// mp-spice -- native KLU + iterative refinement quire study (mp-spice #3/#5)
//
// Companion to klu_quire_study, split out because these two questions use a
// different solve path and shouldn't share a binary:
//
//   klu_quire_study     -- one-shot DIRECT solve, plain vs quire accumulator
//                           INSIDE the factorization (MTL5 sparse_lu, no BTF).
//   klu_quire_IR_study   -- native KLU (full BTF) + iterative refinement:
//
//     Factor-accumulator IR: factor A in posit (plain vs quire), then refine
//     with MTL5's generic DOUBLE-precision residual. Does an exact accumulator
//     inside the factorization still matter once IR refines away its
//     accumulation error? Routed through
//     sw::mp_spice::mixed_refine_with_histogram (mathematically identical to
//     mixed_refine, just instrumented) so each run also reports a
//     product-magnitude histogram: the distribution (by order of magnitude)
//     of every individual term `a_ij * x_j` formed while assembling the
//     residual, looking for the two-cluster ("bimodal") signature of
//     catastrophic cancellation. The LU factorization's own
//     forward/backward/Schur accumulation lengths are a separate concern this
//     app no longer tracks.
//
//     Working-precision residual (John's hypothesis, direct test): NO
//     low-precision factorization at all -- factor once at Working precision
//     (a posit stand-in for "64-bit", plain, never quire -- quire in the
//     solver is a separate, already-answered question) and form the residual
//     at that SAME Working precision, varying only whether the summation is
//     plain or quire. Operands are never downcast further than Working, so
//     this tests summation-order error in isolation, matching John's literal
//     description ("a direct solver with 64-bit precision ... if you use the
//     quire to compute the residual, that residual is computed to infinite
//     precision until you round each entry of b-Ax to the working data
//     type"). Routed through sw::mp_spice::working_precision_refine_with_histogram.
//
//
// Requires Universal + the MTL5 accumulator seam (#122); build with
// -DMPSPICE_MIXED_PRECISION_KLU=ON (default ON).
//
// Usage: klu_quire_IR_study [matrix.mtx]
// No matrix -> an ill-conditioned Hilbert-like system where the effect is clear.

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/io/matrix_market.hpp>

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
    sw::mp_spice::ProductMagnitudeStats plain_hist;
    sw::mp_spice::ProductMagnitudeStats quire_hist;
};

// ASCII bar chart of the product-magnitude distribution: one bar per decade
// of |a_ij * x_j|.
void print_magnitude_histogram(const std::string& label,
                               const sw::mp_spice::ProductMagnitudeStats& stats) {
    std::printf("\nProduct magnitude distribution (%s):\n", label.c_str());
    if (stats.total() == 0) {
        std::printf("  (no products recorded -- solve may have failed)\n");
        return;
    }
    std::printf("  %zu products recorded, %zu exact zeros\n", stats.total(), stats.zero_count());
    std::size_t max_count = 0;
    for (const auto& bucket : stats.buckets()) max_count = std::max(max_count, bucket.second);
    for (const auto& bucket : stats.buckets()) {
        int decade = bucket.first;
        std::size_t count = bucket.second;
        int bar_len = (max_count > 0)
            ? static_cast<int>(50.0 * static_cast<double>(count) / static_cast<double>(max_count))
            : 0;
        std::printf("  1e%-4d [%7zu] %s\n", decade, count, std::string(bar_len, '#').c_str());
    }
}

// Bare filename, extension stripped -- "dense_mixed" if no matrix was loaded.
// Matches the "Matrix" column convention already used by csv/lu_accumulation.csv
// etc. (from the mixed-precision-klu-study.md write-up).
std::string matrix_name_from_path(const std::string& mtx) {
    if (mtx.empty()) return "dense_mixed";
    return std::filesystem::path(mtx).stem().string();
}

// Appends the product-magnitude histogram (all rows, all refinement
// iterations, pooled to csv/product_magnitude_histogram.csv,
// following the existing project convention (Matrix,n,Type,... header). One row per
// non-empty decade bucket; ZeroCount/Total repeated on every row of that
// group so the file stays fully rectangular for pandas.
void write_histogram_csv(std::ofstream& out,
                         const std::string& matrix_name,
                         std::size_t n,
                         const std::string& experiment,
                         const std::string& type,
                         const std::string& variant,
                         const sw::mp_spice::ProductMagnitudeStats& stats) {
    for (const auto& bucket : stats.buckets()) {
        out << matrix_name << ',' << n << ',' << experiment << ",\"" << type << "\","
            << variant << ',' << bucket.first << ',' << bucket.second << ','
            << stats.zero_count() << ',' << stats.total() << '\n';
    }
}

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

        sw::mp_spice::product_magnitude_stats.reset();
        auto plain = sw::mp_spice::mixed_refine_with_histogram<P>(A, b, ones);
        auto plain_hist = sw::mp_spice::product_magnitude_stats;

        sw::mp_spice::product_magnitude_stats.reset();
        auto quire = sw::mp_spice::mixed_refine_with_histogram<P, sw::mp_spice::quire_acc<P>>(A, b, ones);
        auto quire_hist = sw::mp_spice::product_magnitude_stats;

        return {type, plain, quire, plain_hist, quire_hist};
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
    for (const auto& row : ir_rows) {
        print_magnitude_histogram(row.type + " plain factorization", row.plain_hist);
        print_magnitude_histogram(row.type + " quire factorization", row.quire_hist);
    }

    // --- 2. Working-precision residual: plain vs quire summation.
    //        Factorization is ALWAYS plain, at the
    //        same working precision as the residual
    std::printf("\nWorking-precision residual: plain vs quire summation:\n");
    std::printf("%-13s | %11s %11s %5s | %11s %11s %5s\n",
                "type", "plain res", "plain ferr", "it", "quire res", "quire ferr", "it");
    std::printf("%s\n", std::string(74, '-').c_str());
    auto wp_row = [&](const std::string& type, auto tag) -> FactorComparison {
        using Working = decltype(tag);

        sw::mp_spice::product_magnitude_stats.reset();
        auto plain = sw::mp_spice::working_precision_refine_with_histogram<Working>(A, b, ones);
        auto plain_hist = sw::mp_spice::product_magnitude_stats;

        sw::mp_spice::product_magnitude_stats.reset();
        auto quire = sw::mp_spice::working_precision_refine_with_histogram<
            Working, sw::mp_spice::quire_acc<Working>>(A, b, ones);
        auto quire_hist = sw::mp_spice::product_magnitude_stats;

        return {type, plain, quire, plain_hist, quire_hist};
    };
    const std::vector<FactorComparison> wp_rows = {
        wp_row("posit<32,2>", posit<32, 2>{}),
        wp_row("posit<64,3>", posit<64, 3>{})
    };
    for (const auto& row : wp_rows) {
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
    for (const auto& row : wp_rows) {
        print_magnitude_histogram(row.type + " plain residual", row.plain_hist);
        print_magnitude_histogram(row.type + " quire residual", row.quire_hist);
    }

    // CSV export (csv/product_magnitude_histogram.csv, matching the existing
    // project convention -- see csv/lu_accumulation.csv etc.).
    const std::string matrix_name = matrix_name_from_path(mtx);
    std::filesystem::create_directories("csv");
    const std::string csv_path = "csv/product_magnitude_histogram.csv";
    const bool csv_exists = std::filesystem::exists(csv_path);
    std::ofstream csv_out(csv_path, std::ios::app);
    if (!csv_out) {
        std::fprintf(stderr, "warning: could not open %s for writing\n", csv_path.c_str());
    } else {
        if (!csv_exists)
            csv_out << "Matrix,n,Experiment,Type,Variant,Decade,Count,ZeroCount,Total\n";
        for (const auto& row : ir_rows) {
            write_histogram_csv(csv_out, matrix_name, A.num_rows(), "FactorAccumulatorIR",
                                row.type, "Plain", row.plain_hist);
            write_histogram_csv(csv_out, matrix_name, A.num_rows(), "FactorAccumulatorIR",
                                row.type, "Quire", row.quire_hist);
        }
        for (const auto& row : wp_rows) {
            write_histogram_csv(csv_out, matrix_name, A.num_rows(), "WorkingPrecisionResidualIR",
                                row.type, "Plain", row.plain_hist);
            write_histogram_csv(csv_out, matrix_name, A.num_rows(), "WorkingPrecisionResidualIR",
                                row.type, "Quire", row.quire_hist);
        }
        std::printf("\nWrote %s\n", csv_path.c_str());
    }
    return 0;
}
