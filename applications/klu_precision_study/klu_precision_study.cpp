// mp-spice -- comprehensive native-KLU precision study (mp-spice #1 / #3 / #5)
//
// One report comparing, for a circuit-simulation matrix, the accuracy and
// convergence of MTL5's native KLU across number systems and solver strategies:
//
//   number systems:  double, float, bfloat16 (cfloat<16,8>), half (cfloat<16,5>),
//                     posit<16,2>, posit<32,2>
//   strategies:      direct        -- factor and solve in the type
//                    IR            -- factor in the type, refine with a double residual
//                    quire (posit) -- exact fused-dot-product accumulator in the factor
//
// Reports residual ||Ax-b||inf and forward error ||x-1||inf (exact x = 1), plus
// the IR iteration count. Two tables:
//   1. precision sweep (all types):     direct vs IR
//   2. quire comparison (posits only):  plain vs quire, under direct and IR
//
// Requires Universal + the MTL5 accumulator seam; build with
// -DMPSPICE_MIXED_PRECISION_KLU=ON (default ON). Usage:
//   klu_precision_study [matrix.mtx] [--csv out.csv]

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/io/matrix_market.hpp>

#include <universal/number/cfloat/cfloat.hpp>
#include <universal/number/posit/posit.hpp>
#include <sw/mp_spice/klu_study.hpp>
#include <sw/mp_spice/quire_accumulator.hpp>

namespace {

using sw::mp_spice::DSparse;
using sw::mp_spice::solve_stats;
using bfloat16 = sw::universal::cfloat<16, 8, std::uint16_t, true, false, false>;
using half     = sw::universal::cfloat<16, 5, std::uint16_t, true, false, false>;

DSparse make_synthetic_circuit() {
    DSparse A(6, 6);
    mtl::mat::inserter<DSparse> ins(A);
    ins[0][0] << 3.0;  ins[0][1] << -1.0;  ins[0][5] << -2.0;
    ins[1][0] << -1.0; ins[1][1] << 3.0;   ins[1][2] << -1.0;
    ins[2][1] << -1.0; ins[2][2] << 3.0;
    ins[3][3] << 4.0;  ins[3][4] << -1.0;
    ins[4][3] << -1.0; ins[4][4] << 4.0;   ins[4][5] << -1.0;
    ins[5][4] << -1.0; ins[5][5] << 4.0;
    return A;
}

void cell(const solve_stats& s, bool with_iters) {
    if (!s.ok) { std::printf(" %11s %11s", "FAIL", "-"); if (with_iters) std::printf(" %5s", "-"); return; }
    std::printf(" %11.3e %11.3e", s.residual, s.fwd_error);
    if (with_iters) std::printf(" %5d", s.iters);
}

} // namespace

int main(int argc, char** argv) {
    std::string mtx, csv;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) csv = argv[++i]; else mtx = a;
    }

    DSparse A;
    if (!mtx.empty()) { std::printf("Loading %s\n", mtx.c_str()); A = mtl::io::mm_read<double>(mtx); }
    else { std::printf("Synthetic block-triangular circuit matrix (pass a .mtx for a real one).\n");
           A = make_synthetic_circuit(); }
    std::printf("Matrix: %zu x %zu, nnz = %zu, exact solution x = 1\n\n",
                (size_t)A.num_rows(), (size_t)A.num_cols(), (size_t)A.nnz());

    using namespace sw::universal;
    std::vector<double> ones(A.num_rows(), 1.0);
    auto b = sw::mp_spice::rhs_from_ones(A);

    // ---- Table 1: precision sweep, direct vs iterative refinement ----
    std::printf("Table 1 -- precision sweep: direct solve vs double-residual IR\n");
    std::printf("%-13s | %11s %11s | %11s %11s %5s\n",
                "type", "dir res", "dir ferr", "IR res", "IR ferr", "it");
    std::printf("%s\n", std::string(68, '-').c_str());
    struct T1 { std::string t; solve_stats dir, ir; };
    std::vector<T1> t1;
    auto row1 = [&](const std::string& t, auto tag) {
        using P = decltype(tag);
        t1.push_back({t, sw::mp_spice::direct_solve<P>(A, b, ones),
                         sw::mp_spice::mixed_refine<P>(A, b, ones)});
        std::printf("%-13s |", t.c_str()); cell(t1.back().dir, false);
        std::printf(" |"); cell(t1.back().ir, true); std::printf("\n");
    };
    row1("double",      double{});
    row1("float",       float{});
    row1("bfloat16",    bfloat16{});
    row1("half",        half{});
    row1("posit<16,2>", posit<16, 2>{});
    row1("posit<32,2>", posit<32, 2>{});

    // ---- Table 2: quire (exact FDP) comparison, posits only ----
    std::printf("\nTable 2 -- quire (exact fused-dot-product) vs plain, posit only\n");
    std::printf("%-13s | %-25s | %-31s\n", "", "direct (plain | quire)", "IR (plain | quire)");
    std::printf("%-13s | %11s %11s | %11s %11s %5s\n",
                "type", "plain ferr", "quire ferr", "plain ferr", "quire ferr", "");
    std::printf("%s\n", std::string(74, '-').c_str());
    struct T2 { std::string t; solve_stats dp, dq, ip, iq; };
    std::vector<T2> t2;
    auto row2 = [&](const std::string& t, auto tag) {
        using P = decltype(tag);
        T2 r{t,
             sw::mp_spice::direct_solve<P>(A, b, ones),
             sw::mp_spice::direct_solve<P, sw::mp_spice::quire_acc<P>>(A, b, ones),
             sw::mp_spice::mixed_refine<P>(A, b, ones),
             sw::mp_spice::mixed_refine<P, sw::mp_spice::quire_acc<P>>(A, b, ones)};
        std::printf("%-13s | %11.3e %11.3e | %11.3e %11.3e (it %d/%d)\n", t.c_str(),
                    r.dp.fwd_error, r.dq.fwd_error, r.ip.fwd_error, r.iq.fwd_error, r.ip.iters, r.iq.iters);
        t2.push_back(r);
    };
    row2("posit<16,2>", posit<16, 2>{});
    row2("posit<32,2>", posit<32, 2>{});

    // ---- Table 3: unscaled vs scaled IR (extended-precision residual) ----
    // Scaled IR normalizes each residual to O(1) before casting to the type and
    // carries the correction magnitude in double -- rescuing narrow-exponent
    // types (half) whose unscaled IR underflows.
    std::printf("\nTable 3 -- iterative refinement: unscaled vs scaled (extended-precision residual)\n");
    std::printf("%-13s | %11s %5s | %11s %5s\n",
                "type", "unscaled fe", "it", "scaled fe", "it");
    std::printf("%s\n", std::string(54, '-').c_str());
    struct T3 { std::string t; solve_stats u, sc; };
    std::vector<T3> t3;
    auto row3 = [&](const std::string& t, auto tag) {
        using P = decltype(tag);
        T3 r{t, sw::mp_spice::mixed_refine<P>(A, b, ones),
                sw::mp_spice::mixed_refine_scaled<P>(A, b, ones)};
        auto fe = [](const solve_stats& s) {
            if (s.ok) std::printf(" %11.3e %5d", s.fwd_error, s.iters);
            else      std::printf(" %11s %5s", "FAIL", "-");
        };
        std::printf("%-13s |", t.c_str()); fe(r.u); std::printf(" |"); fe(r.sc); std::printf("\n");
        t3.push_back(r);
    };
    row3("float",       float{});
    row3("bfloat16",    bfloat16{});
    row3("half",        half{});
    row3("posit<16,2>", posit<16, 2>{});
    row3("posit<32,2>", posit<32, 2>{});

    if (!csv.empty()) {
        std::ofstream o(csv);
        o << "table,type,a_residual,a_fwd_error,a_iters,b_residual,b_fwd_error,b_iters\n";
        for (const auto& r : t1)
            o << "sweep," << r.t << ',' << r.dir.residual << ',' << r.dir.fwd_error << ",0,"
              << r.ir.residual << ',' << r.ir.fwd_error << ',' << r.ir.iters << '\n';
        for (const auto& r : t2) {
            o << "quire_direct," << r.t << ',' << r.dp.residual << ',' << r.dp.fwd_error << ",0,"
              << r.dq.residual << ',' << r.dq.fwd_error << ",0\n";
            o << "quire_IR," << r.t << ',' << r.ip.residual << ',' << r.ip.fwd_error << ',' << r.ip.iters << ','
              << r.iq.residual << ',' << r.iq.fwd_error << ',' << r.iq.iters << '\n';
        }
        for (const auto& r : t3)
            o << "scaled_ir," << r.t << ',' << r.u.residual << ',' << r.u.fwd_error << ',' << r.u.iters << ','
              << r.sc.residual << ',' << r.sc.fwd_error << ',' << r.sc.iters << '\n';
        std::printf("\nCSV: %s\n", csv.c_str());
    }
    return 0;
}
