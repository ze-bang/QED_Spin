// =============================================================================
// test_thermal_dense_ref  (Catch2 v3)
//
// Every sampling thermal method, through ed::sectors::thermal, against the
// exact answer of a dense diagonalisation.
//
//   Methods   : FTLM, mTPQ
//   Symmetry  : none (one 64-state block), and the Sz sectors (seven blocks,
//               combined by the verb itself)
//   Devices   : CPU; with WITH_CUDA and a device, device='gpu' on the block
//   Trials    : three seeds per method
//   Compared  : E(T) and C(T); S(T) for FTLM
//
// System: the N = 6 periodic Heisenberg chain (dim 64). Every block sits under
// thermal's dense crossover (dense_max_dim, default 512), which would answer
// exactly and leave the estimators untested, so every run sets dense_max_dim = 0.
// test_block_solve [thermal] pins the crossover itself.
//
//   FTLM : T in [1, 10], 15 log-spaced points.
//   mTPQ : T in [3, 10], 10 points, 200 steps per sample.
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/core/operator.h>
#include <ed/sectors/thermal.h>

#ifdef WITH_CUDA
#include <ed/core/select_backend.h>   // ed::have_cuda
#endif

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace ed_tests;
using ed::sectors::ThermalSpec;

namespace {

constexpr int    N_SITES = 6;
constexpr double J       = 1.0;

constexpr double TOL_E  = 0.08;
constexpr double TOL_CV = 0.25;
constexpr double TOL_S  = 0.15;

const std::vector<std::uint64_t> SEEDS = {42ULL, 1337ULL, 99991ULL};

std::vector<double> logspaced(double t_lo, double t_hi, int n) {
    std::vector<double> T;
    const double step = n > 1 ? (std::log(t_hi) - std::log(t_lo)) / (n - 1) : 0.0;
    for (int i = 0; i < n; ++i) T.push_back(std::exp(std::log(t_lo) + i * step));
    return T;
}

const std::vector<double> T_BROAD = logspaced(1.0, 10.0, 15);   // FTLM
const std::vector<double> T_HIGH  = logspaced(3.0, 10.0, 10);   // mTPQ

// Exact canonical E, C and S on the grid T, from the dense spectrum.
struct Exact { std::vector<double> E, C, S; };
Exact dense_reference(const std::vector<double>& T) {
    auto H = build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
    const auto eigs = reference_from_operator(*H, 1ULL << N_SITES).eigs;
    const double e0 = *std::min_element(eigs.begin(), eigs.end());
    Exact r;
    for (double t : T) {
        const double beta = 1.0 / t;
        double z = 0.0, e1 = 0.0, e2 = 0.0;
        for (double e : eigs) {
            const double w = std::exp(-beta * (e - e0));
            z += w; e1 += w * (e - e0); e2 += w * (e - e0) * (e - e0);
        }
        const double m1 = e1 / z, var = e2 / z - m1 * m1;
        r.E.push_back(e0 + m1);
        r.C.push_back(beta * beta * var);
        r.S.push_back(std::log(z) + beta * m1);
    }
    return r;
}

ed::sectors::Spec no_symmetry() {
    ed::sectors::Spec s;
    s.use_sz = false;
    s.spin_flip = 0;
    s.time_reversal = 0;
    return s;
}

ed::sectors::Spec sz_sectors() {
    ed::sectors::Spec s;
    s.spin_flip = 0;
    s.time_reversal = 0;
    return s;
}

ThermalSpec spec_for(ThermalSpec::Method m, std::uint64_t seed, ed::Device device = ed::Device::Cpu) {
    ThermalSpec t;
    t.method        = m;
    t.temperatures  = m == ThermalSpec::Method::mTPQ ? T_HIGH : T_BROAD;
    t.samples       = 50;
    t.krylov        = m == ThermalSpec::Method::mTPQ ? 200 : 60;
    t.dense_max_dim = 0;   // dim 64: the sampling kernels, not the dense crossover
    t.seed          = seed;
    t.device        = device;
    return t;
}

void check_close(const ed::sectors::ThermalCurves& got, const Exact& ref, double tol_E, bool compare_S,
                 const std::string& label) {
    REQUIRE(got.E.size() == ref.E.size());
    REQUIRE(got.C.size() == ref.C.size());
    double dE = 0.0, dC = 0.0, dS = 0.0;
    for (std::size_t i = 0; i < ref.E.size(); ++i) {
        dE = std::max(dE, std::abs(got.E[i] - ref.E[i]));
        dC = std::max(dC, std::abs(got.C[i] - ref.C[i]));
        dS = std::max(dS, std::abs(got.S[i] - ref.S[i]));
    }
    INFO(label << ": max|dE| = " << dE << ", max|dC| = " << dC << ", max|dS| = " << dS);
    REQUIRE(dE <= tol_E);
    REQUIRE(dC <= TOL_CV);
    if (compare_S) REQUIRE(dS <= TOL_S);
}

}  // namespace

TEST_CASE("thermal methods vs dense reference: no symmetry (one block)",
          "[thermal][dense-ref][no-sym]") {
    auto H = build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
    SECTION("FTLM") {
        const auto ref = dense_reference(T_BROAD);
        for (std::uint64_t seed : SEEDS) {
            const auto r = ed::sectors::thermal(*H, N_SITES, no_symmetry(), spec_for(ThermalSpec::Method::FTLM, seed));
            REQUIRE(r.placement.host_krylov == 1);
            check_close(r, ref, TOL_E, true, "FTLM/no-sym seed=" + std::to_string(seed));
        }
    }
    SECTION("mTPQ") {
        const auto ref = dense_reference(T_HIGH);
        for (std::uint64_t seed : SEEDS) {
            const auto r = ed::sectors::thermal(*H, N_SITES, no_symmetry(), spec_for(ThermalSpec::Method::mTPQ, seed));
            REQUIRE(r.placement.host_krylov == 1);
            check_close(r, ref, TOL_E, false, "mTPQ/no-sym seed=" + std::to_string(seed));
        }
    }
}

TEST_CASE("thermal methods vs dense reference: Sz sectors combined by the verb",
          "[thermal][dense-ref][sz-sym]") {
    auto H = build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
    SECTION("FTLM") {
        const auto ref = dense_reference(T_BROAD);
        for (std::uint64_t seed : SEEDS) {
            const auto r = ed::sectors::thermal(*H, N_SITES, sz_sectors(), spec_for(ThermalSpec::Method::FTLM, seed));
            REQUIRE(r.blocks == N_SITES + 1);
            check_close(r, ref, TOL_E, true, "FTLM/sz seed=" + std::to_string(seed));
        }
    }
    SECTION("mTPQ") {
        const auto ref = dense_reference(T_HIGH);
        for (std::uint64_t seed : SEEDS) {
            const auto r = ed::sectors::thermal(*H, N_SITES, sz_sectors(), spec_for(ThermalSpec::Method::mTPQ, seed));
            REQUIRE(r.blocks == N_SITES + 1);
            check_close(r, ref, /*tol_E=*/0.5, false, "mTPQ/sz seed=" + std::to_string(seed));
        }
    }
}

#ifdef WITH_CUDA
TEST_CASE("thermal on the device vs dense reference",
          "[thermal][dense-ref][gpu][with-cuda]") {
    if (!ed::have_cuda()) {
        SUCCEED("no CUDA device: skipped");
        return;
    }
    auto H = build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
    constexpr std::uint64_t GPU_SEED = 42ULL;
    SECTION("FTLM") {
        const auto r = ed::sectors::thermal(*H, N_SITES, no_symmetry(),
                                            spec_for(ThermalSpec::Method::FTLM, GPU_SEED, ed::Device::Gpu));
        REQUIRE(r.placement.device_krylov == 1);
        check_close(r, dense_reference(T_BROAD), TOL_E, true, "FTLM/gpu");
    }
    SECTION("mTPQ") {
        const auto r = ed::sectors::thermal(*H, N_SITES, no_symmetry(),
                                            spec_for(ThermalSpec::Method::mTPQ, GPU_SEED, ed::Device::Gpu));
        REQUIRE(r.placement.device_krylov == 1);
        check_close(r, dense_reference(T_HIGH), TOL_E, false, "mTPQ/gpu");
    }
}
#endif  // WITH_CUDA
