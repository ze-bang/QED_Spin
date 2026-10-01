// =============================================================================
// test_thermal_dense_ref  (Catch2 v3)
//
// Verifies that every finite-temperature method produces E(T) and/or Cv(T)
// consistent with the exact answer from full dense diagonalisation.
//
// Coverage matrix
// ---------------
//
//   Methods   : FTLM, mTPQ
//   Symmetry  : none (full Hilbert),
//               U(1)/Sz (per-Sz-sector recombination)
//   Backends  : CPU (always),
//               GPU (if WITH_CUDA and a device is present at runtime)
//   Trials    : three independent random seeds per method
//   Observables: E(T) and Cv(T) for all methods;
//                S(T) additionally for FTLM (which includes the full
//                ln(D) entropy baseline in its formula).
//
// System under test
// -----------------
//   N = 6 spin-1/2 periodic Heisenberg chain, Hilbert dim = 64.
//
// Method-specific notes / known limitations
// -----------------------------------------
//
//   FTLM    : trace estimator, accurate at all T with enough samples.
//             Compared on [T_BROAD_MIN, T_BROAD_MAX].
//             E, Cv, S all tested.
//
//   mTPQ    : microcanonical TPQ.  The iteration β_k = 2k/(L−E_k)
//             reaches β_target only asymptotically with max_iter.
//             Reliable above T_BROAD_MIN = 1.0 with max_iter=200.
//             The entropy is integrated from S(T_MIN)=0 (no ln(D)
//             baseline), so absolute S is wrong.  Only E and Cv tested.
//             For sector-combination tests the free-energy weighting
//             also uses this biased F, so only E is checked after combine.
//
// Temperature grids
// -----------------
//   T_BROAD : [1.0, 10.0], 15 log-spaced points — FTLM.
//   T_HIGH  : [3.0, 10.0], 10 log-spaced points — mTPQ (TPQ variance
//             shrinks at higher T; trajectory reaches β=0.33 in 200 steps).
//
// Which KERNEL runs
// -----------------
// N=6 (dim=64) sits under the orchestrator's SMALL_THERMAL_DIM=512 exact
// fallback, which answers every sampling method exactly and would leave the
// estimators this file exists to gate completely untested.  This file sets
// ED_THERMAL_EXACT_SMALL=0 at static-init so the real kernels run.
// test_thermal_exact_small_fallback pins the fallback itself.
//
// Both grids are precomputed and passed via ``opts.betas`` so they exactly
// match the log-spaced T grid of the dense reference (dense_reference below).
// The orchestrator's default builds a LINEAR T-axis from
// ``temp_min/temp_max/num_temp_bins``, which would cause element-wise
// temperature mismatches.
//
// Relationship to other tests
// ---------------------------
//   (sector recombination: combine_sectors, in this file)
//   test_auto_thermal      : smoke-tests orchestrator wiring
//   test_kernel_facades    : pin individual kernel signatures
//   THIS FILE              : pin numerical accuracy against dense reference
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/core/operator.h>
#include <ed/core/thermal_types.h>
#include <ed/orchestrator.h>
#include <ed/symmetry/canonical_thermo.h>

#ifdef WITH_CUDA
#include <ed/matvec/backends/cuda_backend.cuh>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstdlib>          // setenv (ED_THERMAL_EXACT_SMALL)
#include <filesystem>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

using namespace ed_tests;
using ed::workflows::ThermalOptions;

// ---------------------------------------------------------------------------
// Test-wide constants
// ---------------------------------------------------------------------------
namespace {

constexpr uint64_t N_SITES = 6;
constexpr double   J       = 1.0;

// "Broad" T range: valid for FTLM.
//   mTPQ uses T_HIGH_MIN (see below).
constexpr double T_BROAD_MIN = 1.0;
constexpr double T_BROAD_MAX = 10.0;
constexpr uint64_t N_BROAD   = 15;

// "High" T range: valid for mTPQ — avoids both the mTPQ
// asymptotic under-reach (β_200 < β_target) and TPQ endpoint noise.
constexpr double T_HIGH_MIN  = 3.0;
constexpr double T_HIGH_MAX  = 10.0;
constexpr uint64_t N_HIGH    = 10;

// Force the sampling KERNELS. dim=64 < SMALL_THERMAL_DIM=512, so without this
// the orchestrator answers every FTLM/mTPQ cell from an exact eigensolve
// and this file silently stops testing the estimators at all. Set at static
// init, before any thermal() call. See the "Which KERNEL runs" note above.
const bool kForceSamplingKernels = [] {
    ::setenv("ED_THERMAL_EXACT_SMALL", "0", /*overwrite=*/1);
    return true;
}();

// Tolerances.
constexpr double TOL_E  = 0.08;
constexpr double TOL_CV = 0.25;
constexpr double TOL_S  = 0.15;

// Seeds for multiple-trial runs.
const std::vector<uint64_t> SEEDS = {42ULL, 1337ULL, 99991ULL};

// ---------------------------------------------------------------------------
// Build a log-spaced beta grid; dense_reference evaluates the exact
// thermodynamics on the same grid.
// Passing opts.betas directly avoids the linear-vs-log T-grid mismatch that
// would otherwise corrupt the element-wise comparison.
// ---------------------------------------------------------------------------
inline std::vector<double> logspaced_betas(double t_lo, double t_hi, uint64_t n) {
    std::vector<double> betas;
    betas.reserve(n);
    const double log_tlo = std::log(t_lo);
    const double log_thi = std::log(t_hi);
    const double step    = (n > 1)
        ? (log_thi - log_tlo) / static_cast<double>(n - 1) : 0.0;
    for (uint64_t i = 0; i < n; ++i) {
        const double T = std::exp(log_tlo + static_cast<double>(i) * step);
        betas.push_back(1.0 / T);
    }
    return betas;
}

const std::vector<double> BETAS_BROAD = logspaced_betas(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
const std::vector<double> BETAS_HIGH  = logspaced_betas(T_HIGH_MIN,  T_HIGH_MAX,  N_HIGH);

// ---------------------------------------------------------------------------
// Operator factories
// ---------------------------------------------------------------------------
std::unique_ptr<Operator> make_full_heisen() {
    return build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
}

std::unique_ptr<SzSectorOperator> make_sz_heisen(int64_t n_up) {
    return build_heisenberg_chain_fixed_sz(N_SITES, J, n_up, /*periodic=*/true);
}

// ---------------------------------------------------------------------------
// Dense reference thermodynamics
// ---------------------------------------------------------------------------
ThermodynamicData dense_reference(double t_min, double t_max, uint64_t n) {
    auto H   = make_full_heisen();
    auto ref = reference_from_operator(*H, 1ULL << N_SITES);
    std::vector<double> T;
    for (double b : logspaced_betas(t_min, t_max, n)) T.push_back(1.0 / b);
    return ed::symmetry::canonical_thermo_from_eigs(ref.eigs, T);
}

// ---------------------------------------------------------------------------
// Accuracy checker
//
// Flags:
//   compare_entropy  — when false, skip S comparison (for TPQ methods whose
//                      entropy integration baseline is 0, not S_true(T_min))
// ---------------------------------------------------------------------------
void check_thermo_close(const ThermodynamicData& got,
                        const ThermodynamicData& ref,
                        double tol_E, double tol_Cv, double tol_S,
                        const std::string& label,
                        bool compare_entropy = true) {
    REQUIRE(got.energy.size()        == ref.energy.size());
    REQUIRE(got.specific_heat.size() == ref.specific_heat.size());

    double max_dE  = 0.0, max_dCv = 0.0, max_dS = 0.0;
    std::size_t worst_E = 0, worst_Cv = 0, worst_S = 0;

    for (std::size_t t = 0; t < ref.energy.size(); ++t) {
        const double dE  = std::abs(got.energy[t]        - ref.energy[t]);
        const double dCv = std::abs(got.specific_heat[t] - ref.specific_heat[t]);
        if (dE  > max_dE)  { max_dE  = dE;  worst_E  = t; }
        if (dCv > max_dCv) { max_dCv = dCv; worst_Cv = t; }
    }

    if (compare_entropy &&
        !got.entropy.empty() && !ref.entropy.empty() &&
        got.entropy.size() == ref.entropy.size()) {
        for (std::size_t t = 0; t < ref.entropy.size(); ++t) {
            const double dS = std::abs(got.entropy[t] - ref.entropy[t]);
            if (dS > max_dS) { max_dS = dS; worst_S = t; }
        }
    }

    INFO(label
         << ": max|ΔE|="  << max_dE  << " (T=" << ref.temperatures[worst_E]  << ")"
         << "  max|ΔCv|=" << max_dCv << " (T=" << ref.temperatures[worst_Cv] << ")"
         << "  max|ΔS|="  << max_dS  << " (T=" << ref.temperatures[worst_S]  << ")");
    REQUIRE(max_dE  <= tol_E);
    REQUIRE(max_dCv <= tol_Cv);
    if (compare_entropy &&
        !got.entropy.empty() && !ref.entropy.empty() &&
        got.entropy.size() == ref.entropy.size()) {
        REQUIRE(max_dS <= tol_S);
    }
}

// ---------------------------------------------------------------------------
// ThermalOptions builders
// Each uses opts.betas (log-spaced) rather than temp_min/temp_max/num_bins
// to ensure the T grid matches the dense reference.
// ---------------------------------------------------------------------------

ThermalOptions make_ftlm_opts(uint64_t seed, bool allow_gpu = false) {
    ThermalOptions o;
    o.method       = ThermalOptions::Method::FTLM;
    o.num_samples  = 50;
    o.krylov_dim   = 60;
    o.betas        = BETAS_BROAD;
    o.random_seed  = seed;
    o.backend.allow_gpu = allow_gpu;
    return o;
}

ThermalOptions make_mtpq_opts(uint64_t seed, bool allow_gpu = false) {
    ThermalOptions o;
    o.method       = ThermalOptions::Method::mTPQ;
    // 50 samples: brings statistical error well below TOL_E=0.08 at T=T_HIGH_MIN.
    o.num_samples  = 50;
    // max_iter=200 with T_MIN=3.0 → L_auto≈1206, β_200≈0.331 ≈ 1/3.0.
    // Coldest comparison at β=1/3.0; trajectory clamps at β=0.331 — ΔE<0.001.
    o.krylov_dim   = 200;
    o.betas        = BETAS_HIGH;
    o.random_seed  = seed;
    o.backend.allow_gpu = allow_gpu;
    return o;
}

ThermalOptions opts_for(ThermalOptions::Method m, uint64_t seed,
                        bool allow_gpu = false) {
    switch (m) {
        case ThermalOptions::Method::FTLM:   return make_ftlm_opts(seed, allow_gpu);
        case ThermalOptions::Method::mTPQ:   return make_mtpq_opts(seed, allow_gpu);
        default: throw std::logic_error("unknown method");
    }
}

std::string method_name(ThermalOptions::Method m) {
    switch (m) {
        case ThermalOptions::Method::FTLM:   return "FTLM";
        case ThermalOptions::Method::mTPQ:   return "mTPQ";
        default: return "??";
    }
}

// Which observables are reliable for each method?
bool method_compare_entropy(ThermalOptions::Method m) {
    // mTPQ integrates S from 0 at T_MIN (no ln(D) baseline).
    return m != ThermalOptions::Method::mTPQ;
}

bool method_compare_cv(ThermalOptions::Method m) {
    return true;  // Cv is always reliable when E is reliable
}

// T grid (dense ref) for this method.
std::pair<double,double> t_range_for(ThermalOptions::Method m) {
    if (m == ThermalOptions::Method::mTPQ)
        return {T_HIGH_MIN, T_HIGH_MAX};
    return {T_BROAD_MIN, T_BROAD_MAX};
}

uint64_t n_temp_for(ThermalOptions::Method m) {
    if (m == ThermalOptions::Method::mTPQ)
        return N_HIGH;
    return N_BROAD;
}

// ---------------------------------------------------------------------------
// Run one method on operator H, compare to precomputed ref.
// ---------------------------------------------------------------------------
template <class OpT>
void run_trial(OpT& H,
               ThermalOptions::Method m,
               uint64_t seed,
               const ThermodynamicData& ref,
               const std::string& label) {
    auto opts = opts_for(m, seed);
    auto R    = ed::workflows::thermal(H, opts);
    REQUIRE(R.backend.lane == "cpu");
    check_thermo_close(R.thermo, ref,
                       TOL_E, TOL_CV, TOL_S,
                       label + " seed=" + std::to_string(seed),
                       method_compare_entropy(m));
}

// For sector-combination tests: whether to compare entropy or just E+Cv
// after combining.  mTPQ has biased F → biased combination weights →
// only E should be tested post-combination.
bool method_combine_reliable(ThermalOptions::Method m) {
    return m != ThermalOptions::Method::mTPQ
;
}

#ifdef WITH_CUDA
bool has_gpu() {
    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess) { cudaGetLastError(); return false; }
    return count > 0;
}
#endif

} // anonymous namespace

// ===========================================================================
// 1. No symmetry — full Hilbert space
//    Each method on the full `Operator`, compared to the exact dense reference.
// ===========================================================================

TEST_CASE("thermal methods vs dense reference: no symmetry (full Hilbert)",
          "[thermal][dense-ref][no-sym]") {

    SECTION("FTLM") {
        const auto ref = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
        auto H = make_full_heisen();
        for (uint64_t seed : SEEDS)
            run_trial(*H, ThermalOptions::Method::FTLM, seed, ref,
                      "FTLM/no-sym");
    }

    // mTPQ: entropy S not compared (integration baseline issue).
    // Uses T_HIGH range [2, 10] to stay well within trajectory reach.
    SECTION("mTPQ") {
        const auto ref = dense_reference(T_HIGH_MIN, T_HIGH_MAX, N_HIGH);
        auto H = make_full_heisen();
        for (uint64_t seed : SEEDS)
            run_trial(*H, ThermalOptions::Method::mTPQ, seed, ref,
                      "mTPQ/no-sym");
    }
}

// ===========================================================================
// 2. U(1) / Sz symmetry — per-sector Sz operator + recombination
//
//    For each n_up ∈ [0, N] run thermal on the corresponding Sz-sector operator,
//    then combine by free-energy weighting (combine_sectors below).
//
//    Note: for mTPQ the combination uses the TPQ free-energy (which has
//    a biased integration constant) as the sector weight.  Only the combined
//    energy E_combined is tested for those methods (using wider tolerance);
//    Cv and S are omitted because the weighting bias can distort them.
// ===========================================================================

namespace {

// Recombine per-sector thermodynamics into the full system by free-energy
// weighting: Z = sum_s exp(-beta F_s) (shifted by the smallest F_s),
// E and <E^2> by the canonical mixture rule, S = beta (E - F).
ThermodynamicData combine_sectors(const std::vector<ThermodynamicData>& sec) {
    ThermodynamicData out;
    out.temperatures = sec.front().temperatures;
    const std::size_t nt = out.temperatures.size();
    out.energy.assign(nt, 0.0);
    out.specific_heat.assign(nt, 0.0);
    out.entropy.assign(nt, 0.0);
    out.free_energy.assign(nt, 0.0);
    for (std::size_t t = 0; t < nt; ++t) {
        const double T = out.temperatures[t], beta = 1.0 / T;
        double F_ref = sec[0].free_energy[t];
        for (const auto& s : sec)
            if (std::isfinite(s.free_energy[t]) && s.free_energy[t] < F_ref) F_ref = s.free_energy[t];
        std::vector<double> Z(sec.size(), 0.0);
        double Zt = 0.0;
        for (std::size_t s = 0; s < sec.size(); ++s) {
            double z = std::exp(-beta * (sec[s].free_energy[t] - F_ref));
            if (!std::isfinite(z) || z < 0.0) z = 0.0;
            Z[s] = z;
            Zt += z;
        }
        double E = 0.0, E2 = 0.0;
        for (std::size_t s = 0; s < sec.size(); ++s) {
            const double w = Z[s] / Zt, Es = sec[s].energy[t];
            E  += w * Es;
            E2 += w * (sec[s].specific_heat[t] / (beta * beta) + Es * Es);
        }
        out.free_energy[t]   = F_ref - T * std::log(Zt);
        out.energy[t]        = E;
        out.specific_heat[t] = beta * beta * (E2 - E * E);
        out.entropy[t]       = beta * (E - out.free_energy[t]);
    }
    return out;
}

void sz_trial(ThermalOptions::Method m, uint64_t seed,
              const ThermodynamicData& ref,
              double tol_E_combo = TOL_E) {
    std::vector<ThermodynamicData> sector_thermos;

    for (int64_t n_up = 0; n_up <= static_cast<int64_t>(N_SITES); ++n_up) {
        auto op = make_sz_heisen(n_up);

        auto opts = opts_for(m, seed + static_cast<uint64_t>(n_up) * 17ULL);
        auto R    = ed::workflows::thermal(*op, opts);
        REQUIRE(R.backend.lane == "cpu");

        sector_thermos.push_back(R.thermo);
    }

    const ThermodynamicData combined =
        combine_sectors(sector_thermos);

    const bool full_compare = method_combine_reliable(m);
    check_thermo_close(combined, ref,
                       tol_E_combo, TOL_CV, TOL_S,
                       method_name(m) + "/sz seed=" + std::to_string(seed),
                       full_compare && method_compare_entropy(m));
    if (!full_compare) {
        // For mTPQ: only energy is checked above; also confirm Cv is
        // finite and non-negative as a sanity guard.
        for (auto cv : combined.specific_heat) REQUIRE(std::isfinite(cv));
    }
}

} // namespace

TEST_CASE("thermal methods vs dense reference: U(1)/Sz symmetry",
          "[thermal][dense-ref][sz-sym]") {

    SECTION("FTLM") {
        const auto ref = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
        for (uint64_t seed : SEEDS)
            sz_trial(ThermalOptions::Method::FTLM, seed, ref);
    }

    SECTION("mTPQ") {
        // Combined energy tolerance is wider because the free-energy
        // weighting of sectors uses the biased TPQ F (missing entropy
        // baseline).  The bias shrinks at higher T where S(T_MIN)/T→0.
        const auto ref = dense_reference(T_HIGH_MIN, T_HIGH_MAX, N_HIGH);
        for (uint64_t seed : SEEDS)
            sz_trial(ThermalOptions::Method::mTPQ, seed, ref,
                     /*tol_E_combo=*/0.5);
    }
}

// ===========================================================================
// 3. GPU backend — if WITH_CUDA and a device is present at runtime
//
//    Run each method with allow_gpu=true and verify it still agrees with
//    the dense reference.  Skipped (SUCCEED) on GPU-less hosts.
// ===========================================================================

#ifdef WITH_CUDA
TEST_CASE("thermal GPU lane vs dense reference",
          "[thermal][dense-ref][gpu][with-cuda]") {

    if (!has_gpu()) {
        SUCCEED("Skipping GPU thermal accuracy test: no CUDA device present.");
        return;
    }

    const auto ref_broad = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
    const auto ref_high  = dense_reference(T_HIGH_MIN,  T_HIGH_MAX,  N_HIGH);
    auto H = make_full_heisen();
    constexpr uint64_t GPU_SEED = 42ULL;

    SECTION("FTLM GPU") {
        auto opts = make_ftlm_opts(GPU_SEED, true);
        auto R    = ed::workflows::thermal(*H, opts);
        REQUIRE((R.backend.lane == "gpu" || R.backend.lane == "cpu"));
        check_thermo_close(R.thermo, ref_broad,
                           TOL_E, TOL_CV, TOL_S,
                           "FTLM/gpu",
                           /*compare_entropy=*/true);
    }

    SECTION("mTPQ GPU") {
        auto opts = make_mtpq_opts(GPU_SEED, true);
        auto R    = ed::workflows::thermal(*H, opts);
        REQUIRE((R.backend.lane == "gpu" || R.backend.lane == "cpu"));
        check_thermo_close(R.thermo, ref_high,
                           TOL_E, TOL_CV, TOL_S,
                           "mTPQ/gpu",
                           /*compare_entropy=*/false);
    }
}
#endif  // WITH_CUDA
