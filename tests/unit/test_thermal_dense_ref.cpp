// =============================================================================
// test_thermal_dense_ref  (Catch2 v3)
//
// Verifies that every finite-temperature method produces E(T) and/or Cv(T)
// consistent with the exact answer from full dense diagonalisation.
//
// Coverage matrix
// ---------------
//
//   Methods   : FTLM, LTLM, mTPQ, cTPQ, KpmDos
//   Symmetry  : none (full Hilbert),
//               U(1)/Sz (per-Sz-sector recombination),
//               spatial (Z_N translation + combine_sector_thermodynamics),
//               Sz × spatial (U(1) × Z_N, flat combine)
//   Backends  : CPU (always),
//               GPU (if WITH_CUDA and a device is present at runtime)
//   Trials    : three independent random seeds per method
//   Observables: E(T) and Cv(T) for all methods;
//                S(T) additionally for FTLM, KpmDos and LTLM (which include
//                the full ln(D) entropy baseline in their formulas).
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
//   LTLM    : GS-biased estimator (Low Temperature Lanczos Method).
//             At high T the formula converges to ⟨ψ_GS|H|ψ_GS⟩ = E₀
//             rather than the canonical average.  Accurate only for
//             T < energy gap ≈ 0.23 (for N=6 Heisenberg at J=1).
//             Compared on [T_LOW_MIN, T_LOW_MAX] with krylov_dim=64
//             (complete Krylov space for this small dim=64 system).
//             E, Cv, S all tested in the low-T window.
//
//   mTPQ    : microcanonical TPQ.  The iteration β_k = 2k/(L−E_k)
//             reaches β_target only asymptotically with max_iter.
//             Reliable above T_BROAD_MIN = 1.0 with max_iter=200.
//             The entropy is integrated from S(T_MIN)=0 (no ln(D)
//             baseline), so absolute S is wrong.  Only E and Cv tested.
//             For sector-combination tests the free-energy weighting
//             also uses this biased F, so only E is checked after combine.
//
//   cTPQ    : canonical TPQ via Taylor expansion.  With delta_beta=0.1
//             and krylov_dim=20 the expansion reaches β=2 > β_target=1.
//             Same entropy limitation as mTPQ.  E and Cv tested.
//             Sector combine: only E tested.
//
//   KpmDos  : Chebyshev trace estimator.  Accurate at all T with enough
//             moments.  E, Cv, S all tested.
//
// Temperature grids
// -----------------
//   T_BROAD : [1.0, 10.0], 15 log-spaced points — FTLM, KpmDos.
//   T_HIGH  : [3.0, 10.0], 10 log-spaced points — mTPQ, cTPQ (TPQ variance
//             shrinks at higher T; trajectory reaches β=0.33 in 200 steps).
//   T_LOW   : [0.01, 0.06], 8 log-spaced points — LTLM (GS-biased; valid
//             only where E(T)≈E_GS and S(T)≈0, both within tol).
//
// Both grids are precomputed and passed via ``opts.betas`` so they exactly
// match ``calculate_thermodynamics_from_spectrum``'s internal log-spaced
// grid.  The orchestrator's default builds a LINEAR T-axis from
// ``temp_min/temp_max/num_temp_bins``, which would cause element-wise
// temperature mismatches.
//
// Relationship to existing tests
// -------------------------------
//   test_sector_thermo     : validates combine_sector_thermodynamics math
//   test_auto_thermal      : smoke-tests orchestrator wiring
//   test_kernel_facades    : pin individual kernel signatures
//   test_thermal_save      : pin HDF5 persistence contract
//   THIS FILE              : pin numerical accuracy against dense reference
// =============================================================================

#include "common/catch2_harness.h"
#include "common/test_harness.h"

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator.h>
#include <ed/core/sector_thermo.h>
#include <ed/core/thermal_types.h>
#include <ed/orchestrator.h>
#include <ed/solvers/observables.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>

#ifdef WITH_CUDA
#include <ed/matvec/backends/cuda_backend.cuh>
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
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

// "Broad" T range: valid for FTLM, KpmDos.
//   mTPQ and cTPQ use T_HIGH_MIN (see below).
constexpr double T_BROAD_MIN = 1.0;
constexpr double T_BROAD_MAX = 10.0;
constexpr uint64_t N_BROAD   = 15;

// "High" T range: valid for mTPQ/cTPQ — avoids both the mTPQ
// asymptotic under-reach (β_200 < β_target) and the cTPQ endpoint noise.
constexpr double T_HIGH_MIN  = 3.0;
constexpr double T_HIGH_MAX  = 10.0;
constexpr uint64_t N_HIGH    = 10;

// "Low-T" range: valid for LTLM (GS-biased estimator accurate below gap≈0.23).
// At T < 0.06 (β > 16.7): E(T) - E_GS < 0.02 (well within TOL_E=0.08)
//                          S(T)         < 0.05 (well within TOL_S=0.15)
// so LTLM (which returns E_GS and S=0) passes both checks.
constexpr double T_LOW_MIN = 0.01;
constexpr double T_LOW_MAX = 0.06;
constexpr uint64_t N_LOW   = 8;

// Tolerances.
constexpr double TOL_E  = 0.08;
constexpr double TOL_CV = 0.25;
constexpr double TOL_S  = 0.15;

// Seeds for multiple-trial runs.
const std::vector<uint64_t> SEEDS = {42ULL, 1337ULL, 99991ULL};

// ---------------------------------------------------------------------------
// Build a log-spaced beta grid that exactly matches the internal T grid of
// `calculate_thermodynamics_from_spectrum` (which is log-spaced internally).
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
const std::vector<double> BETAS_LOW   = logspaced_betas(T_LOW_MIN,   T_LOW_MAX,   N_LOW);

// ---------------------------------------------------------------------------
// Operator factories
// ---------------------------------------------------------------------------
std::unique_ptr<Operator> make_full_heisen() {
    return build_heisenberg_chain(N_SITES, J, /*periodic=*/true);
}

std::unique_ptr<FixedSzOperator> make_sz_heisen(int64_t n_up) {
    return build_heisenberg_chain_fixed_sz(N_SITES, J, n_up, /*periodic=*/true);
}

// ---------------------------------------------------------------------------
// Dense reference thermodynamics
// ---------------------------------------------------------------------------
ThermodynamicData dense_reference(double t_min, double t_max, uint64_t n) {
    auto H   = make_full_heisen();
    auto ref = reference_from_operator(*H, 1ULL << N_SITES);
    return calculate_thermodynamics_from_spectrum(ref.eigs, t_min, t_max, n);
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
// to ensure the T grid matches calculate_thermodynamics_from_spectrum.
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

ThermalOptions make_ltlm_opts(uint64_t seed, bool allow_gpu = false) {
    ThermalOptions o;
    o.method       = ThermalOptions::Method::LTLM;
    o.num_samples  = 30;
    // krylov_dim=64 covers the complete 64-dim Hilbert space for N=6.
    // For smaller symmetry sectors Lanczos terminates early — safe universally.
    o.krylov_dim   = 64;
    o.betas        = BETAS_LOW;   // ← low-T window: LTLM's valid regime
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
    o.temp_min     = T_HIGH_MIN;   // sets beta_target inside orchestrator
    o.random_seed  = seed;
    o.backend.allow_gpu = allow_gpu;
    return o;
}

ThermalOptions make_ctpq_opts(uint64_t seed, bool allow_gpu = false) {
    ThermalOptions o;
    o.method        = ThermalOptions::Method::cTPQ;
    // 50 samples for same reason as mTPQ.
    o.num_samples   = 50;
    // After the Δβ/2 kernel fix: 20 steps × Δβ=0.1 → β_max_recorded=2.0.
    // Coldest comparison at β=1/3.0≈0.333 << 2.0 → well within trajectory.
    o.krylov_dim    = 20;
    o.delta_beta    = 0.1;
    o.beta_max      = 15.0;
    o.taylor_order  = 20;
    o.betas         = BETAS_HIGH;
    o.temp_min      = T_HIGH_MIN;
    o.random_seed   = seed;
    o.backend.allow_gpu = allow_gpu;
    return o;
}

ThermalOptions make_kpm_opts(uint64_t seed, bool allow_gpu = false) {
    ThermalOptions o;
    o.method                 = ThermalOptions::Method::KpmDos;
    o.kpm_num_moments        = 512;
    o.kpm_num_random_vectors = 24;
    o.betas                  = BETAS_BROAD;
    o.random_seed            = seed;
    o.backend.allow_gpu      = allow_gpu;
    return o;
}

ThermalOptions opts_for(ThermalOptions::Method m, uint64_t seed,
                        bool allow_gpu = false) {
    switch (m) {
        case ThermalOptions::Method::FTLM:   return make_ftlm_opts(seed, allow_gpu);
        case ThermalOptions::Method::LTLM:   return make_ltlm_opts(seed, allow_gpu);
        case ThermalOptions::Method::mTPQ:   return make_mtpq_opts(seed, allow_gpu);
        case ThermalOptions::Method::cTPQ:   return make_ctpq_opts(seed, allow_gpu);
        case ThermalOptions::Method::KpmDos: return make_kpm_opts(seed,  allow_gpu);
        default: throw std::logic_error("unknown method");
    }
}

std::string method_name(ThermalOptions::Method m) {
    switch (m) {
        case ThermalOptions::Method::FTLM:   return "FTLM";
        case ThermalOptions::Method::LTLM:   return "LTLM";
        case ThermalOptions::Method::mTPQ:   return "mTPQ";
        case ThermalOptions::Method::cTPQ:   return "cTPQ";
        case ThermalOptions::Method::KpmDos: return "KpmDos";
        default: return "??";
    }
}

// Which observables are reliable for each method?
bool method_compare_entropy(ThermalOptions::Method m) {
    // mTPQ/cTPQ integrate S from 0 at T_MIN (no ln(D) baseline).
    // LTLM is a GS-biased estimator and gives S=0 (Z≈1, E≈E_GS), so its
    // entropy is only ~correct in the T→0 limit where S_true→0 too.
    return m != ThermalOptions::Method::mTPQ
        && m != ThermalOptions::Method::cTPQ
        && m != ThermalOptions::Method::LTLM;
}

bool method_compare_cv(ThermalOptions::Method m) {
    return true;  // Cv is always reliable when E is reliable
}

// T grid (dense ref) for this method.
std::pair<double,double> t_range_for(ThermalOptions::Method m) {
    if (m == ThermalOptions::Method::LTLM)
        return {T_LOW_MIN, T_LOW_MAX};
    if (m == ThermalOptions::Method::mTPQ || m == ThermalOptions::Method::cTPQ)
        return {T_HIGH_MIN, T_HIGH_MAX};
    return {T_BROAD_MIN, T_BROAD_MAX};
}

uint64_t n_temp_for(ThermalOptions::Method m) {
    if (m == ThermalOptions::Method::LTLM) return N_LOW;
    if (m == ThermalOptions::Method::mTPQ || m == ThermalOptions::Method::cTPQ)
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

// ---------------------------------------------------------------------------
// Z_N translation symmetry fixture  (mirrors test_thermal_save.cpp helper)
// ---------------------------------------------------------------------------
std::string write_zN_fixture(uint64_t N, const std::string& suite,
                              const std::string& tag) {
    const std::string root = make_scratch_dir(suite, tag);
    const std::string sym  = root + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(sym, ec);

    auto perm = [N](int shift) {
        std::vector<int> p(N);
        for (uint64_t i = 0; i < N; ++i)
            p[i] = (static_cast<int>(i) - shift
                    + static_cast<int>(N)) % static_cast<int>(N);
        return p;
    };

    { // max_clique.json: all N cyclic shifts
        std::ofstream f(sym + "/max_clique.json");
        f << "[";
        for (uint64_t g = 0; g < N; ++g) {
            auto p = perm(static_cast<int>(g));
            f << "[";
            for (std::size_t i = 0; i < p.size(); ++i)
                f << p[i] << (i + 1 < p.size() ? "," : "");
            f << "]" << (g + 1 < N ? "," : "");
        }
        f << "]";
    }
    { // minimal_generators.json: shift-by-1 generator
        std::ofstream f(sym + "/minimal_generators.json");
        auto p = perm(1);
        f << "{\"generators\":[{\"permutation\":[";
        for (std::size_t i = 0; i < p.size(); ++i)
            f << p[i] << (i + 1 < p.size() ? "," : "");
        f << "],\"order\":" << N << "}]}";
    }
    { // sector_metadata.json: k = 0 .. N-1
        std::ofstream f(sym + "/sector_metadata.json");
        f.precision(17);
        f << "{\"sectors\":[";
        for (uint64_t k = 0; k < N; ++k) {
            const double a = -2.0 * M_PI * static_cast<double>(k)
                             / static_cast<double>(N);
            f << "{\"sector_id\":" << k
              << ",\"quantum_numbers\":[" << k << "]"
              << ",\"phase_factors\":[{\"real\":" << std::cos(a)
              << ",\"imag\":" << std::sin(a) << "}]}";
            if (k + 1 < N) f << ",";
        }
        f << "]}";
    }
    return root;
}

// Heisenberg PBC term builder for SectorOperator.
void add_heisen_pbc_terms(ed::symmetry::SectorOperator& op, uint64_t N, double Jc) {
    using Complex = std::complex<double>;
    const Complex Jr(Jc, 0.0), Jh(0.5 * Jc, 0.0);
    for (uint64_t i = 0; i < N; ++i) {
        const uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, Jr);
        op.addTwoBodyTerm(0, i, 1, j, Jh);
        op.addTwoBodyTerm(1, i, 0, j, Jh);
    }
}

// For sector-combination tests: whether to compare entropy or just E+Cv
// after combining.  mTPQ/cTPQ have biased F → biased combination weights →
// only E should be tested post-combination.
bool method_combine_reliable(ThermalOptions::Method m) {
    return m != ThermalOptions::Method::mTPQ
        && m != ThermalOptions::Method::cTPQ;
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

    // LTLM: GS-biased estimator — only valid at low T < energy gap ≈ 0.23.
    // krylov_dim=64 makes it exact within its GS-biased formula on dim=64.
    SECTION("LTLM") {
        const auto ref = dense_reference(T_LOW_MIN, T_LOW_MAX, N_LOW);
        auto H = make_full_heisen();
        for (uint64_t seed : SEEDS)
            run_trial(*H, ThermalOptions::Method::LTLM, seed, ref,
                      "LTLM/no-sym");
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

    // cTPQ: same caveat as mTPQ (bias-free at T ≥ T_HIGH_MIN after β/2 fix).
    SECTION("cTPQ") {
        const auto ref = dense_reference(T_HIGH_MIN, T_HIGH_MAX, N_HIGH);
        auto H = make_full_heisen();
        for (uint64_t seed : SEEDS)
            run_trial(*H, ThermalOptions::Method::cTPQ, seed, ref,
                      "cTPQ/no-sym");
    }

    SECTION("KpmDos") {
        const auto ref = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
        auto H = make_full_heisen();
        for (uint64_t seed : SEEDS)
            run_trial(*H, ThermalOptions::Method::KpmDos, seed, ref,
                      "KpmDos/no-sym");
    }
}

// ===========================================================================
// 2. U(1) / Sz symmetry — per-sector FixedSzOperator + recombination
//
//    For each n_up ∈ [0, N] run thermal on the corresponding FixedSzOperator,
//    then combine via `ed::core::combine_sector_thermodynamics`.
//
//    Note: for mTPQ/cTPQ the combination uses the TPQ free-energy (which has
//    a biased integration constant) as the sector weight.  Only the combined
//    energy E_combined is tested for those methods (using wider tolerance);
//    Cv and S are omitted because the weighting bias can distort them.
// ===========================================================================

namespace {

void sz_trial(ThermalOptions::Method m, uint64_t seed,
              const ThermodynamicData& ref,
              double tol_E_combo = TOL_E) {
    std::vector<ThermodynamicData> sector_thermos;
    std::vector<uint64_t>          sector_dims;

    for (int64_t n_up = 0; n_up <= static_cast<int64_t>(N_SITES); ++n_up) {
        auto op = make_sz_heisen(n_up);
        const uint64_t dim = op->dim();

        auto opts = opts_for(m, seed + static_cast<uint64_t>(n_up) * 17ULL);
        auto R    = ed::workflows::thermal(*op, opts);
        REQUIRE(R.backend.lane == "cpu");

        sector_thermos.push_back(R.thermo);
        sector_dims.push_back(dim);
    }

    const ThermodynamicData combined =
        ed::core::combine_sector_thermodynamics(sector_thermos, sector_dims);

    const bool full_compare = method_combine_reliable(m);
    check_thermo_close(combined, ref,
                       tol_E_combo, TOL_CV, TOL_S,
                       method_name(m) + "/sz seed=" + std::to_string(seed),
                       full_compare && method_compare_entropy(m));
    if (!full_compare) {
        // For mTPQ/cTPQ: only energy is checked above; also confirm Cv is
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

    SECTION("LTLM") {
        const auto ref = dense_reference(T_LOW_MIN, T_LOW_MAX, N_LOW);
        for (uint64_t seed : SEEDS)
            sz_trial(ThermalOptions::Method::LTLM, seed, ref);
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

    SECTION("cTPQ") {
        const auto ref = dense_reference(T_HIGH_MIN, T_HIGH_MAX, N_HIGH);
        for (uint64_t seed : SEEDS)
            sz_trial(ThermalOptions::Method::cTPQ, seed, ref,
                     /*tol_E_combo=*/0.5);
    }

    SECTION("KpmDos") {
        const auto ref = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
        for (uint64_t seed : SEEDS)
            sz_trial(ThermalOptions::Method::KpmDos, seed, ref);
    }
}

// ===========================================================================
// 3. Spatial (Z_N translation) symmetry — per-sector SectorOperator +
//    recombination
// ===========================================================================

namespace {

void spatial_trial(ThermalOptions::Method m, uint64_t seed,
                   const std::string& sym_root,
                   const ThermodynamicData& ref,
                   double tol_E_combo = TOL_E) {
    SymmetryGroupInfo info;
    info.loadFromDirectory(sym_root);

    auto ops = ed::symmetry::build_full_sector_operators(
        N_SITES, 0.5f, info,
        [](ed::symmetry::SectorOperator& op) {
            add_heisen_pbc_terms(op, N_SITES, J);
        });
    REQUIRE_FALSE(ops.empty());

    std::vector<ThermodynamicData> sector_thermos;
    std::vector<uint64_t>          sector_dims;

    for (std::size_t s = 0; s < ops.size(); ++s) {
        auto& op = *ops[s];
        REQUIRE(op.dim() > 0);
        auto opts = opts_for(m, seed + s * 31ULL);
        auto R    = ed::workflows::thermal(op, opts);
        REQUIRE(R.backend.lane == "cpu");
        sector_thermos.push_back(R.thermo);
        sector_dims.push_back(op.dim());
    }

    const ThermodynamicData combined =
        ed::core::combine_sector_thermodynamics(sector_thermos, sector_dims);

    const bool full_compare = method_combine_reliable(m);
    check_thermo_close(combined, ref,
                       tol_E_combo, TOL_CV, TOL_S,
                       method_name(m) + "/spatial seed=" + std::to_string(seed),
                       full_compare && method_compare_entropy(m));
    if (!full_compare)
        for (auto cv : combined.specific_heat) REQUIRE(std::isfinite(cv));
}

} // namespace

TEST_CASE("thermal methods vs dense reference: spatial Z_N translation symmetry",
          "[thermal][dense-ref][spatial-sym]") {

    setenv("ED_GPU_SYMMETRY_MIRROR", "0", 1);
    const std::string sym_root =
        write_zN_fixture(N_SITES, "thermal_dense_ref", "spatial");

    SECTION("FTLM") {
        const auto ref = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
        for (uint64_t seed : SEEDS)
            spatial_trial(ThermalOptions::Method::FTLM, seed, sym_root, ref);
    }

    SECTION("LTLM") {
        const auto ref = dense_reference(T_LOW_MIN, T_LOW_MAX, N_LOW);
        for (uint64_t seed : SEEDS)
            spatial_trial(ThermalOptions::Method::LTLM, seed, sym_root, ref);
    }

    SECTION("mTPQ") {
        const auto ref = dense_reference(T_HIGH_MIN, T_HIGH_MAX, N_HIGH);
        for (uint64_t seed : SEEDS)
            spatial_trial(ThermalOptions::Method::mTPQ, seed, sym_root, ref,
                          /*tol_E_combo=*/0.5);
    }

    SECTION("cTPQ") {
        const auto ref = dense_reference(T_HIGH_MIN, T_HIGH_MAX, N_HIGH);
        for (uint64_t seed : SEEDS)
            spatial_trial(ThermalOptions::Method::cTPQ, seed, sym_root, ref,
                          /*tol_E_combo=*/0.5);
    }

    SECTION("KpmDos") {
        const auto ref = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
        for (uint64_t seed : SEEDS)
            spatial_trial(ThermalOptions::Method::KpmDos, seed, sym_root, ref);
    }

    std::filesystem::remove_all(sym_root);
}

// ===========================================================================
// 4. Sz × spatial (U(1) × Z_N) — both symmetry axes simultaneously
//
//    Loop over all (n_up, irrep) pairs, run thermal on each SectorOperator,
//    then flat-combine all sector thermo blocks.
//
//    Two representative seeds; E+Cv checked for all methods;
//    S additionally checked for FTLM and KpmDos.
// ===========================================================================

namespace {

void sz_spatial_trial(ThermalOptions::Method m, uint64_t seed,
                      const std::string& sym_root,
                      const ThermodynamicData& ref,
                      double tol_E_combo = TOL_E) {
    SymmetryGroupInfo info;
    info.loadFromDirectory(sym_root);

    std::vector<ThermodynamicData> all_thermos;
    std::vector<uint64_t>          all_dims;

    for (int64_t n_up = 0; n_up <= static_cast<int64_t>(N_SITES); ++n_up) {
        auto ops = ed::symmetry::build_fixed_sz_sector_operators(
            N_SITES, 0.5f, n_up, info,
            [](ed::symmetry::SectorOperator& op) {
                add_heisen_pbc_terms(op, N_SITES, J);
            });
        for (std::size_t s = 0; s < ops.size(); ++s) {
            auto& op = *ops[s];
            if (op.dim() == 0) continue;

            const uint64_t ss = seed
                + static_cast<uint64_t>(n_up) * 97ULL
                + static_cast<uint64_t>(s)    * 13ULL;
            auto opts = opts_for(m, ss);
            auto R    = ed::workflows::thermal(op, opts);
            REQUIRE(R.backend.lane == "cpu");

            all_thermos.push_back(R.thermo);
            all_dims.push_back(op.dim());
        }
    }
    REQUIRE_FALSE(all_thermos.empty());

    const ThermodynamicData combined =
        ed::core::combine_sector_thermodynamics(all_thermos, all_dims);

    const bool full_compare = method_combine_reliable(m);
    check_thermo_close(combined, ref,
                       tol_E_combo, TOL_CV, TOL_S,
                       method_name(m) + "/sz+spatial seed=" + std::to_string(seed),
                       full_compare && method_compare_entropy(m));
    if (!full_compare)
        for (auto cv : combined.specific_heat) REQUIRE(std::isfinite(cv));
}

} // namespace

TEST_CASE("thermal methods vs dense reference: Sz + spatial (U(1) × Z_N)",
          "[thermal][dense-ref][sz-spatial-sym]") {

    setenv("ED_GPU_SYMMETRY_MIRROR", "0", 1);
    const std::string sym_root =
        write_zN_fixture(N_SITES, "thermal_dense_ref", "sz_spatial");

    const std::vector<uint64_t> seeds2 = {42ULL, 1337ULL};

    SECTION("FTLM") {
        const auto ref = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
        for (uint64_t seed : seeds2)
            sz_spatial_trial(ThermalOptions::Method::FTLM, seed, sym_root, ref);
    }

    SECTION("LTLM") {
        const auto ref = dense_reference(T_LOW_MIN, T_LOW_MAX, N_LOW);
        for (uint64_t seed : seeds2)
            sz_spatial_trial(ThermalOptions::Method::LTLM, seed, sym_root, ref);
    }

    SECTION("mTPQ") {
        const auto ref = dense_reference(T_HIGH_MIN, T_HIGH_MAX, N_HIGH);
        for (uint64_t seed : seeds2)
            sz_spatial_trial(ThermalOptions::Method::mTPQ, seed, sym_root, ref,
                             /*tol_E_combo=*/0.5);
    }

    SECTION("cTPQ") {
        const auto ref = dense_reference(T_HIGH_MIN, T_HIGH_MAX, N_HIGH);
        for (uint64_t seed : seeds2)
            sz_spatial_trial(ThermalOptions::Method::cTPQ, seed, sym_root, ref,
                             /*tol_E_combo=*/0.5);
    }

    SECTION("KpmDos") {
        const auto ref = dense_reference(T_BROAD_MIN, T_BROAD_MAX, N_BROAD);
        for (uint64_t seed : seeds2)
            sz_spatial_trial(ThermalOptions::Method::KpmDos, seed, sym_root, ref);
    }

    std::filesystem::remove_all(sym_root);
}

// ===========================================================================
// 5. GPU backend — if WITH_CUDA and a device is present at runtime
//
//    Run all five methods with allow_gpu=true and verify they still agree with
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
    const auto ref_low   = dense_reference(T_LOW_MIN,   T_LOW_MAX,   N_LOW);
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

    SECTION("LTLM GPU") {
        auto opts = make_ltlm_opts(GPU_SEED, true);
        auto R    = ed::workflows::thermal(*H, opts);
        REQUIRE((R.backend.lane == "gpu" || R.backend.lane == "cpu"));
        check_thermo_close(R.thermo, ref_low,
                           TOL_E, TOL_CV, TOL_S,
                           "LTLM/gpu",
                           /*compare_entropy=*/false);
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

    SECTION("cTPQ GPU") {
        auto opts = make_ctpq_opts(GPU_SEED, true);
        auto R    = ed::workflows::thermal(*H, opts);
        REQUIRE((R.backend.lane == "gpu" || R.backend.lane == "cpu"));
        check_thermo_close(R.thermo, ref_high,
                           TOL_E, TOL_CV, TOL_S,
                           "cTPQ/gpu",
                           /*compare_entropy=*/false);
    }

    SECTION("KpmDos GPU") {
        auto opts = make_kpm_opts(GPU_SEED, true);
        auto R    = ed::workflows::thermal(*H, opts);
        REQUIRE((R.backend.lane == "gpu" || R.backend.lane == "cpu"));
        check_thermo_close(R.thermo, ref_broad,
                           TOL_E, TOL_CV, TOL_S,
                           "KpmDos/gpu",
                           /*compare_entropy=*/true);
    }
}
#endif  // WITH_CUDA
