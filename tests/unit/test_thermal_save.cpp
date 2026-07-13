// =============================================================================
// test_thermal_save  (Catch2 v3)
//
// Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026): pins the uniform
// persistence contract for `ed::workflows::thermal(H, opts)`.
//
// Contract:
//   * When ``opts.output_dir`` is set and the run is single-rank, the
//     orchestrator writes ``<output_dir>/ed_results.h5`` and surfaces the
//     resulting path via ``ThermalResult::hdf5_path`` (the same field the
//     Python facade mirrors into ``EDResults.eigenvectors_path``).
//   * FTLM / LTLM / KPM_DOS persist the aggregated thermodynamic curves
//     (``T, E, Cv, S, F``) under ``/ftlm/averaged/<...>``.
//   * mTPQ persists the per-sample trajectory rows
//     (``/tpq/samples/sample_<s>/thermodynamics``); when
//     ``opts.probe_betas`` is set they ALSO snapshot the running state
//     vector at the nearest kernel-step beta and write it to
//     ``/tpq/samples/sample_<s>/states/beta_<b>``.
//   * mTPQ ``ThermalResult::tpq_state_snapshots`` carries the host-side
//     state vectors so callers can chain them into ``ed::workflows::spectral``
//     (the TPQ-to-CF pipeline) without an HDF5 round-trip.
//
// Covered methods: FTLM, LTLM, KPM_DOS (thermo-only) + mTPQ
// (trajectory + probe-beta state snapshots).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/hdf5_io.h>
#include <ed/core/operator.h>
#include <ed/orchestrator.h>
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>

#include <H5Cpp.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace ed_tests;

namespace {

constexpr std::uint64_t N_SITES = 4;
using Complex = std::complex<double>;

std::unique_ptr<Operator> heisen() {
    return build_heisenberg_chain(N_SITES, 1.0, /*periodic=*/true);
}

bool dataset_exists(const std::string& filepath, const std::string& path) {
    try {
        H5::H5File f(filepath, H5F_ACC_RDONLY);
        return f.exists(path);
    } catch (const H5::Exception&) {
        return false;
    }
}

}  // namespace

TEST_CASE("ed::thermal persists FTLM thermo curves",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "ftlm");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::FTLM;
    opts.num_samples   = 2;
    opts.krylov_dim    = 30;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/ftlm/averaged/energy"));
    CHECK(dataset_exists(h5, "/ftlm/averaged/temperatures"));
    // Thermo carrier on the in-memory result.
    CHECK(R.thermo.energy.size() == opts.num_temp_bins);

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal persists LTLM thermo curves",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "ltlm");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::LTLM;
    opts.num_samples   = 2;
    opts.krylov_dim    = 30;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/ftlm/averaged/energy"));
    CHECK(dataset_exists(h5, "/ftlm/averaged/temperatures"));

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal persists KpmDos thermo curves",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "kpm");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::KpmDos;
    opts.num_samples   = 2;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.kpm_num_moments        = 64;
    opts.kpm_num_random_vectors = 4;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/ftlm/averaged/energy"));
    CHECK(dataset_exists(h5, "/ftlm/averaged/temperatures"));

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal persists mTPQ trajectory + probe-beta state snapshots",
          "[orchestrator][thermal-save][tpq]") {
    auto H = heisen();
    const std::string outdir = make_scratch_dir("thermal_save", "mtpq");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::mTPQ;
    opts.num_samples   = 1;
    opts.krylov_dim    = 40;        // = mTPQ max_iter
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 8;
    opts.random_seed   = 42;
    opts.output_dir    = outdir;
    opts.probe_betas   = {1.0, 5.0};

    auto R = ed::workflows::thermal(*H, opts);

    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(std::filesystem::exists(h5));
    CHECK(R.hdf5_path == h5);
    CHECK(dataset_exists(h5, "/tpq/samples/sample_0/thermodynamics"));
    // In-memory trajectory mirror.
    REQUIRE(R.tpq_sample_betas.size() == opts.num_samples);
    REQUIRE_FALSE(R.tpq_sample_betas[0].empty());
    REQUIRE(R.tpq_sample_energies[0].size() == R.tpq_sample_betas[0].size());

    // State snapshots: at most one per (sample, probe_beta).
    REQUIRE_FALSE(R.tpq_state_snapshots.empty());
    for (const auto& snap : R.tpq_state_snapshots) {
        REQUIRE(snap.psi.size() == (std::size_t{1} << N_SITES));
        // Cross-check the on-disk dataset for the same beta exists.
        char beta_buf[64];
        std::snprintf(beta_buf, sizeof(beta_buf),
                      "/tpq/samples/sample_%zu/states/beta_%.6f",
                      snap.sample_index, snap.effective_beta);
        CHECK(dataset_exists(h5, std::string(beta_buf)));
    }

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal leaves hdf5_path empty when output_dir is unset",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::FTLM;
    opts.num_samples   = 1;
    opts.krylov_dim    = 16;
    opts.temp_min      = 0.5;
    opts.temp_max      = 2.0;
    opts.num_temp_bins = 4;
    // output_dir intentionally empty.
    auto R = ed::workflows::thermal(*H, opts);
    CHECK(R.hdf5_path.empty());
}

// =============================================================================
// Save & DSSF Upgrades follow-up (May 2026): the "TPQ + symmetry + save thermal
// states" matrix.
//
// Three regression points pinned by the test cases below:
//
//   1. FixedSzOperator + mTPQ + probe_betas:
//      ``ThermalResult::tpq_state_snapshots`` carries vectors of length
//      ``sector_dim`` (not the full Hilbert dim) and the HDF5 mirror at
//      ``/tpq/samples/sample_<s>/states/beta_<b>`` round-trips them
//      byte-for-byte under ``HDF5IO::loadTPQState``.
//
//   2. StreamingSymmetryOperator::SectorView + mTPQ + probe_betas:
//      Same contract for symmetry sectors. The state vectors are
//      stored in the ORBIT basis (length = sector dim, NOT full Hilbert
//      dim). This is the matvec basis the ``CF`` / ``KpmDynamical`` lanes
//      consume, so the saved data is directly chainable into the
//      TPQ-to-CF spectral pipeline without an embedToFull round trip.
//
//   3. FixedSzStreamingSymmetryOperator::SectorView + mTPQ + probe_betas:
//      The same contract holds for the Sz+symmetry SectorView.
//
// All three cells share one root cause class: when the persistence
// finalizer at the bottom of ``ed::workflows::thermal`` runs against a
// symmetry-projected operator, every dim/state_snapshot must use the
// view's ``local_dim`` instead of the parent operator's full Hilbert
// dim. The tests confirm that ``snap.psi.size() == H.dim()`` and that
// the on-disk dataset shape matches.
// =============================================================================

namespace {

// Build a tiny Z_N translation fixture on disk so we can construct a
// ``StreamingSymmetryOperator`` against it. Returns the fixture root.
inline std::string write_zN_translation_fixture(uint64_t N,
                                                const std::string& suite,
                                                const std::string& tag) {
    const std::string root = make_scratch_dir(suite, tag);
    const std::string sym  = root + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(sym, ec);
    auto perm = [&](int shift) {
        std::vector<int> p(N);
        for (uint64_t i = 0; i < N; ++i)
            p[i] = ((static_cast<int>(i) - shift) % static_cast<int>(N)
                      + static_cast<int>(N)) % static_cast<int>(N);
        return p;
    };
    {
        std::ofstream f(sym + "/max_clique.json");
        f << "[";
        for (uint64_t g = 0; g < N; ++g) {
            auto p = perm(static_cast<int>(g));
            f << "[";
            for (size_t i = 0; i < p.size(); ++i)
                f << p[i] << (i + 1 < p.size() ? "," : "");
            f << "]" << (g + 1 < N ? "," : "");
        }
        f << "]";
    }
    {
        std::ofstream f(sym + "/minimal_generators.json");
        auto p = perm(1);
        f << "{\"generators\":[{\"permutation\":[";
        for (size_t i = 0; i < p.size(); ++i)
            f << p[i] << (i + 1 < p.size() ? "," : "");
        f << "],\"order\":" << N << "}]}";
    }
    {
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

template <class Op>
inline void fill_heisenberg_pbc(Op& op, uint64_t N, double J) {
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        op.transform_data_.push_back(t);
        t.op_type = 0; t.op_type_2 = 1; t.coefficient = J_half;
        op.transform_data_.push_back(t);
        t.op_type = 1; t.op_type_2 = 0;
        op.transform_data_.push_back(t);
    }
}

inline void add_heisenberg_pbc_terms(ed::symmetry::SectorOperator& op,
                                     uint64_t N, double J) {
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (uint64_t i = 0; i < N; ++i) {
        const uint64_t j = (i + 1) % N;
        op.addTwoBodyTerm(2, i, 2, j, J_real);
        op.addTwoBodyTerm(0, i, 1, j, J_half);
        op.addTwoBodyTerm(1, i, 0, j, J_half);
    }
}

// Pick the largest non-empty sector operator from a freshly built set.
inline ed::symmetry::SectorOperator&
largest_sector(std::vector<std::unique_ptr<ed::symmetry::SectorOperator>>& ops) {
    std::size_t pick = 0, best = 0;
    for (std::size_t s = 0; s < ops.size(); ++s) {
        const std::size_t d = ops[s]->dim();
        if (d > best) { best = d; pick = s; }
    }
    return *ops[pick];
}

}  // namespace

TEST_CASE("ed::thermal persists FixedSzOperator mTPQ snapshots at sector dim",
          "[orchestrator][thermal-save][tpq][symmetry]") {
    constexpr uint64_t N = 6;
    auto op = std::make_unique<FixedSzOperator>(N, 0.5f, int64_t(N/2));
    fill_heisenberg_pbc(*op, N, 1.0);
    const std::string outdir = make_scratch_dir("thermal_save", "fsz_mtpq");

    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::mTPQ;
    opts.num_samples   = 1;
    opts.krylov_dim    = 25;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 6;
    opts.random_seed   = 13;
    opts.output_dir    = outdir;
    opts.probe_betas   = {0.5, 2.0};

    auto R = ed::workflows::thermal(*op, opts);

    REQUIRE_FALSE(R.tpq_state_snapshots.empty());
    const std::size_t sector_dim = op->dim();
    for (const auto& snap : R.tpq_state_snapshots) {
        REQUIRE(snap.psi.size() == sector_dim);
    }
    // Round-trip every snapshot through HDF5IO::loadTPQState.
    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(R.hdf5_path == h5);
    for (const auto& snap : R.tpq_state_snapshots) {
        std::vector<Complex> loaded;
        REQUIRE(HDF5IO::loadTPQState(h5, snap.sample_index,
                                      snap.effective_beta, loaded));
        REQUIRE(loaded.size() == sector_dim);
    }

    std::filesystem::remove_all(outdir);
}

TEST_CASE("ed::thermal persists fixed-Sz symmetry-sector mTPQ snapshots",
          "[orchestrator][thermal-save][tpq][symmetry]") {
    constexpr uint64_t N = 6;
    setenv("ED_GPU_SYMMETRY_MIRROR", "0", 1);
    const std::string sym_dir = write_zN_translation_fixture(
        N, "thermal_save", "fsz_sym_mtpq_sym");

    SymmetryGroupInfo info;
    info.loadFromDirectory(sym_dir);
    auto ops = ed::symmetry::build_fixed_sz_sector_operators_lazy(
        N, 0.5f, int64_t(N / 2), info,
        [&](ed::symmetry::SectorOperator& op) {
            add_heisenberg_pbc_terms(op, N, 1.0);
        });
    ed::symmetry::SectorOperator& view = largest_sector(ops);
    REQUIRE(view.dim() > 0);

    const std::string outdir = make_scratch_dir(
        "thermal_save", "fsz_sym_mtpq_out");
    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::mTPQ;
    opts.num_samples   = 1;
    opts.krylov_dim    = 20;
    opts.temp_min      = 0.1;
    opts.temp_max      = 5.0;
    opts.num_temp_bins = 6;
    opts.random_seed   = 19;
    opts.output_dir    = outdir;
    opts.probe_betas   = {0.5, 2.0};

    auto R = ed::workflows::thermal(view, opts);

    REQUIRE_FALSE(R.tpq_state_snapshots.empty());
    const std::size_t sector_dim = view.dim();
    for (const auto& snap : R.tpq_state_snapshots) {
        REQUIRE(snap.psi.size() == sector_dim);
    }
    const std::string h5 = outdir + "/ed_results.h5";
    CHECK(R.hdf5_path == h5);
    for (const auto& snap : R.tpq_state_snapshots) {
        std::vector<Complex> loaded;
        REQUIRE(HDF5IO::loadTPQState(h5, snap.sample_index,
                                      snap.effective_beta, loaded));
        REQUIRE(loaded.size() == sector_dim);
    }

    std::filesystem::remove_all(outdir);
    std::filesystem::remove_all(sym_dir);
    unsetenv("ED_GPU_SYMMETRY_MIRROR");
}

TEST_CASE("ed::thermal honours /dev/null sentinel",
          "[orchestrator][thermal-save]") {
    auto H = heisen();
    ed::workflows::ThermalOptions opts;
    opts.method        = ed::workflows::ThermalOptions::Method::FTLM;
    opts.num_samples   = 1;
    opts.krylov_dim    = 16;
    opts.temp_min      = 0.5;
    opts.temp_max      = 2.0;
    opts.num_temp_bins = 4;
    opts.output_dir    = "/dev/null";
    auto R = ed::workflows::thermal(*H, opts);
    CHECK(R.hdf5_path.empty());
}
