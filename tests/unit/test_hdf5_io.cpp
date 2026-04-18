// =============================================================================
// test_hdf5_io
//
// Round-trip tests for the unified HDF5 layer (`HDF5IO` in
// `ed/core/hdf5_io.h`).
//
// Coverage:
//   1. createOrOpenFile + standard groups: file is created, ensureStandardGroups
//      runs without error, and reopening preserves data.
//   2. saveEigenvalues / loadEigenvalues: byte-for-byte round trip on a
//      hand-written spectrum.
//   3. saveEigenvector / loadEigenvector: complex round trip preserving real
//      and imaginary parts.
//   4. saveDiagonalizationResults: end-to-end driver path used by the real
//      solvers, written + read back.
//   5. saveThermodynamics / loadThermodynamicObservable: per-observable
//      append + load.
//   6. Integration with `full_diagonalization`: actually run the solver,
//      let it write its HDF5 file, then read the eigenvalues back and
//      compare to the dense Eigen reference.
// =============================================================================

#include "common/test_harness.h"

#include <ed/core/hdf5_io.h>
#include <ed/solvers/lanczos.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <vector>

using namespace ed_tests;

// `full_diagonalization` is declared in `lanczos.h` (already included).

// -----------------------------------------------------------------------------
// 1. Basic create/reopen behaviour.
// -----------------------------------------------------------------------------
static void test_create_or_open(TestContext& ctx) {
    std::string dir = make_scratch_dir("hdf5_io", "create");
    std::string path = HDF5IO::createOrOpenFile(dir, "ed_results.h5");

    check(ctx, std::filesystem::exists(path),
          "createOrOpenFile produces an HDF5 file",
          path);
    check(ctx, HDF5IO::fileExists(path),
          "fileExists() returns true for the freshly-created file");

    // Reopen should not destroy contents: write something, reopen, check it
    // is still there.
    std::vector<double> eigs = {1.5, 2.5, 3.5};
    HDF5IO::saveEigenvalues(path, eigs);

    std::string path_again = HDF5IO::createOrOpenFile(dir, "ed_results.h5");
    check(ctx, path == path_again,
          "createOrOpenFile is idempotent on the same directory");

    auto eigs_back = HDF5IO::loadEigenvalues(path_again);
    check(ctx, eigs_back.size() == eigs.size(),
          "createOrOpenFile preserves data on reopen (size)");
    bool match = true;
    for (size_t i = 0; i < eigs.size(); ++i) {
        if (std::abs(eigs_back[i] - eigs[i]) > 1e-15) { match = false; break; }
    }
    check(ctx, match,
          "createOrOpenFile preserves data on reopen (values)");
}

// -----------------------------------------------------------------------------
// 2. Eigenvalue round-trip.
// -----------------------------------------------------------------------------
static void test_eigenvalues_roundtrip(TestContext& ctx) {
    std::string dir = make_scratch_dir("hdf5_io", "eigvals");
    std::string path = HDF5IO::createOrOpenFile(dir);

    std::vector<double> eigs = {-3.14, -2.71, -1.41, 0.0,
                                1.61, 2.23, 3.0, 4.5, 5.0};
    HDF5IO::saveEigenvalues(path, eigs);
    auto loaded = HDF5IO::loadEigenvalues(path);

    check(ctx, loaded.size() == eigs.size(),
          "loadEigenvalues() size matches");
    double max_err = 0.0;
    for (size_t i = 0; i < eigs.size(); ++i) {
        max_err = std::max(max_err, std::abs(loaded[i] - eigs[i]));
    }
    check(ctx, max_err == 0.0,
          "loadEigenvalues() byte-exact round trip",
          "max |Δ| = " + std::to_string(max_err));

    // Overwrite: saving again with a different vector must replace, not
    // append, the dataset.
    std::vector<double> eigs2 = {-1.0, 0.0, 1.0};
    HDF5IO::saveEigenvalues(path, eigs2);
    auto loaded2 = HDF5IO::loadEigenvalues(path);
    check(ctx, loaded2.size() == eigs2.size(),
          "saveEigenvalues() overwrites prior dataset (size)");
}

// -----------------------------------------------------------------------------
// 3. Eigenvector round-trip.
// -----------------------------------------------------------------------------
static void test_eigenvector_roundtrip(TestContext& ctx) {
    std::string dir = make_scratch_dir("hdf5_io", "eigvecs");
    std::string path = HDF5IO::createOrOpenFile(dir);

    // Make a deterministic complex vector with non-trivial real and imag
    // parts so we can detect any half/full silent loss.
    const size_t N = 64;
    std::vector<Complex> v(N);
    for (size_t i = 0; i < N; ++i) {
        v[i] = Complex(std::sin(0.1 * i + 0.3),
                       std::cos(0.1 * i + 0.3) * 0.5);
    }

    HDF5IO::saveEigenvector(path, /*index=*/0, v);
    auto loaded = HDF5IO::loadEigenvector(path, 0);

    check(ctx, loaded.size() == v.size(),
          "loadEigenvector() size matches");
    double max_err = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        max_err = std::max(max_err, std::abs(loaded[i] - v[i]));
    }
    check(ctx, max_err == 0.0,
          "loadEigenvector() byte-exact round trip (complex)",
          "max |Δ| = " + std::to_string(max_err));

    // Multi-index: write index=2, ensure index=0 still readable.
    std::vector<Complex> v2(N);
    for (size_t i = 0; i < N; ++i) v2[i] = Complex(double(i), -double(i));
    HDF5IO::saveEigenvector(path, /*index=*/2, v2);

    auto loaded0_again = HDF5IO::loadEigenvector(path, 0);
    auto loaded2 = HDF5IO::loadEigenvector(path, 2);
    bool ok = (loaded0_again.size() == N && loaded2.size() == N);
    if (ok) {
        for (size_t i = 0; i < N; ++i) {
            if (std::abs(loaded0_again[i] - v[i]) != 0.0 ||
                std::abs(loaded2[i] - v2[i]) != 0.0) { ok = false; break; }
        }
    }
    check(ctx, ok,
          "Multi-index saveEigenvector() preserves all stored vectors");
}

// -----------------------------------------------------------------------------
// 4. saveDiagonalizationResults driver path.
// -----------------------------------------------------------------------------
static void test_save_diag_results(TestContext& ctx) {
    std::string dir = make_scratch_dir("hdf5_io", "diag_results");

    std::vector<double> eigs = {-2.0, -1.0, 0.5, 1.5};
    std::vector<std::vector<Complex>> vecs;
    for (size_t i = 0; i < eigs.size(); ++i) {
        std::vector<Complex> v(8);
        for (size_t k = 0; k < v.size(); ++k) {
            v[k] = Complex(double(i + 1), double(k));
        }
        vecs.push_back(std::move(v));
    }

    HDF5IO::saveDiagonalizationResults(dir, eigs, vecs, "TEST");

    std::string path = dir + "/ed_results.h5";
    check(ctx, std::filesystem::exists(path),
          "saveDiagonalizationResults creates ed_results.h5");

    auto loaded_eigs = HDF5IO::loadEigenvalues(path);
    bool eigs_ok = (loaded_eigs == eigs);
    check(ctx, eigs_ok,
          "saveDiagonalizationResults round-trips eigenvalues");

    bool vecs_ok = true;
    for (size_t i = 0; i < vecs.size(); ++i) {
        auto v = HDF5IO::loadEigenvector(path, i);
        if (v.size() != vecs[i].size()) { vecs_ok = false; break; }
        for (size_t k = 0; k < v.size(); ++k) {
            if (std::abs(v[k] - vecs[i][k]) != 0.0) { vecs_ok = false; break; }
        }
        if (!vecs_ok) break;
    }
    check(ctx, vecs_ok,
          "saveDiagonalizationResults round-trips all eigenvectors");
}

// -----------------------------------------------------------------------------
// 5. Thermodynamics save/load.
// -----------------------------------------------------------------------------
static void test_thermodynamics(TestContext& ctx) {
    std::string dir = make_scratch_dir("hdf5_io", "thermo");
    std::string path = HDF5IO::createOrOpenFile(dir);

    std::vector<double> T  = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0};
    std::vector<double> E  = {-2.0, -1.9, -1.5, -1.0, -0.5, 0.0};
    std::vector<double> S  = {0.0, 0.1, 0.4, 0.9, 1.5, 2.0};
    std::vector<double> Cv = {0.0, 0.5, 0.8, 1.0, 0.7, 0.2};

    HDF5IO::saveThermodynamics(path, T, "energy", E);
    HDF5IO::saveThermodynamics(path, T, "entropy", S);
    HDF5IO::saveThermodynamics(path, T, "specific_heat", Cv);

    auto E_back  = HDF5IO::loadThermodynamicObservable(path, "energy");
    auto S_back  = HDF5IO::loadThermodynamicObservable(path, "entropy");
    auto Cv_back = HDF5IO::loadThermodynamicObservable(path, "specific_heat");

    check(ctx, E_back == E, "energy round trip");
    check(ctx, S_back == S, "entropy round trip");
    check(ctx, Cv_back == Cv, "specific_heat round trip");
}

// -----------------------------------------------------------------------------
// 6. Integration: full_diagonalization writes an HDF5 file we can verify.
// -----------------------------------------------------------------------------
static void test_full_diag_integration(TestContext& ctx) {
    std::string dir = make_scratch_dir("hdf5_io", "full_diag");

    // Tiny system so the full_diagonalization is fast.
    auto op = build_heisenberg_chain(/*N=*/4, /*J=*/1.0, /*periodic=*/true);
    const uint64_t dim = 16;

    auto ref = reference_from_operator(*op, dim);

    std::vector<double> eigs;
    auto Hv = [op_ptr = op.get()](const Complex* in, Complex* out, int n) {
        op_ptr->apply(in, out, static_cast<size_t>(n));
    };
    full_diagonalization(Hv, dim, /*num_eigs=*/dim, eigs, dir,
                         /*compute_eigenvectors=*/false);

    check(ctx, eigs.size() == dim,
          "full_diagonalization returns dim eigenvalues",
          "got " + std::to_string(eigs.size()));

    std::string h5_path = dir + "/ed_results.h5";
    check(ctx, std::filesystem::exists(h5_path),
          "full_diagonalization writes ed_results.h5",
          h5_path);

    auto loaded = HDF5IO::loadEigenvalues(h5_path);
    check_eigs_close(ctx, loaded, ref.eigs, dim, 1e-9,
                     "HDF5 eigenvalues match dense reference");

    // Sanity: the in-memory result and the on-disk result must agree.
    check_eigs_close(ctx, eigs, loaded, dim, 1e-15,
                     "HDF5 round trip matches in-memory eigenvalues");
}

int main() {
    TestContext ctx("test_hdf5_io");
    test_create_or_open(ctx);
    test_eigenvalues_roundtrip(ctx);
    test_eigenvector_roundtrip(ctx);
    test_save_diag_results(ctx);
    test_thermodynamics(ctx);
    test_full_diag_integration(ctx);
    return ctx.summary_exit_code();
}
