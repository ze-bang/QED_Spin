// =============================================================================
// test_hdf5_io (Catch2 v3, P1.8 / audit Q12)
//
// Round-trip tests for the unified HDF5 layer (`HDF5IO` in
// `ed/core/hdf5_io.h`).
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/hdf5_io.h>
#include <ed/solvers/lanczos.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <vector>

using namespace ed_tests;

TEST_CASE("HDF5IO::createOrOpenFile lifecycle", "[hdf5_io][create]") {
    std::string dir = make_scratch_dir("hdf5_io", "create");
    std::string path = HDF5IO::createOrOpenFile(dir, "ed_results.h5");

    INFO("path=" << path);
    REQUIRE(std::filesystem::exists(path));
    REQUIRE(HDF5IO::fileExists(path));

    std::vector<double> eigs = {1.5, 2.5, 3.5};
    HDF5IO::saveEigenvalues(path, eigs);

    std::string path_again = HDF5IO::createOrOpenFile(dir, "ed_results.h5");
    REQUIRE(path == path_again);

    auto eigs_back = HDF5IO::loadEigenvalues(path_again);
    REQUIRE(eigs_back.size() == eigs.size());
    for (size_t i = 0; i < eigs.size(); ++i) {
        INFO("i=" << i << " back=" << eigs_back[i] << " want=" << eigs[i]);
        REQUIRE(std::abs(eigs_back[i] - eigs[i]) <= 1e-15);
    }
}

TEST_CASE("HDF5IO eigenvalue round-trip", "[hdf5_io][eigvals]") {
    std::string dir = make_scratch_dir("hdf5_io", "eigvals");
    std::string path = HDF5IO::createOrOpenFile(dir);

    std::vector<double> eigs = {-3.14, -2.71, -1.41, 0.0,
                                1.61, 2.23, 3.0, 4.5, 5.0};
    HDF5IO::saveEigenvalues(path, eigs);
    auto loaded = HDF5IO::loadEigenvalues(path);

    REQUIRE(loaded.size() == eigs.size());
    for (size_t i = 0; i < eigs.size(); ++i) {
        INFO("i=" << i);
        REQUIRE(loaded[i] == eigs[i]);
    }

    std::vector<double> eigs2 = {-1.0, 0.0, 1.0};
    HDF5IO::saveEigenvalues(path, eigs2);
    auto loaded2 = HDF5IO::loadEigenvalues(path);
    REQUIRE(loaded2.size() == eigs2.size());
}

TEST_CASE("HDF5IO eigenvector round-trip (complex)",
          "[hdf5_io][eigvecs]") {
    std::string dir = make_scratch_dir("hdf5_io", "eigvecs");
    std::string path = HDF5IO::createOrOpenFile(dir);

    const size_t N = 64;
    std::vector<Complex> v(N);
    for (size_t i = 0; i < N; ++i) {
        v[i] = Complex(std::sin(0.1 * i + 0.3),
                       std::cos(0.1 * i + 0.3) * 0.5);
    }

    HDF5IO::saveEigenvector(path, /*index=*/0, v);
    auto loaded = HDF5IO::loadEigenvector(path, 0);

    REQUIRE(loaded.size() == v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        INFO("i=" << i);
        REQUIRE(loaded[i] == v[i]);
    }

    std::vector<Complex> v2(N);
    for (size_t i = 0; i < N; ++i) v2[i] = Complex(double(i), -double(i));
    HDF5IO::saveEigenvector(path, /*index=*/2, v2);

    auto loaded0_again = HDF5IO::loadEigenvector(path, 0);
    auto loaded2 = HDF5IO::loadEigenvector(path, 2);
    REQUIRE(loaded0_again.size() == N);
    REQUIRE(loaded2.size() == N);
    for (size_t i = 0; i < N; ++i) {
        INFO("multi-index i=" << i);
        REQUIRE(loaded0_again[i] == v[i]);
        REQUIRE(loaded2[i] == v2[i]);
    }
}

TEST_CASE("HDF5IO::saveDiagonalizationResults round-trip",
          "[hdf5_io][diag_results]") {
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
    REQUIRE(std::filesystem::exists(path));

    auto loaded_eigs = HDF5IO::loadEigenvalues(path);
    REQUIRE(loaded_eigs == eigs);

    for (size_t i = 0; i < vecs.size(); ++i) {
        auto v = HDF5IO::loadEigenvector(path, i);
        INFO("i=" << i);
        REQUIRE(v.size() == vecs[i].size());
        for (size_t k = 0; k < v.size(); ++k) {
            REQUIRE(v[k] == vecs[i][k]);
        }
    }
}

TEST_CASE("HDF5IO thermodynamics save/load",
          "[hdf5_io][thermo]") {
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

    REQUIRE(E_back == E);
    REQUIRE(S_back == S);
    REQUIRE(Cv_back == Cv);
}

TEST_CASE("full_diagonalization writes valid HDF5",
          "[hdf5_io][full_diag]") {
    std::string dir = make_scratch_dir("hdf5_io", "full_diag");

    auto op = build_heisenberg_chain(/*N=*/4, /*J=*/1.0, /*periodic=*/true);
    const uint64_t dim = 16;

    auto ref = reference_from_operator(*op, dim);

    std::vector<double> eigs;
    auto Hv = [op_ptr = op.get()](const Complex* in, Complex* out, int n) {
        op_ptr->apply(in, out, static_cast<size_t>(n));
    };
    full_diagonalization(Hv, dim, /*num_eigs=*/dim, eigs, dir,
                         /*compute_eigenvectors=*/false);

    INFO("eigs.size()=" << eigs.size());
    REQUIRE(eigs.size() == dim);

    std::string h5_path = dir + "/ed_results.h5";
    INFO(h5_path);
    REQUIRE(std::filesystem::exists(h5_path));

    auto loaded = HDF5IO::loadEigenvalues(h5_path);
    require_eigs_close(loaded, ref.eigs, dim, 1e-9,
                       "HDF5 eigenvalues match dense reference");
    require_eigs_close(eigs, loaded, dim, 1e-15,
                       "HDF5 round trip matches in-memory");
}
