// =============================================================================
// test_bfg_wavefunction_io (Catch2 v3, P2.1 third slice)
//
// Round-trip tests for `ed::bfg::load_wavefunction`,
// `ed::bfg::load_tpq_state`, and `ed::bfg::load_all_tpq_states`.
//
// These are the HDF5 wavefunction loaders extracted from the CPU BFG
// driver into the `ed_bfg` static library so the GPU driver and the Python
// bindings can share the same authoritative implementation.
//
// What we cover:
//   * Single-eigenvector load via the canonical `eigendata/eigenvector_<idx>`
//     dataset path (compound complex type with `(real, imag)` field names).
//   * Single-eigenvector load via the legacy top-level `eigenvector_<idx>`
//     path with `(r, i)` field names (h5py default).
//   * Raw-double fallback layout (interleaved [re, im, re, im, ...]).
//   * `load_tpq_state` returns the lowest-T (highest-beta) snapshot.
//   * `load_all_tpq_states` returns snapshots sorted ascending in T.
//   * Error paths: missing dataset / missing TPQ states group.
// =============================================================================

#include "common/catch2_harness.h"

#include <catch2/catch_approx.hpp>

#include <ed/bfg/wavefunction_io.h>

#include <H5Cpp.h>
#include <unistd.h>

#include <complex>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Complex = std::complex<double>;

std::string make_temp_h5(const std::string& tag) {
    auto dir = std::filesystem::temp_directory_path() /
               ("ed_bfg_wfio_" + tag + "_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    auto path = dir / "out.h5";
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
    return path.string();
}

H5::CompType make_complex_compound(const std::string& real_name,
                                   const std::string& imag_name) {
    H5::CompType ct(sizeof(Complex));
    ct.insertMember(real_name, 0, H5::PredType::NATIVE_DOUBLE);
    ct.insertMember(imag_name, sizeof(double), H5::PredType::NATIVE_DOUBLE);
    return ct;
}

void write_compound_dataset(H5::H5File& file, const std::string& path,
                            const std::vector<Complex>& data,
                            const std::string& real_name,
                            const std::string& imag_name) {
    hsize_t dims[1] = {data.size()};
    H5::DataSpace space(1, dims);
    H5::CompType ct = make_complex_compound(real_name, imag_name);
    auto ds = file.createDataSet(path, ct, space);
    ds.write(data.data(), ct);
}

void write_real_double_dataset(H5::H5File& file, const std::string& path,
                               const std::vector<double>& data) {
    hsize_t dims[1] = {data.size()};
    H5::DataSpace space(1, dims);
    auto ds = file.createDataSet(path, H5::PredType::NATIVE_DOUBLE, space);
    ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}

}  // namespace

TEST_CASE("ed::bfg::load_wavefunction reads canonical eigendata/eigenvector_0",
          "[bfg][wavefunction_io]") {
    const auto path = make_temp_h5("canonical");
    const std::vector<Complex> expected = {
        {1.0, 0.0}, {0.0, -2.5}, {3.5, 4.5}, {-1.0, -1.0},
    };
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        file.createGroup("/eigendata");
        write_compound_dataset(file, "/eigendata/eigenvector_0", expected,
                               "real", "imag");
    }

    auto loaded = ed::bfg::load_wavefunction(path, /*idx=*/0, /*verbose=*/false);
    REQUIRE(loaded.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(loaded[i].real() == Catch::Approx(expected[i].real()));
        REQUIRE(loaded[i].imag() == Catch::Approx(expected[i].imag()));
    }
}

TEST_CASE("ed::bfg::load_wavefunction reads legacy top-level eigenvector_0 "
          "with (r,i) compound",
          "[bfg][wavefunction_io]") {
    const auto path = make_temp_h5("ri_compound");
    const std::vector<Complex> expected = {
        {0.5, 0.5}, {-0.5, -0.5}, {1.0, -1.0},
    };
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        write_compound_dataset(file, "/eigenvector_0", expected, "r", "i");
    }

    auto loaded = ed::bfg::load_wavefunction(path, /*idx=*/0, /*verbose=*/false);
    REQUIRE(loaded.size() == expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(loaded[i].real() == Catch::Approx(expected[i].real()));
        REQUIRE(loaded[i].imag() == Catch::Approx(expected[i].imag()));
    }
}

TEST_CASE("ed::bfg::load_wavefunction reads real-only doubles as Complex(re, 0)",
          "[bfg][wavefunction_io]") {
    const auto path = make_temp_h5("real_only");
    const std::vector<double> reals = {2.0, -1.5, 0.25, 3.0};
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        // The loader probes `ground_state` last; a non-compound (8-byte
        // double) dataset exercises the real-only fallback path. The CPU
        // driver historically treats these as Complex(re, 0), and we
        // preserve that contract.
        write_real_double_dataset(file, "/ground_state", reals);
    }

    auto loaded = ed::bfg::load_wavefunction(path, /*idx=*/0, /*verbose=*/false);
    REQUIRE(loaded.size() == reals.size());
    for (std::size_t i = 0; i < reals.size(); ++i) {
        REQUIRE(loaded[i].real() == Catch::Approx(reals[i]));
        REQUIRE(loaded[i].imag() == Catch::Approx(0.0));
    }
}

TEST_CASE("ed::bfg::load_wavefunction throws on a file with no recognized "
          "dataset",
          "[bfg][wavefunction_io]") {
    const auto path = make_temp_h5("missing_dataset");
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        // Create an unrelated dataset so the file is not empty.
        std::vector<double> dummy{1.0, 2.0, 3.0};
        hsize_t dims[1] = {dummy.size()};
        H5::DataSpace space(1, dims);
        auto ds = file.createDataSet("not_a_wavefunction",
                                     H5::PredType::NATIVE_DOUBLE, space);
        ds.write(dummy.data(), H5::PredType::NATIVE_DOUBLE);
    }
    REQUIRE_THROWS_AS(
        ed::bfg::load_wavefunction(path, /*idx=*/0, /*verbose=*/false),
        std::runtime_error);
}

TEST_CASE("ed::bfg::load_tpq_state returns the lowest-temperature snapshot",
          "[bfg][wavefunction_io]") {
    const auto path = make_temp_h5("tpq_lowT");
    const std::vector<Complex> psi_lowT = {
        {1.0, 0.0}, {0.0, 1.0}, {-1.0, 0.0}, {0.0, -1.0},
    };
    const std::vector<Complex> psi_highT = {
        {0.5, 0.0}, {0.0, 0.5}, {-0.5, 0.0}, {0.0, -0.5},
    };
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        file.createGroup("/tpq");
        file.createGroup("/tpq/samples");
        file.createGroup("/tpq/samples/sample_0");
        file.createGroup("/tpq/samples/sample_0/states");
        // beta=0.1 (high T = 10) and beta=2.0 (low T = 0.5).
        write_compound_dataset(file, "/tpq/samples/sample_0/states/beta_0.1",
                               psi_highT, "real", "imag");
        write_compound_dataset(file, "/tpq/samples/sample_0/states/beta_2.0",
                               psi_lowT, "real", "imag");
    }

    auto [psi, temperature] = ed::bfg::load_tpq_state(path, /*sample=*/0,
                                                      /*verbose=*/false);
    REQUIRE(temperature == Catch::Approx(0.5));
    REQUIRE(psi.size() == psi_lowT.size());
    for (std::size_t i = 0; i < psi.size(); ++i) {
        REQUIRE(psi[i].real() == Catch::Approx(psi_lowT[i].real()));
        REQUIRE(psi[i].imag() == Catch::Approx(psi_lowT[i].imag()));
    }
}

TEST_CASE("ed::bfg::load_all_tpq_states returns snapshots ascending in T",
          "[bfg][wavefunction_io]") {
    const auto path = make_temp_h5("tpq_all");
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        file.createGroup("/tpq");
        file.createGroup("/tpq/samples");
        file.createGroup("/tpq/samples/sample_0");
        file.createGroup("/tpq/samples/sample_0/states");
        // Write betas out of order; loader must sort by ascending T.
        write_compound_dataset(file, "/tpq/samples/sample_0/states/beta_0.5",
                               std::vector<Complex>{{1.0, 0.0}, {0.0, 0.0}},
                               "real", "imag");
        write_compound_dataset(file, "/tpq/samples/sample_0/states/beta_0.1",
                               std::vector<Complex>{{0.0, 1.0}, {0.0, 0.0}},
                               "real", "imag");
        write_compound_dataset(file, "/tpq/samples/sample_0/states/beta_2.0",
                               std::vector<Complex>{{0.0, 0.0}, {1.0, 0.0}},
                               "real", "imag");
    }

    auto snapshots = ed::bfg::load_all_tpq_states(path, /*sample=*/0,
                                                   /*verbose=*/false);
    REQUIRE(snapshots.size() == 3u);

    // Ascending in temperature.
    REQUIRE(snapshots[0].temperature < snapshots[1].temperature);
    REQUIRE(snapshots[1].temperature < snapshots[2].temperature);

    // Lowest-T (= largest beta = 2.0) is first.
    REQUIRE(snapshots[0].beta == Catch::Approx(2.0));
    REQUIRE(snapshots[0].temperature == Catch::Approx(0.5));

    // Highest-T (= smallest beta = 0.1) is last.
    REQUIRE(snapshots.back().beta == Catch::Approx(0.1));
    REQUIRE(snapshots.back().temperature == Catch::Approx(10.0));
}

TEST_CASE("ed::bfg::load_all_tpq_states throws on a file with no TPQ states",
          "[bfg][wavefunction_io]") {
    const auto path = make_temp_h5("no_tpq");
    {
        H5::H5File file(path, H5F_ACC_TRUNC);
        file.createGroup("/eigendata");
    }
    REQUIRE_THROWS_AS(
        ed::bfg::load_all_tpq_states(path, /*sample=*/0, /*verbose=*/false),
        std::runtime_error);
}
