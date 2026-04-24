// =============================================================================
// test_dssf_io (Catch2 v3, P2.3 / DSSF PR-D round-trip tests)
//
// Round-trip tests for the unified `/dssf/...` HDF5 schema introduced in
// P2.3 (audit §3.10). Together with `test_dssf_legacy_schema.cpp` these
// pin the on-disk contract:
//
//   * The legacy `/dynamical/` and `/correlations/` writers stay
//     bit-identical (legacy_schema tests).
//   * The new unified `/dssf/` writer round-trips every Record field,
//     populates the metadata attributes correctly, refuses unknown
//     schema versions, and supports nested operator names.
//
// =============================================================================

#include "common/catch2_harness.h"

#include <catch2/catch_approx.hpp>

#include <ed/dssf/dssf_io.h>

#include <H5Cpp.h>
#include <unistd.h>

#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::string make_temp_h5(const std::string& tag) {
    auto dir = std::filesystem::temp_directory_path() /
               ("ed_dssf_io_" + tag + "_" + std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    auto path = dir / "out.h5";
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
    return path.string();
}

ed::dssf::Metadata make_dyn_meta() {
    ed::dssf::Metadata meta;
    meta.method      = ed::dssf::DSSFMethod::DYNAMICAL_THERMAL;
    meta.num_sites   = 4;
    meta.spin_length = 0.5;
    meta.created_at  = "2026-04-24T00:00:00Z";
    return meta;
}

} // namespace

TEST_CASE("ensure_metadata creates /dssf with the correct attributes",
          "[dssf][hdf5][p2-3][io]") {
    const auto path = make_temp_h5("meta");
    const auto meta = make_dyn_meta();

    ed::dssf::ensure_metadata(path, meta);

    H5::H5File file(path, H5F_ACC_RDONLY);
    REQUIRE(file.nameExists("/dssf"));

    H5::Group root = file.openGroup("/dssf");

    std::uint32_t version{};
    root.openAttribute("schema_version").read(
        H5::PredType::NATIVE_UINT32, &version);
    REQUIRE(version == ed::dssf::kSchemaVersion);

    std::uint64_t num_sites{};
    root.openAttribute("num_sites").read(
        H5::PredType::NATIVE_UINT64, &num_sites);
    REQUIRE(num_sites == meta.num_sites);

    double spin{};
    root.openAttribute("spin_length").read(
        H5::PredType::NATIVE_DOUBLE, &spin);
    REQUIRE(spin == Catch::Approx(meta.spin_length));

    file.close();
    std::filesystem::remove(path);
}

TEST_CASE("write_record + read_record round-trip a dynamical record",
          "[dssf][hdf5][p2-3][io]") {
    const auto path = make_temp_h5("dyn");
    const auto meta = make_dyn_meta();
    ed::dssf::ensure_metadata(path, meta);

    ed::dssf::Record rec;
    rec.method        = ed::dssf::DSSFMethod::DYNAMICAL_THERMAL;
    rec.operator_name = "SpSm_Q_pi_0_0";
    rec.temperature   = 0.5;
    rec.total_samples = 17;
    rec.frequencies   = {-1.0, 0.0, 1.0};
    rec.spectral_real = { 0.10, 0.20, 0.30};
    rec.spectral_imag = {-0.01, 0.02,-0.03};
    rec.error_real    = { 0.001, 0.002, 0.003};
    rec.error_imag    = { 0.0001,0.0002,0.0003};

    ed::dssf::write_record(path, rec);

    auto [out_meta, out_rec] = ed::dssf::read_record(path, rec.operator_name);

    REQUIRE(out_meta.method      == meta.method);
    REQUIRE(out_meta.num_sites   == meta.num_sites);
    REQUIRE(out_meta.spin_length == Catch::Approx(meta.spin_length));

    REQUIRE(out_rec.operator_name == rec.operator_name);
    REQUIRE(out_rec.method        == rec.method);
    REQUIRE(out_rec.temperature   == Catch::Approx(rec.temperature));
    REQUIRE(out_rec.total_samples == rec.total_samples);
    REQUIRE(out_rec.frequencies   == rec.frequencies);
    REQUIRE(out_rec.spectral_real == rec.spectral_real);
    REQUIRE(out_rec.spectral_imag == rec.spectral_imag);
    REQUIRE(out_rec.error_real    == rec.error_real);
    REQUIRE(out_rec.error_imag    == rec.error_imag);

    REQUIRE(out_rec.temperatures.empty());
    REQUIRE(out_rec.expectation.empty());

    std::filesystem::remove(path);
}

TEST_CASE("write_record + read_record round-trip a static record",
          "[dssf][hdf5][p2-3][io]") {
    const auto path = make_temp_h5("static");
    auto meta = make_dyn_meta();
    meta.method = ed::dssf::DSSFMethod::STATIC_THERMAL;
    ed::dssf::ensure_metadata(path, meta);

    ed::dssf::Record rec;
    rec.method               = ed::dssf::DSSFMethod::STATIC_THERMAL;
    rec.operator_name        = "SzSz_Q_pi_pi_0";
    rec.temperature          = 0.0;
    rec.total_samples        = 42;
    rec.temperatures         = {0.1, 0.5, 1.0};
    rec.expectation          = {0.10, 0.05, 0.01};
    rec.expectation_error    = {0.001, 0.002, 0.003};
    rec.variance             = {0.30, 0.20, 0.10};
    rec.variance_error       = {0.01, 0.02, 0.03};
    rec.susceptibility       = {1.0, 0.5, 0.1};
    rec.susceptibility_error = {0.1, 0.05, 0.01};

    ed::dssf::write_record(path, rec);

    auto [out_meta, out_rec] = ed::dssf::read_record(path, rec.operator_name);

    REQUIRE(out_rec.temperatures         == rec.temperatures);
    REQUIRE(out_rec.expectation          == rec.expectation);
    REQUIRE(out_rec.expectation_error    == rec.expectation_error);
    REQUIRE(out_rec.variance             == rec.variance);
    REQUIRE(out_rec.variance_error       == rec.variance_error);
    REQUIRE(out_rec.susceptibility       == rec.susceptibility);
    REQUIRE(out_rec.susceptibility_error == rec.susceptibility_error);

    REQUIRE(out_rec.frequencies.empty());
    REQUIRE(out_rec.spectral_real.empty());

    std::filesystem::remove(path);
}

TEST_CASE("write_record supports slash-separated operator names",
          "[dssf][hdf5][p2-3][io]") {
    const auto path = make_temp_h5("nested");
    ed::dssf::ensure_metadata(path, make_dyn_meta());

    ed::dssf::Record rec;
    rec.method        = ed::dssf::DSSFMethod::DYNAMICAL_THERMAL;
    rec.operator_name = "S+S-_NSF/Q_3/T_0.250000";
    rec.temperature   = 0.25;
    rec.total_samples = 7;
    rec.frequencies   = {0.0, 0.5, 1.0};
    rec.spectral_real = {0.0, 0.5, 0.25};
    rec.spectral_imag = {0.0,-0.5,-0.25};
    rec.error_real    = {1e-3, 1e-3, 1e-3};
    rec.error_imag    = {1e-3, 1e-3, 1e-3};

    ed::dssf::write_record(path, rec);

    H5::H5File file(path, H5F_ACC_RDONLY);
    REQUIRE(file.nameExists("/dssf/S+S-_NSF/Q_3/T_0.250000"));
    file.close();

    auto [out_meta, out_rec] = ed::dssf::read_record(path, rec.operator_name);
    REQUIRE(out_rec.frequencies   == rec.frequencies);
    REQUIRE(out_rec.spectral_real == rec.spectral_real);

    std::filesystem::remove(path);
}

TEST_CASE("write_record rejects mismatched array lengths",
          "[dssf][hdf5][p2-3][io]") {
    const auto path = make_temp_h5("mismatch");
    ed::dssf::ensure_metadata(path, make_dyn_meta());

    ed::dssf::Record rec;
    rec.method        = ed::dssf::DSSFMethod::DYNAMICAL_THERMAL;
    rec.operator_name = "bad";
    rec.frequencies   = {0.0, 1.0};
    rec.spectral_real = {0.0, 1.0, 2.0};         // length mismatch
    rec.spectral_imag = {0.0, 0.0};
    rec.error_real    = {0.0, 0.0};
    rec.error_imag    = {0.0, 0.0};

    REQUIRE_THROWS_AS(ed::dssf::write_record(path, rec),
                      std::invalid_argument);
    std::filesystem::remove(path);
}

TEST_CASE("write_record rejects empty operator_name",
          "[dssf][hdf5][p2-3][io]") {
    const auto path = make_temp_h5("empty_name");
    ed::dssf::ensure_metadata(path, make_dyn_meta());

    ed::dssf::Record rec;
    rec.operator_name = "";
    REQUIRE_THROWS_AS(ed::dssf::write_record(path, rec),
                      std::invalid_argument);
    std::filesystem::remove(path);
}

TEST_CASE("read_record refuses missing /dssf root",
          "[dssf][hdf5][p2-3][io]") {
    const auto path = make_temp_h5("noroot");
    {
        H5::H5File file(path, H5F_ACC_TRUNC);   // empty file -- no /dssf
        file.close();
    }
    REQUIRE_THROWS_AS(ed::dssf::read_record(path, "anything"),
                      std::invalid_argument);
    std::filesystem::remove(path);
}
