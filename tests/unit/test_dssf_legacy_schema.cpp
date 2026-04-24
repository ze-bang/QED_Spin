// =============================================================================
// test_dssf_legacy_schema (Catch2 v3, P2.3 / DSSF PR-D lock-down tests)
//
// Pin the *legacy* HDF5 schemas that the existing CLI workflows
// (`compute_dynamical_response_workflow`, `compute_static_response_workflow`)
// and the legacy `TPQ_DSSF` binary write today, BEFORE the unified
// `/dssf/...` schema lands in P2.3.
//
// What we lock down here (one test per schema family):
//
//   1. `/dynamical/<op>/{frequencies, spectral_real, spectral_imag,
//                        error_real, error_imag}` + attrs
//      `total_samples`, `temperature`. Written by
//      `HDF5IO::saveDynamicalResponseFull`.
//
//   2. `/correlations/<op>/{temperatures, expectation, expectation_error,
//                           variance, variance_error,
//                           susceptibility, susceptibility_error}`
//      + attr `total_samples`. Written by
//      `HDF5IO::saveStaticResponse`.
//
//   3. The "intermediate-group autocreation" behaviour for slash-separated
//      operator names (e.g. `S+S-/Q_0_0_0_T0.500000`). The legacy
//      `saveDynamicalResponseFull` walks the path and creates missing
//      groups; users have come to rely on this for their nested op names.
//
// These tests must keep passing through P2.3 (when the unified schema is
// added alongside) and P2.4 (when CLI workflows start writing the
// unified schema by default). They should only be deleted when P2.14
// retires the legacy writers entirely.
// =============================================================================

#include "common/catch2_harness.h"

#include <catch2/catch_approx.hpp>

#include <ed/core/hdf5_io.h>

#include <H5Cpp.h>
#include <unistd.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace {

std::string make_temp_h5(const std::string& tag) {
    auto dir = std::filesystem::temp_directory_path() /
               ("ed_dssf_legacy_" + tag + "_" +
                std::to_string(::getpid()));
    std::filesystem::create_directories(dir);
    auto path = dir / "out.h5";
    if (std::filesystem::exists(path)) {
        std::filesystem::remove(path);
    }
    // Touch an empty HDF5 file so saveStaticResponse / saveDynamicalResponseFull
    // (which open with H5F_ACC_RDWR) have something to extend.
    H5::H5File file(path.string(), H5F_ACC_TRUNC);
    file.close();
    return path.string();
}

std::vector<double> read_dataset(const H5::H5File& file,
                                 const std::string& path) {
    H5::DataSet ds = file.openDataSet(path);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[1] = {0};
    sp.getSimpleExtentDims(dims);
    std::vector<double> out(dims[0]);
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    return out;
}

double read_double_attr(const H5::H5File& file,
                        const std::string& group_path,
                        const std::string& attr_name) {
    H5::Group g = file.openGroup(group_path);
    H5::Attribute a = g.openAttribute(attr_name);
    double value{};
    a.read(H5::PredType::NATIVE_DOUBLE, &value);
    return value;
}

std::uint64_t read_u64_attr(const H5::H5File& file,
                            const std::string& group_path,
                            const std::string& attr_name) {
    H5::Group g = file.openGroup(group_path);
    H5::Attribute a = g.openAttribute(attr_name);
    std::uint64_t value{};
    a.read(H5::PredType::NATIVE_UINT64, &value);
    return value;
}

} // namespace

TEST_CASE("legacy /dynamical/<op>/ schema is preserved bit-for-bit",
          "[dssf][hdf5][p2-3][lockdown]") {
    const auto path = make_temp_h5("dyn");

    const std::string op_name = "SpSm_Q_pi_0_0";
    const std::vector<double> freqs   = {-1.0, 0.0, 1.0};
    const std::vector<double> sp_real = { 0.10, 0.20, 0.30};
    const std::vector<double> sp_imag = {-0.01, 0.02,-0.03};
    const std::vector<double> er_real = { 0.001, 0.002, 0.003};
    const std::vector<double> er_imag = { 0.0001,0.0002,0.0003};
    const std::uint64_t total_samples = 17;
    const double temperature          = 0.5;

    HDF5IO::saveDynamicalResponseFull(
        path, op_name, freqs, sp_real, sp_imag, er_real, er_imag,
        total_samples, temperature);

    H5::H5File file(path, H5F_ACC_RDONLY);
    const std::string base = "/dynamical/" + op_name;

    REQUIRE(read_dataset(file, base + "/frequencies")   == freqs);
    REQUIRE(read_dataset(file, base + "/spectral_real") == sp_real);
    REQUIRE(read_dataset(file, base + "/spectral_imag") == sp_imag);
    REQUIRE(read_dataset(file, base + "/error_real")    == er_real);
    REQUIRE(read_dataset(file, base + "/error_imag")    == er_imag);

    REQUIRE(read_u64_attr(file, base, "total_samples") == total_samples);
    REQUIRE(read_double_attr(file, base, "temperature") ==
            Catch::Approx(temperature));

    file.close();
    std::filesystem::remove(path);
}

TEST_CASE("legacy /correlations/<op>/ schema is preserved bit-for-bit",
          "[dssf][hdf5][p2-3][lockdown]") {
    const auto path = make_temp_h5("corr");

    const std::string op_name = "SzSz_Q_pi_pi_0";
    const std::vector<double> temps    = {0.1, 0.5, 1.0};
    const std::vector<double> exp_v    = {0.10, 0.05, 0.01};
    const std::vector<double> exp_err  = {0.001, 0.002, 0.003};
    const std::vector<double> var_v    = {0.30, 0.20, 0.10};
    const std::vector<double> var_err  = {0.01, 0.02, 0.03};
    const std::vector<double> sus_v    = {1.0, 0.5, 0.1};
    const std::vector<double> sus_err  = {0.1, 0.05, 0.01};
    const std::uint64_t total_samples  = 42;

    HDF5IO::saveStaticResponse(
        path, op_name, temps, exp_v, exp_err,
        var_v, var_err, sus_v, sus_err, total_samples);

    H5::H5File file(path, H5F_ACC_RDONLY);
    const std::string base = "/correlations/" + op_name;

    REQUIRE(read_dataset(file, base + "/temperatures")          == temps);
    REQUIRE(read_dataset(file, base + "/expectation")           == exp_v);
    REQUIRE(read_dataset(file, base + "/expectation_error")     == exp_err);
    REQUIRE(read_dataset(file, base + "/variance")              == var_v);
    REQUIRE(read_dataset(file, base + "/variance_error")        == var_err);
    REQUIRE(read_dataset(file, base + "/susceptibility")        == sus_v);
    REQUIRE(read_dataset(file, base + "/susceptibility_error")  == sus_err);

    REQUIRE(read_u64_attr(file, base, "total_samples") == total_samples);

    file.close();
    std::filesystem::remove(path);
}

TEST_CASE("legacy saveDynamicalResponseFull autocreates intermediate groups",
          "[dssf][hdf5][p2-3][lockdown]") {
    const auto path = make_temp_h5("nested");

    // The legacy CLI uses slash-separated names like
    // "<op>/Q_<idx>_T<temp>" and relies on the writer to create the
    // intermediate group. Lock that behaviour down so we can't
    // accidentally break j3_h0_scan post-processing scripts in P2.3.
    const std::string nested_name = "S+S-_NSF/Q_3/T_0.250000";
    const std::vector<double> freqs   = {0.0, 0.5, 1.0};
    const std::vector<double> sp_real = {0.0, 0.5, 0.25};
    const std::vector<double> sp_imag = {0.0,-0.5,-0.25};
    const std::vector<double> er      = {1e-3,1e-3,1e-3};

    HDF5IO::saveDynamicalResponseFull(
        path, nested_name, freqs, sp_real, sp_imag, er, er,
        7u, 0.25);

    H5::H5File file(path, H5F_ACC_RDONLY);
    REQUIRE(file.nameExists("/dynamical"));
    REQUIRE(file.nameExists("/dynamical/S+S-_NSF"));
    REQUIRE(file.nameExists("/dynamical/S+S-_NSF/Q_3"));
    REQUIRE(file.nameExists("/dynamical/S+S-_NSF/Q_3/T_0.250000"));
    REQUIRE(read_dataset(file,
        "/dynamical/S+S-_NSF/Q_3/T_0.250000/frequencies") == freqs);

    file.close();
    std::filesystem::remove(path);
}
