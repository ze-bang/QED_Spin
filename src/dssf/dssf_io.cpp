// =============================================================================
// src/dssf/dssf_io.cpp
//
// Implementation of the unified `/dssf/...` HDF5 schema declared in
// `include/ed/dssf/dssf_io.h` (P2.3 / DSSF PR-D, audit §3.10).
//
// Lives in `ed_dssf` (NOT `ed_cli`) so the schema is reachable from any
// callsite that already has `ed_dssf` linked -- including future Python
// bindings (P2.x) and downstream collaborator code via `find_package(ED)`.
// =============================================================================

#include <ed/dssf/dssf_io.h>

#include <ed/core/hdf5_io.h>  // Phase 6.1: HDF5IO::isDisabledOutputPath

#include <H5Cpp.h>

#include <algorithm>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ed::dssf {

namespace {

constexpr const char* kRoot                 = "/dssf";
constexpr const char* kAttrSchemaVersion    = "schema_version";
constexpr const char* kAttrMethod           = "method";
constexpr const char* kAttrNumSites         = "num_sites";
constexpr const char* kAttrSpinLength       = "spin_length";
constexpr const char* kAttrCreatedAt        = "created_at";
constexpr const char* kAttrTemperature      = "temperature";
constexpr const char* kAttrTotalSamples     = "total_samples";

// ----------------------------------------------------------------------------
// Tiny HDF5 helpers. Kept private to the TU so we don't accidentally grow a
// parallel HDF5 utility library next to ed/core/hdf5_io.h.
// ----------------------------------------------------------------------------

void open_or_create_file(const std::string& path, H5::H5File& out) {
    try {
        out = H5::H5File(path, H5F_ACC_RDWR);
    } catch (const H5::FileIException&) {
        out = H5::H5File(path, H5F_ACC_TRUNC);
    }
}

void ensure_group(H5::H5File& file, const std::string& path) {
    // Walk slash-separated components and create any that are missing.
    std::string cur;
    std::size_t pos = 0;
    while (pos < path.size()) {
        std::size_t next = path.find('/', pos + 1);
        if (next == std::string::npos) next = path.size();
        cur = path.substr(0, next);
        if (cur.empty() || cur == "/") { pos = next; continue; }

        bool exists = false;
        H5E_BEGIN_TRY {
            try { exists = file.nameExists(cur); }
            catch (...) { exists = false; }
        } H5E_END_TRY;

        if (!exists) {
            file.createGroup(cur);
        }
        pos = next;
    }
}

void write_double_array(H5::Group& parent, const std::string& name,
                        const std::vector<double>& data) {
    if (data.empty()) return;
    if (parent.nameExists(name)) {
        parent.unlink(name);
    }
    hsize_t dims[1] = {data.size()};
    H5::DataSpace space(1, dims);
    H5::DataSet ds = parent.createDataSet(name,
                                          H5::PredType::NATIVE_DOUBLE,
                                          space);
    ds.write(data.data(), H5::PredType::NATIVE_DOUBLE);
}

std::vector<double> read_double_array(H5::Group& parent,
                                      const std::string& name) {
    if (!parent.nameExists(name)) return {};
    H5::DataSet ds = parent.openDataSet(name);
    H5::DataSpace sp = ds.getSpace();
    hsize_t dims[1] = {0};
    sp.getSimpleExtentDims(dims);
    std::vector<double> out(dims[0]);
    ds.read(out.data(), H5::PredType::NATIVE_DOUBLE);
    return out;
}

template <typename T>
void write_scalar_attr(H5::H5Object& target, const std::string& name,
                       const H5::DataType& type, const T& value) {
    if (target.attrExists(name)) {
        target.removeAttr(name);
    }
    H5::DataSpace space(H5S_SCALAR);
    H5::Attribute a = target.createAttribute(name, type, space);
    a.write(type, &value);
}

void write_string_attr(H5::H5Object& target, const std::string& name,
                       const std::string& value) {
    if (target.attrExists(name)) {
        target.removeAttr(name);
    }
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace space(H5S_SCALAR);
    H5::Attribute a = target.createAttribute(name, str_type, space);
    const char* c = value.c_str();
    a.write(str_type, &c);
}

template <typename T>
T read_scalar_attr(H5::H5Object& target, const std::string& name,
                   const H5::DataType& type) {
    H5::Attribute a = target.openAttribute(name);
    T value{};
    a.read(type, &value);
    return value;
}

std::string read_string_attr(H5::H5Object& target, const std::string& name) {
    H5::Attribute a = target.openAttribute(name);
    H5::StrType str_type = a.getStrType();
    std::string out;
    a.read(str_type, out);
    return out;
}

void require_matching_lengths(const std::vector<const std::vector<double>*>& xs,
                              const char* what) {
    std::size_t expected = 0;
    bool seen = false;
    for (auto* x : xs) {
        if (x->empty()) continue;
        if (!seen) { expected = x->size(); seen = true; continue; }
        if (x->size() != expected) {
            throw std::invalid_argument(
                std::string("ed::dssf::write_record: mismatched lengths in ")
                + what);
        }
    }
}

} // namespace

void ensure_metadata(const std::string& filepath, const Metadata& meta) {
    // Phase 6.1: respect the centralised "disable HDF5 output" sentinel
    // so DSSF writers behave consistently with HDF5IO::saveDynamicalResponse
    // and friends when the caller passes "" or "/dev/null".
    if (HDF5IO::isDisabledOutputPath(filepath)) return;
    H5::H5File file;
    open_or_create_file(filepath, file);
    ensure_group(file, kRoot);

    H5::Group root = file.openGroup(kRoot);

    write_scalar_attr(root, kAttrSchemaVersion,
                      H5::PredType::NATIVE_UINT32,
                      kSchemaVersion);
    write_string_attr(root, kAttrMethod, to_string(meta.method));
    write_scalar_attr(root, kAttrNumSites,
                      H5::PredType::NATIVE_UINT64,
                      meta.num_sites);
    write_scalar_attr(root, kAttrSpinLength,
                      H5::PredType::NATIVE_DOUBLE,
                      meta.spin_length);
    if (!meta.created_at.empty()) {
        write_string_attr(root, kAttrCreatedAt, meta.created_at);
    }

    root.close();
    file.close();
}

void write_record(const std::string& filepath, const Record& record) {
    if (record.operator_name.empty()) {
        throw std::invalid_argument(
            "ed::dssf::write_record: record.operator_name must be non-empty");
    }
    // Phase 6.1: see comment in ensure_metadata; same disabled-path
    // short-circuit so empty / /dev/null filepaths are a silent no-op.
    if (HDF5IO::isDisabledOutputPath(filepath)) return;

    require_matching_lengths(
        {&record.frequencies, &record.spectral_real, &record.spectral_imag,
         &record.error_real,  &record.error_imag},
        "frequency-domain arrays (frequencies / spectral_real / "
        "spectral_imag / error_real / error_imag)");

    require_matching_lengths(
        {&record.temperatures, &record.expectation, &record.expectation_error,
         &record.variance,     &record.variance_error,
         &record.susceptibility, &record.susceptibility_error},
        "time-domain arrays (temperatures / expectation / "
        "expectation_error / variance{,_error} / susceptibility{,_error})");

    H5::H5File file(filepath, H5F_ACC_RDWR);

    if (!file.nameExists(kRoot)) {
        file.close();
        throw std::invalid_argument(
            "ed::dssf::write_record: file has no /dssf root. Call "
            "ensure_metadata(filepath, meta) before write_record.");
    }

    const std::string base = std::string(kRoot) + "/" + record.operator_name;
    ensure_group(file, base);

    H5::Group group = file.openGroup(base);

    write_string_attr(group, kAttrMethod, to_string(record.method));
    write_scalar_attr(group, kAttrTemperature,
                      H5::PredType::NATIVE_DOUBLE,
                      record.temperature);
    write_scalar_attr(group, kAttrTotalSamples,
                      H5::PredType::NATIVE_UINT64,
                      record.total_samples);

    write_double_array(group, "frequencies",          record.frequencies);
    write_double_array(group, "spectral_real",        record.spectral_real);
    write_double_array(group, "spectral_imag",        record.spectral_imag);
    write_double_array(group, "error_real",           record.error_real);
    write_double_array(group, "error_imag",           record.error_imag);

    write_double_array(group, "temperatures",         record.temperatures);
    write_double_array(group, "expectation",          record.expectation);
    write_double_array(group, "expectation_error",    record.expectation_error);
    write_double_array(group, "variance",             record.variance);
    write_double_array(group, "variance_error",       record.variance_error);
    write_double_array(group, "susceptibility",       record.susceptibility);
    write_double_array(group, "susceptibility_error", record.susceptibility_error);

    group.close();
    file.close();
}

std::pair<Metadata, Record>
read_record(const std::string& filepath, const std::string& operator_name) {
    H5::H5File file(filepath, H5F_ACC_RDONLY);

    if (!file.nameExists(kRoot)) {
        file.close();
        throw std::invalid_argument(
            "ed::dssf::read_record: file has no /dssf root.");
    }

    H5::Group root = file.openGroup(kRoot);

    const auto schema_version =
        read_scalar_attr<std::uint32_t>(root, kAttrSchemaVersion,
                                        H5::PredType::NATIVE_UINT32);
    if (schema_version != kSchemaVersion) {
        throw std::invalid_argument(
            "ed::dssf::read_record: unsupported /dssf schema_version=" +
            std::to_string(schema_version) + " (expected " +
            std::to_string(kSchemaVersion) + ")");
    }

    Metadata meta;
    meta.method      = method_from_string(read_string_attr(root, kAttrMethod));
    meta.num_sites   = read_scalar_attr<std::uint64_t>(
        root, kAttrNumSites, H5::PredType::NATIVE_UINT64);
    meta.spin_length = read_scalar_attr<double>(
        root, kAttrSpinLength, H5::PredType::NATIVE_DOUBLE);
    if (root.attrExists(kAttrCreatedAt)) {
        meta.created_at = read_string_attr(root, kAttrCreatedAt);
    }

    const std::string base = std::string(kRoot) + "/" + operator_name;
    if (!file.nameExists(base)) {
        root.close(); file.close();
        throw std::invalid_argument(
            "ed::dssf::read_record: no record at " + base);
    }

    H5::Group group = file.openGroup(base);

    Record record;
    record.operator_name = operator_name;
    record.method        = method_from_string(read_string_attr(group, kAttrMethod));
    record.temperature   = read_scalar_attr<double>(
        group, kAttrTemperature, H5::PredType::NATIVE_DOUBLE);
    record.total_samples = read_scalar_attr<std::uint64_t>(
        group, kAttrTotalSamples, H5::PredType::NATIVE_UINT64);

    record.frequencies          = read_double_array(group, "frequencies");
    record.spectral_real        = read_double_array(group, "spectral_real");
    record.spectral_imag        = read_double_array(group, "spectral_imag");
    record.error_real           = read_double_array(group, "error_real");
    record.error_imag           = read_double_array(group, "error_imag");

    record.temperatures         = read_double_array(group, "temperatures");
    record.expectation          = read_double_array(group, "expectation");
    record.expectation_error    = read_double_array(group, "expectation_error");
    record.variance             = read_double_array(group, "variance");
    record.variance_error       = read_double_array(group, "variance_error");
    record.susceptibility       = read_double_array(group, "susceptibility");
    record.susceptibility_error = read_double_array(group, "susceptibility_error");

    group.close();
    root.close();
    file.close();

    return {meta, record};
}

} // namespace ed::dssf
