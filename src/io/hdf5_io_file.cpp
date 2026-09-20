// =============================================================================
// src/io/hdf5_io_file.cpp
//
// HDF5IO: low-level helpers and file management -- compression/chunking
// tunables, adaptive dataset property lists, group creation, file
// create/open/exists, the generic array reader/writer and the per-rank file
// naming helpers. Declarations live in include/ed/core/hdf5_io.h.
// =============================================================================

#include <ed/core/hdf5_io.h>

#include <ed/config/env_registry.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>

int HDF5IO::hdf5_compression_level() {
    static const int level = []() {
        const char* env = ed::env::raw("ED_HDF5_COMPRESSION_LEVEL");
        if (!env) return 4;
        try { return std::clamp(std::stoi(env), 0, 9); }
        catch (...) { return 4; }
    }();
    return level;
}

size_t HDF5IO::hdf5_chunk_target_bytes() {
    static const size_t bytes = []() -> size_t {
        const char* env = ed::env::raw("ED_HDF5_CHUNK_TARGET_BYTES");
        if (!env) return 256 * 1024;
        try {
            long long v = std::stoll(env);
            if (v < 16 * 1024) return 16 * 1024;
            if (v > 16ll * 1024 * 1024) return 16 * 1024 * 1024;
            return static_cast<size_t>(v);
        } catch (...) { return 256 * 1024; }
    }();
    return bytes;
}

bool HDF5IO::hdf5_shuffle_enabled() {
    static const bool on = []() {
        return ed::env::flag("ED_HDF5_SHUFFLE", true);
    }();
    return on;
}

H5::DSetCreatPropList HDF5IO::makeAdaptiveDsetProps(
    const std::vector<hsize_t>& dims,
    size_t element_size,
    bool last_dim_full_chunk)
{
    H5::DSetCreatPropList plist;
    if (dims.empty()) return plist;

    // Compute chunk shape.
    std::vector<hsize_t> chunk(dims.size());
    if (last_dim_full_chunk && dims.size() >= 2) {
        // Tabular: chunk = (rows_per_chunk, full_remaining_dims)
        hsize_t row_size_elems = 1;
        for (size_t i = 1; i < dims.size(); ++i) {
            chunk[i] = std::max<hsize_t>(1, dims[i]);
            row_size_elems *= chunk[i];
        }
        const size_t row_bytes = static_cast<size_t>(row_size_elems) * element_size;
        hsize_t rows_per_chunk = row_bytes == 0 ? 1
            : std::max<hsize_t>(1, hdf5_chunk_target_bytes() / std::max<size_t>(row_bytes, 1));
        // Cap by total rows.
        rows_per_chunk = std::min<hsize_t>(rows_per_chunk, std::max<hsize_t>(dims[0], 1));
        chunk[0] = rows_per_chunk;
    } else {
        // 1D or last_dim_full_chunk=false: pick a single chunk size by bytes.
        hsize_t total = 1;
        for (auto d : dims) total *= std::max<hsize_t>(d, 1);
        const size_t target_elems = std::max<size_t>(
            hdf5_chunk_target_bytes() / std::max<size_t>(element_size, 1), 1);
        for (size_t i = 0; i < dims.size(); ++i) {
            chunk[i] = std::max<hsize_t>(1, std::min<hsize_t>(dims[i], target_elems));
        }
        (void)total;
    }

    plist.setChunk(static_cast<int>(chunk.size()), chunk.data());

    const int level = hdf5_compression_level();
    if (level > 0) {
        // Shuffle improves compression of float/double payloads by
        // co-locating the bytes that vary most slowly. Cheap; standard.
        if (hdf5_shuffle_enabled()) plist.setShuffle();
        plist.setDeflate(level);
    }
    return plist;
}

void HDF5IO::ensureGroupExists(H5::H5File& file, const std::string& group_path) {
    if (group_path.empty() || group_path == "/") return;

    // Split path into components
    std::vector<std::string> components;
    std::string current_path;
    std::istringstream ss(group_path);
    std::string component;

    while (std::getline(ss, component, '/')) {
        if (!component.empty()) {
            components.push_back(component);
        }
    }

    // Create each component if it doesn't exist
    for (const auto& comp : components) {
        current_path += "/" + comp;
        if (!file.nameExists(current_path)) {
            file.createGroup(current_path);
        }
    }
}

void HDF5IO::ensureStandardGroups(H5::H5File& file) {
    // List of standard groups for ED results
    const std::vector<std::string> standard_groups = {
        "/eigendata",
        "/thermodynamics",
        "/correlations",
        "/dynamical",
        "/dynamical/samples",
        "/ftlm",
        "/ftlm/samples",
        "/ftlm/averaged",
        "/tpq",
        "/tpq/samples",
        "/tpq/averaged"
    };

    for (const auto& group : standard_groups) {
        ensureGroupExists(file, group);
    }
}

bool HDF5IO::isDisabledOutputPath(const std::string& s) {
    if (s.empty()) return true;
    // Exact match on the conventional sentinel.
    if (s == "/dev/null") return true;
    // ``/dev/null/<anything>`` -- e.g. createOrOpenFile concatenates
    // ``dir + "/" + filename`` so a disabled dir yields a disabled path.
    constexpr const char* kDevNull = "/dev/null/";
    if (s.size() > 10 && s.compare(0, 10, kDevNull) == 0) return true;
    return false;
}

std::string HDF5IO::createOrOpenFile(const std::string& directory, 
                                     const std::string& filename) {
    if (isDisabledOutputPath(directory)) {
        return std::string("/dev/null");
    }
    std::string filepath = directory + "/" + filename;

    try {
        if (fileExists(filepath)) {
            // SAFE: Open existing file in read/write mode (preserves all existing data)
            H5::H5File file(filepath, H5F_ACC_RDWR);

            // Ensure standard groups exist (creates only if missing)
            ensureStandardGroups(file);

            file.close();
            std::cout << "Opened existing HDF5 results file: " << filepath << std::endl;
            return filepath;
        }

        // Create new file only if it doesn't exist
        H5::H5File file(filepath, H5F_ACC_TRUNC);

        // Create standard groups
        ensureStandardGroups(file);

        file.close();
        std::cout << "Created new HDF5 results file: " << filepath << std::endl;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to create/open HDF5 file: " + std::string(e.getCDetailMsg()));
    }

    return filepath;
}

std::string HDF5IO::forceCreateFile(const std::string& directory, 
                                    const std::string& filename) {
    if (isDisabledOutputPath(directory)) {
        return std::string("/dev/null");
    }
    std::string filepath = directory + "/" + filename;

    try {
        // Force truncate - WARNING: deletes existing data
        H5::H5File file(filepath, H5F_ACC_TRUNC);
        ensureStandardGroups(file);
        file.close();
        std::cout << "Created new HDF5 results file (truncated): " << filepath << std::endl;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to create HDF5 file: " + std::string(e.getCDetailMsg()));
    }

    return filepath;
}

bool HDF5IO::fileExists(const std::string& filepath) {
    // Phase 6.1: short-circuit on the disabled-output sentinel. Without
    // this, /dev/null would pass std::filesystem::exists (it is a real
    // device node on Linux) and then fail noisily inside the H5::H5File
    // ctor below.
    if (isDisabledOutputPath(filepath)) return false;
    // First check if file exists on filesystem to avoid HDF5 error output
    if (!std::filesystem::exists(filepath)) {
        return false;
    }

    // Temporarily disable HDF5 error printing
    H5E_auto2_t old_func;
    void* old_client_data;
    H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
    H5Eset_auto2(H5E_DEFAULT, NULL, NULL);

    bool result = false;
    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);
        file.close();
        result = true;
    } catch (H5::Exception& e) {
        result = false;
    }

    // Re-enable error printing
    H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    return result;
}

void HDF5IO::saveArray(const std::string& filepath,
                     const std::string& dataset_path,
                     const std::vector<double>& data,
                     const std::map<std::string, std::string>& string_attrs,
                     const std::map<std::string, double>& double_attrs) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        if (file.nameExists(dataset_path)) {
            file.unlink(dataset_path);
        }

        hsize_t dims[1] = {data.size()};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = file.createDataSet(dataset_path,
                                                 H5::PredType::NATIVE_DOUBLE,
                                                 dataspace);
        dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);

        // Add string attributes
        for (const auto& [key, value] : string_attrs) {
            H5::StrType str_type(H5::PredType::C_S1, value.size() + 1);
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::Attribute attr = dataset.createAttribute(key, str_type, attr_space);
            attr.write(str_type, value.c_str());
            attr.close();
        }

        // Add double attributes
        for (const auto& [key, value] : double_attrs) {
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::Attribute attr = dataset.createAttribute(key,
                                                         H5::PredType::NATIVE_DOUBLE,
                                                         attr_space);
            attr.write(H5::PredType::NATIVE_DOUBLE, &value);
            attr.close();
        }

        dataset.close();
        file.close();
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to save array: " + std::string(e.getCDetailMsg()));
    }
}

std::vector<double> HDF5IO::loadArray(const std::string& filepath,
                                    const std::string& dataset_path) {
    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(dataset_path);
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims);

        std::vector<double> data(dims[0]);
        dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);

        dataset.close();
        file.close();

        return data;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to load array: " + std::string(e.getCDetailMsg()));
    }
}

std::string HDF5IO::getPerRankFilePath(const std::string& directory,
                                       int rank,
                                       const std::string& filename) {
    if (isDisabledOutputPath(directory)) {
        return std::string("/dev/null");
    }
    // Extract base name and extension
    size_t dot_pos = filename.rfind('.');
    std::string base = (dot_pos != std::string::npos) ? filename.substr(0, dot_pos) : filename;
    std::string ext = (dot_pos != std::string::npos) ? filename.substr(dot_pos) : "";

    return directory + "/" + base + "_rank" + std::to_string(rank) + ext;
}

std::string HDF5IO::createPerRankFile(const std::string& directory,
                                      int rank,
                                      const std::string& filename) {
    if (isDisabledOutputPath(directory)) {
        return std::string("/dev/null");
    }
    std::string filepath = getPerRankFilePath(directory, rank, filename);

    try {
        // SAFE: Check if file already exists
        if (fileExists(filepath)) {
            // Open existing file in read/write mode (preserve existing data)
            H5::H5File file(filepath, H5F_ACC_RDWR);
            ensureStandardGroups(file);
            file.close();
            std::cout << "Opened existing per-rank HDF5 file: " << filepath << std::endl;
            return filepath;
        }

        // Create new file only if it doesn't exist
        H5::H5File file(filepath, H5F_ACC_TRUNC);
        ensureStandardGroups(file);
        file.close();
        std::cout << "Created per-rank HDF5 file: " << filepath << std::endl;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to create/open per-rank HDF5 file: " + std::string(e.getCDetailMsg()));
    }

    return filepath;
}
