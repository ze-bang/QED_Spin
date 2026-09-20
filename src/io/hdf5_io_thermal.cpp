// =============================================================================
// src/io/hdf5_io_thermal.cpp
//
// HDF5IO: FTLM/LTLM thermal output -- averaged thermodynamics, static and
// dynamical response, per-sample FTLM datasets and time-correlation data.
// Declarations live in include/ed/core/hdf5_io.h.
// =============================================================================

#include <ed/core/hdf5_io.h>

#include <cstdint>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

void HDF5IO::saveFTLMThermodynamics(
    const std::string& filepath,
    const std::vector<double>& temperatures,
    const std::vector<double>& energy,
    const std::vector<double>& energy_error,
    const std::vector<double>& specific_heat,
    const std::vector<double>& specific_heat_error,
    const std::vector<double>& entropy,
    const std::vector<double>& entropy_error,
    const std::vector<double>& free_energy,
    const std::vector<double>& free_energy_error,
    uint64_t total_samples,
    const std::string& method
) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string base_path = "/ftlm/averaged";

        // Ensure group exists
        if (!file.nameExists(base_path)) {
            file.createGroup(base_path);
        }

        // Helper lambda to save an array
        auto saveDataset = [&](const std::string& name, const std::vector<double>& data) {
            std::string dataset_name = base_path + "/" + name;
            if (file.nameExists(dataset_name)) {
                file.unlink(dataset_name);
            }
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(dataset_name,
                                                     H5::PredType::NATIVE_DOUBLE,
                                                     dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            dataset.close();
        };

        // Save all arrays
        saveDataset("temperatures", temperatures);
        saveDataset("energy", energy);
        saveDataset("energy_error", energy_error);
        saveDataset("specific_heat", specific_heat);
        saveDataset("specific_heat_error", specific_heat_error);
        saveDataset("entropy", entropy);
        saveDataset("entropy_error", entropy_error);
        saveDataset("free_energy", free_energy);
        saveDataset("free_energy_error", free_energy_error);

        // Save metadata as attributes on the group
        H5::Group group = file.openGroup(base_path);
        H5::DataSpace attr_space(H5S_SCALAR);

        // Total samples attribute
        if (group.attrExists("total_samples")) {
            group.removeAttr("total_samples");
        }
        H5::Attribute samples_attr = group.createAttribute("total_samples",
                                                           H5::PredType::NATIVE_UINT64,
                                                           attr_space);
        samples_attr.write(H5::PredType::NATIVE_UINT64, &total_samples);
        samples_attr.close();

        // Method attribute
        if (group.attrExists("method")) {
            group.removeAttr("method");
        }
        H5::StrType str_type(H5::PredType::C_S1, method.size() + 1);
        H5::Attribute method_attr = group.createAttribute("method", str_type, attr_space);
        method_attr.write(str_type, method.c_str());
        method_attr.close();

        group.close();
        file.close();

        std::cout << "Saved " << method << " thermodynamic results to HDF5" << std::endl;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to save FTLM thermodynamics: " +
                               std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::saveStaticResponse(
    const std::string& filepath,
    const std::string& operator_name,
    const std::vector<double>& temperatures,
    const std::vector<double>& expectation,
    const std::vector<double>& expectation_error,
    const std::vector<double>& variance,
    const std::vector<double>& variance_error,
    const std::vector<double>& susceptibility,
    const std::vector<double>& susceptibility_error,
    uint64_t total_samples
) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        // Ensure correlations group exists
        if (!file.nameExists("/correlations")) {
            file.createGroup("/correlations");
        }

        std::string base_path = "/correlations/" + operator_name;
        if (!file.nameExists(base_path)) {
            file.createGroup(base_path);
        }

        // Helper lambda to save an array
        auto saveDataset = [&](const std::string& name, const std::vector<double>& data) {
            if (data.empty()) return;
            std::string dataset_name = base_path + "/" + name;
            if (file.nameExists(dataset_name)) {
                file.unlink(dataset_name);
            }
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(dataset_name,
                                                     H5::PredType::NATIVE_DOUBLE,
                                                     dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            dataset.close();
        };

        // Save all arrays
        saveDataset("temperatures", temperatures);
        saveDataset("expectation", expectation);
        saveDataset("expectation_error", expectation_error);
        saveDataset("variance", variance);
        saveDataset("variance_error", variance_error);
        saveDataset("susceptibility", susceptibility);
        saveDataset("susceptibility_error", susceptibility_error);

        // Save metadata
        H5::Group group = file.openGroup(base_path);
        H5::DataSpace attr_space(H5S_SCALAR);

        if (group.attrExists("total_samples")) {
            group.removeAttr("total_samples");
        }
        H5::Attribute samples_attr = group.createAttribute("total_samples",
                                                           H5::PredType::NATIVE_UINT64,
                                                           attr_space);
        samples_attr.write(H5::PredType::NATIVE_UINT64, &total_samples);
        samples_attr.close();

        group.close();
        file.close();

        std::cout << "Saved static response (" << operator_name << ") to HDF5" << std::endl;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to save static response: " +
                               std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::saveDynamicalResponseFull(
    const std::string& filepath,
    const std::string& operator_name,
    const std::vector<double>& frequencies,
    const std::vector<double>& spectral_real,
    const std::vector<double>& spectral_imag,
    const std::vector<double>& error_real,
    const std::vector<double>& error_imag,
    uint64_t total_samples,
    double temperature
) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string base_path = "/dynamical/" + operator_name;
        // Walk the path and create any missing intermediate groups
        // (HDF5 1.10 H5Lexists fails if intermediate components are missing)
        {
            std::string cur;
            size_t pos = 0;
            while (pos < base_path.size()) {
                size_t next = base_path.find('/', pos + 1);
                if (next == std::string::npos) next = base_path.size();
                cur = base_path.substr(0, next);
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

        // Helper lambda to save an array
        auto saveDataset = [&](const std::string& name, const std::vector<double>& data) {
            if (data.empty()) return;
            std::string dataset_name = base_path + "/" + name;
            if (file.nameExists(dataset_name)) {
                file.unlink(dataset_name);
            }
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(dataset_name,
                                                     H5::PredType::NATIVE_DOUBLE,
                                                     dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            dataset.close();
        };

        // Save all arrays
        saveDataset("frequencies", frequencies);
        saveDataset("spectral_real", spectral_real);
        saveDataset("spectral_imag", spectral_imag);
        saveDataset("error_real", error_real);
        saveDataset("error_imag", error_imag);

        // Save metadata
        H5::Group group = file.openGroup(base_path);
        H5::DataSpace attr_space(H5S_SCALAR);

        if (group.attrExists("total_samples")) {
            group.removeAttr("total_samples");
        }
        H5::Attribute samples_attr = group.createAttribute("total_samples",
                                                           H5::PredType::NATIVE_UINT64,
                                                           attr_space);
        samples_attr.write(H5::PredType::NATIVE_UINT64, &total_samples);
        samples_attr.close();

        if (group.attrExists("temperature")) {
            group.removeAttr("temperature");
        }
        H5::Attribute temp_attr = group.createAttribute("temperature",
                                                        H5::PredType::NATIVE_DOUBLE,
                                                        attr_space);
        temp_attr.write(H5::PredType::NATIVE_DOUBLE, &temperature);
        temp_attr.close();

        group.close();
        file.close();

        std::cout << "Saved dynamical response (" << operator_name << ") to HDF5" << std::endl;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to save dynamical response: " +
                               std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::ensureFTLMSampleGroups(const std::string& filepath) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        if (!file.nameExists("/ftlm")) {
            file.createGroup("/ftlm");
        }
        if (!file.nameExists("/ftlm/samples")) {
            file.createGroup("/ftlm/samples");
        }
        if (!file.nameExists("/ftlm/samples/thermodynamic")) {
            file.createGroup("/ftlm/samples/thermodynamic");
        }
        if (!file.nameExists("/ftlm/samples/dynamical")) {
            file.createGroup("/ftlm/samples/dynamical");
        }
        if (!file.nameExists("/ftlm/samples/dynamical_correlation")) {
            file.createGroup("/ftlm/samples/dynamical_correlation");
        }
        if (!file.nameExists("/ftlm/samples/static")) {
            file.createGroup("/ftlm/samples/static");
        }

        file.close();
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to create FTLM sample groups: " + std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::saveFTLMDynamicalSample(const std::string& filepath,
                                     size_t sample_index,
                                     const FTLMDynamicalSample& sample,
                                     bool is_correlation) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string base_group = is_correlation ? "/ftlm/samples/dynamical_correlation"
                                                : "/ftlm/samples/dynamical";
        std::string sample_group = base_group + "/sample_" + std::to_string(sample_index);

        // Create groups if needed
        if (!file.nameExists(base_group)) {
            ensureFTLMSampleGroups(filepath);
        }
        if (file.nameExists(sample_group)) {
            file.unlink(sample_group);
        }
        file.createGroup(sample_group);

        // Helper to save dataset
        auto saveDataset = [&](const std::string& name, const std::vector<double>& data) {
            std::string path = sample_group + "/" + name;
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(path, H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            dataset.close();
        };

        saveDataset("frequencies", sample.frequencies);
        saveDataset("spectral_real", sample.spectral_real);
        saveDataset("spectral_imag", sample.spectral_imag);

        file.close();
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to save FTLM dynamical sample: " + std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::saveFTLMStaticSample(const std::string& filepath,
                                  size_t sample_index,
                                  const FTLMStaticSample& sample,
                                  const std::string& operator_name) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string base_group = "/ftlm/samples/static";
        std::string sample_group = base_group + "/sample_" + std::to_string(sample_index);

        // Create groups if needed
        if (!file.nameExists(base_group)) {
            ensureFTLMSampleGroups(filepath);
        }
        if (file.nameExists(sample_group)) {
            file.unlink(sample_group);
        }
        file.createGroup(sample_group);

        // Helper to save dataset
        auto saveDataset = [&](const std::string& name, const std::vector<double>& data) {
            std::string path = sample_group + "/" + name;
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(path, H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            dataset.close();
        };

        saveDataset("temperatures", sample.temperatures);
        saveDataset("expectation", sample.expectation);
        saveDataset("variance", sample.variance);

        // Add operator name as attribute if provided
        if (!operator_name.empty()) {
            H5::Group group = file.openGroup(sample_group);
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::StrType str_type(H5::PredType::C_S1, 64);
            H5::Attribute attr = group.createAttribute("operator", str_type, attr_space);
            attr.write(str_type, operator_name.c_str());
            attr.close();
            group.close();
        }

        file.close();
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to save FTLM static sample: " + std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::ensureTimeCorrelationGroups(const std::string& filepath) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        if (!file.nameExists("/dynamical")) {
            file.createGroup("/dynamical");
        }
        if (!file.nameExists("/dynamical/time_correlations")) {
            file.createGroup("/dynamical/time_correlations");
        }

        file.close();
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to create time correlation groups: " + std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::saveTimeCorrelation(const std::string& filepath,
                                 const std::string& operator_name,
                                 size_t sample_index,
                                 double beta,
                                 const TimeCorrelationData& data,
                                 const std::string& label) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        // Ensure base groups exist
        if (!file.nameExists("/dynamical/time_correlations")) {
            ensureTimeCorrelationGroups(filepath);
        }

        // Create dataset name: /dynamical/time_correlations/operator_sample_beta_label
        std::stringstream ss;
        ss << "/dynamical/time_correlations/" << operator_name
           << "_sample" << sample_index
           << "_beta" << std::fixed << std::setprecision(4) << beta;
        if (!label.empty()) {
            ss << "_" << label;
        }
        std::string group_path = ss.str();

        // Remove existing if present
        if (file.nameExists(group_path)) {
            file.unlink(group_path);
        }
        file.createGroup(group_path);

        // Helper to save dataset
        auto saveDataset = [&](const std::string& name, const std::vector<double>& arr) {
            std::string path = group_path + "/" + name;
            hsize_t dims[1] = {arr.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(path, H5::PredType::NATIVE_DOUBLE, dataspace);
            dataset.write(arr.data(), H5::PredType::NATIVE_DOUBLE);
            dataset.close();
        };

        saveDataset("times", data.times);
        saveDataset("correlation_real", data.correlation_real);
        saveDataset("correlation_imag", data.correlation_imag);

        // Add metadata as attributes
        H5::Group group = file.openGroup(group_path);
        H5::DataSpace attr_space(H5S_SCALAR);

        // Beta
        H5::Attribute beta_attr = group.createAttribute("beta", H5::PredType::NATIVE_DOUBLE, attr_space);
        beta_attr.write(H5::PredType::NATIVE_DOUBLE, &beta);
        beta_attr.close();

        // Sample index
        uint64_t sample = sample_index;
        H5::Attribute sample_attr = group.createAttribute("sample_index", H5::PredType::NATIVE_UINT64, attr_space);
        sample_attr.write(H5::PredType::NATIVE_UINT64, &sample);
        sample_attr.close();

        // Operator name
        H5::StrType str_type(H5::PredType::C_S1, 64);
        H5::Attribute op_attr = group.createAttribute("operator", str_type, attr_space);
        op_attr.write(str_type, operator_name.c_str());
        op_attr.close();

        // Label
        if (!label.empty()) {
            H5::Attribute label_attr = group.createAttribute("label", str_type, attr_space);
            label_attr.write(str_type, label.c_str());
            label_attr.close();
        }

        group.close();
        file.close();

        std::cout << "Saved time correlation to HDF5: " << group_path << std::endl;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to save time correlation: " + std::string(e.getCDetailMsg()));
    }
}

std::vector<std::string> HDF5IO::listTimeCorrelations(const std::string& filepath) {
    std::vector<std::string> correlations;

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        std::string base_path = "/dynamical/time_correlations";
        if (!file.nameExists(base_path)) {
            file.close();
            return correlations;
        }

        H5::Group group = file.openGroup(base_path);
        hsize_t num_objs = group.getNumObjs();

        for (hsize_t i = 0; i < num_objs; ++i) {
            std::string name = group.getObjnameByIdx(i);
            correlations.push_back(base_path + "/" + name);
        }

        group.close();
        file.close();
    } catch (H5::Exception& e) {
        std::cerr << "Warning: HDF5 error listing time correlations: " << e.getDetailMsg() << std::endl;
    }

    return correlations;
}
