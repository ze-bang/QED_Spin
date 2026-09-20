// =============================================================================
// src/io/hdf5_io_tpq.cpp
//
// HDF5IO: TPQ sample I/O -- states, thermodynamics and norm streams (append,
// truncate-and-rewrite, load, list) and the per-rank file merge/copy paths.
// Declarations live in include/ed/core/hdf5_io.h.
// =============================================================================

#include <ed/core/hdf5_io.h>

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

bool HDF5IO::saveTPQState(const std::string& filepath,
                          size_t sample_index,
                          double beta,
                          const std::vector<Complex>& state,
                          bool overwrite) {
    if (isDisabledOutputPath(filepath)) return true;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        // Create sample group and states subgroup if needed
        std::string sample_group = "/tpq/samples/sample_" + std::to_string(sample_index);
        std::string states_group = sample_group + "/states";

        if (!file.nameExists(sample_group)) {
            file.createGroup(sample_group);
        }
        if (!file.nameExists(states_group)) {
            file.createGroup(states_group);
        }

        std::stringstream ss;
        ss << states_group << "/beta_"
           << std::fixed << std::setprecision(6) << beta;
        std::string dataset_name = ss.str();

        if (file.nameExists(dataset_name)) {
            if (!overwrite) {
                // Skip - state already exists
                file.close();
                return false;
            }
            file.unlink(dataset_name);
        }

        size_t N = state.size();

        // Create compound datatype for complex numbers
        H5::CompType complex_type(2 * sizeof(double));
        complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);

        hsize_t dims[1] = {N};
        H5::DataSpace dataspace(1, dims);
        H5::DataSet dataset = file.createDataSet(dataset_name, complex_type, dataspace);

        struct ComplexPair {
            double real;
            double imag;
        };

        std::vector<ComplexPair> data(N);
        for (size_t i = 0; i < N; ++i) {
            data[i].real = state[i].real();
            data[i].imag = state[i].imag();
        }

        dataset.write(data.data(), complex_type);

        // Add metadata
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::Attribute beta_attr = dataset.createAttribute("beta",
                                                          H5::PredType::NATIVE_DOUBLE,
                                                          attr_space);
        beta_attr.write(H5::PredType::NATIVE_DOUBLE, &beta);
        beta_attr.close();

        H5::Attribute sample_attr = dataset.createAttribute("sample_index",
                                                            H5::PredType::NATIVE_UINT64,
                                                            attr_space);
        uint64_t sample = sample_index;
        sample_attr.write(H5::PredType::NATIVE_UINT64, &sample);
        sample_attr.close();

        dataset.close();
        file.close();
        return true;

    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to save TPQ state: " + std::string(e.getCDetailMsg()));
    }
}

bool HDF5IO::loadTPQState(const std::string& filepath,
                          size_t sample_index,
                          double beta,
                          std::vector<Complex>& state) {
    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        std::stringstream ss;
        ss << "/tpq/samples/sample_" << sample_index << "/states/beta_"
           << std::fixed << std::setprecision(6) << beta;
        std::string dataset_name = ss.str();

        if (!file.nameExists(dataset_name)) {
            file.close();
            return false;
        }

        H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims);
        size_t N = dims[0];

        // Create compound datatype for complex numbers
        H5::CompType complex_type(2 * sizeof(double));
        complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);

        struct ComplexPair {
            double real;
            double imag;
        };

        std::vector<ComplexPair> data(N);
        dataset.read(data.data(), complex_type);

        state.resize(N);
        for (size_t i = 0; i < N; ++i) {
            state[i] = Complex(data[i].real, data[i].imag);
        }

        dataset.close();
        file.close();
        return true;

    } catch (H5::Exception& e) {
        return false;
    }
}

std::vector<HDF5IO::TPQStateInfo> HDF5IO::listTPQStates(const std::string& filepath, 
                                                int sample_filter) {
    std::vector<TPQStateInfo> states;

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        if (!file.nameExists("/tpq/samples")) {
            file.close();
            return states;
        }

        H5::Group samples_group = file.openGroup("/tpq/samples");
        hsize_t num_samples = samples_group.getNumObjs();

        for (hsize_t s = 0; s < num_samples; ++s) {
            std::string sample_name = samples_group.getObjnameByIdx(s);

            // Parse sample_N
            if (sample_name.find("sample_") == 0) {
                try {
                    size_t sample_index = std::stoull(sample_name.substr(7));

                    // Apply sample filter if specified
                    if (sample_filter >= 0 && sample_index != static_cast<size_t>(sample_filter)) {
                        continue;
                    }

                    std::string states_path = "/tpq/samples/" + sample_name + "/states";
                    if (!file.nameExists(states_path)) {
                        continue;
                    }

                    H5::Group states_group = file.openGroup(states_path);
                    hsize_t num_states = states_group.getNumObjs();

                    for (hsize_t i = 0; i < num_states; ++i) {
                        std::string state_name = states_group.getObjnameByIdx(i);

                        // Parse dataset name: beta_<beta>
                        // Example: beta_10.500000
                        if (state_name.find("beta_") == 0) {
                            try {
                                std::string beta_str = state_name.substr(5);

                                TPQStateInfo info;
                                info.sample_index = sample_index;
                                info.beta = std::stod(beta_str);
                                info.dataset_name = states_path + "/" + state_name;

                                states.push_back(info);
                            } catch (const std::exception& e) {
                                std::cerr << "Warning: Failed to parse TPQ state '" << state_name << "': " << e.what() << std::endl;
                            } catch (...) {
                                std::cerr << "Warning: Unknown error parsing TPQ state '" << state_name << "'" << std::endl;
                            }
                        }
                    }
                    states_group.close();
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to process sample directory '" << sample_name << "': " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Warning: Unknown error processing sample directory '" << sample_name << "'" << std::endl;
                }
            }
        }

        samples_group.close();
        file.close();

    } catch (H5::Exception& e) {
        std::cerr << "Warning: HDF5 error listing TPQ states: " << e.getDetailMsg() << std::endl;
    }

    return states;
}

std::vector<HDF5IO::TPQStateInfo> HDF5IO::listTPQStatesForSample(const std::string& filepath,
                                                         size_t sample_index) {
    return listTPQStates(filepath, static_cast<int>(sample_index));
}

bool HDF5IO::loadTPQStateByName(const std::string& filepath,
                                const std::string& dataset_name,
                                std::vector<Complex>& state) {
    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        if (!file.nameExists(dataset_name)) {
            file.close();
            return false;
        }

        H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace dataspace = dataset.getSpace();

        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims);
        size_t N = dims[0];

        // Create compound datatype for complex numbers
        H5::CompType complex_type(2 * sizeof(double));
        complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);

        struct ComplexPair {
            double real;
            double imag;
        };

        std::vector<ComplexPair> data(N);
        dataset.read(data.data(), complex_type);

        state.resize(N);
        for (size_t i = 0; i < N; ++i) {
            state[i] = Complex(data[i].real, data[i].imag);
        }

        dataset.close();
        file.close();
        return true;

    } catch (H5::Exception& e) {
        return false;
    }
}

void HDF5IO::ensureTPQSampleGroup(const std::string& filepath, size_t sample_index) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string sample_group = "/tpq/samples/sample_" + std::to_string(sample_index);

        if (!file.nameExists("/tpq")) {
            file.createGroup("/tpq");
        }
        if (!file.nameExists("/tpq/samples")) {
            file.createGroup("/tpq/samples");
        }
        if (!file.nameExists(sample_group)) {
            file.createGroup(sample_group);
        }

        file.close();
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to create TPQ sample group: " + std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::truncateAndRewriteTPQThermodynamics(const std::string& filepath,
                                                 size_t sample_index,
                                                 const std::vector<TPQThermodynamicPoint>& kept_data,
                                                 const std::vector<TPQThermodynamicPoint>& new_data) {
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/thermodynamics";

        // Delete the existing dataset if it exists
        if (file.nameExists(dataset_path)) {
            file.unlink(dataset_path);
        }

        // Combine kept and new data
        size_t total_rows = kept_data.size() + new_data.size();
        if (total_rows == 0) {
            file.close();
            return;
        }

        const hsize_t num_cols = 5;
        hsize_t dims[2] = {total_rows, num_cols};
        hsize_t maxdims[2] = {H5S_UNLIMITED, num_cols};
        H5::DataSpace dataspace(2, dims, maxdims);

        // Adaptive chunking + tunable compression (see makeAdaptiveDsetProps).
        H5::DSetCreatPropList plist = makeAdaptiveDsetProps(
            {dims[0], dims[1]}, sizeof(double));

        H5::DataSet dataset = file.createDataSet(dataset_path,
                                                  H5::PredType::NATIVE_DOUBLE,
                                                  dataspace, plist);

        // Prepare combined data
        std::vector<double> combined_data(total_rows * num_cols);
        size_t row = 0;

        // Add kept data
        for (const auto& pt : kept_data) {
            combined_data[row * num_cols + 0] = pt.beta;
            combined_data[row * num_cols + 1] = pt.energy;
            combined_data[row * num_cols + 2] = pt.variance;
            combined_data[row * num_cols + 3] = pt.doublon;
            combined_data[row * num_cols + 4] = static_cast<double>(pt.step);
            row++;
        }

        // Add new data
        for (const auto& pt : new_data) {
            combined_data[row * num_cols + 0] = pt.beta;
            combined_data[row * num_cols + 1] = pt.energy;
            combined_data[row * num_cols + 2] = pt.variance;
            combined_data[row * num_cols + 3] = pt.doublon;
            combined_data[row * num_cols + 4] = static_cast<double>(pt.step);
            row++;
        }

        // Write all data
        dataset.write(combined_data.data(), H5::PredType::NATIVE_DOUBLE);

        // Add column labels
        std::string columns_attr = "beta,energy,variance,doublon,step";
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::StrType str_type(H5::PredType::C_S1, 64);
        H5::Attribute attr = dataset.createAttribute("columns", str_type, attr_space);
        attr.write(str_type, columns_attr.c_str());
        attr.close();

        dataset.close();
        file.close();
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to truncate/rewrite TPQ thermodynamics: " + std::string(e.getCDetailMsg()));
    }
}

void HDF5IO::truncateAndRewriteTPQNorm(const std::string& filepath,
                                       size_t sample_index,
                                       const std::vector<TPQNormPoint>& kept_data,
                                       const std::vector<TPQNormPoint>& new_data) {
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/norm";

        // Delete the existing dataset if it exists
        if (file.nameExists(dataset_path)) {
            file.unlink(dataset_path);
        }

        // Combine kept and new data
        size_t total_rows = kept_data.size() + new_data.size();
        if (total_rows == 0) {
            file.close();
            return;
        }

        const hsize_t num_cols = 4;
        hsize_t dims[2] = {total_rows, num_cols};
        hsize_t maxdims[2] = {H5S_UNLIMITED, num_cols};
        H5::DataSpace dataspace(2, dims, maxdims);

        // Adaptive chunking + tunable compression (see makeAdaptiveDsetProps).
        H5::DSetCreatPropList plist = makeAdaptiveDsetProps(
            {dims[0], dims[1]}, sizeof(double));

        H5::DataSet dataset = file.createDataSet(dataset_path,
                                                  H5::PredType::NATIVE_DOUBLE,
                                                  dataspace, plist);

        // Prepare combined data
        std::vector<double> combined_data(total_rows * num_cols);
        size_t row = 0;

        // Add kept data
        for (const auto& pt : kept_data) {
            combined_data[row * num_cols + 0] = pt.beta;
            combined_data[row * num_cols + 1] = pt.norm;
            combined_data[row * num_cols + 2] = pt.first_norm;
            combined_data[row * num_cols + 3] = static_cast<double>(pt.step);
            row++;
        }

        // Add new data
        for (const auto& pt : new_data) {
            combined_data[row * num_cols + 0] = pt.beta;
            combined_data[row * num_cols + 1] = pt.norm;
            combined_data[row * num_cols + 2] = pt.first_norm;
            combined_data[row * num_cols + 3] = static_cast<double>(pt.step);
            row++;
        }

        // Write all data
        dataset.write(combined_data.data(), H5::PredType::NATIVE_DOUBLE);

        // Add column labels
        std::string columns_attr = "beta,norm,first_norm,step";
        H5::DataSpace attr_space(H5S_SCALAR);
        H5::StrType str_type(H5::PredType::C_S1, 64);
        H5::Attribute attr = dataset.createAttribute("columns", str_type, attr_space);
        attr.write(str_type, columns_attr.c_str());
        attr.close();

        dataset.close();
        file.close();
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to truncate/rewrite TPQ norm: " + std::string(e.getCDetailMsg()));
    }
}

bool HDF5IO::appendTPQThermodynamics(const std::string& filepath,
                                     size_t sample_index,
                                     const TPQThermodynamicPoint& point) {
    if (isDisabledOutputPath(filepath)) return false;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/thermodynamics";

        // Ensure parent groups exist
        if (!file.nameExists("/tpq")) {
            file.createGroup("/tpq");
        }
        if (!file.nameExists("/tpq/samples")) {
            file.createGroup("/tpq/samples");
        }

        // Ensure sample group exists
        std::string sample_group = "/tpq/samples/sample_" + std::to_string(sample_index);
        if (!file.nameExists(sample_group)) {
            file.createGroup(sample_group);
        }

        // Data layout: 5 columns [beta, energy, variance, doublon, step]
        const hsize_t num_cols = 5;
        double row_data[5] = {point.beta, point.energy, point.variance, point.doublon, static_cast<double>(point.step)};

        if (!file.nameExists(dataset_path)) {
            // Create extensible dataset
            hsize_t dims[2] = {1, num_cols};
            hsize_t maxdims[2] = {H5S_UNLIMITED, num_cols};
            H5::DataSpace dataspace(2, dims, maxdims);

            // Adaptive chunking + tunable compression (see makeAdaptiveDsetProps).
            H5::DSetCreatPropList plist = makeAdaptiveDsetProps(
                {dims[0], dims[1]}, sizeof(double));

            H5::DataSet dataset = file.createDataSet(dataset_path,
                                                     H5::PredType::NATIVE_DOUBLE,
                                                     dataspace, plist);
            dataset.write(row_data, H5::PredType::NATIVE_DOUBLE);
            dataset.close();
            file.close();
            return true;
        } else {
            // Check if step already exists before appending
            H5::DataSet dataset = file.openDataSet(dataset_path);
            H5::DataSpace filespace = dataset.getSpace();

            hsize_t dims[2];
            filespace.getSimpleExtentDims(dims);

            // Read existing data
            std::vector<double> existing_data(dims[0] * num_cols);
            if (dims[0] > 0) {
                dataset.read(existing_data.data(), H5::PredType::NATIVE_DOUBLE);

                // Check if step already exists
                for (hsize_t i = 0; i < dims[0]; ++i) {
                    uint64_t existing_step = static_cast<uint64_t>(existing_data[i * num_cols + 4]);
                    if (existing_step == point.step) {
                        // Step already exists, skip writing
                        dataset.close();
                        file.close();
                        return false;
                    }
                }
            }

            // Check if dataset is chunked (extensible)
            H5::DSetCreatPropList cplist = dataset.getCreatePlist();
            bool is_chunked = (cplist.getLayout() == H5D_CHUNKED);

            if (is_chunked) {
                // Extend dataset and append new row
                hsize_t new_dims[2] = {dims[0] + 1, num_cols};
                dataset.extend(new_dims);

                // Select hyperslab for the new row
                filespace = dataset.getSpace();
                hsize_t offset[2] = {dims[0], 0};
                hsize_t count[2] = {1, num_cols};
                filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

                // Write the new row
                H5::DataSpace memspace(2, count);
                dataset.write(row_data, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
                dataset.close();
                file.close();
                return true;
            } else {
                // Dataset is not chunked (legacy contiguous storage)
                // Need to recreate it with chunking enabled
                dataset.close();

                // Read any existing column attribute
                std::string columns_attr = "beta,energy,variance,doublon,step";

                // Delete the old dataset
                file.unlink(dataset_path);

                // Create new chunked dataset with existing + new data
                hsize_t new_rows = dims[0] + 1;
                hsize_t new_dims[2] = {new_rows, num_cols};
                hsize_t maxdims[2] = {H5S_UNLIMITED, num_cols};
                H5::DataSpace new_dataspace(2, new_dims, maxdims);

                H5::DSetCreatPropList plist = makeAdaptiveDsetProps(
                    {new_dims[0], new_dims[1]}, sizeof(double));

                H5::DataSet new_dataset = file.createDataSet(dataset_path,
                                                              H5::PredType::NATIVE_DOUBLE,
                                                              new_dataspace, plist);

                // Append new row to existing data
                existing_data.resize(new_rows * num_cols);
                existing_data[dims[0] * num_cols + 0] = row_data[0];
                existing_data[dims[0] * num_cols + 1] = row_data[1];
                existing_data[dims[0] * num_cols + 2] = row_data[2];
                existing_data[dims[0] * num_cols + 3] = row_data[3];
                existing_data[dims[0] * num_cols + 4] = row_data[4];

                new_dataset.write(existing_data.data(), H5::PredType::NATIVE_DOUBLE);

                // Add column labels as attribute
                H5::DataSpace attr_space(H5S_SCALAR);
                H5::StrType str_type(H5::PredType::C_S1, 64);
                H5::Attribute attr = new_dataset.createAttribute("columns", str_type, attr_space);
                attr.write(str_type, columns_attr.c_str());
                attr.close();

                new_dataset.close();
                file.close();
                return true;
            }
        }
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to append TPQ thermodynamics: " + std::string(e.getCDetailMsg()));
    }
}

bool HDF5IO::appendTPQNorm(const std::string& filepath,
                           size_t sample_index,
                           const TPQNormPoint& point) {
    if (isDisabledOutputPath(filepath)) return false;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/norm";

        // Ensure parent groups exist
        if (!file.nameExists("/tpq")) {
            file.createGroup("/tpq");
        }
        if (!file.nameExists("/tpq/samples")) {
            file.createGroup("/tpq/samples");
        }

        // Ensure sample group exists
        std::string sample_group = "/tpq/samples/sample_" + std::to_string(sample_index);
        if (!file.nameExists(sample_group)) {
            file.createGroup(sample_group);
        }

        // Data layout: 4 columns [beta, norm, first_norm, step]
        const hsize_t num_cols = 4;
        double row_data[4] = {point.beta, point.norm, point.first_norm, static_cast<double>(point.step)};

        if (!file.nameExists(dataset_path)) {
            // Create extensible dataset
            hsize_t dims[2] = {1, num_cols};
            hsize_t maxdims[2] = {H5S_UNLIMITED, num_cols};
            H5::DataSpace dataspace(2, dims, maxdims);

            // Adaptive chunking + tunable compression (see makeAdaptiveDsetProps).
            H5::DSetCreatPropList plist = makeAdaptiveDsetProps(
                {dims[0], dims[1]}, sizeof(double));

            H5::DataSet dataset = file.createDataSet(dataset_path,
                                                     H5::PredType::NATIVE_DOUBLE,
                                                     dataspace, plist);
            dataset.write(row_data, H5::PredType::NATIVE_DOUBLE);
            dataset.close();
            file.close();
            return true;
        } else {
            // Check if step already exists before appending
            H5::DataSet dataset = file.openDataSet(dataset_path);
            H5::DataSpace filespace = dataset.getSpace();

            hsize_t dims[2];
            filespace.getSimpleExtentDims(dims);

            // Read existing data
            std::vector<double> existing_data(dims[0] * num_cols);
            if (dims[0] > 0) {
                dataset.read(existing_data.data(), H5::PredType::NATIVE_DOUBLE);

                // Check if step already exists
                for (hsize_t i = 0; i < dims[0]; ++i) {
                    uint64_t existing_step = static_cast<uint64_t>(existing_data[i * num_cols + 3]);
                    if (existing_step == point.step) {
                        // Step already exists, skip writing
                        dataset.close();
                        file.close();
                        return false;
                    }
                }
            }

            // Check if dataset is chunked (extensible)
            H5::DSetCreatPropList cplist = dataset.getCreatePlist();
            bool is_chunked = (cplist.getLayout() == H5D_CHUNKED);

            if (is_chunked) {
                // Extend and append
                hsize_t new_dims[2] = {dims[0] + 1, num_cols};
                dataset.extend(new_dims);

                filespace = dataset.getSpace();
                hsize_t offset[2] = {dims[0], 0};
                hsize_t count[2] = {1, num_cols};
                filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

                H5::DataSpace memspace(2, count);
                dataset.write(row_data, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
                dataset.close();
                file.close();
                return true;
            } else {
                // Dataset is not chunked (legacy contiguous storage)
                // Need to recreate it with chunking enabled
                dataset.close();

                std::string columns_attr = "beta,norm,first_norm,step";

                // Delete the old dataset
                file.unlink(dataset_path);

                // Create new chunked dataset with existing + new data
                hsize_t new_rows = dims[0] + 1;
                hsize_t new_dims[2] = {new_rows, num_cols};
                hsize_t maxdims[2] = {H5S_UNLIMITED, num_cols};
                H5::DataSpace new_dataspace(2, new_dims, maxdims);

                H5::DSetCreatPropList plist = makeAdaptiveDsetProps(
                    {new_dims[0], new_dims[1]}, sizeof(double));

                H5::DataSet new_dataset = file.createDataSet(dataset_path,
                                                              H5::PredType::NATIVE_DOUBLE,
                                                              new_dataspace, plist);

                // Append new row to existing data
                existing_data.resize(new_rows * num_cols);
                existing_data[dims[0] * num_cols + 0] = row_data[0];
                existing_data[dims[0] * num_cols + 1] = row_data[1];
                existing_data[dims[0] * num_cols + 2] = row_data[2];
                existing_data[dims[0] * num_cols + 3] = row_data[3];

                new_dataset.write(existing_data.data(), H5::PredType::NATIVE_DOUBLE);

                // Add column labels as attribute
                H5::DataSpace attr_space(H5S_SCALAR);
                H5::StrType str_type(H5::PredType::C_S1, 64);
                H5::Attribute attr = new_dataset.createAttribute("columns", str_type, attr_space);
                attr.write(str_type, columns_attr.c_str());
                attr.close();

                new_dataset.close();
                file.close();
                return true;
            }
        }
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to append TPQ norm: " + std::string(e.getCDetailMsg()));
    }
}

std::vector<HDF5IO::TPQThermodynamicPoint> HDF5IO::loadTPQThermodynamics(const std::string& filepath,
                                                                 size_t sample_index) {
    std::vector<TPQThermodynamicPoint> points;

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/thermodynamics";

        // Defensive: ``nameExists`` itself throws when an intermediate
        // group (e.g. ``sample_N``) is missing. Callers that iterate
        // from ``sample_index=0`` upward rely on "missing dataset"
        // being a normal stop condition, so trap that case and return
        // an empty vector instead of rethrowing.
        bool dataset_present = false;
        try {
            dataset_present = file.nameExists(dataset_path);
        } catch (const H5::Exception&) {
            dataset_present = false;
        }
        if (!dataset_present) {
            file.close();
            return points;
        }

        H5::DataSet dataset = file.openDataSet(dataset_path);
        H5::DataSpace dataspace = dataset.getSpace();

        // Validate rank + column count up front; the column layout is
        // [beta, energy, variance, doublon, step] (=5) per the writer.
        // Trusting a wrong-shape dataset would silently shuffle columns
        // and produce wrong thermodynamics.
        if (dataspace.getSimpleExtentNdims() != 2) {
            throw std::runtime_error(
                "loadTPQThermodynamics: dataset '" + dataset_path +
                "' is not 2-D (rank=" +
                std::to_string(dataspace.getSimpleExtentNdims()) + ")");
        }
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        hsize_t num_rows = dims[0];
        hsize_t num_cols = dims[1];
        constexpr hsize_t kExpectedCols = 5;
        if (num_cols != kExpectedCols) {
            throw std::runtime_error(
                "loadTPQThermodynamics: dataset '" + dataset_path +
                "' has " + std::to_string(num_cols) +
                " columns; expected " + std::to_string(kExpectedCols) +
                " (beta, energy, variance, doublon, step)");
        }

        std::vector<double> data(num_rows * num_cols);
        dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);

        points.resize(num_rows);
        for (hsize_t i = 0; i < num_rows; ++i) {
            points[i].beta = data[i * num_cols + 0];
            points[i].energy = data[i * num_cols + 1];
            points[i].variance = data[i * num_cols + 2];
            points[i].doublon = data[i * num_cols + 3];
            points[i].step = static_cast<uint64_t>(data[i * num_cols + 4]);
        }

        dataset.close();
        file.close();
    } catch (H5::Exception& e) {
        // Re-throw with a helpful prefix instead of silently dropping data;
        // an HDF5 error on a real read is *not* equivalent to "the file
        // has no samples" and must not be conflated with the
        // "dataset missing" case above.
        throw std::runtime_error(
            "loadTPQThermodynamics(" + filepath + ", sample " +
            std::to_string(sample_index) + "): " +
            std::string(e.getCDetailMsg()));
    }

    return points;
}

std::vector<HDF5IO::TPQNormPoint> HDF5IO::loadTPQNorm(const std::string& filepath,
                                              size_t sample_index) {
    std::vector<TPQNormPoint> points;

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/norm";

        // ``nameExists`` may throw when an intermediate group is
        // missing; treat that as "no such sample" to preserve the
        // iterate-until-empty pattern. See note on loadTPQThermodynamics.
        bool dataset_present = false;
        try {
            dataset_present = file.nameExists(dataset_path);
        } catch (const H5::Exception&) {
            dataset_present = false;
        }
        if (!dataset_present) {
            file.close();
            return points;
        }

        H5::DataSet dataset = file.openDataSet(dataset_path);
        H5::DataSpace dataspace = dataset.getSpace();

        if (dataspace.getSimpleExtentNdims() != 2) {
            throw std::runtime_error(
                "loadTPQNorm: dataset '" + dataset_path +
                "' is not 2-D (rank=" +
                std::to_string(dataspace.getSimpleExtentNdims()) + ")");
        }
        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        hsize_t num_rows = dims[0];
        hsize_t num_cols = dims[1];
        constexpr hsize_t kExpectedCols = 4;
        if (num_cols != kExpectedCols) {
            throw std::runtime_error(
                "loadTPQNorm: dataset '" + dataset_path +
                "' has " + std::to_string(num_cols) +
                " columns; expected " + std::to_string(kExpectedCols));
        }

        std::vector<double> data(num_rows * num_cols);
        dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);

        points.resize(num_rows);
        for (hsize_t i = 0; i < num_rows; ++i) {
            points[i].beta = data[i * num_cols + 0];
            points[i].norm = data[i * num_cols + 1];
            points[i].first_norm = data[i * num_cols + 2];
            points[i].step = static_cast<uint64_t>(data[i * num_cols + 3]);
        }

        dataset.close();
        file.close();
    } catch (const std::runtime_error&) {
        throw;
    } catch (H5::Exception& e) {
        throw std::runtime_error(
            "loadTPQNorm(" + filepath + ", sample " +
            std::to_string(sample_index) + "): " +
            std::string(e.getCDetailMsg()));
    }

    return points;
}

std::vector<size_t> HDF5IO::listTPQSamples(const std::string& filepath) {
    std::vector<size_t> samples;

    try {
        H5::H5File file(filepath, H5F_ACC_RDONLY);

        if (!file.nameExists("/tpq/samples")) {
            file.close();
            return samples;
        }

        H5::Group group = file.openGroup("/tpq/samples");
        hsize_t num_objs = group.getNumObjs();

        for (hsize_t i = 0; i < num_objs; ++i) {
            std::string name = group.getObjnameByIdx(i);
            // Parse "sample_N" format
            if (name.find("sample_") == 0) {
                try {
                    size_t sample_idx = std::stoull(name.substr(7));
                    samples.push_back(sample_idx);
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to parse sample name '" << name << "': " << e.what() << std::endl;
                } catch (...) {
                    std::cerr << "Warning: Unknown error parsing sample name '" << name << "'" << std::endl;
                }
            }
        }

        group.close();
        file.close();

        // Sort samples
        std::sort(samples.begin(), samples.end());
    } catch (H5::Exception& e) {
        std::cerr << "Warning: HDF5 error listing completed samples: " << e.getDetailMsg() << std::endl;
    }

    return samples;
}

bool HDF5IO::mergePerRankTPQFiles(const std::string& directory,
                                  int num_ranks,
                                  const std::string& output_filename,
                                  bool delete_temp_files) {
    std::string output_path = directory + "/" + output_filename;

    try {
        std::cout << "\n==========================================\n";
        std::cout << "Merging per-rank HDF5 files\n";
        std::cout << "==========================================\n";
        std::cout << "  Output: " << output_path << std::endl;
        std::cout << "  Ranks to merge: " << num_ranks << std::endl;

        // Create or open the output file
        std::string final_path = createOrOpenFile(directory, output_filename);

        int total_samples_merged = 0;

        for (int rank = 0; rank < num_ranks; ++rank) {
            std::string rank_file = getPerRankFilePath(directory, rank, output_filename);

            if (!fileExists(rank_file)) {
                std::cout << "  Rank " << rank << ": file not found, skipping" << std::endl;
                continue;
            }

            std::cout << "  Merging rank " << rank << " from: " << rank_file << std::endl;

            // Copy TPQ sample data from rank file to output file
            int samples_copied = copyTPQSamples(rank_file, final_path);
            total_samples_merged += samples_copied;

            std::cout << "    Copied " << samples_copied << " samples" << std::endl;

            // Delete temporary file if requested
            if (delete_temp_files) {
                try {
                    std::filesystem::remove(rank_file);
                    std::cout << "    Deleted temporary file" << std::endl;
                } catch (const std::exception& e) {
                    std::cerr << "    Warning: Could not delete " << rank_file << ": " << e.what() << std::endl;
                }
            }
        }

        std::cout << "==========================================\n";
        std::cout << "Merge complete: " << total_samples_merged << " total samples\n";
        std::cout << "==========================================\n";

        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error merging per-rank files: " << e.what() << std::endl;
        return false;
    }
}

int HDF5IO::copyTPQSamples(const std::string& source_path, const std::string& dest_path) {
    int samples_copied = 0;

    // First pass: collect sample names and determine what needs merging
    std::vector<std::string> sample_names;
    std::vector<bool> sample_needs_merge;  // true if sample exists in dest and needs merging

    try {
        H5::H5File source(source_path, H5F_ACC_RDONLY);
        H5::H5File dest(dest_path, H5F_ACC_RDWR);

        // Check if TPQ samples group exists in source
        if (!source.nameExists("/tpq/samples")) {
            source.close();
            dest.close();
            return 0;
        }

        // Ensure destination groups exist
        if (!dest.nameExists("/tpq")) {
            dest.createGroup("/tpq");
        }
        if (!dest.nameExists("/tpq/samples")) {
            dest.createGroup("/tpq/samples");
        }

        H5::Group src_samples = source.openGroup("/tpq/samples");
        hsize_t num_samples = src_samples.getNumObjs();

        for (hsize_t i = 0; i < num_samples; ++i) {
            std::string sample_name = src_samples.getObjnameByIdx(i);
            std::string src_sample_path = "/tpq/samples/" + sample_name;
            std::string dst_sample_path = "/tpq/samples/" + sample_name;

            sample_names.push_back(sample_name);

            // If sample doesn't exist in destination, copy the entire group
            if (!dest.nameExists(dst_sample_path)) {
                if (H5Ocopy(source.getId(), src_sample_path.c_str(),
                           dest.getId(), dst_sample_path.c_str(),
                           H5P_DEFAULT, H5P_DEFAULT) >= 0) {
                    samples_copied++;
                }
                sample_needs_merge.push_back(false);  // Already fully copied
            } else {
                // Sample exists - need to merge
                sample_needs_merge.push_back(true);

                // Copy any states that don't exist yet (can be done with files open)
                std::string src_states_path = src_sample_path + "/states";
                std::string dst_states_path = dst_sample_path + "/states";

                if (source.nameExists(src_states_path)) {
                    if (!dest.nameExists(dst_states_path)) {
                        dest.createGroup(dst_states_path);
                    }

                    H5::Group src_states = source.openGroup(src_states_path);
                    hsize_t num_states = src_states.getNumObjs();

                    for (hsize_t j = 0; j < num_states; ++j) {
                        std::string state_name = src_states.getObjnameByIdx(j);
                        std::string src_state_path = src_states_path + "/" + state_name;
                        std::string dst_state_path = dst_states_path + "/" + state_name;

                        if (!dest.nameExists(dst_state_path)) {
                            H5Ocopy(source.getId(), src_state_path.c_str(),
                                   dest.getId(), dst_state_path.c_str(),
                                   H5P_DEFAULT, H5P_DEFAULT);
                        }
                    }

                    src_states.close();
                }
            }
        }

        src_samples.close();
        source.close();
        dest.close();

    } catch (H5::Exception& e) {
        std::cerr << "Warning: Error in first pass of TPQ merge: " << e.getDetailMsg() << std::endl;
    }

    // Second pass: merge thermodynamics and norm data for samples that need it
    // This is done separately to avoid issues with keeping files open
    for (size_t i = 0; i < sample_names.size(); ++i) {
        if (!sample_needs_merge[i]) continue;

        const std::string& sample_name = sample_names[i];
        bool any_merged = false;

        // Extract sample index from sample_name (e.g., "sample_0" -> 0)
        size_t sample_idx = 0;
        size_t pos = sample_name.find('_');
        if (pos != std::string::npos) {
            try {
                sample_idx = std::stoul(sample_name.substr(pos + 1));
            } catch (...) {
                continue;  // Skip if parsing fails
            }
        }

        // Merge thermodynamics data
        try {
            auto existing_thermo = loadTPQThermodynamics(dest_path, sample_idx);
            auto new_thermo = loadTPQThermodynamics(source_path, sample_idx);

            if (!new_thermo.empty()) {
                // Find min step in new data (this is the resume_step in continue_quenching)
                uint64_t min_new_step = UINT64_MAX;
                for (const auto& pt : new_thermo) {
                    if (pt.step < min_new_step) {
                        min_new_step = pt.step;
                    }
                }

                // Find max step in existing data
                uint64_t max_existing_step = 0;
                for (const auto& pt : existing_thermo) {
                    if (pt.step > max_existing_step) {
                        max_existing_step = pt.step;
                    }
                }

                int appended = 0;

                // Check if there's overlap - this happens when continue_quenching
                // resumed from a saved state that was before the last measurement
                if (min_new_step <= max_existing_step && !existing_thermo.empty()) {
                    // Need to truncate existing data and replace with new data
                    // Keep only existing data with step < min_new_step
                    std::cout << "      Detected overlap: new data starts at step " << min_new_step
                              << ", existing ends at step " << max_existing_step << std::endl;
                    std::cout << "      Truncating existing data to step < " << min_new_step
                              << " and appending new data" << std::endl;

                    // Filter existing data to keep only steps before the resume point
                    std::vector<TPQThermodynamicPoint> kept_data;
                    for (const auto& pt : existing_thermo) {
                        if (pt.step < min_new_step) {
                            kept_data.push_back(pt);
                        }
                    }

                    // Rewrite the dataset: delete and recreate with kept + new data
                    truncateAndRewriteTPQThermodynamics(dest_path, sample_idx, kept_data, new_thermo);
                    appended = new_thermo.size();
                } else {
                    // No overlap - simple append (step > max_existing_step)
                    for (const auto& pt : new_thermo) {
                        if (pt.step > max_existing_step) {
                            appendTPQThermodynamics(dest_path, sample_idx, pt);
                            appended++;
                        }
                    }
                }

                if (appended > 0) {
                    any_merged = true;
                    std::cout << "      Appended " << appended
                              << " thermodynamics points to " << sample_name << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to merge thermodynamics for " << sample_name
                      << ": " << e.what() << std::endl;
        }

        // Merge norm data
        try {
            auto existing_norm = loadTPQNorm(dest_path, sample_idx);
            auto new_norm = loadTPQNorm(source_path, sample_idx);

            if (!new_norm.empty()) {
                // Find min step in new data
                uint64_t min_new_step = UINT64_MAX;
                for (const auto& pt : new_norm) {
                    if (pt.step < min_new_step) {
                        min_new_step = pt.step;
                    }
                }

                // Find max step in existing data
                uint64_t max_existing_step = 0;
                for (const auto& pt : existing_norm) {
                    if (pt.step > max_existing_step) {
                        max_existing_step = pt.step;
                    }
                }

                int appended = 0;

                // Check if there's overlap
                if (min_new_step <= max_existing_step && !existing_norm.empty()) {
                    // Truncate and rewrite
                    std::vector<TPQNormPoint> kept_data;
                    for (const auto& pt : existing_norm) {
                        if (pt.step < min_new_step) {
                            kept_data.push_back(pt);
                        }
                    }

                    truncateAndRewriteTPQNorm(dest_path, sample_idx, kept_data, new_norm);
                    appended = new_norm.size();
                } else {
                    // No overlap - simple append
                    for (const auto& pt : new_norm) {
                        if (pt.step > max_existing_step) {
                            appendTPQNorm(dest_path, sample_idx, pt);
                            appended++;
                        }
                    }
                }

                if (appended > 0) {
                    any_merged = true;
                    std::cout << "      Appended " << appended
                              << " norm points to " << sample_name << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to merge norm for " << sample_name
                      << ": " << e.what() << std::endl;
        }

        if (any_merged) samples_copied++;
    }

    return samples_copied;
}
