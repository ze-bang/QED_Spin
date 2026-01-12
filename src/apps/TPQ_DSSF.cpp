#include <iostream>
#include <vector>
#include <complex>
#include <string>
#include <fstream>
#include <sstream>
#include <filesystem>
#include <regex>
#include <limits>
#include <cstring>
#include <cmath> // added for M_PI, std::sqrt
#include <iomanip> // added for std::setprecision and std::fixed
#include <algorithm> // for std::sort, std::max_element, std::min_element
#include <numeric> // for std::accumulate
#include <mutex>   // for thread-safe HDF5 access
#include <thread>  // for sleep_for in HDF5 retry
#include <chrono>  // for milliseconds
#include <map>     // for batch SSSF results
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/solvers/TPQ.h>
#include <ed/solvers/observables.h>
#ifdef WITH_MPI
#include <mpi.h>
#endif
#include <ed/solvers/dynamics.h>
#include <ed/solvers/ftlm.h>
#include <H5Cpp.h>
#include <hdf5.h>  // For C API needed by parallel HDF5

#ifdef WITH_CUDA
#include <ed/gpu/gpu_ed_wrapper.h>
#include <ed/gpu/gpu_operator.cuh>
#endif

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;
namespace fs = std::filesystem;

// Global mutex for thread-safe HDF5 writes (HDF5 is not thread-safe by default)
static std::mutex g_hdf5_mutex;

// MPI communicator for parallel HDF5 (set during initialization)
static MPI_Comm g_mpi_comm = MPI_COMM_WORLD;
static MPI_Info g_mpi_info = MPI_INFO_NULL;

/**
 * @brief Open HDF5 file with serial access
 * Uses retry mechanism for concurrent access.
 */
H5::H5File openHDF5Serial(const std::string& h5_path, unsigned int flags, int max_retries = 50, int delay_ms = 50) {
    for (int attempt = 0; attempt < max_retries; ++attempt) {
        try {
            H5::Exception::dontPrint();
            return H5::H5File(h5_path, flags);
        } catch (H5::FileIException& e) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }
    return H5::H5File(h5_path, flags);
}

#ifdef HDF5_PARALLEL
/**
 * @brief Open HDF5 file with MPI-IO for parallel access (independent I/O mode)
 * Uses H5Pset_fapl_mpio for parallel file access.
 * Each process can independently read/write to different datasets.
 */
H5::H5File openHDF5Parallel(const std::string& h5_path, unsigned int flags) {
    // Create file access property list for MPI-IO
    hid_t fapl = H5Pcreate(H5P_FILE_ACCESS);
    if (fapl < 0) {
        throw std::runtime_error("Failed to create HDF5 file access property list");
    }
    
    // Set MPI-IO driver for independent I/O
    herr_t status = H5Pset_fapl_mpio(fapl, g_mpi_comm, g_mpi_info);
    if (status < 0) {
        H5Pclose(fapl);
        throw std::runtime_error("Failed to set MPI-IO file access property");
    }
    
    // Open file using C API with MPI-IO property
    hid_t file_id = H5Fopen(h5_path.c_str(), flags, fapl);
    H5Pclose(fapl);
    
    if (file_id < 0) {
        // Fallback to serial access if parallel open fails
        return openHDF5Serial(h5_path, flags);
    }
    
    // Wrap in C++ H5File object (takes ownership)
    return H5::H5File(file_id);
}
#endif

/**
 * @brief Helper function to open HDF5 file with parallel or serial access
 * For file creation (H5F_ACC_TRUNC): uses serial access (only rank 0 should call this)
 * For read/write (H5F_ACC_RDWR): uses parallel HDF5 if available
 * 
 * NOTE: With parallel HDF5 library, metadata operations (creating groups/datasets)
 * must be done carefully. This implementation uses independent I/O mode where
 * each rank can create its own groups/datasets. The HDF5_USE_FILE_LOCKING=FALSE
 * environment variable should be set to avoid file locking issues.
 */
H5::H5File openHDF5WithRetry(const std::string& h5_path, unsigned int flags, int max_retries = 50, int delay_ms = 50) {
    // Always use serial for file creation (only rank 0 creates the file)
    if (flags == H5F_ACC_TRUNC) {
        return openHDF5Serial(h5_path, flags, max_retries, delay_ms);
    }
    
    // For read/write access, use serial with retry (most reliable for independent I/O)
    // The parallel HDF5 library is linked but we use serial file access mode
    // because each MPI rank writes to different groups/datasets independently.
    // This avoids the collective metadata operation requirement of true parallel HDF5.
    return openHDF5Serial(h5_path, flags, max_retries, delay_ms);
}

// ============================================================================
// Unified DSSF HDF5 Output Management
// ============================================================================
// 
// MPI-Safe Strategy:
// - In COLLECTIVE mode: Only rank 0 writes, so no locking issues
// - In INDEPENDENT mode: Each rank writes to its own per-rank file
//   (dssf_results_rank0.h5, dssf_results_rank1.h5, etc.)
//   Rank 0 merges all files at the end
//
// HDF5 File Structure for TPQ_DSSF:
//   dssf_results.h5
//   ├── /metadata
//   │   ├── num_sites [attr]
//   │   ├── spin_length [attr]
//   │   ├── method [attr]
//   │   ├── operator_type [attr]
//   │   ├── omega_min, omega_max, num_omega_bins [attr]
//   │   └── broadening [attr]
//   ├── /momentum_points
//   │   └── q_vectors [dataset: Nx3 array]
//   ├── /spectral
//   │   ├── frequencies [dataset: 1D array, shared across all]
//   │   └── /<operator_name>
//   │       └── /beta_<value> or /T_<value>
//   │           ├── sample_<idx>
//   │           │   ├── real [dataset]
//   │           │   ├── imag [dataset]
//   │           │   ├── error_real [dataset]
//   │           │   └── error_imag [dataset]
//   │           └── averaged (optional)
//   │               ├── real [dataset]
//   │               └── imag [dataset]
//   ├── /static
//   │   └── /<operator_name>
//   │       ├── temperatures [dataset]
//   │       ├── expectation [dataset]
//   │       ├── variance [dataset]
//   │       └── susceptibility [dataset]
//   └── /correlations (for spin_correlation/spin_configuration)
//       └── ...
// ============================================================================

/**
 * @brief Get per-rank DSSF HDF5 file path for MPI-safe writing
 */
std::string getPerRankDSSFPath(const std::string& output_dir, int rank) {
    return output_dir + "/dssf_results_rank" + std::to_string(rank) + ".h5";
}

/**
 * @brief Initialize per-rank DSSF HDF5 file structure
 */
std::string initPerRankDSSFFile(const std::string& output_dir, int rank,
                                 int num_sites, float spin_length,
                                 const std::string& method, 
                                 const std::string& operator_type,
                                 double omega_min, double omega_max, 
                                 int num_omega_bins, double broadening,
                                 const std::vector<std::vector<double>>& momentum_points) {
    std::string h5_path = getPerRankDSSFPath(output_dir, rank);
    
    try {
        H5::H5File file(h5_path, H5F_ACC_TRUNC);
        
        // Create group structure
        file.createGroup("/metadata");
        file.createGroup("/momentum_points");
        file.createGroup("/spectral");
        file.createGroup("/static");
        file.createGroup("/correlations");
        
        // Save metadata as attributes
        H5::Group meta = file.openGroup("/metadata");
        H5::DataSpace scalar_space(H5S_SCALAR);
        
        H5::Attribute attr_sites = meta.createAttribute("num_sites", H5::PredType::NATIVE_INT, scalar_space);
        attr_sites.write(H5::PredType::NATIVE_INT, &num_sites);
        
        H5::Attribute attr_spin = meta.createAttribute("spin_length", H5::PredType::NATIVE_FLOAT, scalar_space);
        attr_spin.write(H5::PredType::NATIVE_FLOAT, &spin_length);
        
        H5::Attribute attr_omega_min = meta.createAttribute("omega_min", H5::PredType::NATIVE_DOUBLE, scalar_space);
        attr_omega_min.write(H5::PredType::NATIVE_DOUBLE, &omega_min);
        
        H5::Attribute attr_omega_max = meta.createAttribute("omega_max", H5::PredType::NATIVE_DOUBLE, scalar_space);
        attr_omega_max.write(H5::PredType::NATIVE_DOUBLE, &omega_max);
        
        H5::Attribute attr_nbins = meta.createAttribute("num_omega_bins", H5::PredType::NATIVE_INT, scalar_space);
        attr_nbins.write(H5::PredType::NATIVE_INT, &num_omega_bins);
        
        H5::Attribute attr_broad = meta.createAttribute("broadening", H5::PredType::NATIVE_DOUBLE, scalar_space);
        attr_broad.write(H5::PredType::NATIVE_DOUBLE, &broadening);
        
        H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
        H5::Attribute attr_method = meta.createAttribute("method", str_type, scalar_space);
        attr_method.write(str_type, method);
        
        H5::Attribute attr_optype = meta.createAttribute("operator_type", str_type, scalar_space);
        attr_optype.write(str_type, operator_type);
        
        meta.close();
        file.close();
        
        std::cout << "  Rank " << rank << ": created per-rank DSSF file: " << h5_path << std::endl;
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to create per-rank DSSF file: " + std::string(e.getCDetailMsg()));
    }
    
    return h5_path;
}

/**
 * @brief Merge per-rank DSSF HDF5 files into unified output
 */
bool mergePerRankDSSFFiles(const std::string& output_dir, int num_ranks) {
    std::string output_path = output_dir + "/dssf_results.h5";
    
    std::cout << "\n==========================================\n";
    std::cout << "Merging per-rank DSSF HDF5 files\n";
    std::cout << "==========================================\n";
    std::cout << "  Output: " << output_path << std::endl;
    
    int total_groups_merged = 0;
    
    for (int r = 0; r < num_ranks; ++r) {
        std::string rank_file = getPerRankDSSFPath(output_dir, r);
        
        if (!fs::exists(rank_file)) {
            std::cout << "  Rank " << r << ": file not found, skipping" << std::endl;
            continue;
        }
        
        try {
            H5::H5File source(rank_file, H5F_ACC_RDONLY);
            H5::H5File dest(output_path, H5F_ACC_RDWR);
            
            // Copy spectral groups
            if (source.nameExists("/spectral")) {
                H5::Group src_spectral = source.openGroup("/spectral");
                
                // Ensure destination group exists
                if (!dest.nameExists("/spectral")) {
                    dest.createGroup("/spectral");
                }
                
                // Copy frequencies if not already in destination
                if (source.nameExists("/spectral/frequencies") && 
                    !dest.nameExists("/spectral/frequencies")) {
                    H5Ocopy(source.getId(), "/spectral/frequencies",
                           dest.getId(), "/spectral/frequencies",
                           H5P_DEFAULT, H5P_DEFAULT);
                }
                
                // Iterate through operator groups and copy them
                hsize_t num_objs = src_spectral.getNumObjs();
                for (hsize_t i = 0; i < num_objs; ++i) {
                    std::string name = src_spectral.getObjnameByIdx(i);
                    if (name == "frequencies") continue;  // Already handled
                    
                    std::string src_path = "/spectral/" + name;
                    std::string dst_path = "/spectral/" + name;
                    
                    // Copy operator group (contains beta/T subgroups with samples)
                    if (!dest.nameExists(dst_path)) {
                        H5Ocopy(source.getId(), src_path.c_str(),
                               dest.getId(), dst_path.c_str(),
                               H5P_DEFAULT, H5P_DEFAULT);
                        total_groups_merged++;
                    } else {
                        // Merge into existing group by copying subgroups
                        H5::Group src_op = source.openGroup(src_path);
                        hsize_t num_temp_groups = src_op.getNumObjs();
                        for (hsize_t j = 0; j < num_temp_groups; ++j) {
                            std::string temp_name = src_op.getObjnameByIdx(j);
                            std::string src_temp_path = src_path + "/" + temp_name;
                            std::string dst_temp_path = dst_path + "/" + temp_name;
                            
                            if (!dest.nameExists(dst_temp_path)) {
                                H5Ocopy(source.getId(), src_temp_path.c_str(),
                                       dest.getId(), dst_temp_path.c_str(),
                                       H5P_DEFAULT, H5P_DEFAULT);
                                total_groups_merged++;
                            }
                        }
                        src_op.close();
                    }
                }
                src_spectral.close();
            }
            
            // Copy static groups similarly
            if (source.nameExists("/static")) {
                H5::Group src_static = source.openGroup("/static");
                
                if (!dest.nameExists("/static")) {
                    dest.createGroup("/static");
                }
                
                hsize_t num_objs = src_static.getNumObjs();
                for (hsize_t i = 0; i < num_objs; ++i) {
                    std::string name = src_static.getObjnameByIdx(i);
                    std::string src_path = "/static/" + name;
                    std::string dst_path = "/static/" + name;
                    
                    if (!dest.nameExists(dst_path)) {
                        H5Ocopy(source.getId(), src_path.c_str(),
                               dest.getId(), dst_path.c_str(),
                               H5P_DEFAULT, H5P_DEFAULT);
                        total_groups_merged++;
                    } else {
                        // Merge into existing operator group by copying sample subgroups
                        H5::Group src_op = source.openGroup(src_path);
                        hsize_t num_sample_groups = src_op.getNumObjs();
                        for (hsize_t j = 0; j < num_sample_groups; ++j) {
                            std::string sample_name = src_op.getObjnameByIdx(j);
                            std::string src_sample_path = src_path + "/" + sample_name;
                            std::string dst_sample_path = dst_path + "/" + sample_name;
                            
                            if (!dest.nameExists(dst_sample_path)) {
                                // Sample doesn't exist, copy the whole sample group
                                H5Ocopy(source.getId(), src_sample_path.c_str(),
                                       dest.getId(), dst_sample_path.c_str(),
                                       H5P_DEFAULT, H5P_DEFAULT);
                                total_groups_merged++;
                            } else {
                                // Sample exists, need to merge datasets (append temperature data)
                                // For SSSF, each sample contains: temperatures, expectation, variance, etc.
                                // We need to concatenate the arrays from source into destination
                                H5::Group src_sample = source.openGroup(src_sample_path);
                                H5::Group dst_sample = dest.openGroup(dst_sample_path);
                                
                                // Get list of datasets to merge
                                std::vector<std::string> datasets_to_merge = {
                                    "temperatures", "expectation", "expectation_error",
                                    "variance", "variance_error", "susceptibility", "susceptibility_error"
                                };
                                
                                for (const auto& ds_name : datasets_to_merge) {
                                    if (src_sample.nameExists(ds_name) && dst_sample.nameExists(ds_name)) {
                                        // Read source data
                                        H5::DataSet src_ds = src_sample.openDataSet(ds_name);
                                        H5::DataSpace src_space = src_ds.getSpace();
                                        hsize_t src_size;
                                        src_space.getSimpleExtentDims(&src_size);
                                        std::vector<double> src_data(src_size);
                                        src_ds.read(src_data.data(), H5::PredType::NATIVE_DOUBLE);
                                        src_ds.close();
                                        
                                        // Read destination data
                                        H5::DataSet dst_ds = dst_sample.openDataSet(ds_name);
                                        H5::DataSpace dst_space = dst_ds.getSpace();
                                        hsize_t dst_size;
                                        dst_space.getSimpleExtentDims(&dst_size);
                                        std::vector<double> dst_data(dst_size);
                                        dst_ds.read(dst_data.data(), H5::PredType::NATIVE_DOUBLE);
                                        dst_ds.close();
                                        
                                        // Combine data
                                        std::vector<double> combined_data;
                                        combined_data.reserve(dst_size + src_size);
                                        combined_data.insert(combined_data.end(), dst_data.begin(), dst_data.end());
                                        combined_data.insert(combined_data.end(), src_data.begin(), src_data.end());
                                        
                                        // Delete old dataset and create new one with combined data
                                        dst_sample.unlink(ds_name);
                                        hsize_t combined_size = combined_data.size();
                                        H5::DataSpace combined_space(1, &combined_size);
                                        H5::DataSet new_ds = dst_sample.createDataSet(
                                            ds_name, H5::PredType::NATIVE_DOUBLE, combined_space);
                                        new_ds.write(combined_data.data(), H5::PredType::NATIVE_DOUBLE);
                                        new_ds.close();
                                    }
                                }
                                
                                dst_sample.close();
                                src_sample.close();
                                total_groups_merged++;
                            }
                        }
                        src_op.close();
                    }
                }
                src_static.close();
            }
            
            source.close();
            dest.close();
            
            // Delete temporary per-rank file
            try {
                fs::remove(rank_file);
                std::cout << "  Rank " << r << ": merged and deleted" << std::endl;
            } catch (...) {
                std::cerr << "  Warning: Could not delete " << rank_file << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cerr << "  Error merging rank " << r << ": " << e.what() << std::endl;
        }
    }
    
    std::cout << "==========================================\n";
    std::cout << "Merge complete: " << total_groups_merged << " groups merged\n";
    std::cout << "==========================================\n";
    
    return true;
}

/**
 * @brief Initialize the unified DSSF HDF5 file with proper structure
 */
std::string initDSSFHDF5File(const std::string& output_dir, 
                              int num_sites, float spin_length,
                              const std::string& method, 
                              const std::string& operator_type,
                              double omega_min, double omega_max, 
                              int num_omega_bins, double broadening,
                              const std::vector<std::vector<double>>& momentum_points) {
    std::string h5_path = output_dir + "/dssf_results.h5";
    
    try {
        // Create file (truncate if exists) using parallel HDF5 if available
        H5::H5File file = openHDF5WithRetry(h5_path, H5F_ACC_TRUNC);
        
        // Create group structure
        file.createGroup("/metadata");
        file.createGroup("/momentum_points");
        file.createGroup("/spectral");
        file.createGroup("/static");
        file.createGroup("/correlations");
        
        // Save metadata as attributes
        H5::Group meta = file.openGroup("/metadata");
        H5::DataSpace scalar_space(H5S_SCALAR);
        
        // Integer attributes
        H5::Attribute attr_sites = meta.createAttribute("num_sites", H5::PredType::NATIVE_INT, scalar_space);
        attr_sites.write(H5::PredType::NATIVE_INT, &num_sites);
        
        H5::Attribute attr_spin = meta.createAttribute("spin_length", H5::PredType::NATIVE_FLOAT, scalar_space);
        attr_spin.write(H5::PredType::NATIVE_FLOAT, &spin_length);
        
        H5::Attribute attr_omega_min = meta.createAttribute("omega_min", H5::PredType::NATIVE_DOUBLE, scalar_space);
        attr_omega_min.write(H5::PredType::NATIVE_DOUBLE, &omega_min);
        
        H5::Attribute attr_omega_max = meta.createAttribute("omega_max", H5::PredType::NATIVE_DOUBLE, scalar_space);
        attr_omega_max.write(H5::PredType::NATIVE_DOUBLE, &omega_max);
        
        H5::Attribute attr_nbins = meta.createAttribute("num_omega_bins", H5::PredType::NATIVE_INT, scalar_space);
        attr_nbins.write(H5::PredType::NATIVE_INT, &num_omega_bins);
        
        H5::Attribute attr_broad = meta.createAttribute("broadening", H5::PredType::NATIVE_DOUBLE, scalar_space);
        attr_broad.write(H5::PredType::NATIVE_DOUBLE, &broadening);
        
        // String attributes
        H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
        H5::Attribute attr_method = meta.createAttribute("method", str_type, scalar_space);
        attr_method.write(str_type, method);
        
        H5::Attribute attr_optype = meta.createAttribute("operator_type", str_type, scalar_space);
        attr_optype.write(str_type, operator_type);
        
        meta.close();
        
        // Save momentum points as 2D dataset
        if (!momentum_points.empty()) {
            size_t n_q = momentum_points.size();
            hsize_t dims[2] = {n_q, 3};
            H5::DataSpace q_space(2, dims);
            
            // Flatten to contiguous array
            std::vector<double> q_data(n_q * 3);
            for (size_t i = 0; i < n_q; ++i) {
                q_data[i*3 + 0] = momentum_points[i][0];
                q_data[i*3 + 1] = momentum_points[i][1];
                q_data[i*3 + 2] = momentum_points[i][2];
            }
            
            H5::DataSet q_dataset = file.createDataSet("/momentum_points/q_vectors",
                                                       H5::PredType::NATIVE_DOUBLE, q_space);
            q_dataset.write(q_data.data(), H5::PredType::NATIVE_DOUBLE);
            q_dataset.close();
        }
        
        file.close();
        std::cout << "Created unified DSSF HDF5 file: " << h5_path << std::endl;
        
    } catch (H5::Exception& e) {
        throw std::runtime_error("Failed to create DSSF HDF5 file: " + std::string(e.getCDetailMsg()));
    }
    
    return h5_path;
}

/**
 * @brief Save dynamical spectral function to unified HDF5 file
 * 
 * @param h5_path Path to the unified HDF5 file
 * @param operator_name Name of the operator (e.g., "Sz_Sz_q_Qx0.5_Qy0_Qz0")
 * @param beta Inverse temperature (use INFINITY for ground state)
 * @param sample_idx Sample index
 * @param frequencies Frequency grid
 * @param spectral_real Real part of spectral function
 * @param spectral_imag Imaginary part of spectral function
 * @param error_real Error in real part
 * @param error_imag Error in imaginary part
 * @param temperature Temperature (optional, for thermal methods)
 */
void saveDSSFSpectralToHDF5(
    const std::string& h5_path,
    const std::string& operator_name,
    double beta,
    int sample_idx,
    const std::vector<double>& frequencies,
    const std::vector<double>& spectral_real,
    const std::vector<double>& spectral_imag,
    const std::vector<double>& error_real,
    const std::vector<double>& error_imag,
    double temperature = -1.0
) {
    std::lock_guard<std::mutex> lock(g_hdf5_mutex);
    
    try {
        // Use retry-based file opening for MPI safety
        H5::H5File file = openHDF5WithRetry(h5_path, H5F_ACC_RDWR);
        
        // Save frequencies once (shared across all operators)
        if (!file.nameExists("/spectral/frequencies")) {
            hsize_t dims[1] = {frequencies.size()};
            H5::DataSpace freq_space(1, dims);
            H5::DataSet freq_dataset = file.createDataSet("/spectral/frequencies",
                                                          H5::PredType::NATIVE_DOUBLE, freq_space);
            freq_dataset.write(frequencies.data(), H5::PredType::NATIVE_DOUBLE);
            freq_dataset.close();
        }
        
        // Create operator group if needed
        std::string op_path = "/spectral/" + operator_name;
        if (!file.nameExists(op_path)) {
            file.createGroup(op_path);
        }
        
        // Create beta/temperature group
        std::stringstream param_ss;
        if (temperature > 0) {
            param_ss << "T_" << std::fixed << std::setprecision(6) << temperature;
        } else if (std::isinf(beta)) {
            param_ss << "ground_state";
        } else {
            param_ss << "beta_" << std::fixed << std::setprecision(6) << beta;
        }
        std::string param_path = op_path + "/" + param_ss.str();
        if (!file.nameExists(param_path)) {
            file.createGroup(param_path);
        }
        
        // Create sample group
        std::string sample_path = param_path + "/sample_" + std::to_string(sample_idx);
        if (!file.nameExists(sample_path)) {
            file.createGroup(sample_path);
        }
        
        // Helper to save dataset
        auto saveDataset = [&](const std::string& name, const std::vector<double>& data) {
            if (data.empty()) return;
            std::string dataset_path = sample_path + "/" + name;
            if (file.nameExists(dataset_path)) {
                file.unlink(dataset_path);
            }
            hsize_t dims[1] = {data.size()};
            H5::DataSpace dspace(1, dims);
            H5::DataSet dset = file.createDataSet(dataset_path, H5::PredType::NATIVE_DOUBLE, dspace);
            dset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            dset.close();
        };
        
        saveDataset("real", spectral_real);
        saveDataset("imag", spectral_imag);
        saveDataset("error_real", error_real);
        saveDataset("error_imag", error_imag);
        
        // Add metadata attributes to sample group
        H5::Group sample_grp = file.openGroup(sample_path);
        H5::DataSpace scalar_space(H5S_SCALAR);
        
        if (!sample_grp.attrExists("beta")) {
            H5::Attribute beta_attr = sample_grp.createAttribute("beta", H5::PredType::NATIVE_DOUBLE, scalar_space);
            beta_attr.write(H5::PredType::NATIVE_DOUBLE, &beta);
        }
        if (temperature > 0 && !sample_grp.attrExists("temperature")) {
            H5::Attribute temp_attr = sample_grp.createAttribute("temperature", H5::PredType::NATIVE_DOUBLE, scalar_space);
            temp_attr.write(H5::PredType::NATIVE_DOUBLE, &temperature);
        }
        
        sample_grp.close();
        file.close();
        
    } catch (H5::Exception& e) {
        std::cerr << "Warning: Failed to save DSSF spectral to HDF5: " << e.getCDetailMsg() << std::endl;
    }
}

/**
 * @brief Save static structure factor to unified HDF5 file
 */
void saveDSSFStaticToHDF5(
    const std::string& h5_path,
    const std::string& operator_name,
    int sample_idx,
    const std::vector<double>& temperatures,
    const std::vector<double>& expectation,
    const std::vector<double>& expectation_error,
    const std::vector<double>& variance,
    const std::vector<double>& variance_error,
    const std::vector<double>& susceptibility,
    const std::vector<double>& susceptibility_error
) {
    std::lock_guard<std::mutex> lock(g_hdf5_mutex);
    
    try {
        // Use retry-based file opening for MPI safety  
        H5::H5File file = openHDF5WithRetry(h5_path, H5F_ACC_RDWR);
        
        // Create operator group if needed
        std::string op_path = "/static/" + operator_name;
        if (!file.nameExists(op_path)) {
            file.createGroup(op_path);
        }
        
        // Create sample group
        std::string sample_path = op_path + "/sample_" + std::to_string(sample_idx);
        if (!file.nameExists(sample_path)) {
            file.createGroup(sample_path);
        }
        
        // Helper to save or append dataset (for SSSF method which adds one T at a time)
        auto saveDataset = [&](const std::string& name, const std::vector<double>& data) {
            if (data.empty()) return;
            std::string dataset_path = sample_path + "/" + name;
            
            if (file.nameExists(dataset_path)) {
                // Read existing data and append new data
                H5::DataSet existing_dset = file.openDataSet(dataset_path);
                H5::DataSpace existing_space = existing_dset.getSpace();
                hsize_t existing_size;
                existing_space.getSimpleExtentDims(&existing_size);
                
                // Read existing data
                std::vector<double> existing_data(existing_size);
                existing_dset.read(existing_data.data(), H5::PredType::NATIVE_DOUBLE);
                existing_dset.close();
                
                // Combine existing and new data
                existing_data.insert(existing_data.end(), data.begin(), data.end());
                
                // Remove old dataset and create new with combined data
                file.unlink(dataset_path);
                hsize_t dims[1] = {existing_data.size()};
                H5::DataSpace dspace(1, dims);
                H5::DataSet dset = file.createDataSet(dataset_path, H5::PredType::NATIVE_DOUBLE, dspace);
                dset.write(existing_data.data(), H5::PredType::NATIVE_DOUBLE);
                dset.close();
            } else {
                // Create new dataset
                hsize_t dims[1] = {data.size()};
                H5::DataSpace dspace(1, dims);
                H5::DataSet dset = file.createDataSet(dataset_path, H5::PredType::NATIVE_DOUBLE, dspace);
                dset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
                dset.close();
            }
        };
        
        saveDataset("temperatures", temperatures);
        saveDataset("expectation", expectation);
        saveDataset("expectation_error", expectation_error);
        saveDataset("variance", variance);
        saveDataset("variance_error", variance_error);
        saveDataset("susceptibility", susceptibility);
        saveDataset("susceptibility_error", susceptibility_error);
        
        file.close();
        
        // MEMORY FIX: Force HDF5 to release internal caches
        // The HDF5 library keeps metadata and data caches that grow over time
        H5garbage_collect();
        
    } catch (H5::Exception& e) {
        std::cerr << "Warning: Failed to save DSSF static to HDF5: " << e.getCDetailMsg() << std::endl;
    }
}

/**
 * @brief Save spin correlation data to unified HDF5 file
 */
void saveDSSFCorrelationToHDF5(
    const std::string& h5_path,
    double beta,
    int sample_idx,
    const std::vector<std::vector<Complex>>& spin_plus_minus,
    const std::vector<std::vector<Complex>>& spin_z_z
) {
    std::lock_guard<std::mutex> lock(g_hdf5_mutex);
    
    try {
        // Use retry-based file opening for MPI safety
        H5::H5File file = openHDF5WithRetry(h5_path, H5F_ACC_RDWR);
        
        // Create beta group
        std::stringstream param_ss;
        if (std::isinf(beta)) {
            param_ss << "ground_state";
        } else {
            param_ss << "beta_" << std::fixed << std::setprecision(6) << beta;
        }
        std::string corr_path = "/correlations/" + param_ss.str();
        if (!file.nameExists(corr_path)) {
            file.createGroup(corr_path);
        }
        
        std::string sample_path = corr_path + "/sample_" + std::to_string(sample_idx);
        if (!file.nameExists(sample_path)) {
            file.createGroup(sample_path);
        }
        
        size_t n = spin_plus_minus.size();
        if (n == 0) {
            file.close();
            return;
        }
        
        // Save as flattened 2D arrays (real and imag separately)
        hsize_t dims[2] = {n, n};
        H5::DataSpace matrix_space(2, dims);
        
        // Flatten matrices
        std::vector<double> pm_real(n*n), pm_imag(n*n);
        std::vector<double> zz_real(n*n), zz_imag(n*n);
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                pm_real[i*n + j] = spin_plus_minus[i][j].real();
                pm_imag[i*n + j] = spin_plus_minus[i][j].imag();
                zz_real[i*n + j] = spin_z_z[i][j].real();
                zz_imag[i*n + j] = spin_z_z[i][j].imag();
            }
        }
        
        auto saveMatrix = [&](const std::string& name, const std::vector<double>& data) {
            std::string path = sample_path + "/" + name;
            if (file.nameExists(path)) file.unlink(path);
            H5::DataSet dset = file.createDataSet(path, H5::PredType::NATIVE_DOUBLE, matrix_space);
            dset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            dset.close();
        };
        
        saveMatrix("S_plus_minus_real", pm_real);
        saveMatrix("S_plus_minus_imag", pm_imag);
        saveMatrix("S_z_z_real", zz_real);
        saveMatrix("S_z_z_imag", zz_imag);
        
        file.close();
        
    } catch (H5::Exception& e) {
        std::cerr << "Warning: Failed to save DSSF correlation to HDF5: " << e.getCDetailMsg() << std::endl;
    }
}


// Helper function to read num_sites from positions.dat file
int read_num_sites_from_positions(const std::string& positions_file) {
    std::ifstream file(positions_file);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open positions.dat file: " + positions_file);
    }
    
    int num_sites = 0;
    std::string line;
    while (std::getline(file, line)) {
        // Skip comment lines starting with #
        if (line.empty() || line[0] == '#') continue;
        num_sites++;
    }
    
    file.close();
    
    if (num_sites == 0) {
        throw std::runtime_error("No sites found in positions.dat file: " + positions_file);
    }
    
    return num_sites;
}

// Helper function to read ground state energy from ed_results.h5
// Looks in /eigendata/eigenvalues (picks lowest) or /tpq/samples/sample_0/thermodynamics (picks lowest energy)
double read_ground_state_energy(const std::string& directory) {
    // Check possible locations for the HDF5 file
    std::vector<std::string> h5_paths = {
        directory + "/output/ed_results.h5",
        directory + "/ed_results.h5"
    };
    
    std::string h5_path;
    for (const auto& path : h5_paths) {
        if (std::filesystem::exists(path)) {
            h5_path = path;
            break;
        }
    }
    
    if (h5_path.empty()) {
        throw std::runtime_error("Could not find ed_results.h5 in " + directory + "/output/ or " + directory);
    }
    
    try {
        H5::H5File file(h5_path, H5F_ACC_RDONLY);
        
        // Disable automatic error printing for existence checks
        H5::Exception::dontPrint();
        
        // Method 1: Try to read from /eigendata/eigenvalues (from exact diagonalization)
        bool has_eigendata = false;
        try {
            has_eigendata = file.nameExists("/eigendata") && file.nameExists("/eigendata/eigenvalues");
        } catch (...) {
            has_eigendata = false;
        }
        
        if (has_eigendata) {
            H5::DataSet dataset = file.openDataSet("/eigendata/eigenvalues");
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims, nullptr);
            
            if (dims[0] > 0) {
                std::vector<double> eigenvalues(dims[0]);
                dataset.read(eigenvalues.data(), H5::PredType::NATIVE_DOUBLE);
                dataset.close();
                file.close();
                
                double ground_state_energy = *std::min_element(eigenvalues.begin(), eigenvalues.end());
                std::cout << "Ground state energy from " << h5_path << " (/eigendata/eigenvalues): "
                          << std::fixed << std::setprecision(10) << ground_state_energy << std::endl;
                return ground_state_energy;
            }
            dataset.close();
        }
        
        // Method 2: Try to read from /tpq/samples/sample_0/thermodynamics (from TPQ)
        bool has_tpq_thermo = false;
        try {
            has_tpq_thermo = file.nameExists("/tpq") && 
                            file.nameExists("/tpq/samples") && 
                            file.nameExists("/tpq/samples/sample_0") &&
                            file.nameExists("/tpq/samples/sample_0/thermodynamics");
        } catch (...) {
            has_tpq_thermo = false;
        }
        
        if (has_tpq_thermo) {
            H5::DataSet dataset = file.openDataSet("/tpq/samples/sample_0/thermodynamics");
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims, nullptr);
            
            if (dims[0] > 0) {
                // Read the thermodynamics data (columns: beta, energy, variance, doublon, step)
                std::vector<double> data(dims[0] * dims[1]);
                dataset.read(data.data(), H5::PredType::NATIVE_DOUBLE);
                dataset.close();
                file.close();
                
                // Find minimum energy (column 1, 0-indexed)
                double min_energy = std::numeric_limits<double>::max();
                size_t num_cols = dims[1];
                for (size_t i = 0; i < dims[0]; ++i) {
                    double energy = data[i * num_cols + 1];  // Energy is in column 1
                    if (energy < min_energy) {
                        min_energy = energy;
                    }
                }
                
                std::cout << "Ground state energy from " << h5_path << " (/tpq/samples/sample_0/thermodynamics): "
                          << std::fixed << std::setprecision(10) << min_energy << std::endl;
                return min_energy;
            }
            dataset.close();
        }
        
        file.close();
        throw std::runtime_error("No eigenvalues or TPQ thermodynamics data found in " + h5_path);
        
    } catch (const H5::Exception& e) {
        throw std::runtime_error("Failed to read ground state energy from " + h5_path + ": " + e.getCDetailMsg());
    }
}

void printSpinConfiguration(ComplexVector &state, int num_sites, float spin_length, const std::string &dir) {
    // Compute and print <S_i> for all sites
    std::vector<std::vector<Complex>> result(num_sites, std::vector<Complex>(3));

    for (int i = 0; i < num_sites; i++) {
        SingleSiteOperator S_plus(num_sites, spin_length, 0, i);
        SingleSiteOperator S_minus(num_sites, spin_length, 1, i);
        SingleSiteOperator S_z(num_sites, spin_length, 2, i);

        ComplexVector temp_plus(state.size(), Complex(0.0, 0.0));
        ComplexVector temp_minus(state.size(), Complex(0.0, 0.0));
        ComplexVector temp_z(state.size(), Complex(0.0, 0.0));

        S_plus.apply(state.data(), temp_plus.data(), state.size());
        S_minus.apply(state.data(), temp_minus.data(), state.size());
        S_z.apply(state.data(), temp_z.data(), state.size());

        Complex expectation_plus = 0.0;
        Complex expectation_minus = 0.0;
        Complex expectation_z = 0.0;
        for (size_t k = 0; k < state.size(); k++) {
            expectation_plus += std::conj(state[k]) * temp_plus[k];
            expectation_minus += std::conj(state[k]) * temp_minus[k];
            expectation_z += std::conj(state[k]) * temp_z[k];
        }
        result[i][0] = expectation_plus;
        result[i][1] = expectation_minus;
        result[i][2] = expectation_z;
    }
    // Write results to file
    std::ofstream outfile(dir + "/spin_configuration.txt");
    outfile << std::fixed << std::setprecision(6);
    outfile << "Site S+ S- Sz\n";
    for (int i = 0; i < num_sites; i++) {
        outfile << i << " " << result[i][0] << " " << result[i][1] << " " << result[i][2] << "\n";
    }
    outfile.close();
}

void printSpinCorrelation(ComplexVector &state, int num_sites, float spin_length, const std::string &dir, int unit_cell_size) {
    // Compute and print <S_i . S_j> for all pairs (i,j)
    std::vector<std::vector<std::vector<Complex>>> result(2, std::vector<std::vector<Complex>>(num_sites, std::vector<Complex>(num_sites)));

    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            SingleSiteOperator S_plus_i(num_sites, spin_length, 0, i);
            SingleSiteOperator S_plus_j(num_sites, spin_length, 0, j);
            SingleSiteOperator S_z_i(num_sites, spin_length, 2, i);
            SingleSiteOperator S_z_j(num_sites, spin_length, 2, j);

            ComplexVector temp_plus_i(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_z_i(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_plus_j(state.size(), Complex(0.0, 0.0));
            ComplexVector temp_z_j(state.size(), Complex(0.0, 0.0));

            S_plus_i.apply(state.data(), temp_plus_i.data(), state.size());
            S_z_i.apply(state.data(), temp_z_i.data(), state.size());
            S_plus_j.apply(state.data(), temp_plus_j.data(), state.size());
            S_z_j.apply(state.data(), temp_z_j.data(), state.size());

            Complex expectation_plus = 0.0;
            for (size_t k = 0; k < state.size(); k++) {
                expectation_plus += std::conj(temp_plus_i[k]) * temp_plus_j[k];
            }
            result[0][i][j] = expectation_plus;
            Complex expectation_z = 0.0;
            for (size_t k = 0; k < state.size(); k++) {
                expectation_z += std::conj(temp_z_i[k]) * temp_z_j[k];
            }
            result[1][i][j] = expectation_z;
        }
    }
    // Write results to file
    std::ofstream outfile(dir + "/spin_correlation.txt");
    outfile << std::fixed << std::setprecision(6);
    outfile << "i j <S+_i S-_j> <Sz_i Sz_j>\n";
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            outfile << i << " " << j << " " << result[0][i][j] << " " << result[1][i][j] << "\n";
        }
    }
    outfile.close();

    // Print sublattice correlations
    std::ofstream subfile(dir + "/sublattice_correlation.txt");
    subfile << std::fixed << std::setprecision(6);
    subfile << "sub_i sub_j <S+_i S-_j>_sum <Sz_i Sz_j>_sum count\n";
    
    // Compute sublattice sums
    std::vector<std::vector<Complex>> sublattice_sums_plus(unit_cell_size, std::vector<Complex>(unit_cell_size, 0.0));
    std::vector<std::vector<Complex>> sublattice_sums_z(unit_cell_size, std::vector<Complex>(unit_cell_size, 0.0));
    std::vector<std::vector<int>> sublattice_counts(unit_cell_size, std::vector<int>(unit_cell_size, 0));
    
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            int sub_i = i % unit_cell_size;
            int sub_j = j % unit_cell_size;
            sublattice_sums_plus[sub_i][sub_j] += result[0][i][j];
            sublattice_sums_z[sub_i][sub_j] += result[1][i][j];
            sublattice_counts[sub_i][sub_j]++;
        }
    }
    
    // Write sublattice results
    for (int sub_i = 0; sub_i < unit_cell_size; sub_i++) {
        for (int sub_j = 0; sub_j < unit_cell_size; sub_j++) {
            subfile << sub_i << " " << sub_j << " " 
                   << sublattice_sums_plus[sub_i][sub_j] << " "
                   << sublattice_sums_z[sub_i][sub_j] << " "
                   << sublattice_counts[sub_i][sub_j] << "\n";
        }
    }
    subfile.close();

    // Print total sums for verification
    std::ofstream sumfile(dir + "/total_sums.txt");
    sumfile << std::fixed << std::setprecision(6);

    Complex total_plus_sum = 0.0;
    Complex total_z_sum = 0.0;
    for (int i = 0; i < num_sites; i++) {
        for (int j = 0; j < num_sites; j++) {
            total_plus_sum += result[0][i][j];
            total_z_sum += result[1][i][j];
        }
    }

    sumfile << "Total <S+_i S-_j> sum: " << total_plus_sum << "\n";
    sumfile << "Total <Sz_i Sz_j> sum: " << total_z_sum << "\n";
    sumfile.close();

    std::cout << "Total correlation sums:" << std::endl;
    std::cout << "  <S+_i S-_j> sum: " << total_plus_sum << std::endl;
    std::cout << "  <Sz_i Sz_j> sum: " << total_z_sum << std::endl;

    std::cout << "Spin correlation data saved to spin_correlation.txt" << std::endl;
}

// Helper function to load eigenvector from HDF5 file
// Returns true if successful, false otherwise
bool load_eigenvector_from_hdf5(ComplexVector& state, const std::string& h5_path, int eigenvector_idx) {
    try {
        if (!std::filesystem::exists(h5_path)) {
            return false;
        }
        
        H5::H5File file(h5_path, H5F_ACC_RDONLY);
        
        std::string dataset_name = "/eigendata/eigenvector_" + std::to_string(eigenvector_idx);
        if (!file.nameExists(dataset_name)) {
            file.close();
            return false;
        }
        
        H5::DataSet dataset = file.openDataSet(dataset_name);
        H5::DataSpace dataspace = dataset.getSpace();
        hsize_t dims[1];
        dataspace.getSimpleExtentDims(dims, nullptr);
        
        // Eigenvectors are stored as complex numbers (2 doubles per element)
        size_t num_elements = dims[0];
        
        // Create compound datatype for complex numbers matching HDF5IO format
        // HDF5IO uses member names "real" and "imag"
        H5::CompType complex_type(2 * sizeof(double));
        complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
        complex_type.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);
        
        struct ComplexPair {
            double real;
            double imag;
        };
        
        std::vector<ComplexPair> data(num_elements);
        dataset.read(data.data(), complex_type);
        
        state.resize(num_elements);
        for (size_t i = 0; i < num_elements; ++i) {
            state[i] = Complex(data[i].real, data[i].imag);
        }
        
        dataset.close();
        file.close();
        
        std::cout << "Loaded eigenvector " << eigenvector_idx << " from HDF5: " << h5_path 
                  << " (size: " << state.size() << ")" << std::endl;
        return true;
    } catch (const H5::Exception& e) {
        std::cerr << "Warning: HDF5 error loading eigenvector: " << e.getCDetailMsg() << std::endl;
        return false;
    } catch (const std::exception& e) {
        std::cerr << "Warning: Error loading eigenvector from HDF5: " << e.what() << std::endl;
        return false;
    }
}

// Helper function to find HDF5 file containing eigenvectors
// Returns empty string if not found
// Eigenvectors are stored at /eigendata/eigenvector_0 in ed_results.h5
std::string find_eigenvector_hdf5(const std::string& directory) {
    std::string h5_path = directory + "/output/ed_results.h5";
    
    if (std::filesystem::exists(h5_path)) {
        try {
            H5::H5File file(h5_path, H5F_ACC_RDONLY);
            if (file.nameExists("/eigendata/eigenvector_0")) {
                file.close();
                return h5_path;
            }
            file.close();
        } catch (...) {
            // File exists but couldn't be read
        }
    }
    return "";
}

int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Pre-process arguments to extract flags (--use_gpu, --help, etc.)
    // These can appear anywhere in the command line
    bool use_gpu = false;
    bool show_help = false;
    std::vector<char*> positional_args;
    positional_args.push_back(argv[0]);  // Program name is always first
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--use_gpu" || arg == "--gpu" || arg == "-g") {
            use_gpu = true;
        } else if (arg == "--help" || arg == "-h") {
            show_help = true;
        } else if (arg.substr(0, 2) != "--") {
            // Not a flag, treat as positional argument
            positional_args.push_back(argv[i]);
        } else {
            // Unknown flag - treat as positional for backward compatibility
            positional_args.push_back(argv[i]);
        }
    }
    
    // Update argc and argv to use filtered positional arguments
    int pos_argc = positional_args.size();
    char** pos_argv = positional_args.data();
    
#ifndef WITH_CUDA
    if (use_gpu && rank == 0) {
        std::cerr << "Warning: GPU requested but code not compiled with GPU support (WITH_CUDA). Using CPU." << std::endl;
        use_gpu = false;
    }
#endif
    
    if (show_help || pos_argc < 4 || pos_argc > 14) {
        if (rank == 0) {
            std::cerr << "Usage: " << pos_argv[0] << " <directory> <krylov_dim_or_nmax> <spin_combinations> [options...]" << std::endl;
            std::cerr << "\nFLAGS (can appear anywhere):" << std::endl;
            std::cerr << "  --use_gpu, --gpu, -g  Enable GPU acceleration (requires CUDA build)" << std::endl;
            std::cerr << "  --help, -h            Show this help message" << std::endl;
            std::cerr << "\nNote: num_sites is automatically detected from positions.dat, spin_length is fixed at 0.5" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "REQUIRED ARGUMENTS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "  directory: Path containing InterAll.dat, Trans.dat, and positions.dat files" << std::endl;
            std::cerr << "  krylov_dim_or_nmax: Dimension of Krylov/Lanczos subspace for spectral methods" << std::endl;
            std::cerr << "  spin_combinations: Format \"op1,op2;op3,op4;...\" where op is:" << std::endl;
            std::cerr << "    - ladder basis: 0=Sp, 1=Sm, 2=Sz" << std::endl;
            std::cerr << "    - xyz basis: 0=Sx, 1=Sy, 2=Sz" << std::endl;
            std::cerr << "    - Example: \"0,1;2,2\" for SpSm/SxSy, SzSz combinations" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "OPTIONAL ARGUMENTS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "  method (default: spectral): spectral | ftlm_thermal | static | sssf | ground_state | continued_fraction" << std::endl;
            std::cerr << "    - spectral: Frequency-domain spectral function S(ω) via Lanczos eigendecomposition (single state)" << std::endl;
            std::cerr << "    - ftlm_thermal: TRUE thermal DSSF with random state sampling (FTLM multi-sample)" << std::endl;
            std::cerr << "    - static: Static structure factor ⟨O₁†O₂⟩ vs temperature using FTLM (random sampling)" << std::endl;
            std::cerr << "    - sssf: Static structure factor ⟨O₁†O₂⟩ evaluated on pre-computed TPQ states" << std::endl;
            std::cerr << "    - ground_state: Ground state spectral function using continued fraction" << std::endl;
            std::cerr << "    - continued_fraction: TRUE continued fraction on TPQ states (O₁=O₂ only, O(M) per ω)" << std::endl;
            std::cerr << "\n  operator_type (default: sum): sum | transverse | sublattice | experimental | transverse_experimental" << std::endl;
            std::cerr << "    - sum: Standard momentum-resolved sum operators S^{op1}(Q) S^{op2}(-Q)" << std::endl;
            std::cerr << "    - transverse: Polarization-dependent operators for magnetic scattering" << std::endl;
            std::cerr << "    - sublattice: Sublattice-resolved correlations (use unit_cell_size)" << std::endl;
            std::cerr << "    - experimental: General form cos(θ)Sz + sin(θ)Sx (use theta)" << std::endl;
            std::cerr << "    - transverse_experimental: Transverse version with experimental angle" << std::endl;
            std::cerr << "\n  basis (default: ladder): ladder | xyz" << std::endl;
            std::cerr << "    - ladder: Use Sp/Sm/Sz operators (raising/lowering operators)" << std::endl;
            std::cerr << "    - xyz: Use Sx/Sy/Sz operators (Cartesian components)" << std::endl;
            std::cerr << "    - Note: experimental operator type always uses xyz basis internally" << std::endl;
            std::cerr << "\n  spectral_params (format: \"omega_min,omega_max,num_omega_bins,broadening\" e.g., \"-5.0,5.0,200,0.1\"):" << std::endl;
            std::cerr << "      * omega_min: minimum frequency" << std::endl;
            std::cerr << "      * omega_max: maximum frequency" << std::endl;
            std::cerr << "      * num_omega_bins: number of frequency points (resolution)" << std::endl;
            std::cerr << "      * broadening: Lorentzian broadening parameter (eta)" << std::endl;
            std::cerr << "\n  unit_cell_size (for sublattice operators): number of sublattices (default: 4)" << std::endl;
            std::cerr << "  momentum_points (default: (0,0,0);(0,0,2π)): \"Qx1,Qy1,Qz1;Qx2,Qy2,Qz2;...\"" << std::endl;
            std::cerr << "  polarization (for transverse operators): \"px,py,pz\" normalized vector (default: (1/√2,-1/√2,0))" << std::endl;
            std::cerr << "  theta (for experimental operators): angle in radians (default: 0.0)" << std::endl;
            std::cerr << "  n_up (optional): number of up spins for fixed-Sz sector (default: -1 = use full Hilbert space)" << std::endl;
            std::cerr << "    - When n_up >= 0: restricts to fixed total Sz = n_up - n_down = n_up - (num_sites - n_up)" << std::endl;
            std::cerr << "    - Reduces Hilbert space dimension from 2^N to C(N, n_up)" << std::endl;
            std::cerr << "    - Example: for 16 sites with n_up=8, dimension reduces from 65536 to 12870" << std::endl;
            std::cerr << "  T_min,T_max,T_steps (for ftlm_thermal/static): Temperature scan parameters" << std::endl;
            std::cerr << "    - Format: \"T_min,T_max,T_steps\" e.g., \"0.1,10.0,10\"" << std::endl;
            std::cerr << "    - T_min: Minimum temperature" << std::endl;
            std::cerr << "    - T_max: Maximum temperature" << std::endl;
            std::cerr << "    - T_steps: Number of temperature points (logarithmic spacing)" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "SPECTRAL METHOD (method=spectral) DETAILS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "The spectral methods compute the dynamical structure factor (DSF):" << std::endl;
            std::cerr << "  S(q,ω) = (1/π) × Im[⟨ψ|O₁†(q) G(ω) O₂(q)|ψ⟩]" << std::endl;
            std::cerr << "where G(ω) = 1/(ω - H + iη) is the Green's function." << std::endl;
            std::cerr << "\nThis method computes:" << std::endl;
            std::cerr << "  - Single-state calculation (for T=0 or specific TPQ states)" << std::endl;
            std::cerr << "  - Uses the given state |ψ⟩ directly" << std::endl;
            std::cerr << "  - Fastest option for ground state calculations" << std::endl;
            std::cerr << "  - For finite-T, use ftlm_thermal instead" << std::endl;
            std::cerr << "\nParameters:" << std::endl;
            std::cerr << "  krylov_dim_or_nmax: Lanczos order (typical values: 30-100)" << std::endl;
            std::cerr << "    - Higher values = better convergence but more computational cost" << std::endl;
            std::cerr << "    - Recommended: 50-100 for most systems" << std::endl;
            std::cerr << "\n  Broadening (eta) in spectral parameters:" << std::endl;
            std::cerr << "    - Controls smoothing of spectral features (peak width)" << std::endl;
            std::cerr << "    - Small values (0.01-0.05): High resolution, shows sharp features" << std::endl;
            std::cerr << "    - Medium values (0.1): Balanced (default), good for most cases" << std::endl;
            std::cerr << "    - Large values (0.2-0.5): Smoothed, reduces noise artifacts" << std::endl;
            std::cerr << "\n  Frequency resolution:" << std::endl;
            std::cerr << "    - num_omega_bins: Number of frequency points" << std::endl;
            std::cerr << "    - Larger values give finer frequency resolution" << std::endl;
            std::cerr << "    - Typical values: 100-500 depending on energy scale" << std::endl;
            std::cerr << "\nOutput files:" << std::endl;
            std::cerr << "  - spectral method: <operator>_spectral_sample_<idx>_beta_<beta>.txt" << std::endl;
            std::cerr << "  - Columns: frequency(ω) | spectral_intensity S(ω) | error (if applicable)" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "FTLM THERMAL METHOD (method=ftlm_thermal) DETAILS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "True finite-temperature dynamical structure factor using FTLM random sampling:" << std::endl;
            std::cerr << "  S(q,ω,T) = (1/Z) Tr[e^{-βH} O₁†(q) δ(ω-H+E₀) O₂(q)]" << std::endl;
            std::cerr << "\nThis is the most accurate method for finite-temperature spectra:" << std::endl;
            std::cerr << "  - Uses random state sampling with thermal averaging" << std::endl;
            std::cerr << "  - Properly captures thermal population effects" << std::endl;
            std::cerr << "  - Provides error bars from sample statistics" << std::endl;
            std::cerr << "\nParameters:" << std::endl;
            std::cerr << "  - Number of samples: 40 (for thermal averaging)" << std::endl;
            std::cerr << "  - Computes for all temperatures in T_min,T_max,T_steps range" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "STATIC METHOD (method=static) DETAILS - FTLM SSSF:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "Static structure factor S(q) = ⟨O₁†(q) O₂(q)⟩ vs temperature (FTLM):" << std::endl;
            std::cerr << "  - Computes equal-time correlation ⟨O₁†O₂⟩(T)" << std::endl;
            std::cerr << "  - Uses FTLM random sampling for thermal averaging" << std::endl;
            std::cerr << "  - Also computes fluctuation/susceptibility χ = β⟨(O†O - ⟨O†O⟩)²⟩" << std::endl;
            std::cerr << "\nOutput: Temperature | ⟨O₁†O₂⟩ | error | variance | χ" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "SSSF METHOD (method=sssf) DETAILS - TPQ SSSF:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "Static structure factor S(q) = ⟨O₁†(q) O₂(q)⟩ using pre-computed states:" << std::endl;
            std::cerr << "  - Computes equal-time correlation ⟨ψ|O₁†O₂|ψ⟩ for each TPQ state" << std::endl;
            std::cerr << "  - Requires pre-computed TPQ states (from TPQ run)" << std::endl;
            std::cerr << "  - Each state corresponds to a specific β (inverse temperature)" << std::endl;
            std::cerr << "  - Similar to 'spectral' but computes static instead of dynamical response" << std::endl;
            std::cerr << "\nOutput: For each (q, operator, beta): ⟨O₁†O₂⟩ expectation value" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "GROUND STATE METHOD (method=ground_state) DETAILS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "Ground state (T=0) dynamical structure factor using continued fraction:" << std::endl;
            std::cerr << "  S(q,ω) = ⟨GS|O₁†(q) δ(ω-H+E₀) O₂(q)|GS⟩" << std::endl;
            std::cerr << "\n  - Requires pre-computed ground state eigenvector" << std::endl;
            std::cerr << "  - Optimal for large systems where diagonalization is expensive" << std::endl;
            std::cerr << "  - Uses memory-efficient continued fraction representation" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "CONTINUED FRACTION METHOD (method=continued_fraction) DETAILS:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "TRUE continued fraction representation for single-state spectral functions:" << std::endl;
            std::cerr << "  G(z) = ||O|ψ⟩||² / (z - α₀ - β₁²/(z - α₁ - β₂²/(z - α₂ - ...)))" << std::endl;
            std::cerr << "  S(ω) = -(1/π) × Im[G(ω + iη)]" << std::endl;
            std::cerr << "\nKey differences from 'spectral' method:" << std::endl;
            std::cerr << "  - Uses continued fraction evaluation: O(M) per frequency point" << std::endl;
            std::cerr << "  - 'spectral' uses eigendecomposition: O(M³) one-time cost, then O(M) per ω" << std::endl;
            std::cerr << "  - continued_fraction is more memory-efficient (no eigenvector storage)" << std::endl;
            std::cerr << "\nLIMITATION: Only works for self-correlations (O₁ = O₂)!" << std::endl;
            std::cerr << "  - Valid: \"0,0\", \"1,1\", \"2,2\" (S+S+, S-S-, SzSz)" << std::endl;
            std::cerr << "  - Invalid: \"0,1\", \"0,2\", etc. (use 'spectral' for cross-correlations)" << std::endl;
            std::cerr << "\nIMPORTANT for finite-temperature (TPQ states):" << std::endl;
            std::cerr << "  - At high T (low β), continued_fraction gives different results than spectral" << std::endl;
            std::cerr << "  - At low T (high β → ground state), methods converge" << std::endl;
            std::cerr << "  - For thermal averaging, use 'spectral' or 'ftlm_thermal'" << std::endl;
            std::cerr << "\nWhen to use:" << std::endl;
            std::cerr << "  - Ground state or near-ground-state calculations (β > 50)" << std::endl;
            std::cerr << "  - Very large Krylov dimensions where eigendecomp memory is limiting" << std::endl;
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "EXAMPLES:" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "1. Spectral function with momentum resolution (default, single state):" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 50 \"2,2\"" << std::endl;
            std::cerr << "\n2. Spectral function with custom parameters:" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 50 \"2,2\" spectral sum ladder \"-5,5,200,0.1\" 4 \"0,0,0;0,0,1\"" << std::endl;
            std::cerr << "\n3. Transverse scattering with custom polarization:" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 40 \"2,2\" spectral transverse xyz \"-10,10,300,0.2\" 4 \"0,0,0\" \"1,0,0\"" << std::endl;
            std::cerr << "\n4. Experimental geometry with angle:" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 50 \"2,2\" spectral experimental xyz \"-5,5,250,0.1\" 4 \"0,0,0\" \"0,0,0\" 0.785" << std::endl;
            std::cerr << "\n5. Fixed-Sz sector calculation (Sz = 0 for 16 sites):" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 50 \"2,2\" spectral sum ladder \"-5,5,200,0.1\" 4 \"0,0,0\" \"0,0,0\" 0 8" << std::endl;
            std::cerr << "\n6. Finite-T DSSF with FTLM random sampling (with GPU):" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 50 \"2,2\" ftlm_thermal --use_gpu \"0.1,10.0,10\"" << std::endl;
            std::cerr << "\n7. Static structure factor via FTLM (random sampling) vs temperature:" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 50 \"2,2\" static \"0.1,10.0,20\"" << std::endl;
            std::cerr << "\n8. Static structure factor on pre-computed TPQ states (sssf):" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 50 \"2,2\" sssf sum ladder \"0,0,0\" 4 \"0,0,0;0,0,1\"" << std::endl;
            std::cerr << "\n9. Ground state DSSF (T=0, requires eigenvector):" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 100 \"2,2\" ground_state sum ladder \"-5,5,200,0.05\" 4 \"0,0,0;0,0,1\"" << std::endl;
            std::cerr << "\n10. GPU-accelerated FTLM thermal DSSF:" << std::endl;
            std::cerr << "   " << pos_argv[0] << " --use_gpu ./data 50 \"2,2\" ftlm_thermal" << std::endl;
            std::cerr << "\n11. Continued fraction DSSF on TPQ states (finite-T with continued fraction):" << std::endl;
            std::cerr << "   " << pos_argv[0] << " ./data 100 \"2,2\" continued_fraction sum ladder \"-5,5,200,0.05\" 4 \"0,0,0;0,0,1\"" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    std::string directory = pos_argv[1];
    int krylov_dim_or_nmax = std::stoi(pos_argv[2]);
    std::string spin_combinations_str = pos_argv[3];
    std::string method = (pos_argc >= 5) ? std::string(pos_argv[4]) : std::string("spectral");
    std::string operator_type = (pos_argc >= 6) ? std::string(pos_argv[5]) : std::string("sum");
    std::string basis = (pos_argc >= 7) ? std::string(pos_argv[6]) : std::string("ladder");
    
    // Read num_sites from positions.dat and set spin_length to 0.5
    std::string positions_file = directory + "/positions.dat";
    int num_sites;
    try {
        num_sites = read_num_sites_from_positions(positions_file);
    } catch (const std::exception& e) {
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    float spin_length = 0.5f;
    
    // Experimental operators always use XYZ basis internally
    if ((operator_type == "experimental" || operator_type == "transverse_experimental") && basis != "xyz") {
        if (rank == 0) {
            std::cout << "Note: " << operator_type << " operator type requires xyz basis, setting basis=xyz" << std::endl;
        }
        basis = "xyz";
    }
    
    // Spectral method parameters (default values)
    double omega_min = -5.0;
    double omega_max = 5.0;
    int num_omega_bins = 200;
    double broadening = 0.1;
    
    if (pos_argc >= 8) {
        std::string param_str = pos_argv[7];
        
        if (method == "spectral") {
            // Parse omega_min,omega_max,num_omega_bins,broadening for spectral method
            std::stringstream ss(param_str);
            std::string val;
            std::vector<std::string> tokens;
            while (std::getline(ss, val, ',')) {
                tokens.push_back(val);
            }
            
            try {
                if (tokens.size() >= 1) omega_min = std::stod(tokens[0]);
                if (tokens.size() >= 2) omega_max = std::stod(tokens[1]);
                if (tokens.size() >= 3) num_omega_bins = std::stoi(tokens[2]);
                if (tokens.size() >= 4) broadening = std::stod(tokens[3]);
                if (rank == 0) {
                    std::cout << "Spectral method parameters: omega=[" << omega_min << "," << omega_max 
                              << "], bins=" << num_omega_bins << ", broadening=" << broadening << std::endl;
                }
            } catch (...) {
                if (rank == 0) {
                    std::cerr << "Warning: failed to parse spectral parameters. Using defaults." << std::endl;
                }
            }
        }
    }

    int unit_cell_size = 4; // Default for pyrochlore
    if (pos_argc >= 9) {
        try { unit_cell_size = std::stoi(pos_argv[8]); } catch (...) { unit_cell_size = 4; }
    }

    // Parse momentum points
    std::vector<std::vector<double>> momentum_points;
    if (pos_argc >= 10) {
        std::string momentum_str = pos_argv[9];
        std::stringstream mom_ss(momentum_str);
        std::string point_str;
        
        while (std::getline(mom_ss, point_str, ';')) {
            std::stringstream point_ss(point_str);
            std::string coord_str;
            std::vector<double> point;
            
            while (std::getline(point_ss, coord_str, ',')) {
                try {
                    double coord = std::stod(coord_str);
                    coord *= M_PI;  // Scale to π
                    point.push_back(coord);
                } catch (...) {
                    if (rank == 0) {
                        std::cerr << "Warning: Failed to parse momentum coordinate: " << coord_str << std::endl;
                    }
                }
            }
            
            if (point.size() == 3) {
                momentum_points.push_back(point);
            } else if (rank == 0) {
                std::cerr << "Warning: Momentum point must have 3 coordinates, got " << point.size() << std::endl;
            }
        }
    }
    
    // Use default momentum points if none provided or parsing failed
    if (momentum_points.empty()) {
        momentum_points = {
            {0.0, 0.0, 0.0},
            {0.0, 0.0, 2.0 * M_PI}
        };
        if (rank == 0) {
            std::cout << "Using default momentum points: (0,0,0) and (0,0,2π)" << std::endl;
        }
    }

    // Parse polarization vector for transverse operators
    std::vector<double> polarization = {1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0};
    if (pos_argc >= 11) {
        std::string pol_str = pos_argv[10];
        std::stringstream pol_ss(pol_str);
        std::string coord_str;
        std::vector<double> pol_temp;
        
        while (std::getline(pol_ss, coord_str, ',')) {
            try {
                double coord = std::stod(coord_str);
                pol_temp.push_back(coord);
            } catch (...) {
                if (rank == 0) {
                    std::cerr << "Warning: Failed to parse polarization coordinate: " << coord_str << std::endl;
                }
            }
        }
        
        if (pol_temp.size() == 3) {
            // Normalize the polarization vector
            double norm = std::sqrt(pol_temp[0]*pol_temp[0] + pol_temp[1]*pol_temp[1] + pol_temp[2]*pol_temp[2]);
            if (norm > 1e-10) {
                polarization = {pol_temp[0]/norm, pol_temp[1]/norm, pol_temp[2]/norm};
                if (rank == 0) {
                    std::cout << "Using custom polarization: (" << polarization[0] << "," 
                              << polarization[1] << "," << polarization[2] << ")" << std::endl;
                }
            } else if (rank == 0) {
                std::cerr << "Warning: Polarization vector has zero norm, using default" << std::endl;
            }
        } else if (rank == 0) {
            std::cerr << "Warning: Polarization must have 3 coordinates, got " << pol_temp.size() << std::endl;
        }
    }

    // Parse theta for experimental operators
    double theta = 0.0;  // Default to 0
    if (pos_argc >= 12) {
        try {
            theta = std::stod(pos_argv[11]);
            theta *= M_PI;
            if (rank == 0) {
                std::cout << "Using theta = " << theta << " radians" << std::endl;
            }
        } catch (...) {
            if (rank == 0) {
                std::cerr << "Warning: Failed to parse theta, using default 0.0" << std::endl;
            }
            theta = 0.0;
        }
    }

    // Print GPU status (use_gpu was already set during flag pre-processing)
    if (rank == 0 && use_gpu) {
        std::cout << "GPU acceleration enabled" << std::endl;
    }

    // Parse n_up for fixed-Sz sector (optional) - now at position 12
    int n_up = -1;  // Default: -1 means use full Hilbert space
    bool use_fixed_sz = false;
    if (pos_argc >= 13) {
        try {
            n_up = std::stoi(pos_argv[12]);
            if (n_up == -1) {
                // Explicitly using full Hilbert space
                use_fixed_sz = false;
            } else if (n_up >= 0 && n_up <= num_sites) {
                use_fixed_sz = true;
                if (rank == 0) {
                    std::cout << "Using fixed-Sz sector: n_up = " << n_up 
                              << ", Sz = " << (n_up - (num_sites - n_up)) * 0.5 << std::endl;
                    // Calculate and display dimension
                    size_t fixed_sz_dim = 1;
                    for (int i = 0; i < n_up; ++i) {
                        fixed_sz_dim = fixed_sz_dim * (num_sites - i) / (i + 1);
                    }
                    std::cout << "Fixed-Sz dimension: " << fixed_sz_dim 
                              << " (reduced from " << (1ULL << num_sites) << ")" << std::endl;
                }
            } else {
                if (rank == 0) {
                    std::cerr << "Warning: Invalid n_up value " << n_up 
                              << " (must be -1 or 0 <= n_up <= " << num_sites << "), using full Hilbert space" << std::endl;
                }
                n_up = -1;
                use_fixed_sz = false;
            }
        } catch (...) {
            if (rank == 0) {
                std::cerr << "Warning: Failed to parse n_up, using full Hilbert space" << std::endl;
            }
            n_up = -1;
            use_fixed_sz = false;
        }
    }

    // Parse temperature scan parameters for ftlm_thermal/static (optional) - now at position 13
    double T_min = 1e-3;
    double T_max = 1.0;
    int T_steps = 20;
    bool use_temperature_scan = true;
    
    if (pos_argc >= 14) {
        std::string temp_str = pos_argv[13];
        std::stringstream temp_ss(temp_str);
        std::string val;
        std::vector<std::string> tokens;
        
        while (std::getline(temp_ss, val, ',')) {
            tokens.push_back(val);
        }
        
        try {
            if (tokens.size() >= 3) {
                T_min = std::stod(tokens[0]);
                T_max = std::stod(tokens[1]);
                T_steps = std::stoi(tokens[2]);
                
                if (T_min > 0 && T_max > T_min && T_steps > 0) {
                    use_temperature_scan = true;
                    if (rank == 0) {
                        std::cout << "Temperature scan enabled: T ∈ [" << T_min << ", " << T_max 
                                  << "], " << T_steps << " points (log scale)" << std::endl;
                    }
                } else {
                    if (rank == 0) {
                        std::cerr << "Warning: Invalid temperature parameters (need T_min > 0, T_max > T_min, T_steps > 0)" << std::endl;
                        std::cerr << "Using TPQ state temperature instead" << std::endl;
                    }
                    use_temperature_scan = false;
                }
            } else {
                if (rank == 0) {
                    std::cerr << "Warning: Temperature parameter needs 3 values (T_min,T_max,T_steps)" << std::endl;
                    std::cerr << "Using TPQ state temperature instead" << std::endl;
                }
            }
        } catch (...) {
            if (rank == 0) {
                std::cerr << "Warning: Failed to parse temperature parameters" << std::endl;
                std::cerr << "Using TPQ state temperature instead" << std::endl;
            }
            use_temperature_scan = false;
        }
    }

    // Parse spin combinations
    std::vector<std::pair<int, int>> spin_combinations;
    std::stringstream ss(spin_combinations_str);
    std::string pair_str;
    
    while (std::getline(ss, pair_str, ';')) {
        std::stringstream pair_ss(pair_str);
        std::string op1_str, op2_str;
        
        if (std::getline(pair_ss, op1_str, ',') && std::getline(pair_ss, op2_str)) {
            try {
                int op1 = std::stoi(op1_str);
                int op2 = std::stoi(op2_str);
                
                if (op1 >= 0 && op1 <= 2 && op2 >= 0 && op2 <= 2) {
                    spin_combinations.push_back({op1, op2});
                } else {
                    if (rank == 0) {
                        std::cerr << "Warning: Invalid spin operator " << op1 << "," << op2 
                                  << ". Operators must be 0, 1, or 2." << std::endl;
                    }
                }
            } catch (const std::exception& e) {
                if (rank == 0) {
                    std::cerr << "Warning: Failed to parse spin combination: " << pair_str << std::endl;
                }
            }
        }
    }
    
    if (spin_combinations.empty()) {
        if (rank == 0) {
            std::cerr << "Error: No valid spin combinations provided. Using default SzSz." << std::endl;
        }
        spin_combinations = {{2, 2}};
    }

    // Determine if using XYZ basis (Sx, Sy, Sz) vs ladder basis (Sp, Sm, Sz)
    bool use_xyz_basis = (basis == "xyz");

    auto spin_combination_name = [use_xyz_basis](int op) {
        if (use_xyz_basis) {
            // XYZ basis: Sx, Sy, Sz
            switch (op) {
                case 0:
                    return "Sx";
                case 1:
                    return "Sy";
                case 2:
                    return "Sz";
                default:
                    return "Unknown";
            }
        } else {
            // Ladder basis: Sp, Sm, Sz
            switch (op) {
                case 2:
                    return "Sz";
                case 0:
                    return "Sp";
                case 1:
                    return "Sm";
                default:
                    return "Unknown";
            }
        }
    };

    std::vector<const char*> spin_combination_names;
    for (const auto& pair : spin_combinations) {
        int first = pair.first;
        int second = pair.second;
        
        if (!use_xyz_basis) {
            // For ladder basis: Convert 0->1(Sp), 1->0(Sm) for first operator
            first = first == 2 ? 2 : 1 - first;
        }
        // For XYZ basis, use operators as-is (0=Sx, 1=Sy, 2=Sz)
        std::string combined_name = std::string(spin_combination_name(first)) + std::string(spin_combination_name(second));
        char* name = new char[combined_name.size() + 1];
        std::strcpy(name, combined_name.c_str());
        spin_combination_names.push_back(name);
    }

    // Regex to match tpq_state files - support both new format (with step) and legacy format
    // New format: tpq_state_i_beta=*_step=*.dat
    // Legacy format: tpq_state_i_beta=*.dat
    std::regex state_pattern_new("tpq_state_([0-9]+)_beta=([0-9.]+)_step=([0-9]+)\\.dat");
    std::regex state_pattern_legacy("tpq_state_([0-9]+)_beta=([0-9.]+)\\.dat");

    // Load Hamiltonian (all processes need this)
    if (rank == 0) {
        std::cout << "Loading Hamiltonian..." << std::endl;
    }
    
    Operator ham_op(num_sites, spin_length);
    
    std::string interall_file = directory + "/InterAll.dat";
    std::string trans_file = directory + "/Trans.dat";
    std::string counterterm_file = directory + "/CounterTerm.dat";
    std::string three_body_file = directory + "/ThreeBodyG.dat";
    // Check if files exist
    if (!fs::exists(interall_file) || !fs::exists(trans_file)) {
        if (rank == 0) {
            std::cerr << "Error: Hamiltonian files not found in " << directory << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    
    // Load Hamiltonian
    ham_op.loadFromInterAllFile(interall_file);
    ham_op.loadFromFile(trans_file);
    if (fs::exists(three_body_file)) {
        if (rank == 0) {
            std::cout << "Loading three-body interactions from " << three_body_file << std::endl;
        }
        ham_op.loadThreeBodyTerm(three_body_file);
    }
    
    // COUNTERTERM DISABLED
    // if (fs::exists(counterterm_file)) {
    //     ham_op.loadCounterTerm(counterterm_file);
    // }
    
    auto H = [&ham_op](const Complex* in, Complex* out, int size) {
        ham_op.apply(in, out, size);
    };
    
    // Read ground state energy for energy shift in spectral functions
    double ground_state_energy = 0.0;
    bool has_ground_state_energy = false;
    if (method == "spectral" || method == "continued_fraction") {
        try {
            if (rank == 0) {
                std::cout << "Reading ground state energy (minimum across all sources)..." << std::endl;
                ground_state_energy = read_ground_state_energy(directory);
                has_ground_state_energy = true;
                std::cout << "Final ground state energy (for spectral shift): " 
                          << std::fixed << std::setprecision(10) << ground_state_energy << std::endl;
            }
            // Broadcast ground state energy to all ranks
            MPI_Bcast(&ground_state_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            int has_gs_int = has_ground_state_energy ? 1 : 0;
            MPI_Bcast(&has_gs_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
            has_ground_state_energy = (has_gs_int != 0);
        } catch (const std::exception& e) {
            if (rank == 0) {
                std::cerr << "Warning: Could not read ground state energy: " << e.what() << std::endl;
                std::cerr << "Proceeding without energy shift" << std::endl;
            }
            // Ensure all ranks know it failed
            MPI_Bcast(&ground_state_energy, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            int has_gs_int = 0;
            MPI_Bcast(&has_gs_int, 1, MPI_INT, 0, MPI_COMM_WORLD);
        }
    }
    
    // Use 64-bit to compute Hilbert space dimension and guard against int overflow
    size_t N64 = 1ULL << num_sites;
    if (N64 > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::cerr << "Error: 2^num_sites exceeds 32-bit int range (num_sites=" << num_sites
                      << "). Refactor APIs to use size_t for N or reduce num_sites." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    int N = static_cast<int>(N64);
    
    // Helper function for cross product
    auto cross_product = [](const std::vector<double>& a, const std::vector<double>& b) -> std::array<double, 3> {
        return {
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]
        };
    };
    
    // Helper function to normalize a vector
    auto normalize = [](const std::array<double, 3>& v) -> std::array<double, 3> {
        double norm = std::sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
        if (norm < 1e-10) {
            return {0.0, 0.0, 0.0};
        }
        return {v[0]/norm, v[1]/norm, v[2]/norm};
    };
    
    if (rank == 0) {
        std::cout << "\nMomentum points:" << std::endl;
        for (size_t i = 0; i < momentum_points.size(); i++) {
            std::cout << "  Q[" << i << "] = (" << momentum_points[i][0] << ", " 
                      << momentum_points[i][1] << ", " << momentum_points[i][2] << ")" << std::endl;
        }
        std::cout << "\nPolarization vector (transverse_basis_1): (" 
                  << polarization[0] << ", " << polarization[1] << ", " 
                  << polarization[2] << ")" << std::endl;
    }
    
    // Pre-compute time evolution operator and transverse bases if needed
    std::string output_base_dir = directory + "/structure_factor_results";
    std::string unified_h5_path;  // Path to unified DSSF HDF5 file (or per-rank file for INDEPENDENT mode)
    
    // Determine if we're in INDEPENDENT mode (need per-rank files)
    bool uses_independent_mode = (method == "spectral" || method == "ground_state" || method == "sssf" || method == "continued_fraction");
    
    if (rank == 0) {
        ensureDirectoryExists(output_base_dir);
        
        // Initialize unified HDF5 file for all DSSF results
        unified_h5_path = initDSSFHDF5File(
            output_base_dir, num_sites, spin_length,
            method, operator_type,
            omega_min, omega_max, num_omega_bins, broadening,
            momentum_points
        );
        std::cout << "Unified DSSF results will be saved to: " << unified_h5_path << std::endl;
        
        if (uses_independent_mode && size > 1) {
            std::cout << "INDEPENDENT mode: Each MPI rank will write to per-rank file, merged at end" << std::endl;
        }
    }
    
    // Ensure all ranks wait for rank 0 to create the output directory
    MPI_Barrier(MPI_COMM_WORLD);
    
    // For INDEPENDENT mode with MPI: each rank uses its own file to avoid file locking conflicts
    if (uses_independent_mode && size > 1) {
        unified_h5_path = initPerRankDSSFFile(
            output_base_dir, rank, num_sites, spin_length,
            method, operator_type,
            omega_min, omega_max, num_omega_bins, broadening,
            momentum_points
        );
    } else {
        // Broadcast unified HDF5 path to all processes (COLLECTIVE mode)
        int path_len = unified_h5_path.size();
        MPI_Bcast(&path_len, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            unified_h5_path.resize(path_len);
        }
        MPI_Bcast(unified_h5_path.data(), path_len, MPI_CHAR, 0, MPI_COMM_WORLD);
    }
    
    // Synchronize all processes
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Pre-compute transverse bases for transverse operators (needed for both krylov and taylor methods)
    std::vector<std::array<double,3>> transverse_basis_1, transverse_basis_2;
    
    if (operator_type == "transverse" || operator_type == "transverse_experimental") {
        int num_momentum = momentum_points.size();
        transverse_basis_1.resize(num_momentum);
        transverse_basis_2.resize(num_momentum);
        
        // transverse_basis_1 is the same for all momentum points (the polarization vector)
        std::array<double, 3> pol_array = {polarization[0], polarization[1], polarization[2]};
        
        for (int qi = 0; qi < num_momentum; ++qi) {
            transverse_basis_1[qi] = pol_array;
            
            // transverse_basis_2 is Q × polarization (cross product)
            auto cross = cross_product(momentum_points[qi], polarization);
            transverse_basis_2[qi] = normalize(cross);
            
            // Handle special case: if Q is parallel to polarization, cross product is zero
            double cross_norm = std::sqrt(cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]);
            if (cross_norm < 1e-10) {
                // Find an orthogonal vector to polarization
                if (std::abs(pol_array[0]) > 0.5) {
                    // polarization has significant x component, use y-axis as reference
                    auto alt_cross = cross_product({0.0, 1.0, 0.0}, polarization);
                    transverse_basis_2[qi] = normalize(alt_cross);
                } else {
                    // use x-axis as reference
                    auto alt_cross = cross_product({1.0, 0.0, 0.0}, polarization);
                    transverse_basis_2[qi] = normalize(alt_cross);
                }
                if (rank == 0) {
                    std::cout << "Warning: Q[" << qi << "] parallel to polarization, using alternative basis" << std::endl;
                }
            }
        }
        
        if (rank == 0) {
            std::cout << "\nTransverse bases for momentum points:" << std::endl;
            for (int qi = 0; qi < num_momentum; ++qi) {
                const auto &Q = momentum_points[qi];
                const auto &b1 = transverse_basis_1[qi];
                const auto &b2 = transverse_basis_2[qi];
                std::cout << "  Q[" << qi << "] = (" << Q[0] << "," << Q[1] << "," << Q[2] 
                          << "), e1=(" << b1[0] << "," << b1[1] << "," << b1[2] 
                          << "), e2=(" << b2[0] << "," << b2[1] << "," << b2[2] << ")" << std::endl;
            }
        }
    }
    
    // Collect all tpq_state files from HDF5 or output subdirectory (only rank 0)
    std::vector<std::string> tpq_files;
    std::vector<int> sample_indices;
    std::vector<double> beta_values;
    std::vector<std::string> beta_strings;
    
    if (rank == 0) {
        std::string tpq_directory = directory + "/output";
        
        // Check if output directory exists (only critical for methods that need pre-computed states)
        bool needs_precomputed_states = (method == "spectral" || method == "ground_state" || method == "continued_fraction");
        
        // For ftlm_thermal/static, we generate random states internally - no need for pre-computed states
        // This is MUCH faster since we only process once instead of per-state
        bool uses_internal_random_states = (method == "ftlm_thermal" || method == "static");
        
        if (uses_internal_random_states) {
            // For ftlm_thermal/static, create a single dummy entry so the task loop runs once
            std::cout << "Method '" << method << "' uses internal random state generation." << std::endl;
            std::cout << "Creating single task entry for processing..." << std::endl;
            tpq_files.push_back("INTERNAL_RANDOM_STATES");
            sample_indices.push_back(0);
            beta_strings.push_back("scan");
            beta_values.push_back(0.0);
            // Skip loading pre-computed states - go straight to task distribution
        } else if (!fs::exists(tpq_directory)) {
            if (needs_precomputed_states) {
                std::cerr << "\nError: Output directory does not exist: " << tpq_directory << std::endl;
                std::cerr << "\nThe '" << method << "' method requires pre-computed quantum states." << std::endl;
                std::cerr << "Please run one of the following first:" << std::endl;
                std::cerr << "  1. TPQ calculation: ED --config <config_file> (with tpq_mode enabled)" << std::endl;
                std::cerr << "  2. Exact diagonalization: ED --config <config_file> (with save_eigenvectors=true)" << std::endl;
                std::cerr << "\nAlternatively, use 'ftlm_thermal' method which generates random states internally." << std::endl;
                MPI_Abort(MPI_COMM_WORLD, 1);
            } else {
                // Should not reach here, but handle gracefully
                std::cout << "Output directory not found, but '" << method << "' generates states internally." << std::endl;
                std::cout << "Creating dummy task entry for processing..." << std::endl;
                tpq_files.push_back("INTERNAL_RANDOM_STATES");
                sample_indices.push_back(0);
                beta_strings.push_back("scan");
                beta_values.push_back(0.0);
            }
        }
        
        // Only search for pre-computed states if method requires them
        if (!uses_internal_random_states && tpq_files.empty()) {
        // First, check for TPQ states in unified HDF5 file
        std::string hdf5_path = tpq_directory + "/ed_results.h5";
        bool found_hdf5_tpq_states = false;
        
        if (fs::exists(hdf5_path)) {
            std::vector<HDF5IO::TPQStateInfo> hdf5_states = HDF5IO::listTPQStates(hdf5_path);
            
            if (!hdf5_states.empty()) {
                std::cout << "Found " << hdf5_states.size() << " TPQ state(s) in HDF5 file: " << hdf5_path << std::endl;
                
                for (const auto& state_info : hdf5_states) {
                    // Use marker format: "TPQ_HDF5:<path>:<dataset_name>"
                    std::string marker = "TPQ_HDF5:" + hdf5_path + ":" + state_info.dataset_name;
                    tpq_files.push_back(marker);
                    sample_indices.push_back(static_cast<int>(state_info.sample_index));
                    
                    // Format beta string to match legacy behavior
                    std::stringstream beta_ss;
                    beta_ss << std::fixed << std::setprecision(6) << state_info.beta;
                    beta_strings.push_back(beta_ss.str());
                    beta_values.push_back(state_info.beta);
                    
                    std::cout << "  - TPQ state: sample=" << state_info.sample_index 
                              << ", β=" << state_info.beta << std::endl;
                }
                found_hdf5_tpq_states = true;
            }
        }
        
        // Fall back to legacy .dat files if no HDF5 TPQ states found
        if (!found_hdf5_tpq_states) {
            std::cout << "No TPQ states in HDF5, checking for legacy .dat files..." << std::endl;
            
            try {
                for (const auto& entry : fs::directory_iterator(tpq_directory)) {
                    if (!entry.is_regular_file()) continue;
                    
                    std::string filename = entry.path().filename().string();
                    std::smatch match;
                    
                    if (std::regex_match(filename, match, state_pattern_new)) {
                        tpq_files.push_back(entry.path().string());
                        sample_indices.push_back(std::stoi(match[1]));
                        beta_strings.push_back(match[2]);
                        beta_values.push_back(std::stod(match[2]));
                    }
                }
            } catch (const fs::filesystem_error& e) {
                std::cerr << "Warning: Error reading directory " << tpq_directory << ": " << e.what() << std::endl;
            }
        }

        // Include zero-temperature ground-state eigenvector if available
        // Eigenvectors are stored in HDF5 format at /eigendata/eigenvector_0
        std::string gs_hdf5_path = find_eigenvector_hdf5(directory);
        
        if (!gs_hdf5_path.empty()) {
            // Use special marker "HDF5:<path>:0" to indicate HDF5 eigenvector 0
            tpq_files.push_back("HDF5:" + gs_hdf5_path + ":0");
            sample_indices.push_back(0); // use 0 as a conventional index for ground state
            beta_strings.push_back("inf");
            beta_values.push_back(std::numeric_limits<double>::infinity());
            std::cout << "Found ground state eigenvector in HDF5: " << gs_hdf5_path << std::endl;
        } else {
            std::cout << "No ground state eigenvector found in " << tpq_directory << "/ed_results.h5" << std::endl;
        }
        
        // For methods that don't need pre-computed states, add fallback dummy entry
        if (tpq_files.empty()) {
            bool needs_precomputed = (method == "spectral" || method == "ground_state" || 
                                      method == "spin_correlation" || method == "spin_configuration" ||
                                      method == "sssf" || method == "continued_fraction");
            if (!needs_precomputed) {
                std::cout << "No state files found, but '" << method << "' generates states internally." << std::endl;
                tpq_files.push_back("INTERNAL_RANDOM_STATES");
                sample_indices.push_back(0);
                beta_strings.push_back("scan");
                beta_values.push_back(0.0);
            }
        }
        } // End of: if (!uses_internal_random_states && tpq_files.empty())
        
        std::cout << "Found " << tpq_files.size() << " state file(s) to process (including ground state if present)" << std::endl;
        std::cout << "Using " << size << " MPI processes" << std::endl;
    }
    
    // Broadcast the number of files to all processes
    int num_files = tpq_files.size();
    MPI_Bcast(&num_files, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Check for methods that need pre-computed states
    bool needs_precomputed_states = (method == "spectral" || method == "ground_state" || method == "sssf" || method == "continued_fraction");

    if (num_files == 0 && needs_precomputed_states) {
        if (rank == 0) {
            std::cerr << "\n" << std::string(80, '=') << std::endl;
            std::cerr << "ERROR: No quantum states found for '" << method << "' calculation!" << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
            std::cerr << "\nSearched in: " << directory << "/output/ed_results.h5" << std::endl;
            
            if (method == "ground_state") {
                std::cerr << "\nThe 'ground_state' method requires a pre-computed ground state eigenvector." << std::endl;
                std::cerr << "\nExpected location: ed_results.h5 -> /eigendata/eigenvector_0" << std::endl;
                std::cerr << "\nTo generate the ground state, run:" << std::endl;
                std::cerr << "  ED --config <config> (with save_eigenvectors=true and num_eigenvectors>=1)" << std::endl;
            } else {
                std::cerr << "\nThe '" << method << "' method requires pre-computed states. None were found." << std::endl;
                std::cerr << "\nExpected locations in ed_results.h5:" << std::endl;
                std::cerr << "  - TPQ states: /tpq/states/* datasets" << std::endl;
                std::cerr << "  - Eigenvector: /eigendata/eigenvector_0" << std::endl;
                std::cerr << "\nTo generate states, run one of:" << std::endl;
                std::cerr << "  1. TPQ calculation:  ED --config <config> (with tpq_mode=microcanonical or canonical)" << std::endl;
                std::cerr << "  2. Diagonalization:  ED --config <config> (with save_eigenvectors=true)" << std::endl;
            }
            std::cerr << "\nAlternative: Use 'ftlm_thermal' method instead, which generates random" << std::endl;
            std::cerr << "states internally and doesn't require pre-computed states." << std::endl;
            std::cerr << std::string(80, '=') << std::endl;
        }
        MPI_Finalize();
        return 1;  // Return error code instead of 0
    } else if (num_files == 0) {
        // For ftlm_thermal/static, create dummy entry on all ranks
        // (Already done on rank 0, but make vectors consistent)
        if (rank == 0) {
            std::cout << "Using internal random state generation for '" << method << "' method" << std::endl;
        }
        // Vectors already populated on rank 0 in earlier block
    }
    
    // Get file sizes for workload estimation (only rank 0)
    std::vector<size_t> file_sizes(num_files, 0);
    if (rank == 0) {
        for (int i = 0; i < num_files; i++) {
            try {
                // Handle HDF5 marker paths specially (eigenvector and TPQ state markers)
                if (tpq_files[i].rfind("HDF5:", 0) == 0) {
                    // Parse HDF5 marker to get actual file path: "HDF5:<path>:<idx>"
                    std::string marker = tpq_files[i].substr(5);
                    size_t last_colon = marker.rfind(':');
                    if (last_colon != std::string::npos) {
                        std::string h5_path = marker.substr(0, last_colon);
                        file_sizes[i] = fs::file_size(h5_path);
                    } else {
                        file_sizes[i] = 1; // Default weight
                    }
                } else if (tpq_files[i].rfind("TPQ_HDF5:", 0) == 0) {
                    // Parse TPQ_HDF5 marker: "TPQ_HDF5:<path>:<dataset_name>"
                    std::string marker = tpq_files[i].substr(9);
                    size_t first_colon = marker.find(':');
                    if (first_colon != std::string::npos) {
                        std::string h5_path = marker.substr(0, first_colon);
                        file_sizes[i] = fs::file_size(h5_path);
                    } else {
                        file_sizes[i] = 1; // Default weight
                    }
                } else {
                    file_sizes[i] = fs::file_size(tpq_files[i]);
                }
            } catch (const std::exception& e) {
                // Use default weight if file size cannot be determined
                file_sizes[i] = 1;
            }
        }
    }
    MPI_Bcast(file_sizes.data(), num_files, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    
    // ============================================================================
    // TWO-LEVEL MPI PARALLELIZATION STRATEGY
    // ============================================================================
    // 
    // There are two distinct parallelization strategies based on the method:
    //
    // 1. INDEPENDENT TASKS (method = "spectral", "ground_state", etc.):
    //    - Task structure: (state × momentum × operator)
    //    - Each task is INDEPENDENT and processes a single TPQ/ground state
    //    - MPI uses master-worker pattern: different ranks work on different tasks
    //    - NO internal MPI communication within a task
    //    - Good for: many pre-computed states, parallelizes over states
    //
    // 2. COLLECTIVE TASKS (method = "ftlm_thermal", "static"):
    //    - Task structure: (momentum × operator) only - NO state dimension!
    //    - These methods generate random states INTERNALLY and compute ALL
    //      temperatures at once per task
    //    - MPI uses COLLECTIVE mode: ALL ranks process EACH task TOGETHER
    //    - INTERNAL MPI parallelization: random Ritz samples distributed across ranks
    //    - Each rank processes (num_samples / num_ranks) samples, then MPI_Reduce
    //    - Good for: finite-temperature spectral functions from scratch
    //
    // Key insight: ftlm_thermal/static compute ALL temperatures in one call,
    // so there's NO temperature parallelization at the task level. Temperature
    // parallelization only applies to "spectral" method with pre-computed TPQ states.
    // ============================================================================
    
    // Build fine-grained task list following (operators) × (method) structure
    // Each task is (state_idx, momentum_idx, combo_idx, sublattice_i, sublattice_j)
    struct Task {
        int state_idx;
        int momentum_idx;
        int combo_idx;
        int sublattice_i;
        int sublattice_j;
        size_t weight;  // file_size as proxy for cost
    };
    
    std::vector<Task> all_tasks;
    int num_momentum = momentum_points.size();
    int num_combos = spin_combinations.size();
    
    if (rank == 0) {
        // Special handling for spin_correlation, spin_configuration (process entire states atomically)
        if (method == "spin_correlation" || method == "spin_configuration") {
            // These methods process entire states atomically
            for (int s = 0; s < num_files; s++) {
                all_tasks.push_back({s, -1, -1, -1, -1, file_sizes[s]});
            }
            std::cout << "Parallelization: per-state (" << num_files << " tasks)" << std::endl;
        } else {
            // For krylov and taylor methods: parallelize based on operator type
            if (operator_type == "sublattice") {
                // Sublattice operators: parallelize across (state, momentum, combo, sublattice pairs)
                // Only compute upper triangle: sublattice_i <= sublattice_j (symmetry)
                int num_sublattice_pairs = unit_cell_size * (unit_cell_size + 1) / 2;
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        for (int c = 0; c < num_combos; c++) {
                            for (int sub_i = 0; sub_i < unit_cell_size; sub_i++) {
                                for (int sub_j = sub_i; sub_j < unit_cell_size; sub_j++) {
                                    size_t task_weight = file_sizes[s] / (num_momentum * num_combos * num_sublattice_pairs);
                                    all_tasks.push_back({s, q, c, sub_i, sub_j, task_weight});
                                }
                            }
                        }
                    }
                }
                std::cout << "Parallelization: per-sublattice-pair (upper triangle, " << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta × "
                          << num_combos << " combos × " << num_sublattice_pairs << " unique sublattice pairs)" << std::endl;
            } else if (operator_type == "transverse") {
                // Transverse operators: create 2 tasks per (state, momentum, combo) for SF/NSF
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        for (int c = 0; c < num_combos; c++) {
                            // Use sublattice_i as a flag: 0=SF, 1=NSF
                            size_t task_weight = file_sizes[s] / (num_momentum * num_combos * 2);
                            all_tasks.push_back({s, q, c, 0, -1, task_weight}); // SF component
                            all_tasks.push_back({s, q, c, 1, -1, task_weight}); // NSF component
                        }
                    }
                }
                std::cout << "Parallelization: per-transverse-component (" << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta × "
                          << num_combos << " combos × 2 components)" << std::endl;
            } else if (operator_type == "transverse_experimental") {
                // Transverse experimental operators: create 2 tasks per (state, momentum) for SF/NSF
                // Does NOT depend on combo (only uses theta parameter)
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        // Use sublattice_i as a flag: 0=SF, 1=NSF
                        // Set combo_idx to 0 (dummy value, not used)
                        size_t task_weight = file_sizes[s] / (num_momentum * 2);
                        all_tasks.push_back({s, q, 0, 0, -1, task_weight}); // SF component
                        all_tasks.push_back({s, q, 0, 1, -1, task_weight}); // NSF component
                    }
                }
                std::cout << "Parallelization: per-transverse-component (" << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta × 2 components)" << std::endl;
            } else if (operator_type == "experimental") {
                // Experimental operators: parallelize across (state, momentum) only
                // Does NOT depend on combo (only uses theta parameter)
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        // Set combo_idx to 0 (dummy value, not used)
                        size_t task_weight = file_sizes[s] / num_momentum;
                        all_tasks.push_back({s, q, 0, -1, -1, task_weight});
                    }
                }
                std::cout << "Parallelization: per-operator (" << all_tasks.size() << " tasks = "
                          << num_files << " states × " << num_momentum << " momenta)" << std::endl;
            } else {
                // Sum operators: parallelize across (state, momentum, combo)
                for (int s = 0; s < num_files; s++) {
                    for (int q = 0; q < num_momentum; q++) {
                        for (int c = 0; c < num_combos; c++) {
                            size_t task_weight = file_sizes[s] / (num_momentum * num_combos);
                            all_tasks.push_back({s, q, c, -1, -1, task_weight});
                        }
                    }
                }
                
                // Different output message for collective vs independent methods
                bool uses_collective = (method == "ftlm_thermal" || method == "static");
                if (uses_collective) {
                    // For ftlm_thermal/static: num_files=1 (dummy), so just momentum × combos
                    std::cout << "\n==========================================\n";
                    std::cout << "TASK GENERATION: COLLECTIVE MPI MODE\n";
                    std::cout << "==========================================\n";
                    std::cout << "Method: " << method << "\n";
                    std::cout << "Tasks: " << all_tasks.size() << " = " 
                              << num_momentum << " momenta × " << num_combos << " operators\n";
                    std::cout << "NOTE: Each task computes ALL temperatures internally\n";
                    std::cout << "NOTE: Random samples distributed across MPI ranks within each task\n";
                    std::cout << "==========================================\n";
                } else {
                    std::cout << "Parallelization: per-operator (" << all_tasks.size() << " tasks = "
                              << num_files << " states × " << num_momentum << " momenta × "
                              << num_combos << " combos)" << std::endl;
                }
            }
        }
        
        // Sort by weight (descending) for better load balance
        std::sort(all_tasks.begin(), all_tasks.end(), 
                  [](const Task& a, const Task& b) { return a.weight > b.weight; });
    }
    
    // Broadcast task list
    int num_tasks = all_tasks.size();
    MPI_Bcast(&num_tasks, 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    if (rank != 0) {
        all_tasks.resize(num_tasks);
    }
    
    // Broadcast all tasks (struct is POD-like)
    for (int i = 0; i < num_tasks; i++) {
        int buf[5] = {all_tasks[i].state_idx, all_tasks[i].momentum_idx, all_tasks[i].combo_idx, 
                      all_tasks[i].sublattice_i, all_tasks[i].sublattice_j};
        size_t w = all_tasks[i].weight;
        MPI_Bcast(buf, 5, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&w, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        if (rank != 0) {
            all_tasks[i] = {buf[0], buf[1], buf[2], buf[3], buf[4], w};
        }
    }
    
    // Resize vectors on non-root processes
    if (rank != 0) {
        tpq_files.resize(num_files);
        sample_indices.resize(num_files);
        beta_values.resize(num_files);
        beta_strings.resize(num_files);
    }
    
    // Optimized broadcast: use single buffer for all strings
    if (rank == 0) {
        // Pack all string data into a single buffer
        std::vector<char> string_buffer;
        std::vector<int> offsets;
        std::vector<int> lengths;
        
        for (int i = 0; i < num_files; i++) {
            offsets.push_back(string_buffer.size());
            lengths.push_back(tpq_files[i].size());
            string_buffer.insert(string_buffer.end(), tpq_files[i].begin(), tpq_files[i].end());
            
            offsets.push_back(string_buffer.size());
            lengths.push_back(beta_strings[i].size());
            string_buffer.insert(string_buffer.end(), beta_strings[i].begin(), beta_strings[i].end());
        }
        
        int buffer_size = string_buffer.size();
        MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(string_buffer.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(offsets.data(), offsets.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(lengths.data(), lengths.size(), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int buffer_size;
        MPI_Bcast(&buffer_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        
        std::vector<char> string_buffer(buffer_size);
        std::vector<int> offsets(num_files * 2);
        std::vector<int> lengths(num_files * 2);
        
        MPI_Bcast(string_buffer.data(), buffer_size, MPI_CHAR, 0, MPI_COMM_WORLD);
        MPI_Bcast(offsets.data(), offsets.size(), MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(lengths.data(), lengths.size(), MPI_INT, 0, MPI_COMM_WORLD);
        
        // Unpack strings
        for (int i = 0; i < num_files; i++) {
            int file_offset = offsets[i * 2];
            int file_length = lengths[i * 2];
            tpq_files[i].assign(string_buffer.begin() + file_offset, 
                               string_buffer.begin() + file_offset + file_length);
            
            int beta_offset = offsets[i * 2 + 1];
            int beta_length = lengths[i * 2 + 1];
            beta_strings[i].assign(string_buffer.begin() + beta_offset, 
                                  string_buffer.begin() + beta_offset + beta_length);
        }
    }
    
    MPI_Bcast(sample_indices.data(), num_files, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(beta_values.data(), num_files, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // Lambda to process a single task following (operators) × (method) structure
    auto process_task = [&](const Task& task) -> bool {
        int state_idx = task.state_idx;
        int momentum_idx = task.momentum_idx;
        int combo_idx = task.combo_idx;
        int sublattice_i = task.sublattice_i;
        int sublattice_j = task.sublattice_j;
        
        int sample_index = sample_indices[state_idx];
        double beta = beta_values[state_idx];
        std::string beta_str = beta_strings[state_idx];
        std::string file_path = tpq_files[state_idx];
        std::string filename = fs::path(file_path).filename().string();
        std::string output_dir = output_base_dir + "/beta_" + beta_str;
        
        // Load state (or skip for methods that generate internal random states)
        ComplexVector tpq_state;
        bool loaded_ok = false;
        bool needs_loaded_state = (method == "spectral" || method == "ground_state" || 
                                   method == "spin_correlation" || method == "spin_configuration" ||
                                   method == "sssf" || method == "continued_fraction");
        
        // For ftlm_thermal/static, we don't need to load a state
        if (file_path == "INTERNAL_RANDOM_STATES") {
            // Methods that generate their own random states internally
            loaded_ok = true;
            filename = "internal random states";
        }
        // Check for TPQ state HDF5 marker format: "TPQ_HDF5:<path>:<dataset_name>"
        else if (file_path.rfind("TPQ_HDF5:", 0) == 0) {
            // Parse TPQ_HDF5 marker: "TPQ_HDF5:<h5_path>:<dataset_name>"
            std::string marker = file_path.substr(9); // Remove "TPQ_HDF5:" prefix
            size_t first_colon = marker.find(':');
            if (first_colon != std::string::npos) {
                std::string h5_path = marker.substr(0, first_colon);
                std::string dataset_name = marker.substr(first_colon + 1);
                loaded_ok = HDF5IO::loadTPQStateByName(h5_path, dataset_name, tpq_state);
                filename = "TPQ state (HDF5): " + dataset_name;
                if (loaded_ok) {
                    std::cout << "Loaded TPQ state from HDF5: " << dataset_name 
                              << " (size: " << tpq_state.size() << ")" << std::endl;
                }
            }
        }
        // Check for HDF5 eigenvector marker format: "HDF5:<path>:<idx>"
        else if (file_path.rfind("HDF5:", 0) == 0) {
            // Parse HDF5 marker: "HDF5:<h5_path>:<eigenvector_idx>"
            std::string marker = file_path.substr(5); // Remove "HDF5:" prefix
            size_t last_colon = marker.rfind(':');
            if (last_colon != std::string::npos) {
                std::string h5_path = marker.substr(0, last_colon);
                int eigenvector_idx = std::stoi(marker.substr(last_colon + 1));
                loaded_ok = load_eigenvector_from_hdf5(tpq_state, h5_path, eigenvector_idx);
                filename = "eigenvector_" + std::to_string(eigenvector_idx) + " (HDF5)";
            }
        } else {
            // Legacy TPQ state file (.dat format)
            loaded_ok = load_tpq_state(tpq_state, file_path);
        }
        
        // Validate state loading only for methods that need it
        if (needs_loaded_state && (!loaded_ok || (int)tpq_state.size() != N)) {
            std::cerr << "Rank " << rank << " failed to load/validate state from " << filename << std::endl;
            return false;
        }
        
        // Note: output_dir is defined earlier in the lambda for legacy methods
        
        // Special methods that don't follow (operators) × (method) structure
        if (method == "spin_correlation") {
            ensureDirectoryExists(output_dir);
            printSpinCorrelation(tpq_state, num_sites, spin_length, output_dir, unit_cell_size);
            // TODO: Also save to unified HDF5 via saveDSSFCorrelationToHDF5
            return true;
        } else if (method == "spin_configuration") {
            ensureDirectoryExists(output_dir);
            printSpinConfiguration(tpq_state, num_sites, spin_length, output_dir);
            // TODO: Also save to unified HDF5
            return true;
        }
        
        // ============================================================
        // Main (operators) × (method) structure
        // ============================================================
        
        // STEP 1: Construct operators based on operator_type
        std::vector<Operator> obs_1, obs_2;
        std::vector<std::string> obs_names;
        std::string method_dir = output_dir + "/" + operator_type;
        
        // Add suffix to directory if using fixed-Sz
        if (use_fixed_sz) {
            method_dir += "_fixed_sz_nup" + std::to_string(n_up);
        }
        
        const auto &Q = momentum_points[momentum_idx];
        int op_type_1 = spin_combinations[combo_idx].first;
        int op_type_2 = spin_combinations[combo_idx].second;
        std::string base_name = std::string(spin_combination_names[combo_idx]);
        
        try {
            if (operator_type == "sum") {
                // Standard sum operators: S^{op1}(Q) S^{op2}(-Q)
                std::stringstream name_ss;
                name_ss << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2];
                obs_names.push_back(name_ss.str());
                
                if (use_fixed_sz) {
                    // Fixed-Sz operators
                    if (use_xyz_basis) {
                        FixedSzSumOperatorXYZ sum_op_1(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], positions_file);
                        FixedSzSumOperatorXYZ sum_op_2(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], positions_file);
                        obs_1.push_back(Operator(sum_op_1));
                        obs_2.push_back(Operator(sum_op_2));
                    } else {
                        FixedSzSumOperator sum_op_1(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], positions_file);
                        FixedSzSumOperator sum_op_2(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], positions_file);
                        obs_1.push_back(Operator(sum_op_1));
                        obs_2.push_back(Operator(sum_op_2));
                    }
                } else {
                    // Full Hilbert space operators
                    if (use_xyz_basis) {
                        SumOperatorXYZ sum_op_1(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], positions_file);
                        SumOperatorXYZ sum_op_2(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], positions_file);
                        obs_1.push_back(Operator(sum_op_1));
                        obs_2.push_back(Operator(sum_op_2));
                    } else {
                        SumOperator sum_op_1(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], positions_file);
                        SumOperator sum_op_2(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], positions_file);
                        obs_1.push_back(Operator(sum_op_1));
                        obs_2.push_back(Operator(sum_op_2));
                    }
                }
                
            } else if (operator_type == "transverse") {
                // Transverse operators for SF/NSF separation
                const auto &b1 = transverse_basis_1[momentum_idx];
                const auto &b2 = transverse_basis_2[momentum_idx];
                std::vector<double> e1_vec = {b1[0], b1[1], b1[2]};
                std::vector<double> e2_vec = {b2[0], b2[1], b2[2]};
                
                // SF component (transverse_basis_1)
                std::stringstream name_sf;
                name_sf << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_NSF";
                
                // NSF component (transverse_basis_2)
                std::stringstream name_nsf;
                name_nsf << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_SF";
                
                if (use_fixed_sz) {
                    // Fixed-Sz transverse operators
                    if (use_xyz_basis) {
                        FixedSzTransverseOperatorXYZ op1_sf(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op2_sf(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op1_nsf(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                        FixedSzTransverseOperatorXYZ op2_nsf(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                        
                        obs_1.push_back(Operator(op1_sf));
                        obs_2.push_back(Operator(op2_sf));
                        obs_1.push_back(Operator(op1_nsf));
                        obs_2.push_back(Operator(op2_nsf));
                    } else {
                        FixedSzTransverseOperator op1_sf(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                        FixedSzTransverseOperator op2_sf(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                        FixedSzTransverseOperator op1_nsf(num_sites, spin_length, n_up, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                        FixedSzTransverseOperator op2_nsf(num_sites, spin_length, n_up, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                        
                        obs_1.push_back(Operator(op1_sf));
                        obs_2.push_back(Operator(op2_sf));
                        obs_1.push_back(Operator(op1_nsf));
                        obs_2.push_back(Operator(op2_nsf));
                    }
                } else {
                    // Full Hilbert space transverse operators
                    if (use_xyz_basis) {
                        TransverseOperatorXYZ op1_sf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                        TransverseOperatorXYZ op2_sf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                        TransverseOperatorXYZ op1_nsf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                        TransverseOperatorXYZ op2_nsf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                        
                        obs_1.push_back(Operator(op1_sf));
                        obs_2.push_back(Operator(op2_sf));
                        obs_1.push_back(Operator(op1_nsf));
                        obs_2.push_back(Operator(op2_nsf));
                    } else {
                        TransverseOperator op1_sf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e1_vec, positions_file);
                        TransverseOperator op2_sf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e1_vec, positions_file);
                        TransverseOperator op1_nsf(num_sites, spin_length, op_type_1, momentum_points[momentum_idx], e2_vec, positions_file);
                        TransverseOperator op2_nsf(num_sites, spin_length, op_type_2, momentum_points[momentum_idx], e2_vec, positions_file);
                        
                        obs_1.push_back(Operator(op1_sf));
                        obs_2.push_back(Operator(op2_sf));
                        obs_1.push_back(Operator(op1_nsf));
                        obs_2.push_back(Operator(op2_nsf));
                    }
                }
                
                obs_names.push_back(name_sf.str());
                obs_names.push_back(name_nsf.str());
                
            } else if (operator_type == "sublattice") {
                // Sublattice-resolved operators
                std::stringstream name_ss;
                name_ss << base_name << "_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2]
                        << "_sub" << sublattice_i << "_sub" << sublattice_j;
                obs_names.push_back(name_ss.str());
                
                std::vector<double> Q_vec = {Q[0], Q[1], Q[2]};
                
                if (use_fixed_sz) {
                    // Fixed-Sz sublattice operators
                    FixedSzSublatticeOperator sub_op_1(sublattice_i, unit_cell_size, num_sites, spin_length, n_up, op_type_1, Q_vec, positions_file);
                    FixedSzSublatticeOperator sub_op_2(sublattice_j, unit_cell_size, num_sites, spin_length, n_up, op_type_2, Q_vec, positions_file);
                    
                    obs_1.push_back(Operator(sub_op_1));
                    obs_2.push_back(Operator(sub_op_2));
                } else {
                    // Full Hilbert space sublattice operators
                    SublatticeOperator sub_op_1(sublattice_i, unit_cell_size, num_sites, spin_length, op_type_1, Q_vec, positions_file);
                    SublatticeOperator sub_op_2(sublattice_j, unit_cell_size, num_sites, spin_length, op_type_2, Q_vec, positions_file);
                    
                    obs_1.push_back(Operator(sub_op_1));
                    obs_2.push_back(Operator(sub_op_2));
                }
                
            } else if (operator_type == "experimental") {
                // Experimental operators: cos(θ)Sz + sin(θ)Sx
                std::stringstream name_ss;
                name_ss << "Experimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] << "_theta" << theta;
                obs_names.push_back(name_ss.str());
                
                if (use_fixed_sz) {
                    // Fixed-Sz experimental operators
                    FixedSzExperimentalOperator exp_op_1(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], positions_file);
                    FixedSzExperimentalOperator exp_op_2(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], positions_file);
                    
                    obs_1.push_back(Operator(exp_op_1));
                    obs_2.push_back(Operator(exp_op_2));
                } else {
                    // Full Hilbert space experimental operators
                    ExperimentalOperator exp_op_1(num_sites, spin_length, theta, momentum_points[momentum_idx], positions_file);
                    ExperimentalOperator exp_op_2(num_sites, spin_length, theta, momentum_points[momentum_idx], positions_file);
                    
                    obs_1.push_back(Operator(exp_op_1));
                    obs_2.push_back(Operator(exp_op_2));
                }
                
            } else if (operator_type == "transverse_experimental") {
                // Transverse experimental operators with SF/NSF separation: cos(θ)Sz + sin(θ)Sx
                const auto &b1 = transverse_basis_1[momentum_idx];
                const auto &b2 = transverse_basis_2[momentum_idx];
                std::vector<double> e1_vec = {b1[0], b1[1], b1[2]};
                std::vector<double> e2_vec = {b2[0], b2[1], b2[2]};
                
                // SF component (transverse_basis_1)
                std::stringstream name_sf;
                name_sf << "TransverseExperimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] 
                        << "_theta" << theta << "_NSF";
                
                // NSF component (transverse_basis_2)
                std::stringstream name_nsf;
                name_nsf << "TransverseExperimental_q_Qx" << Q[0] << "_Qy" << Q[1] << "_Qz" << Q[2] 
                         << "_theta" << theta << "_SF";
                
                if (use_fixed_sz) {
                    // Fixed-Sz transverse experimental operators
                    FixedSzTransverseExperimentalOperator op1_sf(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], e1_vec, positions_file);
                    FixedSzTransverseExperimentalOperator op2_sf(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], e1_vec, positions_file);
                    FixedSzTransverseExperimentalOperator op1_nsf(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], e2_vec, positions_file);
                    FixedSzTransverseExperimentalOperator op2_nsf(num_sites, spin_length, n_up, theta, momentum_points[momentum_idx], e2_vec, positions_file);
                    
                    obs_1.push_back(Operator(op1_sf));
                    obs_2.push_back(Operator(op2_sf));
                    obs_1.push_back(Operator(op1_nsf));
                    obs_2.push_back(Operator(op2_nsf));
                } else {
                    // Full Hilbert space transverse experimental operators
                    TransverseExperimentalOperator op1_sf(num_sites, spin_length, theta, momentum_points[momentum_idx], e1_vec, positions_file);
                    TransverseExperimentalOperator op2_sf(num_sites, spin_length, theta, momentum_points[momentum_idx], e1_vec, positions_file);
                    TransverseExperimentalOperator op1_nsf(num_sites, spin_length, theta, momentum_points[momentum_idx], e2_vec, positions_file);
                    TransverseExperimentalOperator op2_nsf(num_sites, spin_length, theta, momentum_points[momentum_idx], e2_vec, positions_file);
                    
                    obs_1.push_back(Operator(op1_sf));
                    obs_2.push_back(Operator(op2_sf));
                    obs_1.push_back(Operator(op1_nsf));
                    obs_2.push_back(Operator(op2_nsf));
                }
                
                obs_names.push_back(name_sf.str());
                obs_names.push_back(name_nsf.str());
            }
            
        } catch (const std::exception &e) {
            std::cerr << "Rank " << rank << " failed operator construction: " << e.what() << std::endl;
            return false;
        }
        
        // STEP 2: Apply method to the constructed operators
        // All methods use unified HDF5 output (dssf_results.h5)
        
        try {
            if (method == "spectral") {
                    // Use spectral method with FTLM approach
                    int krylov_dim = krylov_dim_or_nmax;
                    
#ifdef WITH_CUDA
                    if (use_gpu) {
                        std::cout << "Using GPU-accelerated FTLM spectral calculation..." << std::endl;
                        
                        // Convert CPU Hamiltonian to GPU
                        GPUOperator gpu_ham(num_sites, spin_length);
                        if (!convertOperatorToGPU(ham_op, gpu_ham)) {
                            std::cerr << "Failed to convert Hamiltonian to GPU, falling back to CPU" << std::endl;
                            use_gpu = false;
                        } else {
                            // Process each operator pair on GPU
                            for (size_t i = 0; i < obs_1.size(); i++) {
                                std::cout << "  Processing operator pair " << (i+1) << "/" << obs_1.size() 
                                          << ": " << obs_names[i] << std::endl;
                                
                                // Convert observable operators to GPU
                                GPUOperator gpu_obs1(num_sites, spin_length);
                                GPUOperator gpu_obs2(num_sites, spin_length);
                                
                                if (!convertOperatorToGPU(obs_1[i], gpu_obs1) || 
                                    !convertOperatorToGPU(obs_2[i], gpu_obs2)) {
                                    std::cerr << "  Failed to convert operators to GPU, skipping..." << std::endl;
                                    continue;
                                }
                                
                                // Allocate device memory for TPQ state
                                cuDoubleComplex* d_psi;
                                cudaMalloc(&d_psi, N * sizeof(cuDoubleComplex));
                                cudaMemcpy(d_psi, tpq_state.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
                                
                                // Check if O1 == O2 based on spin operator types (enables memory optimization)
                                // Pass as int: 1 = identical, 0 = not identical, -1 = auto-detect
                                int operators_identical_flag = (op_type_1 == op_type_2) ? 1 : 0;
                                
                                // Call GPU wrapper for dynamical correlation on a single state
                                auto [frequencies, S_real, S_imag] = GPUEDWrapper::runGPUDynamicalCorrelationState(
                                    &gpu_ham, &gpu_obs1, &gpu_obs2,
                                    d_psi,  // Device pointer to TPQ state
                                    N,
                                    krylov_dim,
                                    omega_min, omega_max, num_omega_bins,
                                    broadening,
                                    0.0,  // temperature (not used for single-state)
                                    ground_state_energy,
                                    operators_identical_flag  // enables memory-efficient CF for large systems
                                );
                                
                                // Package results
                                DynamicalResponseResults results;
                                results.frequencies = frequencies;
                                results.spectral_function = S_real;
                                results.spectral_function_imag = S_imag;
                                // Initialize error vectors to zero (single-state, no error bars)
                                results.spectral_error.resize(frequencies.size(), 0.0);
                                results.spectral_error_imag.resize(frequencies.size(), 0.0);
                                results.total_samples = 1;
                                
                                // Save results to unified HDF5 file
                                saveDSSFSpectralToHDF5(
                                    unified_h5_path,
                                    obs_names[i],
                                    beta,
                                    sample_index,
                                    results.frequencies,
                                    results.spectral_function,
                                    results.spectral_function_imag,
                                    results.spectral_error,
                                    results.spectral_error_imag
                                );
                                
                                if (rank == 0) {
                                    std::cout << "  Saved GPU spectral to HDF5: " << obs_names[i] << std::endl;
                                }
                                
                                // Cleanup device memory
                                cudaFree(d_psi);
                            }
                        }
                    }
#endif
                    
                    if (!use_gpu) {
                        // CPU spectral calculation
                        // Set up FTLM parameters
                        DynamicalResponseParameters params;
                        params.krylov_dim = krylov_dim;
                        params.broadening = broadening;
                        params.tolerance = 1e-10;
                        // MEMORY OPTIMIZATION: For large systems (>16M states = 24 bits), disable
                        // all reorthogonalization to prevent storing krylov_dim basis vectors.
                        // Each vector for 27 sites is ~2GB, storing 50 would be ~100GB!
                        // The 3-term recurrence alone gives acceptable results for spectral functions.
                        bool large_system = (N > (1ULL << 24));
                        params.full_reorthogonalization = !large_system;
                        params.reorth_frequency = large_system ? 0 : 10;  // 0 = no reorth, no basis storage
                        if (large_system && rank == 0) {
                            std::cout << "  *** MEMORY OPTIMIZATION ENABLED ***" << std::endl;
                            std::cout << "  System size: " << N << " states (" << (N * 16.0 / (1024*1024*1024)) << " GB per vector)" << std::endl;
                            std::cout << "  Disabled reorthogonalization to avoid storing " << krylov_dim << " basis vectors" << std::endl;
                            std::cout << "  This saves ~" << (krylov_dim * N * 16.0 / (1024*1024*1024)) << " GB of memory" << std::endl;
                        }
                        
                        // Process each operator pair
                        for (size_t i = 0; i < obs_1.size(); i++) {
                            // Check if operators are identical (can use memory-efficient continued fraction)
                            bool operators_identical = (op_type_1 == op_type_2);
                            
                            DynamicalResponseResults results;
                            
                            if (large_system && operators_identical) {
                                // Use memory-efficient continued fraction for large systems with O1=O2
                                if (rank == 0) {
                                    std::cout << "  Using MEMORY-EFFICIENT continued fraction (O1=O2)" << std::endl;
                                }
                                
                                auto O_func = [&obs_1, i](const Complex* in, Complex* out, int size) {
                                    obs_1[i].apply(in, out, size);
                                };
                                
                                results = compute_dynamical_correlation_state_cf(
                                    H, O_func, tpq_state, N, params,
                                    omega_min, omega_max, num_omega_bins, ground_state_energy
                                );
                            } else {
                                // Standard path with basis storage (for O1≠O2 or small systems)
                                auto O1_func = [&obs_1, i](const Complex* in, Complex* out, int size) {
                                    obs_1[i].apply(in, out, size);
                                };
                                
                                auto O2_func = [&obs_2, i](const Complex* in, Complex* out, int size) {
                                    obs_2[i].apply(in, out, size);
                                };
                                
                                results = compute_dynamical_correlation_state(
                                    H, O1_func, O2_func, tpq_state, N, params,
                                    omega_min, omega_max, num_omega_bins, 0.0, ground_state_energy
                                );
                            }
                            
                            // Save results to unified HDF5 file
                            saveDSSFSpectralToHDF5(
                                unified_h5_path,
                                obs_names[i],
                                beta,
                                sample_index,
                                results.frequencies,
                                results.spectral_function,
                                results.spectral_function_imag,
                                results.spectral_error,
                                results.spectral_error_imag
                            );
                            
                            if (rank == 0) {
                                std::cout << "  Saved spectral to HDF5: " << obs_names[i] << std::endl;
                            }
                        }
                    }
                } else if (method == "ftlm_thermal") {
                    // ============================================================
                    // TRUE FTLM THERMAL DSSF: Random state sampling with thermal averaging
                    // This is the most accurate method for finite-temperature spectra
                    // ============================================================
                    // 
                    // MPI PARALLELIZATION (internal, within this task):
                    //   - Total random samples = num_samples (e.g., 40)
                    //   - Each MPI rank processes (num_samples / num_ranks) samples
                    //   - Results aggregated via MPI_Reduce at the end
                    //   - Temperature loop is NOT parallelized - all temps computed at once
                    //
                    // WHAT THIS COMPUTES:
                    //   - For one (momentum, operator) pair, computes spectral function
                    //     S(q, ω, T) for ALL temperatures in the temperature list
                    //   - Uses Lanczos algorithm with random state sampling (FTLM)
                    // ============================================================
                    int krylov_dim = krylov_dim_or_nmax;
                    int num_samples = 40;  // Number of random samples for FTLM thermal averaging
                    unsigned int random_seed = state_idx * 1000 + momentum_idx * 100 + combo_idx;
                    
                    // Determine temperature(s) to compute - ALL temps done in one call
                    std::vector<double> temperatures;
                    
                    if (use_temperature_scan) {
                        // Use user-specified temperature range (log spacing)
                        double log_T_min = std::log(T_min);
                        double log_T_max = std::log(T_max);
                        double log_step = (log_T_max - log_T_min) / std::max(1, T_steps - 1);
                        
                        for (int i = 0; i < T_steps; i++) {
                            double log_T = log_T_min + i * log_step;
                            temperatures.push_back(std::exp(log_T));
                        }
                    } else {
                        // Default temperature range if not specified
                        temperatures = {0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0};
                    }
                    
                    if (rank == 0) {
                        std::cout << "\n============================================" << std::endl;
                        std::cout << "FTLM THERMAL DSSF: TRUE random state sampling" << std::endl;
                        std::cout << "============================================" << std::endl;
                        std::cout << "  Krylov dimension: " << krylov_dim << std::endl;
                        std::cout << "  Number of random samples: " << num_samples << std::endl;
                        std::cout << "  MPI ranks: " << size << " (samples/rank: " << (num_samples / size) << ")" << std::endl;
                        std::cout << "  Temperature points: " << temperatures.size() << " (ALL computed in this task)" << std::endl;
                        std::cout << "  Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
                        std::cout << "  Broadening: " << broadening << std::endl;
                        std::cout << "  GPU acceleration: " << (use_gpu ? "enabled" : "disabled") << std::endl;
                    }
                    
                    // Set up FTLM parameters
                    DynamicalResponseParameters params;
                    params.krylov_dim = krylov_dim;
                    params.broadening = broadening;
                    params.tolerance = 1e-10;
                    // MEMORY OPTIMIZATION: For large systems (>16M states), disable all
                    // reorthogonalization. This prevents storing krylov_dim basis vectors
                    // which would require ~100GB for 27 sites with krylov_dim=50.
                    bool large_system = (N > (1ULL << 24));  // > 16M states
                    params.full_reorthogonalization = !large_system;
                    params.reorth_frequency = large_system ? 0 : 10;  // 0 = pure 3-term recurrence
                    if (large_system && rank == 0) {
                        std::cout << "  *** MEMORY OPTIMIZATION ENABLED ***" << std::endl;
                        std::cout << "  System size: " << N << " states (" << (N * 16.0 / (1024*1024*1024)) << " GB per vector)" << std::endl;
                        std::cout << "  Disabled reorthogonalization to avoid storing basis vectors" << std::endl;
                        std::cout << "  Memory saved: ~" << (krylov_dim * N * 16.0 / (1024*1024*1024)) << " GB" << std::endl;
                    }
                    params.num_samples = num_samples;
                    params.random_seed = random_seed;
                    params.store_intermediate = false;
                    
#ifdef WITH_CUDA
                    if (use_gpu) {
                        // ============================================================
                        // GPU-accelerated FTLM using CORRECT multi-temperature formulation
                        // Uses computeDynamicalCorrelationMultiTemp which matches CPU algorithm
                        // ============================================================
                        
                        if (rank == 0) {
                            std::cout << "Using GPU-accelerated FTLM thermal spectral (CORRECT formulation)" << std::endl;
                            std::cout << "  MPI ranks: " << size << std::endl;
                            std::cout << "  Total samples: " << num_samples << std::endl;
                            std::cout << "  Temperatures: " << temperatures.size() << std::endl;
                        }
                        
                        // Distribute samples across MPI ranks
                        int samples_per_rank = num_samples / size;
                        int remainder = num_samples % size;
                        int my_samples = samples_per_rank + (rank < remainder ? 1 : 0);
                        // Use the SAME seed formula as CPU to ensure identical results
                        // Both CPU and GPU use: base_seed + sample_idx * 12345
                        // For MPI: each rank processes different samples with the same base seed
                        unsigned int my_seed = random_seed;  // Same base seed as CPU
                        
                        if (rank == 0) {
                            std::cout << "  Samples per rank: ~" << samples_per_rank 
                                      << " (some ranks +1 for remainder)" << std::endl;
                        }
                        std::cout << "  Rank " << rank << ": " << my_samples << " samples with seed " << my_seed << std::endl;
                        
                        // Convert CPU Hamiltonian to GPU
                        GPUOperator gpu_ham(num_sites, spin_length);
                        if (!convertOperatorToGPU(ham_op, gpu_ham)) {
                            std::cerr << "Failed to convert Hamiltonian to GPU, falling back to CPU" << std::endl;
                            use_gpu = false;
                        } else {
                            // Process each operator pair on GPU
                            for (size_t i = 0; i < obs_1.size(); i++) {
                                if (rank == 0) {
                                    std::cout << "  Processing operator pair " << (i+1) << "/" << obs_1.size() 
                                              << ": " << obs_names[i] << std::endl;
                                }
                                
                                // Convert observable operators to GPU
                                GPUOperator gpu_obs1(num_sites, spin_length);
                                GPUOperator gpu_obs2(num_sites, spin_length);
                                
                                if (!convertOperatorToGPU(obs_1[i], gpu_obs1) || 
                                    !convertOperatorToGPU(obs_2[i], gpu_obs2)) {
                                    std::cerr << "  Failed to convert operators to GPU, skipping..." << std::endl;
                                    continue;
                                }
                                
                                // Use CORRECT multi-temperature FTLM (matches CPU implementation)
                                // Each rank computes its share of samples with the full algorithm
                                auto local_results = GPUEDWrapper::runGPUDynamicalCorrelationMultiTemp(
                                    &gpu_ham, &gpu_obs1, &gpu_obs2,
                                    N, my_samples, krylov_dim,
                                    omega_min, omega_max, num_omega_bins,
                                    broadening, temperatures,
                                    my_seed,
                                    ground_state_energy
                                );
                                
                                // MPI reduction: average spectral functions from all ranks
                                for (double T : temperatures) {
                                    std::vector<double> global_real(num_omega_bins, 0.0);
                                    std::vector<double> global_imag(num_omega_bins, 0.0);
                                    std::vector<double> frequencies;
                                    
                                    // Get local results for this temperature
                                    std::vector<double> local_real(num_omega_bins, 0.0);
                                    std::vector<double> local_imag(num_omega_bins, 0.0);
                                    
                                    if (local_results.count(T) > 0) {
                                        auto& [freqs, S_real, S_imag] = local_results[T];
                                        frequencies = freqs;
                                        // Weight by number of samples this rank processed
                                        for (int j = 0; j < num_omega_bins && j < (int)S_real.size(); j++) {
                                            local_real[j] = S_real[j] * my_samples;
                                            local_imag[j] = S_imag[j] * my_samples;
                                        }
                                    }
                                    
                                    // Share frequencies (use the first non-empty one)
                                    if (frequencies.empty() && local_results.size() > 0) {
                                        auto it = local_results.begin();
                                        frequencies = std::get<0>(it->second);
                                    }
                                    
                                    MPI_Reduce(local_real.data(), global_real.data(), 
                                               num_omega_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                                    MPI_Reduce(local_imag.data(), global_imag.data(),
                                               num_omega_bins, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
                                    
                                    // Normalize and save results (only rank 0)
                                    if (rank == 0) {
                                        // Normalize by total samples
                                        for (int j = 0; j < num_omega_bins; j++) {
                                            global_real[j] /= num_samples;
                                            global_imag[j] /= num_samples;
                                        }
                                        
                                        // Initialize error vectors to zero
                                        std::vector<double> spectral_error(num_omega_bins, 0.0);
                                        std::vector<double> spectral_error_imag(num_omega_bins, 0.0);
                                        
                                        saveDSSFSpectralToHDF5(
                                            unified_h5_path,
                                            obs_names[i],
                                            1.0/T,  // beta
                                            sample_index,
                                            frequencies,
                                            global_real,
                                            global_imag,
                                            spectral_error,
                                            spectral_error_imag,
                                            T
                                        );
                                        
                                        std::cout << "  Saved GPU FTLM thermal spectral to HDF5: " << obs_names[i] 
                                                  << " T=" << T << std::endl;
                                    }
                                }
                                
                                // Synchronize before next operator
                                MPI_Barrier(MPI_COMM_WORLD);
                            }
                        }
                    }
#endif
                    
                    if (!use_gpu) {
                        // CPU computation path
                        // Process each operator pair with TRUE multi-sample FTLM
                        for (size_t i = 0; i < obs_1.size(); i++) {
                            std::cout << "  Processing operator " << obs_names[i] << std::endl;
                            
                            // Create function wrappers for operators
                            auto O1_func = [&obs_1, i](const Complex* in, Complex* out, int size) {
                                obs_1[i].apply(in, out, size);
                            };
                            
                            auto O2_func = [&obs_2, i](const Complex* in, Complex* out, int size) {
                                obs_2[i].apply(in, out, size);
                            };
                            
                            // Use TRUE multi-sample multi-temperature FTLM
                            auto results_map = compute_dynamical_correlation_multi_sample_multi_temperature(
                                H, O1_func, O2_func, N, params,
                                omega_min, omega_max, num_omega_bins, 
                                temperatures, ground_state_energy, output_base_dir
                            );
                            
                            // Save results for each temperature to unified HDF5 file
                            // Only rank 0 saves since results have been MPI_Reduced internally
                            if (rank == 0) {
                                for (const auto& [temperature, results] : results_map) {
                                    saveDSSFSpectralToHDF5(
                                        unified_h5_path,
                                        obs_names[i],
                                        1.0/temperature,  // beta
                                        sample_index,
                                        results.frequencies,
                                        results.spectral_function,
                                        results.spectral_function_imag,
                                        results.spectral_error,
                                        results.spectral_error_imag,
                                        temperature
                                    );
                                
                                    std::cout << "  Saved FTLM thermal spectral to HDF5: " << obs_names[i] 
                                              << " T=" << temperature << std::endl;
                                }
                            }
                        }
                    }
                    
                } else if (method == "static") {
                    // ============================================================
                    // STATIC STRUCTURE FACTOR: ⟨O₁†O₂⟩ vs temperature (SSSF)
                    // ============================================================
                    int krylov_dim = krylov_dim_or_nmax;
                    int num_samples = 40;
                    unsigned int random_seed = state_idx * 1000 + momentum_idx * 100 + combo_idx;
                    
                    // Determine temperature range
                    double temp_min_static = T_min;
                    double temp_max_static = T_max;
                    int num_temp_bins = T_steps;
                    
                    if (!use_temperature_scan) {
                        // Default temperature range
                        temp_min_static = 0.1;
                        temp_max_static = 10.0;
                        num_temp_bins = 20;
                    }
                    
                    if (rank == 0) {
                        std::cout << "\n============================================" << std::endl;
                        std::cout << "STATIC STRUCTURE FACTOR (SSSF)" << std::endl;
                        std::cout << "============================================" << std::endl;
                        std::cout << "  Krylov dimension: " << krylov_dim << std::endl;
                        std::cout << "  Number of random samples: " << num_samples << std::endl;
                        std::cout << "  Temperature range: [" << temp_min_static << ", " << temp_max_static << "]" << std::endl;
                        std::cout << "  Temperature bins: " << num_temp_bins << std::endl;
                    }
                    
                    // Set up static response parameters
                    StaticResponseParameters params;
                    params.krylov_dim = krylov_dim;
                    params.tolerance = 1e-10;
                    // MEMORY OPTIMIZATION: For large systems (>16M states), disable ALL
                    // reorthogonalization to prevent storing krylov_dim basis vectors.
                    bool large_system = (N > (1ULL << 24));
                    params.full_reorthogonalization = !large_system;
                    params.reorth_frequency = large_system ? 0 : 10;  // 0 = no basis storage
                    if (large_system && rank == 0) {
                        std::cout << "  *** MEMORY OPTIMIZATION: Disabled reorthogonalization ***" << std::endl;
                    }
                    params.num_samples = num_samples;
                    params.random_seed = random_seed;
                    params.compute_error_bars = true;
                    params.store_intermediate = false;
                    
                    // Process each operator pair
                    for (size_t i = 0; i < obs_1.size(); i++) {
                        std::cout << "  Processing operator " << obs_names[i] << std::endl;
                        
                        // Create function wrappers for operators
                        auto O1_func = [&obs_1, i](const Complex* in, Complex* out, int size) {
                            obs_1[i].apply(in, out, size);
                        };
                        
                        auto O2_func = [&obs_2, i](const Complex* in, Complex* out, int size) {
                            obs_2[i].apply(in, out, size);
                        };
                        
                        // Compute static response function
                        auto results = compute_static_response(
                            H, O1_func, O2_func, N, params,
                            temp_min_static, temp_max_static, num_temp_bins, output_base_dir
                        );
                        
                        // Save results to unified HDF5 file
                        // Only rank 0 saves since results have been MPI_Reduced internally
                        if (rank == 0) {
                            saveDSSFStaticToHDF5(
                                unified_h5_path,
                                obs_names[i],
                                sample_index,
                                results.temperatures,
                                results.expectation,
                                results.expectation_error,
                                results.variance,
                                results.variance_error,
                                results.susceptibility,
                                results.susceptibility_error
                            );
                            
                            std::cout << "  Saved static structure factor to HDF5: " << obs_names[i] << std::endl;
                        }
                    }
                    
                } else if (method == "sssf") {
                    // ============================================================
                    // SSSF: Static structure factor on pre-computed TPQ states
                    // Computes ⟨ψ|O₁†O₂|ψ⟩ for each loaded TPQ state
                    // ============================================================
                    // OPTIMIZATIONS APPLIED:
                    // 1. GPU acceleration (if --gpu flag and WITH_CUDA)
                    // 2. O₁=O₂ detection: only apply operator once when identical
                    // 3. OpenMP parallelization of inner product
                    // ============================================================
                    
                    if (rank == 0) {
                        std::cout << "\n============================================" << std::endl;
                        std::cout << "SSSF: Static Structure Factor on TPQ State" << std::endl;
                        std::cout << "============================================" << std::endl;
                        std::cout << "  Beta: " << beta << std::endl;
                        std::cout << "  Sample index: " << sample_index << std::endl;
                        std::cout << "  Momentum: (" << Q[0] << ", " << Q[1] << ", " << Q[2] << ")" << std::endl;
                        std::cout << "  GPU acceleration: " << (use_gpu ? "enabled" : "disabled") << std::endl;
                    }
                    
                    // Process each operator pair
                    for (size_t i = 0; i < obs_1.size(); i++) {
                        std::cout << "  Processing operator " << obs_names[i] << std::endl;
                        
                        Complex expectation_value(0.0, 0.0);
                        
                        // OPTIMIZATION: Check if O₁ and O₂ are the same operator
                        // For sum operators with same spin_combination (e.g., "0,0" = SmSp),
                        // obs_1[i] and obs_2[i] are constructed identically
                        bool operators_identical = (op_type_1 == op_type_2);
                        
#ifdef WITH_CUDA
                        if (use_gpu) {
                            // Check GPU memory availability first
                            size_t free_mem, total_mem;
                            cudaMemGetInfo(&free_mem, &total_mem);
                            size_t required_mem = (operators_identical ? 2 : 3) * N * sizeof(cuDoubleComplex);
                            
                            if (required_mem > free_mem * 0.9) {
                                std::cerr << "  GPU memory insufficient for SSSF (need " 
                                          << required_mem / (1024.0*1024.0*1024.0) << " GB, have "
                                          << free_mem / (1024.0*1024.0*1024.0) << " GB free). Falling back to CPU." << std::endl;
                                use_gpu = false;
                            } else {
                                // GPU-accelerated SSSF computation
                                // Convert operators to GPU (only copies operator data, no state vector allocation)
                                GPUOperator gpu_obs1(num_sites, spin_length);
                                GPUOperator gpu_obs2(num_sites, spin_length);
                                
                                bool gpu_ok = convertOperatorToGPU(obs_1[i], gpu_obs1);
                                if (!operators_identical && gpu_ok) {
                                    gpu_ok = gpu_ok && convertOperatorToGPU(obs_2[i], gpu_obs2);
                                }
                                
                                if (gpu_ok) {
                                    // Allocate device memory with error checking
                                    cuDoubleComplex* d_psi = nullptr;
                                    cuDoubleComplex* d_chi = nullptr;
                                    cuDoubleComplex* d_phi = nullptr;
                                    
                                    cudaError_t err1 = cudaMalloc(&d_psi, N * sizeof(cuDoubleComplex));
                                    cudaError_t err2 = cudaMalloc(&d_chi, N * sizeof(cuDoubleComplex));
                                    cudaError_t err3 = cudaSuccess;
                                    if (!operators_identical) {
                                        err3 = cudaMalloc(&d_phi, N * sizeof(cuDoubleComplex));
                                    }
                                    
                                    if (err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
                                        std::cerr << "  GPU memory allocation failed. Falling back to CPU." << std::endl;
                                        if (d_psi) cudaFree(d_psi);
                                        if (d_chi) cudaFree(d_chi);
                                        if (d_phi) cudaFree(d_phi);
                                        use_gpu = false;
                                    } else {
                                        // Copy state to GPU
                                        cudaMemcpy(d_psi, tpq_state.data(), N * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);
                                        cudaMemset(d_chi, 0, N * sizeof(cuDoubleComplex));
                                
                                        // Apply O₂ on GPU: chi = O₂|ψ⟩
                                        if (operators_identical) {
                                            gpu_obs1.matVecGPU(d_psi, d_chi, N);
                                        } else {
                                            gpu_obs2.matVecGPU(d_psi, d_chi, N);
                                            cudaMemset(d_phi, 0, N * sizeof(cuDoubleComplex));
                                            gpu_obs1.matVecGPU(d_psi, d_phi, N);
                                        }
                                        
                                        // Compute inner product using cuBLAS
                                        cublasHandle_t cublas_handle;
                                        cublasCreate(&cublas_handle);
                                        
                                        cuDoubleComplex result;
                                        if (operators_identical) {
                                            // ⟨chi|chi⟩ = ||O|ψ⟩||²
                                            cublasZdotc(cublas_handle, N, d_chi, 1, d_chi, 1, &result);
                                        } else {
                                            // ⟨phi|chi⟩
                                            cublasZdotc(cublas_handle, N, d_phi, 1, d_chi, 1, &result);
                                        }
                                        
                                        expectation_value = Complex(cuCreal(result), cuCimag(result));
                                        
                                        // Cleanup
                                        cublasDestroy(cublas_handle);
                                        cudaFree(d_psi);
                                        cudaFree(d_chi);
                                        if (!operators_identical) {
                                            cudaFree(d_phi);
                                        }
                                    }  // end cudaMalloc success
                                } else {
                                    std::cerr << "  GPU operator conversion failed, falling back to CPU" << std::endl;
                                    use_gpu = false;
                                }
                            }  // end memory check success
                        }  // end use_gpu
#endif
                        
                        if (!use_gpu) {
                            // CPU computation with optimizations
                            if (operators_identical) {
                                // OPTIMIZATION: O₁ = O₂, only need to apply once
                                // ⟨ψ|O†O|ψ⟩ = ||O|ψ⟩||²
                                ComplexVector chi(N, Complex(0.0, 0.0));
                                obs_1[i].apply(tpq_state.data(), chi.data(), N);
                                
                                // Compute ||chi||² with OpenMP
                                double norm_sq = 0.0;
                                #pragma omp parallel for reduction(+:norm_sq) if(N > 10000)
                                for (int j = 0; j < N; j++) {
                                    norm_sq += std::norm(chi[j]);
                                }
                                expectation_value = Complex(norm_sq, 0.0);
                            } else {
                                // O₁ ≠ O₂, need both applications
                                ComplexVector chi(N, Complex(0.0, 0.0));  // O₂|ψ⟩
                                ComplexVector phi(N, Complex(0.0, 0.0));  // O₁|ψ⟩
                                
                                // Apply operators
                                obs_2[i].apply(tpq_state.data(), chi.data(), N);
                                obs_1[i].apply(tpq_state.data(), phi.data(), N);
                                
                                // Compute inner product ⟨φ|χ⟩ with OpenMP
                                double real_sum = 0.0, imag_sum = 0.0;
                                #pragma omp parallel for reduction(+:real_sum,imag_sum) if(N > 10000)
                                for (int j = 0; j < N; j++) {
                                    Complex prod = std::conj(phi[j]) * chi[j];
                                    real_sum += prod.real();
                                    imag_sum += prod.imag();
                                }
                                expectation_value = Complex(real_sum, imag_sum);
                            }
                        }
                        
                        // Temperature from beta (for output)
                        double temperature = (beta > 0) ? 1.0 / beta : std::numeric_limits<double>::infinity();
                        
                        // Save to HDF5 (appending to existing data)
                        std::vector<double> temperatures_vec = {temperature};
                        std::vector<double> expectation_vec = {expectation_value.real()};
                        std::vector<double> expectation_error_vec = {0.0};
                        std::vector<double> variance_vec = {0.0};
                        std::vector<double> variance_error_vec = {0.0};
                        std::vector<double> susceptibility_vec = {0.0};
                        std::vector<double> susceptibility_error_vec = {0.0};
                        
                        saveDSSFStaticToHDF5(
                            unified_h5_path,
                            obs_names[i],
                            sample_index,
                            temperatures_vec,
                            expectation_vec,
                            expectation_error_vec,
                            variance_vec,
                            variance_error_vec,
                            susceptibility_vec,
                            susceptibility_error_vec
                        );
                        
                        // MEMORY FIX: Flush HDF5 library caches periodically to prevent memory accumulation
                        // HDF5 library caches metadata and data chunks internally, which can grow unbounded
                        H5garbage_collect();
                        
                        if (rank == 0) {
                            std::cout << "  Saved SSSF: " << obs_names[i] 
                                      << " beta=" << beta 
                                      << " S(q)=" << expectation_value.real();
                            if (std::abs(expectation_value.imag()) > 1e-10) {
                                std::cout << " + " << expectation_value.imag() << "i";
                            }
                            if (operators_identical) {
                                std::cout << " [O₁=O₂ optimized]";
                            }
                            std::cout << std::endl;
                        }
                    }
                    
                } else if (method == "continued_fraction") {
                    // ============================================================
                    // CONTINUED FRACTION DSSF on a single state
                    // ============================================================
                    // Uses the continued fraction representation:
                    //   G(z) = ||O|ψ⟩||² / (z - α₀ - β₁²/(z - α₁ - β₂²/...))
                    //   S(ω) = -Im[G(ω + iη)] / π
                    //
                    // This computes the spectral function for a SINGLE state |ψ⟩:
                    //   S(ω) = ⟨ψ|O† δ(ω - H + E₀) O|ψ⟩
                    //
                    // IMPORTANT: For TPQ states at finite temperature, this gives
                    // slightly different results than the spectral method because:
                    // - continued_fraction: Uses diagonal resolvent ⟨φ|(z-H)⁻¹|φ⟩
                    // - spectral: Uses eigendecomposition with exact weights
                    //
                    // The methods converge at low temperature (large β) where the
                    // TPQ state approaches a pure ground state.
                    //
                    // NOTE: Only works for O₁ = O₂ case (self-correlation)
                    // ============================================================
                    int krylov_dim = krylov_dim_or_nmax;
                    
                    // Check if operators are identical (required for continued fraction)
                    bool operators_identical = (op_type_1 == op_type_2);
                    
                    if (!operators_identical) {
                        std::cerr << "Error: continued_fraction method only supports O₁ = O₂ case" << std::endl;
                        std::cerr << "  For cross-correlations (O₁ ≠ O₂), use the 'spectral' method instead" << std::endl;
                        return false;
                    }
                    
                    if (rank == 0) {
                        std::cout << "\n============================================" << std::endl;
                        std::cout << "CONTINUED FRACTION DSSF (single state)" << std::endl;
                        std::cout << "============================================" << std::endl;
                        std::cout << "  Beta: " << beta << std::endl;
                        std::cout << "  Temperature: " << (beta > 0 ? 1.0/beta : std::numeric_limits<double>::infinity()) << std::endl;
                        std::cout << "  Krylov dimension: " << krylov_dim << std::endl;
                        std::cout << "  Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
                        std::cout << "  Broadening: " << broadening << std::endl;
                        std::cout << "  Method: Continued fraction (O(" << krylov_dim << ") per ω point)" << std::endl;
                        if (beta < 50) {
                            std::cout << "  WARNING: At high T (low β), results may differ from spectral method" << std::endl;
                            std::cout << "           For thermal averaging, use spectral or ftlm_thermal instead" << std::endl;
                        }
                    }
                    
                    // Set up ground state DSSF parameters with continued fraction enabled
                    GroundStateDSSFParameters cf_params;
                    cf_params.krylov_dim = krylov_dim;
                    cf_params.omega_min = omega_min;
                    cf_params.omega_max = omega_max;
                    cf_params.num_omega_points = num_omega_bins;
                    cf_params.broadening = broadening;
                    cf_params.tolerance = 1e-10;
                    cf_params.use_continued_fraction = true;  // TRUE continued fraction!
                    // Reorthogonalization settings
                    bool large_system = (N > (1ULL << 24));
                    cf_params.full_reorthogonalization = !large_system;
                    cf_params.reorth_frequency = large_system ? 0 : 10;
                    
                    // Process each operator pair (since O₁ = O₂, only use one)
                    for (size_t i = 0; i < obs_1.size(); i++) {
                        std::cout << "  Processing operator " << obs_names[i] << std::endl;
                        
                        // Create function wrapper for single operator (O₁ = O₂)
                        auto O_func = [&obs_1, i](const Complex* in, Complex* out, int size) {
                            obs_1[i].apply(in, out, size);
                        };
                        
                        // Use compute_ground_state_dssf which computes:
                        //   S(ω) = ||O|ψ⟩||² × (-Im[G(ω+iη)]/π)
                        // where G is the continued fraction of the Lanczos tridiagonal
                        auto results = compute_ground_state_dssf(
                            H, O_func, tpq_state, ground_state_energy, N, cf_params
                        );
                        
                        // Save results to unified HDF5 file
                        saveDSSFSpectralToHDF5(
                            unified_h5_path,
                            obs_names[i],
                            beta,  // Use the TPQ state's beta (finite temperature)
                            sample_index,
                            results.frequencies,
                            results.spectral_function,
                            results.spectral_function_imag,
                            results.spectral_error,
                            results.spectral_error_imag
                        );
                        
                        if (rank == 0) {
                            std::cout << "  Saved continued fraction DSSF to HDF5: " << obs_names[i] << std::endl;
                        }
                    }
                    
                } else if (method == "ground_state") {
                    // ============================================================
                    // GROUND STATE DSSF: T=0 spectral function via continued fraction
                    // ============================================================
                    int krylov_dim = krylov_dim_or_nmax;
                    
                    if (rank == 0) {
                        std::cout << "\n============================================" << std::endl;
                        std::cout << "GROUND STATE DSSF (T=0)" << std::endl;
                        std::cout << "============================================" << std::endl;
                        std::cout << "  Krylov dimension: " << krylov_dim << std::endl;
                        std::cout << "  Frequency range: [" << omega_min << ", " << omega_max << "]" << std::endl;
                        std::cout << "  Broadening: " << broadening << std::endl;
                    }
                    
                    // Load ground state from file
                    ComplexVector ground_state;
                    double gs_energy = ground_state_energy;
                    
                    bool loaded_gs = load_ground_state_from_file(
                        directory + "/output", ground_state, gs_energy, N
                    );
                    
                    if (!loaded_gs) {
                        std::cerr << "Error: Could not load ground state for ground_state method" << std::endl;
                        std::cerr << "  Run diagonalization first to compute ground state eigenvector" << std::endl;
                        return false;
                    }
                    
                    if (rank == 0) {
                        std::cout << "  Ground state energy: " << gs_energy << std::endl;
                        std::cout << "  Ground state dimension: " << ground_state.size() << std::endl;
                    }
                    
                    // Set up ground state DSSF parameters
                    GroundStateDSSFParameters gs_params;
                    gs_params.krylov_dim = krylov_dim;
                    gs_params.omega_min = omega_min;
                    gs_params.omega_max = omega_max;
                    gs_params.num_omega_points = num_omega_bins;
                    gs_params.broadening = broadening;
                    gs_params.tolerance = 1e-10;
                    
                    // Process each operator pair
                    for (size_t i = 0; i < obs_1.size(); i++) {
                        std::cout << "  Processing operator " << obs_names[i] << std::endl;
                        
                        // Create function wrappers for operators
                        auto O1_func = [&obs_1, i](const Complex* in, Complex* out, int size) {
                            obs_1[i].apply(in, out, size);
                        };
                        
                        auto O2_func = [&obs_2, i](const Complex* in, Complex* out, int size) {
                            obs_2[i].apply(in, out, size);
                        };
                        
                        // Compute ground state cross-correlation DSSF
                        auto results = compute_ground_state_cross_correlation(
                            H, O1_func, O2_func, ground_state, gs_energy, N, gs_params
                        );
                        
                        // Save results to unified HDF5 file
                        saveDSSFSpectralToHDF5(
                            unified_h5_path,
                            obs_names[i],
                            std::numeric_limits<double>::infinity(),  // beta=infinity for ground state
                            sample_index,
                            results.frequencies,
                            results.spectral_function,
                            results.spectral_function_imag,
                            results.spectral_error,
                            results.spectral_error_imag
                        );
                        
                        if (rank == 0) {
                            std::cout << "  Saved ground state DSSF to HDF5: " << obs_names[i] << std::endl;
                        }
                    }
                    
                } else {
                    std::cerr << "Rank " << rank << " unknown method: " << method << std::endl;
                    return false;
                }
        } catch (const std::exception &e) {
            std::cerr << "Rank " << rank << " failed time evolution: " << e.what() << std::endl;
            return false;
        }
        
        // MEMORY FIX: Force cleanup of HDF5 internal caches after each task
        // This prevents memory accumulation across tasks due to HDF5's internal caching
        H5garbage_collect();
        
        return true;
    };
    
    // Dynamic work distribution using master-worker pattern
    int local_processed_count = 0;
    double start_time = MPI_Wtime();
    
    // ============================================================================
    // MPI EXECUTION MODE SELECTION
    // ============================================================================
    // 
    // COLLECTIVE MODE (ftlm_thermal, static):
    //   - ALL ranks call process_task() for EACH task TOGETHER
    //   - Inside the task, random Ritz samples are distributed across ranks
    //   - Each rank computes (num_samples / num_ranks) samples
    //   - Results are aggregated via MPI_Reduce within the FTLM function
    //   - Task-level parallelism: NONE (sequential over tasks)
    //   - Sample-level parallelism: YES (MPI distributes random samples)
    //
    // INDEPENDENT MODE (spectral, ground_state, etc.):
    //   - Master-worker pattern: rank 0 dispatches tasks to workers
    //   - Each task processes a single pre-computed quantum state
    //   - No MPI communication within a task
    //   - Task-level parallelism: YES (different ranks work on different tasks)
    //   - Sample-level parallelism: N/A (uses pre-computed states)
    // ============================================================================
    
    bool uses_collective_mpi = (method == "ftlm_thermal" || method == "static");
    
    if (size == 1) {
        // Serial execution
        for (const auto& task : all_tasks) {
            if (process_task(task)) {
                local_processed_count++;
            }
        }
    } else if (uses_collective_mpi) {
        // COLLECTIVE MODE: ALL ranks process each task together
        // Internal FTLM/static functions distribute random samples across ranks
        if (rank == 0) {
            std::cout << "\n==========================================\n";
            std::cout << "COLLECTIVE MPI MODE\n";
            std::cout << "==========================================\n";
            std::cout << "All " << size << " ranks cooperating on each task\n";
            std::cout << "Random samples distributed: ~" << (40 / size) << " samples/rank\n";
            std::cout << "Tasks (momentum × operator): " << num_tasks << "\n";
            std::cout << "Note: ALL temperatures computed within each task\n";
            std::cout << "==========================================\n\n";
        }
        
        for (const auto& task : all_tasks) {
            // All ranks call process_task together - internal MPI distributes samples
            if (process_task(task)) {
                local_processed_count++;
            }
            // Synchronize after each task to ensure HDF5 writes complete
            MPI_Barrier(MPI_COMM_WORLD);
        }
    } else {
        // Master-worker dynamic scheduling with rank 0 also processing tasks
        const int TASK_TAG = 1;
        const int STOP_TAG = 2;
        const int DONE_TAG = 3;
        const int REQUEST_TAG = 4;
        
        if (rank == 0) {
            // Master: dispatch tasks and also process tasks itself
            int next_task = 0;
            int active_workers = std::min(size - 1, num_tasks);
            
            // Send initial batch to other workers
            for (int r = 1; r <= active_workers; ++r) {
                MPI_Send(&next_task, 1, MPI_INT, r, TASK_TAG, MPI_COMM_WORLD);
                next_task++;
            }
            
            // Idle remaining workers
            for (int r = active_workers + 1; r < size; ++r) {
                int dummy = -1;
                MPI_Send(&dummy, 1, MPI_INT, r, STOP_TAG, MPI_COMM_WORLD);
            }
            
            // Process tasks on rank 0 while managing other workers
            int completed = 0;
            while (completed < num_tasks) {
                // Check if rank 0 can grab a task
                if (next_task < num_tasks) {
                    int my_task = next_task;
                    next_task++;
                    
                    std::cout << "Rank 0 processing task " << my_task << "/" << num_tasks;
                    if (all_tasks[my_task].momentum_idx >= 0) {
                        std::cout << " (state=" << all_tasks[my_task].state_idx 
                                  << ", Q=" << all_tasks[my_task].momentum_idx
                                  << ", combo=" << all_tasks[my_task].combo_idx;
                        if (all_tasks[my_task].sublattice_i >= 0) {
                            std::cout << ", sub_i=" << all_tasks[my_task].sublattice_i 
                                      << ", sub_j=" << all_tasks[my_task].sublattice_j;
                        }
                        std::cout << ")";
                    }
                    std::cout << std::endl;
                    
                    if (process_task(all_tasks[my_task])) {
                        local_processed_count++;
                    }
                    completed++;
                }
                
                // Check for completed tasks from other workers (non-blocking)
                int flag;
                MPI_Status status;
                MPI_Iprobe(MPI_ANY_SOURCE, DONE_TAG, MPI_COMM_WORLD, &flag, &status);
                
                if (flag) {
                    int done_task;
                    MPI_Recv(&done_task, 1, MPI_INT, status.MPI_SOURCE, DONE_TAG, MPI_COMM_WORLD, &status);
                    completed++;
                    
                    if (next_task < num_tasks) {
                        MPI_Send(&next_task, 1, MPI_INT, status.MPI_SOURCE, TASK_TAG, MPI_COMM_WORLD);
                        next_task++;
                    } else {
                        int dummy = -1;
                        MPI_Send(&dummy, 1, MPI_INT, status.MPI_SOURCE, STOP_TAG, MPI_COMM_WORLD);
                    }
                }
            }
        } else {
            // Worker: request and process tasks
            while (true) {
                int task_id;
                MPI_Status status;
                MPI_Recv(&task_id, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
                
                if (status.MPI_TAG == STOP_TAG) {
                    break;
                }
                
                std::cout << "Rank " << rank << " processing task " << task_id << "/" << num_tasks;
                if (all_tasks[task_id].momentum_idx >= 0) {
                    std::cout << " (state=" << all_tasks[task_id].state_idx 
                              << ", Q=" << all_tasks[task_id].momentum_idx
                              << ", combo=" << all_tasks[task_id].combo_idx;
                    if (all_tasks[task_id].sublattice_i >= 0) {
                        std::cout << ", sub_i=" << all_tasks[task_id].sublattice_i 
                                  << ", sub_j=" << all_tasks[task_id].sublattice_j;
                    }
                    std::cout << ")";
                }
                std::cout << std::endl;
                
                if (process_task(all_tasks[task_id])) {
                    local_processed_count++;
                }
                
                MPI_Send(&task_id, 1, MPI_INT, 0, DONE_TAG, MPI_COMM_WORLD);
            }
        }
    }
    
    double end_time = MPI_Wtime();
    double elapsed_time = end_time - start_time;
    
    // Gather timing and count statistics
    std::vector<double> all_times;
    if (rank == 0) {
        all_times.resize(size);
    }
    MPI_Gather(&elapsed_time, 1, MPI_DOUBLE, rank == 0 ? all_times.data() : nullptr, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    int total_processed_count;
    MPI_Reduce(&local_processed_count, &total_processed_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    // =======================================================================
    // MPI-SAFE HDF5 MERGE: For INDEPENDENT mode, merge per-rank files
    // =======================================================================
    // In INDEPENDENT mode (spectral, ground_state), each MPI rank writes to 
    // its own per-rank file (dssf_results_rank0.h5, etc.) to avoid HDF5 file 
    // locking conflicts. Now we need to merge all per-rank files into the 
    // unified output file on rank 0.
    // =======================================================================
    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks have finished writing
    
    if (uses_independent_mode && size > 1 && rank == 0) {
        std::cout << "\nMerging per-rank DSSF HDF5 files..." << std::endl;
        mergePerRankDSSFFiles(output_base_dir, size);
        unified_h5_path = output_base_dir + "/dssf_results.h5";  // Update to merged file path
    }
    
    if (rank == 0) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "Processing complete!" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Processed " << total_processed_count << "/" << num_tasks << " tasks successfully." << std::endl;
        std::cout << "All results saved to unified HDF5 file: " << unified_h5_path << std::endl;
        
        // Print timing statistics
        std::cout << "\nTiming statistics:" << std::endl;
        auto max_it = std::max_element(all_times.begin(), all_times.end());
        auto min_it = std::min_element(all_times.begin(), all_times.end());
        double max_time = *max_it;
        double min_time = *min_it;
        double avg_time = std::accumulate(all_times.begin(), all_times.end(), 0.0) / size;
        double load_imbalance = 0.0;
        if (max_time > 0.0) {
            load_imbalance = (max_time - min_time) / max_time * 100.0;
        }
        
        std::cout << "  Max time: " << max_time << " seconds (Rank " 
                  << std::distance(all_times.begin(), max_it) << ")" << std::endl;
        std::cout << "  Min time: " << min_time << " seconds (Rank " 
                  << std::distance(all_times.begin(), min_it) << ")" << std::endl;
        std::cout << "  Avg time: " << avg_time << " seconds" << std::endl;
        std::cout << "  Load imbalance: " << std::fixed << std::setprecision(2) << load_imbalance << "%" << std::endl;
        
        std::cout << "\nPer-rank timing:" << std::endl;
        for (int r = 0; r < size; r++) {
            std::cout << "  Rank " << r << ": " << all_times[r] << " seconds" << std::endl;
        }
    }
    
    MPI_Finalize();
    return 0;
}
