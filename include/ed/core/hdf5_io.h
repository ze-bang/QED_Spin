#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <H5Cpp.h>
#include <vector>
#include <complex>
#include <string>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <fstream>
#include <map>
#include <ed/core/thermal_types.h>

using Complex = std::complex<double>;

/**
 * @brief Comprehensive HDF5 I/O utilities for all exact diagonalization intermediate data
 * 
 * This class provides unified file management for:
 * - Eigenvalues and eigenvectors
 * - Thermodynamic observables (energy, entropy, specific heat, etc.)
 * - Correlation functions (static and dynamical)
 * - Structure factors and dynamical structure factors
 * - FTLM samples and thermal states
 * - TPQ states and intermediate results
 * 
 * File structure:
 *   ed_results.h5
 *   ├── /eigendata
 *   │   ├── eigenvalues [dataset: array of double]
 *   │   ├── eigenvector_0 [dataset: complex vector]
 *   │   ├── eigenvector_1 [dataset: complex vector]
 *   │   └── ...
 *   ├── /thermodynamics
 *   │   ├── temperatures [dataset: array of double]
 *   │   ├── energy [dataset: array of double]
 *   │   ├── entropy [dataset: array of double]
 *   │   ├── specific_heat [dataset: array of double]
 *   │   ├── magnetization [dataset: array of double]
 *   │   └── susceptibility [dataset: array of double]
 *   ├── /correlations
 *   │   ├── spin_spin [dataset: 2D matrix]
 *   │   ├── spin_configuration [dataset: array]
 *   │   └── sublattice_correlation [dataset: array]
 *   ├── /dynamical
 *   │   ├── frequencies [dataset: array of double]
 *   │   ├── spectral_function [dataset: array of double]
 *   │   ├── structure_factor [dataset: 2D or 3D array]
 *   │   └── samples/
 *   │       ├── sample_0 [dataset]
 *   │       └── ...
 *   ├── /ftlm
 *   │   ├── samples/
 *   │   │   ├── sample_0/
 *   │   │   │   ├── eigenvalues
 *   │   │   │   ├── eigenvectors
 *   │   │   │   └── thermodynamics
 *   │   │   └── ...
 *   │   └── averaged/
 *   │       └── observables
 *   └── /tpq
 *       ├── states/
 *       │   ├── state_0_beta_X
 *       │   └── ...
 *       └── observables/
 *           └── ...
 */
class HDF5IO {
public:
    
    // ============================================================================
    // File Management
    // ============================================================================
    
    /**
     * @brief Create or open an HDF5 file for results storage
     * @param directory Directory to create/open the file in
     * @param filename Name of the HDF5 file (default: ed_results.h5)
     * @return Full path to the HDF5 file
     */
    static std::string createOrOpenFile(const std::string& directory, 
                                        const std::string& filename = "ed_results.h5") {
        std::string filepath = directory + "/" + filename;
        
        try {
            // Try to open existing file
            if (fileExists(filepath)) {
                return filepath;
            }
            
            // Create new file
            H5::H5File file(filepath, H5F_ACC_TRUNC);
            
            // Create standard groups
            file.createGroup("/eigendata");
            file.createGroup("/thermodynamics");
            file.createGroup("/correlations");
            file.createGroup("/dynamical");
            file.createGroup("/dynamical/samples");
            file.createGroup("/ftlm");
            file.createGroup("/ftlm/samples");
            file.createGroup("/ftlm/averaged");
            file.createGroup("/tpq");
            file.createGroup("/tpq/states");
            file.createGroup("/tpq/observables");
            
            file.close();
            std::cout << "Created HDF5 results file: " << filepath << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to create/open HDF5 file: " + std::string(e.getCDetailMsg()));
        }
        
        return filepath;
    }
    
    /**
     * @brief Check if HDF5 file exists and is valid
     */
    static bool fileExists(const std::string& filepath) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            file.close();
            return true;
        } catch (H5::Exception& e) {
            return false;
        }
    }
    
    // ============================================================================
    // Eigenvalue/Eigenvector I/O
    // ============================================================================
    
    /**
     * @brief Save eigenvalues to HDF5
     * @param filepath Path to HDF5 file
     * @param eigenvalues Vector of eigenvalues
     */
    static void saveEigenvalues(const std::string& filepath, 
                                const std::vector<double>& eigenvalues) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            hsize_t dims[1] = {eigenvalues.size()};
            H5::DataSpace dataspace(1, dims);
            
            // Delete if exists
            if (file.nameExists("/eigendata/eigenvalues")) {
                file.unlink("/eigendata/eigenvalues");
            }
            
            H5::DataSet dataset = file.createDataSet("/eigendata/eigenvalues", 
                                                     H5::PredType::NATIVE_DOUBLE, 
                                                     dataspace);
            dataset.write(eigenvalues.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Add metadata
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::Attribute count_attr = dataset.createAttribute("count", 
                                                               H5::PredType::NATIVE_UINT64, 
                                                               attr_space);
            uint64_t count = eigenvalues.size();
            count_attr.write(H5::PredType::NATIVE_UINT64, &count);
            count_attr.close();
            
            dataset.close();
            file.close();
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save eigenvalues: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Load eigenvalues from HDF5
     * @param filepath Path to HDF5 file
     * @return Vector of eigenvalues
     */
    static std::vector<double> loadEigenvalues(const std::string& filepath) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            H5::DataSet dataset = file.openDataSet("/eigendata/eigenvalues");
            H5::DataSpace dataspace = dataset.getSpace();
            
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims);
            
            std::vector<double> eigenvalues(dims[0]);
            dataset.read(eigenvalues.data(), H5::PredType::NATIVE_DOUBLE);
            
            dataset.close();
            file.close();
            
            return eigenvalues;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to load eigenvalues: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save a single eigenvector to HDF5 (stored as real, imag pairs)
     * @param filepath Path to HDF5 file
     * @param index Index of the eigenvector
     * @param eigenvector Complex vector
     */
    static void saveEigenvector(const std::string& filepath, 
                                size_t index, 
                                const std::vector<Complex>& eigenvector) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string dataset_name = "/eigendata/eigenvector_" + std::to_string(index);
            
            // Delete if exists
            if (file.nameExists(dataset_name)) {
                file.unlink(dataset_name);
            }
            
            size_t N = eigenvector.size();
            
            // Create compound datatype for complex numbers
            H5::CompType complex_type(2 * sizeof(double));
            complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
            complex_type.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);
            
            // Create dataspace
            hsize_t dims[1] = {N};
            H5::DataSpace dataspace(1, dims);
            
            // Create dataset
            H5::DataSet dataset = file.createDataSet(dataset_name, complex_type, dataspace);
            
            // Prepare data for writing
            struct ComplexPair {
                double real;
                double imag;
            };
            
            std::vector<ComplexPair> data(N);
            for (size_t i = 0; i < N; ++i) {
                data[i].real = eigenvector[i].real();
                data[i].imag = eigenvector[i].imag();
            }
            
            dataset.write(data.data(), complex_type);
            
            // Add dimension as attribute
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::Attribute dim_attr = dataset.createAttribute("dimension", 
                                                             H5::PredType::NATIVE_UINT64, 
                                                             attr_space);
            uint64_t dim = N;
            dim_attr.write(H5::PredType::NATIVE_UINT64, &dim);
            dim_attr.close();
            
            dataset.close();
            file.close();
            
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save eigenvector: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Unified function to save all diagonalization results (eigenvalues + eigenvectors)
     * 
     * This is the preferred method for all solvers to save their results.
     * Creates the output directory and HDF5 file if needed.
     * 
     * @param output_dir Base output directory (e.g., "output")
     * @param eigenvalues Vector of eigenvalues
     * @param eigenvectors Vector of eigenvectors (can be empty if not computed)
     * @param solver_name Name of the solver for logging (e.g., "LANCZOS", "LOBPCG")
     */
    static void saveDiagonalizationResults(
        const std::string& output_dir,
        const std::vector<double>& eigenvalues,
        const std::vector<std::vector<Complex>>& eigenvectors = {},
        const std::string& solver_name = ""
    ) {
        if (output_dir.empty()) return;
        
        // Create eigenvectors subdirectory
        std::string evec_dir = output_dir + "/eigenvectors";
        
        // Use system call to create directory (cross-platform would need filesystem)
        std::string cmd = "mkdir -p " + evec_dir;
        int result = system(cmd.c_str());
        if (result != 0) {
            std::cerr << "Warning: Could not create directory " << evec_dir << std::endl;
        }
        
        // Create/open HDF5 file
        std::string h5_path = createOrOpenFile(evec_dir);
        
        // Save eigenvalues
        saveEigenvalues(h5_path, eigenvalues);
        
        // Save eigenvectors if provided
        for (size_t i = 0; i < eigenvectors.size(); ++i) {
            saveEigenvector(h5_path, i, eigenvectors[i]);
        }
        
        // Log results
        if (!solver_name.empty()) {
            std::cout << solver_name << ": ";
        }
        std::cout << "Saved " << eigenvalues.size() << " eigenvalues";
        if (!eigenvectors.empty()) {
            std::cout << " and " << eigenvectors.size() << " eigenvectors";
        }
        std::cout << " to " << h5_path << std::endl;
    }
    
    /**
     * @brief Load a single eigenvector from HDF5
     * @param filepath Path to HDF5 file
     * @param index Index of the eigenvector
     * @return Complex vector
     */
    static std::vector<Complex> loadEigenvector(const std::string& filepath, size_t index) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            std::string dataset_name = "/eigendata/eigenvector_" + std::to_string(index);
            H5::DataSet dataset = file.openDataSet(dataset_name);
            H5::DataSpace dataspace = dataset.getSpace();
            
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims);
            size_t N = dims[0];
            
            // Define compound type
            H5::CompType complex_type(2 * sizeof(double));
            complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
            complex_type.insertMember("imag", sizeof(double), H5::PredType::NATIVE_DOUBLE);
            
            struct ComplexPair {
                double real;
                double imag;
            };
            
            std::vector<ComplexPair> data(N);
            dataset.read(data.data(), complex_type);
            
            std::vector<Complex> eigenvector(N);
            for (size_t i = 0; i < N; ++i) {
                eigenvector[i] = Complex(data[i].real, data[i].imag);
            }
            
            dataset.close();
            file.close();
            
            return eigenvector;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to load eigenvector: " + std::string(e.getCDetailMsg()));
        }
    }
    
    // ============================================================================
    // Thermodynamics I/O
    // ============================================================================
    
    /**
     * @brief Save thermodynamic data (temperature-dependent observables)
     * @param filepath Path to HDF5 file
     * @param temperatures Vector of temperatures
     * @param observable_name Name of the observable (e.g., "energy", "entropy")
     * @param values Vector of observable values
     */
    static void saveThermodynamics(const std::string& filepath,
                                   const std::vector<double>& temperatures,
                                   const std::string& observable_name,
                                   const std::vector<double>& values) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            // Save temperatures if not already saved
            if (!file.nameExists("/thermodynamics/temperatures")) {
                hsize_t dims[1] = {temperatures.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = file.createDataSet("/thermodynamics/temperatures",
                                                         H5::PredType::NATIVE_DOUBLE,
                                                         dataspace);
                dataset.write(temperatures.data(), H5::PredType::NATIVE_DOUBLE);
                dataset.close();
            }
            
            // Save observable
            std::string dataset_name = "/thermodynamics/" + observable_name;
            if (file.nameExists(dataset_name)) {
                file.unlink(dataset_name);
            }
            
            hsize_t dims[1] = {values.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(dataset_name,
                                                     H5::PredType::NATIVE_DOUBLE,
                                                     dataspace);
            dataset.write(values.data(), H5::PredType::NATIVE_DOUBLE);
            dataset.close();
            
            file.close();
            
            std::cout << "Saved thermodynamic data: " << observable_name << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save thermodynamics: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Load thermodynamic observable
     */
    static std::vector<double> loadThermodynamicObservable(const std::string& filepath,
                                                           const std::string& observable_name) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            std::string dataset_name = "/thermodynamics/" + observable_name;
            H5::DataSet dataset = file.openDataSet(dataset_name);
            H5::DataSpace dataspace = dataset.getSpace();
            
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims);
            
            std::vector<double> values(dims[0]);
            dataset.read(values.data(), H5::PredType::NATIVE_DOUBLE);
            
            dataset.close();
            file.close();
            
            return values;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to load thermodynamic observable: " + 
                                   std::string(e.getCDetailMsg()));
        }
    }
    
    // ============================================================================
    // Correlation Functions I/O
    // ============================================================================
    
    /**
     * @brief Save 2D correlation matrix (e.g., spin-spin correlations)
     * @param filepath Path to HDF5 file
     * @param correlation_name Name of correlation (e.g., "spin_spin", "density_density")
     * @param matrix 2D matrix of correlations (num_sites x num_sites)
     */
    static void saveCorrelationMatrix(const std::string& filepath,
                                      const std::string& correlation_name,
                                      const std::vector<std::vector<Complex>>& matrix) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string dataset_name = "/correlations/" + correlation_name;
            if (file.nameExists(dataset_name)) {
                file.unlink(dataset_name);
            }
            
            size_t n_rows = matrix.size();
            size_t n_cols = matrix[0].size();
            
            // Flatten matrix
            std::vector<double> real_part(n_rows * n_cols);
            std::vector<double> imag_part(n_rows * n_cols);
            
            for (size_t i = 0; i < n_rows; ++i) {
                for (size_t j = 0; j < n_cols; ++j) {
                    real_part[i * n_cols + j] = matrix[i][j].real();
                    imag_part[i * n_cols + j] = matrix[i][j].imag();
                }
            }
            
            // Create datasets for real and imaginary parts
            hsize_t dims[2] = {n_rows, n_cols};
            H5::DataSpace dataspace(2, dims);
            
            H5::DataSet dataset_real = file.createDataSet(dataset_name + "_real",
                                                          H5::PredType::NATIVE_DOUBLE,
                                                          dataspace);
            dataset_real.write(real_part.data(), H5::PredType::NATIVE_DOUBLE);
            dataset_real.close();
            
            H5::DataSet dataset_imag = file.createDataSet(dataset_name + "_imag",
                                                          H5::PredType::NATIVE_DOUBLE,
                                                          dataspace);
            dataset_imag.write(imag_part.data(), H5::PredType::NATIVE_DOUBLE);
            dataset_imag.close();
            
            file.close();
            
            std::cout << "Saved correlation matrix: " << correlation_name 
                      << " (" << n_rows << "x" << n_cols << ")" << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save correlation matrix: " + 
                                   std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save 1D correlation data (e.g., spin configuration)
     * @param filepath Path to HDF5 file
     * @param dataset_name Name of the dataset
     * @param data Vector of values
     */
    static void saveCorrelationData(const std::string& filepath,
                                    const std::string& dataset_name,
                                    const std::vector<Complex>& data) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string full_name = "/correlations/" + dataset_name;
            if (file.nameExists(full_name)) {
                file.unlink(full_name);
            }
            
            size_t N = data.size();
            std::vector<double> real_part(N);
            std::vector<double> imag_part(N);
            
            for (size_t i = 0; i < N; ++i) {
                real_part[i] = data[i].real();
                imag_part[i] = data[i].imag();
            }
            
            hsize_t dims[1] = {N};
            H5::DataSpace dataspace(1, dims);
            
            H5::DataSet dataset_real = file.createDataSet(full_name + "_real",
                                                          H5::PredType::NATIVE_DOUBLE,
                                                          dataspace);
            dataset_real.write(real_part.data(), H5::PredType::NATIVE_DOUBLE);
            dataset_real.close();
            
            H5::DataSet dataset_imag = file.createDataSet(full_name + "_imag",
                                                          H5::PredType::NATIVE_DOUBLE,
                                                          dataspace);
            dataset_imag.write(imag_part.data(), H5::PredType::NATIVE_DOUBLE);
            dataset_imag.close();
            
            file.close();
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save correlation data: " + 
                                   std::string(e.getCDetailMsg()));
        }
    }
    
    // ============================================================================
    // Dynamical Response / Structure Factor I/O
    // ============================================================================
    
    /**
     * @brief Save dynamical structure factor S(q, ω)
     * @param filepath Path to HDF5 file
     * @param operator_name Name of operator (e.g., "S_zz", "S_pm")
     * @param frequencies Frequency grid
     * @param spectral_function Spectral function S(ω)
     * @param metadata Additional metadata as key-value pairs
     */
    static void saveDynamicalResponse(const std::string& filepath,
                                      const std::string& operator_name,
                                      const std::vector<double>& frequencies,
                                      const std::vector<double>& spectral_function,
                                      const std::map<std::string, double>& metadata = {}) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            // Save frequencies if not already saved
            if (!file.nameExists("/dynamical/frequencies")) {
                hsize_t dims[1] = {frequencies.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = file.createDataSet("/dynamical/frequencies",
                                                         H5::PredType::NATIVE_DOUBLE,
                                                         dataspace);
                dataset.write(frequencies.data(), H5::PredType::NATIVE_DOUBLE);
                dataset.close();
            }
            
            // Save spectral function
            std::string dataset_name = "/dynamical/" + operator_name;
            if (file.nameExists(dataset_name)) {
                file.unlink(dataset_name);
            }
            
            hsize_t dims[1] = {spectral_function.size()};
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet(dataset_name,
                                                     H5::PredType::NATIVE_DOUBLE,
                                                     dataspace);
            dataset.write(spectral_function.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Add metadata as attributes
            for (const auto& [key, value] : metadata) {
                H5::DataSpace attr_space(H5S_SCALAR);
                H5::Attribute attr = dataset.createAttribute(key, 
                                                             H5::PredType::NATIVE_DOUBLE, 
                                                             attr_space);
                attr.write(H5::PredType::NATIVE_DOUBLE, &value);
                attr.close();
            }
            
            dataset.close();
            file.close();
            
            std::cout << "Saved dynamical response: " << operator_name << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save dynamical response: " + 
                                   std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save FTLM sample data
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param eigenvalues Sample eigenvalues
     * @param observables Map of observable name to values
     */
    static void saveFTLMSample(const std::string& filepath,
                               size_t sample_index,
                               const std::vector<double>& eigenvalues,
                               const std::map<std::string, std::vector<double>>& observables) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string sample_group = "/ftlm/samples/sample_" + std::to_string(sample_index);
            
            // Create group if it doesn't exist
            if (!file.nameExists(sample_group)) {
                file.createGroup(sample_group);
            }
            
            // Save eigenvalues
            hsize_t dims[1] = {eigenvalues.size()};
            H5::DataSpace dataspace(1, dims);
            
            std::string evals_name = sample_group + "/eigenvalues";
            if (file.nameExists(evals_name)) {
                file.unlink(evals_name);
            }
            
            H5::DataSet eval_dataset = file.createDataSet(evals_name,
                                                          H5::PredType::NATIVE_DOUBLE,
                                                          dataspace);
            eval_dataset.write(eigenvalues.data(), H5::PredType::NATIVE_DOUBLE);
            eval_dataset.close();
            
            // Save observables
            for (const auto& [name, values] : observables) {
                std::string obs_name = sample_group + "/" + name;
                if (file.nameExists(obs_name)) {
                    file.unlink(obs_name);
                }
                
                dims[0] = values.size();
                H5::DataSpace obs_dataspace(1, dims);
                H5::DataSet obs_dataset = file.createDataSet(obs_name,
                                                             H5::PredType::NATIVE_DOUBLE,
                                                             obs_dataspace);
                obs_dataset.write(values.data(), H5::PredType::NATIVE_DOUBLE);
                obs_dataset.close();
            }
            
            file.close();
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save FTLM sample: " + 
                                   std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save TPQ state vector
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param beta Inverse temperature
     * @param state State vector
     */
    static void saveTPQState(const std::string& filepath,
                             size_t sample_index,
                             double beta,
                             const std::vector<Complex>& state) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::stringstream ss;
            ss << "/tpq/states/state_" << sample_index << "_beta_" 
               << std::fixed << std::setprecision(6) << beta;
            std::string dataset_name = ss.str();
            
            if (file.nameExists(dataset_name)) {
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
            
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save TPQ state: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Load TPQ state vector from HDF5
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param beta Inverse temperature
     * @param state Output state vector
     * @return true if successful, false otherwise
     */
    static bool loadTPQState(const std::string& filepath,
                             size_t sample_index,
                             double beta,
                             std::vector<Complex>& state) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            std::stringstream ss;
            ss << "/tpq/states/state_" << sample_index << "_beta_" 
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
    
    /**
     * @brief TPQ state info structure
     */
    struct TPQStateInfo {
        size_t sample_index;
        double beta;
        std::string dataset_name;
    };
    
    /**
     * @brief List all TPQ states stored in an HDF5 file
     * @param filepath Path to HDF5 file
     * @return Vector of TPQStateInfo for each stored state
     */
    static std::vector<TPQStateInfo> listTPQStates(const std::string& filepath) {
        std::vector<TPQStateInfo> states;
        
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            if (!file.nameExists("/tpq/states")) {
                file.close();
                return states;
            }
            
            H5::Group tpq_group = file.openGroup("/tpq/states");
            
            hsize_t num_objs = tpq_group.getNumObjs();
            for (hsize_t i = 0; i < num_objs; ++i) {
                std::string name = tpq_group.getObjnameByIdx(i);
                
                // Parse dataset name: state_<sample>_beta_<beta>
                // Example: state_0_beta_10.500000
                if (name.find("state_") == 0) {
                    size_t beta_pos = name.find("_beta_");
                    if (beta_pos != std::string::npos) {
                        try {
                            std::string sample_str = name.substr(6, beta_pos - 6);
                            std::string beta_str = name.substr(beta_pos + 6);
                            
                            TPQStateInfo info;
                            info.sample_index = std::stoull(sample_str);
                            info.beta = std::stod(beta_str);
                            info.dataset_name = "/tpq/states/" + name;
                            
                            states.push_back(info);
                        } catch (...) {
                            // Skip malformed entries
                        }
                    }
                }
            }
            
            tpq_group.close();
            file.close();
            
        } catch (H5::Exception& e) {
            // Return empty list on error
        }
        
        return states;
    }
    
    /**
     * @brief Load TPQ state by dataset name
     * @param filepath Path to HDF5 file
     * @param dataset_name Full dataset path (e.g., /tpq/states/state_0_beta_10.500000)
     * @param state Output state vector
     * @return true if successful, false otherwise
     */
    static bool loadTPQStateByName(const std::string& filepath,
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
    
    // ============================================================================
    // FTLM/LTLM/Hybrid Thermal Results I/O
    // ============================================================================
    
    /**
     * @brief Save FTLM thermodynamic results with error bars to HDF5
     * @param filepath Path to HDF5 file
     * @param temperatures Temperature array
     * @param energy Energy values
     * @param energy_error Energy error bars
     * @param specific_heat Specific heat values
     * @param specific_heat_error Specific heat error bars
     * @param entropy Entropy values
     * @param entropy_error Entropy error bars
     * @param free_energy Free energy values
     * @param free_energy_error Free energy error bars
     * @param total_samples Number of samples used
     * @param method Method name (FTLM, LTLM, Hybrid)
     */
    static void saveFTLMThermodynamics(
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
        const std::string& method = "FTLM"
    ) {
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
    
    /**
     * @brief Save static response results to HDF5
     * @param filepath Path to HDF5 file
     * @param operator_name Name of operator
     * @param temperatures Temperature array
     * @param expectation Expectation values
     * @param expectation_error Error bars
     * @param variance Variance values (optional)
     * @param variance_error Variance error (optional)
     * @param susceptibility Susceptibility values (optional)
     * @param susceptibility_error Susceptibility error (optional)
     * @param total_samples Number of samples
     */
    static void saveStaticResponse(
        const std::string& filepath,
        const std::string& operator_name,
        const std::vector<double>& temperatures,
        const std::vector<double>& expectation,
        const std::vector<double>& expectation_error,
        const std::vector<double>& variance = {},
        const std::vector<double>& variance_error = {},
        const std::vector<double>& susceptibility = {},
        const std::vector<double>& susceptibility_error = {},
        uint64_t total_samples = 1
    ) {
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
    
    /**
     * @brief Save dynamical response results with complex values and errors to HDF5
     * @param filepath Path to HDF5 file
     * @param operator_name Name of operator
     * @param frequencies Frequency array
     * @param spectral_real Real part of spectral function
     * @param spectral_imag Imaginary part of spectral function
     * @param error_real Real part of error
     * @param error_imag Imaginary part of error
     * @param total_samples Number of samples
     * @param temperature Temperature (optional metadata)
     */
    static void saveDynamicalResponseFull(
        const std::string& filepath,
        const std::string& operator_name,
        const std::vector<double>& frequencies,
        const std::vector<double>& spectral_real,
        const std::vector<double>& spectral_imag,
        const std::vector<double>& error_real,
        const std::vector<double>& error_imag,
        uint64_t total_samples = 1,
        double temperature = 0.0
    ) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            // Ensure dynamical group exists
            if (!file.nameExists("/dynamical")) {
                file.createGroup("/dynamical");
            }
            
            std::string base_path = "/dynamical/" + operator_name;
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
    
    /**
     * @brief Save hybrid thermal method results with method indicators
     */
    static void saveHybridThermalResults(
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
        double ground_state_energy,
        double crossover_temperature,
        uint64_t crossover_index,
        uint64_t ltlm_points,
        uint64_t ftlm_points,
        uint64_t ftlm_samples_used
    ) {
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
            
            // Create method indicator array (0 = LTLM, 1 = FTLM)
            std::vector<int> method_indicator(temperatures.size());
            for (size_t i = 0; i < temperatures.size(); i++) {
                method_indicator[i] = (i < crossover_index) ? 0 : 1;
            }
            
            // Save method indicator
            std::string mi_name = base_path + "/method_indicator";
            if (file.nameExists(mi_name)) {
                file.unlink(mi_name);
            }
            hsize_t dims[1] = {method_indicator.size()};
            H5::DataSpace mi_dataspace(1, dims);
            H5::DataSet mi_dataset = file.createDataSet(mi_name,
                                                        H5::PredType::NATIVE_INT,
                                                        mi_dataspace);
            mi_dataset.write(method_indicator.data(), H5::PredType::NATIVE_INT);
            mi_dataset.close();
            
            // Save metadata as attributes on the group
            H5::Group group = file.openGroup(base_path);
            H5::DataSpace attr_space(H5S_SCALAR);
            
            auto setDoubleAttr = [&](const std::string& name, double value) {
                if (group.attrExists(name)) group.removeAttr(name);
                H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_DOUBLE, attr_space);
                attr.write(H5::PredType::NATIVE_DOUBLE, &value);
                attr.close();
            };
            
            auto setUint64Attr = [&](const std::string& name, uint64_t value) {
                if (group.attrExists(name)) group.removeAttr(name);
                H5::Attribute attr = group.createAttribute(name, H5::PredType::NATIVE_UINT64, attr_space);
                attr.write(H5::PredType::NATIVE_UINT64, &value);
                attr.close();
            };
            
            setDoubleAttr("ground_state_energy", ground_state_energy);
            setDoubleAttr("crossover_temperature", crossover_temperature);
            setUint64Attr("crossover_index", crossover_index);
            setUint64Attr("ltlm_points", ltlm_points);
            setUint64Attr("ftlm_points", ftlm_points);
            setUint64Attr("ftlm_samples_used", ftlm_samples_used);
            
            // Method string
            std::string method = "Hybrid";
            if (group.attrExists("method")) group.removeAttr("method");
            H5::StrType str_type(H5::PredType::C_S1, method.size() + 1);
            H5::Attribute method_attr = group.createAttribute("method", str_type, attr_space);
            method_attr.write(str_type, method.c_str());
            method_attr.close();
            
            group.close();
            file.close();
            
            std::cout << "Saved Hybrid thermal results to HDF5" << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save hybrid thermal results: " + 
                                   std::string(e.getCDetailMsg()));
        }
    }
    
    // ============================================================================
    // Generic Array Save/Load
    // ============================================================================
    
    /**
     * @brief Generic save for 1D double array with custom path
     */
    static void saveArray(const std::string& filepath,
                         const std::string& dataset_path,
                         const std::vector<double>& data,
                         const std::map<std::string, std::string>& string_attrs = {},
                         const std::map<std::string, double>& double_attrs = {}) {
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
    
    /**
     * @brief Generic load for 1D double array
     */
    static std::vector<double> loadArray(const std::string& filepath,
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
};

#endif // HDF5_IO_H
