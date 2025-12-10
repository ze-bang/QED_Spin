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
#include <algorithm>
#include <filesystem>
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
 *       ├── samples/
 *       │   ├── sample_0/
 *       │   │   ├── thermodynamics [dataset: beta, energy, variance, doublon, step]
 *       │   │   ├── norm [dataset: beta, norm, first_norm, step]
 *       │   │   ├── fluctuations [dataset: spin fluctuation data]
 *       │   │   └── states/
 *       │   │       ├── beta_0.100000 [dataset: complex state vector]
 *       │   │       ├── beta_1.000000 [dataset: complex state vector]
 *       │   │       └── ...
 *       │   ├── sample_1/
 *       │   │   └── ...
 *       │   └── ...
 *       └── averaged/
 *           └── thermodynamics [dataset: averaged over all samples]
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
            file.createGroup("/tpq/samples");   // Per-sample TPQ data (thermodynamics, norm, states)
            file.createGroup("/tpq/averaged");  // Averaged thermodynamic results
            
            file.close();
            std::cout << "Created HDF5 results file: " << filepath << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to create/open HDF5 file: " + std::string(e.getCDetailMsg()));
        }
        
        return filepath;
    }
    
    /**
     * @brief Check if HDF5 file exists and is valid
     * Uses filesystem check first to avoid HDF5 error messages when file doesn't exist
     */
    static bool fileExists(const std::string& filepath) {
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
        
        // Create output directory if needed (for .dat files and HDF5)
        std::string cmd = "mkdir -p " + output_dir;
        int result = system(cmd.c_str());
        if (result != 0) {
            std::cerr << "Warning: Could not create directory " << output_dir << std::endl;
        }
        
        // Create/open HDF5 file in main output directory (unified ed_results.h5)
        std::string h5_path = createOrOpenFile(output_dir);
        
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
     * @param sample_index Optional: filter by sample index (-1 for all samples)
     * @return Vector of TPQStateInfo for each stored state
     */
    static std::vector<TPQStateInfo> listTPQStates(const std::string& filepath, 
                                                    int sample_filter = -1) {
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
                                } catch (...) {
                                    // Skip malformed entries
                                }
                            }
                        }
                        states_group.close();
                    } catch (...) {
                        // Skip malformed sample directories
                    }
                }
            }
            
            samples_group.close();
            file.close();
            
        } catch (H5::Exception& e) {
            // Return empty list on error
        }
        
        return states;
    }
    
    /**
     * @brief List TPQ states for a specific sample
     * @param filepath Path to HDF5 file  
     * @param sample_index Sample index
     * @return Vector of TPQStateInfo for the specified sample
     */
    static std::vector<TPQStateInfo> listTPQStatesForSample(const std::string& filepath,
                                                             size_t sample_index) {
        return listTPQStates(filepath, static_cast<int>(sample_index));
    }

    /**
     * @brief Load TPQ state by dataset name
     * @param filepath Path to HDF5 file
     * @param dataset_name Full dataset path (e.g., /tpq/samples/sample_0/states/beta_10.500000)
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
    // TPQ Per-Sample Thermodynamic Data I/O (replaces SS_rand*.dat / norm_rand*.dat)
    // ============================================================================
    
    /**
     * @brief Structure to hold TPQ thermodynamic data for a single measurement point
     */
    struct TPQThermodynamicPoint {
        double beta;        // Inverse temperature
        double energy;      // Energy expectation value
        double variance;    // Energy variance
        double doublon;     // Doublon expectation (or other observable)
        uint64_t step;      // TPQ step number
    };
    
    /**
     * @brief Structure to hold TPQ norm data for a single measurement point
     */
    struct TPQNormPoint {
        double beta;        // Inverse temperature
        double norm;        // Current norm
        double first_norm;  // Initial norm
        uint64_t step;      // TPQ step number
    };
    
    /**
     * @brief Ensure TPQ sample group exists in HDF5 file
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     */
    static void ensureTPQSampleGroup(const std::string& filepath, size_t sample_index) {
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
    
    /**
     * @brief Append TPQ thermodynamic data point to HDF5 (replaces SS_rand*.dat writing)
     * 
     * This function appends a single measurement point to the sample's thermodynamics dataset.
     * Data is stored as: [beta, energy, variance, doublon, step]
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index (0, 1, 2, ...)
     * @param point Thermodynamic data point to append
     */
    static void appendTPQThermodynamics(const std::string& filepath,
                                        size_t sample_index,
                                        const TPQThermodynamicPoint& point) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/thermodynamics";
            
            // Ensure group exists
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
                
                // Enable chunking for extensible dataset
                H5::DSetCreatPropList plist;
                hsize_t chunk_dims[2] = {100, num_cols};  // Chunk by 100 rows
                plist.setChunk(2, chunk_dims);
                plist.setDeflate(6);  // Compression level
                
                H5::DataSet dataset = file.createDataSet(dataset_path, 
                                                         H5::PredType::NATIVE_DOUBLE, 
                                                         dataspace, plist);
                dataset.write(row_data, H5::PredType::NATIVE_DOUBLE);
                dataset.close();
            } else {
                // Append to existing dataset
                H5::DataSet dataset = file.openDataSet(dataset_path);
                H5::DataSpace filespace = dataset.getSpace();
                
                hsize_t dims[2];
                filespace.getSimpleExtentDims(dims);
                
                // Extend dataset
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
            }
            
            file.close();
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to append TPQ thermodynamics: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Append TPQ norm data point to HDF5 (replaces norm_rand*.dat writing)
     * 
     * Data is stored as: [beta, norm, first_norm, step]
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param point Norm data point to append
     */
    static void appendTPQNorm(const std::string& filepath,
                              size_t sample_index,
                              const TPQNormPoint& point) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/norm";
            
            // Ensure group exists
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
                
                H5::DSetCreatPropList plist;
                hsize_t chunk_dims[2] = {100, num_cols};
                plist.setChunk(2, chunk_dims);
                plist.setDeflate(6);
                
                H5::DataSet dataset = file.createDataSet(dataset_path, 
                                                         H5::PredType::NATIVE_DOUBLE, 
                                                         dataspace, plist);
                dataset.write(row_data, H5::PredType::NATIVE_DOUBLE);
                dataset.close();
            } else {
                // Append to existing dataset
                H5::DataSet dataset = file.openDataSet(dataset_path);
                H5::DataSpace filespace = dataset.getSpace();
                
                hsize_t dims[2];
                filespace.getSimpleExtentDims(dims);
                
                hsize_t new_dims[2] = {dims[0] + 1, num_cols};
                dataset.extend(new_dims);
                
                filespace = dataset.getSpace();
                hsize_t offset[2] = {dims[0], 0};
                hsize_t count[2] = {1, num_cols};
                filespace.selectHyperslab(H5S_SELECT_SET, count, offset);
                
                H5::DataSpace memspace(2, count);
                dataset.write(row_data, H5::PredType::NATIVE_DOUBLE, memspace, filespace);
                dataset.close();
            }
            
            file.close();
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to append TPQ norm: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save complete TPQ thermodynamic trajectory for a sample
     * 
     * Batch write of all thermodynamic data points (more efficient than append)
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param points Vector of thermodynamic data points
     */
    static void saveTPQThermodynamics(const std::string& filepath,
                                      size_t sample_index,
                                      const std::vector<TPQThermodynamicPoint>& points) {
        if (points.empty()) return;
        
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string sample_group = "/tpq/samples/sample_" + std::to_string(sample_index);
            std::string dataset_path = sample_group + "/thermodynamics";
            
            // Ensure groups exist
            if (!file.nameExists("/tpq/samples")) {
                file.createGroup("/tpq/samples");
            }
            if (!file.nameExists(sample_group)) {
                file.createGroup(sample_group);
            }
            if (file.nameExists(dataset_path)) {
                file.unlink(dataset_path);
            }
            
            // Prepare data array
            const hsize_t num_rows = points.size();
            const hsize_t num_cols = 5;
            std::vector<double> data(num_rows * num_cols);
            
            for (size_t i = 0; i < num_rows; ++i) {
                data[i * num_cols + 0] = points[i].beta;
                data[i * num_cols + 1] = points[i].energy;
                data[i * num_cols + 2] = points[i].variance;
                data[i * num_cols + 3] = points[i].doublon;
                data[i * num_cols + 4] = static_cast<double>(points[i].step);
            }
            
            // Create dataset
            hsize_t dims[2] = {num_rows, num_cols};
            H5::DataSpace dataspace(2, dims);
            
            H5::DSetCreatPropList plist;
            plist.setDeflate(6);
            
            H5::DataSet dataset = file.createDataSet(dataset_path, 
                                                     H5::PredType::NATIVE_DOUBLE, 
                                                     dataspace, plist);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Add column labels as attribute
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::StrType str_type(H5::PredType::C_S1, 64);
            std::string columns = "beta,energy,variance,doublon,step";
            H5::Attribute attr = dataset.createAttribute("columns", str_type, attr_space);
            attr.write(str_type, columns.c_str());
            attr.close();
            
            dataset.close();
            file.close();
            
            std::cout << "Saved TPQ thermodynamics for sample " << sample_index 
                      << " (" << num_rows << " points)" << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save TPQ thermodynamics: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save complete TPQ norm trajectory for a sample
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param points Vector of norm data points
     */
    static void saveTPQNorm(const std::string& filepath,
                            size_t sample_index,
                            const std::vector<TPQNormPoint>& points) {
        if (points.empty()) return;
        
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string sample_group = "/tpq/samples/sample_" + std::to_string(sample_index);
            std::string dataset_path = sample_group + "/norm";
            
            // Ensure groups exist
            if (!file.nameExists("/tpq/samples")) {
                file.createGroup("/tpq/samples");
            }
            if (!file.nameExists(sample_group)) {
                file.createGroup(sample_group);
            }
            if (file.nameExists(dataset_path)) {
                file.unlink(dataset_path);
            }
            
            // Prepare data array
            const hsize_t num_rows = points.size();
            const hsize_t num_cols = 4;
            std::vector<double> data(num_rows * num_cols);
            
            for (size_t i = 0; i < num_rows; ++i) {
                data[i * num_cols + 0] = points[i].beta;
                data[i * num_cols + 1] = points[i].norm;
                data[i * num_cols + 2] = points[i].first_norm;
                data[i * num_cols + 3] = static_cast<double>(points[i].step);
            }
            
            // Create dataset
            hsize_t dims[2] = {num_rows, num_cols};
            H5::DataSpace dataspace(2, dims);
            
            H5::DSetCreatPropList plist;
            plist.setDeflate(6);
            
            H5::DataSet dataset = file.createDataSet(dataset_path, 
                                                     H5::PredType::NATIVE_DOUBLE, 
                                                     dataspace, plist);
            dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
            
            // Add column labels as attribute
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::StrType str_type(H5::PredType::C_S1, 64);
            std::string columns = "beta,norm,first_norm,step";
            H5::Attribute attr = dataset.createAttribute("columns", str_type, attr_space);
            attr.write(str_type, columns.c_str());
            attr.close();
            
            dataset.close();
            file.close();
            
            std::cout << "Saved TPQ norm for sample " << sample_index 
                      << " (" << num_rows << " points)" << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save TPQ norm: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Load TPQ thermodynamic data for a sample
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @return Vector of thermodynamic data points
     */
    static std::vector<TPQThermodynamicPoint> loadTPQThermodynamics(const std::string& filepath,
                                                                     size_t sample_index) {
        std::vector<TPQThermodynamicPoint> points;
        
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/thermodynamics";
            
            if (!file.nameExists(dataset_path)) {
                file.close();
                return points;
            }
            
            H5::DataSet dataset = file.openDataSet(dataset_path);
            H5::DataSpace dataspace = dataset.getSpace();
            
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims);
            hsize_t num_rows = dims[0];
            hsize_t num_cols = dims[1];
            
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
            // Return empty vector on error
        }
        
        return points;
    }
    
    /**
     * @brief Load TPQ norm data for a sample
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @return Vector of norm data points
     */
    static std::vector<TPQNormPoint> loadTPQNorm(const std::string& filepath,
                                                  size_t sample_index) {
        std::vector<TPQNormPoint> points;
        
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            std::string dataset_path = "/tpq/samples/sample_" + std::to_string(sample_index) + "/norm";
            
            if (!file.nameExists(dataset_path)) {
                file.close();
                return points;
            }
            
            H5::DataSet dataset = file.openDataSet(dataset_path);
            H5::DataSpace dataspace = dataset.getSpace();
            
            hsize_t dims[2];
            dataspace.getSimpleExtentDims(dims);
            hsize_t num_rows = dims[0];
            hsize_t num_cols = dims[1];
            
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
        } catch (H5::Exception& e) {
            // Return empty vector on error
        }
        
        return points;
    }
    
    /**
     * @brief List all TPQ samples in an HDF5 file
     * 
     * @param filepath Path to HDF5 file
     * @return Vector of sample indices that have data
     */
    static std::vector<size_t> listTPQSamples(const std::string& filepath) {
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
                    } catch (...) {
                        // Skip malformed entries
                    }
                }
            }
            
            group.close();
            file.close();
            
            // Sort samples
            std::sort(samples.begin(), samples.end());
        } catch (H5::Exception& e) {
            // Return empty vector on error
        }
        
        return samples;
    }
    
    /**
     * @brief Save TPQ averaged thermodynamics (combined from all samples)
     * 
     * @param filepath Path to HDF5 file
     * @param beta Inverse temperature array
     * @param energy Energy (mean over samples)
     * @param energy_error Energy error (std error)
     * @param specific_heat Specific heat
     * @param specific_heat_error Specific heat error
     * @param entropy Entropy
     * @param entropy_error Entropy error
     * @param num_samples Number of samples used
     */
    static void saveTPQAveragedThermodynamics(
        const std::string& filepath,
        const std::vector<double>& beta,
        const std::vector<double>& energy,
        const std::vector<double>& energy_error,
        const std::vector<double>& specific_heat,
        const std::vector<double>& specific_heat_error,
        const std::vector<double>& entropy,
        const std::vector<double>& entropy_error,
        uint64_t num_samples
    ) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string base_path = "/tpq/averaged";
            
            // Ensure group exists
            if (!file.nameExists("/tpq")) {
                file.createGroup("/tpq");
            }
            if (!file.nameExists(base_path)) {
                file.createGroup(base_path);
            }
            
            // Helper to save dataset
            auto saveDataset = [&](const std::string& name, const std::vector<double>& data) {
                std::string path = base_path + "/" + name;
                if (file.nameExists(path)) {
                    file.unlink(path);
                }
                hsize_t dims[1] = {data.size()};
                H5::DataSpace dataspace(1, dims);
                H5::DataSet dataset = file.createDataSet(path, H5::PredType::NATIVE_DOUBLE, dataspace);
                dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
                dataset.close();
            };
            
            // Convert beta to temperature for convenience
            std::vector<double> temperatures(beta.size());
            for (size_t i = 0; i < beta.size(); ++i) {
                temperatures[i] = (beta[i] > 0) ? 1.0 / beta[i] : 0.0;
            }
            
            saveDataset("beta", beta);
            saveDataset("temperatures", temperatures);
            saveDataset("energy", energy);
            saveDataset("energy_error", energy_error);
            saveDataset("specific_heat", specific_heat);
            saveDataset("specific_heat_error", specific_heat_error);
            saveDataset("entropy", entropy);
            saveDataset("entropy_error", entropy_error);
            
            // Save metadata
            H5::Group group = file.openGroup(base_path);
            H5::DataSpace attr_space(H5S_SCALAR);
            
            if (group.attrExists("num_samples")) {
                group.removeAttr("num_samples");
            }
            H5::Attribute attr = group.createAttribute("num_samples", H5::PredType::NATIVE_UINT64, attr_space);
            attr.write(H5::PredType::NATIVE_UINT64, &num_samples);
            attr.close();
            
            if (group.attrExists("method")) {
                group.removeAttr("method");
            }
            std::string method = "TPQ";
            H5::StrType str_type(H5::PredType::C_S1, 16);
            H5::Attribute method_attr = group.createAttribute("method", str_type, attr_space);
            method_attr.write(str_type, method.c_str());
            method_attr.close();
            
            group.close();
            file.close();
            
            std::cout << "Saved TPQ averaged thermodynamics to HDF5" << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save TPQ averaged thermodynamics: " + std::string(e.getCDetailMsg()));
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
