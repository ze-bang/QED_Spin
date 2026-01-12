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
#include <sstream>
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
    // File Management - Safe Writing Protocol
    // ============================================================================
    // 
    // SAFE WRITING PROTOCOL:
    // The HDF5 I/O system uses a safe writing protocol that:
    // 1. Opens existing files in read/write mode (H5F_ACC_RDWR) - never truncates
    // 2. Creates new files only if they don't exist
    // 3. Ensures required groups exist without overwriting existing data
    // 4. Uses segment-based writing where each run writes to its own segment
    //    (e.g., different TPQ samples, different temperatures, different operators)
    // 5. Existing segments from previous runs are preserved
    //
    // This allows:
    // - Multiple runs to accumulate data in the same file
    // - Restarting failed runs without losing previous data
    // - Adding new TPQ samples to existing files
    // - Computing different operators/temperatures incrementally
    // ============================================================================
    
    /**
     * @brief Ensure a group exists in an HDF5 file, creating it only if needed
     * 
     * This is a safe operation that does not overwrite existing groups.
     * It also handles nested paths by creating parent groups as needed.
     * 
     * @param file Reference to open HDF5 file
     * @param group_path Full path to the group (e.g., "/tpq/samples/sample_0")
     */
    static void ensureGroupExists(H5::H5File& file, const std::string& group_path) {
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
    
    /**
     * @brief Ensure standard ED result groups exist in an HDF5 file
     * 
     * Creates the standard group structure if groups don't already exist.
     * This is safe to call on files with existing data.
     * 
     * @param file Reference to open HDF5 file
     */
    static void ensureStandardGroups(H5::H5File& file) {
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
    
    /**
     * @brief Create or open an HDF5 file for results storage (SAFE - preserves existing data)
     * 
     * SAFE WRITING PROTOCOL:
     * - If the file exists, opens it in read/write mode without truncating
     * - If the file doesn't exist, creates a new file with standard groups
     * - Always ensures standard groups exist (creates if missing)
     * - NEVER overwrites or truncates existing data
     * 
     * @param directory Directory to create/open the file in
     * @param filename Name of the HDF5 file (default: ed_results.h5)
     * @return Full path to the HDF5 file
     */
    static std::string createOrOpenFile(const std::string& directory, 
                                        const std::string& filename = "ed_results.h5") {
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
    
    /**
     * @brief Force create a new HDF5 file (UNSAFE - truncates existing data)
     * 
     * WARNING: This function will DELETE all existing data in the file.
     * Use only when you explicitly want to start fresh.
     * 
     * @param directory Directory to create the file in
     * @param filename Name of the HDF5 file (default: ed_results.h5)
     * @return Full path to the HDF5 file
     */
    static std::string forceCreateFile(const std::string& directory, 
                                       const std::string& filename = "ed_results.h5") {
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
     * 
     * SAFE WRITING: By default, skips saving if state at this beta already exists.
     * Use overwrite=true to replace existing state.
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param beta Inverse temperature
     * @param state State vector
     * @param overwrite If true, overwrite existing state; if false (default), skip if exists
     * @return true if state was saved, false if skipped (already exists and overwrite=false)
     */
    static bool saveTPQState(const std::string& filepath,
                             size_t sample_index,
                             double beta,
                             const std::vector<Complex>& state,
                             bool overwrite = false) {
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
     * SAFE WRITING: Skips writing if this step already exists in the dataset.
     * This supports continue_quenching mode where runs may overlap.
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index (0, 1, 2, ...)
     * @param point Thermodynamic data point to append
     * @return true if data was written, false if step already existed (skipped)
     */
    static bool appendTPQThermodynamics(const std::string& filepath,
                                        size_t sample_index,
                                        const TPQThermodynamicPoint& point) {
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
                file.close();
                return true;
            } else {
                // Check if step already exists before appending
                H5::DataSet dataset = file.openDataSet(dataset_path);
                H5::DataSpace filespace = dataset.getSpace();
                
                hsize_t dims[2];
                filespace.getSimpleExtentDims(dims);
                
                // Read existing step values (column 4) to check for duplicates
                if (dims[0] > 0) {
                    std::vector<double> existing_data(dims[0] * num_cols);
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
            }
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to append TPQ thermodynamics: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Append TPQ norm data point to HDF5 (replaces norm_rand*.dat writing)
     * 
     * Data is stored as: [beta, norm, first_norm, step]
     * 
     * SAFE WRITING: Skips writing if this step already exists in the dataset.
     * This supports continue_quenching mode where runs may overlap.
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param point Norm data point to append
     * @return true if data was written, false if step already existed (skipped)
     */
    static bool appendTPQNorm(const std::string& filepath,
                              size_t sample_index,
                              const TPQNormPoint& point) {
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
                
                H5::DSetCreatPropList plist;
                hsize_t chunk_dims[2] = {100, num_cols};
                plist.setChunk(2, chunk_dims);
                plist.setDeflate(6);
                
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
                
                // Read existing step values (column 3) to check for duplicates
                if (dims[0] > 0) {
                    std::vector<double> existing_data(dims[0] * num_cols);
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
            }
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
    
    // ============================================================================
    // FTLM Sample Data I/O (replaces ftlm_samples/*.dat and dynamical_samples/*.txt)
    // ============================================================================
    
    /**
     * @brief Structure to hold FTLM thermodynamic sample data
     */
    struct FTLMThermodynamicSample {
        std::vector<double> temperatures;
        std::vector<double> energy;
        std::vector<double> specific_heat;
        std::vector<double> entropy;
        std::vector<double> free_energy;
    };
    
    /**
     * @brief Structure to hold FTLM dynamical sample data (spectral function)
     */
    struct FTLMDynamicalSample {
        std::vector<double> frequencies;
        std::vector<double> spectral_real;
        std::vector<double> spectral_imag;
    };
    
    /**
     * @brief Structure to hold FTLM static response sample data
     */
    struct FTLMStaticSample {
        std::vector<double> temperatures;
        std::vector<double> expectation;
        std::vector<double> variance;
    };
    
    /**
     * @brief Ensure FTLM sample groups exist in HDF5 file
     */
    static void ensureFTLMSampleGroups(const std::string& filepath) {
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
    
    /**
     * @brief Save FTLM thermodynamic sample to HDF5 (replaces ftlm_samples/sample_*.dat)
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param sample Sample data
     */
    static void saveFTLMThermodynamicSample(const std::string& filepath,
                                            size_t sample_index,
                                            const FTLMThermodynamicSample& sample) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string sample_group = "/ftlm/samples/thermodynamic/sample_" + std::to_string(sample_index);
            
            // Create group if needed
            if (!file.nameExists("/ftlm/samples/thermodynamic")) {
                if (!file.nameExists("/ftlm/samples")) {
                    if (!file.nameExists("/ftlm")) {
                        file.createGroup("/ftlm");
                    }
                    file.createGroup("/ftlm/samples");
                }
                file.createGroup("/ftlm/samples/thermodynamic");
            }
            if (file.nameExists(sample_group)) {
                // Delete existing group contents
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
            saveDataset("energy", sample.energy);
            saveDataset("specific_heat", sample.specific_heat);
            saveDataset("entropy", sample.entropy);
            saveDataset("free_energy", sample.free_energy);
            
            file.close();
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save FTLM thermodynamic sample: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save FTLM dynamical sample to HDF5 (replaces dynamical_samples/sample_*.txt)
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param sample Sample data
     * @param is_correlation If true, saves to dynamical_correlation group
     */
    static void saveFTLMDynamicalSample(const std::string& filepath,
                                        size_t sample_index,
                                        const FTLMDynamicalSample& sample,
                                        bool is_correlation = false) {
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
    
    /**
     * @brief Save FTLM static response sample to HDF5 (replaces static_samples/sample_*.txt)
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param sample Sample data
     * @param operator_name Name of the operator (optional, for labeling)
     */
    static void saveFTLMStaticSample(const std::string& filepath,
                                     size_t sample_index,
                                     const FTLMStaticSample& sample,
                                     const std::string& operator_name = "") {
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
    
    // ============================================================================
    // Time Correlation I/O (replaces time_corr_*.dat files)
    // ============================================================================
    
    /**
     * @brief Structure to hold time correlation data
     */
    struct TimeCorrelationData {
        std::vector<double> times;
        std::vector<double> correlation_real;
        std::vector<double> correlation_imag;
    };
    
    /**
     * @brief Ensure time correlation groups exist in HDF5 file
     */
    static void ensureTimeCorrelationGroups(const std::string& filepath) {
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
    
    /**
     * @brief Save time correlation data to HDF5 (replaces time_corr_*.dat files)
     * 
     * @param filepath Path to HDF5 file
     * @param operator_name Operator name (e.g., "Sz_Sz", "Sp_Sm")
     * @param sample_index Sample index
     * @param beta Inverse temperature
     * @param data Time correlation data
     * @param label Additional label (e.g., "ground_state", "thermal")
     */
    static void saveTimeCorrelation(const std::string& filepath,
                                    const std::string& operator_name,
                                    size_t sample_index,
                                    double beta,
                                    const TimeCorrelationData& data,
                                    const std::string& label = "") {
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
    
    /**
     * @brief Load time correlation data from HDF5
     * 
     * @param filepath Path to HDF5 file
     * @param group_path Full path to the time correlation group
     * @param data Output time correlation data
     * @return true if successful, false otherwise
     */
    static bool loadTimeCorrelation(const std::string& filepath,
                                    const std::string& group_path,
                                    TimeCorrelationData& data) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            if (!file.nameExists(group_path)) {
                file.close();
                return false;
            }
            
            // Helper to load dataset
            auto loadDataset = [&](const std::string& name) -> std::vector<double> {
                std::string path = group_path + "/" + name;
                H5::DataSet dataset = file.openDataSet(path);
                H5::DataSpace dataspace = dataset.getSpace();
                hsize_t dims[1];
                dataspace.getSimpleExtentDims(dims);
                std::vector<double> arr(dims[0]);
                dataset.read(arr.data(), H5::PredType::NATIVE_DOUBLE);
                dataset.close();
                return arr;
            };
            
            data.times = loadDataset("times");
            data.correlation_real = loadDataset("correlation_real");
            data.correlation_imag = loadDataset("correlation_imag");
            
            file.close();
            return true;
        } catch (H5::Exception& e) {
            return false;
        }
    }
    
    /**
     * @brief List all time correlation datasets in an HDF5 file
     * 
     * @param filepath Path to HDF5 file
     * @return Vector of group paths for each time correlation dataset
     */
    static std::vector<std::string> listTimeCorrelations(const std::string& filepath) {
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
    
    // ============================================================================
    // MPI-Safe HDF5 I/O Functions
    // ============================================================================
    // Industry standard approach: each MPI rank writes to its own file, then
    // rank 0 merges all per-rank files at the end.
    
    /**
     * @brief Get MPI-safe filename for per-rank HDF5 file
     * @param directory Base output directory
     * @param rank MPI rank (0 for serial execution)
     * @param filename Base filename (default: ed_results.h5)
     * @return Full path to per-rank file (e.g., ed_results_rank0.h5)
     */
    static std::string getPerRankFilePath(const std::string& directory,
                                          int rank,
                                          const std::string& filename = "ed_results.h5") {
        // Extract base name and extension
        size_t dot_pos = filename.rfind('.');
        std::string base = (dot_pos != std::string::npos) ? filename.substr(0, dot_pos) : filename;
        std::string ext = (dot_pos != std::string::npos) ? filename.substr(dot_pos) : "";
        
        return directory + "/" + base + "_rank" + std::to_string(rank) + ext;
    }
    
    /**
     * @brief Create or open per-rank HDF5 file for MPI-safe writing (SAFE)
     * 
     * SAFE WRITING PROTOCOL:
     * - If file exists, opens in read/write mode (preserves existing data)
     * - If file doesn't exist, creates new file with standard groups
     * - Never truncates existing data
     * 
     * @param directory Output directory
     * @param rank MPI rank
     * @param filename Base filename (default: ed_results.h5)
     * @return Full path to created/opened file
     */
    static std::string createPerRankFile(const std::string& directory,
                                         int rank,
                                         const std::string& filename = "ed_results.h5") {
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
    
    /**
     * @brief Merge TPQ data from per-rank HDF5 files into unified output
     * 
     * This is called by rank 0 after all MPI ranks complete their work.
     * It reads TPQ samples from each rank's file and writes them to the
     * final unified HDF5 file.
     * 
     * @param directory Output directory containing per-rank files
     * @param num_ranks Total number of MPI ranks
     * @param output_filename Name of final output file (default: ed_results.h5)
     * @param delete_temp_files Whether to delete per-rank files after merging
     * @return true if successful
     */
    static bool mergePerRankTPQFiles(const std::string& directory,
                                     int num_ranks,
                                     const std::string& output_filename = "ed_results.h5",
                                     bool delete_temp_files = true) {
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
    
    /**
     * @brief Copy all TPQ samples from source file to destination file
     * 
     * For continue_quenching support, this function properly merges:
     * - thermodynamics: Appends new rows (by step number) to existing data
     * - norm: Appends new rows (by step number) to existing data
     * - states: Copies new states (by beta value) that don't already exist
     * 
     * @param source_path Path to source HDF5 file
     * @param dest_path Path to destination HDF5 file
     * @return Number of samples copied/merged
     */
    static int copyTPQSamples(const std::string& source_path, const std::string& dest_path) {
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
                    // Find max step in existing data
                    uint64_t max_existing_step = 0;
                    for (const auto& pt : existing_thermo) {
                        if (pt.step > max_existing_step) {
                            max_existing_step = pt.step;
                        }
                    }
                    
                    // Append only new data points (step > max_existing_step)
                    int appended = 0;
                    for (const auto& pt : new_thermo) {
                        if (pt.step > max_existing_step) {
                            appendTPQThermodynamics(dest_path, sample_idx, pt);
                            appended++;
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
                    // Find max step in existing data
                    uint64_t max_existing_step = 0;
                    for (const auto& pt : existing_norm) {
                        if (pt.step > max_existing_step) {
                            max_existing_step = pt.step;
                        }
                    }
                    
                    // Append only new data points
                    int appended = 0;
                    for (const auto& pt : new_norm) {
                        if (pt.step > max_existing_step) {
                            appendTPQNorm(dest_path, sample_idx, pt);
                            appended++;
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
};

#endif // HDF5_IO_H
