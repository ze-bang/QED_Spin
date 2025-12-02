#ifndef HDF5_IO_H
#define HDF5_IO_H

#include <H5Cpp.h>
#include <vector>
#include <complex>
#include <string>
#include <iostream>
#include <stdexcept>
#include <iomanip>
#include <map>

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
            
            std::cout << "Saved " << eigenvalues.size() << " eigenvalues to HDF5" << std::endl;
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
