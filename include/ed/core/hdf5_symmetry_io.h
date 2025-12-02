#ifndef HDF5_SYMMETRY_IO_H
#define HDF5_SYMMETRY_IO_H

#include <H5Cpp.h>
#include <vector>
#include <complex>
#include <string>
#include <iostream>
#include <stdexcept>
#include <Eigen/Sparse>

using Complex = std::complex<double>;

/**
 * @brief HDF5 I/O utilities for symmetry-adapted basis and block Hamiltonians
 * 
 * This class provides efficient file management for:
 * - Symmetrized basis vectors (sparse complex vectors)
 * - Block-diagonal Hamiltonian matrices (sparse complex matrices)
 * - Sector metadata (dimensions, quantum numbers)
 * 
 * File structure:
 *   symmetry_data.h5
 *   ├── /metadata
 *   │   ├── sector_dimensions [array of uint64_t]
 *   │   └── num_sectors [scalar uint64_t]
 *   ├── /basis
 *   │   ├── vector_0 [dataset: indices, real, imag]
 *   │   ├── vector_1
 *   │   └── ...
 *   └── /blocks
 *       ├── block_0 [dataset: rows, cols, real, imag, nnz]
 *       ├── block_1
 *       └── ...
 */
class HDF5SymmetryIO {
public:
    /**
     * @brief Create a new HDF5 file for symmetry data
     * @param directory Directory to create the file in
     * @return Full path to the created HDF5 file
     */
    static std::string createFile(const std::string& directory) {
        std::string filepath = directory + "/symmetry_data.h5";
        
        try {
            // Create file (overwrite if exists)
            H5::H5File file(filepath, H5F_ACC_TRUNC);
            
            // Create groups
            file.createGroup("/metadata");
            file.createGroup("/basis");
            file.createGroup("/blocks");
            
            file.close();
            std::cout << "Created HDF5 file: " << filepath << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to create HDF5 file: " + std::string(e.getCDetailMsg()));
        }
        
        return filepath;
    }
    
    /**
     * @brief Save sector dimensions (block sizes) to HDF5 file
     * @param filepath Path to HDF5 file
     * @param dimensions Vector of sector dimensions
     */
    static void saveSectorDimensions(const std::string& filepath, 
                                     const std::vector<uint64_t>& dimensions) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            // Save number of sectors
            hsize_t dims[1] = {1};
            H5::DataSpace scalar_space(1, dims);
            H5::DataSet num_sectors_ds = file.createDataSet("/metadata/num_sectors", 
                                                           H5::PredType::NATIVE_UINT64, 
                                                           scalar_space);
            uint64_t num_sectors = dimensions.size();
            num_sectors_ds.write(&num_sectors, H5::PredType::NATIVE_UINT64);
            num_sectors_ds.close();
            
            // Save sector dimensions array
            dims[0] = dimensions.size();
            H5::DataSpace dataspace(1, dims);
            H5::DataSet dataset = file.createDataSet("/metadata/sector_dimensions", 
                                                     H5::PredType::NATIVE_UINT64, 
                                                     dataspace);
            dataset.write(dimensions.data(), H5::PredType::NATIVE_UINT64);
            dataset.close();
            
            file.close();
            
            std::cout << "Saved " << dimensions.size() << " sector dimensions to HDF5" << std::endl;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save sector dimensions: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Load sector dimensions from HDF5 file
     * @param filepath Path to HDF5 file
     * @return Vector of sector dimensions
     */
    static std::vector<uint64_t> loadSectorDimensions(const std::string& filepath) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            // Read number of sectors
            H5::DataSet num_sectors_ds = file.openDataSet("/metadata/num_sectors");
            uint64_t num_sectors;
            num_sectors_ds.read(&num_sectors, H5::PredType::NATIVE_UINT64);
            num_sectors_ds.close();
            
            // Read sector dimensions
            H5::DataSet dataset = file.openDataSet("/metadata/sector_dimensions");
            H5::DataSpace dataspace = dataset.getSpace();
            
            std::vector<uint64_t> dimensions(num_sectors);
            dataset.read(dimensions.data(), H5::PredType::NATIVE_UINT64);
            dataset.close();
            file.close();
            
            return dimensions;
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to load sector dimensions: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save a sparse symmetrized basis vector to HDF5
     * @param filepath Path to HDF5 file
     * @param index Index of the basis vector
     * @param vec Complex vector (will store only non-zero elements)
     */
    static void saveBasisVector(const std::string& filepath, 
                                size_t index, 
                                const std::vector<Complex>& vec) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            // Collect non-zero elements
            std::vector<uint64_t> indices;
            std::vector<double> real_parts;
            std::vector<double> imag_parts;
            
            for (size_t i = 0; i < vec.size(); ++i) {
                if (std::abs(vec[i]) > 1e-12) {
                    indices.push_back(i);
                    real_parts.push_back(vec[i].real());
                    imag_parts.push_back(vec[i].imag());
                }
            }
            
            size_t nnz = indices.size();
            if (nnz == 0) {
                file.close();
                return; // Don't save empty vectors
            }
            
            // Create dataset name
            std::string dataset_name = "/basis/vector_" + std::to_string(index);
            
            // Create compound datatype for sparse vector
            H5::CompType sparse_vec_type(sizeof(uint64_t) + 2 * sizeof(double));
            sparse_vec_type.insertMember("index", 0, H5::PredType::NATIVE_UINT64);
            sparse_vec_type.insertMember("real", sizeof(uint64_t), H5::PredType::NATIVE_DOUBLE);
            sparse_vec_type.insertMember("imag", sizeof(uint64_t) + sizeof(double), H5::PredType::NATIVE_DOUBLE);
            
            // Create dataspace
            hsize_t dims[1] = {nnz};
            H5::DataSpace dataspace(1, dims);
            
            // Create dataset
            H5::DataSet dataset = file.createDataSet(dataset_name, sparse_vec_type, dataspace);
            
            // Write data
            struct SparseElement {
                uint64_t index;
                double real;
                double imag;
            };
            
            std::vector<SparseElement> data(nnz);
            for (size_t i = 0; i < nnz; ++i) {
                data[i] = {indices[i], real_parts[i], imag_parts[i]};
            }
            
            dataset.write(data.data(), sparse_vec_type);
            
            // Store vector dimension as attribute
            H5::DataSpace attr_space(H5S_SCALAR);
            H5::Attribute attr = dataset.createAttribute("dimension", H5::PredType::NATIVE_UINT64, attr_space);
            uint64_t dim = vec.size();
            attr.write(H5::PredType::NATIVE_UINT64, &dim);
            attr.close();
            
            dataset.close();
            file.close();
            
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save basis vector: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Load a sparse symmetrized basis vector from HDF5
     * @param filepath Path to HDF5 file
     * @param index Index of the basis vector
     * @param dimension Full dimension of the vector (2^n_bits)
     * @return Complex vector
     */
    static std::vector<Complex> loadBasisVector(const std::string& filepath, 
                                                size_t index,
                                                size_t dimension) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            std::string dataset_name = "/basis/vector_" + std::to_string(index);
            
            // Initialize result vector with zeros
            std::vector<Complex> vec(dimension, Complex(0.0, 0.0));
            
            // Check if dataset exists
            if (!file.nameExists(dataset_name)) {
                file.close();
                return vec; // Return zero vector if not found
            }
            
            H5::DataSet dataset = file.openDataSet(dataset_name);
            
            // Get stored dimension from attribute
            H5::Attribute attr = dataset.openAttribute("dimension");
            uint64_t stored_dim;
            attr.read(H5::PredType::NATIVE_UINT64, &stored_dim);
            attr.close();
            
            if (stored_dim != dimension) {
                throw std::runtime_error("Dimension mismatch: expected " + std::to_string(dimension) 
                                       + ", got " + std::to_string(stored_dim));
            }
            
            // Get dataspace and number of elements
            H5::DataSpace dataspace = dataset.getSpace();
            hsize_t dims[1];
            dataspace.getSimpleExtentDims(dims);
            size_t nnz = dims[0];
            
            // Define compound type
            H5::CompType sparse_vec_type(sizeof(uint64_t) + 2 * sizeof(double));
            sparse_vec_type.insertMember("index", 0, H5::PredType::NATIVE_UINT64);
            sparse_vec_type.insertMember("real", sizeof(uint64_t), H5::PredType::NATIVE_DOUBLE);
            sparse_vec_type.insertMember("imag", sizeof(uint64_t) + sizeof(double), H5::PredType::NATIVE_DOUBLE);
            
            // Read data
            struct SparseElement {
                uint64_t index;
                double real;
                double imag;
            };
            
            std::vector<SparseElement> data(nnz);
            dataset.read(data.data(), sparse_vec_type);
            
            // Fill result vector
            for (size_t i = 0; i < nnz; ++i) {
                vec[data[i].index] = Complex(data[i].real, data[i].imag);
            }
            
            dataset.close();
            file.close();
            
            return vec;
            
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to load basis vector: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Save a sparse block matrix to HDF5
     * @param filepath Path to HDF5 file
     * @param block_idx Index of the block
     * @param matrix Sparse matrix to save
     */
    static void saveBlockMatrix(const std::string& filepath, 
                               size_t block_idx, 
                               const Eigen::SparseMatrix<Complex>& matrix) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDWR);
            
            std::string dataset_name = "/blocks/block_" + std::to_string(block_idx);
            
            // Extract triplets from sparse matrix
            std::vector<int> rows;
            std::vector<int> cols;
            std::vector<double> real_parts;
            std::vector<double> imag_parts;
            
            for (int k = 0; k < matrix.outerSize(); ++k) {
                for (Eigen::SparseMatrix<Complex>::InnerIterator it(matrix, k); it; ++it) {
                    rows.push_back(it.row());
                    cols.push_back(it.col());
                    real_parts.push_back(it.value().real());
                    imag_parts.push_back(it.value().imag());
                }
            }
            
            size_t nnz = rows.size();
            
            // Create compound datatype for sparse matrix
            H5::CompType sparse_mat_type(2 * sizeof(int) + 2 * sizeof(double));
            sparse_mat_type.insertMember("row", 0, H5::PredType::NATIVE_INT);
            sparse_mat_type.insertMember("col", sizeof(int), H5::PredType::NATIVE_INT);
            sparse_mat_type.insertMember("real", 2 * sizeof(int), H5::PredType::NATIVE_DOUBLE);
            sparse_mat_type.insertMember("imag", 2 * sizeof(int) + sizeof(double), H5::PredType::NATIVE_DOUBLE);
            
            // Create dataspace
            hsize_t dims[1] = {nnz};
            H5::DataSpace dataspace(1, dims);
            
            // Create dataset
            H5::DataSet dataset = file.createDataSet(dataset_name, sparse_mat_type, dataspace);
            
            // Write data
            struct SparseMatElement {
                int row;
                int col;
                double real;
                double imag;
            };
            
            std::vector<SparseMatElement> data(nnz);
            for (size_t i = 0; i < nnz; ++i) {
                data[i] = {rows[i], cols[i], real_parts[i], imag_parts[i]};
            }
            
            dataset.write(data.data(), sparse_mat_type);
            
            // Store matrix dimensions as attributes
            H5::DataSpace attr_space(H5S_SCALAR);
            
            H5::Attribute rows_attr = dataset.createAttribute("rows", H5::PredType::NATIVE_UINT64, attr_space);
            uint64_t num_rows = matrix.rows();
            rows_attr.write(H5::PredType::NATIVE_UINT64, &num_rows);
            rows_attr.close();
            
            H5::Attribute cols_attr = dataset.createAttribute("cols", H5::PredType::NATIVE_UINT64, attr_space);
            uint64_t num_cols = matrix.cols();
            cols_attr.write(H5::PredType::NATIVE_UINT64, &num_cols);
            cols_attr.close();
            
            H5::Attribute nnz_attr = dataset.createAttribute("nnz", H5::PredType::NATIVE_UINT64, attr_space);
            nnz_attr.write(H5::PredType::NATIVE_UINT64, &nnz);
            nnz_attr.close();
            
            dataset.close();
            file.close();
            
            std::cout << " done (saved to HDF5: " << nnz << " nnz, "
                      << std::fixed << std::setprecision(2)
                      << (100.0 * nnz / (matrix.rows() * matrix.cols())) << "% fill)" << std::endl;
            
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to save block matrix: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Load a sparse block matrix from HDF5
     * @param filepath Path to HDF5 file
     * @param block_idx Index of the block
     * @return Sparse complex matrix
     */
    static Eigen::SparseMatrix<Complex> loadBlockMatrix(const std::string& filepath, 
                                                        size_t block_idx) {
        try {
            H5::H5File file(filepath, H5F_ACC_RDONLY);
            
            std::string dataset_name = "/blocks/block_" + std::to_string(block_idx);
            
            H5::DataSet dataset = file.openDataSet(dataset_name);
            
            // Read dimensions from attributes
            H5::Attribute rows_attr = dataset.openAttribute("rows");
            H5::Attribute cols_attr = dataset.openAttribute("cols");
            H5::Attribute nnz_attr = dataset.openAttribute("nnz");
            
            uint64_t rows, cols, nnz;
            rows_attr.read(H5::PredType::NATIVE_UINT64, &rows);
            cols_attr.read(H5::PredType::NATIVE_UINT64, &cols);
            nnz_attr.read(H5::PredType::NATIVE_UINT64, &nnz);
            
            rows_attr.close();
            cols_attr.close();
            nnz_attr.close();
            
            // Get dataspace
            H5::DataSpace dataspace = dataset.getSpace();
            
            // Define compound type
            H5::CompType sparse_mat_type(2 * sizeof(int) + 2 * sizeof(double));
            sparse_mat_type.insertMember("row", 0, H5::PredType::NATIVE_INT);
            sparse_mat_type.insertMember("col", sizeof(int), H5::PredType::NATIVE_INT);
            sparse_mat_type.insertMember("real", 2 * sizeof(int), H5::PredType::NATIVE_DOUBLE);
            sparse_mat_type.insertMember("imag", 2 * sizeof(int) + sizeof(double), H5::PredType::NATIVE_DOUBLE);
            
            // Read data
            struct SparseMatElement {
                int row;
                int col;
                double real;
                double imag;
            };
            
            std::vector<SparseMatElement> data(nnz);
            dataset.read(data.data(), sparse_mat_type);
            
            // Build triplet list
            std::vector<Eigen::Triplet<Complex>> triplets;
            triplets.reserve(nnz);
            
            for (size_t i = 0; i < nnz; ++i) {
                triplets.emplace_back(data[i].row, data[i].col, 
                                     Complex(data[i].real, data[i].imag));
            }
            
            // Create sparse matrix
            Eigen::SparseMatrix<Complex> matrix(rows, cols);
            matrix.setFromTriplets(triplets.begin(), triplets.end());
            matrix.makeCompressed();
            
            dataset.close();
            file.close();
            
            return matrix;
            
        } catch (H5::Exception& e) {
            throw std::runtime_error("Failed to load block matrix: " + std::string(e.getCDetailMsg()));
        }
    }
    
    /**
     * @brief Check if HDF5 file exists and is valid
     * @param filepath Path to HDF5 file
     * @return true if file exists and can be opened
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
};

#endif // HDF5_SYMMETRY_IO_H
