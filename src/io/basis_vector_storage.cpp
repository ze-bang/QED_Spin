#include <ed/io/basis_vector_storage.h>
#include <iostream>
#include <stdexcept>
#include <cstring>

BasisVectorStorage::BasisVectorStorage(const std::string& filename, uint64_t dimension, bool read_only)
    : filename_(filename), dimension_(dimension), file_id_(-1), is_open_(false), read_only_(read_only) {
    
    // Turn off HDF5 error messages to handle errors programmatically
    H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
    
    if (read_only) {
        // Open existing file for reading
        file_id_ = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id_ < 0) {
            throw std::runtime_error("Failed to open HDF5 file for reading: " + filename);
        }
    } else {
        // Try to open existing file for read/write
        file_id_ = H5Fopen(filename.c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
        
        // If file doesn't exist, create it
        if (file_id_ < 0) {
            file_id_ = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            if (file_id_ < 0) {
                throw std::runtime_error("Failed to create HDF5 file: " + filename);
            }
            
            // Store dimension as attribute in root group
            hid_t root_group = H5Gopen(file_id_, "/", H5P_DEFAULT);
            hid_t attr_space = H5Screate(H5S_SCALAR);
            hid_t attr = H5Acreate(root_group, "dimension", H5T_NATIVE_UINT64, attr_space, 
                                  H5P_DEFAULT, H5P_DEFAULT);
            H5Awrite(attr, H5T_NATIVE_UINT64, &dimension_);
            H5Aclose(attr);
            H5Sclose(attr_space);
            H5Gclose(root_group);
        } else {
            // Read dimension from existing file
            hid_t root_group = H5Gopen(file_id_, "/", H5P_DEFAULT);
            if (H5Aexists(root_group, "dimension")) {
                hid_t attr = H5Aopen(root_group, "dimension", H5P_DEFAULT);
                uint64_t stored_dim;
                H5Aread(attr, H5T_NATIVE_UINT64, &stored_dim);
                H5Aclose(attr);
                
                if (stored_dim != dimension) {
                    std::cerr << "Warning: Stored dimension (" << stored_dim 
                             << ") differs from requested dimension (" << dimension << ")\n";
                    dimension_ = stored_dim;
                }
            }
            H5Gclose(root_group);
        }
    }
    
    is_open_ = true;
}

BasisVectorStorage::~BasisVectorStorage() {
    close();
}

void BasisVectorStorage::close() {
    if (is_open_ && file_id_ >= 0) {
        H5Fclose(file_id_);
        file_id_ = -1;
        is_open_ = false;
    }
}

std::string BasisVectorStorage::get_dataset_name(uint64_t index) {
    return "/basis_" + std::to_string(index);
}

bool BasisVectorStorage::write_vector(uint64_t index, const ComplexVector& vec) {
    if (!is_open_ || read_only_) {
        std::cerr << "Error: File not open for writing\n";
        return false;
    }
    
    if (vec.size() != dimension_) {
        std::cerr << "Error: Vector size (" << vec.size() 
                 << ") does not match expected dimension (" << dimension_ << ")\n";
        return false;
    }
    
    std::string dataset_name = get_dataset_name(index);
    
    // Check if dataset already exists and delete it
    if (H5Lexists(file_id_, dataset_name.c_str(), H5P_DEFAULT) > 0) {
        H5Ldelete(file_id_, dataset_name.c_str(), H5P_DEFAULT);
    }
    
    // Create dataspace for complex vector
    hsize_t dims[2] = {dimension_, 2}; // [real, imag]
    hid_t dataspace = H5Screate_simple(2, dims, nullptr);
    
    // Create dataset
    hid_t dataset = H5Dcreate(file_id_, dataset_name.c_str(), H5T_NATIVE_DOUBLE, 
                             dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    
    if (dataset < 0) {
        std::cerr << "Error: Failed to create dataset " << dataset_name << "\n";
        H5Sclose(dataspace);
        return false;
    }
    
    // Convert complex vector to interleaved real/imag format
    std::vector<double> data(2 * dimension_);
    for (uint64_t i = 0; i < dimension_; ++i) {
        data[2*i] = vec[i].real();
        data[2*i + 1] = vec[i].imag();
    }
    
    // Write data
    herr_t status = H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                            H5P_DEFAULT, data.data());
    
    H5Dclose(dataset);
    H5Sclose(dataspace);
    
    // Flush to disk to ensure data is written
    H5Fflush(file_id_, H5F_SCOPE_LOCAL);
    
    return (status >= 0);
}

ComplexVector BasisVectorStorage::read_vector(uint64_t index) {
    if (!is_open_) {
        throw std::runtime_error("Error: File not open for reading");
    }
    
    std::string dataset_name = get_dataset_name(index);
    
    // Check if dataset exists
    if (H5Lexists(file_id_, dataset_name.c_str(), H5P_DEFAULT) <= 0) {
        throw std::runtime_error("Error: Dataset " + dataset_name + " does not exist");
    }
    
    // Open dataset
    hid_t dataset = H5Dopen(file_id_, dataset_name.c_str(), H5P_DEFAULT);
    if (dataset < 0) {
        throw std::runtime_error("Error: Failed to open dataset " + dataset_name);
    }
    
    // Read data
    std::vector<double> data(2 * dimension_);
    herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, 
                           H5P_DEFAULT, data.data());
    
    H5Dclose(dataset);
    
    if (status < 0) {
        throw std::runtime_error("Error: Failed to read dataset " + dataset_name);
    }
    
    // Convert interleaved real/imag format to complex vector
    ComplexVector vec(dimension_);
    for (uint64_t i = 0; i < dimension_; ++i) {
        vec[i] = Complex(data[2*i], data[2*i + 1]);
    }
    
    return vec;
}

bool BasisVectorStorage::vector_exists(uint64_t index) {
    if (!is_open_) {
        return false;
    }
    
    std::string dataset_name = get_dataset_name(index);
    return (H5Lexists(file_id_, dataset_name.c_str(), H5P_DEFAULT) > 0);
}

uint64_t BasisVectorStorage::get_num_vectors() {
    if (!is_open_) {
        return 0;
    }
    
    // Count datasets in root group that start with "basis_"
    uint64_t count = 0;
    hid_t root_group = H5Gopen(file_id_, "/", H5P_DEFAULT);
    
    H5G_info_t group_info;
    H5Gget_info(root_group, &group_info);
    
    for (hsize_t i = 0; i < group_info.nlinks; ++i) {
        char name[256];
        H5Lget_name_by_idx(root_group, ".", H5_INDEX_NAME, H5_ITER_NATIVE, 
                          i, name, 256, H5P_DEFAULT);
        
        if (strncmp(name, "basis_", 6) == 0) {
            count++;
        }
    }
    
    H5Gclose(root_group);
    return count;
}

// Legacy interface functions for backward compatibility

ComplexVector read_basis_vector_h5(const std::string& h5_file, uint64_t index, uint64_t N) {
    try {
        BasisVectorStorage storage(h5_file, N, true);
        return storage.read_vector(index);
    } catch (const std::exception& e) {
        std::cerr << "Error reading basis vector: " << e.what() << "\n";
        throw;
    }
}

bool write_basis_vector_h5(const std::string& h5_file, uint64_t index, const ComplexVector& vec, uint64_t N) {
    try {
        BasisVectorStorage storage(h5_file, N, false);
        return storage.write_vector(index, vec);
    } catch (const std::exception& e) {
        std::cerr << "Error writing basis vector: " << e.what() << "\n";
        return false;
    }
}
