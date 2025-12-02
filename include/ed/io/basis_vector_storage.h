#ifndef BASIS_VECTOR_STORAGE_H
#define BASIS_VECTOR_STORAGE_H

#include <string>
#include <vector>
#include <complex>
#include <hdf5.h>

// Type alias for complex vectors
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// Class for managing HDF5-based storage of Lanczos basis vectors
class BasisVectorStorage {
public:
    // Constructor: opens or creates HDF5 file
    BasisVectorStorage(const std::string& filename, uint64_t dimension, bool read_only = false);
    
    // Destructor: closes HDF5 file
    ~BasisVectorStorage();
    
    // Write a basis vector to the file
    bool write_vector(uint64_t index, const ComplexVector& vec);
    
    // Read a basis vector from the file
    ComplexVector read_vector(uint64_t index);
    
    // Get the dimension of stored vectors
    uint64_t get_dimension() const { return dimension_; }
    
    // Check if a vector exists at given index
    bool vector_exists(uint64_t index);
    
    // Get number of stored vectors
    uint64_t get_num_vectors();
    
    // Close the file explicitly (also called in destructor)
    void close();

private:
    std::string filename_;
    uint64_t dimension_;
    hid_t file_id_;
    bool is_open_;
    bool read_only_;
    
    // Helper function to create dataset name
    std::string get_dataset_name(uint64_t index);
};

// Legacy interface functions for backward compatibility
// These wrap the HDF5 implementation
ComplexVector read_basis_vector_h5(const std::string& h5_file, uint64_t index, uint64_t N);
bool write_basis_vector_h5(const std::string& h5_file, uint64_t index, const ComplexVector& vec, uint64_t N);

#endif // BASIS_VECTOR_STORAGE_H
