// =============================================================================
// src/io/hdf5_io_eigen.cpp
//
// HDF5IO: eigenvalue / eigenvector I/O plus the thermodynamic-observable and
// correlation-matrix datasets. Declarations live in
// include/ed/core/hdf5_io.h.
// =============================================================================

#include <ed/core/hdf5_io.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <system_error>

void HDF5IO::saveEigenvalues(const std::string& filepath, 
                             const std::vector<double>& eigenvalues) {
    if (isDisabledOutputPath(filepath)) return;
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

std::vector<double> HDF5IO::loadEigenvalues(const std::string& filepath) {
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

void HDF5IO::saveEigenvector(const std::string& filepath, 
                             size_t index,
                             const std::vector<Complex>& eigenvector) {
    if (isDisabledOutputPath(filepath)) return;
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

void HDF5IO::saveDiagonalizationResults(
    const std::string& output_dir,
    const std::vector<double>& eigenvalues,
    const std::vector<std::vector<Complex>>& eigenvectors,
    const std::string& solver_name
) {
    if (isDisabledOutputPath(output_dir)) return;

    // Create output directory if needed (for .dat files and HDF5)
    std::error_code ec;
    std::filesystem::create_directories(output_dir, ec);
    if (ec) {
        std::cerr << "Warning: Could not create directory " << output_dir
                  << ": " << ec.message() << std::endl;
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

std::vector<Complex> HDF5IO::loadEigenvector(const std::string& filepath, size_t index) {
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

void HDF5IO::saveThermodynamics(const std::string& filepath,
                                const std::vector<double>& temperatures,
                                const std::string& observable_name,
                                const std::vector<double>& values) {
    if (isDisabledOutputPath(filepath)) return;
    if (temperatures.size() != values.size()) {
        throw std::invalid_argument(
            "saveThermodynamics: temperatures.size()=" +
            std::to_string(temperatures.size()) +
            " disagrees with values.size()=" +
            std::to_string(values.size()) +
            " for observable '" + observable_name + "'");
    }
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

std::vector<double> HDF5IO::loadThermodynamicObservable(const std::string& filepath,
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

void HDF5IO::saveCorrelationMatrix(const std::string& filepath,
                                   const std::string& correlation_name,
                                   const std::vector<std::vector<Complex>>& matrix) {
    if (isDisabledOutputPath(filepath)) return;
    try {
        H5::H5File file(filepath, H5F_ACC_RDWR);

        std::string dataset_name = "/correlations/" + correlation_name;
        if (file.nameExists(dataset_name)) {
            file.unlink(dataset_name);
        }

        if (matrix.empty()) {
            throw std::invalid_argument(
                "saveCorrelationMatrix: refusing to save an empty matrix for '" +
                correlation_name + "'");
        }
        size_t n_rows = matrix.size();
        size_t n_cols = matrix[0].size();

        // Flatten matrix
        std::vector<double> real_part(n_rows * n_cols);
        std::vector<double> imag_part(n_rows * n_cols);

        for (size_t i = 0; i < n_rows; ++i) {
            if (matrix[i].size() != n_cols) {
                throw std::invalid_argument(
                    "saveCorrelationMatrix: jagged matrix for '" +
                    correlation_name + "' (row " + std::to_string(i) +
                    " has size " + std::to_string(matrix[i].size()) +
                    " != n_cols=" + std::to_string(n_cols) + ")");
            }
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
