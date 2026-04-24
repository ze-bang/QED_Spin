// =============================================================================
// src/bfg/wavefunction_io.cpp
//
// HDF5 wavefunction loaders for the BFG order-parameter pipeline. See
// include/ed/bfg/wavefunction_io.h for context. Audit ref: P2.1 (third slice).
// =============================================================================

#include "ed/bfg/wavefunction_io.h"

#include <algorithm>
#include <iostream>
#include <stdexcept>

#include <H5Cpp.h>

namespace ed::bfg {

namespace {

using Complex = std::complex<double>;

/**
 * Try to open one of several candidate dataset paths in `file`. Returns the
 * first that succeeds together with the path that matched; throws if none
 * of the candidates exist.
 */
H5::DataSet open_first_existing(H5::H5File& file,
                                const std::vector<std::string>& candidates,
                                std::string& matched_path) {
    for (const auto& path : candidates) {
        try {
            H5::DataSet ds = file.openDataSet(path);
            matched_path = path;
            return ds;
        } catch (...) {
            continue;
        }
    }
    throw std::runtime_error("Wavefunction dataset not found");
}

/**
 * Probe the dataset's actual on-disk compound member names so we read with
 * matching field labels. Different writers use different conventions:
 *
 *   * The legacy ED CPU/GPU writer uses ("real", "imag").
 *   * h5py's complex128 default uses ("r", "i").
 *
 * H5Cpp's `dataset.read(buf, requested_compound)` does NOT throw when the
 * requested member names don't exist on the source dataset -- it silently
 * fills the unmatched fields with zeros. Discovering the actual member
 * names first and then constructing a matching CompType is the only
 * robust way to round-trip both layouts.
 */
std::pair<std::string, std::string> probe_complex_field_names(
    H5::DataSet& dataset) {
    H5::DataType base = dataset.getDataType();
    if (base.getClass() != H5T_COMPOUND) {
        return {"real", "imag"};  // not compound; caller will fall back
    }
    H5::CompType compound(dataset);
    const int n_members = compound.getNmembers();
    std::string real_name = "real";
    std::string imag_name = "imag";
    bool have_real = false;
    bool have_imag = false;
    for (int idx = 0; idx < n_members; ++idx) {
        std::string member = compound.getMemberName(idx);
        if (member == "real" || member == "r" || member == "Re" ||
            member == "RE") {
            real_name = member;
            have_real = true;
        } else if (member == "imag" || member == "i" || member == "Im" ||
                   member == "IM") {
            imag_name = member;
            have_imag = true;
        }
    }
    if (!have_real || !have_imag) {
        // Fall back to the first two members in declaration order.
        if (n_members >= 2) {
            real_name = compound.getMemberName(0);
            imag_name = compound.getMemberName(1);
        }
    }
    return {real_name, imag_name};
}

/**
 * Read a dataset into `out` as a complex vector of length `total_size`.
 *
 * Two on-disk layouts are supported, mirroring the legacy CPU driver in
 * `compute_bfg_order_parameters.cpp` exactly:
 *
 *   * Compound-typed datasets (HDF5 compound, 16 bytes per element):
 *       try (real, imag), then (r, i) compound member names; fall back
 *       to interpreting the raw bytes as interleaved double pairs
 *       [re_0, im_0, re_1, im_1, ...] if neither matches.
 *   * Non-compound (8-byte double) datasets are read as **real-only**
 *       wavefunctions (imag = 0). This matches the behaviour the CPU
 *       driver had before the refactor for `is_complex == false`.
 *
 * `total_size` is the number of complex amplitudes -- equal to the
 * dataset's reported element count for compound layouts, and to the
 * dataset's element count for the real-only layout.
 */
void read_complex_dataset(H5::DataSet& dataset,
                          std::size_t total_size,
                          std::vector<Complex>& out) {
    out.assign(total_size, Complex{});

    H5::DataType base = dataset.getDataType();
    const bool is_compound = (base.getClass() == H5T_COMPOUND);
    if (is_compound) {
        const auto [real_name, imag_name] = probe_complex_field_names(dataset);
        try {
            H5::CompType compound(sizeof(Complex));
            compound.insertMember(real_name, 0, H5::PredType::NATIVE_DOUBLE);
            compound.insertMember(imag_name, sizeof(double),
                                  H5::PredType::NATIVE_DOUBLE);
            dataset.read(out.data(), compound);
            return;
        } catch (...) {
            // Fall through to the interleaved-double interpretation.
        }

        // Interleaved-double fallback. Used by some legacy writers that
        // emit compound types whose member names neither the (real, imag)
        // nor (r, i) probe matches.
        std::vector<double> buffer(total_size * 2);
        dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
        for (std::size_t i = 0; i < total_size; ++i) {
            out[i] = Complex(buffer[2 * i], buffer[2 * i + 1]);
        }
        return;
    }

    // Non-compound = real-only wavefunction. The CPU driver historically
    // treats these as Complex(re, 0); we preserve that contract here.
    std::vector<double> real_data(total_size);
    dataset.read(real_data.data(), H5::PredType::NATIVE_DOUBLE);
    for (std::size_t i = 0; i < total_size; ++i) {
        out[i] = Complex(real_data[i], 0.0);
    }
}

std::size_t dataspace_total_size(const H5::DataSpace& dataspace) {
    int rank = dataspace.getSimpleExtentNdims();
    std::vector<hsize_t> dims(rank);
    dataspace.getSimpleExtentDims(dims.data());
    std::size_t total = 1;
    for (int i = 0; i < rank; ++i) {
        total *= dims[i];
    }
    return total;
}

}  // namespace

std::vector<Complex> load_wavefunction(const std::string& filename,
                                       int eigenvector_idx,
                                       bool verbose) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        const std::string idx_str = std::to_string(eigenvector_idx);
        const std::vector<std::string> candidates = {
            "eigendata/eigenvector_" + idx_str,
            "eigenvector_" + idx_str,
            "eigendata/eigenvectors",
            "eigenvectors",
            "psi",
            "wavefunction",
            "ground_state",
        };

        std::string matched_path;
        H5::DataSet dataset = open_first_existing(file, candidates, matched_path);
        if (verbose) {
            std::cout << "Found wavefunction in dataset: " << matched_path << std::endl;
        }

        H5::DataSpace dataspace = dataset.getSpace();
        const std::size_t total_size = dataspace_total_size(dataspace);

        std::vector<Complex> psi;
        H5::DataType dtype = dataset.getDataType();
        const bool is_complex = (dtype.getSize() == 16);  // 2 doubles
        if (is_complex) {
            read_complex_dataset(dataset, total_size, psi);
        } else {
            std::vector<double> real_data(total_size);
            dataset.read(real_data.data(), H5::PredType::NATIVE_DOUBLE);
            psi.resize(total_size);
            for (std::size_t i = 0; i < total_size; ++i) {
                psi[i] = Complex(real_data[i], 0.0);
            }
        }

        if (verbose) {
            std::cout << "Loaded wavefunction: " << psi.size() << " amplitudes" << std::endl;
        }
        return psi;
    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 error reading " + filename + ": " +
                                 e.getCDetailMsg());
    }
}

std::vector<TPQState> load_all_tpq_states(const std::string& filename,
                                          int sample_idx,
                                          bool verbose) {
    try {
        H5::H5File file(filename, H5F_ACC_RDONLY);

        const std::string sample_path =
            "tpq/samples/sample_" + std::to_string(sample_idx) + "/states";

        H5::Group states_group;
        try {
            states_group = file.openGroup(sample_path);
        } catch (...) {
            throw std::runtime_error("TPQ states not found at: " + sample_path);
        }

        std::vector<std::pair<std::string, double>> beta_datasets;
        const hsize_t num_objs = states_group.getNumObjs();
        for (hsize_t i = 0; i < num_objs; ++i) {
            std::string name = states_group.getObjnameByIdx(i);
            if (name.substr(0, 5) == "beta_") {
                try {
                    double beta = std::stod(name.substr(5));
                    beta_datasets.emplace_back(name, beta);
                } catch (...) {
                    continue;
                }
            }
        }

        if (beta_datasets.empty()) {
            throw std::runtime_error("No beta_* states found in TPQ data");
        }

        std::sort(beta_datasets.begin(), beta_datasets.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });

        if (verbose) {
            std::cout << "Loading " << beta_datasets.size() << " TPQ states from sample "
                      << sample_idx << "..." << std::endl;
        }

        std::vector<TPQState> states;
        states.reserve(beta_datasets.size());
        for (const auto& [name, beta] : beta_datasets) {
            const std::string dataset_path = sample_path + "/" + name;
            H5::DataSet dataset = file.openDataSet(dataset_path);

            H5::DataSpace dataspace = dataset.getSpace();
            const std::size_t total_size = dataspace_total_size(dataspace);

            TPQState state;
            state.beta = beta;
            state.temperature = 1.0 / beta;
            read_complex_dataset(dataset, total_size, state.psi);
            states.push_back(std::move(state));
        }

        std::sort(states.begin(), states.end(),
                  [](const TPQState& a, const TPQState& b) {
                      return a.temperature < b.temperature;
                  });

        if (verbose) {
            std::cout << "Loaded " << states.size() << " TPQ states, T range: ["
                      << states.front().temperature << ", "
                      << states.back().temperature << "]" << std::endl;
        }
        return states;
    } catch (H5::Exception& e) {
        throw std::runtime_error("HDF5 error loading TPQ: " +
                                 std::string(e.getCDetailMsg()));
    }
}

std::pair<std::vector<Complex>, double> load_tpq_state(const std::string& filename,
                                                       int sample_idx,
                                                       bool verbose) {
    auto states = load_all_tpq_states(filename, sample_idx, verbose);
    if (states.empty()) {
        throw std::runtime_error("No TPQ states found in " + filename);
    }
    // load_all_tpq_states returns ascending temperature; lowest-T first.
    auto state = std::move(states.front());
    if (verbose) {
        std::cout << "Selected TPQ state: beta=" << state.beta
                  << ", T=" << state.temperature << std::endl;
    }
    return {std::move(state.psi), state.temperature};
}

}  // namespace ed::bfg
