#include <ed/solvers/TPQ.h>
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/parallel/thread_budget.h>  // Phase 6.1: dim-aware OMP+BLAS cap
#include <algorithm>
#include <cctype>
#include <filesystem>
#include <regex>
#include <sstream>
#include <map>

#ifdef WITH_MPI
#include <mpi.h>
#endif

// Note: Time evolution and dynamics computation methods have been moved to dynamics.cpp
// This file now focuses on TPQ-specific functionality and wraps the general dynamics module

// Forward declarations
bool save_tpq_state_hdf5(const ComplexVector& tpq_state, const std::string& dir,
                         size_t sample, double beta, FixedSzOperator* fixed_sz_op);

// ============================================================================
// Legacy TPQ-SPECIFIC WRAPPER FUNCTIONS retired in the minimalist-
// architecture rev (May 2026). Every wrapper here was a one-line
// thunk into the ``dynamics`` module (time_evolve_taylor /
// time_evolve_krylov / time_evolve_chebyshev / time_evolve_rk4 /
// time_evolve_adaptive); none had external callers. Call the dynamics
// module directly.
// ============================================================================


// ============================================================================
// TPQ-SPECIFIC UTILITY FUNCTIONS
// ============================================================================

/**
 * Generate a random normalized vector for TPQ initial state
 * 
 * @param N Dimension of the Hilbert space
 * @param seed Random seed to use
 * @return Random normalized vector
 */
ComplexVector generateTPQVector(int N, uint64_t seed) {
    // i.i.d. complex Gaussian components: real and imag parts ~ N(0, 1).
    // After L2 normalisation this is the canonical isotropic sample on the
    // complex unit sphere -- the standard TPQ initial-state distribution
    // (Sugiura-Shimizu 2012/2013, Hams-De Raedt 2000).
    //
    // Note: gpu_tpq.cu (`fill_random_vector_kernel`) and gpu_lanczos.cu both
    // already use curand_normal_double, so this brings the CPU path in line
    // with the GPU path.
    std::mt19937 gen(seed);
    std::normal_distribution<double> ndist(0.0, 1.0);

    ComplexVector v(N);
    for (int i = 0; i < N; i++) {
        v[i] = Complex(ndist(gen), ndist(gen));
    }

    double norm = cblas_dznrm2(N, v.data(), 1);
    Complex scale_factor = Complex(1.0 / norm, 0.0);
    cblas_zscal(N, &scale_factor, v.data(), 1);

    return v;  // NRVO; std::move on return value would inhibit it.
}

/**
 * Create directory if it doesn't exist (P0.12: replaced shell mkdir).
 */
bool ensureDirectoryExists(const std::string& path) {
    std::error_code ec;
    if (std::filesystem::is_directory(path, ec)) {
        return true;
    }
    if (std::filesystem::exists(path, ec)) {
        // Path exists but is not a directory.
        return false;
    }
    std::filesystem::create_directories(path, ec);
    return !ec || std::filesystem::is_directory(path);
}

/**
 * Calculate energy and variance for a TPQ state
 * 
 * @param H Hamiltonian operator function
 * @param v Current TPQ state vector
 * @param N Dimension of the Hilbert space
 * @return Pair of energy and variance
 */
std::pair<double, double> calculateEnergyAndVariance(
    std::function<void(const Complex*, Complex*, int)> H,
    const ComplexVector& v,
    uint64_t N
) {
    // E  = <v | H v>,  Var = <v | H^2 v> - E^2
    // The two scalar reductions are now BLAS-1 zdotc calls — vectorised,
    // deterministic for fixed N, and avoid allocating Complex temporaries
    // in a hot inner loop. For large N this drops the call cost from
    // ~6N flops at scalar speed to ~6N at peak SIMD throughput.
    ComplexVector Hv(N);
    H(v.data(), Hv.data(), N);
    Complex energy_complex;
    cblas_zdotc_sub(N, v.data(), 1, Hv.data(), 1, &energy_complex);
    double energy = energy_complex.real();

    ComplexVector H2v(N);
    H(Hv.data(), H2v.data(), N);
    Complex h2_complex;
    cblas_zdotc_sub(N, v.data(), 1, H2v.data(), 1, &h2_complex);
    double variance = h2_complex.real() - energy * energy;

    return {energy, variance};
}

std::vector<SingleSiteOperator> createSzOperators(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Sz_ops;
    for (int site = 0; site < num_sites; site++) {
        Sz_ops.emplace_back(num_sites, spin_length, 2, site);
    }
    return Sz_ops;
}

std::vector<SingleSiteOperator> createSxOperators(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Sx_ops;
    for (int site = 0; site < num_sites; site++) {
        Sx_ops.emplace_back(num_sites, spin_length, 3, site);
    }
    return Sx_ops;
}

std::vector<SingleSiteOperator> createSyOperators(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Sy_ops;
    for (int site = 0; site < num_sites; site++) {
        Sy_ops.emplace_back(num_sites, spin_length, 4, site);
    }
    return Sy_ops;
}

std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzandSz2(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sz_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit to avoid overflow)
    
    ComplexVector Sz_exps(sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Sz2_exps(sublattice_size+1, Complex(0.0, 0.0));
    
    // OPTIMIZED: Pre-allocate reusable buffers outside loop
    std::vector<Complex> Sz_psi(N);
    std::vector<Complex> Sz2_psi(N);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        uint64_t i = site % sublattice_size;

        // Apply operator into pre-allocated buffer
        Sz_ops[i].apply(tpq_state.data(), Sz_psi.data(), N);
        
        // Calculate expectation value using BLAS
        Complex Sz_exp;
        cblas_zdotc_sub(N, tpq_state.data(), 1, Sz_psi.data(), 1, &Sz_exp);
        Sz_exps[i] += Sz_exp;

        // Apply operator again for Sz^2
        Sz_ops[i].apply(Sz_psi.data(), Sz2_psi.data(), N);
        
        // Calculate Sz^2 expectation using BLAS
        Complex Sz2_exp;
        cblas_zdotc_sub(N, tpq_state.data(), 1, Sz2_psi.data(), 1, &Sz2_exp);
        Sz2_exps[i] += Sz2_exp;
    }

    for (int i = 0; i < sublattice_size; i++) {
        Sz_exps[i] /= double(num_sites);
        Sz2_exps[i] /= double(num_sites);
        Sz_exps[sublattice_size] += Sz_exps[i];
        Sz2_exps[sublattice_size] += Sz2_exps[i];
    }

    return {Sz_exps, Sz2_exps};
}


Complex calculateSpm_onsite(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Spm_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)

    Complex Spm_exp(0.0, 0.0);
    
    // OPTIMIZED: Pre-allocate reusable buffer
    std::vector<Complex> Spm_psi(N);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        uint64_t i = site % sublattice_size;

        // Apply operator into pre-allocated buffer
        Spm_ops[i].apply(tpq_state.data(), Spm_psi.data(), N);

        // Calculate <Spm_psi|Spm_psi> using BLAS
        Complex site_exp;
        cblas_zdotc_sub(N, Spm_psi.data(), 1, Spm_psi.data(), 1, &site_exp);
        Spm_exp += site_exp;
    }

    return Spm_exp / double(num_sites);
}


std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> createDoubleSiteOperators(int num_sites, float spin_length) {
    std::vector<DoubleSiteOperator> Szz_ops;
    std::vector<DoubleSiteOperator> Spm_ops;

    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            Szz_ops.emplace_back(num_sites, spin_length, 2, site, 2, site2);
            Spm_ops.emplace_back(num_sites, spin_length, 0, site, 1, site2);
        }
    }
    return {Szz_ops, Spm_ops};
}


std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> createSingleOperators_pair(int num_sites, float spin_length) {
    std::vector<SingleSiteOperator> Szz_ops;
    std::vector<SingleSiteOperator> Spm_ops;

    for (int site = 0; site < num_sites; site++) {
        Szz_ops.emplace_back(num_sites, spin_length, 2, site);
        Spm_ops.emplace_back(num_sites, spin_length, 0, site);
    }
    return {Szz_ops, Spm_ops};
}



std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    std::pair<std::vector<DoubleSiteOperator>, std::vector<DoubleSiteOperator>> double_site_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));

    // Reference operators (avoid copy)
    const std::vector<DoubleSiteOperator>& Szz_ops = double_site_ops.first;
    const std::vector<DoubleSiteOperator>& Spm_ops = double_site_ops.second;
    
    // OPTIMIZED: Pre-allocate reusable buffers outside nested loop
    std::vector<Complex> Szz_psi(N);
    std::vector<Complex> Spm_psi(N);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            uint64_t n1 = site % sublattice_size;
            uint64_t n2 = site2 % sublattice_size;

            // Apply operators into pre-allocated buffers
            Szz_ops[site*num_sites+site2].apply(tpq_state.data(), Szz_psi.data(), N);
            Spm_ops[site*num_sites+site2].apply(tpq_state.data(), Spm_psi.data(), N);

            // Calculate expectation values using BLAS
            Complex Szz_exp, Spm_exp;
            cblas_zdotc_sub(N, tpq_state.data(), 1, Szz_psi.data(), 1, &Szz_exp);
            cblas_zdotc_sub(N, tpq_state.data(), 1, Spm_psi.data(), 1, &Spm_exp);
            
            Spm_exps[n1*sublattice_size+n2] += Spm_exp;
            Szz_exps[n1*sublattice_size+n2] += Szz_exp;
        }
    }

    for (int i = 0; i < sublattice_size*sublattice_size; i++) {
        Spm_exps[i] /= double(num_sites);
        Szz_exps[i] /= double(num_sites);
        Spm_exps[sublattice_size*sublattice_size] += Spm_exps[i];
        Szz_exps[sublattice_size*sublattice_size] += Szz_exps[i];
    }
    
    return {Szz_exps, Spm_exps};

}

std::tuple<std::vector<Complex>, std::vector<Complex>, std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>> double_site_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spp_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    
    // Reference operators (avoid copy)
    const std::vector<SingleSiteOperator>& Szz_ops = double_site_ops.first;
    const std::vector<SingleSiteOperator>& Spm_ops = double_site_ops.second;
    
    // OPTIMIZED: Pre-allocate reusable buffers outside nested loop
    std::vector<Complex> Szz_psi(N);
    std::vector<Complex> Szz_psi2(N);
    std::vector<Complex> Spm_psi(N);
    std::vector<Complex> Spm_psi2(N);
    std::vector<Complex> Spp_psi(N);
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        for (int site2 = 0; site2 < num_sites; site2++) {
            uint64_t n1 = site % sublattice_size;
            uint64_t n2 = site2 % sublattice_size;

            // Apply operators into pre-allocated buffers
            Szz_ops[site].apply(tpq_state.data(), Szz_psi.data(), N);
            Szz_ops[site2].apply(tpq_state.data(), Szz_psi2.data(), N);
            Spm_ops[site].apply(tpq_state.data(), Spm_psi.data(), N);
            Spm_ops[site2].apply(tpq_state.data(), Spm_psi2.data(), N);
            Spm_ops[site2].apply(Spm_psi.data(), Spp_psi.data(), N);

            // Calculate expectation values using BLAS
            Complex Szz_exp, Spm_exp, Spp_exp, Spz_exp;
            cblas_zdotc_sub(N, Szz_psi.data(), 1, Szz_psi2.data(), 1, &Szz_exp);
            cblas_zdotc_sub(N, Spm_psi.data(), 1, Spm_psi2.data(), 1, &Spm_exp);
            cblas_zdotc_sub(N, tpq_state.data(), 1, Spp_psi.data(), 1, &Spp_exp);
            cblas_zdotc_sub(N, Spm_psi.data(), 1, Szz_psi2.data(), 1, &Spz_exp);
            
            Spm_exps[n1*sublattice_size+n2] += Spm_exp;
            Szz_exps[n1*sublattice_size+n2] += Szz_exp;
            Spp_exps[n1*sublattice_size+n2] += Spp_exp;
            Spz_exps[n1*sublattice_size+n2] += Spz_exp;
        }
    }

    for (int i = 0; i < sublattice_size*sublattice_size; i++) {
        Spm_exps[i] /= double(num_sites);
        Szz_exps[i] /= double(num_sites);
        Spp_exps[i] /= double(num_sites);
        Spz_exps[i] /= double(num_sites);
        Spm_exps[sublattice_size*sublattice_size] += Spm_exps[i];
        Szz_exps[sublattice_size*sublattice_size] += Szz_exps[i];
        Spp_exps[sublattice_size*sublattice_size] += Spp_exps[i];
        Spz_exps[sublattice_size*sublattice_size] += Spz_exps[i];
    }
    
    return {Szz_exps, Spm_exps, Spp_exps, Spz_exps};

}


// writeTPQData was retired in the minimalist-architecture rev (May 2026):
// the text SS_rand*.dat sidecar was eliminated in favour of the unified
// HDF5 store, so this append-only helper had no callers. Use
// writeTPQDataHDF5 instead.

/**
 * Write TPQ thermodynamic data to both text file and HDF5
 * 
 * @param text_file Path to text file (SS_rand*.dat style)
 * @param h5_file Path to HDF5 file (ed_results.h5)
 * @param sample Sample index
 * @param inv_temp Inverse temperature (beta)
 * @param energy Energy expectation value
 * @param variance Energy variance
 * @param doublon Additional observable (e.g., doublon count)
 * @param step TPQ step number
 */
void writeTPQDataHDF5(const std::string& text_file, const std::string& h5_file,
                      size_t sample, double inv_temp, double energy, 
                      double variance, double doublon, uint64_t step) {
    // Write to HDF5 only (text file output removed - data already in HDF5)
    if (!h5_file.empty() && HDF5IO::fileExists(h5_file)) {
        try {
            HDF5IO::TPQThermodynamicPoint point;
            point.beta = inv_temp;
            point.energy = energy;
            point.variance = variance;
            point.doublon = doublon;
            point.step = step;
            HDF5IO::appendTPQThermodynamics(h5_file, sample, point);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to write TPQ thermodynamics to HDF5: " << e.what() << std::endl;
        }
    }
}

/**
 * Write TPQ norm data to both text file and HDF5
 * 
 * @param text_file Path to text file (norm_rand*.dat style)
 * @param h5_file Path to HDF5 file (ed_results.h5)
 * @param sample Sample index
 * @param inv_temp Inverse temperature (beta)
 * @param norm Current norm
 * @param first_norm Initial norm
 * @param step TPQ step number
 */
void writeTPQNormHDF5(const std::string& text_file, const std::string& h5_file,
                      size_t sample, double inv_temp, double norm, 
                      double first_norm, uint64_t step) {
    // Write to HDF5 only (text file output removed - data already in HDF5)
    if (!h5_file.empty() && HDF5IO::fileExists(h5_file)) {
        try {
            HDF5IO::TPQNormPoint point;
            point.beta = inv_temp;
            point.norm = norm;
            point.first_norm = first_norm;
            point.step = step;
            HDF5IO::appendTPQNorm(h5_file, sample, point);
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to write TPQ norm to HDF5: " << e.what() << std::endl;
        }
    }
}

// readTPQData / readTPQDataHDF5 / save_tpq_state (binary sidecar) were
// retired in the minimalist-architecture rev (May 2026): the
// legacy SS_rand*.dat reader path and the binary-sidecar writer were
// only ever consumed by the equally-dead calculate_spectrum_from_tpq /
// get_tpq_state_at_temperature flows. The live thermal-spectrum path
// goes through HDF5IO::saveTPQState + HDF5IO::loadTPQState, and
// post-processing reads the unified HDF5 store directly.

/**
 * Save a TPQ state to the unified HDF5 file
 * 
 * MPI-Safe Implementation:
 * When running with MPI, each rank writes to its own HDF5 file to avoid
 * file locking conflicts. The per-rank files are merged at the end.
 * 
 * @param tpq_state TPQ state vector to save
 * @param dir Output directory (HDF5 file will be created at dir/ed_results.h5)
 * @param sample Sample index
 * @param beta Inverse temperature
 * @param fixed_sz_op Optional FixedSzOperator - if provided, transforms to full basis before saving
 * @return True if successful
 */
bool save_tpq_state_hdf5(const ComplexVector& tpq_state, const std::string& dir,
                         size_t sample, double beta, FixedSzOperator* fixed_sz_op = nullptr) {
    try {
        // MPI-safe: determine the correct HDF5 file path. Guard with
        // MPI_Initialized so this helper is callable from a non-MPI
        // process (unit tests, Python single-process workflows,
        // qed.thermal()).
        std::string hdf5_path;
        bool mpi_path = false;
        #ifdef WITH_MPI
        {
            int mpi_inited = 0;
            MPI_Initialized(&mpi_inited);
            if (mpi_inited) {
                int mpi_rank = 0;
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                hdf5_path = HDF5IO::getPerRankFilePath(dir, mpi_rank, "ed_results.h5");
                if (!HDF5IO::fileExists(hdf5_path)) {
                    HDF5IO::createPerRankFile(dir, mpi_rank, "ed_results.h5");
                }
                mpi_path = true;
            }
        }
        #endif
        if (!mpi_path) {
            hdf5_path = HDF5IO::createOrOpenFile(dir, "ed_results.h5");
        }
        
        // Ensure sample group exists
        HDF5IO::ensureTPQSampleGroup(hdf5_path, sample);
        
        // Transform to full basis if using fixed-Sz
        if (fixed_sz_op != nullptr) {
            std::vector<Complex> full_state = fixed_sz_op->embedToFull(tpq_state);
            HDF5IO::saveTPQState(hdf5_path, sample, beta, full_state);
            std::cout << "  [Fixed-Sz] Transformed state from dim " << tpq_state.size() 
                      << " to full space dim " << full_state.size() << " before saving to HDF5" << std::endl;
        } else {
            HDF5IO::saveTPQState(hdf5_path, sample, beta, tpq_state);
        }
        
        std::cout << "  Saved TPQ state to HDF5: sample=" << sample << ", β=" << beta << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error saving TPQ state to HDF5: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Load a TPQ state from a file
 * 
 * @param tpq_state TPQ state vector to load into
 * @param filename Name of the file to load from
 * @return True if successful
 */
bool load_tpq_state(ComplexVector& tpq_state, const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return false;
    }
    
    size_t size;
    in.read(reinterpret_cast<char*>(&size), sizeof(size_t));
    
    tpq_state.resize(size);
    in.read(reinterpret_cast<char*>(tpq_state.data()), size * sizeof(Complex));
    
    in.close();
    return true;
}

/**
 * Load a TPQ state from a file with optional projection from full to reduced basis
 * 
 * @param tpq_state TPQ state vector to load into (will be in reduced basis if fixed_sz_op provided)
 * @param filename Name of the file to load from
 * @param fixed_sz_op Optional FixedSzOperator - if provided, projects from full to reduced basis
 * @param expected_reduced_dim Expected dimension of reduced basis
 * @return True if successful
 */
bool load_tpq_state(ComplexVector& tpq_state, const std::string& filename, 
                    FixedSzOperator* fixed_sz_op, uint64_t expected_reduced_dim) {
    std::ifstream in(filename, std::ios::binary);
    if (!in.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for reading" << std::endl;
        return false;
    }
    
    size_t file_size;
    in.read(reinterpret_cast<char*>(&file_size), sizeof(size_t));
    
    if (fixed_sz_op != nullptr) {
        size_t full_dim = fixed_sz_op->getFullDim();
        
        if (file_size == full_dim) {
            // State is in full basis - read and project to reduced basis
            ComplexVector full_state(full_dim);
            in.read(reinterpret_cast<char*>(full_state.data()), full_dim * sizeof(Complex));
            in.close();
            
            // Project from full to reduced basis
            tpq_state = fixed_sz_op->projectToReduced(full_state);
            
            if (tpq_state.size() != expected_reduced_dim) {
                std::cerr << "Error: Projected state dimension mismatch. Expected " << expected_reduced_dim 
                          << ", got " << tpq_state.size() << std::endl;
                return false;
            }
            
            std::cout << "  [Fixed-Sz] Projected from full basis (dim=" << full_dim 
                      << ") to reduced basis (dim=" << tpq_state.size() << ")" << std::endl;
            return true;
        } else if (file_size == expected_reduced_dim) {
            // State is already in reduced basis (legacy) - read directly
            tpq_state.resize(expected_reduced_dim);
            in.read(reinterpret_cast<char*>(tpq_state.data()), expected_reduced_dim * sizeof(Complex));
            in.close();
            
            std::cout << "Loaded TPQ state (already in reduced basis) from: " << filename << std::endl;
            return true;
        } else {
            std::cerr << "Error: TPQ state dimension mismatch. Expected " << full_dim 
                      << " (full) or " << expected_reduced_dim << " (reduced), got " << file_size << std::endl;
            in.close();
            return false;
        }
    } else {
        // No fixed_sz_op, just load as-is
        tpq_state.resize(file_size);
        in.read(reinterpret_cast<char*>(tpq_state.data()), file_size * sizeof(Complex));
        in.close();
        return true;
    }
}


// load_raw_data was retired in the minimalist-architecture rev (May 2026):
// no external callers. State persistence goes through HDF5IO::loadTPQState
// / saveTPQState now.

/**
 * Compute spin expectations (S^+, S^-, S^z) at each site using a TPQ state
 * 
 * @param tpq_state The TPQ state vector
 * @param num_sites Number of lattice sites
 * @param spin_l Spin value (e.g., 0.5 for spin-1/2)
 * @param output_file Output file path
 * @param print_output Whether to print results to console
 * @return Vector of spin expectation values organized as [site][S+,S-,Sz]
 */
std::vector<std::vector<Complex>> compute_spin_expectations_from_tpq(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_l,
    const std::string& output_file,
    bool print_output
) {
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    // Initialize expectations matrix: 3 rows (S^+, S^-, S^z) x num_sites columns
    std::vector<std::vector<Complex>> expectations(3, std::vector<Complex>(num_sites, Complex(0.0, 0.0)));
    
    // Create S operators for each site
    std::vector<SingleSiteOperator> Sp_ops;
    std::vector<SingleSiteOperator> Sm_ops;
    std::vector<SingleSiteOperator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sp_ops.emplace_back(num_sites, spin_l, 0, site);
        Sm_ops.emplace_back(num_sites, spin_l, 1, site);
        Sz_ops.emplace_back(num_sites, spin_l, 2, site);
    }
    
    // For each site, compute the expectation values
    for (int site = 0; site < num_sites; site++) {
        // Apply operators
        std::vector<Complex> Sp_psi(N);
        std::vector<Complex> Sm_psi(N);
        std::vector<Complex> Sz_psi(N);
        Sp_ops[site].apply(tpq_state.data(), Sp_psi.data(), N);
        Sm_ops[site].apply(tpq_state.data(), Sm_psi.data(), N);
        Sz_ops[site].apply(tpq_state.data(), Sz_psi.data(), N);
        
        // <psi|S^a|psi> = zdotc(psi, S^a psi). Three independent BLAS-1
        // calls instead of one fused triple-loop — each is vectorised
        // and the loop fusion the compiler could do here was negligible
        // (the operator-application cost dominates anyway).
        Complex Sp_exp, Sm_exp, Sz_exp;
        cblas_zdotc_sub(N, tpq_state.data(), 1, Sp_psi.data(), 1, &Sp_exp);
        cblas_zdotc_sub(N, tpq_state.data(), 1, Sm_psi.data(), 1, &Sm_exp);
        cblas_zdotc_sub(N, tpq_state.data(), 1, Sz_psi.data(), 1, &Sz_exp);

        expectations[0][site] = Sp_exp;
        expectations[1][site] = Sm_exp;
        expectations[2][site] = Sz_exp;
    }
    
    // Print results if requested
    if (print_output) {
        std::cout << "\nSpin Expectation Values from TPQ state:" << std::endl;
        std::cout << std::setw(5) << "Site" 
                << std::setw(20) << "S^+ (real)" 
                << std::setw(20) << "S^+ (imag)" 
                << std::setw(20) << "S^- (real)"
                << std::setw(20) << "S^- (imag)"
                << std::setw(20) << "S^z (real)"
                << std::setw(20) << "S^z (imag)" << std::endl;
        
        for (int site = 0; site < num_sites; site++) {
            std::cout << std::setw(5) << site 
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[0][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[1][site].imag()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].real()
                    << std::setw(20) << std::setprecision(10) << expectations[2][site].imag() << std::endl;
        }
    }
    
    // Save to file if output_file is specified
    if (!output_file.empty()) {
        std::ofstream out(output_file);
        if (out.is_open()) {
            out << "# Site S+_real S+_imag S-_real S-_imag Sz_real Sz_imag" << std::endl;
            for (int site = 0; site < num_sites; site++) {
                out << site << " "
                    << std::setprecision(10) << expectations[0][site].real() << " "
                    << std::setprecision(10) << expectations[0][site].imag() << " "
                    << std::setprecision(10) << expectations[1][site].real() << " "
                    << std::setprecision(10) << expectations[1][site].imag() << " "
                    << std::setprecision(10) << expectations[2][site].real() << " "
                    << std::setprecision(10) << expectations[2][site].imag() << std::endl;
            }
            out.close();
            std::cout << "Spin expectations saved to " << output_file << std::endl;
        }
    }
    
    return expectations;
}





void writeFluctuationData(
    const std::string& flct_file,
    const std::vector<std::string>& spin_corr,
    double inv_temp,
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    const std::vector<SingleSiteOperator>& Sx_ops,
    const std::vector<SingleSiteOperator>& Sy_ops,
    const std::vector<SingleSiteOperator>& Sz_ops,
    const std::pair<std::vector<SingleSiteOperator>, std::vector<SingleSiteOperator>>& double_site_ops,
    uint64_t sublattice_size,
    uint64_t step
) {
    // Compute and write Sz on-demand (memory is freed after computation)
    auto [Sz, Sz2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sz_ops, sublattice_size);
    
    std::ofstream flct_out(flct_file, std::ios::app);
    flct_out << std::setprecision(16) << inv_temp 
             << " " << Sz[sublattice_size].real() << " " << Sz[sublattice_size].imag() 
             << " " << Sz2[sublattice_size].real() << " " << Sz2[sublattice_size].imag();
    
    for (int i = 0; i < sublattice_size; i++) {
        flct_out << " " << Sz[i].real() << " " << Sz[i].imag() 
                 << " " << Sz2[i].real() << " " << Sz2[i].imag();
    }

    auto Spm2exp = calculateSpm_onsite(tpq_state, num_sites, spin_length, double_site_ops.second, sublattice_size);
    flct_out << std::setprecision(16) << " " << Spm2exp.real() << " " << Spm2exp.imag();
    flct_out << " " << step << std::endl;
    flct_out.close();
    
    // Compute and write Sx on-demand (Sz memory is freed before this)
    auto [Sx, Sx2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sx_ops, sublattice_size);
    
    std::string flct_file_x_string = flct_file.substr(0,flct_file.size()-4) + "_Sx.dat";
    std::ofstream flct_out_x(flct_file_x_string, std::ios::app);
    flct_out_x << std::setprecision(16) << inv_temp 
               << " " << Sx[sublattice_size].real() << " " << Sx[sublattice_size].imag() 
               << " " << Sx2[sublattice_size].real() << " " << Sx2[sublattice_size].imag();
    
    for (int i = 0; i < sublattice_size; i++) {
        flct_out_x << " " << Sx[i].real() << " " << Sx[i].imag() 
                   << " " << Sx2[i].real() << " " << Sx2[i].imag();
    }

    flct_out_x << std::setprecision(16) << " " << Spm2exp.real() << " " << Spm2exp.imag();
    flct_out_x << " " << step << std::endl;
    flct_out_x.close();

    // Compute and write Sy on-demand (Sx memory is freed before this)
    auto [Sy, Sy2] = calculateSzandSz2(tpq_state, num_sites, spin_length, Sy_ops, sublattice_size);
    
    std::string flct_file_y_string = flct_file.substr(0,flct_file.size()-4) + "_Sy.dat";
    std::ofstream flct_out_y(flct_file_y_string, std::ios::app);
    flct_out_y << std::setprecision(16) << inv_temp 
               << " " << Sy[sublattice_size].real() << " " << Sy[sublattice_size].imag() 
               << " " << Sy2[sublattice_size].real() << " " << Sy2[sublattice_size].imag();
    
    for (int i = 0; i < sublattice_size; i++) {
        flct_out_y << " " << Sy[i].real() << " " << Sy[i].imag() 
                   << " " << Sy2[i].real() << " " << Sy2[i].imag();
    }

    flct_out_y << std::setprecision(16) << " " << Spm2exp.real() << " " << Spm2exp.imag();
    flct_out_y << " " << step << std::endl;
    flct_out_y.close();

    // Compute and stream correlation data one type at a time
    auto [Szz, Spm, Spp, Spz] = calculateSzzSpm(tpq_state, num_sites, spin_length, double_site_ops, sublattice_size);
    
    for (size_t idx = 0; idx < spin_corr.size(); idx++) {
        std::ofstream corr_out(spin_corr[idx], std::ios::app);
        corr_out << std::setprecision(16) << inv_temp;
        
        // Write total (last element)
        std::vector<Complex>* data_ptr = nullptr;
        if (idx == 0) data_ptr = &Szz;
        else if (idx == 1) data_ptr = &Spm;
        else if (idx == 2) data_ptr = &Spp;
        else if (idx == 3) data_ptr = &Spz;
        
        corr_out << " " << (*data_ptr)[sublattice_size*sublattice_size].real() 
                 << " " << (*data_ptr)[sublattice_size*sublattice_size].imag();
        
        // Write individual correlations
        for (int i = 0; i < sublattice_size*sublattice_size; i++) {
            corr_out << " " << (*data_ptr)[i].real() 
                     << " " << (*data_ptr)[i].imag();
        }
        
        corr_out << " " << step << std::endl;
        corr_out.close();
    }
}

// get_tpq_state_at_temperature was retired in the minimalist-architecture
// rev (May 2026): the SS_rand*.dat sidecar that it scanned for the closest
// beta has been deleted; nearest-beta lookup now reads the unified HDF5
// store via HDF5IO::loadTPQStateByName. No external callers remained.

/**
 * Find the lowest energy state from saved TPQ state files
 * Searches through:
 *   1. HDF5 file (ed_results.h5) for /tpq/samples/sample_N/states/beta_X datasets
 *   2. Binary .dat files: tpq_state_N_beta=X_step=Y.dat (or legacy tpq_state_N_beta=X.dat)
 * Returns the state with highest beta (lowest energy)
 */
bool find_lowest_energy_tpq_state(
    const std::string& tpq_dir,
    uint64_t N,
    uint64_t& out_sample,
    double& out_beta,
    uint64_t& out_step
) {
    namespace fs = std::filesystem;
    
    double max_beta = -1.0;
    bool found = false;
    
    std::cout << "Searching for lowest energy state (highest beta) in " << tpq_dir << std::endl;
    
    // Search through directory for tpq_state files
    if (!fs::exists(tpq_dir) || !fs::is_directory(tpq_dir)) {
        std::cerr << "Error: Directory " << tpq_dir << " does not exist" << std::endl;
        return false;
    }
    
    // ===========================================================================
    // First, search HDF5 file for saved TPQ states (preferred modern format)
    // ===========================================================================
    std::string h5_file = tpq_dir + "/ed_results.h5";
    if (fs::exists(h5_file)) {
        std::cout << "  Checking HDF5 file: " << h5_file << std::endl;
        
        try {
            auto hdf5_states = HDF5IO::listTPQStates(h5_file, -1);  // -1 = all samples
            
            for (const auto& state : hdf5_states) {
                if (state.beta > max_beta) {
                    max_beta = state.beta;
                    out_sample = state.sample_index;
                    out_beta = state.beta;
                    out_step = 0;  // Step not stored with HDF5 states, will look up from thermodynamics
                    found = true;
                }
            }
            
            if (found) {
                // Try to find the step number from thermodynamics data
                auto thermo_data = HDF5IO::loadTPQThermodynamics(h5_file, out_sample);
                if (!thermo_data.empty()) {
                    // Find the step with closest beta
                    double min_diff = std::numeric_limits<double>::max();
                    for (const auto& pt : thermo_data) {
                        double diff = std::abs(pt.beta - out_beta);
                        if (diff < min_diff) {
                            min_diff = diff;
                            out_step = pt.step;
                        }
                    }
                }
                
                std::cout << "Found state in HDF5 (highest beta):" << std::endl;
                std::cout << "  Sample: " << out_sample << std::endl;
                std::cout << "  Beta: " << out_beta << std::endl;
                std::cout << "  Step: " << out_step << std::endl;
                std::cout << "  Dataset: " << "HDF5:" << h5_file << ":/tpq/samples/sample_" 
                          << out_sample << "/states/beta_" << out_beta << std::endl;
                return true;
            }
        } catch (const std::exception& e) {
            std::cout << "  Warning: Could not read HDF5 states: " << e.what() << std::endl;
        }
    }
    
    // ===========================================================================
    // Fall back to binary .dat files (legacy format)
    // ===========================================================================
    std::cout << "  Checking binary .dat files..." << std::endl;
    
    // Regex to match tpq_state_i_beta=*_step=*.dat files (new format with step)
    std::regex state_pattern_new("tpq_state_([0-9]+)_beta=([0-9.]+)_step=([0-9]+)\\.dat");
    // Also support legacy pattern: tpq_state_i_beta=*.dat
    std::regex state_pattern_legacy("tpq_state_([0-9]+)_beta=([0-9.]+)\\.dat");
    
    for (const auto& entry : fs::directory_iterator(tpq_dir)) {
        if (!entry.is_regular_file()) continue;
        
        std::string filename = entry.path().filename().string();
        std::smatch matches;
        
        uint64_t sample = 0;
        double beta = 0.0;
        uint64_t step = 0;
        bool matched = false;
        
        // Try new format first
        if (std::regex_match(filename, matches, state_pattern_new)) {
            if (matches.size() == 4) {
                try {
                    sample = std::stoull(matches[1].str());
                    beta = std::stod(matches[2].str());
                    step = std::stoull(matches[3].str());
                    matched = true;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to parse filename: " << filename << std::endl;
                    continue;
                }
            }
        }
        // Fall back to legacy format
        else if (std::regex_match(filename, matches, state_pattern_legacy)) {
            if (matches.size() == 3) {
                try {
                    sample = std::stoull(matches[1].str());
                    beta = std::stod(matches[2].str());
                    step = 0; // Will need to look up from SS_rand
                    matched = true;
                } catch (const std::exception& e) {
                    std::cerr << "Warning: Failed to parse filename: " << filename << std::endl;
                    continue;
                }
            }
        }
        
        if (matched) {
            // Higher beta = lower energy, so we want the maximum beta
            if (beta > max_beta) {
                max_beta = beta;
                out_sample = sample;
                out_beta = beta;
                out_step = step;
                found = true;
            }
        }
    }
    
    if (found) {
        // If step was not in filename (legacy format), look it up from SS_rand file
        if (out_step == 0) {
            std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(out_sample) + ".dat";
            std::ifstream file(ss_file);
            
            if (file.is_open()) {
                std::string line;
                std::getline(file, line); // Skip header
                
                // Find the step corresponding to this beta
                double min_diff = std::numeric_limits<double>::max();
                while (std::getline(file, line)) {
                    std::istringstream iss(line);
                    double beta, energy, variance, norm, doublon;
                    uint64_t step;
                    
                    if (!(iss >> beta >> energy >> variance >> norm >> doublon >> step)) {
                        continue;
                    }
                    
                    double diff = std::abs(beta - out_beta);
                    if (diff < min_diff) {
                        min_diff = diff;
                        out_step = step;
                    }
                }
                file.close();
            } else {
                std::cout << "Warning: Could not find SS_rand file to determine step number" << std::endl;
            }
        }
        
        std::cout << "Found lowest energy state (highest beta):" << std::endl;
        std::cout << "  Sample: " << out_sample << std::endl;
        std::cout << "  Beta: " << out_beta << std::endl;
        std::cout << "  Step: " << out_step << std::endl;
    } else {
        std::cerr << "Error: No valid TPQ state files found in " << tpq_dir << std::endl;
        std::cerr << "  Looking for files matching patterns: tpq_state_*_beta=*_step=*.dat or tpq_state_*_beta=*.dat" << std::endl;
    }
    
    return found;
}



/**
 * Initialize TPQ output files with appropriate headers
 * 
 * MPI-Safe Implementation:
 * When running with MPI, each rank writes to its own HDF5 file to avoid
 * file locking conflicts. Files are named ed_results_rankN.h5 and are
 * merged by rank 0 at the end of computation.
 * 
 * @param dir Directory for output files
 * @param sample Current sample index
 * @param sublattice_size Size of sublattice for measurements
 * @param measure_sz Whether spin measurements are enabled (controls flct/spin_corr file creation)
 * @return Tuple of filenames (ss_file, norm_file, flct_file, spin_corr, h5_file)
 */
std::tuple<std::string, std::string, std::string, std::vector<std::string>, std::string> initializeTPQFilesWithHDF5(
    const std::string& dir,
    uint64_t sample,
    uint64_t sublattice_size,
    bool measure_sz = false
) {
    std::string ss_file = dir + "/SS_rand" + std::to_string(sample) + ".dat";
    std::string norm_file = dir + "/norm_rand" + std::to_string(sample) + ".dat";
    std::string flct_file = dir + "/flct_rand" + std::to_string(sample) + ".dat";
    
    // MPI-safe HDF5 file naming: each rank gets its own file. The
    // MPI_Initialized guard is needed because this helper can be
    // reached from non-MPI processes (unit tests, Python single-process
    // qed.thermal()) when the binary was built with -DWITH_MPI=ON.
    std::string h5_file;
    int mpi_rank = 0;
    bool mpi_path = false;
    #ifdef WITH_MPI
    {
        int mpi_inited = 0;
        MPI_Initialized(&mpi_inited);
        if (mpi_inited) {
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            h5_file = HDF5IO::getPerRankFilePath(dir, mpi_rank, "ed_results.h5");
            mpi_path = true;
        }
    }
    #endif
    if (!mpi_path) {
        h5_file = dir + "/ed_results.h5";
    }
    
    // Create vector of spin correlation files
    std::vector<std::string> spin_corr_files;
    std::vector<std::string> suffixes = {"SzSz", "SpSm", "SmSm", "SpSz"};
    
    for (const auto& suffix : suffixes) {
        std::string filename = dir + "/spin_corr_" + suffix + "_rand" + std::to_string(sample) + ".dat";
        spin_corr_files.push_back(filename);
    }
    
    // Initialize output files - only fluctuation and correlation files are created
    // SS_rand and norm_rand data is stored in HDF5 only
    {
        // Only create fluctuation and correlation files if measurements are enabled
        if (measure_sz) {
            std::ofstream flct_out(flct_file);
            flct_out << "# inv_temp sz(real) sz(imag) sz2(real) sz2(imag)";

            for (int i = 0; i < sublattice_size; i++) {
                flct_out << " sz" << i << "(real) sz" << i << "(imag)"  << " sz2" << i << "(real) sz2" << i << "(imag)";
            }

            flct_out << " Spm2(real) Spm2(imag)";

            flct_out << " step" << std::endl;

            // Initialize each spin correlation file
            for (const auto& file : spin_corr_files) {
                std::ofstream spin_out(file);
                spin_out << "# inv_temp total(real) total(imag)";
                
                for (int i = 0; i < sublattice_size*sublattice_size; i++) {
                    spin_out << " site" << i << "(real) site" << i << "(imag)";
                }
                spin_out << " step" << std::endl;
            }
        }
    }
    
    // Initialize HDF5 file and sample group (MPI-safe: per-rank file
    // when MPI is active; single file otherwise).
    try {
        if (!HDF5IO::fileExists(h5_file)) {
            if (mpi_path) {
                HDF5IO::createPerRankFile(dir, mpi_rank, "ed_results.h5");
            } else {
                HDF5IO::createOrOpenFile(dir, "ed_results.h5");
            }
        }
        HDF5IO::ensureTPQSampleGroup(h5_file, sample);
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not initialize HDF5 TPQ storage: " << e.what() << std::endl;
        h5_file = "";  // Disable HDF5 writing
    }
    
    return {ss_file, norm_file, flct_file, spin_corr_files, h5_file};
}

/**
 * Initialize TPQ output files with appropriate headers (legacy version without HDF5)
 * 
 * @param dir Directory for output files
 * @param sample Current sample index
 * @param sublattice_size Size of sublattice for measurements
 * @param measure_sz Whether spin measurements are enabled (controls flct/spin_corr file creation)
 * @return Tuple of filenames (ss_file, norm_file, flct_file, spin_corr)
 */
std::tuple<std::string, std::string, std::string, std::vector<std::string>> initializeTPQFiles(
    const std::string& dir,
    uint64_t sample,
    uint64_t sublattice_size,
    bool measure_sz
) {
    auto [ss_file, norm_file, flct_file, spin_corr_files, h5_file] = 
        initializeTPQFilesWithHDF5(dir, sample, sublattice_size, measure_sz);
    return {ss_file, norm_file, flct_file, spin_corr_files};
}


// calculate_spectrum_from_tpq was retired in the minimalist-architecture
// rev (May 2026): the Gaussian-broadening single-temperature spectrum
// approximation was never wired into any workflow, and TPQ dynamical
// spectra now go through ed::observables::cf_dynamical_correlator /
// time_evolution_correlator with the proper continued-fraction
// expansion. No external callers.


/**
 * Standard TPQ (microcanonical) implementation
 * 
 * @param H Hamiltonian operator function
 * @param N Dimension of the Hilbert space (fixed-Sz dimension if using fixed-Sz)
 * @param max_iter Maximum number of iterations
 * @param num_samples Number of random samples
 * @param temp_interval Interval for calculating physical quantities
 * @param eigenvalues Optional output vector for final state energies
 * @param dir Output directory
 * @param compute_spectrum Whether to compute spectrum
 * @param fixed_sz_op Optional FixedSzOperator - if provided, transforms states to full basis before saving
 * @param target_beta Target inverse temperature at which to stop iteration (default 1000.0)
 */
void microcanonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N, 
    uint64_t max_iter,
    uint64_t num_samples,
    uint64_t temp_interval,
    std::vector<double>& eigenvalues,
    std::string dir,
    bool compute_spectrum,
    double LargeValue,
    bool compute_observables,
    std::vector<Operator> observables,
    std::vector<std::string> observable_names,
    double omega_min,
    double omega_max,
    uint64_t num_points,
    double t_end,
    double dt,
    float spin_length,
    bool measure_sz,
    uint64_t sublattice_size,
    uint64_t num_sites,
    FixedSzOperator* fixed_sz_op,
    bool continue_quenching,
    uint64_t continue_sample,
    double continue_beta,
    double target_beta,
    uint64_t num_measure_points,
    double measure_beta_min,
    double measure_beta_max
) {
    // Phase 6.1: dim-aware OMP+BLAS thread cap (see lanczos() rationale).
    // Microcanonical TPQ is dominated by H * v applies; same memory-bandwidth
    // cliff as a Lanczos chain on the same dim.
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    // Audit follow-up: guard with MPI_Initialized so this routine is
    // safe to call from a non-MPI environment (unit tests, Python
    // single-process scripts, qed.thermal()). Without the guard, the
    // first MPI_Comm_rank below aborts because OpenMPI requires
    // MPI_Init beforehand.
    uint64_t start_sample = 0;
    uint64_t end_sample = num_samples;
    uint64_t local_num_samples = num_samples;
    int rank = 0, size = 1;
    bool mpi_active = false;
    #ifdef WITH_MPI
    {
        int mpi_inited = 0;
        MPI_Initialized(&mpi_inited);
        mpi_active = (mpi_inited != 0);
    }
    if (mpi_active) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Distribute samples across ranks using balanced assignment
        uint64_t samples_per_rank = num_samples / size;
        uint64_t remainder = num_samples % size;

        // Ranks with index < remainder get one extra sample
        start_sample = rank * samples_per_rank + std::min((uint64_t)rank, remainder);
        end_sample = start_sample + samples_per_rank + (rank < remainder ? 1 : 0);
        local_num_samples = end_sample - start_sample;

        if (rank == 0) {
            std::cout << "\n==========================================\n";
            std::cout << "MPI-Parallel TPQ Calculation\n";
            std::cout << "==========================================\n";
            std::cout << "Total MPI ranks: " << size << "\n";
            std::cout << "Total samples: " << num_samples << "\n";
            std::cout << "Samples per rank: " << samples_per_rank << " (+ " << remainder << " remainder)\n";
            std::cout << "==========================================\n\n";
        }

        std::cout << "Rank " << rank << " processing samples ["
                  << start_sample << ", " << end_sample << ")\n";

        // Synchronize before starting computation
        MPI_Barrier(MPI_COMM_WORLD);
    }
    #endif
    
    // Create output directory if needed
    if (!dir.empty()) {
        ensureDirectoryExists(dir);
    }
    double D_S = std::log2(N);
    eigenvalues.clear();
    eigenvalues.reserve(local_num_samples);

    // NOTE: Operators are now created INSIDE the sample loop on-demand
    // to avoid keeping all operators in memory simultaneously
    std::cout << "Begin TPQ calculation with dimension " << N << std::endl;
    if (measure_sz) {
        std::cout << "Observable measurements enabled (operators created on-demand per sample)" << std::endl;
    }
    std::string position_file;
    if (!dir.empty()) {
        size_t last_slash_pos = dir.find_last_of('/');
        if (last_slash_pos != std::string::npos) {
            // Check if the last character is a slash, if so, find the previous one
            if (last_slash_pos == dir.length() - 1) {
                last_slash_pos = dir.find_last_of('/', last_slash_pos - 1);
            }
            if (last_slash_pos != std::string::npos) {
                position_file = dir.substr(0, last_slash_pos) + "/positions.dat";
            } else {
                position_file = "positions.dat"; // In case dir is just a name without slashes
            }
        } else {
            position_file = "positions.dat"; // Relative path
        }
    }



    const uint64_t num_temp_points = num_measure_points;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(measure_beta_min);
    double log_max = std::log10(measure_beta_max);
    for (uint64_t i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    std::cout << "Setting LargeValue: " << LargeValue << std::endl;
    std::cout << "Measurement grid: " << num_temp_points << " log-spaced points from β=" << measure_beta_min << " to β=" << measure_beta_max << std::endl;
    std::cout << "Target beta: " << target_beta << std::endl;
    
    // Handle continue-quenching mode
    bool is_continuing = false;
    uint64_t resume_sample = 0;
    uint64_t original_sample = 0;  // Track original sample for loading
    double resume_beta = 0.0;
    uint64_t resume_step = 0;
    
    if (continue_quenching) {
        std::cout << "\n==========================================\n";
        std::cout << "CONTINUE QUENCHING MODE ENABLED\n";
        std::cout << "==========================================\n";
        
        if (continue_sample == 0) {
            // Auto-detect lowest energy state (highest beta) from any sample
            std::cout << "Auto-detecting lowest energy state (highest beta)..." << std::endl;
            if (find_lowest_energy_tpq_state(dir, N, original_sample, resume_beta, resume_step)) {
                is_continuing = true;
                resume_sample = 0;  // Will write to sample 0
            } else {
                std::cout << "Warning: Could not find saved state to continue from. Falling back to normal TPQ (starting fresh)." << std::endl;
                is_continuing = false;
            }
        } else {
            // Use specified sample
            original_sample = continue_sample;
            resume_sample = 0;  // Will write to sample 0
            std::cout << "Continuing from sample " << continue_sample << std::endl;
            
            // Try to find the state file directly
            if (find_lowest_energy_tpq_state(dir, N, original_sample, resume_beta, resume_step)) {
                // Verify it's the requested sample
                if (original_sample == continue_sample) {
                    is_continuing = true;
                } else {
                    std::cout << "Warning: Found sample " << original_sample 
                              << " but requested " << continue_sample << std::endl;
                    is_continuing = false;
                }
            } else {
                std::cout << "Warning: Could not find state file for sample " << continue_sample 
                          << ". Falling back to normal TPQ (starting fresh)." << std::endl;
                is_continuing = false;
            }
        }
        
        if (is_continuing) {
            std::cout << "Resuming from:" << std::endl;
            std::cout << "  Original sample: " << original_sample << std::endl;
            std::cout << "  Continuing as sample: 0 (output to SS_rand0.dat)" << std::endl;
            std::cout << "  Beta: " << resume_beta << std::endl;
            std::cout << "  Step: " << resume_step << std::endl;
            std::cout << "==========================================\n" << std::endl;
        }
    }
    
    // Modified loop: only process samples assigned to this rank
    for (uint64_t sample = start_sample; sample < end_sample; sample++) {
        if (mpi_active) {
            std::cout << "[Rank " << rank << "] TPQ sample " << sample
                      << " of " << num_samples
                      << " (local: " << (sample - start_sample + 1)
                      << " of " << local_num_samples << ")" << std::endl;
        } else {
            std::cout << "TPQ sample " << (sample + 1) << " of " << num_samples << std::endl;
        }
        
        std::vector<bool> temp_measured(num_temp_points, false);
        auto [ss_file, norm_file, flct_file, spin_corr, h5_file] = initializeTPQFilesWithHDF5(dir, sample, sublattice_size, measure_sz);
        
        // Variables that will be initialized differently for continue mode
        ComplexVector v0;
        uint64_t step;
        double inv_temp;
        double energy1, variance1;
        double first_norm, current_norm;
        
        // Temp buffer for Hamiltonian applications (reused throughout)
        ComplexVector temp(N);
        Complex minus_one(-1.0, 0.0);
        
        // Check if we should continue from saved state (only for first sample = 0)
        if (is_continuing && sample == 0) {
            std::cout << "Loading saved state to continue quenching..." << std::endl;
            
            v0.resize(N);
            bool loaded = false;
            
            // ===========================================================================
            // First, try to load from HDF5 (preferred modern format)
            // ===========================================================================
            std::string h5_file = dir + "/ed_results.h5";
            if (std::filesystem::exists(h5_file)) {
                std::cout << "  Trying HDF5 file: " << h5_file << std::endl;
                
                std::vector<Complex> hdf5_state;
                if (HDF5IO::loadTPQState(h5_file, original_sample, resume_beta, hdf5_state)) {
                    if (hdf5_state.size() == N) {
                        v0 = std::move(hdf5_state);
                        std::cout << "  Loaded state from HDF5: sample=" << original_sample 
                                  << ", beta=" << resume_beta << std::endl;
                        loaded = true;
                    } else if (fixed_sz_op && hdf5_state.size() > N) {
                        // State is from full space, need to project to reduced space
                        std::cout << "  Projecting state from full space (" << hdf5_state.size() 
                                  << ") to reduced space (" << N << ")..." << std::endl;
                        v0 = fixed_sz_op->projectToReduced(hdf5_state);
                        loaded = true;
                    } else {
                        std::cout << "  Warning: HDF5 state size mismatch (" << hdf5_state.size() 
                                  << " vs " << N << ")" << std::endl;
                    }
                }
            }
            
            // ===========================================================================
            // Fall back to binary .dat files (legacy format)
            // ===========================================================================
            if (!loaded) {
                // Construct state file path - try new format first, then fall back to legacy
                std::string state_file_new = dir + "/tpq_state_" + std::to_string(original_sample) 
                                           + "_beta=" + std::to_string(resume_beta) 
                                           + "_step=" + std::to_string(resume_step) + ".dat";
                std::string state_file_legacy = dir + "/tpq_state_" + std::to_string(original_sample) 
                                              + "_beta=" + std::to_string(resume_beta) + ".dat";
                
                // Try new format first - use overload that can project from full to reduced
                if (load_tpq_state(v0, state_file_new, fixed_sz_op, N)) {
                    std::cout << "  Loaded state from binary: " << state_file_new << std::endl;
                    loaded = true;
                } 
                // Fall back to legacy format
                else if (load_tpq_state(v0, state_file_legacy, fixed_sz_op, N)) {
                    std::cout << "  Loaded state from binary: " << state_file_legacy << std::endl;
                    loaded = true;
                }
            }
            
            if (!loaded) {
                std::cerr << "Error: Could not load TPQ state from HDF5 or binary files" << std::endl;
                std::cerr << "  Tried HDF5: " << h5_file << " (sample=" << original_sample 
                          << ", beta=" << resume_beta << ")" << std::endl;
                std::cerr << "Starting fresh for this sample." << std::endl;
                goto fresh_start;
            }
            
            // Set starting point from loaded state
            step = resume_step;
            inv_temp = resume_beta;
            
            // Calculate energy and variance of loaded state
            auto [loaded_energy, loaded_variance] = calculateEnergyAndVariance(H, v0, N);
            energy1 = loaded_energy;
            variance1 = loaded_variance;
            
            first_norm = cblas_dznrm2(N, v0.data(), 1);
            current_norm = first_norm;
            
            std::cout << "Loaded state properties:" << std::endl;
            std::cout << "  Energy: " << energy1 << std::endl;
            std::cout << "  Variance: " << variance1 << std::endl;
            std::cout << "  Beta: " << inv_temp << std::endl;
            std::cout << "  Will run " << max_iter << " additional iterations" << std::endl;
            std::cout << "  Target final step: " << (step + max_iter) << std::endl;
            std::cout << "Continuing from step " << step << "..." << std::endl;
        } else {
            fresh_start:
            // Generate initial random state (already normalized).
            // Unified seeding: see tpq_per_sample_seed() in TPQ.h. Set
            // ED_TPQ_BASE_SEED=<unsigned> to make CPU and GPU TPQ runs
            // bit-identical for cross-validation.
            uint64_t seed = tpq_per_sample_seed(sample);
            v0 = generateTPQVector(N, seed);
            
            // Apply initial transformation: v0 = (L*D_S - H)|v0⟩
            H(v0.data(), temp.data(), N);
            Complex scale_factor_large(LargeValue * D_S, 0.0);
            cblas_zscal(N, &scale_factor_large, v0.data(), 1);  // v0 *= L*D_S
            cblas_zaxpy(N, &minus_one, temp.data(), 1, v0.data(), 1);  // v0 = v0 - temp
            
            // Step 1 is initialization only - do NOT write to file
            // The data at step 1 is unphysical (not yet thermalized)
            step = 1;
            
            // Calculate energy and variance for step 1 (for internal use only)
            auto [e1, v1] = calculateEnergyAndVariance(H, v0, N);
            energy1 = e1;
            variance1 = v1;
            inv_temp = (2.0) / (LargeValue* D_S - energy1);

            first_norm = cblas_dznrm2(N, v0.data(), 1);
            Complex scale_factor = Complex(1.0/first_norm, 0.0);

            cblas_zscal(N, &scale_factor, v0.data(), 1);

            current_norm = first_norm;
            
            // Skip writing step 1 - it contains unphysical initialization data
            // Physical data starts from step 2
            
            step = 2; // Start main loop from step 2
        }
        
        // Determine final step: if continuing, run for additional max_iter iterations
        uint64_t final_step = is_continuing && sample == 0 ? (step - 1 + max_iter) : max_iter;
        
        // Main TPQ loop - using in-place operations with single temp buffer
        for (; step <= final_step; step++) {
            // Report progress
            if (step % (max_iter/10) == 0 || step == final_step) {
                std::cout << "  Step " << step << " of " << final_step << std::endl;
            }
            
            // In-place evolution: v0 = (L*D_S - H)|v0⟩
            // First compute temp = H|v0⟩
            H(v0.data(), temp.data(), N);
            
            // Then v0 = L*D_S*v0 - temp (in-place)
            Complex scale_ld(LargeValue * D_S, 0.0);
            cblas_zscal(N, &scale_ld, v0.data(), 1);  // v0 *= L*D_S
            cblas_zaxpy(N, &minus_one, temp.data(), 1, v0.data(), 1);  // v0 = v0 - temp

            current_norm = cblas_dznrm2(N, v0.data(), 1);
            Complex scale_factor = Complex(1.0/current_norm, 0.0);
            cblas_zscal(N, &scale_factor, v0.data(), 1);
            
            // Check if we should measure observables at target temperatures
            // We need to check this at EVERY step to avoid missing temperature points
            bool should_measure_observables = false;
            int target_temp_idx = -1;
            
            // First, do a quick check using estimated temperature
            // Estimate current inverse temperature (using last known energy)
            double estimated_inv_temp = (2.0 * step) / (LargeValue * D_S - energy1);
            
            // Check if we're potentially near any target temperature
            // Use a wider search window (5% instead of 1%) for the initial check
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i]) {
                    double search_tolerance = 0.05 * measure_inv_temp[i];  // 5% search window
                    if (std::abs(estimated_inv_temp - measure_inv_temp[i]) < search_tolerance) {
                        // We're potentially close - need to compute actual energy to be sure
                        should_measure_observables = true;
                        target_temp_idx = i;
                        break;
                    }
                }
            }
            
            // Determine if we should do measurements this step
            bool do_regular_measurement = (step % temp_interval == 0 || step == final_step);
            bool do_measurement = do_regular_measurement || should_measure_observables;
            
            // OPTIMIZED: Calculate energy and variance only when needed
            // This significantly reduces computational cost for large systems
            if (do_measurement) {
                // Calculate energy and variance
                auto [energy_step, variance_step] = calculateEnergyAndVariance(H, v0, N);
                // Update inverse temperature with accurate energy
                inv_temp = (2.0*step) / (LargeValue * D_S - energy_step);
                
                // Update energy for next iteration's estimate
                energy1 = energy_step;
                
                // Now check with accurate temperature if we're really at the target
                bool actually_at_target = false;
                if (should_measure_observables && target_temp_idx >= 0) {
                    double precise_tolerance = 0.01 * measure_inv_temp[target_temp_idx];  // 1% precise tolerance
                    if (std::abs(inv_temp - measure_inv_temp[target_temp_idx]) < precise_tolerance) {
                        actually_at_target = true;
                    }
                }
                
                // Write data (always write when we compute energy) - now to both text and HDF5
                writeTPQDataHDF5(ss_file, h5_file, sample, inv_temp, energy_step, variance_step, 0.0, step);
                
                // Write norm data to both text and HDF5
                writeTPQNormHDF5(norm_file, h5_file, sample, inv_temp, current_norm, first_norm, step);
                
                // Report detailed progress
                if (step % (temp_interval * 10) == 0 || step == final_step) {
                    std::cout << "  Step " << step << ": E = " << energy_step 
                              << ", var = " << variance_step 
                              << ", β = " << inv_temp << std::endl;
                }
                
                // Write fluctuation data only at regular intervals
                if (measure_sz && do_regular_measurement){
                    // Create operators on-demand only when needed (they are freed after use)
                    std::cout << "  Creating operators on-demand for fluctuation measurement..." << std::endl;
                    auto Sx_ops = createSxOperators(num_sites, spin_length);
                    auto Sy_ops = createSyOperators(num_sites, spin_length);
                    auto Sz_ops = createSzOperators(num_sites, spin_length);
                    auto double_site_ops = createSingleOperators_pair(num_sites, spin_length);
                    
                    writeFluctuationData(flct_file, spin_corr, inv_temp, v0, num_sites, spin_length, Sx_ops, Sy_ops, Sz_ops, double_site_ops, sublattice_size, step);
                    // Operators are automatically freed here when they go out of scope
                }
                
                // Save observables at target temperatures (with accurate inv_temp)
                if (actually_at_target) {
                    std::cout << "  *** Saving TPQ state at β = " << inv_temp 
                              << " (target: " << measure_inv_temp[target_temp_idx] << ") ***" << std::endl;
                    // Save to unified HDF5 file (only if computing observables)
                    if (compute_observables) {
                        save_tpq_state_hdf5(v0, dir, sample, inv_temp, fixed_sz_op);
                    }
                    temp_measured[target_temp_idx] = true;
                }
                
                // Check if we've reached the target beta - early termination
                // Note: This is different from continue_quenching - target_beta is a stopping condition
                // regardless of whether we're continuing from a previous run or starting fresh
                if (inv_temp >= target_beta) {
                    std::cout << "  *** Reached target beta " << target_beta 
                              << " (current β = " << inv_temp << ") - stopping iteration ***" << std::endl;
                    break;  // Exit the main TPQ loop for this sample
                }
            }
        }
        
        // Always save final state to HDF5 for continue_quenching
        if (!dir.empty()) {
            save_tpq_state_hdf5(v0, dir, sample, inv_temp, fixed_sz_op);
            std::cout << "Saved final TPQ state to HDF5: sample=" << sample << ", beta=" << inv_temp << std::endl;
        }
        
        // Store final energy for this sample
        eigenvalues.push_back(energy1);
    }
    
    #ifdef WITH_MPI
    if (mpi_active) {
        // Gather all eigenvalues from all ranks to rank 0
        std::vector<double> all_eigenvalues;
        if (rank == 0) {
            all_eigenvalues.resize(num_samples);
        }

        // Calculate receive counts and displacements for MPI_Gatherv
        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int r = 0; r < size; r++) {
            uint64_t r_samples_per_rank = num_samples / size;
            uint64_t r_remainder = num_samples % size;
            uint64_t r_start = r * r_samples_per_rank + std::min((uint64_t)r, r_remainder);
            uint64_t r_count = r_samples_per_rank + (r < r_remainder ? 1 : 0);

            recvcounts[r] = static_cast<int>(r_count);
            displs[r] = static_cast<int>(r_start);
        }

        // Gather eigenvalues from all ranks
        MPI_Gatherv(eigenvalues.data(), static_cast<int>(eigenvalues.size()), MPI_DOUBLE,
                    all_eigenvalues.data(), recvcounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Barrier to ensure all ranks have finished writing their per-rank HDF5 files
        MPI_Barrier(MPI_COMM_WORLD);

        // Update eigenvalues on rank 0 with complete set
        if (rank == 0) {
            eigenvalues = std::move(all_eigenvalues);
            std::cout << "\n==========================================\n";
            std::cout << "MPI TPQ Computation Complete\n";
            std::cout << "Collected " << eigenvalues.size() << " sample energies\n";
            std::cout << "==========================================\n";

            // Merge per-rank HDF5 files into unified output
            HDF5IO::mergePerRankTPQFiles(dir, size, "ed_results.h5", true);

            // Convert TPQ results to unified thermodynamic format
            convert_tpq_to_unified_thermodynamics(dir, num_samples);
        } else {
            // Clear eigenvalues on non-root ranks to save memory
            eigenvalues.clear();
        }

        // Final barrier before returning
        MPI_Barrier(MPI_COMM_WORLD);
    } else
    #endif
    {
        // Non-MPI (or MPI not initialised): convert TPQ results to
        // unified thermodynamic format on a single process.
        convert_tpq_to_unified_thermodynamics(dir, num_samples);
    }
}

// ---------------------------------------------------------------------------
// Krylov-Lanczos propagator for exp(-Δβ H)|ψ>.
//
// Computes |ψ_new> = exp(-Δβ H) |ψ> using a small Lanczos subspace (Saad,
// 1992; Sidje, Expokit). For Δβ * ||H|| moderate, this converges to
// double precision in m ≈ 20–30 iterations and is dramatically more
// accurate than the truncated Taylor series at large Δβ (where Taylor
// can produce coefficients of order (-Δβ * ||H||)^n / n! that overflow
// before being summed).
//
// Algorithm:
//   1. Build orthonormal Lanczos basis V_m (cols of V) and tridiagonal
//      T_m starting from v_1 = ψ / ||ψ||. Full reorthogonalisation against
//      stored basis (m is small, cost negligible).
//   2. Diagonalise T_m via LAPACK dstevd: T_m = Q D Q^T.
//   3. exp(-Δβ T_m) e_1 = Q exp(-Δβ D) Q^T e_1.
//   4. ψ_new = ||ψ|| * V_m * (Q exp(-Δβ D) Q^T e_1).
//
// Returns true on success. On total Lanczos breakdown (β=0 vector) the
// state is left untouched so the caller can fall back.
// ---------------------------------------------------------------------------
inline bool imaginary_time_evolve_tpq_krylov(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_beta,
    int m,
    bool normalize
){
    if (m < 2) m = 2;
    if (static_cast<uint64_t>(m) > N) m = static_cast<int>(N);

    const double initial_norm = cblas_dznrm2(N, state.data(), 1);
    if (!(initial_norm > 0.0)) return false;

    // V is N x m, column-major, contiguous.
    std::vector<ComplexVector> V(m, ComplexVector(N));
    {
        Complex inv_norm(1.0 / initial_norm, 0.0);
        std::copy(state.begin(), state.end(), V[0].begin());
        cblas_zscal(N, &inv_norm, V[0].data(), 1);
    }

    std::vector<double> alpha(m, 0.0);
    std::vector<double> beta(m, 0.0);  // beta[k] = ||r_k||, k>=1; beta[0] = ||v0||
    ComplexVector w(N);

    int actual_m = m;
    for (int k = 0; k < m; ++k) {
        H(V[k].data(), w.data(), N);

        // alpha_k = <V_k, w>  (real for Hermitian H)
        Complex a;
        cblas_zdotc_sub(N, V[k].data(), 1, w.data(), 1, &a);
        alpha[k] = a.real();

        // w := w - alpha_k V_k - beta_{k-1} V_{k-1}
        Complex neg_alpha(-alpha[k], 0.0);
        cblas_zaxpy(N, &neg_alpha, V[k].data(), 1, w.data(), 1);
        if (k > 0) {
            Complex neg_beta(-beta[k], 0.0);
            cblas_zaxpy(N, &neg_beta, V[k - 1].data(), 1, w.data(), 1);
        }

        // Full reorthogonalisation (m is tiny, this is cheap and fixes
        // the well-known drift of the three-term recurrence).
        for (int j = 0; j <= k; ++j) {
            Complex h;
            cblas_zdotc_sub(N, V[j].data(), 1, w.data(), 1, &h);
            Complex neg_h(-h.real(), -h.imag());
            cblas_zaxpy(N, &neg_h, V[j].data(), 1, w.data(), 1);
        }

        if (k + 1 < m) {
            beta[k + 1] = cblas_dznrm2(N, w.data(), 1);
            if (beta[k + 1] < 1e-14) {
                // Krylov subspace exhausted: H is invariant on span(V_0..V_k).
                // The reduced problem is exact on this span.
                actual_m = k + 1;
                break;
            }
            Complex inv_b(1.0 / beta[k + 1], 0.0);
            std::copy(w.begin(), w.end(), V[k + 1].begin());
            cblas_zscal(N, &inv_b, V[k + 1].data(), 1);
        }
    }

    // Diagonalise the symmetric tridiagonal T_m (alpha[0..m-1], beta[1..m-1]).
    // dstevd computes all eigenvalues + eigenvectors of a real symmetric
    // tridiagonal — much cheaper than zheevd and exact for our case.
    std::vector<double> d(actual_m), e(actual_m > 0 ? actual_m - 1 : 0);
    for (int i = 0; i < actual_m; ++i) d[i] = alpha[i];
    for (int i = 0; i + 1 < actual_m; ++i) e[i] = beta[i + 1];
    std::vector<double> Z(actual_m * actual_m, 0.0);
    int info = LAPACKE_dstevd(LAPACK_COL_MAJOR, 'V', actual_m,
                              d.data(), e.data(), Z.data(), actual_m);
    if (info != 0) return false;

    // f = exp(-Δβ T_m) e_1 = Z * diag(exp(-Δβ d_i)) * Z^T * e_1
    //   = sum_i Z[:,i] * exp(-Δβ d_i) * Z[0, i]
    // Energy-shift trick: subtract min(d) before exponentiating to avoid
    // overflow at large Δβ; the global scale is restored by *initial_norm
    // and re-normalisation by the caller.
    double dmin = d[0];
    for (int i = 1; i < actual_m; ++i) if (d[i] < dmin) dmin = d[i];

    std::vector<double> f(actual_m, 0.0);
    for (int i = 0; i < actual_m; ++i) {
        double w_i = std::exp(-delta_beta * (d[i] - dmin)) * Z[0 * actual_m + i];
        for (int row = 0; row < actual_m; ++row) {
            f[row] += Z[row * actual_m + i] * w_i;
        }
    }

    // ψ_new = initial_norm * V * f  — direct gather, m is small so the
    // fused loop beats a full ZGEMV (we'd pay a transpose to pack V).
    ComplexVector new_state(N, Complex(0.0, 0.0));
    for (int j = 0; j < actual_m; ++j) {
        Complex coef(initial_norm * f[j], 0.0);
        cblas_zaxpy(N, &coef, V[j].data(), 1, new_state.data(), 1);
    }

    if (normalize) {
        double nn = cblas_dznrm2(N, new_state.data(), 1);
        if (nn > 0.0) {
            Complex sc(1.0 / nn, 0.0);
            cblas_zscal(N, &sc, new_state.data(), 1);
        }
    }

    state.swap(new_state);
    return true;
}

// Choose the cTPQ propagator at runtime. Default = "taylor" preserves the
// historical behaviour byte-for-byte; "krylov" enables the Saad/Expokit
// path above (recommended for δβ ||H|| ≳ 1).
//
// Krylov subspace size m is read from ED_CTPQ_KRYLOV_M (default 20) so
// users can balance accuracy vs. memory on a per-job basis.
inline std::string ctpq_propagator_choice() {
    static const std::string choice = []() {
        const char* s = std::getenv("ED_CTPQ_PROPAGATOR");
        if (s && s[0] != '\0') {
            std::string v(s);
            std::transform(v.begin(), v.end(), v.begin(),
                           [](unsigned char c) { return std::tolower(c); });
            if (v == "krylov" || v == "taylor") return v;
            std::cerr << "[cTPQ] Unknown ED_CTPQ_PROPAGATOR='" << s
                      << "'; expected 'taylor' or 'krylov'. Falling back to 'taylor'.\n";
        }
        return std::string("taylor");
    }();
    return choice;
}

inline int ctpq_krylov_m_default() {
    static const int m = []() {
        const char* s = std::getenv("ED_CTPQ_KRYLOV_M");
        if (s && s[0] != '\0') {
            try {
                int v = std::stoi(s);
                if (v >= 2 && v <= 200) return v;
            } catch (...) {}
        }
        return 20;
    }();
    return m;
}

// Canonical TPQ using imaginary-time propagation e^{-βH} |r>
inline void imaginary_time_evolve_tpq_taylor(
    std::function<void(const Complex*, Complex*, int)> H,
    ComplexVector& state,
    uint64_t N,
    double delta_beta,
    uint64_t n_max,
    bool normalize
){
    // result = sum_{n=0}^{n_max} (-Δβ H)^n / n! |ψ⟩
    ComplexVector term(N), Hterm(N), result(N);
    std::copy(state.begin(), state.end(), term.begin());
    std::copy(state.begin(), state.end(), result.begin());

    // Iteratively build coefficients to avoid factorial overflow
    // c0 = 1; c_{k} = c_{k-1} * (-Δβ) / k
    double coef_real = 1.0;
    for (int order = 1; order <= n_max; ++order) {
        // term <- H * term
        H(term.data(), Hterm.data(), N);
        std::swap(term, Hterm);

        coef_real *= (-delta_beta) / double(order);
        Complex coef(coef_real, 0.0);

        // result += coef * term  (BLAS-1 zaxpy — vectorised, fused MA)
        cblas_zaxpy(N, &coef, term.data(), 1, result.data(), 1);
    }

    std::swap(state, result);

    if (normalize) {
        double norm = cblas_dznrm2(N, state.data(), 1);
        if (norm > 0.0) {
            Complex scale(1.0 / norm, 0.0);
            cblas_zscal(N, &scale, state.data(), 1);
        }
    }
}

void canonical_tpq(
    std::function<void(const Complex*, Complex*, int)> H,
    uint64_t N,
    double beta_max,
    uint64_t num_samples,
    uint64_t temp_interval,
    std::vector<double>& energies,
    std::string dir,
    double delta_beta,
    uint64_t taylor_order,
    bool compute_observables,
    std::vector<Operator> observables,
    std::vector<std::string> observable_names,
    double omega_min,
    double omega_max,
    uint64_t num_points,
    double t_end,
    double dt,
    float spin_length,
    bool measure_sz,
    uint64_t sublattice_size,
    uint64_t num_sites,
    FixedSzOperator* fixed_sz_op,
    uint64_t num_measure_points,
    double measure_beta_min,
    double measure_beta_max
){
    // Phase 6.1: dim-aware OMP+BLAS thread cap (see lanczos() rationale).
    const ed::parallel::ThreadBudgetScope budget(
        ed::parallel::auto_threads_for_dim(N));

    // Audit follow-up: guard with MPI_Initialized so this routine is
    // safe to call from a non-MPI environment.
    uint64_t start_sample = 0;
    uint64_t end_sample = num_samples;
    uint64_t local_num_samples = num_samples;
    int rank = 0, size = 1;
    bool mpi_active = false;
    #ifdef WITH_MPI
    {
        int mpi_inited = 0;
        MPI_Initialized(&mpi_inited);
        mpi_active = (mpi_inited != 0);
    }
    if (mpi_active) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        uint64_t samples_per_rank = num_samples / size;
        uint64_t remainder = num_samples % size;

        start_sample = rank * samples_per_rank + std::min((uint64_t)rank, remainder);
        end_sample = start_sample + samples_per_rank + (rank < remainder ? 1 : 0);
        local_num_samples = end_sample - start_sample;

        if (rank == 0) {
            std::cout << "\n==========================================\n";
            std::cout << "MPI-Parallel Canonical TPQ Calculation\n";
            std::cout << "==========================================\n";
            std::cout << "Total MPI ranks: " << size << "\n";
            std::cout << "Total samples: " << num_samples << "\n";
            std::cout << "Samples per rank: " << samples_per_rank << " (+ " << remainder << " remainder)\n";
            std::cout << "==========================================\n\n";
        }

        std::cout << "Rank " << rank << " processing samples ["
                  << start_sample << ", " << end_sample << ")\n";

        // Synchronize before starting computation
        MPI_Barrier(MPI_COMM_WORLD);
    }
    #endif

    if (!dir.empty()) { ensureDirectoryExists(dir); }
    energies.clear();
    energies.reserve(local_num_samples);

    // NOTE: Operators are now created INSIDE the sample loop on-demand
    // to avoid keeping all operators in memory simultaneously
    std::cout << "Begin Canonical TPQ calculation with dimension " << N << std::endl;
    if (measure_sz) {
        std::cout << "Observable measurements enabled (operators created on-demand per sample)" << std::endl;
    }

    // Temperature checkpoints (log-spaced β for saving states)
    const uint64_t num_temp_points = num_measure_points;
    std::vector<double> measure_inv_temp(num_temp_points);
    double log_min = std::log10(measure_beta_min);
    double log_max = std::log10(measure_beta_max);
    for (uint64_t i = 0; i < num_temp_points; ++i) {
        measure_inv_temp[i] = std::pow(10.0, log_min + i * (log_max - log_min) / (num_temp_points - 1));
    }

    uint64_t max_steps = std::max(1, int(std::ceil(beta_max / delta_beta)));

    // Modified loop: only process samples assigned to this rank
    for (uint64_t sample = start_sample; sample < end_sample; ++sample) {
        if (mpi_active) {
            std::cout << "[Rank " << rank << "] Canonical TPQ sample " << sample
                      << " of " << num_samples
                      << " (local: " << (sample - start_sample + 1)
                      << " of " << local_num_samples << ")" << std::endl;
        } else {
            std::cout << "Canonical TPQ sample " << (sample + 1)
                      << " of " << num_samples << std::endl;
        }
        
        std::vector<bool> temp_measured(num_temp_points, false);
        
        // Setup filenames for this sample (with HDF5 support)
        auto [ss_file, norm_file, flct_file, spin_corr, h5_file] = initializeTPQFilesWithHDF5(dir, sample, sublattice_size, measure_sz);

        // Initial random normalized state (β=0). Unified seeding for cross-
        // validation against GPU TPQ — see tpq_per_sample_seed() in TPQ.h.
        uint64_t seed = tpq_per_sample_seed(sample);
        ComplexVector psi = generateTPQVector(N, seed);

        // Step 1: record β=0
        {
            auto [e0, var0] = calculateEnergyAndVariance(H, psi, N);
            double inv_temp = 0.0;
            writeTPQDataHDF5(ss_file, h5_file, sample, inv_temp, e0, var0, 0.0, 1);
            writeTPQNormHDF5(norm_file, h5_file, sample, inv_temp, 1.0, 1.0, 1);
        }

        // Main imaginary-time loop
        uint64_t step = 2;
        double beta = 0.0;
        for (int k = 1; k <= max_steps; ++k, ++step) {
            beta += delta_beta;
            if (beta > beta_max + 1e-15) { beta = beta_max; }

            // Evolve by Δβ. Default = Taylor (back-compat); set
            // ED_CTPQ_PROPAGATOR=krylov to enable the Lanczos/Expokit
            // path which converges to machine precision in m ~ 20 steps
            // even when delta_beta * ||H|| is large enough to make the
            // Taylor truncation lossy.
            if (ctpq_propagator_choice() == "krylov") {
                bool ok = imaginary_time_evolve_tpq_krylov(
                    H, psi, N, delta_beta, ctpq_krylov_m_default(), /*normalize=*/true);
                if (!ok) {
                    // Krylov breakdown (e.g. invariant subspace too small):
                    // fall back to Taylor for this step rather than crash.
                    imaginary_time_evolve_tpq_taylor(
                        H, psi, N, delta_beta, taylor_order, /*normalize=*/true);
                }
            } else {
                imaginary_time_evolve_tpq_taylor(
                    H, psi, N, delta_beta, taylor_order, /*normalize=*/true);
            }

            // Check if we should measure observables at target temperatures
            // In canonical TPQ, beta is known exactly, so we can check directly
            bool should_measure_observables = false;
            int target_temp_idx = -1;
            
            for (int i = 0; i < num_temp_points; ++i) {
                if (!temp_measured[i]) {
                    // Use relative tolerance
                    double tolerance = 0.01 * measure_inv_temp[i];  // 1% tolerance
                    if (std::abs(beta - measure_inv_temp[i]) < tolerance) {
                        should_measure_observables = true;
                        target_temp_idx = i;
                        break;
                    }
                }
            }

            // Determine if we should do measurements this step
            bool do_regular_measurement = (k % temp_interval == 0 || k == max_steps);
            bool do_measurement = do_regular_measurement || should_measure_observables;

            // OPTIMIZED: Measurements only when needed
            // This significantly reduces computational cost for large systems
            if (do_measurement) {
                auto [e, var] = calculateEnergyAndVariance(H, psi, N);
                double inv_temp = beta;

                // Write data to both text and HDF5
                writeTPQDataHDF5(ss_file, h5_file, sample, inv_temp, e, var, 0.0, step);
                writeTPQNormHDF5(norm_file, h5_file, sample, inv_temp, 1.0, 1.0, step);

                // Write fluctuation data only at regular intervals
                if (measure_sz && do_regular_measurement){
                    // Create operators on-demand only when needed (they are freed after use)
                    std::cout << "  Creating operators on-demand for fluctuation measurement..." << std::endl;
                    auto Sx_ops = createSxOperators(num_sites, spin_length);
                    auto Sy_ops = createSyOperators(num_sites, spin_length);
                    auto Sz_ops = createSzOperators(num_sites, spin_length);
                    auto double_site_ops = createSingleOperators_pair(num_sites, spin_length);
                    
                    writeFluctuationData(flct_file, spin_corr, inv_temp, psi,
                                         num_sites, spin_length, Sx_ops, Sy_ops, Sz_ops, double_site_ops, sublattice_size, step);
                    // Operators are automatically freed here when they go out of scope
                }
                
                // Save state at target temperature checkpoints
                if (should_measure_observables && target_temp_idx >= 0) {
                    std::cout << "  *** Saving TPQ state at β = " << inv_temp 
                              << " (target: " << measure_inv_temp[target_temp_idx] << ") ***" << std::endl;
                    if (compute_observables) {
                        // Save to unified HDF5 file
                        save_tpq_state_hdf5(psi, dir, sample, inv_temp, fixed_sz_op);
                    }
                    temp_measured[target_temp_idx] = true;
                }
                
                if (k % std::max(static_cast<uint64_t>(1), max_steps / 10) == 0 || k == max_steps) {
                    std::cout << "  β = " << beta << " (" << k << "/" << max_steps << "), E = " << e << std::endl;
                }
            }
        }

        // Final energy at β_max
        auto [ef, _varf] = calculateEnergyAndVariance(H, psi, N);
        energies.push_back(ef);
    }
    
    #ifdef WITH_MPI
    if (mpi_active) {
        // Gather all energies from all ranks to rank 0
        std::vector<double> all_energies;
        if (rank == 0) {
            all_energies.resize(num_samples);
        }

        std::vector<int> recvcounts(size);
        std::vector<int> displs(size);

        for (int r = 0; r < size; r++) {
            uint64_t r_samples_per_rank = num_samples / size;
            uint64_t r_remainder = num_samples % size;
            uint64_t r_start = r * r_samples_per_rank + std::min((uint64_t)r, r_remainder);
            uint64_t r_count = r_samples_per_rank + (r < r_remainder ? 1 : 0);

            recvcounts[r] = static_cast<int>(r_count);
            displs[r] = static_cast<int>(r_start);
        }

        // Gather energies from all ranks
        MPI_Gatherv(energies.data(), static_cast<int>(energies.size()), MPI_DOUBLE,
                    all_energies.data(), recvcounts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Barrier to ensure all ranks have finished writing their per-rank HDF5 files
        MPI_Barrier(MPI_COMM_WORLD);

        // Update energies on rank 0 with complete set
        if (rank == 0) {
            energies = std::move(all_energies);
            std::cout << "\n==========================================\n";
            std::cout << "MPI Canonical TPQ Computation Complete\n";
            std::cout << "Collected " << energies.size() << " sample energies\n";
            std::cout << "==========================================\n";

            // Merge per-rank HDF5 files into unified output
            HDF5IO::mergePerRankTPQFiles(dir, size, "ed_results.h5", true);

            // Convert TPQ results to HDF5 format
            std::string h5_path = HDF5IO::createOrOpenFile(dir);
            convert_tpq_to_unified_thermo(dir, h5_path);
        } else {
            // Clear energies on non-root ranks to save memory
            energies.clear();
        }

        // Final barrier before returning
        MPI_Barrier(MPI_COMM_WORLD);
    } else
    #endif
    {
        // Non-MPI (or MPI not initialised): single-process conversion.
        std::string h5_path = HDF5IO::createOrOpenFile(dir);
        convert_tpq_to_unified_thermo(dir, h5_path);
    }
}

/**
 * @brief Convert TPQ SS_rand files to unified thermodynamic format
 * 
 * Reads all SS_rand*.dat files in the directory, interpolates/bins them
 * to a common temperature grid, and outputs a unified thermo file.
 * 
 * TPQ provides raw microcanonical data:
 *   - beta (inverse temperature)
 *   - energy
 *   - variance
 * 
 * From which we compute:
 *   - Cv = beta^2 * variance
 *   - F = E - T*S (requires integration)
 *   - S = integral(Cv/T) or from free energy
 * 
 * @param tpq_dir Directory containing SS_rand*.dat files
 * @param output_file Output unified thermodynamic file
 * @param temp_min Minimum temperature for output grid
 * @param temp_max Maximum temperature for output grid
 * @param num_temp_bins Number of temperature bins
 * @return true if successful
 */
/**
 * @brief In-memory TPQ post-processor.
 *
 * Implementation notes (audit follow-up): this function is now the
 * canonical place where per-sample TPQ trajectories are interpolated
 * onto a common temperature grid, averaged, and integrated to obtain
 * S(T) and F(T). Both the legacy HDF5-writing convert function and
 * the new unified ``ed::auto_pilot::thermal`` entry point delegate
 * here, so any future tweak to the interpolation / integration logic
 * needs to happen in only one place.
 */
ThermodynamicData compute_tpq_unified_thermo(
    const std::string& tpq_dir,
    double temp_min,
    double temp_max,
    std::uint64_t num_temp_bins
) {
    ThermodynamicData thermo{};

    struct TPQDataPoint {
        double beta;
        double energy;
        double variance;
    };

    std::vector<std::vector<TPQDataPoint>> all_sample_data;

    // Try to read from HDF5 file first (preferred).
    std::string h5_file = HDF5IO::createOrOpenFile(tpq_dir, "ed_results.h5");
    if (HDF5IO::fileExists(h5_file) && !HDF5IO::isDisabledOutputPath(h5_file)) {
        H5E_auto2_t old_func;
        void* old_client_data;
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);

        for (int sample = 0; sample < 1000; ++sample) {
            auto points = HDF5IO::loadTPQThermodynamics(h5_file, sample);
            if (points.empty()) break;

            std::vector<TPQDataPoint> sample_data;
            for (const auto& pt : points) {
                if (pt.beta > 0 && std::isfinite(pt.energy) && std::isfinite(pt.variance)) {
                    sample_data.push_back({pt.beta, pt.energy, pt.variance});
                }
            }
            if (!sample_data.empty()) {
                all_sample_data.push_back(sample_data);
            }
        }
        H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
    }

    // Fallback: legacy text files.
    if (all_sample_data.empty()) {
        for (int sample = 0; sample < 1000; ++sample) {
            std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(sample) + ".dat";
            std::ifstream file(ss_file);
            if (!file.is_open()) break;
            std::vector<TPQDataPoint> sample_data;
            std::string line;
            std::getline(file, line);  // skip header
            while (std::getline(file, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream iss(line);
                double inv_temp, energy, variance, num_acc, step;
                if (iss >> inv_temp >> energy >> variance >> num_acc >> step) {
                    if (inv_temp > 0 && std::isfinite(energy) && std::isfinite(variance)) {
                        sample_data.push_back({inv_temp, energy, variance});
                    }
                }
            }
            if (!sample_data.empty()) {
                all_sample_data.push_back(sample_data);
            }
        }
    }

    if (all_sample_data.empty() || num_temp_bins == 0) {
        return thermo;  // empty
    }

    // Build log-spaced temperature grid.
    std::vector<double> temperatures(num_temp_bins);
    if (num_temp_bins == 1) {
        temperatures[0] = temp_min;
    } else {
        double log_T_min = std::log(std::max(temp_min, 1e-300));
        double log_T_max = std::log(std::max(temp_max, 1e-300));
        double log_T_step = (log_T_max - log_T_min) / (num_temp_bins - 1);
        for (std::uint64_t i = 0; i < num_temp_bins; ++i) {
            temperatures[i] = std::exp(log_T_min + i * log_T_step);
        }
    }

    std::vector<double> energy_mean(num_temp_bins, 0.0);
    std::vector<double> cv_mean(num_temp_bins, 0.0);
    std::vector<std::uint64_t> counts(num_temp_bins, 0);

    for (const auto& sample_data : all_sample_data) {
        std::vector<TPQDataPoint> sorted_data = sample_data;
        std::sort(sorted_data.begin(), sorted_data.end(),
                  [](const TPQDataPoint& a, const TPQDataPoint& b) { return a.beta < b.beta; });

        for (std::uint64_t t_idx = 0; t_idx < num_temp_bins; ++t_idx) {
            double T = temperatures[t_idx];
            double target_beta = 1.0 / std::max(T, 1e-300);

            std::size_t i_low = 0, i_high = sorted_data.size() - 1;
            for (std::size_t i = 0; i < sorted_data.size() - 1; ++i) {
                if (sorted_data[i].beta <= target_beta && sorted_data[i + 1].beta >= target_beta) {
                    i_low = i;
                    i_high = i + 1;
                    break;
                }
            }
            if (target_beta < sorted_data.front().beta || target_beta > sorted_data.back().beta) {
                continue;
            }
            double beta_low = sorted_data[i_low].beta;
            double beta_high = sorted_data[i_high].beta;
            double alpha = (beta_high > beta_low) ?
                           (target_beta - beta_low) / (beta_high - beta_low) : 0.0;
            double E = sorted_data[i_low].energy * (1.0 - alpha) +
                       sorted_data[i_high].energy * alpha;
            double var = sorted_data[i_low].variance * (1.0 - alpha) +
                         sorted_data[i_high].variance * alpha;
            double Cv = target_beta * target_beta * var;

            ++counts[t_idx];
            energy_mean[t_idx] += (E - energy_mean[t_idx]) / counts[t_idx];
            cv_mean[t_idx]     += (Cv - cv_mean[t_idx])     / counts[t_idx];
        }
    }

    // Fill missing bins (no samples at this T) by linear extrapolation from
    // nearest covered bin so the returned grid is dense and the recombiner
    // doesn't see NaNs / random zeros.
    auto fill_holes = [&](std::vector<double>& v) {
        std::int64_t first = -1, last = -1;
        for (std::int64_t i = 0; i < static_cast<std::int64_t>(num_temp_bins); ++i) {
            if (counts[i] > 0) { first = i; break; }
        }
        for (std::int64_t i = static_cast<std::int64_t>(num_temp_bins) - 1; i >= 0; --i) {
            if (counts[i] > 0) { last = i; break; }
        }
        if (first < 0) return;
        for (std::int64_t i = 0; i < first; ++i) v[i] = v[first];
        for (std::int64_t i = last + 1; i < static_cast<std::int64_t>(num_temp_bins); ++i) {
            v[i] = v[last];
        }
    };
    fill_holes(energy_mean);
    fill_holes(cv_mean);

    // Trapezoidal integration of Cv/T → entropy
    std::vector<double> entropy(num_temp_bins, 0.0);
    std::vector<double> free_energy(num_temp_bins, 0.0);
    for (std::uint64_t i = 1; i < num_temp_bins; ++i) {
        double T1 = temperatures[i - 1];
        double T2 = temperatures[i];
        double Cv1 = cv_mean[i - 1];
        double Cv2 = cv_mean[i];
        entropy[i] = entropy[i - 1] + 0.5 * (T2 - T1) * (Cv1 / T1 + Cv2 / T2);
    }
    for (std::uint64_t i = 0; i < num_temp_bins; ++i) {
        free_energy[i] = energy_mean[i] - temperatures[i] * entropy[i];
    }

    thermo.temperatures   = std::move(temperatures);
    thermo.energy         = std::move(energy_mean);
    thermo.specific_heat  = std::move(cv_mean);
    thermo.entropy        = std::move(entropy);
    thermo.free_energy    = std::move(free_energy);
    return thermo;
}

bool convert_tpq_to_unified_thermo(
    const std::string& tpq_dir,
    const std::string& output_file,
    double temp_min,
    double temp_max,
    uint64_t num_temp_bins
) {
    std::cout << "\n=== Converting TPQ data to unified thermodynamic format ===" << std::endl;
    
    // Audit follow-up: delegate the actual interpolation /
    // integration to ``compute_tpq_unified_thermo``. We still need to
    // read the sample trajectories here to compute error bars (which
    // the unified ThermodynamicData struct doesn't carry), but the
    // averaged grid + entropy / free energy must match what
    // ``ed::auto_pilot::thermal`` sees in-memory.
    ThermodynamicData unified =
        compute_tpq_unified_thermo(tpq_dir, temp_min, temp_max, num_temp_bins);
    if (unified.temperatures.empty()) {
        std::cerr << "No TPQ data found in HDF5 or SS_rand*.dat files in " << tpq_dir << std::endl;
        return false;
    }

    // Re-read per-sample data once more for error bar computation. This is
    // unavoidable until ``ThermodynamicData`` gains error-bar fields, but
    // it's still a single ``compute_tpq_unified_thermo`` source of truth
    // for the averaged grid.
    struct TPQDataPoint {
        double beta;
        double energy;
        double variance;
    };

    std::vector<std::vector<TPQDataPoint>> all_sample_data;

    // Try to read from HDF5 file first (preferred)
    std::string h5_file = output_file;  // output_file is the HDF5 path
    if (HDF5IO::fileExists(h5_file)) {
        // Temporarily suppress HDF5 error messages (expected when checking for non-existent samples)
        H5E_auto2_t old_func;
        void* old_client_data;
        H5Eget_auto2(H5E_DEFAULT, &old_func, &old_client_data);
        H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
        
        // Find all available samples in HDF5
        for (int sample = 0; sample < 1000; ++sample) {
            auto points = HDF5IO::loadTPQThermodynamics(h5_file, sample);
            if (points.empty()) {
                break;  // No more samples
            }
            
            std::vector<TPQDataPoint> sample_data;
            for (const auto& pt : points) {
                if (pt.beta > 0 && std::isfinite(pt.energy) && std::isfinite(pt.variance)) {
                    sample_data.push_back({pt.beta, pt.energy, pt.variance});
                }
            }
            
            if (!sample_data.empty()) {
                all_sample_data.push_back(sample_data);
            }
        }
        
        // Restore HDF5 error handling
        H5Eset_auto2(H5E_DEFAULT, old_func, old_client_data);
        
        if (!all_sample_data.empty()) {
            std::cout << "Read TPQ data from HDF5: " << all_sample_data.size() << " samples" << std::endl;
        }
    }
    
    // Fallback: try legacy SS_rand*.dat text files (for backward compatibility)
    if (all_sample_data.empty()) {
        std::cout << "Trying legacy SS_rand*.dat files..." << std::endl;
        for (int sample = 0; sample < 1000; ++sample) {
            std::string ss_file = tpq_dir + "/SS_rand" + std::to_string(sample) + ".dat";
            std::ifstream file(ss_file);
            if (!file.is_open()) {
                break;  // Assume files are numbered consecutively
            }
            
            std::vector<TPQDataPoint> sample_data;
            std::string line;
            
            // Skip header
            std::getline(file, line);
            
            while (std::getline(file, line)) {
                if (line.empty() || line[0] == '#') continue;
                
                std::istringstream iss(line);
                double beta, energy, variance, norm, doublon;
                uint64_t step;
                
                if (iss >> beta >> energy >> variance >> norm >> doublon >> step) {
                    if (beta > 0 && std::isfinite(energy) && std::isfinite(variance)) {
                        sample_data.push_back({beta, energy, variance});
                    }
                }
            }
            
            if (!sample_data.empty()) {
                all_sample_data.push_back(sample_data);
            }
        }
        
        if (!all_sample_data.empty()) {
            std::cout << "Read TPQ data from legacy text files: " << all_sample_data.size() << " samples" << std::endl;
        }
    }
    
    std::cout << "Processing " << all_sample_data.size() << " TPQ samples for error bars" << std::endl;

    // Re-walk samples to estimate per-bin standard error (the unified
    // averaging itself was already done by compute_tpq_unified_thermo).
    const auto& temperatures = unified.temperatures;
    const auto& energy_mean  = unified.energy;
    const auto& cv_mean      = unified.specific_heat;
    std::vector<double> energy_var(num_temp_bins, 0.0);
    std::vector<double> cv_var(num_temp_bins, 0.0);
    std::vector<std::uint64_t> counts(num_temp_bins, 0);

    for (const auto& sample_data : all_sample_data) {
        std::vector<TPQDataPoint> sorted_data = sample_data;
        std::sort(sorted_data.begin(), sorted_data.end(),
                  [](const TPQDataPoint& a, const TPQDataPoint& b) { return a.beta < b.beta; });

        for (std::size_t t_idx = 0; t_idx < num_temp_bins; ++t_idx) {
            double T = temperatures[t_idx];
            double target_beta = 1.0 / std::max(T, 1e-300);
            if (target_beta < sorted_data.front().beta ||
                target_beta > sorted_data.back().beta) {
                continue;
            }
            std::size_t i_low = 0, i_high = sorted_data.size() - 1;
            for (std::size_t i = 0; i < sorted_data.size() - 1; ++i) {
                if (sorted_data[i].beta <= target_beta &&
                    sorted_data[i + 1].beta >= target_beta) {
                    i_low = i;
                    i_high = i + 1;
                    break;
                }
            }
            double beta_low = sorted_data[i_low].beta;
            double beta_high = sorted_data[i_high].beta;
            double alpha = (beta_high > beta_low)
                               ? (target_beta - beta_low) / (beta_high - beta_low)
                               : 0.0;
            double E = sorted_data[i_low].energy * (1.0 - alpha) +
                       sorted_data[i_high].energy * alpha;
            double var = sorted_data[i_low].variance * (1.0 - alpha) +
                         sorted_data[i_high].variance * alpha;
            double Cv = target_beta * target_beta * var;

            ++counts[t_idx];
            double dE = E - energy_mean[t_idx];
            energy_var[t_idx] += dE * dE;
            double dCv = Cv - cv_mean[t_idx];
            cv_var[t_idx] += dCv * dCv;
        }
    }

    std::vector<double> energy_error(num_temp_bins, 0.0);
    std::vector<double> cv_error(num_temp_bins, 0.0);
    for (std::size_t i = 0; i < num_temp_bins; ++i) {
        if (counts[i] > 1) {
            energy_error[i] = std::sqrt(energy_var[i] / (counts[i] - 1) / counts[i]);
            cv_error[i]     = std::sqrt(cv_var[i] / (counts[i] - 1) / counts[i]);
        }
    }

    try {
        HDF5IO::saveFTLMThermodynamics(
            output_file,
            temperatures,
            energy_mean,
            energy_error,
            cv_mean,
            cv_error,
            unified.entropy,
            {},  // entropy errors come from integration
            unified.free_energy,
            {},  // free energy errors come from integration
            all_sample_data.size(),
            "TPQ"
        );

        std::cout << "Saved TPQ thermodynamic data to HDF5: " << output_file << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error writing TPQ output to HDF5: " << e.what() << std::endl;
        return false;
    }
}

/**
 * Convenience wrapper for convert_tpq_to_unified_thermo that auto-generates output path
 */
void convert_tpq_to_unified_thermodynamics(
    const std::string& tpq_dir,
    uint64_t num_samples,
    double temp_min,
    double temp_max,
    uint64_t num_temp_points
) {
    // tpq_dir IS the output directory, use it directly
    std::string h5_path = HDF5IO::createOrOpenFile(tpq_dir);
    
    std::cout << "Converting TPQ results to HDF5..." << std::endl;
    std::cout << "  Input directory: " << tpq_dir << std::endl;
    std::cout << "  Output file: " << h5_path << std::endl;
    std::cout << "  Number of samples: " << num_samples << std::endl;
    std::cout << "  Temperature range: " << temp_min << " to " << temp_max << std::endl;
    
    bool success = convert_tpq_to_unified_thermo(
        tpq_dir, h5_path, temp_min, temp_max, num_temp_points
    );
    
    if (success) {
        std::cout << "Successfully saved TPQ data to HDF5." << std::endl;
    } else {
        std::cerr << "Warning: Failed to save TPQ data to HDF5." << std::endl;
    }
}
