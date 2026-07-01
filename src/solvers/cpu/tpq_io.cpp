#include <ed/solvers/TPQ.h>
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/parallel/thread_budget.h>  // Phase 6.1: dim-aware OMP+BLAS cap
#include <ed/parallel/fused_blas1.h>    // Wave 3.6: fused_mtpq_step_complex
#include <algorithm>
#include <array>     // compute_tpq_thermo_from_trajectories aggregator
#include <cctype>
#include <cmath>     // std::isfinite in trajectory aggregator
#include <cstdint>
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

// tpq_io.cpp - split out of TPQ.cpp (architecture hardening D2).
// TPQ state IO + setup utilities (generate/save/load state, Sx/Sy/Sz operator
// builders, HDF5 writers, spin-expectation + fluctuation helpers). All are
// header-declared in <ed/solvers/TPQ.h> (external linkage); the TPQ algorithms
// stay in TPQ.cpp.

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

std::vector<Operator> createSzOperators(int num_sites, float spin_length) {
    std::vector<Operator> Sz_ops;
    for (int site = 0; site < num_sites; site++) {
        Sz_ops.push_back(ed::ops::make_single_site(num_sites, spin_length, 2, site));
    }
    return Sz_ops;
}

std::vector<Operator> createSxOperators(int num_sites, float spin_length) {
    std::vector<Operator> Sx_ops;
    for (int site = 0; site < num_sites; site++) {
        Sx_ops.push_back(ed::ops::make_single_site(num_sites, spin_length, 3, site));
    }
    return Sx_ops;
}

std::vector<Operator> createSyOperators(int num_sites, float spin_length) {
    std::vector<Operator> Sy_ops;
    for (int site = 0; site < num_sites; site++) {
        Sy_ops.push_back(ed::ops::make_single_site(num_sites, spin_length, 4, site));
    }
    return Sy_ops;
}

std::pair<std::vector<Complex>, std::vector<Complex>> calculateSzandSz2(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    const std::vector<Operator>& Sz_ops,
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
    const std::vector<Operator>& Spm_ops,
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


std::pair<std::vector<Operator>, std::vector<Operator>> createSingleOperators_pair(int num_sites, float spin_length) {
    std::vector<Operator> Szz_ops;
    std::vector<Operator> Spm_ops;

    for (int site = 0; site < num_sites; site++) {
        Szz_ops.push_back(ed::ops::make_single_site(num_sites, spin_length, 2, site));
        Spm_ops.push_back(ed::ops::make_single_site(num_sites, spin_length, 0, site));
    }
    return {Szz_ops, Spm_ops};
}



std::tuple<std::vector<Complex>, std::vector<Complex>, std::vector<Complex>, std::vector<Complex>> calculateSzzSpm(
    const ComplexVector& tpq_state,
    uint64_t num_sites,
    float spin_length,
    std::pair<std::vector<Operator>, std::vector<Operator>> double_site_ops,
    uint64_t sublattice_size
){
    // Calculate the dimension of the Hilbert space
    size_t N = 1ULL << num_sites;  // 2^num_sites (64-bit)
    
    ComplexVector Szz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spm_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spp_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    ComplexVector Spz_exps(sublattice_size*sublattice_size+1, Complex(0.0, 0.0));
    
    // Reference operators (avoid copy)
    const std::vector<Operator>& Szz_ops = double_site_ops.first;
    const std::vector<Operator>& Spm_ops = double_site_ops.second;
    
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
    std::vector<Operator> Sp_ops;
    std::vector<Operator> Sm_ops;
    std::vector<Operator> Sz_ops;
    
    for (int site = 0; site < num_sites; site++) {
        Sp_ops.push_back(ed::ops::make_single_site(num_sites, spin_l, 0, site));
        Sm_ops.push_back(ed::ops::make_single_site(num_sites, spin_l, 1, site));
        Sz_ops.push_back(ed::ops::make_single_site(num_sites, spin_l, 2, site));
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
    const std::vector<Operator>& Sx_ops,
    const std::vector<Operator>& Sy_ops,
    const std::vector<Operator>& Sz_ops,
    const std::pair<std::vector<Operator>, std::vector<Operator>>& double_site_ops,
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