/**
 * @file dmrg_config.h
 * @brief Configuration structures for DMRG algorithms
 */
#ifndef DMRG_CONFIG_H
#define DMRG_CONFIG_H

#include <string>
#include <vector>
#include <cstdint>

namespace dmrg {

/**
 * @brief Model type for DMRG
 */
enum class ModelType {
    HEISENBERG_XXZ,     // J(Sx Sx + Sy Sy) + Δ Sz Sz
    HEISENBERG_XXX,     // Isotropic: J(S·S)
    ISING_TRANSVERSE,   // J Sz Sz + h Sx
    CUSTOM              // User-defined MPO
};

/**
 * @brief Truncation strategy for SVD
 */
enum class TruncationMode {
    FIXED_CHI,          // Keep exactly chi states
    FIXED_TRUNCATION,   // Keep states until truncation error < epsilon
    ADAPTIVE            // Start small, grow chi as needed
};

/**
 * @brief Configuration for DMRG calculations
 */
struct DMRGConfig {
    // ========== Model Parameters ==========
    ModelType model = ModelType::HEISENBERG_XXX;
    double J = 1.0;              // Exchange coupling
    double Delta = 1.0;          // Anisotropy (XXZ)
    double h = 0.0;              // Transverse field
    std::vector<double> custom_couplings;  // For custom models
    
    // ========== DMRG Parameters ==========
    uint64_t chi_max = 100;      // Maximum bond dimension
    uint64_t chi_init = 10;      // Initial bond dimension
    TruncationMode truncation = TruncationMode::FIXED_CHI;
    double truncation_error = 1e-10;  // For FIXED_TRUNCATION mode
    
    // ========== Convergence ==========
    uint64_t max_sweeps = 100;   // Maximum number of sweeps (finite DMRG)
    uint64_t max_sites = 1000;   // Maximum sites for iDMRG growth
    double energy_tol = 1e-10;   // Energy convergence tolerance
    double entropy_tol = 1e-6;   // Entropy convergence tolerance
    
    // ========== Eigensolver ==========
    uint64_t lanczos_max_iter = 200;  // Max Lanczos iterations
    double lanczos_tol = 1e-12;       // Lanczos convergence tolerance
    uint64_t num_eigenstates = 1;     // Number of states to compute
    
    // ========== Random ==========
    unsigned int seed = 12345;        // Random seed for initialization
    
    // ========== I/O ==========
    std::string output_dir = "./dmrg_output";
    bool save_mps = false;            // Save full MPS to HDF5
    bool save_environments = false;   // Save environment blocks
    uint64_t checkpoint_interval = 10;  // Sweeps between checkpoints
    int verbosity = 1;                // 0=quiet, 1=normal, 2=debug
    
    // ========== Physical ==========
    double spin = 0.5;           // Local spin (0.5 for spin-1/2)
    uint64_t local_dim() const { 
        return static_cast<uint64_t>(2 * spin + 1); 
    }
    
    // Validation
    bool validate() const {
        if (chi_max < 1) return false;
        if (spin < 0) return false;
        if (lanczos_tol <= 0) return false;
        return true;
    }
};

/**
 * @brief Results from DMRG calculation
 */
struct DMRGResults {
    // Ground state energy
    double energy = 0.0;
    double energy_per_site = 0.0;
    double energy_variance = 0.0;
    
    // Entanglement
    double entanglement_entropy = 0.0;
    std::vector<double> entanglement_spectrum;
    
    // Convergence history
    std::vector<double> energy_history;
    std::vector<double> entropy_history;
    std::vector<double> truncation_error_history;
    
    // System info
    uint64_t num_sites = 0;
    uint64_t bond_dimension = 0;
    uint64_t num_sweeps = 0;
    bool converged = false;
    
    // Observables (per site)
    std::vector<double> sz_expectation;    // <Sz_i>
    std::vector<double> sz_sz_correlation; // <Sz_i Sz_j> - <Sz_i><Sz_j>
};

} // namespace dmrg

#endif // DMRG_CONFIG_H
