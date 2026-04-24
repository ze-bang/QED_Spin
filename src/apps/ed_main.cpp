// =============================================================================
// src/apps/ed_main.cpp
//
// Main entry point for the `ED` driver. Thin argv → workflow dispatcher
// after P1.11 (DSSF PR-B / audit §3.10): the ~1.9 kLOC of run_*_workflow /
// compute_*_workflow / parse_* helpers that used to live above `main()`
// have moved to `src/cli/workflows.cpp` (declared in
// `include/ed/cli/workflows.h`). This file is now responsible only for:
//
//   * Printing --help and --method-info=<name>.
//   * The legacy `--dssf` shim (`run_dssf_mode`) which preserves the
//     half-positional/half-flag CLI used by the j3_h0_scan workflows.
//     This shim will be replaced by the proper `ED dssf` subcommand in
//     P2.4 (DSSF PR-E) and removed in P2.14 (PR-H).
//   * `main()` itself: argv parsing → EDConfig → call the appropriate
//     workflow function from `<ed/cli/workflows.h>`.
// =============================================================================

#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <limits>
#include <filesystem>
#include <fstream>

#include <ed/cli/workflows.h>
#include <ed/core/ed_config.h>
#include <ed/core/ed_config_adapter.h>
#include <ed/core/ed_wrapper.h>
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/dssf/operator_spec.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/observables.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#ifdef WITH_CUDA
#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/gpu_ed_wrapper.h>
#include <cuda_runtime.h>
#endif

/**
 * @brief Print help message
 */
void print_help(const char* prog_name) {
    std::cout << "Exact Diagonalization Pipeline\n";
    std::cout << "==============================\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " <directory> [options]\n";
    std::cout << "  " << prog_name << " --config=<file> [options]\n\n";
    
    std::cout << "Quick Examples:\n";
    std::cout << "  # Basic ground state calculation\n";
    std::cout << "  " << prog_name << " ./data --method=LANCZOS\n\n";
    std::cout << "  # Full spectrum with thermodynamics\n";
    std::cout << "  " << prog_name << " ./data --method=FULL --thermo\n\n";
    std::cout << "  # Symmetry-exploiting calculation (auto-selects best mode)\n";
    std::cout << "  " << prog_name << " ./data --symm --eigenvalues=10\n\n";
    std::cout << "  # Use config file\n";
    std::cout << "  " << prog_name << " --config=ed_config.txt\n\n";
    
    std::cout << "General Options:\n";
    std::cout << "  --config=<file>         Load configuration from file\n";
    std::cout << "  --method=<name>         Diagonalization method (LANCZOS, FULL, mTPQ, etc.)\n";
    std::cout << "  --method-info=<name>    Show detailed parameters for specific method\n";
    std::cout << "  --num_sites=<n>         Number of sites (auto-detected if omitted)\n";
    std::cout << "  --output=<dir>          Output directory\n\n";
    
    std::cout << "Diagonalization Options:\n";
    std::cout << "  --eigenvalues=<n>       Number of eigenvalues (or FULL for complete spectrum)\n";
    std::cout << "  --eigenvectors          Compute eigenvectors\n";
    std::cout << "  --tolerance=<tol>       Convergence tolerance (default: 1e-10)\n";
    std::cout << "  --iterations=<n>        Maximum iterations\n\n";
    
    std::cout << "Workflow Options:\n";
    std::cout << "  --standard              Run standard diagonalization\n";
    std::cout << "  --symm                  Run symmetry-exploiting diagonalization (auto-selects best mode)\n";
    std::cout << "  --symm-threshold=<n>    Hilbert dim threshold for streaming mode (default: 4096)\n";
    std::cout << "  --disk-threshold=<n>    Hilbert dim threshold for disk-streaming mode (default: 67108864)\n";
    std::cout << "  --symmetrized           Run symmetrized diagonalization (exploits symmetries)\n";
    std::cout << "  --streaming-symmetry    Run streaming symmetry diagonalization (memory-efficient,\n";
    std::cout << "                          recommended for systems ≥12 sites)\n";
    std::cout << "  --disk-streaming        Run ultra-low-memory disk-based symmetry diagonalization\n";
    std::cout << "                          (processes one sector at a time, uses disk cache)\n";
    std::cout << "  --thermo                Compute thermodynamic properties\n";
    std::cout << "  --dynamical-response    Compute dynamical response (spectral functions)\n";
    std::cout << "  --static-response       Compute static response (thermal expectation values)\n";
    std::cout << "  --ground-state-dssf     Compute T=0 DSSF using continued fraction (optimal for 32-site ED)\n";
    std::cout << "  --translation-only      Use only translation symmetries for symmetry sectors\n";
    std::cout << "                          (requires positions.dat and *_lattice_parameters.dat)\n\n";
    
    std::cout << "TPQ Observable Options:\n";
    std::cout << "  --save-thermal-states   Save TPQ states at target temperatures (for TPQ_DSSF post-processing)\n";
    std::cout << "  --compute-spin-correlations  Compute ⟨Si⟩ and ⟨Si·Sj⟩ correlations during TPQ evolution\n";
    std::cout << "  --calc_observables      (deprecated) Alias for --save-thermal-states\n";
    std::cout << "  --measure_spin          (deprecated) Alias for --compute-spin-correlations\n\n";
    
    std::cout << "Thermal Options (for mTPQ/cTPQ/FULL):\n";
    std::cout << "  --samples=<n>           Number of TPQ samples\n";
    std::cout << "  --temp_min=<T>          Minimum temperature\n";
    std::cout << "  --temp_max=<T>          Maximum temperature\n";
    std::cout << "  --temp_bins=<n>         Number of temperature bins\n";
    std::cout << "  --continue_quenching    Continue TPQ from saved state (requires prior run)\n";
    std::cout << "  --continue_sample=<n>   Sample to continue from (0 = auto-detect lowest energy)\n";
    std::cout << "  --continue_beta=<β>     Beta to continue from (0.0 = use saved beta)\n";
    std::cout << "  --target_beta=<β>       Target beta at which to stop mTPQ iteration (default: 1000)\n\n";
    
    std::cout << "Dynamical Response Options:\n";
    std::cout << "  --dyn-thermal           Use thermal averaging (multiple random states)\n";
    std::cout << "  --dyn-samples=<n>       Number of random states (default: 20)\n";
    std::cout << "  --dyn-krylov=<n>        Krylov dimension per sample (default: 100)\n";
    std::cout << "  --dyn-omega-min=<ω>     Minimum frequency (default: -10)\n";
    std::cout << "  --dyn-omega-max=<ω>     Maximum frequency (default: 10)\n";
    std::cout << "  --dyn-omega-points=<n>  Number of frequency points (default: 1000)\n";
    std::cout << "  --dyn-broadening=<η>    Lorentzian broadening (default: 0.1)\n";
    std::cout << "  --dyn-temp-min=<T>      Minimum temperature (default: 0.001)\n";
    std::cout << "  --dyn-temp-max=<T>      Maximum temperature (default: 1.0)\n";
    std::cout << "  --dyn-temp-bins=<n>     Number of temperature points (default: 50)\n";
    std::cout << "  --dyn-correlation       Compute two-operator dynamical correlation\n";
    std::cout << "  --dyn-operator=<file>   Operator file to probe (legacy mode)\n";
    std::cout << "  --dyn-operator2=<file>  Second operator for correlation (legacy mode)\n";
    std::cout << "  --dyn-output=<prefix>   Output file prefix (default: dynamical_response)\n";
    std::cout << "  --dyn-seed=<n>          Random seed (0 = auto)\n\n";
    
    std::cout << "Dynamical Response Operator Configuration (like TPQ_DSSF):\n";
    std::cout << "  --dyn-operator-type=<type>     Operator type: sum | transverse | sublattice |\n";
    std::cout << "                                 experimental | transverse_experimental (default: sum)\n";
    std::cout << "  --dyn-basis=<basis>            Basis: ladder (Sp,Sm,Sz) | xyz (Sx,Sy,Sz) (default: ladder)\n";
    std::cout << "  --dyn-spin-combinations=<str>  Spin operators: \"op1,op2;op3,op4\" (default: \"0,0;2,2\")\n";
    std::cout << "                                 For ladder: 0=Sp, 1=Sm, 2=Sz\n";
    std::cout << "                                 For xyz: 0=Sx, 1=Sy, 2=Sz\n";
    std::cout << "  --dyn-unit-cell-size=<n>       Unit cell size for sublattice operators (default: 1)\n";
    std::cout << "  --dyn-momentum-points=<str>    Q-points: \"Qx,Qy,Qz;...\" (default: \"0,0,0;0,0,2\")\n";
    std::cout << "                                 Values are multiples of π\n";
    std::cout << "  --dyn-polarization=<str>       Polarization vector: \"ex,ey,ez\" (default: auto)\n";
    std::cout << "  --dyn-theta=<θ>                Rotation angle for experimental operators (radians)\n\n";
    
    std::cout << "GPU Acceleration Options:\n";
    std::cout << "  --use-gpu               Enable GPU acceleration for both dynamical and static response\n";
    std::cout << "  --dyn-use-gpu           Enable GPU acceleration for dynamical response only\n";
    std::cout << "  --static-use-gpu        Enable GPU acceleration for static response only\n";
    std::cout << "                          Note: Fixed-Sz GPU support is not yet implemented\n\n";
    
    std::cout << "Static Response Options:\n";
    std::cout << "  --static-samples=<n>    Number of random states (default: 20)\n";
    std::cout << "  --static-krylov=<n>     Krylov dimension per sample (default: 400)\n";
    std::cout << "  --static-temp-min=<T>   Minimum temperature (default: 0.001)\n";
    std::cout << "  --static-temp-max=<T>   Maximum temperature (default: 1.0)\n";
    std::cout << "  --static-temp-points=<n> Number of temperature points (default: 100)\n";
    std::cout << "  --static-no-susceptibility  Don't compute susceptibility\n";
    std::cout << "  --static-correlation    Compute two-operator correlation\n";
    std::cout << "  --static-expectation    Compute single-operator <O> (implies --static-response)\n";
    std::cout << "  --static-operator=<file>    Operator file to probe (legacy mode)\n";
    std::cout << "  --static-operator2=<file>   Second operator for correlation (legacy mode)\n";
    std::cout << "  --static-output=<prefix>    Output file prefix (default: static_response)\n";
    std::cout << "  --static-seed=<n>       Random seed (0 = auto)\n\n";
    
    std::cout << "Static Response Operator Configuration (like TPQ_DSSF):\n";
    std::cout << "  --static-operator-type=<type>     Operator type: sum | transverse | sublattice |\n";
    std::cout << "                                    experimental | transverse_experimental (default: sum)\n";
    std::cout << "  --static-basis=<basis>            Basis: ladder (Sp,Sm,Sz) | xyz (Sx,Sy,Sz) (default: ladder)\n";
    std::cout << "  --static-spin-combinations=<str>  Spin operators: \"op1,op2;op3,op4\" (default: \"0,0;2,2\")\n";
    std::cout << "                                    For ladder: 0=Sp, 1=Sm, 2=Sz\n";
    std::cout << "                                    For xyz: 0=Sx, 1=Sy, 2=Sz\n";
    std::cout << "  --static-unit-cell-size=<n>       Unit cell size for sublattice operators (default: 1)\n";
    std::cout << "  --static-momentum-points=<str>    Q-points: \"Qx,Qy,Qz;...\" (default: \"0,0,0;0,0,2\")\n";
    std::cout << "                                    Values are multiples of π\n";
    std::cout << "  --static-polarization=<str>       Polarization vector: \"ex,ey,ez\" (default: auto)\n";
    std::cout << "  --static-theta=<θ>                Rotation angle for experimental operators (radians)\n\n";
    
    std::cout << "Ground State DSSF Options (T=0 Dynamical Correlations):\n";
    std::cout << "  --ground-state-dssf     Compute T=0 DSSF using continued fraction method\n";
    std::cout << "                          Uses same operator options as --dynamical-response\n";
    std::cout << "                          Optimal for 32-site ED with fixed-Sz sector\n";
    std::cout << "                          (Only needs 2-3 Lanczos vectors instead of full spectrum)\n\n";
    
    std::cout << "Fixed-Sz Sector Options:\n";
    std::cout << "  --fixed-sz              Use fixed-Sz sector (reduced Hilbert space)\n";
    std::cout << "  --n-up=<n>              Number of up spins (determines Sz sector, default: N/2)\n\n";
    
    std::cout << "MPI Options:\n";
    std::cout << "  Use mpirun/mpiexec to run with multiple ranks:\n";
    std::cout << "    mpirun -np <N> " << prog_name << " <directory> [options]\n";
    std::cout << "  MPI parallelization automatically applies to:\n";
    std::cout << "    - TPQ samples (mTPQ, cTPQ)\n";
    std::cout << "    - Dynamical response (temperature × operator tasks)\n";
    std::cout << "    - Static response (operator tasks)\n\n";
    
    std::cout << "Available Methods:\n";
    std::cout << "  Lanczos Variants:\n";
    std::cout << "    LANCZOS                Standard Lanczos (default)\n";
    std::cout << "    LANCZOS_SELECTIVE      Lanczos with selective reorthogonalization\n";
    std::cout << "    LANCZOS_NO_ORTHO       Lanczos without reorthogonalization (fastest, least stable)\n";
    std::cout << "    BLOCK_LANCZOS          Block Lanczos for degenerate eigenvalues\n";
    std::cout << "    SHIFT_INVERT           Shift-invert Lanczos for interior eigenvalues\n";
    std::cout << "    SHIFT_INVERT_ROBUST    Robust shift-invert (fallback to standard)\n";
    std::cout << "    KRYLOV_SCHUR           Krylov-Schur method (restarted Lanczos)\n";
    std::cout << "\n";
    std::cout << "  Conjugate Gradient Variants:\n";
    std::cout << "    BICG                   Biconjugate gradient\n";
    std::cout << "    LOBPCG                 Locally optimal block preconditioned CG\n";
    std::cout << "\n";
    std::cout << "  Other Iterative Methods:\n";
    std::cout << "    DAVIDSON               Davidson method\n";
    std::cout << "\n";
    std::cout << "  Full Diagonalization:\n";
    std::cout << "    FULL                   Complete spectrum (exact, memory intensive)\n";
    std::cout << "    OSS                    Optimal spectrum solver (adaptive slicing)\n";
    std::cout << "\n";
    std::cout << "  Thermal Methods:\n";
    std::cout << "    mTPQ                   Microcanonical TPQ\n";
    std::cout << "    cTPQ                   Canonical TPQ\n";
    std::cout << "    mTPQ_MPI               MPI parallel mTPQ (requires MPI build)\n";
    std::cout << "    mTPQ_CUDA              GPU-accelerated mTPQ (requires CUDA build)\n";
    std::cout << "    FTLM                   Finite Temperature Lanczos Method\n";
    std::cout << "    LTLM                   Low Temperature Lanczos Method\n";
    std::cout << "    HYBRID                 Hybrid Thermal Method (LTLM+FTLM auto-switch)\n";
    std::cout << "\n";
    std::cout << "  ARPACK Methods:\n";
    std::cout << "    ARPACK_SM              ARPACK (smallest eigenvalues)\n";
    std::cout << "    ARPACK_LM              ARPACK (largest eigenvalues)\n";
    std::cout << "    ARPACK_SHIFT_INVERT    ARPACK with shift-invert\n";
    std::cout << "    ARPACK_ADVANCED        ARPACK with advanced multi-attempt strategy\n";
    std::cout << "\n";
    std::cout << "  GPU Methods (require CUDA build):\n";
    std::cout << "    LANCZOS_GPU            GPU-accelerated Lanczos\n";
    std::cout << "    LANCZOS_GPU_FIXED_SZ   GPU Lanczos for fixed Sz sector\n";
    std::cout << "    DAVIDSON_GPU           GPU-accelerated Davidson method (recommended)\n";
    std::cout << "    LOBPCG_GPU             GPU-accelerated LOBPCG method\n";
    std::cout << "    FTLM_GPU               GPU-accelerated Finite Temperature Lanczos\n";
    std::cout << "    FTLM_GPU_FIXED_SZ      GPU FTLM for fixed Sz sector\n";
    std::cout << "    mTPQ_GPU               GPU-accelerated microcanonical TPQ\n";
    std::cout << "    cTPQ_GPU               GPU-accelerated canonical TPQ\n\n";
    
    std::cout << "For detailed parameters of any method, use:\n";
    std::cout << "  " << prog_name << " --method-info=<METHOD_NAME>\n";
    std::cout << "\nExample:\n";
    std::cout << "  " << prog_name << " --method-info=LANCZOS\n";
    std::cout << "  " << prog_name << " --method-info=LOBPCG\n";
    std::cout << "  " << prog_name << " --method-info=mTPQ\n";
    std::cout << "  " << prog_name << " --method-info=DAVIDSON_GPU\n\n";
    
    std::cout << "For more options, see documentation or generated config file.\n\n";
    
    std::cout << "================================================================================\n";
    std::cout << "DSSF MODE (TPQ_DSSF-style interface)\n";
    std::cout << "================================================================================\n";
    std::cout << "For spectral function calculations with simpler command-line interface:\n\n";
    std::cout << "  " << prog_name << " --dssf <directory> <krylov_dim> <spin_combinations> [options]\n\n";
    std::cout << "  Required arguments:\n";
    std::cout << "    <directory>          Path containing InterAll.dat, Trans.dat, positions.dat\n";
    std::cout << "    <krylov_dim>         Krylov subspace dimension (30-100 typical)\n";
    std::cout << "    <spin_combinations>  \"op1,op2;op3,op4\" (0=Sp/Sx, 1=Sm/Sy, 2=Sz)\n\n";
    std::cout << "  DSSF-specific options:\n";
    std::cout << "    --dssf-method=<m>    Method: spectral | ftlm_thermal | static | ground_state\n";
    std::cout << "    --dssf-operator=<o>  Operator: sum | transverse | sublattice | experimental\n";
    std::cout << "    --dssf-basis=<b>     Basis: ladder | xyz (default: ladder)\n";
    std::cout << "    --dssf-omega=<min,max,bins,eta>  Frequency grid and broadening\n";
    std::cout << "    --dssf-temps=<min,max,steps>     Temperature range (log spacing)\n";
    std::cout << "    --dssf-momentum=<Qx,Qy,Qz;...>   Momentum points (in units of π)\n";
    std::cout << "    --dssf-samples=<n>   Number of FTLM random samples (default: 40)\n\n";
    std::cout << "  Examples:\n";
    std::cout << "    # SzSz spectral function at Q=0\n";
    std::cout << "    " << prog_name << " --dssf ./data 50 \"2,2\" --dssf-method=spectral\n\n";
    std::cout << "    # Finite-T DSSF with FTLM averaging\n";
    std::cout << "    " << prog_name << " --dssf ./data 50 \"2,2\" --dssf-method=ftlm_thermal \\\n";
    std::cout << "                   --dssf-temps=0.1,10.0,20\n\n";
    std::cout << "    # Static structure factor (SSSF)\n";
    std::cout << "    " << prog_name << " --dssf ./data 50 \"2,2\" --dssf-method=static\n\n";
    std::cout << "  Note: DSSF mode uses TPQ states from ed_results.h5 if available.\n";
    std::cout << "        Run diagonalization/mTPQ first, then use --dssf for post-processing.\n";
}

// ============================================================================
// DSSF MODE (TPQ_DSSF-style interface)
// ============================================================================

/**
 * @brief Run DSSF mode with TPQ_DSSF-style arguments
 * 
 * This provides a simpler interface for spectral function calculations,
 * using positional arguments like TPQ_DSSF.
 */
int run_dssf_mode(int argc, char* argv[]) {
    int rank = 0, size = 1;
    #ifdef WITH_MPI
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    #endif
    
    // Find positional arguments after --dssf
    int dssf_idx = -1;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dssf") {
            dssf_idx = i;
            break;
        }
    }
    
    if (dssf_idx < 0 || dssf_idx + 3 >= argc) {
        if (rank == 0) {
            std::cerr << "DSSF mode requires: --dssf <directory> <krylov_dim> <spin_combinations>\n";
            std::cerr << "Use --help for more information.\n";
        }
        return 1;
    }
    
    // Parse positional arguments
    std::string directory = argv[dssf_idx + 1];
    int krylov_dim = std::stoi(argv[dssf_idx + 2]);
    std::string spin_combinations_str = argv[dssf_idx + 3];
    
    // Parse optional arguments
    std::string method = "spectral";
    std::string operator_type = "sum";
    std::string basis = "ladder";
    double omega_min = -5.0, omega_max = 5.0;
    int num_omega_bins = 200;
    double broadening = 0.1;
    double T_min = 0.1, T_max = 10.0;
    int T_steps = 20;
    bool use_temperature_scan = false;
    std::string momentum_str = "0,0,0";
    std::string polarization_str = "";
    double theta = 0.0;
    int num_samples = 40;
    int n_up = -1;
    bool use_fixed_sz = false;
    int unit_cell_size = 4;
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        
        if (arg.find("--dssf-method=") == 0) {
            method = arg.substr(14);
        } else if (arg.find("--dssf-operator=") == 0) {
            operator_type = arg.substr(16);
        } else if (arg.find("--dssf-basis=") == 0) {
            basis = arg.substr(13);
        } else if (arg.find("--dssf-omega=") == 0) {
            std::string omega_str = arg.substr(13);
            std::stringstream ss(omega_str);
            std::string val;
            std::vector<double> vals;
            while (std::getline(ss, val, ',')) {
                vals.push_back(std::stod(val));
            }
            if (vals.size() >= 1) omega_min = vals[0];
            if (vals.size() >= 2) omega_max = vals[1];
            if (vals.size() >= 3) num_omega_bins = static_cast<int>(vals[2]);
            if (vals.size() >= 4) broadening = vals[3];
        } else if (arg.find("--dssf-temps=") == 0) {
            std::string temps_str = arg.substr(13);
            std::stringstream ss(temps_str);
            std::string val;
            std::vector<double> vals;
            while (std::getline(ss, val, ',')) {
                vals.push_back(std::stod(val));
            }
            if (vals.size() >= 1) T_min = vals[0];
            if (vals.size() >= 2) T_max = vals[1];
            if (vals.size() >= 3) T_steps = static_cast<int>(vals[2]);
            use_temperature_scan = true;
        } else if (arg.find("--dssf-momentum=") == 0) {
            momentum_str = arg.substr(16);
        } else if (arg.find("--dssf-polarization=") == 0) {
            polarization_str = arg.substr(20);
        } else if (arg.find("--dssf-theta=") == 0) {
            theta = std::stod(arg.substr(13));
        } else if (arg.find("--dssf-samples=") == 0) {
            num_samples = std::stoi(arg.substr(15));
        } else if (arg.find("--dssf-n-up=") == 0 || arg.find("--n-up=") == 0) {
            std::string val = (arg.find("--dssf-n-up=") == 0) ? arg.substr(12) : arg.substr(7);
            n_up = std::stoi(val);
            use_fixed_sz = (n_up >= 0);
        } else if (arg == "--fixed-sz") {
            use_fixed_sz = true;
        } else if (arg.find("--dssf-unit-cell=") == 0) {
            unit_cell_size = std::stoi(arg.substr(17));
        } else if (arg.find("--output=") == 0) {
            // output directory can be specified
        }
    }
    
    // Read num_sites from positions.dat
    std::string positions_file = directory + "/positions.dat";
    int num_sites = 0;
    {
        std::ifstream file(positions_file);
        if (!file.is_open()) {
            if (rank == 0) {
                std::cerr << "Error: Cannot open positions.dat at " << positions_file << "\n";
            }
            return 1;
        }
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty() && line[0] != '#') num_sites++;
        }
    }
    
    float spin_length = 0.5f;
    
    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "ED DSSF MODE (TPQ_DSSF-style interface)\n";
        std::cout << "==========================================\n";
        std::cout << "Directory: " << directory << "\n";
        std::cout << "Sites: " << num_sites << ", Spin: " << spin_length << "\n";
        std::cout << "Krylov dimension: " << krylov_dim << "\n";
        std::cout << "Spin combinations: " << spin_combinations_str << "\n";
        std::cout << "Method: " << method << "\n";
        std::cout << "Operator type: " << operator_type << "\n";
        std::cout << "Basis: " << basis << "\n";
        if (method != "static") {
            std::cout << "Omega range: [" << omega_min << ", " << omega_max << "]\n";
            std::cout << "Omega bins: " << num_omega_bins << "\n";
            std::cout << "Broadening: " << broadening << "\n";
        }
        if (use_temperature_scan) {
            std::cout << "Temperature range: [" << T_min << ", " << T_max << "]\n";
            std::cout << "Temperature steps: " << T_steps << "\n";
        }
        if (use_fixed_sz) {
            std::cout << "Fixed-Sz sector: n_up = " << (n_up >= 0 ? std::to_string(n_up) : "N/2") << "\n";
        }
    }
    
    // Load Hamiltonian
    Operator ham_op(num_sites, spin_length);
    std::string interall_file = directory + "/InterAll.dat";
    std::string trans_file = directory + "/Trans.dat";
    std::string three_body_file = directory + "/ThreeBodyG.dat";
    
    if (!std::filesystem::exists(interall_file) || !std::filesystem::exists(trans_file)) {
        if (rank == 0) {
            std::cerr << "Error: Missing Hamiltonian files (InterAll.dat, Trans.dat)\n";
        }
        return 1;
    }
    
    ham_op.loadFromInterAllFile(interall_file);
    ham_op.loadFromFile(trans_file);
    if (std::filesystem::exists(three_body_file)) {
        ham_op.loadThreeBodyTerm(three_body_file);
    }
    
    // Determine Hilbert space dimension
    if (n_up < 0 && use_fixed_sz) {
        n_up = num_sites / 2;
    }
    
    uint64_t N;
    if (use_fixed_sz) {
        N = 1;
        for (uint64_t i = 0; i < static_cast<uint64_t>(n_up); i++) {
            N = N * (num_sites - i) / (i + 1);
        }
    } else {
        N = 1ULL << num_sites;
    }
    
    if (rank == 0) {
        std::cout << "Hilbert space dimension: " << N << "\n";
    }
    
    // Guard against dimension overflow in int-based solver API
    if (N > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        if (rank == 0) {
            std::cerr << "Error: Hilbert space dimension " << N
                      << " exceeds INT_MAX for solver API." << std::endl;
        }
        return 1;
    }
    
    // Create Hamiltonian function wrapper
    auto H = [&ham_op](const Complex* in, Complex* out, int size) {
        ham_op.apply(in, out, size);
    };
    
    // Read ground state energy for energy shift
    double ground_state_energy = 0.0;
    std::string h5_file = directory + "/output/ed_results.h5";
    if (HDF5IO::fileExists(h5_file)) {
        try {
            auto eigenvalues = HDF5IO::loadEigenvalues(h5_file);
            if (!eigenvalues.empty()) {
                ground_state_energy = eigenvalues[0];
                if (rank == 0) {
                    std::cout << "Ground state energy: " << ground_state_energy << "\n";
                }
            }
        } catch (...) {}
    }
    
    // Parse operators
    auto spin_combinations = parse_spin_combinations(spin_combinations_str);
    auto momentum_points = parse_momentum_points(momentum_str);
    auto polarization = polarization_str.empty() ? 
        std::vector<double>{1.0/std::sqrt(2.0), -1.0/std::sqrt(2.0), 0.0} :
        parse_polarization(polarization_str);
    
    // Construct operators
    std::vector<Operator> obs_1, obs_2;
    std::vector<std::string> obs_names;
    
    construct_operators_from_config(
        operator_type, basis, spin_combinations, momentum_points,
        polarization, theta, unit_cell_size, num_sites, spin_length,
        use_fixed_sz, n_up, positions_file,
        obs_1, obs_2, obs_names
    );
    
    if (rank == 0) {
        std::cout << "Number of operator pairs: " << obs_1.size() << "\n";
    }
    
    // Create output directory
    std::string output_dir = directory + "/output/dssf_" + method;
    create_directory_mpi_safe(output_dir);
    
    // Generate temperature grid if needed
    std::vector<double> temperatures;
    if (use_temperature_scan || method == "ftlm_thermal" || method == "static") {
        double log_T_min = std::log(T_min);
        double log_T_max = std::log(T_max);
        double log_step = (log_T_max - log_T_min) / std::max(1, T_steps - 1);
        for (int i = 0; i < T_steps; i++) {
            temperatures.push_back(std::exp(log_T_min + i * log_step));
        }
    }
    
    // Execute the requested method
    if (method == "spectral") {
        // Single-state spectral function
        // Load ground state or TPQ state
        ComplexVector state;
        bool loaded = load_ground_state_from_file(directory + "/output", state, ground_state_energy, N);
        
        if (!loaded) {
            if (rank == 0) {
                std::cerr << "Error: Could not load state for spectral method\n";
                std::cerr << "Run diagonalization first with --eigenvectors\n";
            }
            return 1;
        }
        
        DynamicalResponseParameters params;
        params.krylov_dim = krylov_dim;
        params.broadening = broadening;
        params.tolerance = 1e-10;
        params.full_reorthogonalization = true;
        
        for (size_t i = 0; i < obs_1.size(); i++) {
            auto O1 = [&obs_1, i](const Complex* in, Complex* out, int sz) { obs_1[i].apply(in, out, sz); };
            auto O2 = [&obs_2, i](const Complex* in, Complex* out, int sz) { obs_2[i].apply(in, out, sz); };
            
            auto results = compute_dynamical_correlation_state(
                H, O1, O2, state, N, params,
                omega_min, omega_max, num_omega_bins, 0.0, ground_state_energy
            );
            
            std::string filename = output_dir + "/" + obs_names[i] + "_spectral.txt";
            save_dynamical_response_results(results, filename);
            
            if (rank == 0) {
                std::cout << "Saved: " << filename << "\n";
            }
        }
        
    } else if (method == "ftlm_thermal") {
        // TRUE FTLM thermal averaging
        DynamicalResponseParameters params;
        params.krylov_dim = krylov_dim;
        params.broadening = broadening;
        params.tolerance = 1e-10;
        params.full_reorthogonalization = true;
        params.num_samples = num_samples;
        
        for (size_t i = 0; i < obs_1.size(); i++) {
            auto O1 = [&obs_1, i](const Complex* in, Complex* out, int sz) { obs_1[i].apply(in, out, sz); };
            auto O2 = [&obs_2, i](const Complex* in, Complex* out, int sz) { obs_2[i].apply(in, out, sz); };
            
            auto results_map = compute_dynamical_correlation_multi_sample_multi_temperature(
                H, O1, O2, N, params,
                omega_min, omega_max, num_omega_bins,
                temperatures, ground_state_energy, output_dir
            );
            
            for (const auto& [T, results] : results_map) {
                std::stringstream ss;
                ss << output_dir << "/" << obs_names[i] << "_ftlm_T" << std::fixed << std::setprecision(4) << T << ".txt";
                save_dynamical_response_results(results, ss.str());
                
                if (rank == 0) {
                    std::cout << "Saved: " << ss.str() << "\n";
                }
            }
        }
        
    } else if (method == "static") {
        // Static structure factor
        StaticResponseParameters params;
        params.krylov_dim = krylov_dim;
        params.tolerance = 1e-10;
        params.full_reorthogonalization = true;
        params.num_samples = num_samples;
        params.compute_error_bars = true;
        
        for (size_t i = 0; i < obs_1.size(); i++) {
            auto O1 = [&obs_1, i](const Complex* in, Complex* out, int sz) { obs_1[i].apply(in, out, sz); };
            auto O2 = [&obs_2, i](const Complex* in, Complex* out, int sz) { obs_2[i].apply(in, out, sz); };
            
            auto results = compute_static_response(
                H, O1, O2, N, params,
                T_min, T_max, T_steps, output_dir
            );
            
            std::string filename = output_dir + "/" + obs_names[i] + "_static.txt";
            save_static_response_results(results, filename);
            
            if (rank == 0) {
                std::cout << "Saved: " << filename << "\n";
            }
        }
        
    } else if (method == "ground_state") {
        // Ground state DSSF using continued fraction
        ComplexVector ground_state;
        bool loaded = load_ground_state_from_file(directory + "/output", ground_state, ground_state_energy, N);
        
        if (!loaded) {
            if (rank == 0) {
                std::cerr << "Error: Could not load ground state for ground_state method\n";
            }
            return 1;
        }
        
        GroundStateDSSFParameters gs_params;
        gs_params.krylov_dim = krylov_dim;
        gs_params.omega_min = omega_min;
        gs_params.omega_max = omega_max;
        gs_params.num_omega_points = num_omega_bins;
        gs_params.broadening = broadening;
        gs_params.tolerance = 1e-10;
        
        for (size_t i = 0; i < obs_1.size(); i++) {
            auto O1 = [&obs_1, i](const Complex* in, Complex* out, int sz) { obs_1[i].apply(in, out, sz); };
            auto O2 = [&obs_2, i](const Complex* in, Complex* out, int sz) { obs_2[i].apply(in, out, sz); };
            
            auto results = compute_ground_state_cross_correlation(
                H, O1, O2, ground_state, ground_state_energy, N, gs_params
            );
            
            std::string filename = output_dir + "/" + obs_names[i] + "_ground_state_dssf.txt";
            save_dynamical_response_results(results, filename);
            
            if (rank == 0) {
                std::cout << "Saved: " << filename << "\n";
            }
        }
        
    } else {
        if (rank == 0) {
            std::cerr << "Unknown DSSF method: " << method << "\n";
            std::cerr << "Available methods: spectral, ftlm_thermal, static, ground_state\n";
        }
        return 1;
    }
    
    if (rank == 0) {
        std::cout << "\nDSSF calculation complete.\n";
        std::cout << "Results saved in: " << output_dir << "\n";
    }
    
    return 0;
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char* argv[]) {
    #ifdef WITH_MPI
    // Initialize MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (rank == 0 && size > 1) {
        std::cout << "ED: MPI enabled (" << size << " ranks)\n";
    }
    #endif
    
    // Check for help
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_help(argv[0]);
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return 0;
        }
        
        // Check for --method-info
        if (arg.find("--method-info=") == 0) {
            std::string method_name = arg.substr(14);
            auto method = ed_config::parseMethod(method_name);
            if (method.has_value()) {
                std::cout << ed_config::getMethodParameterInfo(method.value());
            } else {
                std::cerr << "Error: Unknown method '" << method_name << "'\n";
                std::cerr << "Use --help to see available methods.\n";
                #ifdef WITH_MPI
                MPI_Finalize();
                #endif
                return 1;
            }
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return 0;
        }
    }
    
    if (argc < 2) {
        print_help(argv[0]);
        #ifdef WITH_MPI
        MPI_Finalize();
        #endif
        return 1;
    }
    
    // Check for DSSF mode (TPQ_DSSF-style interface)
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dssf") {
            int result = run_dssf_mode(argc, argv);
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return result;
        }
    }
    
    // Parse configuration
    EDConfig config = EDConfig::fromCommandLine(argc, argv);

    
    // Validate
    if (!config.validate()) {
        std::cerr << "\nConfiguration validation failed. Use --help for usage.\n";
        #ifdef WITH_MPI
        MPI_Finalize();
        #endif
        return 1;
    }
    
    // Print configuration summary
    config.print();
    
    // Create output directory first (needed for config.save)
    create_directory_mpi_safe(config.workflow.output_dir);
    
    // Save configuration for reproducibility
    config.save(config.workflow.output_dir + "/ed_config.txt");
    
    // Execute workflows
    EDResults standard_results, sym_results;
    
    try {
        // Handle --precompute-basis: generate orbit basis and cache, then exit
        if (config.workflow.precompute_basis_only) {
            std::cout << "\n========================================\n";
            std::cout << "  Precompute Basis Mode\n";
            std::cout << "========================================\n\n";
            // Force streaming-symmetry path with precompute flag
            run_streaming_symmetry_workflow(config);
            std::cout << "\nBasis precomputation complete. Use --basis-cache-dir="
                      << (config.workflow.basis_cache_dir.empty() 
                          ? config.system.hamiltonian_dir + "/basis_cache" 
                          : config.workflow.basis_cache_dir)
                      << " on subsequent runs to skip sector generation.\n";
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return 0;
        }

        // Handle unified --symm flag: always use streaming-symmetry
        // (supports GPU, basis caching, and works for all system sizes)
        if (config.workflow.run_symm_auto && !config.workflow.skip_ed) {
            sym_results = run_streaming_symmetry_workflow(config);
            print_eigenvalue_summary(sym_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !sym_results.eigenvalues.empty()) {
                compute_thermodynamics(sym_results.eigenvalues, config);
            }
        }
        
        if (config.workflow.run_standard && !config.workflow.skip_ed) {
            standard_results = run_standard_workflow(config);
            print_eigenvalue_summary(standard_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !standard_results.eigenvalues.empty()) {
                compute_thermodynamics(standard_results.eigenvalues, config);
            }
        }
        
        if (config.workflow.run_disk_streaming && !config.workflow.skip_ed) {
            EDResults disk_results = run_disk_streaming_workflow(config);
            print_eigenvalue_summary(disk_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !disk_results.eigenvalues.empty()) {
                compute_thermodynamics(disk_results.eigenvalues, config);
            }
        }
        
        if (config.workflow.run_chunked_symmetry && !config.workflow.skip_ed) {
            EDResults chunked_results = run_chunked_symmetry_workflow(config);
            print_eigenvalue_summary(chunked_results.eigenvalues);
            
            if (config.workflow.compute_thermo && !chunked_results.eigenvalues.empty()) {
                compute_thermodynamics(chunked_results.eigenvalues, config);
            }
        }
        
        // Standalone response calculations (don't require prior diagonalization)
        if (config.workflow.compute_dynamical_response) {
            compute_dynamical_response_workflow(config);
        }
        
        if (config.workflow.compute_static_response) {
            compute_static_response_workflow(config);
        }
        
        // Ground state DSSF (T=0 dynamical correlations using continued fraction)
        if (config.workflow.compute_ground_state_dssf) {
            compute_ground_state_dssf_workflow(config);
        }
        
        // Compare results if both were run
        if (config.workflow.run_standard && config.workflow.run_symm_auto) {
            uint64_t n = std::min(standard_results.eigenvalues.size(), sym_results.eigenvalues.size());
            double max_diff = 0.0;
            for (int i = 0; i < n; i++) {
                double diff = std::abs(standard_results.eigenvalues[i] - sym_results.eigenvalues[i]);
                max_diff = std::max(max_diff, diff);
            }
            std::cout << "\nStandard vs Symmetrized max difference: " << max_diff << "\n";
        }
        
        std::cout << "\nComplete. Results: " << config.workflow.output_dir << "\n";
        
    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        #ifdef WITH_MPI
        MPI_Finalize();
        #endif
        return 1;
    }
    
    #ifdef WITH_MPI
    // Finalize MPI
    MPI_Finalize();
    #endif
    
    return 0;
}
