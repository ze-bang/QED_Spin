// =============================================================================
// src/apps/ed_main.cpp
//
// Main entry point for the `ED` driver -- the *one* canonical CLI for every
// ED core routine in the project (ground state / full spectrum / TPQ /
// thermo / dynamical & static structure factors). Thin argv → workflow
// dispatcher: the run_*_workflow / compute_*_workflow / parse_* helpers
// live in `src/cli/workflows.cpp` (declared in
// `include/ed/cli/workflows.h`); the canonical DSSF dispatcher lives in
// `src/cli/dssf_engine.cpp` (declared in `include/ed/dssf/dssf_engine.h`).
//
// This file is responsible only for:
//
//   * Printing --help and --method-info=<name>.
//   * Recognising the `ED dssf <method>` subcommand and re-dispatching
//     into ed::dssf::run() with a parsed EDConfig.
//   * `main()` itself: argv parsing → EDConfig → calling the appropriate
//     workflow function from `<ed/cli/workflows.h>`.
//
// The historical `TPQ_DSSF` standalone binary and the deprecated `--dssf`
// half-positional shim were both deleted in P2.14: `ED dssf <method>` is
// now the only supported DSSF entry point.
// =============================================================================

#include <algorithm>
#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

#include <ed/cli/workflows.h>
#include <ed/core/ed_config.h>
#include <ed/core/ed_config_adapter.h>
#include <ed/core/ed_wrapper.h>
#include <ed/dssf/dssf_engine.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

/**
 * @brief Print help message
 */
void print_help(const char* prog_name) {
    std::cout << "Exact Diagonalization Pipeline\n";
    std::cout << "==============================\n\n";
    std::cout << "Usage:\n";
    std::cout << "  " << prog_name << " <directory> [options]\n";
    std::cout << "  " << prog_name << " --config=<file> [options]\n";
    std::cout << "  " << prog_name << " dssf <method> <directory> [options]\n";
    std::cout << "                          (P2.4 subcommand wired through ed::dssf::run)\n";
    std::cout << "                          method = dynamical_thermal | static_thermal |\n";
    std::cout << "                                   ground_state_dssf  | single_expectation\n\n";

    std::cout << "Quick Examples:\n";
    std::cout << "  # Basic ground state calculation\n";
    std::cout << "  " << prog_name << " ./data --method=LANCZOS\n\n";
    std::cout << "  # T=0 dynamical structure factor via the dssf subcommand\n";
    std::cout << "  " << prog_name << " dssf ground_state_dssf ./data\n\n";
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
    std::cout << "  --symm                  Run symmetry-exploiting diagonalization (canonical flag,\n";
    std::cout << "                          auto-selects best mode; works on CPU and GPU)\n";
    std::cout << "  --symm-threshold=<n>    Hilbert dim threshold for streaming mode (default: 4096)\n";
    std::cout << "  --disk-threshold=<n>    Hilbert dim threshold for disk-streaming mode (default: 67108864)\n";
    std::cout << "  --symmetrized           (deprecated alias for --symm)\n";
    std::cout << "  --streaming-symmetry    (deprecated alias for --symm)\n";
    std::cout << "  --disk-streaming        Run ultra-low-memory disk-based symmetry diagonalization\n";
    std::cout << "                          (processes one sector at a time, uses disk cache;\n";
    std::cout << "                          GPU methods auto-fall back to CPU Lanczos)\n";
    std::cout << "  --chunked-symm          Two-pass chunked symmetry build (lowest-memory basis;\n";
    std::cout << "                          GPU methods auto-fall back to CPU Lanczos)\n";
    std::cout << "  --thermo                Compute thermodynamic properties\n";
    std::cout << "  --dynamical-response    Compute dynamical response (spectral functions)\n";
    std::cout << "  --static-response       Compute static response (thermal expectation values)\n";
    std::cout << "  --ground-state-dssf     Compute T=0 DSSF using continued fraction (optimal for 32-site ED)\n";
    std::cout << "  --translation-only      Use only translation symmetries for symmetry sectors\n";
    std::cout << "                          (requires positions.dat and *_lattice_parameters.dat)\n\n";
    
    std::cout << "TPQ Observable Options:\n";
    std::cout << "  --save-thermal-states   Save TPQ states at target temperatures (for `ED dssf` post-processing)\n";
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
    
    std::cout << "Dynamical Response Operator Configuration:\n";
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
    std::cout << "                          Notes: --fixed-sz response GPU paths are not yet\n";
    std::cout << "                          implemented and silently fall back to CPU.\n";
    std::cout << "                          Single-T dynamical and --ground-state-dssf are CPU-only.\n";
    std::cout << "                          The multi-temperature dynamical workflow is the\n";
    std::cout << "                          GPU-accelerated path.\n\n";
    
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
    
    std::cout << "Static Response Operator Configuration:\n";
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
    std::cout << "DSSF / SSSF SUBCOMMAND\n";
    std::cout << "================================================================================\n";
    std::cout << "All dynamical / static structure-factor calculations route through the\n";
    std::cout << "`ED dssf` subcommand, which dispatches into `ed::dssf::run(...)` -- the\n";
    std::cout << "single canonical engine seam shared by the C++ CLI and the Python\n";
    std::cout << "`quantum_ed.dssf` bindings.\n\n";
    std::cout << "  " << prog_name << " dssf <method> <directory> [options]\n\n";
    std::cout << "  Methods (case-insensitive):\n";
    std::cout << "    dynamical_thermal    Finite-T S(Q,ω) via FTLM continued fraction.\n";
    std::cout << "    static_thermal       Finite-T S(Q) (no ω axis), FTLM thermal averaging.\n";
    std::cout << "    ground_state_dssf    T=0 S(Q,ω) via Lanczos + continued fraction.\n";
    std::cout << "    single_expectation   Single ⟨ψ|O|ψ⟩ (no Hermitian conjugate).\n\n";
    std::cout << "  Per-method knobs reuse the standard --dyn-* / --static-* /\n";
    std::cout << "  --gs-dssf-* flags parsed by EDConfig::fromCommandLine, e.g.:\n";
    std::cout << "    --dyn-omega-min, --dyn-omega-max, --dyn-omega-points, --dyn-broadening\n";
    std::cout << "    --dyn-temp-min, --dyn-temp-max, --dyn-temp-bins, --dyn-samples\n";
    std::cout << "    --dyn-operator-type, --dyn-basis, --dyn-spin-combinations\n";
    std::cout << "    --dyn-momentum-points, --dyn-polarization, --dyn-theta\n";
    std::cout << "    --static-samples, --static-temp-min/--max/--points, --static-no-susceptibility\n";
    std::cout << "    --use-gpu, --n-up=<n>\n\n";
    std::cout << "  Examples:\n";
    std::cout << "    # T=0 ground-state DSSF\n";
    std::cout << "    " << prog_name << " dssf ground_state_dssf ./data\n\n";
    std::cout << "    # Finite-T DSSF with FTLM averaging\n";
    std::cout << "    " << prog_name << " dssf dynamical_thermal ./data \\\n";
    std::cout << "                   --dyn-temp-min=0.1 --dyn-temp-max=10 --dyn-temp-bins=20\n\n";
    std::cout << "    # Static structure factor (SSSF) via FTLM thermal averaging\n";
    std::cout << "    " << prog_name << " dssf static_thermal ./data --static-samples=40\n\n";
    std::cout << "  Note: methods that consume eigenstates / TPQ states read them from\n";
    std::cout << "        <directory>/output/ed_results.h5; run diagonalization or mTPQ\n";
    std::cout << "        first, then use `dssf` for the response calculation.\n";
}

// ============================================================================
// MAIN
//
// Note (P2.14): the legacy `--dssf` half-positional shim and its
// `run_dssf_mode` implementation were removed alongside the `TPQ_DSSF`
// standalone binary. All structure-factor calculations now go through the
// `ED dssf <method>` subcommand handled below, which dispatches into the
// canonical `ed::dssf::run(...)` engine seam from `<ed/dssf/dssf_engine.h>`.
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

    // -------------------------------------------------------------------
    // `ED dssf <method>` subcommand -- the canonical, single entry point
    // for every dynamical / static structure-factor calculation. Routes
    // through `ed::dssf::run(...)` (the same engine seam that the Python
    // `quantum_ed.dssf` bindings consume), so the C++ CLI and the Python
    // wrapper share identical observable construction, dispatch, and
    // on-disk output.
    //
    // Recognised methods: dynamical_thermal | static_thermal |
    //                     ground_state_dssf | single_expectation
    //
    // All other CLI flags continue to be parsed by EDConfig::fromCommandLine
    // (so existing --dyn-* / --static-* / --gs-dssf-* knobs still work);
    // the subcommand only chooses which engine method is invoked.
    // -------------------------------------------------------------------
    if (argc >= 2 && std::string(argv[1]) == "dssf") {
        if (argc < 3) {
            std::cerr << "Error: `ED dssf` requires a method argument.\n"
                      << "Usage: ED dssf <dynamical_thermal|static_thermal|"
                         "ground_state_dssf|single_expectation> "
                         "<directory> [options]\n";
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return 1;
        }

        ed::dssf::DSSFMethod method;
        try {
            method = ed::dssf::method_from_string(argv[2]);
        } catch (const std::invalid_argument& e) {
            std::cerr << "Error: " << e.what() << "\n";
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return 1;
        }

        // Strip the "dssf <method>" prefix so EDConfig parses the rest of
        // argv as a normal ED invocation.
        std::vector<char*> cfg_argv;
        cfg_argv.reserve(argc - 1);
        cfg_argv.push_back(argv[0]);
        for (int i = 3; i < argc; ++i) cfg_argv.push_back(argv[i]);
        EDConfig sub_config = EDConfig::fromCommandLine(
            static_cast<int>(cfg_argv.size()), cfg_argv.data());

        if (!sub_config.validate()) {
            std::cerr << "\nConfiguration validation failed. Use --help.\n";
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return 1;
        }

        create_directory_mpi_safe(sub_config.workflow.output_dir);

        ed::dssf::DSSFRequest request;
        request.method     = method;
        request.output_dir = sub_config.workflow.output_dir;
        request.config     = &sub_config;
        // operators left default-constructed: P2.2 transitional cut still
        // routes operator construction through the workflow body (which
        // reads sub_config.dynamical / .static_resp). P2.3 will populate
        // request.operators here from the same EDConfig fields.

        try {
            const auto result = ed::dssf::run(request);
            std::cout << "\n[ED dssf] method=" << ed::dssf::to_string(result.method)
                      << " tasks=" << result.num_tasks_attempted
                      << " output=" << result.output_dir << "\n";
        } catch (const std::exception& e) {
            std::cerr << "\nError: " << e.what() << "\n";
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

    // Reject the historical `--dssf` flag with a friendly migration hint.
    // The flag, its `run_dssf_mode` shim, and the `TPQ_DSSF` standalone
    // binary were all removed in P2.14; `ED dssf <method>` is the only
    // supported entry point.
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--dssf") {
            std::cerr << "Error: the legacy `--dssf` flag was removed in "
                         "P2.14. Use the `ED dssf <method>` subcommand "
                         "instead (see `ED --help`).\n";
            #ifdef WITH_MPI
            MPI_Finalize();
            #endif
            return 1;
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
        
        // Standalone response calculations (don't require prior diagonalization).
        // P2.2 (DSSF PR-C): all three paths now route through the canonical
        // ed::dssf::run dispatcher so the legacy --dynamical-response /
        // --static-response / --ground-state-dssf flags share the same
        // engine seam as the new `ED dssf` subcommand (P2.4) and the
        // upcoming pybind11 binding (P2.x).
        auto dispatch_dssf = [&config](ed::dssf::DSSFMethod method) {
            ed::dssf::DSSFRequest request;
            request.method     = method;
            request.output_dir = config.workflow.output_dir;
            request.config     = &config;
            ed::dssf::run(request);
        };

        if (config.workflow.compute_dynamical_response) {
            dispatch_dssf(ed::dssf::DSSFMethod::DYNAMICAL_THERMAL);
        }

        if (config.workflow.compute_static_response) {
            dispatch_dssf(ed::dssf::DSSFMethod::STATIC_THERMAL);
        }

        if (config.workflow.compute_ground_state_dssf) {
            dispatch_dssf(ed::dssf::DSSFMethod::GROUND_STATE_DSSF);
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
