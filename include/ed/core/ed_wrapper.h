#pragma once

// ============================================================================
// INCLUDES
// ============================================================================
#include <ed/solvers/TPQ.h>
#include <ed/solvers/CG.h>
#include <ed/solvers/arpack.h>
#include <ed/solvers/lanczos.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/hybrid_thermal.h>
#include <ed/solvers/kpm_dos.h>
#include <ed/core/construct_ham.h>
#include <ed/core/hdf5_io.h>
#include <ed/solvers/observables.h>
#include <ed/core/ed_config.h>
#include <ed/core/system_utils.h>
#include <ed/core/ed_logging.h>  // for ed_log::debug() — D-5 routes [DEBUG] prints through here
#include <sys/stat.h>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <memory>  // For std::unique_ptr

// ScaLAPACK support for distributed diagonalization
#ifdef WITH_SCALAPACK
#include <ed/solvers/scalapack_diag.h>
#endif

// GPU support
#ifdef WITH_CUDA
#include <ed/gpu/gpu_ed_wrapper.h>
#endif

// ============================================================================
// Phase 7: this header is the legacy CPU/GPU dispatcher and must keep
// branching on the deprecated `_GPU` / `_CUDA` / `_MPI` / `_FIXED_SZ`
// enum variants for backwards compatibility (HDF5 metadata, pre-Phase-7
// CLI configs, pre-Phase-7 Python code). The orthogonal axes
// (use_gpu / use_mpi / use_fixed_sz on EDParameters) are folded onto
// these enum values inside the dispatcher entry points via
// `ed::canonicalize_method_and_flags()` + `ed::legacy_method_for_dispatch()`,
// so external code only ever sees the deprecation warning if it uses
// the deprecated enum values directly.
//
// We push/pop the deprecated-declarations diagnostic across the entire
// header. The pop is at the very end of the file. Downstream
// translation units that #include <ed/core/ed_wrapper.h> are NOT
// affected: the pop restores their default diagnostic state before
// any user code is parsed.
// ============================================================================
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"


// ============================================================================
// VECTOR OPERATIONS FOR COMPLEX VECTORS
// ============================================================================

/**
 * @brief Vector addition operator for complex vectors
 * @param a First vector
 * @param b Second vector
 * @return Element-wise sum of the two vectors
 * @throws std::invalid_argument if vectors have different sizes
 */
inline std::vector<Complex> operator+ (const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for addition.");
    }
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] + b[i];
    }
    return result;
}

/**
 * @brief Vector subtraction operator for complex vectors
 */
inline std::vector<Complex> operator- (const std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction.");
    }
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] - b[i];
    }
    return result;
}

/**
 * @brief In-place addition operator for complex vectors
 */
inline std::vector<Complex>& operator+= (std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for addition.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] += b[i];
    }
    return a;
}

/**
 * @brief In-place subtraction operator for complex vectors
 */
inline std::vector<Complex>& operator-= (std::vector<Complex>& a, const std::vector<Complex>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vectors must be of the same size for subtraction.");
    }
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] -= b[i];
    }
    return a;
}

/**
 * @brief Scalar multiplication operator for complex vectors
 */
inline std::vector<Complex> operator* (const std::vector<Complex>& a, const Complex& b) {
    std::vector<Complex> result(a.size());
    for (size_t i = 0; i < a.size(); ++i) {
        result[i] = a[i] * b;
    }
    return result;
}

// ============================================================================
// ENUMS AND STRUCTURES
// ============================================================================

// DiagonalizationMethod has been moved to <ed/core/ed_types.h> so that the
// implementation in src/core/ed_config.cpp does not need to include this
// 4500-line header just to mention the enum. (P0.14 / audit Q6.)
#include <ed/core/ed_types.h>
// Method-classification predicates (is_tpq_method, is_gpu_method, ...) live
// in their own header so other TUs don't have to include this monolith
// just to call them. (D-4 in the audit.)
#include <ed/core/ed_method_traits.h>

// ============================================================================
// FEATURE AVAILABILITY CHECKS
// ============================================================================

/**
 * @brief Check if ScaLAPACK support was compiled in
 * @return true if WITH_SCALAPACK was defined at compile time
 */
inline bool is_scalapack_compiled() {
#ifdef WITH_SCALAPACK
    return true;
#else
    return false;
#endif
}

/**
 * @brief Check if CUDA/GPU support was compiled in
 * @return true if WITH_CUDA was defined at compile time
 */
inline bool is_cuda_compiled() {
#ifdef WITH_CUDA
    return true;
#else
    return false;
#endif
}

/**
 * @brief Get fallback method when requested method is unavailable
 * 
 * Provides graceful degradation when optional features aren't compiled in:
 * - ScaLAPACK methods -> FULL (dense diagonalization)
 * - GPU methods -> CPU equivalent
 * 
 * @param method The originally requested method
 * @param verbose Print warning message about fallback
 * @return The method to actually use (may be same as input)
 */
inline DiagonalizationMethod get_fallback_method(DiagonalizationMethod method, bool verbose = true) {
    DiagonalizationMethod fallback = method;
    const char* reason = nullptr;
    
    // Check ScaLAPACK methods
    if (method == DiagonalizationMethod::SCALAPACK || 
        method == DiagonalizationMethod::SCALAPACK_MIXED) {
        if (!is_scalapack_compiled()) {
            fallback = DiagonalizationMethod::FULL;
            reason = "ScaLAPACK not compiled (build with -DWITH_MPI=ON and ScaLAPACK-compatible BLAS)";
        }
    }
    
    // Check GPU methods
    if (!is_cuda_compiled()) {
        switch (method) {
            case DiagonalizationMethod::LANCZOS_GPU:
            case DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ:
                fallback = DiagonalizationMethod::LANCZOS;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::BLOCK_LANCZOS_GPU:
            case DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ:
                fallback = DiagonalizationMethod::BLOCK_LANCZOS;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::DAVIDSON_GPU:
                fallback = DiagonalizationMethod::DAVIDSON;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::LOBPCG_GPU:
                fallback = DiagonalizationMethod::LOBPCG;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::KRYLOV_SCHUR_GPU:
                fallback = DiagonalizationMethod::KRYLOV_SCHUR;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU:
                fallback = DiagonalizationMethod::BLOCK_KRYLOV_SCHUR;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::FULL_GPU:
                fallback = DiagonalizationMethod::FULL;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::mTPQ_GPU:
                fallback = DiagonalizationMethod::mTPQ;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::cTPQ_GPU:
                fallback = DiagonalizationMethod::cTPQ;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            case DiagonalizationMethod::FTLM_GPU:
            case DiagonalizationMethod::FTLM_GPU_FIXED_SZ:
                fallback = DiagonalizationMethod::FTLM;
                reason = "CUDA not compiled (build with -DWITH_CUDA=ON)";
                break;
            default:
                break;
        }
    }
    
    // Print warning if method changed
    if (verbose && fallback != method && reason != nullptr) {
        std::cerr << "Warning: " << reason << "\n";
        std::cerr << "         Falling back to ";
        switch (fallback) {
            case DiagonalizationMethod::FULL: std::cerr << "FULL"; break;
            case DiagonalizationMethod::LANCZOS: std::cerr << "LANCZOS"; break;
            case DiagonalizationMethod::BLOCK_LANCZOS: std::cerr << "BLOCK_LANCZOS"; break;
            case DiagonalizationMethod::DAVIDSON: std::cerr << "DAVIDSON"; break;
            case DiagonalizationMethod::LOBPCG: std::cerr << "LOBPCG"; break;
            case DiagonalizationMethod::KRYLOV_SCHUR: std::cerr << "KRYLOV_SCHUR"; break;
            case DiagonalizationMethod::BLOCK_KRYLOV_SCHUR: std::cerr << "BLOCK_KRYLOV_SCHUR"; break;
            case DiagonalizationMethod::mTPQ: std::cerr << "mTPQ"; break;
            case DiagonalizationMethod::cTPQ: std::cerr << "cTPQ"; break;
            case DiagonalizationMethod::FTLM: std::cerr << "FTLM"; break;
            default: std::cerr << "alternative method"; break;
        }
        std::cerr << " instead.\n\n";
    }
    
    return fallback;
}

/**
 * @brief Structure to hold exact diagonalization results
 */
struct EDResults {
    std::vector<double> eigenvalues;
    bool eigenvectors_computed;
    std::string eigenvectors_path;
    ThermodynamicData thermo_data;  // For thermal calculations
    FTLMResults ftlm_results;       // For FTLM calculations (includes per-sector data)
};

// EDParameters lives in <ed/core/ed_parameters.h> so that the legacy-config
// adapter (and any other code that only needs the parameter bag) doesn't have
// to drag in TPQ.h / CG.h / lanczos.h / ftlm.h / ltlm.h / arpack.h / hdf5_io.h /
// observables.h / system_utils.h / GPU wrappers via this header. (D-2 in the
// modernization audit.)
#include <ed/core/ed_parameters.h>


/**
 * @brief Enum for Hamiltonian file formats
 */
enum class HamiltonianFileFormat {
    STANDARD,       // InterAll.dat and Trans.dat format
    SPARSE_MATRIX,  // Sparse matrix format
    CUSTOM          // Custom format requiring a parser function
};

// ============================================================================
// FORWARD DECLARATIONS
// ============================================================================

// Core diagonalization function
EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    uint64_t hilbert_space_dim,
    DiagonalizationMethod method,
    const EDParameters& params
);

// Helper functions
namespace ed_internal {

    // ========== Method Classification Helpers ==========
    //
    // The actual predicates (is_tpq_method, is_ftlm_method, is_gpu_method,
    // is_deprecated_fixed_sz_method, normalize_method,
    // requires_ground_state_sector, requires_sector_combination) live in
    // <ed/core/ed_method_traits.h>. They are constexpr-friendly and
    // depend only on <ed/core/ed_types.h>, so other translation units
    // (workflows.cpp, ed_config.cpp, future Python bindings) can include
    // *just* the traits header instead of dragging in this whole monolith
    // every time they want to ask "is this method GPU?". (D-4 in the
    // modernization audit.)
    //
    // The `using` declarations inside ed_method_traits.h re-export the
    // ed:: predicates into ed_internal::, so existing call sites keep
    // working unchanged.
    
    /**
     * @brief Normalize method and update use_fixed_sz flag if needed
     * 
     * If a deprecated _FIXED_SZ method is used, converts to base method
     * and sets use_fixed_sz to true. This provides backwards compatibility.
     * 
     * @param method The method enum (will be normalized in-place)
     * @param use_fixed_sz The fixed-Sz flag (will be set to true if _FIXED_SZ method used)
     */
    inline void normalize_method_and_fixed_sz(DiagonalizationMethod& method, bool& use_fixed_sz) {
        if (is_deprecated_fixed_sz_method(method)) {
            std::cerr << "Warning: Using deprecated _FIXED_SZ method variant. "
                      << "Use the base method with --fixed-sz flag instead.\n";
            use_fixed_sz = true;  // Force fixed-Sz mode
            method = normalize_method(method);  // Convert to base method
        }
    }
    
    /**
     * @brief Check if method supports fixed-Sz operation
     * 
     * All methods support fixed-Sz via the appropriate operator class:
     * - CPU methods: FixedSzOperator
     * - GPU methods: GPUFixedSzOperator (for LANCZOS_GPU, BLOCK_LANCZOS_GPU, FTLM_GPU)
     */
    inline bool supports_fixed_sz(DiagonalizationMethod method) {
        // Normalize first to handle deprecated variants
        DiagonalizationMethod base = normalize_method(method);
        
        // All CPU methods support fixed-Sz
        if (!is_gpu_method(base)) {
            return true;
        }
        
        // GPU methods that support fixed-Sz
        return base == DiagonalizationMethod::LANCZOS_GPU ||
               base == DiagonalizationMethod::BLOCK_LANCZOS_GPU ||
               base == DiagonalizationMethod::FTLM_GPU ||
               base == DiagonalizationMethod::DAVIDSON_GPU ||
               base == DiagonalizationMethod::LOBPCG_GPU ||
               base == DiagonalizationMethod::KRYLOV_SCHUR_GPU ||
               base == DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU ||
               base == DiagonalizationMethod::mTPQ_GPU ||
               base == DiagonalizationMethod::cTPQ_GPU ||
               base == DiagonalizationMethod::FULL_GPU;
    }

    // ========== Forward Declarations ==========
    
    void process_thermal_correlations(
        const EDParameters& params,
        uint64_t hilbert_space_dim
    );
    
    Operator load_hamiltonian_from_files(
        const std::string& interaction_file,
        const std::string& single_site_file,
        const std::string& counterterm_file,
        const std::string& three_body_file,
        uint64_t num_sites,
        float spin_length,
        DiagonalizationMethod method,
        HamiltonianFileFormat format
    );
    
    std::function<void(const Complex*, Complex*, int)> create_hamiltonian_apply_function(
        Operator& hamiltonian
    );

    /**
     * @brief Matrix-free diagonalization using a custom apply function
     *
     * This enables truly matrix-free diagonalization where the Hamiltonian
     * is never explicitly stored. The apply_func computes H*v on-the-fly.
     *
     * @param apply_func Function that computes out = H * in
     * @param dim Dimension of the Hilbert space
     * @param method Diagonalization method (LANCZOS recommended)
     * @param params Diagonalization parameters
     * @return EDResults with eigenvalues and optionally eigenvectors
     */
    EDResults diagonalize_matrix_free(
        std::function<void(const Complex*, Complex*, uint64_t)> apply_func,
        uint64_t dim,
        DiagonalizationMethod method,
        const EDParameters& params
    );

    // Phase 9 cleanup: the explicit-block symmetry helpers
    // (``diagonalize_symmetry_block``, ``setup_*_symmetry_basis``,
    // ``find_ground_state_sector*``, ``diagonalize_fixed_sz_sector``,
    // ``transform_sector_to_*``, ``transform_and_save_*``, plus the
    // ``GroundStateSectorInfo``/``SectorInfo``/``SectorResult`` PODs)
    // were removed together with the deprecated public
    // ``exact_diagonalization_*_symmetrized`` entry points. The
    // canonical symmetry path is the streaming kernel in
    // ``ed/core/ed_wrapper_streaming.h`` -- it is matrix-free
    // (no per-sector sparse blocks on disk), GPU-capable per
    // sector, and reachable via
    // ``exact_diagonalization_from_directory(...)`` with
    // ``EDParameters::use_symmetry = true``.
}

// ============================================================================
// MAIN EXACT DIAGONALIZATION FUNCTION
// ============================================================================

/**
 * @brief Main wrapper function for exact diagonalization
 * 
 * @param H Hamiltonian matrix-vector product function
 * @param hilbert_space_dim Dimension of the Hilbert space
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_core(
    std::function<void(const Complex*, Complex*, int)> H, 
    uint64_t hilbert_space_dim,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params_in = EDParameters()
) {
    EDResults results;

    // Phase 6.1: empty params.output_dir historically meant "current working
    // directory" -- underlying solvers (e.g. thick_restart_lanczos,
    // shift_invert_lanczos) would dump eigenvalues.dat / eigenvalues.txt
    // into wherever the process happened to be running, polluting cwd
    // every time the Python dispatcher was used. Mirror the
    // ``output_dir_or_devnull`` convention used by the standalone Python
    // wrappers: empty -> "/dev/null", which the C++ HDF5 layer
    // (``HDF5IO::isDisabledOutputPath``) and the ``ofstream`` writes (via
    // path concatenation that produces a non-openable path) both treat
    // as "skip all I/O". Pass ``"."`` explicitly to restore legacy
    // cwd-dump behaviour.
    EDParameters params = params_in;
    if (params.output_dir.empty()) {
        params.output_dir = "/dev/null";
    }

    // Phase 7: collapse legacy `_GPU` / `_CUDA` / `_MPI` / `_FIXED_SZ`
    // enum variants onto the canonical (base, use_fixed_sz, use_gpu,
    // use_mpi) tuple, then rebuild the legacy "_GPU" enum value from
    // the flag for the existing CPU-vs-GPU dispatcher branches below.
    // This is the single normalization point: every code path beneath
    // this one sees a fully orthogonal ``(method, params)``.
    {
        const auto canon = ed::canonicalize_method_and_flags(
            method, params.use_fixed_sz, params.use_gpu, params.use_mpi);
        params.use_fixed_sz = canon.use_fixed_sz;
        params.use_gpu      = canon.use_gpu;
        params.use_mpi      = canon.use_mpi;
        method = ed::legacy_method_for_dispatch(canon.method, canon.use_gpu);
    }

    // Initialize output directory if needed (skip the disabled sentinel).
    if (!HDF5IO::isDisabledOutputPath(params.output_dir)) {
        std::string cmd = "mkdir -p " + params.output_dir;
        safe_system_call(cmd);
    }

    // Set eigenvectors flag in results
    results.eigenvectors_computed = params.compute_eigenvectors;
    if (params.compute_eigenvectors &&
        !HDF5IO::isDisabledOutputPath(params.output_dir)) {
        results.eigenvectors_path = params.output_dir;
    }
    
    // Call the appropriate diagonalization method
    switch (method) {
        case DiagonalizationMethod::FULL:
            full_diagonalization(H, hilbert_space_dim, params.num_eigenvalues, results.eigenvalues, 
                                 params.output_dir, params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::SCALAPACK:
        case DiagonalizationMethod::SCALAPACK_MIXED:
#ifdef WITH_SCALAPACK
            {
                ScaLAPACKConfig scalapack_config;
                scalapack_config.nprow = params.scalapack_nprow;
                scalapack_config.npcol = params.scalapack_npcol;
                // Phase 8 #5: route the new auto flag through to the
                // solver. When auto is on (the default), the explicit
                // mb/nb values are ignored at solve time and replaced
                // by ``get_optimal_block_size`` -- so we still set them
                // from params for the case where the user disabled auto
                // via the CLI / Python API.
                scalapack_config.mb = params.scalapack_block_size;
                scalapack_config.nb = params.scalapack_block_size;
                scalapack_config.block_size_auto = params.scalapack_block_size_auto;
                // SCALAPACK uses double precision, SCALAPACK_MIXED uses single with refinement
                scalapack_config.use_mixed_precision = (method == DiagonalizationMethod::SCALAPACK_MIXED);
                scalapack_config.refinement_tol = params.scalapack_refinement_tol;
                scalapack_config.max_refinement_iter = params.scalapack_max_refinement_iter;
                scalapack_config.compute_eigenvectors = params.compute_eigenvectors;
                scalapack_config.num_eigenvalues = params.num_eigenvalues;
                scalapack_config.output_dir = params.output_dir;
                scalapack_config.verbose = params.scalapack_verbose;
                
                ScaLAPACKResults scalapack_results = scalapack_diagonalization(H, hilbert_space_dim, scalapack_config);
                results.eigenvalues = std::move(scalapack_results.eigenvalues);
                
                std::cout << "ScaLAPACK completed in " << scalapack_results.total_time << " s" << std::endl;
                std::cout << "  Matrix construction: " << scalapack_results.construction_time << " s" << std::endl;
                std::cout << "  Diagonalization: " << scalapack_results.diagonalization_time << " s" << std::endl;
                if (scalapack_config.use_mixed_precision) {
                    std::cout << "  Refinement: " << scalapack_results.refinement_time << " s" << std::endl;
                    std::cout << "  Max residual: " << scalapack_results.max_residual << std::endl;
                }
            }
#else
            // Graceful fallback to FULL diagonalization when ScaLAPACK is not available
            std::cerr << "Warning: ScaLAPACK not available (build with -DWITH_MPI=ON and ScaLAPACK-compatible BLAS)\n";
            std::cerr << "         Falling back to FULL diagonalization instead.\n\n";
            full_diagonalization(H, hilbert_space_dim, params.num_eigenvalues, results.eigenvalues, 
                                 params.output_dir, params.compute_eigenvectors);
#endif
            break;

        case DiagonalizationMethod::LANCZOS: {
            // Audit follow-up: ARPACK (IRLM, ncv-restart) typically beats a plain
            // Lanczos by 2-4x at our problem sizes -- it's what scipy.sparse.linalg.eigsh
            // and quspin both wrap. Make it opt-in via env var so the default is
            // identical to before, but a single flag (ED_USE_ARPACK_DEFAULT=1)
            // drops the user onto the faster solver for k<=32 ground-state requests.
            //  - ED_USE_ARPACK_DEFAULT=0 (or unset, default): plain in-house lanczos
            //  - ED_USE_ARPACK_DEFAULT=1: route to arpack_ground_state when k<=32
            const char* arpack_env = std::getenv("ED_USE_ARPACK_DEFAULT");
            const bool use_arpack = (arpack_env && arpack_env[0] == '1');
            if (use_arpack && params.num_eigenvalues <= 32) {
                arpack_ground_state(H, hilbert_space_dim,
                                    params.max_iterations, params.num_eigenvalues,
                                    params.tolerance, results.eigenvalues,
                                    params.output_dir, params.compute_eigenvectors);
            } else {
                lanczos(H, hilbert_space_dim, params.max_iterations, params.num_eigenvalues,
                        params.tolerance, results.eigenvalues, params.output_dir,
                        params.compute_eigenvectors);
            }
            break;
        }
            
        case DiagonalizationMethod::LANCZOS_SELECTIVE:
            lanczos_selective_reorth(H, hilbert_space_dim, params.max_iterations, 
                                    params.num_eigenvalues, params.tolerance, 
                                    results.eigenvalues, params.output_dir, 
                                    params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::LANCZOS_NO_ORTHO:
            lanczos_no_ortho(H, hilbert_space_dim, params.max_iterations, 
                           params.num_eigenvalues, params.tolerance, 
                           results.eigenvalues, params.output_dir, 
                           params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::SHIFT_INVERT:
            shift_invert_lanczos(H, hilbert_space_dim, params.max_iterations, 
                                params.num_eigenvalues, params.shift, 
                                params.tolerance, results.eigenvalues, 
                                params.output_dir, params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::DAVIDSON:
            {
                std::vector<ComplexVector> eigenvectors;
                davidson_method(H, hilbert_space_dim, params.max_iterations, 
                             params.max_subspace, params.num_eigenvalues, 
                             params.tolerance, results.eigenvalues, 
                             eigenvectors, params.output_dir);
            }
            break;
            
        case DiagonalizationMethod::LOBPCG:
            lobpcg_diagonalization(H, hilbert_space_dim, params.max_iterations, 
                            params.num_eigenvalues, params.tolerance, 
                            results.eigenvalues, params.output_dir, 
                            params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::KRYLOV_SCHUR:
            krylov_schur(H, hilbert_space_dim, params.max_iterations, 
                       params.num_eigenvalues, params.tolerance, 
                       results.eigenvalues, params.output_dir, 
                       params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::BLOCK_KRYLOV_SCHUR:
            block_krylov_schur(H, hilbert_space_dim, params.max_iterations,
                              params.num_eigenvalues, params.block_size, params.tolerance,
                              results.eigenvalues, params.output_dir,
                              params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::IMPLICIT_RESTART_LANCZOS:
            implicitly_restarted_lanczos(H, hilbert_space_dim, params.max_iterations, 
                                       params.num_eigenvalues, params.tolerance, 
                                       results.eigenvalues, params.output_dir, 
                                       params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::THICK_RESTART_LANCZOS:
            thick_restart_lanczos(H, hilbert_space_dim, params.max_iterations, 
                                params.num_eigenvalues, params.tolerance, 
                                results.eigenvalues, params.output_dir, 
                                params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::OSS:
            optimal_spectrum_solver(
                H, hilbert_space_dim, params.max_iterations,
                results.eigenvalues, params.output_dir, 
                params.compute_eigenvectors
            );
            break;
            
        case DiagonalizationMethod::mTPQ:
            microcanonical_tpq(H, hilbert_space_dim,
                            params.max_iterations, params.num_samples,
                            params.tpq_measurement_interval,
                            results.eigenvalues,
                            params.output_dir,
                            params.compute_eigenvectors,
                            params.tpq_energy_shift,
                            params.save_thermal_states, params.observables, params.observable_names,
                            params.omega_min, params.omega_max,
                            params.num_points, params.t_end, params.dt, params.spin_length, 
                            params.compute_spin_correlations, params.sublattice_size, params.num_sites,
                            params.fixed_sz_op,
                            params.tpq_continue,
                            params.tpq_continue_sample,
                            params.tpq_continue_beta,
                            params.tpq_target_beta,
                            params.tpq_num_measure_points,
                            params.tpq_measure_beta_min,
                            params.tpq_measure_beta_max); 
            break;

        case DiagonalizationMethod::cTPQ:
            canonical_tpq(
                H,                      // Hamiltonian matvec
                hilbert_space_dim,      // N
                params.temp_max,        // beta_max (use configured max inverse temperature)
                params.num_samples,     // num_samples
                params.tpq_measurement_interval, // temp_interval / measurement frequency
                results.eigenvalues,    // energies output vector
                params.output_dir,      // output dir
                params.tpq_delta_beta,  // delta_beta (imaginary-time step)
                params.tpq_taylor_order, // taylor_order
                params.save_thermal_states, // compute_observables
                params.observables,     // observables
                params.observable_names,// observable names
                params.omega_min,       // omega_min
                params.omega_max,       // omega_max
                params.num_points,      // num_points
                params.t_end,           // t_end
                params.dt,              // dt
                params.spin_length,     // spin length
                params.compute_spin_correlations, // measure Sz and fluctuations
                params.sublattice_size, // sublattice size
                params.num_sites,       // number of sites
                params.fixed_sz_op,      // Fixed-Sz operator for embedding
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max
            );
            break;


        // Note: DiagonalizationMethod::mTPQ_CUDA used to fall through here
        // as a no-op (silently returning empty EDResults). Phase 7 collapses
        // mTPQ_CUDA -> mTPQ + use_gpu=true via canonicalize_method_and_flags,
        // so it now reaches the GPU-error branch below (`case mTPQ_GPU:`)
        // and produces a clear "GPU methods must be called via
        // exact_diagonalization_from_files" diagnostic instead.

        case DiagonalizationMethod::BLOCK_LANCZOS:
            block_lanczos(H, hilbert_space_dim, 
                        params.max_iterations, params.num_eigenvalues, params.block_size, 
                        params.tolerance, results.eigenvalues, 
                        params.output_dir, params.compute_eigenvectors);
            break;
            
        case DiagonalizationMethod::CHEBYSHEV_FILTERED:
            chebyshev_filtered_lanczos(H, hilbert_space_dim, 
                                     params.max_iterations, params.num_eigenvalues, 
                                     params.tolerance, results.eigenvalues, 
                                     params.output_dir, params.compute_eigenvectors,
                                     params.target_lower, params.target_upper);
            break;
            
        
        case DiagonalizationMethod::ARPACK_SM:
            arpack_ground_state(H, hilbert_space_dim,
                                params.max_iterations, params.num_eigenvalues, params.tolerance,
                                results.eigenvalues, params.output_dir, params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::ARPACK_LM:
            arpack_largest(H, hilbert_space_dim,
                            params.max_iterations, params.num_eigenvalues, params.tolerance,
                            results.eigenvalues, params.output_dir, params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::ARPACK_SHIFT_INVERT:
            arpack_shift_invert(H, hilbert_space_dim,
                                params.max_iterations, params.num_eigenvalues, params.tolerance,
                                params.shift,
                                results.eigenvalues, params.output_dir,
                                params.compute_eigenvectors);
            break;

        case DiagonalizationMethod::ARPACK_ADVANCED: {
            detail_arpack::ArpackAdvancedOptions opts;
            opts.nev = params.num_eigenvalues;
            opts.which = params.arpack_which;
            opts.tol = params.tolerance;
            opts.max_iter = params.max_iterations;
            opts.ncv = params.arpack_ncv;
            opts.auto_enlarge_ncv = params.arpack_auto_enlarge_ncv;
            opts.max_restarts = params.arpack_max_restarts;
            opts.ncv_growth = params.arpack_ncv_growth;
            opts.two_phase_refine = params.arpack_two_phase_refine;
            opts.relaxed_tol = params.arpack_relaxed_tol;
            opts.shift_invert = params.arpack_shift_invert;
            opts.sigma = params.arpack_sigma;
            opts.auto_switch_to_shift_invert = params.arpack_auto_switch_shift_invert;
            opts.switch_sigma = params.arpack_switch_sigma;
            opts.adaptive_inner_tol = params.arpack_adaptive_inner_tol;
            opts.inner_tol_factor = params.arpack_inner_tol_factor;
            opts.inner_tol_min = params.arpack_inner_tol_min;
            opts.inner_max_iter = params.arpack_inner_max_iter;
            opts.verbose = params.arpack_advanced_verbose;
            std::vector<Complex> evecs; // optionally capture
            uint64_t info = arpack_eigs_advanced(H, hilbert_space_dim, opts,
                                            results.eigenvalues,
                                            params.output_dir,
                                            params.compute_eigenvectors,
                                            params.compute_eigenvectors ? &evecs : nullptr);
            if (info != 0) {
                std::cerr << "ARPACK advanced solver returned info=" << info << std::endl;
            }
            break; }
        
        // Methods not yet fully implemented
        case DiagonalizationMethod::SHIFT_INVERT_ROBUST:
            std::cerr << "SHIFT_INVERT_ROBUST not yet implemented. Using standard SHIFT_INVERT instead." << std::endl;
            shift_invert_lanczos(H, hilbert_space_dim, params.max_iterations, 
                                params.num_eigenvalues, params.shift, 
                                params.tolerance, results.eigenvalues, 
                                params.output_dir, params.compute_eigenvectors);
            break;
        
        case DiagonalizationMethod::mTPQ_MPI:
            std::cerr << "Error: mTPQ_MPI requires MPI build. Use standard mTPQ instead." << std::endl;
            throw std::runtime_error("mTPQ_MPI not available");
            break;
        
        case DiagonalizationMethod::FTLM:
            {
                // Setup FTLM parameters
                FTLMParameters ftlm_params;
                ftlm_params.krylov_dim = params.ftlm_krylov_dim;
                ftlm_params.num_samples = params.num_samples;
                ftlm_params.max_iterations = params.max_iterations;
                ftlm_params.tolerance = params.tolerance;
                ftlm_params.full_reorthogonalization = params.ftlm_full_reorth;
                ftlm_params.reorth_frequency = params.ftlm_reorth_freq;
                ftlm_params.random_seed = params.ftlm_seed;
                ftlm_params.store_intermediate = params.ftlm_store_samples;
                ftlm_params.compute_error_bars = params.ftlm_error_bars;
                
                // Run FTLM
                FTLMResults ftlm_results = finite_temperature_lanczos(
                    H, hilbert_space_dim, ftlm_params,
                    params.temp_min, params.temp_max, params.num_temp_bins,
                    params.output_dir
                );
                
                // Store results (both thermodynamics and full FTLM data)
                results.thermo_data = ftlm_results.thermo_data;
                results.ftlm_results = ftlm_results;  // Store for sector combination
                
                // Store ground state estimate as eigenvalue
                if (ftlm_results.ground_state_estimate != 0.0) {
                    results.eigenvalues.push_back(ftlm_results.ground_state_estimate);
                }
                
                // Save FTLM results to file (HDF5 goes to main output dir)
                if (!params.output_dir.empty()) {
                    safe_system_call("mkdir -p " + params.output_dir);
                    save_ftlm_results(ftlm_results, params.output_dir + "/ftlm_thermo.txt");
                }
            }
            break;
        
        case DiagonalizationMethod::LTLM:
            {
                // The `use_hybrid_method` legacy flag was removed in
                // matvec-unification Phase 7.3 (it had been
                // [[deprecated]] for several releases and only emitted
                // a warning + fell back to standard LTLM). Use
                // DiagonalizationMethod::HYBRID instead.

                // Standard LTLM
                // Setup LTLM parameters
                LTLMParameters ltlm_params;
                ltlm_params.krylov_dim = params.ltlm_krylov_dim;
                ltlm_params.ground_state_krylov = params.ltlm_ground_krylov;
                ltlm_params.num_samples = 1;  // LTLM typically uses 1 sample
                ltlm_params.max_iterations = params.max_iterations;
                ltlm_params.tolerance = params.tolerance;
                ltlm_params.full_reorthogonalization = params.ltlm_full_reorth;
                ltlm_params.reorth_frequency = params.ltlm_reorth_freq;
                ltlm_params.random_seed = params.ltlm_seed;
                ltlm_params.store_intermediate = params.ltlm_store_data;
                ltlm_params.compute_error_bars = false;  // Not needed for single sample
                
                // Run LTLM
                LTLMResults ltlm_results = low_temperature_lanczos(
                    H, hilbert_space_dim, ltlm_params,
                    params.temp_min, params.temp_max, params.num_temp_bins,
                    nullptr, params.output_dir
                );
                
                // Store results
                results.thermo_data = ltlm_results.thermo_data;
                results.eigenvalues.push_back(ltlm_results.ground_state_energy);
                
                // Save LTLM results to file (HDF5 goes to main output dir)
                if (!params.output_dir.empty()) {
                    safe_system_call("mkdir -p " + params.output_dir);
                    save_ltlm_results(ltlm_results, params.output_dir + "/ltlm_thermo.txt");
                }
            }
            break;
        
        case DiagonalizationMethod::HYBRID:
            {
                // Setup Hybrid Thermal parameters
                HybridThermalParameters hybrid_params;
                
                // Temperature and crossover settings
                hybrid_params.crossover_temperature = params.hybrid_crossover;
                hybrid_params.auto_crossover = params.hybrid_auto_crossover;
                
                // LTLM parameters (low temperature)
                hybrid_params.ltlm_krylov_dim = params.ltlm_krylov_dim;
                hybrid_params.ltlm_ground_krylov = params.ltlm_ground_krylov;
                hybrid_params.ltlm_full_reorth = params.ltlm_full_reorth;
                hybrid_params.ltlm_reorth_freq = params.ltlm_reorth_freq;
                hybrid_params.ltlm_seed = params.ltlm_seed;
                hybrid_params.ltlm_store_data = params.ltlm_store_data;
                
                // FTLM parameters (high temperature)
                hybrid_params.ftlm_num_samples = params.num_samples;
                hybrid_params.ftlm_krylov_dim = params.ftlm_krylov_dim;
                hybrid_params.ftlm_full_reorth = params.ftlm_full_reorth;
                hybrid_params.ftlm_reorth_freq = params.ftlm_reorth_freq;
                hybrid_params.ftlm_seed = params.ftlm_seed;
                hybrid_params.ftlm_store_samples = params.ftlm_store_samples;
                hybrid_params.ftlm_error_bars = params.ftlm_error_bars;
                
                // General parameters
                hybrid_params.max_iterations = params.max_iterations;
                hybrid_params.tolerance = params.tolerance;
                
                // Run hybrid thermal method
                HybridThermalResults hybrid_results = hybrid_thermal_method(
                    H, hilbert_space_dim, hybrid_params,
                    params.temp_min, params.temp_max, params.num_temp_bins,
                    params.output_dir
                );
                
                // Store results
                results.thermo_data = hybrid_results.thermo_data;
                results.eigenvalues.push_back(hybrid_results.ground_state_energy);
                
                // Save results to file (HDF5 goes to main output dir)
                if (!params.output_dir.empty()) {
                    safe_system_call("mkdir -p " + params.output_dir);
                    save_hybrid_thermal_results(hybrid_results, params.output_dir + "/hybrid_thermo.txt");
                }
            }
            break;

        case DiagonalizationMethod::KPM_DOS:
            {
                // Build temperature grid (linear in T, matching what FTLM/LTLM do)
                std::vector<double> betas;
                std::vector<double> temps;
                betas.reserve(params.num_temp_bins);
                temps.reserve(params.num_temp_bins);
                if (params.num_temp_bins == 1) {
                    temps.push_back(params.temp_min);
                    betas.push_back(1.0 / std::max(params.temp_min, 1e-300));
                } else {
                    double dT = (params.temp_max - params.temp_min) /
                                static_cast<double>(params.num_temp_bins - 1);
                    for (uint64_t i = 0; i < params.num_temp_bins; ++i) {
                        double T = params.temp_min + dT * static_cast<double>(i);
                        temps.push_back(T);
                        betas.push_back(1.0 / std::max(T, 1e-300));
                    }
                }

                ed::kpm_dos::KPMDOSParameters kpm_params;
                kpm_params.num_moments              = params.kpm_num_moments;
                kpm_params.num_random_vectors       = params.kpm_num_random_vectors;
                kpm_params.num_quadrature_nodes     = params.kpm_num_quadrature_nodes;
                kpm_params.spectral_bounds_krylov   = params.kpm_spectral_bounds_krylov;
                kpm_params.spectral_bound_buffer    = params.kpm_spectral_bound_buffer;
                kpm_params.use_jackson_kernel       = params.kpm_use_jackson_kernel;
                kpm_params.lorentz_lambda           = params.kpm_lorentz_lambda;
                kpm_params.tolerance                = params.tolerance;
                kpm_params.full_reorthogonalization = params.kpm_full_reorth;
                kpm_params.reorth_frequency         = params.kpm_reorth_freq;
                kpm_params.random_seed              = params.kpm_seed;

                ed::kpm_dos::KPMDOSResult kpm_res =
                    ed::kpm_dos::compute_kpm_dos(H, hilbert_space_dim, betas,
                                                 /*dos_energies=*/{}, kpm_params);

                results.thermo_data.temperatures   = temps;
                results.thermo_data.energy         = kpm_res.energy;
                results.thermo_data.specific_heat  = kpm_res.specific_heat;
                results.thermo_data.entropy        = kpm_res.entropy;
                results.thermo_data.free_energy    = kpm_res.free_energy;
                // Provide e_min so the resummation pipeline has the same
                // ground-state reference convention as FTLM/LTLM.
                results.thermo_data.e_min          = kpm_res.e_min_estimate;
                results.eigenvalues.push_back(kpm_res.e_min_estimate);

                // Persist /thermodynamics/{temperatures,energy,specific_heat,
                // entropy,free_energy} so the NLCE summation kernel
                // (NLC_sum_ftlm.py) finds the same dataset layout it reads
                // for FTLM.
                if (!params.output_dir.empty()) {
                    safe_system_call("mkdir -p " + params.output_dir);
                    try {
                        std::string h5_file = HDF5IO::createOrOpenFile(params.output_dir);
                        HDF5IO::saveThermodynamics(h5_file, temps, "energy",        kpm_res.energy);
                        HDF5IO::saveThermodynamics(h5_file, temps, "specific_heat", kpm_res.specific_heat);
                        HDF5IO::saveThermodynamics(h5_file, temps, "entropy",       kpm_res.entropy);
                        HDF5IO::saveThermodynamics(h5_file, temps, "free_energy",   kpm_res.free_energy);
                    } catch (const std::exception& e) {
                        std::cerr << "  KPM-DOS: failed to write thermodynamics HDF5: "
                                  << e.what() << std::endl;
                    }
                }
            }
            break;

        case DiagonalizationMethod::LANCZOS_GPU:
        case DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ:
        case DiagonalizationMethod::DAVIDSON_GPU:
        case DiagonalizationMethod::LOBPCG_GPU:
        case DiagonalizationMethod::KRYLOV_SCHUR_GPU:
        case DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU:
        case DiagonalizationMethod::BLOCK_LANCZOS_GPU:
        case DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ:
        case DiagonalizationMethod::mTPQ_GPU:
        case DiagonalizationMethod::cTPQ_GPU:
        case DiagonalizationMethod::FTLM_GPU:
        case DiagonalizationMethod::FTLM_GPU_FIXED_SZ:
            // These should be handled in exact_diagonalization_from_files
            // If we reach here, it means they were called incorrectly
            std::cerr << "Error: GPU methods must be called via exact_diagonalization_from_files" << std::endl;
            std::cerr << "Use: ED <directory> --method=<GPU_METHOD>" << std::endl;
            throw std::runtime_error("GPU methods require file-based interface");
            break;

        default:
            // Hard-fail on an unrecognised method instead of silently
            // returning empty results: in non-interactive runs the cerr
            // line was getting buried and downstream code would happily
            // operate on an empty EDResults.
            throw std::runtime_error(
                "exact_diagonalization_core: unknown DiagonalizationMethod "
                "(enum value " + std::to_string(static_cast<int>(method)) +
                "). Either the enum was extended without updating the "
                "dispatcher or the caller is passing an uninitialised method.");
    }

    if (params.compute_eigenvectors) {
        std::cout << "Eigenvectors computed and saved to " << params.output_dir << std::endl;
    }

    // Calculate thermal observables if requested
    // if (params.calc_observables) {
    //     ed_internal::process_thermal_correlations(params, hilbert_space_dim);
    // }

    return results;
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

namespace ed_internal {

/**
 * @brief Process thermal correlation functions
 * 
 * Searches for correlation files, loads operators, and calculates
 * thermal expectation values at different temperatures.
 * 
 * @param params ED parameters containing temperature ranges and observables
 * @param hilbert_space_dim Dimension of the Hilbert space
 */
inline void process_thermal_correlations(
    const EDParameters& params,
    uint64_t hilbert_space_dim
) {
    std::cout << "Calculating custom observables..." << std::endl;
    std::cout << "Calculating thermal expectation values for correlation operators..." << std::endl;

    // Create output directory for thermal correlation results
    std::string output_correlations_dir = params.output_dir + "/thermal_correlations";
    std::string cmd_mkdir = "mkdir -p " + output_correlations_dir;
    safe_system_call(cmd_mkdir);

    // Determine base directory where correlation files might be located
    std::string base_dir;
    if (!params.output_dir.empty()) {
        size_t pos = params.output_dir.find_last_of("/\\");
        base_dir = (pos != std::string::npos) ? params.output_dir.substr(0, pos) : ".";
    } else {
        base_dir = ".";
    }

    std::cout << "Looking for correlation files in: " << base_dir << std::endl;

    // Define correlation file patterns to search for
    std::vector<std::pair<std::string, std::string>> patterns = {
        {"one_body_correlations", "one_body_correlations*.dat"},
        {"two_body_correlations", "two_body_correlations*.dat"}
    };

    // Process each type of correlation file
    for (const auto& [prefix, pattern] : patterns) {
            // Find matching files
            std::string temp_list_file = output_correlations_dir + "/" + prefix + "_files.txt";
            std::string find_command = "find \"" + base_dir + "\" -name \"" + pattern + "\" 2>/dev/null > \"" + temp_list_file + "\"";
            safe_system_call(find_command);
            
            // Read the list of files
            std::ifstream file_list(temp_list_file);
            if (!file_list.is_open()) continue;
            
            std::string correlation_file;
            uint64_t file_count = 0;

            // Compute thermal expectations at different temperatures
            std::string results_file_path = output_correlations_dir + "/thermal_expectation_" + 
            prefix + ".dat";
            std::ofstream results_file(results_file_path);
            
            if (!results_file.is_open()) {
                std::cerr << "Error: Could not open output file: " << results_file_path << std::endl;
                continue;
            }
                        
            while (std::getline(file_list, correlation_file)) {
                if (correlation_file.empty()) continue;
                file_count++;
                
                // Extract operator type from filename
                size_t prefix_pos = correlation_file.find(prefix);
                if (prefix_pos == std::string::npos) continue;
                
                
                std::cout << "Processing " << prefix << " file: " << correlation_file << std::endl;
                
                try {
                    // Load the operator
                    

                    std::ifstream file(correlation_file);
                    if (!file.is_open()) {
                        throw std::runtime_error("Could not open file: " + correlation_file);
                    }
                    std::cout << "Reading file: " << correlation_file << std::endl;
                    std::string line;
                    
                    // Skip the first line (header)
                    std::getline(file, line);
                    
                    // Read the number of lines
            
                    std::getline(file, line);
                    std::istringstream iss(line);
                    uint64_t numLines;
                    std::string m;
                    iss >> m >> numLines;
                    // std::cout << "Number of lines: " << numLines << std::endl;
                    
                    // Skip the next 3 lines (separators/headers)
                    for (uint64_t i = 0; i < 3; ++i) {
                        std::getline(file, line);
                    }
                                            
                    if (prefix == "one_body_correlations") {
                        results_file << std::setw(12) << "Temperatures" << " "
                                    << std::setw(12) << "Beta" << " "
                                    << std::setw(12) << "Op1" << " "
                                    << std::setw(12) << "Index1" << " "
                                    << std::setw(12) << "Expectation" << std::endl;
                    } else if (prefix == "two_body_correlations") {
                        results_file << std::setw(12) << "Temperatures" << " "
                                    << std::setw(12) << "Beta" << " "
                                    << std::setw(12) << "Op1" << " "
                                    << std::setw(12) << "Op2" << " "
                                    << std::setw(12) << "Index1" << " "
                                    << std::setw(12) << "Index2" << " "
                                    << std::setw(12) << "Expectation" << std::endl;
                    }

                    // Process transform data
                    uint64_t lineCount = 0;
                    while (std::getline(file, line) && lineCount < numLines) {
                        Operator correlation_op(params.num_sites, params.spin_length);
                        std::istringstream lineStream(line);
                        uint64_t Op1, indx1, Op2, indx2;
                        double E, F;
                        if (prefix == "one_body_correlations") {

                            // std::cout << "Reading line: " << line << std::endl;
                            if (!(lineStream >> Op1 >> indx1 >> E >> F)) {
                                continue; // Skip invalid lines
                            }

                            correlation_op.loadonebodycorrelation(Op1, indx1);
                        } else if (prefix == "two_body_correlations") {

                            // std::cout << "Reading line: " << line << std::endl;
                            if (!(lineStream >> Op1 >> indx1 >> Op2 >> indx2 >> E >> F)) {
                                continue; // Skip invalid lines
                            }
                            correlation_op.loadtwobodycorrelation(Op1, indx1, Op2, indx2);
                        }

                        // Create a lambda to apply the operator
                        auto apply_correlation_op = [&correlation_op](const Complex* in, Complex* out, uint64_t n) {
                            correlation_op.apply(in, out, n);
                        };
                        

                        // Calculate thermal expectations at temperature points
                        uint64_t num_temps = std::min(params.num_temp_bins, static_cast<uint64_t>(20));
                        double log_temp_min = std::log(params.temp_min);
                        double log_temp_max = std::log(params.temp_max);
                        double log_temp_step = (log_temp_max - log_temp_min) / std::max(1, static_cast<int>(num_temps - 1));

                        for (uint64_t i = 0; i < num_temps; i++) {
                            double T = std::exp(log_temp_min + i * log_temp_step);
                            double beta = 1.0 / T;
                            
                            // TODO: Fix thermal expectation calculation - 
                            // compute_thermal_expectation_value has a different signature
                            // and requires StaticResponseParameters. This code path is broken.
                            Complex expectation(0.0, 0.0);
                            std::cerr << "Warning: Thermal expectation calculation not implemented for correlation operators" << std::endl;
                            
                            /*
                            // Old broken code:
                            Complex expectation = calculate_thermal_expectation(
                                apply_correlation_op, hilbert_space_dim, beta, params.output_dir + "/eigenvectors/");
                            */
                            
                            std::cout << "T: " << T << ", beta: " << beta << ", expectation: " 
                                        << expectation.real() << " + " << expectation.imag() << "i" << std::endl;

                            // Write to file
                            if (prefix == "one_body_correlations") {
                                results_file << std::setw(12) << std::setprecision(6) << T << " "
                                            << std::setw(12) << std::setprecision(6) << beta << " "
                                            << std::setw(12) << std::setprecision(6) << Op1 << " "
                                            << std::setw(12) << std::setprecision(6) << indx1 << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.real() << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.imag() << std::endl;
                            } else if (prefix == "two_body_correlations") {
                                results_file << std::setw(12) << std::setprecision(6) << T << " "
                                            << std::setw(12) << std::setprecision(6) << beta << " "
                                            << std::setw(12) << std::setprecision(6) << Op1 << " "
                                            << std::setw(12) << std::setprecision(6) << Op2 << " "
                                            << std::setw(12) << std::setprecision(6) << indx1 << " "
                                            << std::setw(12) << std::setprecision(6) << indx2 << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.real() << " "
                                            << std::setw(12) << std::setprecision(6) << expectation.imag() << std::endl;
                            }
                        }
                    }
                }
                catch (const std::exception& e) {
                    std::cerr << "Error processing " << correlation_file << ": " << e.what() << std::endl;
                }
            }
            results_file.close();
            std::cout << "Thermal expectations saved to: " << results_file_path << std::endl;
            file_list.close();
            std::cout << "Processed " << file_count << " " << prefix << " files" << std::endl;
            std::remove(temp_list_file.c_str());
        }

    std::cout << "Thermal expectation calculations complete!" << std::endl;
}

/**
 * @brief Factory function to create the appropriate operator type
 * Creates either a standard Operator or FixedSzOperator based on config
 */
template<typename OperatorType = Operator>
OperatorType* create_operator(const SystemConfig& config) {
    if (config.use_fixed_sz) {
        int64_t n_up = (config.n_up >= 0) ? config.n_up : config.num_sites / 2;
        return new FixedSzOperator(config.num_sites, config.spin_length, n_up);
    } else {
        return new OperatorType(config.num_sites, config.spin_length);
    }
}

/**
 * @brief Load Hamiltonian from files based on format
 */
inline Operator load_hamiltonian_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file,
    const std::string& counterterm_file,
    const std::string& three_body_file,
    uint64_t num_sites,
    float spin_length,
    DiagonalizationMethod method,
    HamiltonianFileFormat format
) {
    Operator hamiltonian(num_sites, spin_length);
    
    switch (format) {
        case HamiltonianFileFormat::STANDARD:
            // Load terms from files
            if (!single_site_file.empty()) {
                hamiltonian.loadFromFile(single_site_file);
            }
            if (!interaction_file.empty()) {
                hamiltonian.loadFromInterAllFile(interaction_file);
            }
            // Load three-body terms if provided
            if (!three_body_file.empty() && std::filesystem::exists(three_body_file)) {
                std::cout << "Loading three-body terms from: " << three_body_file << std::endl;
                hamiltonian.loadThreeBodyTerm(three_body_file);
            }
            // COUNTERTERM DISABLED
            // if (!counterterm_file.empty()){
            //     hamiltonian.loadCounterTerm(counterterm_file);
            // }
            // Build sparse matrix (except for full diagonalization)
            // if (method == DiagonalizationMethod::FULL) {
            //     hamiltonian.buildSparseMatrix();
            // }
            break;
            
        case HamiltonianFileFormat::SPARSE_MATRIX:
            throw std::runtime_error("Sparse matrix format not yet implemented");
            
        case HamiltonianFileFormat::CUSTOM:
            throw std::runtime_error("Custom format requires a parser function");
            
        default:
            throw std::runtime_error("Unknown Hamiltonian file format");
    }
    
    return hamiltonian;
}

/**
 * @brief Create a lambda function to apply the Hamiltonian
 */
inline std::function<void(const Complex*, Complex*, int)> create_hamiltonian_apply_function(
    Operator& hamiltonian
) {
    return [&hamiltonian](const Complex* in, Complex* out, uint64_t n) {
        // Directly use pointer-based apply to avoid temporary vector allocations
        hamiltonian.apply(in, out, n);
    };
}

/**
 * @brief Matrix-free diagonalization using a custom apply function
 * 
 * This enables truly matrix-free diagonalization where the Hamiltonian
 * is never explicitly stored. The apply_func computes H*v on-the-fly.
 * 
 * Supports Lanczos and Davidson methods. ARPACK and FULL require matrix construction.
 */
inline EDResults diagonalize_matrix_free(
    std::function<void(const Complex*, Complex*, uint64_t)> apply_func,
    uint64_t dim,
    DiagonalizationMethod method,
    const EDParameters& params
) {
    // Validate method - only Lanczos and Davidson support matrix-free
    if (method == DiagonalizationMethod::FULL) {
        throw std::runtime_error("FULL diagonalization requires explicit matrix. Use LANCZOS for matrix-free.");
    }
    if (method == DiagonalizationMethod::ARPACK_SM ||
        method == DiagonalizationMethod::ARPACK_LM) {
        std::cerr << "Warning: ARPACK requires matrix construction. Falling back to LANCZOS." << std::endl;
        method = DiagonalizationMethod::LANCZOS;
    }
    
    // Wrap the apply function to match expected signature
    auto apply_wrapper = [&apply_func](const Complex* in, Complex* out, int size) {
        apply_func(in, out, static_cast<uint64_t>(size));
    };
    
    return exact_diagonalization_core(apply_wrapper, dim, method, params);
}

} // namespace ed_internal

// ============================================================================
// Sz CONSERVATION CHECKER
// ============================================================================

/**
 * @brief Check if a Hamiltonian conserves total Sz by scanning its term files.
 *
 * For spin-1/2 in the ladder basis (0=S+, 1=S-, 2=Sz):
 *   - Two-body (InterAll.dat): Sz is conserved iff every (Op_i, Op_j) pair is
 *     one of (0,1), (1,0), or (2,2)  — i.e. S+S-, S-S+, SzSz.
 *   - Single-site (Trans.dat):  Sz is conserved iff every Op is 2 (Sz).
 *     An on-site S+ or S- (transverse field) breaks conservation.
 *
 * @return true if the Hamiltonian conserves total Sz
 */
inline bool hamiltonian_conserves_sz(const std::string& interaction_file,
                                     const std::string& single_site_file) {
    // --- Check two-body terms (InterAll.dat) ---
    if (!interaction_file.empty()) {
        std::ifstream file(interaction_file);
        if (file.is_open()) {
            std::string line;
            // Skip header: 3 separator/num lines + 3 more lines (matches loadFromInterAllFile)
            for (int i = 0; i < 5; ++i) std::getline(file, line);

            while (std::getline(file, line)) {
                std::istringstream iss(line);
                uint64_t op_i, site_i, op_j, site_j;
                double re, im;
                if (!(iss >> op_i >> site_i >> op_j >> site_j >> re >> im)) continue;
                if (std::abs(re) < 1e-15 && std::abs(im) < 1e-15) continue;

                // Allowed Sz-conserving pairs: (S+,S-)=(0,1), (S-,S+)=(1,0), (Sz,Sz)=(2,2)
                bool ok = (op_i == 0 && op_j == 1) ||
                          (op_i == 1 && op_j == 0) ||
                          (op_i == 2 && op_j == 2);
                if (!ok) {
                    std::cout << "[Sz check] Non-conserving two-body term: Op("
                              << op_i << "," << op_j << ") on sites ("
                              << site_i << "," << site_j << ") with coeff ("
                              << re << "," << im << ")" << std::endl;
                    return false;
                }
            }
        }
    }

    // --- Check single-site terms (Trans.dat) ---
    if (!single_site_file.empty()) {
        std::ifstream file(single_site_file);
        if (file.is_open()) {
            std::string line;
            // Skip header (matches loadFromFile): 1 separator + "num N" + 3 separator lines
            for (int i = 0; i < 5; ++i) std::getline(file, line);

            while (std::getline(file, line)) {
                std::istringstream iss(line);
                uint64_t op, site;
                double re, im;
                if (!(iss >> op >> site >> re >> im)) continue;
                if (std::abs(re) < 1e-15 && std::abs(im) < 1e-15) continue;

                // Only Sz (op=2) preserves total Sz
                if (op != 2) {
                    std::cout << "[Sz check] Non-conserving single-site term: Op("
                              << op << ") on site " << site << " with coeff ("
                              << re << "," << im << ")" << std::endl;
                    return false;
                }
            }
        }
    }

    return true;
}

// ============================================================================
// FULL DIAGONALIZATION VIA Sz SECTOR SPLITTING
// ============================================================================

/**
 * @brief Full diagonalization by splitting into Sz sectors
 * 
 * Instead of diagonalizing the full 2^N Hilbert space at once, this function:
 * 1. Loops over all Sz sectors (n_up = 0, 1, ..., num_sites)
 * 2. Builds a FixedSzOperator per sector
 * 3. Constructs the dense Hamiltonian using the sparse-to-dense conversion
 *    (via buildFixedSzMatrix() + toDense()), which is far more efficient than
 *    N column-by-column SpMV applications
 * 4. Diagonalizes each sector independently with LAPACKE_zheevd
 * 5. Collects and sorts all eigenvalues
 * 6. Saves the merged result as a single HDF5 file (transparent to downstream code)
 * 
 * Memory advantage: The largest sector has dimension C(N, N/2), e.g.:
 *   N=15: 2^15 = 32768 -> C(15,7) = 6435  (5.1x reduction, 26x less memory)
 *   N=17: 2^17 = 131072 -> C(17,8) = 24310 (5.4x reduction, 29x less memory)
 *   N=19: 2^19 = 524288 -> C(19,9) = 92378 (5.7x reduction, 32x less memory)
 * 
 * IMPORTANT: This assumes Sz is a good quantum number (total Sz commutes with H).
 * Models that break Sz conservation (e.g., anisotropic J±±, Jz±, in-plane field)
 * will produce INCORRECT results. The caller must verify Sz conservation.
 * 
 * @param interaction_file Path to interaction file (InterAll.dat)
 * @param single_site_file Path to single-site file (Trans.dat)
 * @param num_sites Number of spin sites
 * @param spin_length Spin length (usually 0.5)
 * @param params Parameters for diagonalization
 * @return EDResults containing ALL eigenvalues from ALL sectors, sorted
 */
inline EDResults exact_diagonalization_all_sz_sectors(
    const std::string& interaction_file,
    const std::string& single_site_file,
    uint64_t num_sites,
    float spin_length,
    const EDParameters& params
) {
    EDResults results;
    
    uint64_t full_dim = 1ULL << num_sites;
    uint64_t num_sectors = num_sites + 1;  // n_up = 0, 1, ..., num_sites
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "FULL DIAGONALIZATION VIA Sz SECTOR SPLITTING" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Full Hilbert space dimension: " << full_dim << std::endl;
    std::cout << "Number of Sz sectors: " << num_sectors << std::endl;
    
    // Pre-compute sector dimensions for progress reporting
    std::vector<uint64_t> sector_dims(num_sectors);
    uint64_t largest_sector_dim = 0;
    uint64_t total_sector_dim = 0;
    for (uint64_t n_up = 0; n_up <= num_sites; ++n_up) {
        auto basis = generateFixedSzBasis(num_sites, n_up);
        sector_dims[n_up] = basis.size();
        total_sector_dim += sector_dims[n_up];
        if (sector_dims[n_up] > largest_sector_dim) {
            largest_sector_dim = sector_dims[n_up];
        }
    }
    
    std::cout << "Largest sector dimension: " << largest_sector_dim 
              << " (reduction: " << std::fixed << std::setprecision(1) 
              << (double)full_dim / largest_sector_dim << "x)" 
              << std::defaultfloat << std::endl;
    std::cout << "Memory for full matrix:    " 
              << (double)full_dim * full_dim * sizeof(Complex) / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    std::cout << "Memory for largest sector: " 
              << (double)largest_sector_dim * largest_sector_dim * sizeof(Complex) / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    
    // Collect all eigenvalues from all sectors
    std::vector<double> all_eigenvalues;
    all_eigenvalues.reserve(full_dim);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (uint64_t n_up = 0; n_up <= num_sites; ++n_up) {
        uint64_t dim = sector_dims[n_up];
        double sz_value = spin_length * num_sites - n_up;
        
        std::cout << "\n--- Sector " << n_up + 1 << "/" << num_sectors 
                  << ": n_up=" << n_up << ", Sz=" << sz_value
                  << ", dim=" << dim << " ---" << std::endl;
        
        if (dim == 0) {
            std::cout << "  Empty sector, skipping." << std::endl;
            continue;
        }
        
        auto sector_start = std::chrono::high_resolution_clock::now();
        
        // Create FixedSzOperator for this sector
        FixedSzOperator op(num_sites, spin_length, n_up);
        
        // Load Hamiltonian terms
        if (!single_site_file.empty()) {
            std::ifstream file(single_site_file);
            if (file.is_open()) {
                op.loadFromFile(single_site_file);
            }
        }
        if (!interaction_file.empty()) {
            std::ifstream file(interaction_file);
            if (file.is_open()) {
                op.loadFromInterAllFile(interaction_file);
            }
        }
        
        // Build the sparse matrix, then convert to dense for LAPACK
        // This is O(nnz) — much faster than dim SpMV calls which is O(dim * nnz)
        op.buildFixedSzMatrix();
        auto sparse_matrix = op.getFixedSzMatrix();
        
        // Convert sparse -> dense (column-major for LAPACK)
        Eigen::MatrixXcd dense = Eigen::MatrixXcd(sparse_matrix);
        
        // Eigenvalue array
        std::vector<double> evals(dim);
        
        // Call LAPACKE_zheevd (divide-and-conquer) — eigenvalues only, no eigenvectors needed
        lapack_int info = LAPACKE_zheevd(
            LAPACK_COL_MAJOR,
            params.compute_eigenvectors ? 'V' : 'N',   // Eigenvalues only for NLCE thermodynamics
            'U',                         // Upper triangle
            dim,                         // Matrix dimension
            reinterpret_cast<lapack_complex_double*>(dense.data()),
            dim,                         // Leading dimension
            evals.data()                 // Output eigenvalues
        );
        
        if (info != 0) {
            std::cerr << "  LAPACKE_zheevd failed for sector n_up=" << n_up 
                      << " with error code " << info << std::endl;
            continue;
        }
        
        // Append eigenvalues to the combined list
        all_eigenvalues.insert(all_eigenvalues.end(), evals.begin(), evals.end());
        
        auto sector_end = std::chrono::high_resolution_clock::now();
        double sector_time = std::chrono::duration<double>(sector_end - sector_start).count();
        
        std::cout << "  Diagonalized in " << std::fixed << std::setprecision(2) 
                  << sector_time << " s, found " << dim << " eigenvalues"
                  << " [" << evals.front() << ", " << evals.back() << "]" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    // Sort all eigenvalues globally
    std::sort(all_eigenvalues.begin(), all_eigenvalues.end());
    
    // Store results
    if (params.num_eigenvalues > 0 && params.num_eigenvalues < all_eigenvalues.size()) {
        results.eigenvalues.assign(all_eigenvalues.begin(), 
                                   all_eigenvalues.begin() + params.num_eigenvalues);
    } else {
        results.eigenvalues = std::move(all_eigenvalues);
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "Sz-SPLIT FULL DIAG COMPLETE" << std::endl;
    std::cout << "Total eigenvalues: " << results.eigenvalues.size() << std::endl;
    std::cout << "Total wall time: " << std::fixed << std::setprecision(2) << total_time << " s" << std::endl;
    if (!results.eigenvalues.empty()) {
        std::cout << "Ground state energy: " << results.eigenvalues[0] << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
    
    // Save results to HDF5
    if (!params.output_dir.empty()) {
        safe_system_call("mkdir -p " + params.output_dir);
        
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
            std::cout << "Saved " << results.eigenvalues.size() 
                      << " eigenvalues to " << hdf5_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save to HDF5: " << e.what() << std::endl;
            // Fallback to text file
            std::string txt_file = params.output_dir + "/eigenvalues.txt";
            std::ofstream ofs(txt_file);
            if (ofs.is_open()) {
                ofs << std::setprecision(15);
                for (double ev : results.eigenvalues) {
                    ofs << ev << "\n";
                }
                std::cout << "Saved eigenvalues to " << txt_file << std::endl;
            }
        }
    }
    
    return results;
}

// ============================================================================
// GPU FULL DIAGONALIZATION VIA Sz SECTOR SPLITTING
// ============================================================================

/**
 * @brief GPU full diagonalization by splitting into Sz sectors
 * 
 * GPU equivalent of exact_diagonalization_all_sz_sectors(). Instead of building
 * a CPU sparse matrix per sector, this function:
 * 1. Parses interaction files once into vectors
 * 2. Loops over all Sz sectors (n_up = 0, 1, ..., num_sites)
 * 3. Creates a GPUFixedSzOperator per sector
 * 4. Calls gpuFullDiagonalization (builds dense H via matVec, cuSOLVER zheevd)
 * 5. Collects and sorts all eigenvalues
 * 6. Saves the merged result to HDF5
 * 
 * Same memory advantage as CPU sector splitting - the largest sector is C(N,N/2).
 * Each sector's GPU operator is created and destroyed independently, so only one
 * sector's dense matrix is in GPU memory at a time.
 * 
 * @param interaction_file Path to interaction file (InterAll.dat)
 * @param single_site_file Path to single-site file (Trans.dat)
 * @param num_sites Number of spin sites
 * @param spin_length Spin length (usually 0.5)
 * @param params Parameters for diagonalization
 * @return EDResults containing ALL eigenvalues from ALL sectors, sorted
 */
#ifdef WITH_CUDA
inline EDResults exact_diagonalization_all_sz_sectors_gpu(
    const std::string& interaction_file,
    const std::string& single_site_file,
    uint64_t num_sites,
    float spin_length,
    const EDParameters& params
) {
    EDResults results;
    
    uint64_t full_dim = 1ULL << num_sites;
    uint64_t num_sectors = num_sites + 1;  // n_up = 0, 1, ..., num_sites
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "GPU FULL DIAGONALIZATION VIA Sz SECTOR SPLITTING" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    std::cout << "Full Hilbert space dimension: " << full_dim << std::endl;
    std::cout << "Number of Sz sectors: " << num_sectors << std::endl;
    
    // Pre-compute sector dimensions using binomial coefficients
    auto binomial = [](uint64_t n, uint64_t k) -> uint64_t {
        if (k > n - k) k = n - k;
        uint64_t result = 1;
        for (uint64_t i = 0; i < k; ++i) {
            result *= (n - i);
            result /= (i + 1);
        }
        return result;
    };
    
    std::vector<uint64_t> sector_dims(num_sectors);
    uint64_t largest_sector_dim = 0;
    for (uint64_t n_up = 0; n_up <= num_sites; ++n_up) {
        sector_dims[n_up] = binomial(num_sites, n_up);
        if (sector_dims[n_up] > largest_sector_dim) {
            largest_sector_dim = sector_dims[n_up];
        }
    }
    
    std::cout << "Largest sector dimension: " << largest_sector_dim 
              << " (reduction: " << std::fixed << std::setprecision(1) 
              << (double)full_dim / largest_sector_dim << "x)" 
              << std::defaultfloat << std::endl;
    std::cout << "Memory for full dense matrix:    " 
              << (double)full_dim * full_dim * 16.0 / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    std::cout << "Memory for largest sector:       " 
              << (double)largest_sector_dim * largest_sector_dim * 16.0 / (1024.0*1024.0*1024.0)
              << " GB" << std::endl;
    
    // Parse interaction files ONCE into vectors for reuse across sectors
    std::vector<std::tuple<int, int, char, char, double>> gpu_interactions;
    std::vector<std::tuple<int, char, double>> gpu_single_site_ops;
    
    {
        std::ifstream inter_file(interaction_file);
        if (inter_file.is_open()) {
            std::string line;
            std::getline(inter_file, line);
            std::getline(inter_file, line);
            std::istringstream iss(line);
            uint64_t numLines;
            std::string m;
            iss >> m >> numLines;
            for (uint64_t i = 0; i < 3; ++i) std::getline(inter_file, line);
            
            auto mapOp = [](uint64_t op) -> char {
                if (op == 0) return '+';
                if (op == 1) return '-';
                return 'z';
            };
            
            uint64_t lineCount = 0;
            while (std::getline(inter_file, line) && lineCount < numLines) {
                std::istringstream lineStream(line);
                uint64_t Op_i, indx_i, Op_j, indx_j;
                double E, F;
                if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) { lineCount++; continue; }
                gpu_interactions.push_back(std::make_tuple(indx_i, indx_j, mapOp(Op_i), mapOp(Op_j), E));
                lineCount++;
            }
        }
        
        if (!single_site_file.empty()) {
            std::ifstream ss_file(single_site_file);
            if (ss_file.is_open()) {
                std::string line;
                std::getline(ss_file, line);
                std::getline(ss_file, line);
                std::istringstream iss(line);
                uint64_t numLines;
                std::string m;
                iss >> m >> numLines;
                for (uint64_t i = 0; i < 3; ++i) std::getline(ss_file, line);
                
                auto mapOp = [](uint64_t op) -> char {
                    if (op == 0) return '+';
                    if (op == 1) return '-';
                    return 'z';
                };
                
                uint64_t lineCount = 0;
                while (std::getline(ss_file, line) && lineCount < numLines) {
                    std::istringstream lineStream(line);
                    uint64_t Op_i, indx_i;
                    double E, F;
                    if (!(lineStream >> Op_i >> indx_i >> E >> F)) { lineCount++; continue; }
                    gpu_single_site_ops.push_back(std::make_tuple(indx_i, mapOp(Op_i), E));
                    lineCount++;
                }
            }
        }
    }
    
    std::cout << "Loaded " << gpu_interactions.size() << " interactions, "
              << gpu_single_site_ops.size() << " single-site terms" << std::endl;
    
    // Collect all eigenvalues from all sectors
    std::vector<double> all_eigenvalues;
    all_eigenvalues.reserve(full_dim);
    
    auto total_start = std::chrono::high_resolution_clock::now();
    
    for (uint64_t n_up = 0; n_up <= num_sites; ++n_up) {
        uint64_t dim = sector_dims[n_up];
        double sz_value = spin_length * num_sites - n_up;
        
        std::cout << "\n--- Sector " << n_up + 1 << "/" << num_sectors 
                  << ": n_up=" << n_up << ", Sz=" << sz_value
                  << ", dim=" << dim << " ---" << std::endl;
        
        if (dim == 0) {
            std::cout << "  Empty sector, skipping." << std::endl;
            continue;
        }
        
        auto sector_start = std::chrono::high_resolution_clock::now();
        
        // Create GPU Fixed Sz operator for this sector
        void* gpu_op_handle = GPUEDWrapper::createGPUFixedSzOperatorDirect(
            num_sites, n_up, spin_length,
            gpu_interactions, gpu_single_site_ops);
        
        if (!gpu_op_handle) {
            std::cerr << "  Error: Failed to create GPU operator for sector n_up=" << n_up << std::endl;
            continue;
        }
        
        // Run GPU full diag on this sector (no HDF5 save per sector)
        std::vector<double> sector_eigenvalues;
        GPUEDWrapper::runGPUFullDiag(
            gpu_op_handle,
            static_cast<int>(dim),
            static_cast<int>(dim),  // Get ALL eigenvalues in each sector
            sector_eigenvalues,
            "",   // Empty dir = don't save per-sector HDF5
            false // No eigenvectors needed for thermodynamics
        );
        
        // Destroy GPU operator for this sector (frees GPU memory)
        GPUEDWrapper::destroyGPUOperator(gpu_op_handle);
        
        // Append eigenvalues
        all_eigenvalues.insert(all_eigenvalues.end(), 
                                sector_eigenvalues.begin(), sector_eigenvalues.end());
        
        auto sector_end = std::chrono::high_resolution_clock::now();
        double sector_time = std::chrono::duration<double>(sector_end - sector_start).count();
        
        if (!sector_eigenvalues.empty()) {
            std::cout << "  GPU diag: " << std::fixed << std::setprecision(2) 
                      << sector_time << " s, " << sector_eigenvalues.size() << " eigenvalues"
                      << " [" << sector_eigenvalues.front() << ", " << sector_eigenvalues.back() << "]" 
                      << std::defaultfloat << std::endl;
        }
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();
    
    // Sort all eigenvalues globally
    std::sort(all_eigenvalues.begin(), all_eigenvalues.end());
    
    // Store results
    if (params.num_eigenvalues > 0 && static_cast<size_t>(params.num_eigenvalues) < all_eigenvalues.size()) {
        results.eigenvalues.assign(all_eigenvalues.begin(), 
                                   all_eigenvalues.begin() + params.num_eigenvalues);
    } else {
        results.eigenvalues = std::move(all_eigenvalues);
    }
    
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "GPU Sz-SPLIT FULL DIAG COMPLETE" << std::endl;
    std::cout << "Total eigenvalues: " << results.eigenvalues.size() << std::endl;
    std::cout << "Total wall time: " << std::fixed << std::setprecision(2) << total_time << " s" << std::endl;
    if (!results.eigenvalues.empty()) {
        std::cout << "Ground state energy: " << results.eigenvalues[0] << std::endl;
    }
    std::cout << std::string(80, '=') << std::endl;
    
    // Save results to HDF5
    if (!params.output_dir.empty()) {
        safe_system_call("mkdir -p " + params.output_dir);
        
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            HDF5IO::saveEigenvalues(hdf5_file, results.eigenvalues);
            std::cout << "Saved " << results.eigenvalues.size() 
                      << " eigenvalues to " << hdf5_file << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to save to HDF5: " << e.what() << std::endl;
        }
    }
    
    return results;
}
#endif // WITH_CUDA


// ============================================================================
// FIXED SZ EXACT DIAGONALIZATION
// ============================================================================

/**
 * @brief Exact diagonalization in fixed Sz sector
 * 
 * Performs diagonalization on a restricted Hilbert space with fixed
 * total Sz quantum number, significantly reducing memory and computational cost.
 * 
 * @param interaction_file Path to interaction file
 * @param single_site_file Path to single-site file
 * @param num_sites Number of sites
 * @param spin_length Spin length (usually 0.5)
 * @param n_up Number of up spins (determines Sz sector)
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_fixed_sz(
    const std::string& interaction_file,
    const std::string& single_site_file,
    uint64_t num_sites,
    float spin_length,
    int64_t n_up,
    DiagonalizationMethod method,
    const EDParameters& params_in
) {
    // Phase 7: idempotent canonicalization (also covers callers that
    // skip exact_diagonalization_from_files and jump straight to here,
    // e.g. workflows.cpp::run_standard_workflow when --fixed-sz is set
    // in the config).
    EDParameters params = params_in;
    {
        const auto canon = ed::canonicalize_method_and_flags(
            method, params.use_fixed_sz, params.use_gpu, params.use_mpi);
        params.use_fixed_sz = canon.use_fixed_sz;
        params.use_gpu      = canon.use_gpu;
        params.use_mpi      = canon.use_mpi;
        method = ed::legacy_method_for_dispatch(canon.method, canon.use_gpu);
    }

    FixedSzOperator hamiltonian(num_sites, spin_length, n_up);
    
    // Load Hamiltonian terms
    if (!single_site_file.empty()) {
        std::ifstream file(single_site_file);
        if (file.is_open()) {
            hamiltonian.loadFromFile(single_site_file);
        }
    }
    if (!interaction_file.empty()) {
        std::ifstream file(interaction_file);
        if (file.is_open()) {
            hamiltonian.loadFromInterAllFile(interaction_file);
        }
    }
    
    // Get dimension of fixed Sz sector
    uint64_t fixed_sz_dim = hamiltonian.getFixedSzDim();
    uint64_t full_dim = 1ULL << num_sites;
    
    std::cout << "Fixed Sz basis: dim=" << fixed_sz_dim 
              << " (reduction: " << std::fixed << std::setprecision(1) 
              << (double)full_dim / fixed_sz_dim << "x)" << std::defaultfloat << std::endl;
    
    // Check if GPU method requested
    bool is_gpu_method = (method == DiagonalizationMethod::DAVIDSON_GPU ||
                          method == DiagonalizationMethod::LOBPCG_GPU ||
                          method == DiagonalizationMethod::KRYLOV_SCHUR_GPU ||
                          method == DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU ||
                          method == DiagonalizationMethod::LANCZOS_GPU ||
                          method == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ ||
                          method == DiagonalizationMethod::BLOCK_LANCZOS_GPU ||
                          method == DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ ||
                          method == DiagonalizationMethod::mTPQ_GPU ||
                          method == DiagonalizationMethod::cTPQ_GPU ||
                          method == DiagonalizationMethod::FTLM_GPU_FIXED_SZ ||
                          method == DiagonalizationMethod::FULL_GPU);
    
    EDResults results;
    
    if (is_gpu_method) {
#ifdef WITH_CUDA
        // Prepare interactions and single-site operators
        std::vector<std::tuple<int, int, char, char, double>> gpu_interactions;
        std::vector<std::tuple<int, char, double>> gpu_single_site_ops;
        
        // Load from files
        std::ifstream inter_file(interaction_file);
        if (inter_file.is_open()) {
            std::string line;
            std::getline(inter_file, line);
            std::getline(inter_file, line);
            std::istringstream iss(line);
            uint64_t numLines;
            std::string m;
            iss >> m >> numLines;
            
            for (uint64_t i = 0; i < 3; ++i) std::getline(inter_file, line);
            
            uint64_t lineCount = 0;
            while (std::getline(inter_file, line) && lineCount < numLines) {
                std::istringstream lineStream(line);
                uint64_t Op_i, indx_i, Op_j, indx_j;
                double E, F;
                
                if (!(lineStream >> Op_i >> indx_i >> Op_j >> indx_j >> E >> F)) continue;
                
                // File operator codes: 0=S+, 1=S-, 2=Sz
                // Map to chars: '+'=S+, '-'=S-, 'z'=Sz
                auto mapOp = [](uint64_t op) -> char {
                    if (op == 0) return '+';  // S+
                    if (op == 1) return '-';  // S-
                    return 'z';  // Sz
                };
                
                gpu_interactions.push_back(std::make_tuple(indx_i, indx_j, mapOp(Op_i), mapOp(Op_j), E));
                lineCount++;
            }
        }
        
        // Load single-site terms if present
        if (!single_site_file.empty()) {
            std::ifstream ss_file(single_site_file);
            if (ss_file.is_open()) {
                std::string line;
                std::getline(ss_file, line);
                std::getline(ss_file, line);
                std::istringstream iss(line);
                uint64_t numLines;
                std::string m;
                iss >> m >> numLines;
                
                for (uint64_t i = 0; i < 3; ++i) std::getline(ss_file, line);
                
                uint64_t lineCount = 0;
                while (std::getline(ss_file, line) && lineCount < numLines) {
                    std::istringstream lineStream(line);
                    uint64_t Op_i, indx_i;
                    double E, F;
                    
                    if (!(lineStream >> Op_i >> indx_i >> E >> F)) continue;
                    
                    // File operator codes: 0=S+, 1=S-, 2=Sz
                    // Map to chars: '+'=S+, '-'=S-, 'z'=Sz
                    auto mapOp = [](uint64_t op) -> char {
                        if (op == 0) return '+';  // S+
                        if (op == 1) return '-';  // S-
                        return 'z';  // Sz
                    };
                    
                    gpu_single_site_ops.push_back(std::make_tuple(indx_i, mapOp(Op_i), E));
                    lineCount++;
                }
            }
        }
        
        // Create GPU operator
        void* gpu_op_handle = GPUEDWrapper::createGPUFixedSzOperatorDirect(
            num_sites, n_up, spin_length,
            gpu_interactions, gpu_single_site_ops);
        
        std::cout << "Loaded " << gpu_interactions.size() << " interactions and " 
                  << gpu_single_site_ops.size() << " single-site terms\n";
        
        // Run appropriate GPU method
        std::vector<double> eigenvalues;
        
        if (method == DiagonalizationMethod::DAVIDSON_GPU) {
            GPUEDWrapper::runGPUDavidsonFixedSz(
                gpu_op_handle, n_up,
                params.num_eigenvalues,
                params.max_iterations,
                params.max_subspace,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::LOBPCG_GPU) {
            GPUEDWrapper::runGPULOBPCGFixedSz(
                gpu_op_handle, n_up,
                params.num_eigenvalues,
                params.max_iterations,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::LANCZOS_GPU || method == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ) {
            GPUEDWrapper::runGPULanczosFixedSz(
                gpu_op_handle, n_up,
                params.max_iterations,
                params.num_eigenvalues,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::mTPQ_GPU) {
            GPUEDWrapper::runGPUMicrocanonicalTPQFixedSz(
                gpu_op_handle, n_up,
                params.max_iterations,
                params.num_samples,
                params.tpq_measurement_interval,
                eigenvalues,
                params.output_dir,
                params.tpq_energy_shift,
                params.tpq_continue,
                params.tpq_continue_sample,
                params.tpq_continue_beta,
                params.save_thermal_states,
                params.tpq_target_beta,
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max);
        } else if (method == DiagonalizationMethod::cTPQ_GPU) {
            GPUEDWrapper::runGPUCanonicalTPQFixedSz(
                gpu_op_handle, n_up,
                params.temp_max,  // beta_max
                params.num_samples,
                params.tpq_measurement_interval,
                eigenvalues,
                params.output_dir,
                params.tpq_delta_beta,  // delta_beta
                params.tpq_taylor_order,  // taylor_order
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max);
        } else if (method == DiagonalizationMethod::BLOCK_LANCZOS_GPU || 
                   method == DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ) {
            GPUEDWrapper::runGPUBlockLanczosFixedSz(
                gpu_op_handle, n_up,
                params.max_iterations,
                params.num_eigenvalues,
                params.block_size,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::KRYLOV_SCHUR_GPU) {
            GPUEDWrapper::runGPUKrylovSchurFixedSz(
                gpu_op_handle, n_up,
                params.num_eigenvalues,
                params.max_iterations,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU) {
            GPUEDWrapper::runGPUBlockKrylovSchurFixedSz(
                gpu_op_handle, n_up,
                params.num_eigenvalues,
                params.max_iterations,
                params.block_size,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        } else if (method == DiagonalizationMethod::FULL_GPU) {
            GPUEDWrapper::runGPUFullDiag(
                gpu_op_handle,
                fixed_sz_dim,
                params.num_eigenvalues,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors);
        }
        
        results.eigenvalues = eigenvalues;
        
        // Note: GPU TPQ states are now automatically transformed during save (via saveTPQState)
        // No post-processing transformation needed
        if (method == DiagonalizationMethod::mTPQ_GPU || method == DiagonalizationMethod::cTPQ_GPU) {
            std::cout << "\nGPU TPQ states were automatically transformed to full Hilbert space during save." << std::endl;
            
            // MPI-safe HDF5 merge: merge per-rank files on rank 0
#ifdef WITH_MPI
            int mpi_size, mpi_rank;
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            
            if (mpi_size > 1) {
                MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks have finished writing
            }
            if (mpi_rank == 0) {
                std::cout << "\nMerging per-rank HDF5 files..." << std::endl;
                HDF5IO::mergePerRankTPQFiles(params.output_dir, mpi_size, "ed_results.h5", true);
                // Convert TPQ results to unified thermodynamic format
                convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
            }
#else
            // Non-MPI: convert TPQ results to unified thermodynamic format
            convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
#endif
        }
        
        // Cleanup
        GPUEDWrapper::destroyGPUOperator(gpu_op_handle);
        
        std::cout << "GPU diagonalization complete\n";
#else
        throw std::runtime_error("GPU methods require CUDA support (compile with -DWITH_CUDA=ON)");
#endif
    } else {

        
        // Create apply function
        auto apply_hamiltonian = [&hamiltonian, fixed_sz_dim](const Complex* in, Complex* out, uint64_t n) {
            if (n != fixed_sz_dim) {
                throw std::runtime_error("Dimension mismatch in fixed Sz apply");
            }
            hamiltonian.apply(in, out, n);
        };
        
        // Set the fixed_sz_op in params so TPQ can transform states before saving
        params.fixed_sz_op = &hamiltonian;
        
        // Perform diagonalization
        std::cout << "\nDiagonalizing..." << std::endl;
        results = exact_diagonalization_core(apply_hamiltonian, fixed_sz_dim, method, params);
    }

    // Check if this is a TPQ method
    bool is_tpq_method = (method == DiagonalizationMethod::mTPQ || 
                          method == DiagonalizationMethod::mTPQ_CUDA || 
                          method == DiagonalizationMethod::cTPQ);

    // Transform eigenvectors from fixed-Sz basis to full basis
    size_t n_eigs = results.eigenvalues.size();
    if (!params.output_dir.empty() && params.compute_eigenvectors && n_eigs > 0) {
        // Transform eigenvectors if computed - load from HDF5, transform, save back
        try {
            std::string hdf5_file = HDF5IO::createOrOpenFile(params.output_dir);
            
            for (size_t i = 0; i < n_eigs; ++i) {
                // Load eigenvector from HDF5 (in fixed-Sz basis)
                std::vector<Complex> fixed_sz_vec = HDF5IO::loadEigenvector(hdf5_file, i);
                
                if (fixed_sz_vec.size() == fixed_sz_dim) {
                    // Transform to full basis
                    std::vector<Complex> full_vec = hamiltonian.embedToFull(fixed_sz_vec);
                    
                    // Save back to HDF5 with full-space eigenvector
                    HDF5IO::saveEigenvector(hdf5_file, i, full_vec);
                }
            }
            std::cout << "Transformed " << n_eigs << " eigenvectors to full space" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "Warning: Failed to transform eigenvectors: " << e.what() << std::endl;
        }
    }

    return results;
}

// ============================================================================
// FILE-BASED WRAPPER FUNCTIONS
// ============================================================================

/**
 * @brief Wrapper function to perform exact diagonalization from Hamiltonian files
 * 
 * Loads Hamiltonian from input files and performs diagonalization.
 * 
 * @param interaction_file Path to interaction file (e.g., InterAll.dat)
 * @param single_site_file Path to single-site file (e.g., Trans.dat)
 * @param counterterm_file Path to counter term file (optional)
 * @param three_body_file Path to three-body interaction file (e.g., ThreeBodyG.dat, optional)
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @param format File format for Hamiltonian
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_from_files(
    const std::string& interaction_file,
    const std::string& single_site_file = "",
    const std::string& counterterm_file = "",
    const std::string& three_body_file = "",
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params_in = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD
) {
    if (ed_log::isVerbose()) {
        std::ostringstream _dbg;
        _dbg << "exact_diagonalization_from_files: num_sites=" << params_in.num_sites
             << ", method=" << static_cast<int>(method);
        ed_log::debug(_dbg.str());
    }

    // ========== Phase 7: Single canonicalization point ==========
    // Collapse legacy _FIXED_SZ / _GPU / _CUDA / _MPI enum variants onto
    // the canonical (base_method, use_fixed_sz, use_gpu, use_mpi) tuple.
    // This is the only place where the deprecated variants are observed;
    // every dispatch branch below sees flags only.
    EDParameters params = params_in;
    {
        const auto canon = ed::canonicalize_method_and_flags(
            method, params.use_fixed_sz, params.use_gpu, params.use_mpi);
        params.use_fixed_sz = canon.use_fixed_sz;
        params.use_gpu      = canon.use_gpu;
        params.use_mpi      = canon.use_mpi;
        method = ed::legacy_method_for_dispatch(canon.method, canon.use_gpu);
    }

    bool use_fixed_sz = params.use_fixed_sz;

    // ========== Phase 7.1: Symmetry routing (5th orthogonal axis) ==========
    // When `use_symmetry == true` the canonical dispatch is to route to
    // the streaming symmetry kernel in ed_wrapper_streaming.h. We CANNOT
    // call it from this header because ed_wrapper_streaming.h includes
    // ed_wrapper.h (circular). Instead, the routing happens in the
    // higher-level dispatcher seam:
    //   * Python:  ed_dispatch::exact_diagonalization_from_directory(...)
    //              in ed_dispatch_symmetry.h (included by the Python
    //              binding, which sees both headers).
    //   * CLI:     ed_main.cpp routes via run_streaming_symmetry_workflow
    //              when WorkflowConfig::run_symm_auto is set.
    //   * C++:     direct callers of from_files() that set use_symmetry=true
    //              get a hard error here so the bug is loud, not silent.
    if (params.use_symmetry) {
        throw std::runtime_error(
            "EDParameters::use_symmetry=true is not supported by "
            "exact_diagonalization_from_files() directly. Call "
            "ed_dispatch::exact_diagonalization_from_directory() "
            "(declared in ed/core/ed_dispatch_symmetry.h) or invoke "
            "exact_diagonalization_streaming_symmetry[_fixed_sz]() "
            "explicitly. See docs/history/PHASE_7_SYMMETRY_AXIS.md.");
    }

    // Check if method supports fixed-Sz when requested
    if (use_fixed_sz && !ed_internal::supports_fixed_sz(method)) {
        std::cerr << "Warning: Method does not support fixed-Sz mode. "
                  << "Proceeding with full Hilbert space.\n";
        use_fixed_sz = false;
    }
    
    // Route to fixed-Sz function if use_fixed_sz is true
    // This ensures all fixed-Sz GPU logic is handled in one place
    if (use_fixed_sz) {
        int64_t n_up = (params.n_up >= 0) ? params.n_up : params.num_sites / 2;
        return exact_diagonalization_fixed_sz(
            interaction_file,
            single_site_file,
            params.num_sites,
            params.spin_length,
            n_up,
            method,
            params
        );
    }
    
    // ========== Full Sz-Sector Split ==========
    // Auto-enable Sz sector splitting for FULL diag when the Hamiltonian
    // conserves total Sz.  This is always beneficial: it reduces the largest
    // dense block from 2^N to C(N,N/2) (e.g. 29x less memory at N=17).
    // The flag --full-sz-split can force it on (caller takes responsibility)
    // or it is auto-detected by scanning InterAll.dat / Trans.dat.
    if (method == DiagonalizationMethod::FULL || 
        method == DiagonalizationMethod::SCALAPACK ||
        method == DiagonalizationMethod::SCALAPACK_MIXED) {
        
        bool use_sz_split = params.full_sz_split;  // Explicitly requested?
        
        if (!use_sz_split) {
            // Auto-detect: check if Hamiltonian conserves Sz
            bool sz_conserved = hamiltonian_conserves_sz(interaction_file, single_site_file);
            if (sz_conserved) {
                std::cout << "[Auto] Hamiltonian conserves Sz — enabling sector splitting for full diag" << std::endl;
                use_sz_split = true;
            }
        }
        
        if (use_sz_split) {
            std::cout << "Using Sz-sector splitting for full diagonalization" << std::endl;
            return exact_diagonalization_all_sz_sectors(
                interaction_file,
                single_site_file,
                params.num_sites,
                params.spin_length,
                params
            );
        }
    }
    
    // ========== GPU FULL_GPU Sz-Sector Split ==========
    // Same auto-detection for GPU full diag: split into Sz sectors on GPU
#ifdef WITH_CUDA
    if (method == DiagonalizationMethod::FULL_GPU) {
        bool use_sz_split = params.full_sz_split;
        
        if (!use_sz_split) {
            bool sz_conserved = hamiltonian_conserves_sz(interaction_file, single_site_file);
            if (sz_conserved) {
                std::cout << "[Auto] Hamiltonian conserves Sz — enabling GPU sector splitting for full diag" << std::endl;
                use_sz_split = true;
            }
        }
        
        if (use_sz_split) {
            std::cout << "Using GPU Sz-sector splitting for full diagonalization" << std::endl;
            return exact_diagonalization_all_sz_sectors_gpu(
                interaction_file,
                single_site_file,
                params.num_sites,
                params.spin_length,
                params
            );
        }
    }
#endif
    
    // Handle GPU methods separately (they don't need CPU Operator)
#ifdef WITH_CUDA
    if (ed_internal::is_gpu_method(method)) {
        
        std::cout << "Running GPU-accelerated algorithm (full Hilbert space)..." << std::endl;
        
        // Check if GPU is available
        if (!GPUEDWrapper::isGPUAvailable()) {
            std::cerr << "Error: No CUDA-capable GPU found!" << std::endl;
            throw std::runtime_error("GPU not available");
        }
        
        GPUEDWrapper::printGPUInfo();
        
        EDResults results;
        uint64_t hilbert_space_dim = static_cast<int>(1ULL << params.num_sites);
        
        if (method == DiagonalizationMethod::LANCZOS_GPU) {
            // Check if files exist
            if (!std::filesystem::exists(interaction_file)) {
                std::cerr << "Error: " << interaction_file << " not found!" << std::endl;
                throw std::runtime_error("InterAll.dat file not found");
            }
            
            // Create GPU operator from files (full Hilbert space)
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            // Run GPU Lanczos
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPULanczos(
                gpu_op,
                hilbert_space_dim,
                params.max_iterations,
                params.num_eigenvalues,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            // Store results
            results.eigenvalues = eigenvalues;
            
            // Clean up
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Lanczos completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ) {
            // Deprecated variant — normalize_method_and_fixed_sz() upstream
            // should have collapsed this to LANCZOS_GPU + use_fixed_sz=true
            // and routed it through exact_diagonalization_fixed_sz. If we
            // ever hit here it means the dispatcher contract was broken.
            // (D-6: removed a follow-up unreachable throw.)
            throw std::runtime_error(
                "Internal error: LANCZOS_GPU_FIXED_SZ reached the file-based "
                "dispatcher without being normalized; this is a bug in the "
                "method-dispatch path.");

        } else if (method == DiagonalizationMethod::DAVIDSON_GPU) {
            std::cout << "Running GPU Davidson method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUDavidson(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                params.max_iterations,
                params.max_subspace,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Davidson completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::LOBPCG_GPU) {
            std::cout << "Running GPU LOBPCG method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPULOBPCG(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                params.max_iterations,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU LOBPCG completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::mTPQ_GPU) {
            std::cout << "Running GPU microcanonical TPQ..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUMicrocanonicalTPQ(
                gpu_op,
                hilbert_space_dim,
                params.max_iterations,
                params.num_samples,
                params.tpq_measurement_interval,
                eigenvalues,
                params.output_dir,
                params.tpq_energy_shift,
                params.tpq_continue,
                params.tpq_continue_sample,
                params.tpq_continue_beta,
                params.save_thermal_states,
                params.tpq_target_beta,
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            // MPI-safe HDF5 merge: merge per-rank files on rank 0
#ifdef WITH_MPI
            int mpi_size, mpi_rank;
            MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
            MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
            
            if (mpi_size > 1) {
                MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks have finished writing
            }
            if (mpi_rank == 0) {
                std::cout << "\nMerging per-rank HDF5 files..." << std::endl;
                HDF5IO::mergePerRankTPQFiles(params.output_dir, mpi_size, "ed_results.h5", true);
                // Convert TPQ results to unified thermodynamic format
                convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
            }
#else
            // Non-MPI: convert TPQ results to unified thermodynamic format
            convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
#endif
            
            std::cout << "GPU mTPQ completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::cTPQ_GPU) {
            std::cout << "Running GPU canonical TPQ..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUCanonicalTPQ(
                gpu_op,
                hilbert_space_dim,
                params.temp_max,
                params.num_samples,
                params.tpq_measurement_interval,
                eigenvalues,
                params.output_dir,
                params.tpq_delta_beta,
                params.tpq_taylor_order,
                params.tpq_num_measure_points,
                params.tpq_measure_beta_min,
                params.tpq_measure_beta_max
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            // MPI-safe HDF5 merge: merge per-rank files on rank 0
#ifdef WITH_MPI
            {
                int mpi_size, mpi_rank;
                MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
                MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
                
                if (mpi_size > 1) {
                    MPI_Barrier(MPI_COMM_WORLD);  // Ensure all ranks have finished writing
                }
                if (mpi_rank == 0) {
                    std::cout << "\nMerging per-rank HDF5 files..." << std::endl;
                    HDF5IO::mergePerRankTPQFiles(params.output_dir, mpi_size, "ed_results.h5", true);
                    // Convert TPQ results to unified thermodynamic format
                    convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
                }
            }
#else
            // Non-MPI: convert TPQ results to unified thermodynamic format
            convert_tpq_to_unified_thermodynamics(params.output_dir, params.num_samples);
#endif
            
            std::cout << "GPU cTPQ completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::FTLM_GPU) {
            std::cout << "Running GPU Finite Temperature Lanczos Method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            GPUEDWrapper::runGPUFTLM(
                gpu_op,
                hilbert_space_dim,
                params.ftlm_krylov_dim,
                params.num_samples,
                params.temp_min,
                params.temp_max,
                params.num_temp_bins,
                params.tolerance,
                params.output_dir,
                params.ftlm_full_reorth,
                params.ftlm_reorth_freq,
                params.ftlm_seed
            );
            
            // FTLM doesn't return eigenvalues in the traditional sense
            // Results are thermodynamic quantities written to files
            
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU FTLM completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::FTLM_GPU_FIXED_SZ) {
            std::cerr << "Error: FTLM_GPU_FIXED_SZ file interface not yet implemented." << std::endl;
            std::cerr << "Please use the fixed_sz wrapper function directly." << std::endl;
            throw std::runtime_error("Fixed Sz GPU FTLM not yet integrated with file interface");
        } else if (method == DiagonalizationMethod::BLOCK_LANCZOS_GPU) {
            std::cout << "Running GPU Block Lanczos method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUBlockLanczos(
                gpu_op,
                hilbert_space_dim,
                params.max_iterations,
                params.num_eigenvalues,
                params.block_size,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Block Lanczos completed successfully!" << std::endl;
            
        } else if (method == DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ) {
            std::cerr << "Error: BLOCK_LANCZOS_GPU_FIXED_SZ file interface not yet implemented." << std::endl;
            std::cerr << "Please use the fixed_sz wrapper function directly." << std::endl;
            throw std::runtime_error("Fixed Sz GPU Block Lanczos not yet integrated with file interface");
        } else if (method == DiagonalizationMethod::KRYLOV_SCHUR_GPU) {
            std::cout << "Running GPU Krylov-Schur method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUKrylovSchur(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                params.max_iterations,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Krylov-Schur completed successfully!" << std::endl;
        } else if (method == DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU) {
            std::cout << "Running GPU Block Krylov-Schur method..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUBlockKrylovSchur(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                params.max_iterations,
                params.block_size,
                params.tolerance,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU Block Krylov-Schur completed successfully!" << std::endl;
        } else if (method == DiagonalizationMethod::FULL_GPU) {
            std::cout << "Running GPU full diagonalization (cuSOLVER zheevd)..." << std::endl;
            
            void* gpu_op = GPUEDWrapper::createGPUOperatorFromFiles(
                params.num_sites, interaction_file, single_site_file);
            
            if (!gpu_op) {
                std::cerr << "Error: Failed to create GPU operator" << std::endl;
                throw std::runtime_error("GPU operator creation failed");
            }
            
            std::vector<double> eigenvalues;
            GPUEDWrapper::runGPUFullDiag(
                gpu_op,
                hilbert_space_dim,
                params.num_eigenvalues,
                eigenvalues,
                params.output_dir,
                params.compute_eigenvectors
            );
            
            results.eigenvalues = eigenvalues;
            GPUEDWrapper::destroyGPUOperator(gpu_op);
            
            std::cout << "GPU full diagonalization completed successfully!" << std::endl;
        }
        
        return results;
    }
#endif
    
    // Load Hamiltonian (for CPU methods)
    Operator hamiltonian = ed_internal::load_hamiltonian_from_files(
        interaction_file, single_site_file, counterterm_file, three_body_file, 
        params.num_sites, params.spin_length, method, format
    );
    
    // Calculate Hilbert space dimension
    uint64_t hilbert_space_dim = static_cast<int>(1ULL << params.num_sites);
    if (ed_log::isVerbose()) {
        ed_log::debug("hilbert_space_dim=" + std::to_string(hilbert_space_dim));
    }
    
    // Create Hamiltonian apply function
    auto apply_hamiltonian = ed_internal::create_hamiltonian_apply_function(hamiltonian);
    
    // Perform diagonalization
    return exact_diagonalization_core(apply_hamiltonian, hilbert_space_dim, method, params);
}

/**
 * @brief Wrapper function to perform exact diagonalization from a directory
 * 
 * Convenience function that constructs file paths from a directory and
 * calls exact_diagonalization_from_files.
 * 
 * @param directory Directory containing Hamiltonian files
 * @param method Diagonalization method to use
 * @param params Parameters for diagonalization
 * @param format File format for Hamiltonian
 * @param interaction_filename Name of interaction file (default: "InterAll.dat")
 * @param single_site_filename Name of single-site file (default: "Trans.dat")
 * @return EDResults containing eigenvalues and metadata
 */
inline EDResults exact_diagonalization_from_directory(
    const std::string& directory,
    DiagonalizationMethod method = DiagonalizationMethod::LANCZOS,
    const EDParameters& params = EDParameters(),
    HamiltonianFileFormat format = HamiltonianFileFormat::STANDARD,
    const std::string& interaction_filename = "InterAll.dat",
    const std::string& single_site_filename = "Trans.dat",
    const std::string& counterterm_filename = "CounterTerm.dat",
    const std::string& three_body_filename = "ThreeBodyG.dat"
) {
    // Construct full file paths
    std::string interaction_file = directory + "/" + interaction_filename;
    std::string single_site_file = directory + "/" + single_site_filename;
    std::string counterterm_file = directory + "/" + counterterm_filename;
    std::string three_body_file = directory + "/" + three_body_filename;
    
    // Check if counter term file exists
    struct stat buffer;
    if (stat(counterterm_file.c_str(), &buffer) != 0) {
        counterterm_file = "";  // File doesn't exist, pass empty string
    }
    
    // Check if three-body file exists
    if (stat(three_body_file.c_str(), &buffer) != 0) {
        three_body_file = "";  // File doesn't exist, pass empty string
    }
    
    // Call the file-based wrapper
    return exact_diagonalization_from_files(
        interaction_file, single_site_file, counterterm_file, three_body_file, method, params, format
    );
}


// End of Phase 7 deprecated-declarations diagnostic scope (push at top of file).
#pragma GCC diagnostic pop
