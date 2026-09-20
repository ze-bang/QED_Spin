#ifndef HDF5_IO_H
#define HDF5_IO_H

// Declarations only. The definitions live in src/io/hdf5_io_file.cpp (helpers,
// chunking, file management), src/io/hdf5_io_eigen.cpp (eigen/thermodynamic/
// correlation datasets), src/io/hdf5_io_tpq.cpp (TPQ samples, per-rank merge)
// and src/io/hdf5_io_thermal.cpp (FTLM/LTLM response, time correlations).

#include <H5Cpp.h>
#include <vector>
#include <complex>
#include <string>
#include <map>
#include <cstddef>
#include <cstdint>
#include <ed/core/thermal_types.h>

using Complex = std::complex<double>;

/**
 * @brief Comprehensive HDF5 I/O utilities for all exact diagonalization intermediate data
 * 
 * This class provides unified file management for:
 * - Eigenvalues and eigenvectors
 * - Thermodynamic observables (energy, entropy, specific heat, etc.)
 * - Correlation functions (static and dynamical)
 * - Structure factors and dynamical structure factors
 * - FTLM samples and thermal states
 * - TPQ states and intermediate results
 * 
 * File structure:
 *   ed_results.h5
 *   ├── /eigendata
 *   │   ├── eigenvalues [dataset: array of double]
 *   │   ├── eigenvector_0 [dataset: complex vector]
 *   │   ├── eigenvector_1 [dataset: complex vector]
 *   │   └── ...
 *   ├── /thermodynamics
 *   │   ├── temperatures [dataset: array of double]
 *   │   ├── energy [dataset: array of double]
 *   │   ├── entropy [dataset: array of double]
 *   │   ├── specific_heat [dataset: array of double]
 *   │   ├── magnetization [dataset: array of double]
 *   │   └── susceptibility [dataset: array of double]
 *   ├── /correlations
 *   │   ├── spin_spin [dataset: 2D matrix]
 *   │   ├── spin_configuration [dataset: array]
 *   │   └── sublattice_correlation [dataset: array]
 *   ├── /dynamical
 *   │   ├── frequencies [dataset: array of double]
 *   │   ├── spectral_function [dataset: array of double]
 *   │   ├── structure_factor [dataset: 2D or 3D array]
 *   │   └── samples/
 *   │       ├── sample_0 [dataset]
 *   │       └── ...
 *   ├── /ftlm
 *   │   ├── samples/
 *   │   │   ├── sample_0/
 *   │   │   │   ├── eigenvalues
 *   │   │   │   ├── eigenvectors
 *   │   │   │   └── thermodynamics
 *   │   │   └── ...
 *   │   └── averaged/
 *   │       └── observables
 *   └── /tpq
 *       ├── samples/
 *       │   ├── sample_0/
 *       │   │   ├── thermodynamics [dataset: beta, energy, variance, doublon, step]
 *       │   │   ├── norm [dataset: beta, norm, first_norm, step]
 *       │   │   ├── fluctuations [dataset: spin fluctuation data]
 *       │   │   └── states/
 *       │   │       ├── beta_0.100000 [dataset: complex state vector]
 *       │   │       ├── beta_1.000000 [dataset: complex state vector]
 *       │   │       └── ...
 *       │   ├── sample_1/
 *       │   │   └── ...
 *       │   └── ...
 *       └── averaged/
 *           └── thermodynamics [dataset: averaged over all samples]
 */
class HDF5IO {
public:

    // ============================================================================
    // Adaptive chunking + tunable compression
    // ============================================================================
    //
    // HDF5 best-practice chunking targets a chunk size in [16 KiB, 1 MiB] for
    // good throughput against the chunk cache and the storage layer. Hardcoded
    // chunk shapes (e.g. {100, num_cols}) silently underperform when the
    // dataset is much smaller (one tiny chunk -> compressor overhead dominates)
    // or much larger (millions of rows / zillions of chunks -> per-chunk
    // metadata dominates).
    //
    // Tunables (env, evaluated lazily on first use):
    //   ED_HDF5_COMPRESSION_LEVEL : 0 (off) .. 9 (max). Default: 4 (balanced).
    //   ED_HDF5_CHUNK_TARGET_BYTES: target chunk size in bytes. Default: 256 KiB.
    //   ED_HDF5_SHUFFLE          : 0/1 to disable/enable shuffle filter. Default: 1.
    //
    // Defaults are chosen for typical ED outputs (TPQ thermodynamics, MELs,
    // streaming append): deflate-4 + shuffle gives ~70-90% of deflate-6's
    // compression at 2-3x the encode throughput, and 256 KiB chunks leave
    // headroom for the default 1 MiB chunk cache while keeping the chunk
    // count below ~10 for typical run sizes.
    static int hdf5_compression_level();

    static size_t hdf5_chunk_target_bytes();

    static bool hdf5_shuffle_enabled();

    /**
     * @brief Build a chunked + (optionally) compressed dataset property list,
     *        adaptively sizing the chunk to ED_HDF5_CHUNK_TARGET_BYTES.
     *
     * @param dims        full dimensions of the dataset (rank == dims.size())
     * @param element_size size in bytes of one element (e.g. sizeof(double))
     * @param last_dim_full_chunk if true, chunk the trailing axis as one block
     *                            (typical for tabular data: chunk full row,
     *                            many rows per chunk).
     */
    static H5::DSetCreatPropList makeAdaptiveDsetProps(
        const std::vector<hsize_t>& dims,
        size_t element_size,
        bool last_dim_full_chunk = true);

    // ============================================================================
    // File Management - Safe Writing Protocol
    // ============================================================================
    // 
    // SAFE WRITING PROTOCOL:
    // The HDF5 I/O system uses a safe writing protocol that:
    // 1. Opens existing files in read/write mode (H5F_ACC_RDWR) - never truncates
    // 2. Creates new files only if they don't exist
    // 3. Ensures required groups exist without overwriting existing data
    // 4. Uses segment-based writing where each run writes to its own segment
    //    (e.g., different TPQ samples, different temperatures, different operators)
    // 5. Existing segments from previous runs are preserved
    //
    // This allows:
    // - Multiple runs to accumulate data in the same file
    // - Restarting failed runs without losing previous data
    // - Adding new TPQ samples to existing files
    // - Computing different operators/temperatures incrementally
    // ============================================================================
    
    /**
     * @brief Ensure a group exists in an HDF5 file, creating it only if needed
     * 
     * This is a safe operation that does not overwrite existing groups.
     * It also handles nested paths by creating parent groups as needed.
     * 
     * @param file Reference to open HDF5 file
     * @param group_path Full path to the group (e.g., "/tpq/samples/sample_0")
     */
    static void ensureGroupExists(H5::H5File& file, const std::string& group_path);
    
    /**
     * @brief Ensure standard ED result groups exist in an HDF5 file
     * 
     * Creates the standard group structure if groups don't already exist.
     * This is safe to call on files with existing data.
     * 
     * @param file Reference to open HDF5 file
     */
    static void ensureStandardGroups(H5::H5File& file);
    
    /**
     * @brief Universal "disable HDF5 output" sentinel.
     *
     * Phase 6.1: every solver write path goes through ``createOrOpenFile`` /
     * ``saveEigenvalues`` / ``saveEigenvector`` / ``saveDiagonalizationResults``
     * / ``ensureTPQSampleGroup`` / ``saveTPQState``. Centralising the
     * "skip all I/O" check here means that **any** caller (CPU / GPU /
     * MPI / DSSF / Python / CLI) gets the no-write fast path simply by
     * passing an empty string or ``"/dev/null"`` for the output dir, with
     * zero per-call-site changes.
     *
     * Treats ``"/dev/null"`` and any path of the form ``"/dev/null/..."``
     * as disabled. The Python bindings remap a default ``output_dir=""``
     * to ``"/dev/null"`` so interactive / benchmarking calls never write
     * to disk unless the caller asked for it.
     */
    static bool isDisabledOutputPath(const std::string& s);

    /**
     * @brief Create or open an HDF5 file for results storage (SAFE - preserves existing data)
     * 
     * SAFE WRITING PROTOCOL:
     * - If the file exists, opens it in read/write mode without truncating
     * - If the file doesn't exist, creates a new file with standard groups
     * - Always ensures standard groups exist (creates if missing)
     * - NEVER overwrites or truncates existing data
     *
     * If ``directory`` is empty or "/dev/null", returns the
     * sentinel ``"/dev/null"`` immediately and performs no I/O. All
     * downstream HDF5IO writers detect this sentinel and short-circuit.
     *
     * @param directory Directory to create/open the file in
     * @param filename Name of the HDF5 file (default: ed_results.h5)
     * @return Full path to the HDF5 file
     */
    static std::string createOrOpenFile(const std::string& directory, 
                                        const std::string& filename = "ed_results.h5");
    
    /**
     * @brief Force create a new HDF5 file (UNSAFE - truncates existing data)
     * 
     * WARNING: This function will DELETE all existing data in the file.
     * Use only when you explicitly want to start fresh.
     * 
     * @param directory Directory to create the file in
     * @param filename Name of the HDF5 file (default: ed_results.h5)
     * @return Full path to the HDF5 file
     */
    static std::string forceCreateFile(const std::string& directory, 
                                       const std::string& filename = "ed_results.h5");
    
    /**
     * @brief Check if HDF5 file exists and is valid
     * Uses filesystem check first to avoid HDF5 error messages when file doesn't exist
     */
    static bool fileExists(const std::string& filepath);
    
    // ============================================================================
    // Eigenvalue/Eigenvector I/O
    // ============================================================================
    
    /**
     * @brief Save eigenvalues to HDF5
     * @param filepath Path to HDF5 file
     * @param eigenvalues Vector of eigenvalues
     */
    static void saveEigenvalues(const std::string& filepath, 
                                const std::vector<double>& eigenvalues);
    
    /**
     * @brief Load eigenvalues from HDF5
     * @param filepath Path to HDF5 file
     * @return Vector of eigenvalues
     */
    static std::vector<double> loadEigenvalues(const std::string& filepath);
    
    /**
     * @brief Save a single eigenvector to HDF5 (stored as real, imag pairs)
     * @param filepath Path to HDF5 file
     * @param index Index of the eigenvector
     * @param eigenvector Complex vector
     */
    static void saveEigenvector(const std::string& filepath, 
                                size_t index, 
                                const std::vector<Complex>& eigenvector);
    
    /**
     * @brief Unified function to save all diagonalization results (eigenvalues + eigenvectors)
     * 
     * This is the preferred method for all solvers to save their results.
     * Creates the output directory and HDF5 file if needed.
     * 
     * @param output_dir Base output directory (e.g., "output")
     * @param eigenvalues Vector of eigenvalues
     * @param eigenvectors Vector of eigenvectors (can be empty if not computed)
     * @param solver_name Name of the solver for logging (e.g., "LANCZOS", "BLOCK_LANCZOS")
     */
    static void saveDiagonalizationResults(
        const std::string& output_dir,
        const std::vector<double>& eigenvalues,
        const std::vector<std::vector<Complex>>& eigenvectors = {},
        const std::string& solver_name = ""
    );
    
    /**
     * @brief Load a single eigenvector from HDF5
     * @param filepath Path to HDF5 file
     * @param index Index of the eigenvector
     * @return Complex vector
     */
    static std::vector<Complex> loadEigenvector(const std::string& filepath, size_t index);
    
    // ============================================================================
    // Thermodynamics I/O
    // ============================================================================
    
    /**
     * @brief Save thermodynamic data (temperature-dependent observables)
     * @param filepath Path to HDF5 file
     * @param temperatures Vector of temperatures
     * @param observable_name Name of the observable (e.g., "energy", "entropy")
     * @param values Vector of observable values
     */
    static void saveThermodynamics(const std::string& filepath,
                                   const std::vector<double>& temperatures,
                                   const std::string& observable_name,
                                   const std::vector<double>& values);
    
    /**
     * @brief Load thermodynamic observable
     */
    static std::vector<double> loadThermodynamicObservable(const std::string& filepath,
                                                           const std::string& observable_name);
    
    // ============================================================================
    // Correlation Functions I/O
    // ============================================================================
    
    /**
     * @brief Save 2D correlation matrix (e.g., spin-spin correlations)
     * @param filepath Path to HDF5 file
     * @param correlation_name Name of correlation (e.g., "spin_spin", "density_density")
     * @param matrix 2D matrix of correlations (num_sites x num_sites)
     */
    static void saveCorrelationMatrix(const std::string& filepath,
                                      const std::string& correlation_name,
                                      const std::vector<std::vector<Complex>>& matrix);
    
    /**
     * @brief Save 1D correlation data (e.g., spin configuration)
     * @param filepath Path to HDF5 file
     * @param dataset_name Name of the dataset
     * @param data Vector of values
     */
    // saveCorrelationData, saveDynamicalResponse, and saveFTLMSample
    // were retired in the minimalist-architecture rev (May 2026): all
    // current correlator / dynamical-response / FTLM-sample writes go
    // through saveCorrelationMatrix, saveDynamicalResponseFull, and
    // saveFTLMDynamicalSample / saveFTLMStaticSample respectively. None
    // of the three deleted helpers had any callers. (The per-sample
    // thermodynamic writer saveFTLMThermodynamicSample went with the
    // Gen-1 FTLM driver, its only caller, in WP10 C6.)

    
    /**
     * @brief Save TPQ state vector
     * 
     * SAFE WRITING: By default, skips saving if state at this beta already exists.
     * Use overwrite=true to replace existing state.
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param beta Inverse temperature
     * @param state State vector
     * @param overwrite If true, overwrite existing state; if false (default), skip if exists
     * @return true if state was saved, false if skipped (already exists and overwrite=false)
     */
    static bool saveTPQState(const std::string& filepath,
                             size_t sample_index,
                             double beta,
                             const std::vector<Complex>& state,
                             bool overwrite = false);
    
    /**
     * @brief Load TPQ state vector from HDF5
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param beta Inverse temperature
     * @param state Output state vector
     * @return true if successful, false otherwise
     */
    static bool loadTPQState(const std::string& filepath,
                             size_t sample_index,
                             double beta,
                             std::vector<Complex>& state);
    
    /**
     * @brief TPQ state info structure
     */
    struct TPQStateInfo {
        size_t sample_index;
        double beta;
        std::string dataset_name;
    };
    
    /**
     * @brief List all TPQ states stored in an HDF5 file
     * @param filepath Path to HDF5 file
     * @param sample_index Optional: filter by sample index (-1 for all samples)
     * @return Vector of TPQStateInfo for each stored state
     */
    static std::vector<TPQStateInfo> listTPQStates(const std::string& filepath, 
                                                    int sample_filter = -1);
    
    /**
     * @brief List TPQ states for a specific sample
     * @param filepath Path to HDF5 file  
     * @param sample_index Sample index
     * @return Vector of TPQStateInfo for the specified sample
     */
    static std::vector<TPQStateInfo> listTPQStatesForSample(const std::string& filepath,
                                                             size_t sample_index);

    /**
     * @brief Load TPQ state by dataset name
     * @param filepath Path to HDF5 file
     * @param dataset_name Full dataset path (e.g., /tpq/samples/sample_0/states/beta_10.500000)
     * @param state Output state vector
     * @return true if successful, false otherwise
     */
    static bool loadTPQStateByName(const std::string& filepath,
                                   const std::string& dataset_name,
                                   std::vector<Complex>& state);
    
    // ============================================================================
    // TPQ Per-Sample Thermodynamic Data I/O (replaces SS_rand*.dat / norm_rand*.dat)
    // ============================================================================
    
    /**
     * @brief Structure to hold TPQ thermodynamic data for a single measurement point
     */
    struct TPQThermodynamicPoint {
        double beta;        // Inverse temperature
        double energy;      // Energy expectation value
        double variance;    // Energy variance
        double doublon;     // Doublon expectation (or other observable)
        uint64_t step;      // TPQ step number
    };
    
    /**
     * @brief Structure to hold TPQ norm data for a single measurement point
     */
    struct TPQNormPoint {
        double beta;        // Inverse temperature
        double norm;        // Current norm
        double first_norm;  // Initial norm
        uint64_t step;      // TPQ step number
    };
    
    /**
     * @brief Ensure TPQ sample group exists in HDF5 file
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     */
    static void ensureTPQSampleGroup(const std::string& filepath, size_t sample_index);
    
    /**
     * @brief Truncate and rewrite TPQ thermodynamics dataset
     * 
     * Used during continue_quenching merge when the new data overlaps with existing data.
     * Deletes the existing dataset and recreates it with kept_data + new_data.
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param kept_data Existing data points to keep (steps before resume point)
     * @param new_data New data points to append (from the continued run)
     */
    static void truncateAndRewriteTPQThermodynamics(const std::string& filepath,
                                                     size_t sample_index,
                                                     const std::vector<TPQThermodynamicPoint>& kept_data,
                                                     const std::vector<TPQThermodynamicPoint>& new_data);
    
    /**
     * @brief Truncate and rewrite TPQ norm dataset
     * 
     * Used during continue_quenching merge when the new data overlaps with existing data.
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param kept_data Existing data points to keep
     * @param new_data New data points to append
     */
    static void truncateAndRewriteTPQNorm(const std::string& filepath,
                                           size_t sample_index,
                                           const std::vector<TPQNormPoint>& kept_data,
                                           const std::vector<TPQNormPoint>& new_data);

    /**
     * @brief Append TPQ thermodynamic data point to HDF5 (replaces SS_rand*.dat writing)
     * 
     * This function appends a single measurement point to the sample's thermodynamics dataset.
     * Data is stored as: [beta, energy, variance, doublon, step]
     * 
     * SAFE WRITING: Skips writing if this step already exists in the dataset.
     * This supports continue_quenching mode where runs may overlap.
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index (0, 1, 2, ...)
     * @param point Thermodynamic data point to append
     * @return true if data was written, false if step already existed (skipped)
     */
    static bool appendTPQThermodynamics(const std::string& filepath,
                                        size_t sample_index,
                                        const TPQThermodynamicPoint& point);
    
    /**
     * @brief Append TPQ norm data point to HDF5 (replaces norm_rand*.dat writing)
     * 
     * Data is stored as: [beta, norm, first_norm, step]
     * 
     * SAFE WRITING: Skips writing if this step already exists in the dataset.
     * This supports continue_quenching mode where runs may overlap.
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param point Norm data point to append
     * @return true if data was written, false if step already existed (skipped)
     */
    static bool appendTPQNorm(const std::string& filepath,
                              size_t sample_index,
                              const TPQNormPoint& point);
    
    // saveTPQThermodynamics / saveTPQNorm were retired in the
    // minimalist-architecture rev (May 2026): all TPQ trajectory writes go
    // through appendTPQThermodynamics / appendTPQNorm now (which extend
    // an existing chunked dataset rather than reallocating it). Neither
    // batch writer had any callers after the streaming refactor.

    
    /**
     * @brief Load TPQ thermodynamic data for a sample
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @return Vector of thermodynamic data points
     */
    static std::vector<TPQThermodynamicPoint> loadTPQThermodynamics(const std::string& filepath,
                                                                     size_t sample_index);
    
    /**
     * @brief Load TPQ norm data for a sample
     *
     * Internal helper used by `copyTPQSamples` while merging per-rank
     * HDF5 files. No external callers; if you need to consume norm
     * trajectories outside the merge path, prefer
     * `loadTPQThermodynamics` and store norm alongside the existing
     * energy/variance/doublon columns instead of introducing a separate
     * load surface.
     */
    static std::vector<TPQNormPoint> loadTPQNorm(const std::string& filepath,
                                                  size_t sample_index);

    
    /**
     * @brief List all TPQ samples in an HDF5 file
     * 
     * @param filepath Path to HDF5 file
     * @return Vector of sample indices that have data
     */
    static std::vector<size_t> listTPQSamples(const std::string& filepath);
    
    // saveTPQAveragedThermodynamics was retired in the
    // minimalist-architecture rev (May 2026): the post-process step that
    // would have called it now lives in src/cli/workflows.cpp where the
    // averaging logic writes directly to per-sample groups. The helper had
    // zero callers.

    
    // ============================================================================
    // FTLM/LTLM/Hybrid Thermal Results I/O
    // ============================================================================
    
    /**
     * @brief Save FTLM thermodynamic results with error bars to HDF5
     * @param filepath Path to HDF5 file
     * @param temperatures Temperature array
     * @param energy Energy values
     * @param energy_error Energy error bars
     * @param specific_heat Specific heat values
     * @param specific_heat_error Specific heat error bars
     * @param entropy Entropy values
     * @param entropy_error Entropy error bars
     * @param free_energy Free energy values
     * @param free_energy_error Free energy error bars
     * @param total_samples Number of samples used
     * @param method Method name (FTLM, LTLM, Hybrid)
     */
    static void saveFTLMThermodynamics(
        const std::string& filepath,
        const std::vector<double>& temperatures,
        const std::vector<double>& energy,
        const std::vector<double>& energy_error,
        const std::vector<double>& specific_heat,
        const std::vector<double>& specific_heat_error,
        const std::vector<double>& entropy,
        const std::vector<double>& entropy_error,
        const std::vector<double>& free_energy,
        const std::vector<double>& free_energy_error,
        uint64_t total_samples,
        const std::string& method = "FTLM"
    );
    
    /**
     * @brief Save static response results to HDF5
     * @param filepath Path to HDF5 file
     * @param operator_name Name of operator
     * @param temperatures Temperature array
     * @param expectation Expectation values
     * @param expectation_error Error bars
     * @param variance Variance values (optional)
     * @param variance_error Variance error (optional)
     * @param susceptibility Susceptibility values (optional)
     * @param susceptibility_error Susceptibility error (optional)
     * @param total_samples Number of samples
     */
    static void saveStaticResponse(
        const std::string& filepath,
        const std::string& operator_name,
        const std::vector<double>& temperatures,
        const std::vector<double>& expectation,
        const std::vector<double>& expectation_error,
        const std::vector<double>& variance = {},
        const std::vector<double>& variance_error = {},
        const std::vector<double>& susceptibility = {},
        const std::vector<double>& susceptibility_error = {},
        uint64_t total_samples = 1
    );
    
    /**
     * @brief Save dynamical response results with complex values and errors to HDF5
     * @param filepath Path to HDF5 file
     * @param operator_name Name of operator
     * @param frequencies Frequency array
     * @param spectral_real Real part of spectral function
     * @param spectral_imag Imaginary part of spectral function
     * @param error_real Real part of error
     * @param error_imag Imaginary part of error
     * @param total_samples Number of samples
     * @param temperature Temperature (optional metadata)
     */
    static void saveDynamicalResponseFull(
        const std::string& filepath,
        const std::string& operator_name,
        const std::vector<double>& frequencies,
        const std::vector<double>& spectral_real,
        const std::vector<double>& spectral_imag,
        const std::vector<double>& error_real,
        const std::vector<double>& error_imag,
        uint64_t total_samples = 1,
        double temperature = 0.0
    );
    
    // ============================================================================
    // Generic Array Save/Load
    // ============================================================================
    
    /**
     * @brief Generic save for 1D double array with custom path
     */
    static void saveArray(const std::string& filepath,
                         const std::string& dataset_path,
                         const std::vector<double>& data,
                         const std::map<std::string, std::string>& string_attrs = {},
                         const std::map<std::string, double>& double_attrs = {});
    
    /**
     * @brief Generic load for 1D double array
     */
    static std::vector<double> loadArray(const std::string& filepath,
                                        const std::string& dataset_path);
    
    // ============================================================================
    // FTLM Sample Data I/O (replaces ftlm_samples/*.dat and dynamical_samples/*.txt)
    // ============================================================================
    
    /**
     * @brief Structure to hold FTLM dynamical sample data (spectral function)
     */
    struct FTLMDynamicalSample {
        std::vector<double> frequencies;
        std::vector<double> spectral_real;
        std::vector<double> spectral_imag;
    };
    
    /**
     * @brief Structure to hold FTLM static response sample data
     */
    struct FTLMStaticSample {
        std::vector<double> temperatures;
        std::vector<double> expectation;
        std::vector<double> variance;
    };
    
    /**
     * @brief Ensure FTLM sample groups exist in HDF5 file
     */
    static void ensureFTLMSampleGroups(const std::string& filepath);
    
    /**
     * @brief Save FTLM dynamical sample to HDF5 (replaces dynamical_samples/sample_*.txt)
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param sample Sample data
     * @param is_correlation If true, saves to dynamical_correlation group
     */
    static void saveFTLMDynamicalSample(const std::string& filepath,
                                        size_t sample_index,
                                        const FTLMDynamicalSample& sample,
                                        bool is_correlation = false);
    
    /**
     * @brief Save FTLM static response sample to HDF5 (replaces static_samples/sample_*.txt)
     * 
     * @param filepath Path to HDF5 file
     * @param sample_index Sample index
     * @param sample Sample data
     * @param operator_name Name of the operator (optional, for labeling)
     */
    static void saveFTLMStaticSample(const std::string& filepath,
                                     size_t sample_index,
                                     const FTLMStaticSample& sample,
                                     const std::string& operator_name = "");
    
    // ============================================================================
    // Time Correlation I/O (replaces time_corr_*.dat files)
    // ============================================================================
    
    /**
     * @brief Structure to hold time correlation data
     */
    struct TimeCorrelationData {
        std::vector<double> times;
        std::vector<double> correlation_real;
        std::vector<double> correlation_imag;
    };
    
    /**
     * @brief Ensure time correlation groups exist in HDF5 file
     */
    static void ensureTimeCorrelationGroups(const std::string& filepath);
    
    /**
     * @brief Save time correlation data to HDF5 (replaces time_corr_*.dat files)
     * 
     * @param filepath Path to HDF5 file
     * @param operator_name Operator name (e.g., "Sz_Sz", "Sp_Sm")
     * @param sample_index Sample index
     * @param beta Inverse temperature
     * @param data Time correlation data
     * @param label Additional label (e.g., "ground_state", "thermal")
     */
    static void saveTimeCorrelation(const std::string& filepath,
                                    const std::string& operator_name,
                                    size_t sample_index,
                                    double beta,
                                    const TimeCorrelationData& data,
                                    const std::string& label = "");
    
    // loadTimeCorrelation was retired in the minimalist-architecture rev
    // (May 2026): write-only path. Re-introduce by copying loadDataset
    // boilerplate from listTimeCorrelations / saveTimeCorrelation if any
    // post-processing tool ever needs to read this back.

    
    /**
     * @brief List all time correlation datasets in an HDF5 file
     * 
     * @param filepath Path to HDF5 file
     * @return Vector of group paths for each time correlation dataset
     */
    static std::vector<std::string> listTimeCorrelations(const std::string& filepath);
    
    // ============================================================================
    // MPI-Safe HDF5 I/O Functions
    // ============================================================================
    // Industry standard approach: each MPI rank writes to its own file, then
    // rank 0 merges all per-rank files at the end.
    
    /**
     * @brief Get MPI-safe filename for per-rank HDF5 file
     * @param directory Base output directory
     * @param rank MPI rank (0 for serial execution)
     * @param filename Base filename (default: ed_results.h5)
     * @return Full path to per-rank file (e.g., ed_results_rank0.h5)
     */
    static std::string getPerRankFilePath(const std::string& directory,
                                          int rank,
                                          const std::string& filename = "ed_results.h5");
    
    /**
     * @brief Create or open per-rank HDF5 file for MPI-safe writing (SAFE)
     * 
     * SAFE WRITING PROTOCOL:
     * - If file exists, opens in read/write mode (preserves existing data)
     * - If file doesn't exist, creates new file with standard groups
     * - Never truncates existing data
     * 
     * @param directory Output directory
     * @param rank MPI rank
     * @param filename Base filename (default: ed_results.h5)
     * @return Full path to created/opened file
     */
    static std::string createPerRankFile(const std::string& directory,
                                         int rank,
                                         const std::string& filename = "ed_results.h5");
    
    /**
     * @brief Merge TPQ data from per-rank HDF5 files into unified output
     * 
     * This is called by rank 0 after all MPI ranks complete their work.
     * It reads TPQ samples from each rank's file and writes them to the
     * final unified HDF5 file.
     * 
     * @param directory Output directory containing per-rank files
     * @param num_ranks Total number of MPI ranks
     * @param output_filename Name of final output file (default: ed_results.h5)
     * @param delete_temp_files Whether to delete per-rank files after merging
     * @return true if successful
     */
    static bool mergePerRankTPQFiles(const std::string& directory,
                                     int num_ranks,
                                     const std::string& output_filename = "ed_results.h5",
                                     bool delete_temp_files = true);
    
    /**
     * @brief Copy all TPQ samples from source file to destination file
     * 
     * For continue_quenching support, this function properly merges:
     * - thermodynamics: Appends new rows (by step number) to existing data
     * - norm: Appends new rows (by step number) to existing data
     * - states: Copies new states (by beta value) that don't already exist
     * 
     * @param source_path Path to source HDF5 file
     * @param dest_path Path to destination HDF5 file
     * @return Number of samples copied/merged
     */
    static int copyTPQSamples(const std::string& source_path, const std::string& dest_path);
};

#endif // HDF5_IO_H
