#pragma once
// =============================================================================
// include/ed/config/env_registry.h -- the ONE table of runtime environment
// variables the library reads.
//
// Every ED_* / QED_* variable is declared here exactly once: name, value type,
// subsystem, default, meaning. Nothing else in the tree may spell a variable name
// in a getenv call (scripts/check_env_registry.sh enforces this in both directions).
//
// Why a table. A run's numerical behaviour depended on ~100 undeclared inputs read
// ad hoc at their consumption sites, with no inventory, no protection against a
// misspelt name, and no record of which ones were set. Through this header
//   * env::dump()      renders every variable with its live value (bug reports);
//   * env::snapshot()  returns the variables that ARE set, for result metadata;
//   * env::unknown()   returns ED_* names found in the environment that no
//                      row declares -- almost always a typo that silently did nothing.
//
// Reading a variable. Use the typed accessors; they define ONE meaning of "set":
//   flag      unset or "" -> the default; "0", "false", "off", "no" -> false;
//             anything else -> true.   (Presence alone never enables a flag:
//             FOO=0 used to switch several of them ON.)
//   tristate  unset or "" -> nullopt (the engine decides); otherwise as flag.
//   integer / real   unset, "" or unparsable -> the default.
//   text      unset -> the default; "" is returned as "".
// Accessors read the environment on every call (tests toggle gates without
// restarting the process). A caller that needs a value fixed for the process
// lifetime caches it in a function-local static, and says so in its row.
//
// PRECEDENCE: an explicit function argument or option field always wins over the
// environment, which wins over the built-in default. An accessor is therefore
// consulted only when the corresponding option is unset.
// =============================================================================

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <utility>
#include <vector>

extern char** environ;

namespace ed::env {

enum class Kind { Flag, Tristate, Integer, Real, Text, Path };

struct Row {
    const char* name;
    Kind        kind;
    const char* scope;
    const char* default_text;
    const char* meaning;
};

// X(name, kind, scope, default_text, meaning)
#define ED_ENV_TABLE(X)                                                        \
    X("ED_SYM_PROFILE", Flag, "symmetry", "0",                                 \
      "Prints symmetry-construction / little-group phase timers and lane-decline reasons to stderr")\
    X("ED_SYM_CACHE", Flag, "symmetry", "true (enabled)",                      \
      "=0 disables the .otab orbit-table disk cache layer")                    \
    X("ED_SYM_CACHE_DIR", Path, "symmetry", "\"\" -> caller falls back to <lattice_dir>/basis_cache",\
      "Overrides the on-disk symmetry-cache location")                         \
    X("ED_SYM_PERM_LUT", Flag, "symmetry", "true (LUT built)",                 \
      "=0 keeps the scalar bit-walk permutation path instead of the byte-decomposition LUT (test gate: pin LUT == scalar)")\
    X("ED_SYM_REP_RANKTABLE", Tristate, "symmetry", "unset -> budget decides", \
      "=0 forces binary-search rank lookup, =1 forces the dense combinadic rank table regardless of budget")\
    X("ED_SYM_REP_RANKTABLE_BUDGET_GIB", Real, "symmetry", "8.0",              \
      "Memory budget for the per-sector dense int32 rank table")               \
    X("ED_SYM_REQUIRE_ABELIAN", Flag, "symmetry", "false (auto-restrict)",     \
      "=1 makes a non-abelian generator set a hard error instead of auto-restricting to a maximal abelian subgroup")\
    X("ED_SYM_SPIN_FLIP", Flag, "symmetry", "true",                            \
      "=0 disables the Sz flip transport (spec(n_up) = spec(N - n_up))")       \
    X("ED_SYM_SPIN_FLIP_PROJECT", Flag, "symmetry", "true",                    \
      "=0 disables the (k,+-) in-sector flip projection")                      \
    X("ED_SYM_TIME_REVERSAL", Flag, "symmetry", "true",                        \
      "=0 disables conjugate-sector TR pairing")                               \
    X("ED_SYM_SU2", Flag, "symmetry", "true",                                  \
      "=0 vetoes the SU(2) total-spin axis (labels, Lowdin targeting, S-resolution)")\
    X("ED_SYM_SU2_REPROJECT_FREQ", Integer, "symmetry", "1",                   \
      "Lowdin drift-scrub cadence: project every k-th wrapped apply; 0 = seed projection only")\
    X("ED_SYM_REDUCED_CSR", Tristate, "symmetry", "Auto -> RepReducedCsr",     \
      "=1 forces the reduced-CSR symmetry matvec, =0 forces the CSR-free rep walk")\
    X("ED_SYM_SECTOR_CSR_BUDGET_GIB", Real, "symmetry", "8.0",                 \
      "AGGREGATE reduced-CSR byte budget; over-budget sectors fall back to the CSR-free walk")\
    X("ED_SYM_SECTOR_PARALLEL", Flag, "symmetry", "auto (many-tiny-sectors heuristic)",\
      "=1/=0 force the sector-parallel outer OMP loops on/off")                \
    X("ED_SYM_CSR_DIM_MAX", Integer, "symmetry", "0 -> falls back to ED_CSR_DIM_MAX, then the caller default 1<<13",\
      "Symmetry-lane CSR-vs-matrix-free dimension cutoff")                     \
    X("ED_SYM_LG_FLIP", Flag, "little-group", "true",                          \
      "=0 disables A' = A x Z2 inside the little-group engine")                \
    X("ED_SYM_LG_TR", Flag, "little-group", "true",                            \
      "=0 disables TR star/sigma folding inside the engine")                   \
    X("ED_SYM_LG_GPU", Tristate, "little-group", "auto (device present + block >= 2^20 reps)",\
      "=0 vetoes the little-group GPU lanes; =1 drops the 2^20-rep dim floor") \
    X("ED_SYM_LG_SEED", Integer, "little-group", "0 (base seed 0x51ED0B70)",   \
      "Offsets the lowest-k Lanczos start vector (multi-seed degeneracy verification)")\
    X("ED_SYM_LG_DENSE_FLOOR", Integer, "little-group", "max(dense_max_dim, 4 * max_iter_cap)",\
      "Raises the dense/Lanczos crossover so larger blocks solve exactly (=1 in tests forces the Lanczos path at toy dims)")\
    X("ED_SYM_LG_DENSE_BATCH_GIB", Real, "little-group", "8.0 (8ULL << 30 bytes)",\
      "Byte budget for the CPU deferred dense-eigensolve batch; =0 solves every block inline")\
    X("ED_SYM_LG_TWO_PASS_MIN_DIM", Integer, "little-group", "1 << 22 (4.2M)", \
      "Dim floor above which the GS vector uses two-pass no-reorth Lanczos instead of FullCGS2 + kept basis (memory cap)")\
    X("ED_SYM_LG_LOWEST_MAX_ITER", Integer, "little-group", "dflt argument if > 0, else max(40*k, 400)",\
      "Absolute Lanczos budget for the lowest-k eigenvalue scan")              \
    X("ED_SYM_LG_GS_MAX_ITER", Integer, "little-group", "the caller's dflt argument (600 two-pass lane / 200 small-n lane)",\
      "Per-attempt Lanczos budget for the certified GS vector")                \
    X("ED_SYM_LG_GS_RESTARTS", Integer, "little-group", "4",                   \
      "Restart count for the two-pass GS lane; total work <= (1+restarts) x GS_MAX_ITER")\
    X("ED_SYM_LG_GS_RESID_TOL", Real, "little-group", "1e-8",                  \
      "Residual acceptance tolerance for the certified little-group GS vector")\
    X("ED_SYM_LG_ONLY_K0", Text, "little-group", "unset (solve every star)",   \
      "=\"7,43\" solves only those star reps (job splitting); =\"plan\" lists (k0, |star|, dim) per star and solves nothing")\
    X("ED_SYM_LITTLE_GROUP", Flag, "little-group", "\"1\"",                    \
      "=0: point_group='auto' degrades to abelian folds; 'full' raises")       \
    X("ED_SYM_LG_THERMAL", Flag, "little-group", "\"1\"",                      \
      "=0 restores pre-U1b behaviour: thermal 'auto' never projects, sampling methods keep the abelian sector lane")\
    X("ED_SYM_CLIQUE_BUDGET", Integer, "symmetry", "512 (_DEFAULT_CLIQUE_BUDGET)",\
      "|Aut| above which find_symmetries switches from exact max-clique to greedy maximal-abelian (hang guard)")\
    X("ED_SYM_NO_DETECT_MEMO", Flag, "symmetry", "\"0\" (memo on)",            \
      "=1 disables the find_symmetries content memo")                          \
    X("ED_SYM_SKIP_COMMUTE_CHECK", Flag, "symmetry", "\"0\" (check on)",       \
      "=1 skips the [H, U_g] = 0 validation of explicit generators")           \
    X("ED_CSR_FORCE", Tristate, "krylov", "-1 (use the dim cutoff)",           \
      "=1 always assemble CSR, =0 never (matrix-free always), unset -> use csr_cutoff_dim")\
    X("ED_CSR_DIM_MAX", Integer, "krylov", "the factory's default_cutoff argument: 1<<20 full-Hilbert, 1<<22 fi...",\
      "Projected-basis dim below which assembled CSR is preferred over matrix-free")\
    X("ED_MATVEC_SCATTER", Flag, "krylov", "false (gather kernel)",            \
      "=1 falls back to the legacy atomic-SCATTER SpMV kernel (bisection escape hatch) instead of the lock-free row-GATHER")\
    X("ED_FIXED_SZ_TABLELESS", Tristate, "memory-guard", "unset -> planner slot, then the dimension/budget heuristic",\
      "=1 forces the tableless combinadic fixed-Sz basis, =0 forces the materialized list")\
    X("ED_FIXED_SZ_TABLE_BUDGET_GIB", Real, "memory-guard", "16.0",            \
      "Byte budget for the materialized fixed-Sz basis list; above it the basis flips to tableless")\
    X("ED_LANCZOS_VERBOSE", Flag, "krylov", "false",                           \
      "=1 re-enables per-iteration progress prints inside the Lanczos inner loops")\
    X("ED_LANCZOS_PROFILE", Flag, "krylov", "false",                           \
      "=1 enables the per-iteration us timing breakdown in lanczos_real")      \
    X("ED_LANCZOS_KERNEL_PROFILE", Flag, "krylov", "false",                    \
      "=1 enables per-bucket us timers inside lanczos_kernel (A/B against lanczos_real)")\
    X("ED_LANCZOS_COMPLEX_SEED", Flag, "krylov", "false (real-only seed)",     \
      "=1 reverts the random Krylov seed to fully complex (legacy behaviour)") \
    X("ED_LANCZOS_REORTH_K", Integer, "krylov", "1 (kernel lanes); legacy real lane clamps to [0,4]",\
      "Local-reorthogonalisation ring width for the LocalDGKS3 policy")        \
    X("ED_LANCZOS_CHECK_EVERY", Integer, "krylov", "5 at both sites",          \
      "Ritz-convergence check cadence (=1 restores per-iteration checking)")   \
    X("ED_LANCZOS_EIGVEC_TWOPASS", Flag, "krylov", "true (two-pass on)",       \
      "=0 restores the kept-basis FullCGS2 eigenvector lane instead of the two-pass no-reorth reconstruction")\
    X("ED_FORCE_COMPLEX_LANCZOS", Flag, "krylov", "false",                     \
      "=1 returns the pre-Wave-1.1 unified complex Lanczos kernel (A/B + bisection)")\
    X("ED_LANCZOS_REAL_DISPATCH", Flag, "krylov", "true (real dispatch on)",   \
      "=0 opts out of dispatching eigenvalue-only Lanczos to the real-arithmetic fast path")\
    X("ED_BLOCK_LANCZOS_LEAN", Flag, "krylov", "false (honour the caller)",    \
      "=1 forces keep_basis = false in the block-Lanczos eigenvalue-only lane")\
    X("ED_GPU_LANCZOS_FULL_CGS2", Flag, "krylov", "false",                     \
      "=1 restores FullCGS2 + keep_basis in the GPU Lanczos facade (pre-Wave-4.1/4.2 defaults)")\
    X("ED_FULLDIAG_DENSE_MAX", Integer, "krylov", "120000",                    \
      "Dimension threshold below which full diagonalization uses the dense LAPACK path")\
    X("ED_FULLDIAG_THREADS", Integer, "krylov", "max(1, N / 1024)",            \
      "OMP/BLAS team size for the dense dsytrd/zhetrd reduction")              \
    X("ED_FULLDIAG_FORCE_COMPLEX", Flag, "krylov", "false (real fast path allowed)",\
      "Forces the complex LAPACK driver even when the assembled matrix is real (A/B + equivalence checks)")\
    X("ED_LANCZOS_REORTH_TILE", Integer, "krylov", "16",                       \
      "Tile width for the disk-backed full-reorthogonalisation pass")          \
    X("ED_LANCZOS_DISK", Flag, "krylov", "false",                              \
      "=1 forces the Lanczos basis onto disk instead of in-memory buffers")    \
    X("ED_LANCZOS_CHECKPOINT_DIR", Path, "io-hdf5", "\"\" (checkpointing disabled)",\
      "Directory for the HDF5 Lanczos checkpoint; non-empty enables checkpointing")\
    X("ED_LANCZOS_CHECKPOINT_INTERVAL", Integer, "io-hdf5", "100",             \
      "Iterations between checkpoint writes")                                  \
    X("ED_LANCZOS_RESUME", Flag, "io-hdf5", "false",                           \
      "Resume a Lanczos run from the checkpoint file if one exists")           \
    X("ED_THERMAL_EXACT_SMALL", Flag, "thermal-kpm", "true",                   \
      "=0 forces the real sampling kernel even at D <= SMALL_THERMAL_DIM instead of the exact dense fallback")\
    X("ED_MTPQ_VERBOSE", Flag, "thermal-kpm", "false",                         \
      "Prints the mTPQ fp32-vs-double lane-selection decision to stderr")      \
    X("ED_TPQ_BASE_SEED", Integer, "thermal-kpm", "0 -> non-deterministic, time-seeded",\
      "Non-zero value puts TPQ per-sample seeding in deterministic mode (identical CPU and GPU)")\
    X("ED_DSSF_VERBOSE", Flag, "thermal-kpm", "false",                         \
      "Gates per-sample / per-iteration progress prints in the DSSF/SSSF/FTLM/LTLM kernels")\
    X("ED_FTLM_PARALLEL", Flag, "thermal-kpm", "false (serial samples)",       \
      "=1 opts in to OMP-parallel FTLM samples (only safe with a thread-safe Hv callback)")\
    X("ED_KPM_VERBOSE", Flag, "thermal-kpm", "false",                          \
      "Verbose logging in the FTLM-KPM kernel")                                \
    X("ED_KPM_DOS_VERBOSE", Flag, "thermal-kpm", "false",                      \
      "Verbose logging in the CPU KPM-DOS kernel; also the fallback name for the GPU kernel")\
    X("ED_KPM_DOS_GPU_VERBOSE", Flag, "thermal-kpm", "false -> falls back to ED_KPM_DOS_VERBOSE",\
      "Verbose logging in the GPU KPM-DOS kernel")                             \
    X("ED_KPM_SAMPLE_THREADS", Integer, "thermal-kpm", "1 (serial)",           \
      "Outer OMP team size over KPM random samples")                           \
    X("ED_KPM_NUM_MOMENTS", Integer, "thermal-kpm", "KPMDOSParameters::num_moments (from config.dynamical.krylov_dim, se...",\
      "Chebyshev moment count for the KPM-DOS workflow")                       \
    X("ED_KPM_NUM_QUAD", Integer, "thermal-kpm", "KPMDOSParameters::num_quadrature_nodes default",\
      "Quadrature-node count for the KPM thermodynamic integrals")             \
    X("ED_KPM_BOUND_BUFFER", Real, "thermal-kpm", "KPMDOSParameters::spectral_bound_buffer default",\
      "Safety buffer on the estimated spectral bounds before Chebyshev rescaling")\
    X("ED_KPM_KERNEL", Text, "thermal-kpm", "Jackson kernel (use_jackson_kernel = true)",\
      "Selects the Chebyshev damping kernel")                                  \
    X("ED_KPM_LORENTZ_LAMBDA", Real, "thermal-kpm", "KPMDOSParameters::lorentz_lambda default",\
      "lambda parameter of the Lorentz kernel")                                \
    X("ED_DSSF_PAIR_THREADS", Integer, "thermal-kpm", "min(n_my_pairs, max(1, omp_max_threads / 2))",\
      "Outer OMP team size over DSSF (q, omega) pair tasks")                   \
    X("ED_XSEC_CSR_BUDGET_GIB", Real, "thermal-kpm", "4.0",                    \
      "Byte budget for the cross-sector orbit-observable triplet CSR; over budget -> csr_refused_")\
    X("ED_GPU_OPERATOR_MIRROR", Flag, "gpu", "true (mirror on)",               \
      "=0 disables the full-Hilbert / fixed-Sz device mirror (CPU-vs-GPU bisection)")\
    X("ED_GPU_SYMMETRY_MIRROR", Flag, "gpu", "true (mirror on)",               \
      "=0 disables the symmetry-sector device mirror, so select_backend stays on the CPU lane")\
    X("ED_GPU_SYM_CACHE_GIB", Real, "gpu", "24 (rank-table cache) / 16 (sector mirror)",\
      "Byte budget for the device-side strong caches that pin recently-used symmetry tables/mirrors")\
    X("ED_GPU_SYNC_LAUNCH", Flag, "gpu", "false (async launches)",             \
      "=1 restores a host sync after every matvec launch (diagnostic for the 2026-09-11 async change)")\
    X("ED_GPU_GATHER_WARP", Flag, "gpu", "false (thread-per-row kernel)",      \
      "=1 selects the warp-per-row GPU gather kernel instead of thread-per-row")\
    X("ED_GPU_TIMING", Flag, "gpu", "false",                                   \
      "=1 enables per-call CUDA event timing (forces a host sync, 30-50% of wall time at small N)")\
    X("ED_GPU_MIXED_PRECISION_SPMV", Flag, "gpu", "false",                     \
      "=1 enables the FP32 CSR SpMV cache on the GPU lane")                    \
    X("ED_GPU_CUSPARSE_MIN_DIM", Integer, "gpu", "32768",                      \
      "Dimension below which cuSPARSE CSR is skipped in favour of the matrix-free fused kernel")\
    X("ED_GPU_DISABLE_CUSPARSE", Flag, "gpu", "false",                         \
      "=1 disables the cuSPARSE assembled-CSR pathway entirely")               \
    X("ED_GPU_ALLOW_DROPPED_THREEBODY", Flag, "gpu", "false (hard error)",     \
      "=1 acknowledges that three-body terms are silently dropped by the GPU kernel; without it the load THROWS")\
    X("ED_AUTO_THREADS", Flag, "threads-numa", "false (auto-threading enabled)",\
      "=0/false/no disables the dim-aware automatic thread-budget scaling")    \
    X("ED_AUTO_THREADS_PER_K", Integer, "threads-numa", "8",                   \
      "Aim for one OMP/BLAS worker per K * 1024 basis states")                 \
    X("ED_AUTO_THREADS_CEIL", Integer, "threads-numa", "8",                    \
      "Soft cap on the auto-derived thread count; =0 disables the soft cap (use min(dim/per_k, max_t))")\
    X("ED_NUMA_FIRST_TOUCH", Flag, "threads-numa", "false",                    \
      "Enables NUMA-aware first-touch page placement for large vectors")       \
    X("ED_NUMA_PIN_THREADS", Flag, "threads-numa", "false",                    \
      "Pins OMP worker threads to cores (irreversible, applied once per process via std::once_flag)")\
    X("ED_MEM_GUARD_OFF", Flag, "memory-guard", "false (guard active)",        \
      "Disables the \"estimated working set exceeds ~90% of available RAM\" pre-allocation throw")\
    X("ED_HDF5_COMPRESSION_LEVEL", Integer, "io-hdf5", "4",                    \
      "Deflate level 0 (off) .. 9 (max) for HDF5 datasets")                    \
    X("ED_HDF5_CHUNK_TARGET_BYTES", Integer, "io-hdf5", "256 * 1024 (256 KiB)",\
      "Target HDF5 chunk size in bytes")                                       \
    X("ED_HDF5_SHUFFLE", Flag, "io-hdf5", "true",                              \
      "Enables the HDF5 shuffle filter before deflate")                        \
    X("ED_TIME_CONSTRUCTION", Flag, "debug", "false",                          \
      "Times and reports the sector/operator construction phase")              \
    X("ED_DEBUG_BALANCE", Flag, "debug", "false",                              \
      "Dumps Burnside sector dims and the per-MPI-rank load after greedy owner assignment")\
    X("ED_PYTHON", Path, "debug", "\"python3\"",                               \
      "Custom Python interpreter used to shell out to automorphism_finder.py") \
    X("QED_CORE_DIR", Path, "python", "unset (extension inside the package)",  \
      "Prepends a build directory containing _core*.so to qed.__path__")       \
    X("QED_SZ_WORKERS", Integer, "python", "len(_n_up_values) (one worker per Sz sector)",\
      "Process/thread fan-out across Sz sectors in the thermal sweep")         \
    X("ED_VERBOSE_TRILINEAR", Flag, "debug", "\"0\"",                          \
      "Prints trilinear-triplet counts when building the pyrochlore super-exchange term list")\
    X("ED_ENV_STRICT", Flag, "python", "false",                                  \
      "=1 makes an undeclared ED_* variable in the environment an import error instead of a warning") \
    /* end of table */

inline const std::vector<Row>& rows() {
    static const std::vector<Row> table = {
#define ED_ENV_ROW(NAME, KIND, SCOPE, DEF, MEANING) Row{NAME, Kind::KIND, SCOPE, DEF, MEANING},
        ED_ENV_TABLE(ED_ENV_ROW)
#undef ED_ENV_ROW
    };
    return table;
}

[[nodiscard]] inline bool is_registered(const char* name) {
    for (const auto& r : rows())
        if (std::strcmp(r.name, name) == 0) return true;
    return false;
}

// ---- typed accessors ---------------------------------------------------------
/// The raw value, or nullptr when unset. Prefer the typed accessors.
[[nodiscard]] inline const char* raw(const char* name) { return std::getenv(name); }

[[nodiscard]] inline bool is_false_word(const char* v) {
    return std::strcmp(v, "0") == 0 || std::strcmp(v, "false") == 0 || std::strcmp(v, "FALSE") == 0 ||
           std::strcmp(v, "off") == 0 || std::strcmp(v, "OFF") == 0 || std::strcmp(v, "no") == 0 ||
           std::strcmp(v, "NO") == 0;
}

[[nodiscard]] inline std::optional<bool> tristate(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return std::nullopt;
    return !is_false_word(v);
}

[[nodiscard]] inline bool flag(const char* name, bool dflt) {
    return tristate(name).value_or(dflt);
}

[[nodiscard]] inline long long integer(const char* name, long long dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return dflt;
    errno = 0;
    char* end = nullptr;
    const long long x = std::strtoll(v, &end, 10);
    return (errno != 0 || end == v) ? dflt : x;
}

[[nodiscard]] inline double real(const char* name, double dflt) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return dflt;
    errno = 0;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    return (errno != 0 || end == v) ? dflt : x;
}

[[nodiscard]] inline std::string text(const char* name, const char* dflt = "") {
    const char* v = std::getenv(name);
    return std::string(v != nullptr ? v : dflt);
}

// ---- introspection -----------------------------------------------------------
/// (name, value) for every registered variable that is set, in table order.
[[nodiscard]] inline std::vector<std::pair<std::string, std::string>> snapshot() {
    std::vector<std::pair<std::string, std::string>> out;
    for (const auto& r : rows())
        if (const char* v = std::getenv(r.name)) out.emplace_back(r.name, v);
    return out;
}

/// ED_* names present in the environment that no row declares. QED_* is not scanned:
/// that prefix is shared with job scripts and sibling packages (QED_NLCE_CACHE, ...),
/// ED_BUILD_* belongs to the build scripts, ED_TEST_* / ED_BENCH_* / ED_KILL_HASH_GATE_* to
/// the test and benchmark harnesses.
[[nodiscard]] inline std::vector<std::string> unknown() {
    std::vector<std::string> out;
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        const char* s = *e;
        if (std::strncmp(s, "ED_", 3) != 0) continue;
        bool harness = false;
        for (const char* p : {"ED_BUILD_", "ED_TEST_", "ED_BENCH_", "ED_KILL_HASH_GATE_"})
            harness = harness || std::strncmp(s, p, std::strlen(p)) == 0;
        if (harness) continue;
        const char* eq = std::strchr(s, '=');
        std::string name(s, eq != nullptr ? static_cast<std::size_t>(eq - s) : std::strlen(s));
        if (!is_registered(name.c_str())) out.push_back(std::move(name));
    }
    return out;
}

/// Every variable with its live value, default and meaning; `prefix` filters by name.
[[nodiscard]] inline std::string dump(const char* prefix = "") {
    std::string out = "environment variables (live value | default | meaning)\n";
    const char* scope = "";
    char line[640];
    for (const auto& r : rows()) {
        if (std::strncmp(r.name, prefix, std::strlen(prefix)) != 0) continue;
        if (std::strcmp(scope, r.scope) != 0) {
            scope = r.scope;
            std::snprintf(line, sizeof(line), " [%s]\n", scope);
            out += line;
        }
        const char* v = std::getenv(r.name);
        std::snprintf(line, sizeof(line), "  %-34s %-12s | %s | %s\n", r.name,
                      v != nullptr ? v : "(unset)", r.default_text, r.meaning);
        out += line;
    }
    return out;
}

}  // namespace ed::env
