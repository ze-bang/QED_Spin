#pragma once
// =============================================================================
// include/ed/config/env_registry.h -- the ONE table of runtime environment
// variables the library reads.
//
// Every ED_* / QED_* variable is declared here exactly once: name, value type,
// subsystem, default, meaning. Nothing else in the tree may spell a variable name
// in a getenv call (scripts/check_env_registry.sh enforces this in both directions).
//
// Why a table. Variables read ad hoc at their consumption sites have no
// inventory, no protection against a misspelt name, and no record of which ones
// were set, yet they change a run's numerical behaviour. Through this header
//   * env::dump()      renders every variable with its live value (bug reports);
//   * env::snapshot()  returns the variables that ARE set, for result metadata;
//   * env::unknown()   returns ED_* names found in the environment that no
//                      row declares -- almost always a typo that silently did nothing.
//
// Reading a variable. Use the typed accessors; they define ONE meaning of "set":
//   flag      unset or "" -> the default; "0", "false", "off", "no" -> false;
//             anything else -> true.   (Presence alone never enables a flag:
//             FOO=0 means off.)
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

extern "C" {
extern char** environ;   // POSIX; C linkage so it agrees with <unistd.h> when that is included too
}

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
    X("ED_SYM_REP_RANKTABLE_BUDGET_GIB", Real, "symmetry", "8.0",              \
      "Memory budget for the per-sector dense int32 rank table")               \
    X("ED_SYM_REDUCED_CSR", Tristate, "symmetry", "unset -> RepReducedCsr",    \
      "Exactly \"1\" forces the reduced-CSR symmetry matvec, exactly \"0\" the CSR-free rep walk (read once per process)")\
    X("ED_SYM_SECTOR_CSR_BUDGET_GIB", Real, "symmetry", "8.0",                 \
      "AGGREGATE reduced-CSR byte budget; over-budget sectors fall back to the CSR-free walk")\
    X("ED_SYM_LG_GPU", Tristate, "little-group", "auto (device present + block >= 2^20 reps)",\
      "=0 vetoes the little-group GPU lanes; =1 drops the 2^20-rep dim floor") \
    X("ED_SYM_LG_DENSE_FLOOR", Integer, "little-group", "max(dense_max_dim, 4 * max_iter_cap)",\
      "Raises the dense/Lanczos crossover so larger blocks solve exactly (=1 in tests forces the Lanczos path at toy dims)")\
    X("ED_CSR_FORCE", Tristate, "krylov", "-1 (use the dim cutoff)",           \
      "=1 always assemble CSR, =0 never (matrix-free always), unset -> use csr_cutoff_dim")\
    X("ED_CSR_DIM_MAX", Integer, "krylov", "the caller's default cutoff",\
      "Projected-basis dim below which assembled CSR is preferred over matrix-free")\
    X("ED_MATVEC_SCATTER", Flag, "krylov", "false (gather kernel)",            \
      "=1 uses the atomic-scatter SpMV kernel instead of the lock-free row gather (for bisection)")\
    X("ED_LANCZOS_KERNEL_PROFILE", Flag, "krylov", "false",                    \
      "=1 enables per-bucket us timers inside lanczos_kernel (A/B against lanczos_real)")\
    X("ED_THERMAL_EXACT_SMALL", Flag, "thermal", "true",                       \
      "=0 forces the real sampling kernel even at D <= SMALL_THERMAL_DIM instead of the exact dense fallback")\
    X("ED_XSEC_CSR_BUDGET_GIB", Real, "thermal", "4.0",                        \
      "Byte budget for the cross-sector orbit-observable triplet CSR; over budget -> csr_refused_")\
    X("ED_GPU_SYM_CACHE_GIB", Real, "gpu", "24 (rank-table cache) / 16 (sector mirror)",\
      "Byte budget for the device-side strong caches that pin recently-used symmetry tables/mirrors")\
    X("ED_AUTO_THREADS", Flag, "threads-numa", "true (auto-threading enabled)",\
      "=0/false/FALSE/no/NO disables the dim-aware automatic thread-budget scaling; any other value leaves it on")\
    X("ED_NUMA_PIN_THREADS", Flag, "threads-numa", "false",                    \
      "Pins OMP worker threads to cores (irreversible, applied once per process via std::once_flag)")\
    X("ED_MEM_GUARD_OFF", Flag, "memory-guard", "false (guard active)",        \
      "Disables the \"estimated working set exceeds ~90% of available RAM\" pre-allocation throw")\
    X("QED_CORE_DIR", Path, "python", "unset (extension inside the package)",  \
      "Prepends a build directory containing _core*.so to qed.__path__")       \
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
