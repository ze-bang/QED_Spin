#pragma once
// =============================================================================
// include/ed/core/config.h -- the ONE table of runtime environment
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
//             "1", "true", "on", "yes" or another integer -> true (any case).
//             (Presence alone never enables a flag: FOO=0 means off.)
//   tristate  unset or "" -> nullopt (the engine decides); otherwise as flag.
//   integer / real   unset or "" -> the default; the whole value must parse ("8GB" does not).
//   text      unset -> the default; "" is returned as "".
// A set value that does not parse as its kind is refused: malformed() lists them, and the
// Python package (at import) and every ed::sectors verb (before any work) raise on it. The
// accessors themselves never throw (they run deep inside the engine) and fall back to the
// default, which the up-front check makes unreachable.
// Accessors read the environment on every call (tests toggle gates without
// restarting the process). A caller that needs a value fixed for the process
// lifetime caches it in a function-local static, and says so in its row.
//
// PRECEDENCE: an explicit function argument or option field always wins over the
// environment, which wins over the built-in default. An accessor is therefore
// consulted only when the corresponding option is unset.
// =============================================================================

#include <cctype>
#include <cerrno>
#include <cmath>
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
      "Logs symmetry-construction / little-group phase timers and lane-decline reasons at Info")\
    X("ED_SYM_REP_RANKTABLE_BUDGET_GIB", Real, "symmetry", "0.5",              \
      "Memory budget for the per-sector dense int32 rank table; 0 builds none")\
    X("ED_SYM_REDUCED_CSR", Tristate, "symmetry", "unset -> RepReducedCsr",    \
      "A true word (1) forces the reduced-CSR symmetry matvec, a false word (0) the CSR-free rep walk (read once per process)")\
    X("ED_SYM_SECTOR_CSR_BUDGET_GIB", Real, "symmetry", "unset -> 0.55 x available RAM less the solver working set", \
      "AGGREGATE reduced-CSR byte budget (an absolute cap when set); over-budget sectors (all of them at 0) fall back to the CSR-free walk")\
    X("ED_SYM_REAL", Flag, "symmetry", "1",                                    \
      "Runs the host Krylov lanes of a real block (real matrix elements in a dictionary CSR) in real arithmetic; 0 keeps every block complex")\
    X("ED_SYM_LG_GPU", Tristate, "little-group", "auto (device present + block >= 2^20 reps)",\
      "=0 vetoes the little-group GPU lanes; =1 drops the 2^20-rep dim floor") \
    X("ED_CSR_FORCE", Tristate, "krylov", "-1 (use the dim cutoff)",           \
      "Full-space Operator.apply only: =1 always assemble CSR, =0 never (matrix-free), unset -> the dim cutoff")\
    X("ED_CSR_DIM_MAX", Integer, "krylov", "the caller's default cutoff",\
      "Full-space Operator.apply only: dim below which the assembled CSR is preferred over matrix-free")\
    X("ED_LANCZOS_KERNEL_PROFILE", Flag, "krylov", "false",                    \
      "=1 logs per-bucket us timers inside lanczos_kernel at Info")\
    X("ED_XSEC_CSR_BUDGET_GIB", Real, "dynamics", "4.0",                       \
      "Byte budget (exact pre-merge bound) for the host CSR of O between two sectors in dynamics; over budget -> row walk")\
    X("ED_GPU_SYM_CACHE_GIB", Real, "gpu", "25% (rank-table cache) / 15% (sector mirror) of the device, at most 24 / 16",\
      "Byte budget for the device-side strong caches that pin recently-used symmetry tables/mirrors; 0 pins none")\
    X("ED_GPU_DENSE_BATCH_GIB", Real, "gpu", "2 (and at most a quarter of the free device memory and of the RAM)",\
      "Matrices per batched device dense solve; a larger block is solved by itself on the device when it fits")\
    X("ED_GPU_CSR_BUDGET_GIB", Real, "gpu", "unset -> half the free device memory at the bind",\
      "Byte budget for an operator's reduced CSR built on the device; over it (or 0) the device walk serves")\
    X("ED_AUTO_THREADS", Flag, "threads-numa", "true (auto-threading enabled)",\
      "A false word (0, false, off, no) disables the dim-aware automatic thread-budget scaling")\
    X("ED_NUMA_PIN_THREADS", Flag, "threads-numa", "false",                    \
      "Pins OMP worker threads to cores (irreversible, applied once per process via std::once_flag)")\
    X("ED_MEM_GUARD_OFF", Flag, "memory-guard", "false (guard active)",        \
      "Disables the \"estimated working set exceeds ~90% of available RAM\" pre-allocation throw")\
    X("QED_CORE_DIR", Path, "python", "unset (extension inside the package)",  \
      "Prepends a build directory containing _core*.so to qed.__path__")       \
    X("ED_ENV_STRICT", Flag, "python", "false",                                  \
      "=1 makes an undeclared ED_* variable in the environment an import error instead of a warning") \
    X("QED_LOG_LEVEL", Text, "python", "warn (ED_SYM_PROFILE=1: info)",          \
      "Log level at import (off|error|warn|info|debug); above warn the records also stream to stderr") \
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

/// Strict parses: the whole value, trailing blanks allowed.
[[nodiscard]] inline bool parses_flag(const char* v) {
    std::string s(v);
    for (auto& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    for (const char* w : {"0", "1", "false", "true", "off", "on", "no", "yes"})
        if (s == w) return true;
    errno = 0;
    char* end = nullptr;
    (void)std::strtoll(v, &end, 10);
    if (errno != 0 || end == v) return false;
    while (*end == ' ' || *end == '\t') ++end;
    return *end == '\0';
}
[[nodiscard]] inline bool parses_integer(const char* v) {
    errno = 0;
    char* end = nullptr;
    (void)std::strtoll(v, &end, 10);
    if (errno != 0 || end == v) return false;
    while (*end == ' ' || *end == '\t') ++end;
    return *end == '\0';
}
[[nodiscard]] inline bool parses_real(const char* v) {
    errno = 0;
    char* end = nullptr;
    const double x = std::strtod(v, &end);
    if (errno != 0 || end == v || !std::isfinite(x)) return false;
    while (*end == ' ' || *end == '\t') ++end;
    return *end == '\0';
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

/// "NAME=value" for every set registered variable whose value does not parse as its kind
/// (flag / tristate / integer / real), in table order; empty when all parse.
[[nodiscard]] inline std::vector<std::string> malformed() {
    std::vector<std::string> out;
    for (const auto& r : rows()) {
        const char* v = std::getenv(r.name);
        if (v == nullptr || v[0] == '\0') continue;
        bool ok = true;
        switch (r.kind) {
            case Kind::Flag: case Kind::Tristate: ok = parses_flag(v); break;
            case Kind::Integer: ok = parses_integer(v); break;
            case Kind::Real: ok = parses_real(v); break;
            default: break;
        }
        if (!ok) out.push_back(std::string(r.name) + "=" + v);
    }
    return out;
}

/// ED_* names present in the environment that no row declares. QED_* is not scanned:
/// that prefix is shared with job scripts and sibling packages (QED_NLCE_CACHE, ...),
/// ED_BUILD_* belongs to the build scripts, ED_TEST_* / ED_BENCH_* to the test and
/// benchmark harnesses.
[[nodiscard]] inline std::vector<std::string> unknown() {
    std::vector<std::string> out;
    for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
        const char* s = *e;
        if (std::strncmp(s, "ED_", 3) != 0) continue;
        bool harness = false;
        for (const char* p : {"ED_BUILD_", "ED_TEST_", "ED_BENCH_"})
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
