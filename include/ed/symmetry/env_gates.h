#pragma once
// =============================================================================
// include/ed/symmetry/env_gates.h -- Stage 10b: ONE inventory of the
// symmetry-stack environment gates.
//
// The stack had grown 104 distinct ED_* variables read by ad-hoc getenv at
// their consumption sites: no single inventory (docs listed ~10 by hand), no
// typo protection, and machine-to-machine behaviour differences (the
// legacy-driver class of failure) were undiagnosable without grepping the
// tree. This header is the SINGLE table for the ED_SYM_* family: every
// gate, its default, and one sentence of meaning --
// and ``dump_env_gates()`` renders the live values for bug reports.
//
// Deliberately an INVENTORY-plus-dump, not a forced rewiring: the existing
// named accessors (``spin_flip_transport_enabled`` & co.) keep their per-call
// getenv semantics (tests toggle without restart); new gates should be added
// HERE first and read through a named accessor. The X-list is the one place
// a gate's name is spelled.
//
// Retired gates (rows removed when their lane died): ED_SYM_FUSED_PASS15 /
// ED_SYM_STREAMING_ENUM (Stage 10e -- fused OrbitTable is the only enum
// path); ED_SYM_LAZY_SECTORS / ED_SYM_LAZY_SECTORS_BYTES_MAX (Stage 11c-1
// -- the lazy rep-first builders are the only construction lane);
// ED_SYM_REP / ED_GPU_SYMMETRY_REP / ED_GPU_SYMMETRY_MIRROR_V2 (Stage 11c-2b
// -- the orbit-CSR matvec backends themselves are gone: the CSR-free rep
// kernel is the ONE representation on host and device).
// =============================================================================

#include <cstdio>
#include <cstdlib>
#include <string>

namespace ed::symmetry {

// X-list: ED_SYM_GATE(name, default_text, description)
#define ED_SYM_GATES(X)                                                        \
    X("ED_SYM_REDUCED_CSR", "unset (auto: reduced-CSR default)",               \
      "=0 forces the CSR-free rep walk; =1 forces reduced-CSR")                \
    X("ED_SYM_SECTOR_CSR_BUDGET_GIB", "8",                                     \
      "per-sector reduced-CSR memory budget; over-budget sectors fall "       \
      "back to the CSR-free walk (both lanes)")                                \
    X("ED_SYM_SECTOR_PARALLEL", "auto (many-tiny-sectors regime)",             \
      "=1/=0 force sector-parallel outer loops on/off")                        \
    X("ED_SYM_PROFILE", "0",                                                   \
      "=1 prints construction + little-group phase timers")                    \
    X("ED_SYM_CACHE", "1",                                                     \
      "=0 disables the .otab disk cache layer")                                \
    X("ED_SYM_CACHE_DIR", "<lattice_dir>/basis_cache",                         \
      "overrides the disk-cache location")                                     \
    X("ED_SYM_SPIN_FLIP", "1",                                                 \
      "=0 disables flip transport (Stage 5a; Auto only, require wins)")        \
    X("ED_SYM_SPIN_FLIP_PROJECT", "1",                                         \
      "=0 disables the (k,+/-) in-sector flip projection (Stage 5b)")          \
    X("ED_SYM_TIME_REVERSAL", "1",                                             \
      "=0 disables conjugate-sector TR pairing (Stage 6)")                     \
    X("ED_SYM_LITTLE_GROUP", "1  [read in Python routing]",                    \
      "=0: point_group=auto degrades to abelian folds; 'full' raises")         \
    X("ED_SYM_LG_FLIP", "1",                                                   \
      "=0 disables A' = A x Z2 inside the little-group engine (Stage 9a)")     \
    X("ED_SYM_LG_TR", "1",                                                     \
      "=0 disables TR star/sigma folding inside the engine (Stage 9b)")        \
    X("ED_SYM_LG_GPU", "auto (device present + block >= 2^20 reps)",           \
      "=0 vetoes the little-group GPU rep gather; =1 drops the dim floor")     \
    X("ED_SYM_LG_SEED", "0",                                                   \
      "offsets the lowest-k Lanczos start vector (multi-seed verification)")   \
    X("ED_SYM_LG_DENSE_FLOOR", "auto (32x Lanczos cap)",                       \
      "raise the dense/Lanczos crossover so larger (e.g. degenerate) blocks "  \
      "solve dense/exactly, at the cost of memory (S1 mitigation)")            \
    X("ED_SYM_PERM_LUT", "1",                                                  \
      "=0 keeps the scalar bit walk (test gate: pin LUT == scalar)")           \
    X("ED_SYM_LG_ONLY_K0", "unset (solve every star)",                         \
      "=\"7,43\" solves only these star reps (job splitting); =\"plan\" "     \
      "lists (k0, |star|, dim) per star and solves nothing")                   \
    X("ED_SYM_SAB_CACHE_BYTES", "536870912",                                   \
      "byte budget of the SAB sector cache (test oracle)")                     \
    X("ED_SYM_CSR_DIM_MAX", "engine default",                                  \
      "dimension cap for assembled sector CSRs (legacy lane)")                 \
    X("ED_SYM_REP_RANKTABLE", "auto (budget)",                                 \
      "=0 disables the dense rank table (binary-search fallback)")             \
    X("ED_SYM_REP_RANKTABLE_BUDGET_GIB", "engine default",                     \
      "memory budget for per-sector dense rank tables")                        \
    X("ED_SYM_SKIP_COMMUTE_CHECK", "0  [read in Python]",                      \
      "=1 skips the [H,U_g]=0 validation of explicit generators")              \
    X("ED_SYM_NO_DETECT_MEMO", "0  [read in Python]",                          \
      "=1 disables the find_symmetries content memo")

/// Render every symmetry gate with its LIVE value (or "(unset)"), default,
/// and meaning -- one call to paste into a bug report. Bound to Python as
/// ``_core.dump_env_gates()``.
[[nodiscard]] inline std::string dump_env_gates() {
    std::string out;
    out += "symmetry env gates (live value | default | meaning)\n";
    char line[512];
#define ED_SYM_GATE_ROW(NAME, DEF, DESC)                                       \
    {                                                                          \
        const char* v = std::getenv(NAME);                                     \
        std::snprintf(line, sizeof(line), "  %-33s %-10s | %s | %s\n", NAME,   \
                      v ? v : "(unset)", DEF, DESC);                           \
        out += line;                                                           \
    }
    ED_SYM_GATES(ED_SYM_GATE_ROW)
#undef ED_SYM_GATE_ROW
    return out;
}

}  // namespace ed::symmetry
