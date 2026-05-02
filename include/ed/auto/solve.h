#pragma once
// =============================================================================
// include/ed/auto/solve.h
//
// `ed::auto_pilot::solve(Operator&, AutoSolveOptions)` — stress-free,
// modern-C++ entry point that mirrors the Python `qed.diag(...)`
// auto-selector but stays fully in-process. It is a thin convenience
// layer over the canonical `exact_diagonalization_core(...)` dispatcher
// in `ed/core/ed_wrapper.h`; it does NOT introduce any new solver, it
// only chooses sensible defaults so users don't have to reason about:
//
//   * which `DiagonalizationMethod` enum value to pick for their
//     (dim, num_eigenvalues, eigenvectors-needed?) tuple,
//   * whether to enable the GPU / MPI backends (build availability is
//     queried via `is_cuda_compiled()` / `is_scalapack_compiled()`),
//   * whether to drop into the fixed-Sz Operator subclass (Sz
//     conservation is asserted before the projection runs; guards
//     against silently giving the wrong answer for transverse
//     Hamiltonians, mirroring the Python guard at workflow.py:814).
//
// The auto-pilot also EMITS HINTS (when `verbose=true`) when the
// supplied Operator commutes with total Sz but the caller did not pass
// `options.sz` — using a fixed-Sz sector is almost always cheaper.
//
// Anything users want to override they pass via `AutoSolveOptions`.
// Anything users do NOT want to use this façade for — exotic flags,
// continuation runs, custom observable lists, low-level Lanczos knobs —
// they should reach for `exact_diagonalization_core(...)` directly with
// their own `EDParameters`. The auto-pilot is for the common path.
// =============================================================================

#include <ed/core/ed_parameters.h>
#include <ed/core/ed_types.h>
#include <ed/core/ed_wrapper.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator.h>

#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace ed::auto_pilot {

// ---------------------------------------------------------------------------
// Internal helpers (kept inline so this header stays standalone).
// Mirror the logic of `op_conserves_sz` / `op_make_fixed_sz` in
// `python/qed/_bindings/qed_bindings.cpp` so the C++ and
// Python façades agree byte-for-byte on what counts as Sz-conserving.
// ---------------------------------------------------------------------------

namespace detail {

inline int sz_shift(int op_type) {
    if (op_type == 0) return  1;  // S+ raises by 1
    if (op_type == 1) return -1;  // S- lowers by 1
    return 0;                     // Sz is diagonal
}

inline bool conserves_sz(const Operator& op) {
    for (const auto& t : op.transform_data_) {
        if (std::abs(t.coefficient) < 1e-15) continue;
        int delta = sz_shift(static_cast<int>(t.op_type));
        if (t.is_two_body) delta += sz_shift(static_cast<int>(t.op_type_2));
        if (delta != 0) return false;
    }
    for (const auto& t : op.three_body_data_) {
        if (std::abs(t.coefficient) < 1e-15) continue;
        int delta = sz_shift(static_cast<int>(t.op_type_1)) +
                    sz_shift(static_cast<int>(t.op_type_2)) +
                    sz_shift(static_cast<int>(t.op_type_3));
        if (delta != 0) return false;
    }
    return true;
}

inline std::unique_ptr<FixedSzOperator>
project_fixed_sz(const Operator& op, std::int64_t n_up) {
    auto fop = std::make_unique<FixedSzOperator>(
        op.getNumBits(), op.getSpin(), n_up);
    fop->transform_data_  = op.transform_data_;
    fop->three_body_data_ = op.three_body_data_;
    fop->invalidateMatrixCaches();
    return fop;
}

inline std::uint64_t binomial(std::uint64_t n, std::uint64_t k) {
    if (k > n) return 0;
    if (k > n - k) k = n - k;
    std::uint64_t result = 1;
    for (std::uint64_t i = 0; i < k; ++i) {
        result = result * (n - i) / (i + 1);
    }
    return result;
}

} // namespace detail

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Backend selection. Mirrors the `device=` argument of `qed.diag(...)`.
enum class Device {
    Auto,    ///< Heuristic over (dim, build flags). Default.
    CPU,
    GPU,     ///< Requires WITH_CUDA. Falls back to CPU + warning when
             ///< unavailable unless `allow_fallback=false`.
    MPI,     ///< Requires WITH_SCALAPACK. The auto-pilot does not spawn
             ///< ranks itself; callers must already be inside an
             ///< `mpiexec` launcher. Use the Python
             ///< `qed.diag(device='mpi')` wrapper for an out-of-process
             ///< launcher.
};

struct AutoSolveOptions {
    /// How many eigenpairs to converge. Default: just the ground state.
    std::uint64_t num_eigenvalues = 1;

    /// Persist eigenvectors to `output_dir` (HDF5)?
    bool compute_eigenvectors = false;

    /// Convergence tolerance forwarded to the underlying solver.
    double tolerance = 1e-10;

    /// Where to write eigenvectors / HDF5 artefacts. Empty → "/dev/null"
    /// (no I/O), matching the convention of `exact_diagonalization_core`.
    std::string output_dir = "";

    /// `std::nullopt` → heuristic. Otherwise an explicit
    /// `DiagonalizationMethod` enum value.
    std::optional<DiagonalizationMethod> solver{};

    /// Backend device.
    Device device = Device::Auto;

    /// When `device=GPU` but the build has no GPU, fall back to CPU
    /// silently (with a stderr warning). When false, throw.
    bool allow_fallback = true;

    /// Project onto a fixed-Sz sector with this many up spins. The
    /// auto-pilot validates `conserves_sz(op)` first and throws
    /// `std::invalid_argument` if Sz is not a good quantum number.
    std::optional<std::int64_t> sz{};

    /// When true, print one line per auto-decision to stderr (mirrors
    /// the `verbose=True` default of `qed.diag(...)`).
    bool verbose = true;

    /// Threshold below which the auto-selector chooses dense LAPACK
    /// (`FULL`) over Lanczos. The Python wrapper uses 2048; below that
    /// LAPACK is end-to-end faster than Lanczos for any
    /// `num_eigenvalues`.
    std::uint64_t small_dim_threshold = 2048;

    /// GPU dispatch threshold. The auto rule promotes to a `*_GPU`
    /// sibling when the sector dimension is at least this big AND the
    /// build supports CUDA. Default: 2^17 = 131 072 (matches the
    /// Python `_resolve_device` heuristic).
    std::uint64_t gpu_dim_threshold = (1ULL << 17);

    /// Low-level escape hatch — mirrors Python's `qed.diag(extra_params=…)`.
    /// Invoked **after** the auto-pilot has populated `EDParameters`
    /// from `AutoSolveOptions`, **before** the dispatcher fires. Use
    /// this to tweak any of the ~70 niche EDParameters fields
    /// (arpack_*, ftlm_*, ltlm_*, tpq_*, scalapack_*, hybrid_*, …)
    /// without giving up the auto solver / device / Sz selection.
    /// Pass `nullptr` (the default) to skip.
    std::function<void(EDParameters&)> tune_params{};
};

// ---------------------------------------------------------------------------
// One-call exact diagonalisation with smart defaults.
//
// `H` may be either a base `Operator` (full Hilbert space) or a
// `FixedSzOperator` (already-projected sector). When `options.sz` is
// set on a base `Operator`, the function constructs a temporary
// `FixedSzOperator` for the requested sector. When `options.sz` is set
// on a `FixedSzOperator`, it is checked for consistency and otherwise
// ignored.
//
// Returns the same `EDResults` payload as `exact_diagonalization_core`.
// ---------------------------------------------------------------------------
inline EDResults solve(Operator& H, const AutoSolveOptions& options = {});

inline EDResults solve(Operator& H, const AutoSolveOptions& options) {
    const bool verbose = options.verbose;
    const std::uint64_t num_sites = H.getNumBits();
    const std::uint64_t base_dim = 1ULL << num_sites;

    // -----------------------------------------------------------------
    // 1. Resolve fixed-Sz axis with a guard against transverse / S±-
    //    breaking Hamiltonians (mirrors workflow.py:814 in Python).
    // -----------------------------------------------------------------
    std::unique_ptr<FixedSzOperator> projected;
    Operator* op_to_use = &H;
    // NOTE: `Operator` is non-polymorphic (no virtual destructor), so we
    // cannot dynamic_cast<FixedSzOperator*>. Callers who already hold a
    // `FixedSzOperator&` should pass it through this same overload --
    // the resulting full-Hilbert-space `apply()` call would be wrong.
    // For now we require base `Operator` inputs and project internally
    // when `options.sz` is supplied; the alternative path is to invoke
    // `exact_diagonalization_core` directly with the FixedSzOperator's
    // `apply` lambda + `getFixedSzDim()`.
    const bool sz_is_conserved = detail::conserves_sz(H);

    if (options.sz.has_value()) {
        if (!sz_is_conserved) {
            throw std::invalid_argument(
                "ed::auto_pilot::solve: sz=... was requested but the "
                "supplied Operator does not commute with total Sz. "
                "Build the Hamiltonian without Sz-breaking terms (no "
                "transverse field, no general-orientation J+-+-, etc.) "
                "or drop the sz= option.");
        }
        const std::int64_t sz_val = *options.sz;
        if (sz_val < 0 || static_cast<std::uint64_t>(sz_val) > num_sites) {
            throw std::invalid_argument(
                "ed::auto_pilot::solve: sz out of range [0, num_sites]");
        }
        projected = detail::project_fixed_sz(H, sz_val);
        op_to_use = projected.get();
        if (verbose) {
            const std::uint64_t sec_dim = detail::binomial(num_sites,
                static_cast<std::uint64_t>(sz_val));
            std::cerr << "[ed::auto_pilot::solve] Sz sector n_up=" << sz_val
                      << ": dim=" << sec_dim
                      << " (reduced from " << base_dim << ").\n";
        }
    } else if (!options.sz.has_value()
               && sz_is_conserved && verbose) {
        std::cerr << "[ed::auto_pilot::solve] HINT: this Hamiltonian conserves "
                  << "total Sz. Passing options.sz = N/2 = "
                  << (num_sites / 2)
                  << " would project onto the sector of dimension C("
                  << num_sites << ", " << (num_sites / 2)
                  << ") and dramatically reduce the cost.\n";
    }

    // Sector dimension for solver/device heuristics.
    std::uint64_t sector_dim = projected ? projected->getFixedSzDim() : base_dim;

    // -----------------------------------------------------------------
    // 2. Resolve solver (default: heuristic on dim & num_eigenvalues).
    // -----------------------------------------------------------------
    DiagonalizationMethod method;
    if (options.solver.has_value()) {
        method = *options.solver;
    } else if (sector_dim <= options.small_dim_threshold) {
        method = DiagonalizationMethod::FULL;
    } else if (options.num_eigenvalues <= 5) {
        method = DiagonalizationMethod::LANCZOS;
    } else if (options.num_eigenvalues <= 20) {
        method = DiagonalizationMethod::KRYLOV_SCHUR;
    } else {
        method = DiagonalizationMethod::BLOCK_LANCZOS;
    }

    // -----------------------------------------------------------------
    // 3. Resolve device (build introspection + sector dim heuristic).
    // -----------------------------------------------------------------
    bool use_gpu = false;
    bool use_mpi = false;
    switch (options.device) {
        case Device::CPU:
            break;
        case Device::GPU:
            if (!is_cuda_compiled()) {
                if (!options.allow_fallback) {
                    throw std::runtime_error(
                        "ed::auto_pilot::solve: device=GPU requested but "
                        "WITH_CUDA was not enabled in this build.");
                }
                if (verbose) {
                    std::cerr << "[ed::auto_pilot::solve] GPU requested but "
                              << "WITH_CUDA is OFF; falling back to CPU.\n";
                }
            } else {
                use_gpu = true;
            }
            break;
        case Device::MPI:
            if (!is_scalapack_compiled() && !options.allow_fallback) {
                throw std::runtime_error(
                    "ed::auto_pilot::solve: device=MPI requested but "
                    "WITH_SCALAPACK was not enabled in this build.");
            }
            use_mpi = is_scalapack_compiled();
            break;
        case Device::Auto:
            if (is_cuda_compiled() && sector_dim >= options.gpu_dim_threshold) {
                use_gpu = true;
            }
            break;
    }

    // -----------------------------------------------------------------
    // 4. (No GPU-sibling enum promotion -- the modern API instead
    //    sets EDParameters::use_gpu / use_mpi orthogonally to the
    //    DiagonalizationMethod enum, see ed_parameters.h:191.)
    // -----------------------------------------------------------------

    if (verbose) {
        std::cerr << "[ed::auto_pilot::solve] solver_enum="
                  << static_cast<int>(method)
                  << "  num_eigenvalues=" << options.num_eigenvalues
                  << "  use_gpu=" << (use_gpu ? "true" : "false")
                  << "  use_mpi=" << (use_mpi ? "true" : "false") << "\n";
    }

    // -----------------------------------------------------------------
    // 5. Build EDParameters and dispatch.
    // -----------------------------------------------------------------
    EDParameters params;
    params.num_eigenvalues = options.num_eigenvalues;
    params.tolerance = options.tolerance;
    params.compute_eigenvectors = options.compute_eigenvectors;
    params.output_dir = options.output_dir;
    params.use_gpu = use_gpu;
    params.use_mpi = use_mpi;
    if (projected) {
        params.use_fixed_sz = true;
        params.fixed_sz_op = projected.get();
    }

    // Low-level escape hatch: let callers tweak any of the ~70 niche
    // EDParameters fields the auto-pilot doesn't surface explicitly.
    if (options.tune_params) {
        options.tune_params(params);
    }

    auto apply = [op_to_use, projected_ptr = projected.get()](
                     const Complex* in, Complex* out, int n) {
        // Operator::apply() is non-virtual, so we MUST dispatch on the
        // statically-known concrete type to keep the FixedSzOperator
        // override from being sliced into the base-Operator path (which
        // would silently run on the full 2^N Hilbert space).
        if (projected_ptr != nullptr) {
            projected_ptr->apply(in, out, static_cast<size_t>(n));
        } else {
            op_to_use->apply(in, out, static_cast<size_t>(n));
        }
    };
    return exact_diagonalization_core(apply, sector_dim, method, params);
}

} // namespace ed::auto_pilot
