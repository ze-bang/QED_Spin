#pragma once
// =============================================================================
// include/ed/core/make_operator.h
//
// `ed::make_operator(OperatorSpec)`: the unified factory that supersedes
// the legacy collection of construction-plus-solve helpers:
//
//   * (deleted) `ed::exact_diagonalization_from_files`            (ed_wrapper.h)
//   * (deleted) `ed::exact_diagonalization_from_directory`        (ed_wrapper.h)
//   * (deleted) `ed::exact_diagonalization_streaming_symmetry*`   (ed_wrapper_streaming.h)
//   * `DistributedOperator(...)` / `DistributedSymmetryOperator(...)`
//     manual construction in CLI binaries
//
// The factory returns ONLY the constructed `LinearOperator` (or a derived
// concrete type --- distributed / streaming / GPU as the spec requests);
// the actual solve is then a separate `ed::workflows::solve(*op, opts)` call.
//
// Spec shape:
//
//   OperatorSpec spec;
//   spec.source                  = ed::DirectoryPath{"/path/to/dir"};
//   spec.num_sites               = 16;
//   spec.spin_l                  = 0.5f;
//   spec.fixed_sz                = std::nullopt;          // fixed-Sz off
//   spec.streaming_symmetry      = false;
//   spec.distributed             = false;
//
//   auto op = ed::make_operator(spec);
//   auto gs = ed::workflows::solve(*op, ed::SolveOptions{ .num_eigs = 5 });
//
// Phase 4.3 of the Minimalist ED Collapse (May 2026), extended for the full
// unified-interface collapse to honour every axis the legacy CLI consumes
// (streaming_symmetry, distributed, fixed_sz, plus their cross products).
//
// Return-type policy: `std::unique_ptr<LinearOperator>` is the lowest common
// type that covers the full matrix of subclasses the spec axes can
// materialise:
//
//   * plain `Operator`                       (default lane)
//   * `FixedSzOperator`                      (`fixed_sz` set)
//   * `StreamingSymmetryOperator`            (`streaming_symmetry = true`)
//   * `FixedSzStreamingSymmetryOperator`     (streaming_symmetry + fixed_sz)
//   * `ed::distributed::DistributedOperator` (distributed = true)
//   * `ed::distributed::DistributedSymmetryOperator`
//                                            (distributed + streaming_symmetry
//                                             + sector_index)
//
// Both `Operator` and the distributed operators derive from
// `ed::LinearOperator`, so a single owning pointer covers every case
// without losing dispatchability.
// =============================================================================

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/linear_operator.h>
#include <ed/core/operator.h>
#include <ed/core/streaming_symmetry.h>

#ifdef WITH_MPI
#  include <mpi.h>
#  include <ed/distributed/distributed_operator.h>
#  include <ed/distributed/distributed_symmetry_operator.h>
#endif

namespace ed {

// ---------------------------------------------------------------------------
// Source discriminators
// ---------------------------------------------------------------------------

/// Explicit per-file paths. Each empty path is skipped.
struct FilePaths {
    std::string interaction_file;
    std::string single_site_file;
    std::string counterterm_file;
    std::string three_body_file;
};

/// Directory route: load InterAll.dat / Trans.dat / CounterTerm.dat /
/// ThreeBodyG.dat from a fixed directory. Sub-paths are configurable so
/// callers can rename if needed.
struct DirectoryPath {
    std::string directory;
    std::string interaction_filename = "InterAll.dat";
    std::string single_site_filename = "Trans.dat";
    std::string counterterm_filename = "CounterTerm.dat";
    std::string three_body_filename  = "ThreeBodyG.dat";
};

/// In-memory route: caller passes a pre-built `Operator` (typically built
/// programmatically via `Operator::addInteraction(...)` etc.). The factory
/// just forwards the supplied pointer.
struct InMemoryOperator {
    std::unique_ptr<Operator> op;
};

// ---------------------------------------------------------------------------
// OperatorSpec
// ---------------------------------------------------------------------------

struct OperatorSpec {
    /// Hamiltonian source. Exactly one variant alternative is consulted.
    std::variant<FilePaths, DirectoryPath, InMemoryOperator> source;

    /// Total number of sites (= number of qubits in the spin-1/2 mapping).
    std::uint64_t         num_sites = 0;

    /// Spin magnitude (0.5 for spin-1/2). Affects how single-site operators
    /// are scaled when loaded from text files.
    float                 spin_l    = 0.5f;

    /// If set, restrict to a single Sz sector. The integer is the n_up
    /// (number of "up" spins). For Heisenberg-like models without a
    /// Zeeman field, the ground state typically lives in
    /// `n_up = num_sites / 2`.
    std::optional<int>    fixed_sz;

    /// If true, materialise a `StreamingSymmetryOperator`. Requires the
    /// source to be a `DirectoryPath` containing an `automorphism_results/`
    /// subdirectory. The returned operator carries all symmetry sectors;
    /// callers drive one sector at a time via the streaming op's
    /// `SectorView` nested class (consumed transparently by the
    /// orchestrator when iterating sectors).
    bool                  streaming_symmetry = false;

    /// If true, materialise a `DistributedOperator` (or
    /// `DistributedSymmetryOperator` when `streaming_symmetry` is also
    /// true). Requires `WITH_MPI` at compile time and `MPI_Init` at
    /// runtime. The MPI communicator defaults to `MPI_COMM_WORLD`.
    bool                  distributed = false;

    /// For the distributed + streaming-symmetry lane: which symmetry
    /// sector to materialise. Required in that lane; ignored otherwise.
    /// (The non-distributed streaming-symmetry operator carries every
    /// sector internally and the caller iterates them as
    /// `SectorView`s.)
    std::optional<std::size_t> sector_index;

#ifdef WITH_MPI
    /// MPI communicator for the distributed lanes. Defaults to
    /// `MPI_COMM_WORLD`. Single-rank communicators are accepted (the
    /// distributed operator degenerates to a single-slab apply).
    MPI_Comm              comm = MPI_COMM_WORLD;
#endif
};

// ---------------------------------------------------------------------------
// File-loading helper
// ---------------------------------------------------------------------------

/// Populate a freshly-constructed `Operator` from the given file paths.
/// Skips files whose path is empty or that don't exist on disk. The
/// `Operator::loadFrom*` methods are the well-tested parser frontends;
/// this helper just wires them up.
inline void populate_operator_from_files(Operator& op, const FilePaths& f) {
    namespace fs = std::filesystem;
    if (!f.interaction_file.empty() && fs::exists(f.interaction_file)) {
        op.loadFromInterAllFile(f.interaction_file);
    }
    if (!f.single_site_file.empty() && fs::exists(f.single_site_file)) {
        op.loadFromFile(f.single_site_file);
    }
    // CounterTerm.dat and ThreeBodyG.dat use the same `loadFromFile`
    // header format (op_type, site, complex coeff). Presence-check above
    // covers the common case where these auxiliary decks are absent.
    if (!f.counterterm_file.empty() && fs::exists(f.counterterm_file)) {
        op.loadFromFile(f.counterterm_file);
    }
    if (!f.three_body_file.empty() && fs::exists(f.three_body_file)) {
        op.loadFromFile(f.three_body_file);
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (unnamespaced detail::)
// ---------------------------------------------------------------------------

namespace detail {

/// Resolve a `DirectoryPath` variant into the concrete file paths.
inline FilePaths file_paths_from_directory(const DirectoryPath& d) {
    FilePaths f;
    const std::filesystem::path dir{d.directory};
    f.interaction_file = (dir / d.interaction_filename).string();
    f.single_site_file = (dir / d.single_site_filename).string();
    f.counterterm_file = (dir / d.counterterm_filename).string();
    f.three_body_file  = (dir / d.three_body_filename).string();
    return f;
}

/// Build a base `Operator` or `FixedSzOperator` from the spec's
/// num_sites / spin_l / fixed_sz fields. Used by every downstream
/// lane (full-Hilbert, fixed-Sz, streaming, distributed) as the
/// term-storage carrier.
inline std::shared_ptr<Operator> build_base_op(const OperatorSpec& spec) {
    if (spec.fixed_sz.has_value()) {
        return std::make_shared<FixedSzOperator>(
            static_cast<uint64_t>(spec.num_sites), spec.spin_l,
            static_cast<int64_t>(*spec.fixed_sz));
    }
    return std::make_shared<Operator>(
        static_cast<uint64_t>(spec.num_sites), spec.spin_l);
}

/// Apply the spec's source variant to an existing `Operator&`, loading
/// term storage from files / directory / in-memory as appropriate.
/// Used after `build_base_op` to populate term storage.
inline void load_terms_into(Operator& op, const OperatorSpec& spec) {
    std::visit([&](auto&& src) {
        using T = std::decay_t<decltype(src)>;
        if constexpr (std::is_same_v<T, FilePaths>) {
            populate_operator_from_files(op, src);
        } else if constexpr (std::is_same_v<T, DirectoryPath>) {
            populate_operator_from_files(op,
                file_paths_from_directory(src));
        } else if constexpr (std::is_same_v<T, InMemoryOperator>) {
            // The in-memory route is incompatible with the file-loading
            // helper -- the caller pre-built the term storage and we
            // should never reach this branch via build_base_op +
            // load_terms_into. The streaming / distributed lanes route
            // around this case explicitly.
            (void)op;
            (void)src;
        }
    }, spec.source);
}

/// Resolve the directory string from a `DirectoryPath`-typed source,
/// or throw if the spec is using a different source variant. Used by
/// the streaming-symmetry lane which mandatorily needs a directory
/// (it loads `automorphism_results/` from disk).
inline const std::string& require_directory(const OperatorSpec& spec) {
    return std::visit([&](auto&& src) -> const std::string& {
        using T = std::decay_t<decltype(src)>;
        if constexpr (std::is_same_v<T, DirectoryPath>) {
            return src.directory;
        } else {
            throw std::runtime_error(
                "ed::make_operator: streaming_symmetry = true requires "
                "OperatorSpec::source to be a DirectoryPath (the "
                "streaming-symmetry operator loads its automorphism "
                "metadata from <directory>/automorphism_results/).");
        }
    }, spec.source);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Streaming-symmetry lane (CPU, multi-sector)
// ---------------------------------------------------------------------------

/// Build a `StreamingSymmetryOperator` (or its fixed-Sz sibling). The
/// returned operator carries every symmetry sector loaded from
/// `<directory>/automorphism_results/`. Callers drive one sector at a
/// time via the operator's `SectorView` nested class.
inline std::unique_ptr<LinearOperator>
make_streaming_symmetry_operator(const OperatorSpec& spec) {
    const std::string& dir = detail::require_directory(spec);

    if (spec.fixed_sz.has_value()) {
        auto op = std::make_unique<FixedSzStreamingSymmetryOperator>(
            static_cast<uint64_t>(spec.num_sites), spec.spin_l,
            static_cast<int64_t>(*spec.fixed_sz));
        detail::load_terms_into(*op, spec);
        op->generateSymmetrySectorsStreamingFixedSz(dir);
        return op;
    }

    auto op = std::make_unique<StreamingSymmetryOperator>(
        static_cast<uint64_t>(spec.num_sites), spec.spin_l);
    detail::load_terms_into(*op, spec);
    op->generateSymmetrySectorsStreaming(dir);
    return op;
}

#ifdef WITH_MPI
// ---------------------------------------------------------------------------
// Distributed lanes (WITH_MPI only)
// ---------------------------------------------------------------------------

/// Build a `DistributedOperator` wrapping a serial `Operator` slab. The
/// communicator defaults to `MPI_COMM_WORLD`; pass `spec.comm` to
/// override.
inline std::unique_ptr<LinearOperator>
make_distributed_operator(const OperatorSpec& spec) {
    auto base = detail::build_base_op(spec);
    detail::load_terms_into(*base, spec);
    return std::make_unique<ed::distributed::DistributedOperator>(
        std::move(base), spec.comm);
}

/// Build a `DistributedSymmetryOperator` for a single symmetry sector.
/// Requires `spec.sector_index`; the symmetry metadata is loaded from
/// the directory's `automorphism_results/` before the distributed wrap.
inline std::unique_ptr<LinearOperator>
make_distributed_symmetry_operator(const OperatorSpec& spec) {
    if (!spec.sector_index.has_value()) {
        throw std::runtime_error(
            "ed::make_operator: distributed + streaming_symmetry "
            "requires OperatorSpec::sector_index (the "
            "DistributedSymmetryOperator carries one sector at a "
            "time; iterate sector indices in the caller for the full "
            "spectrum).");
    }
    const std::string& dir = detail::require_directory(spec);

    auto base = detail::build_base_op(spec);
    detail::load_terms_into(*base, spec);
    // The DistributedSymmetryOperator constructor reads
    // `base->symmetry_info`, which is populated by `loadFromDirectory`.
    base->symmetry_info.loadFromDirectory(dir);

    return std::make_unique<ed::distributed::DistributedSymmetryOperator>(
        std::move(base), *spec.sector_index, spec.comm);
}
#endif  // WITH_MPI

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Construct a `LinearOperator` (or derived) from `spec`. Returns a
/// polymorphic owner; caller passes `*op` to `ed::workflows::solve` /
/// `ed::workflows::thermal` / `ed::workflows::spectral`.
///
/// Dispatch order (axis values):
///
///   distributed | streaming_symmetry | fixed_sz | returned type
///   ------------|--------------------|----------|--------------------------
///   false       | false              | nullopt  | Operator
///   false       | false              | set      | FixedSzOperator
///   false       | true               | nullopt  | StreamingSymmetryOperator
///   false       | true               | set      | FixedSzStreamingSymmetryOperator
///   true        | false              | any      | DistributedOperator
///   true        | true               | any      | DistributedSymmetryOperator
///
/// Compile guards:
///   * `distributed = true` requires `WITH_MPI`. Without WITH_MPI, the
///     factory throws `std::runtime_error` to keep the surface
///     uniformly defined regardless of build flags.
inline std::unique_ptr<LinearOperator> make_operator(OperatorSpec spec) {
    // The InMemoryOperator route is incompatible with the
    // streaming_symmetry / distributed axes (those require directory
    // loading or term-storage manipulation that the caller has not
    // performed). Forward the bare unique_ptr in the simple case.
    if (auto* mem = std::get_if<InMemoryOperator>(&spec.source)) {
        if (spec.streaming_symmetry || spec.distributed) {
            throw std::runtime_error(
                "ed::make_operator: InMemoryOperator source is "
                "incompatible with streaming_symmetry / distributed "
                "axes; use FilePaths or DirectoryPath instead.");
        }
        return std::unique_ptr<LinearOperator>(mem->op.release());
    }

    if (spec.distributed) {
#ifdef WITH_MPI
        if (spec.streaming_symmetry) {
            return make_distributed_symmetry_operator(spec);
        }
        return make_distributed_operator(spec);
#else
        throw std::runtime_error(
            "ed::make_operator: distributed = true requires WITH_MPI "
            "at compile time.");
#endif
    }

    if (spec.streaming_symmetry) {
        return make_streaming_symmetry_operator(spec);
    }

    // Default lane: plain Operator or FixedSzOperator, file-loaded.
    // Resolve the source into a concrete FilePaths first; the InMemory
    // branch is handled at the top of the function.
    const FilePaths fp = std::visit([&](auto&& src) -> FilePaths {
        using T = std::decay_t<decltype(src)>;
        if constexpr (std::is_same_v<T, FilePaths>) return src;
        else if constexpr (std::is_same_v<T, DirectoryPath>)
            return detail::file_paths_from_directory(src);
        else return {};  // unreachable; InMemoryOperator handled above.
    }, spec.source);

    if (spec.fixed_sz.has_value()) {
        auto fop = std::make_unique<FixedSzOperator>(
            static_cast<uint64_t>(spec.num_sites), spec.spin_l,
            static_cast<int64_t>(*spec.fixed_sz));
        populate_operator_from_files(*fop, fp);
        return fop;
    }

    auto op = std::make_unique<Operator>(
        static_cast<uint64_t>(spec.num_sites), spec.spin_l);
    populate_operator_from_files(*op, fp);
    return op;
}

}  // namespace ed
