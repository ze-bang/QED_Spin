#pragma once
// =============================================================================
// include/ed/core/make_operator.h
//
// `ed::make_operator(OperatorSpec)`: the unified factory that supersedes
// the legacy collection of construction-plus-solve helpers:
//
//   * `exact_diagonalization_from_files`     (ed_wrapper.h, ~450 lines)
//   * `exact_diagonalization_from_directory` (ed_wrapper.h, ~280 lines)
//   * `exact_diagonalization_streaming_symmetry*` (ed_wrapper_streaming.h)
//
// The new factory returns ONLY the constructed `Operator` (or a derived
// concrete type --- distributed / streaming / GPU as the spec requests);
// the actual solve is then a separate `ed::solve(*op, opts)` call
// against the Phase 4.2 orchestrator. This is the central
// construction/solve split the plan calls out.
//
// Spec shape mirrors the plan:
//
//   OperatorSpec spec;
//   spec.source                  = "/path/to/dir";         // directory route
//   spec.num_sites               = 16;
//   spec.spin_l                  = 0.5f;
//   spec.fixed_sz                = std::nullopt;            // fixed-Sz off
//   spec.streaming_symmetry      = false;
//   spec.distributed             = false;
//
//   auto op = ed::make_operator(spec);
//   auto gs = ed::solve(*op, ed::SolveOptions{ .num_eigs = 5 });
//
// Phase 4.3 of the Minimalist ED Collapse (May 2026).
// =============================================================================

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/operator.h>

namespace ed {

/// Discriminator: explicit per-file paths, vs a single directory.
struct FilePaths {
    std::string interaction_file;
    std::string single_site_file;
    std::string counterterm_file;
    std::string three_body_file;
};

struct DirectoryPath {
    std::string directory;
    std::string interaction_filename = "InterAll.dat";
    std::string single_site_filename = "Trans.dat";
    std::string counterterm_filename = "CounterTerm.dat";
    std::string three_body_filename  = "ThreeBodyG.dat";
};

/// In-memory builder route --- caller-driven `Operator` construction.
/// The factory just forwards the supplied unique_ptr. Useful for tests
/// and for programmatically-built Hamiltonians.
struct InMemoryOperator {
    std::unique_ptr<Operator> op;
};

struct OperatorSpec {
    /// Where to load the Hamiltonian from. Exactly one of the three
    /// alternatives is consulted; the others are ignored.
    std::variant<FilePaths, DirectoryPath, InMemoryOperator> source;

    std::uint64_t         num_sites = 0;
    float                 spin_l    = 0.5f;
    /// If set, restrict to a single Sz sector (legacy `fixed_sz_*` lane).
    std::optional<int>    fixed_sz;
    /// If true, materialise a `StreamingSymmetryOperator`. Otherwise a
    /// plain `Operator` is returned.
    bool                  streaming_symmetry = false;
    /// If true, materialise a `DistributedOperator` (currently
    /// CPU-only; GPU+MPI requires a separate factory entry).
    bool                  distributed = false;
};

/// Construct an `Operator` (or derived) from `spec`. Returns a
/// polymorphic owner; caller passes `*op` to `ed::solve` / `ed::thermal`
/// / `ed::spectral`.
///
/// Implementation routes to the existing `Operator::loadFrom*`
/// methods, plus the optional symmetry / distributed wrappers. The
/// factory is intentionally a thin wrapper --- it owns the source-type
/// discrimination but delegates parsing to the well-tested
/// `Operator::loadFrom*` paths.
/// Implementation detail: populate a freshly-constructed `Operator`
/// from the given file paths. Skips files whose path is empty or that
/// don't exist on disk. The `Operator::loadFrom*` methods are the
/// well-tested parser frontends; this helper just wires them up.
inline void populate_operator_from_files(Operator& op, const FilePaths& f) {
    namespace fs = std::filesystem;
    if (!f.interaction_file.empty() && fs::exists(f.interaction_file)) {
        op.loadFromInterAllFile(f.interaction_file);
    }
    if (!f.single_site_file.empty() && fs::exists(f.single_site_file)) {
        op.loadFromFile(f.single_site_file);
    }
    // CounterTerm.dat and ThreeBodyG.dat use the same `loadFromFile`
    // header format. The presence-check above also covers the common
    // case where these auxiliary decks are absent.
    if (!f.counterterm_file.empty() && fs::exists(f.counterterm_file)) {
        op.loadFromFile(f.counterterm_file);
    }
    if (!f.three_body_file.empty() && fs::exists(f.three_body_file)) {
        op.loadFromFile(f.three_body_file);
    }
}

/// Construct an `Operator` (or `FixedSzOperator` when `spec.fixed_sz`
/// is set) from `spec`. Returns a polymorphic owner; caller passes
/// `*op` to `ed::workflows::solve` / `ed::workflows::thermal` /
/// `ed::workflows::spectral`.
///
/// Implementation routes to the existing `Operator::loadFrom*`
/// methods; the factory is intentionally a thin wrapper.
///
/// ED Cleanup Sweep Phase 4 (May 2026): extended to cover the full
/// CLI file matrix (InterAll + Trans + CounterTerm + ThreeBodyG) and
/// the fixed-Sz projection axis, so `src/cli/workflows.cpp` can route
/// through this factory + `ed::workflows::solve` instead of the
/// legacy `ed::exact_diagonalization(directory, ...)` dispatcher.
/// Streaming-symmetry / distributed routes remain on their legacy
/// entry points until the corresponding orchestrator paths land
/// (Phase 6).
inline std::unique_ptr<Operator> make_operator(OperatorSpec spec) {
    auto load_paths = [&](const FilePaths& f) -> std::unique_ptr<Operator> {
        if (spec.fixed_sz.has_value()) {
            auto fop = std::make_unique<FixedSzOperator>(
                static_cast<uint64_t>(spec.num_sites), spec.spin_l,
                static_cast<int64_t>(*spec.fixed_sz));
            populate_operator_from_files(*fop, f);
            return fop;
        }
        auto op = std::make_unique<Operator>(
            static_cast<uint64_t>(spec.num_sites), spec.spin_l);
        populate_operator_from_files(*op, f);
        return op;
    };

    return std::visit([&](auto&& src) -> std::unique_ptr<Operator> {
        using T = std::decay_t<decltype(src)>;
        if constexpr (std::is_same_v<T, FilePaths>) {
            return load_paths(src);
        } else if constexpr (std::is_same_v<T, DirectoryPath>) {
            FilePaths f;
            const std::filesystem::path d{src.directory};
            f.interaction_file = (d / src.interaction_filename).string();
            f.single_site_file = (d / src.single_site_filename).string();
            f.counterterm_file = (d / src.counterterm_filename).string();
            f.three_body_file  = (d / src.three_body_filename).string();
            return load_paths(f);
        } else {  // InMemoryOperator
            return std::move(const_cast<T&>(src).op);
        }
    }, spec.source);
}

}  // namespace ed
