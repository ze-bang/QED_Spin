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

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <ed/core/fixed_sz_operator.h>
#include <ed/core/linear_operator.h>
#include <ed/core/operator.h>
#include <ed/core/results.h>            // SectorTag
#include <ed/symmetry/sector_operator.h>
#include <ed/symmetry/sector_set.h>

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

    /// Stage 8c (SymmetryEngine v2): split the half-filling Sz block into
    /// flip-parity (k, +/-) sectors when the fixed-Sz streaming lane builds
    /// n_up == num_sites/2 (caller must have verified [H, X] == 0 and an
    /// eigenvalues-only workload). Forces the lazy builder variant.
    bool                  flip_project_half = false;

    /// Full-space prod-sigma^x sectors (monomial-group consolidation,
    /// Jul 2026): when no fixed-Sz axis applies but [H, X] == 0, split
    /// every spatial irrep into (k, +/-) flip sectors over the full
    /// 2^N space (caller verifies the symmetry + eigenvalues-only
    /// workload). Forces the lazy full-space builder.
    bool                  flip_sectors_full = false;

    /// Sz-PARITY sectors (diagonal Z2 remnant when U(1) is broken but
    /// every term changes n_up by an even amount): 0 = even half,
    /// 1 = odd half, 2 = BOTH halves in one sector set (pooled GS /
    /// thermal / full-dense mode). Unset = off. Mutually exclusive
    /// with fixed_sz. ``flip_sectors_full`` additionally splits each
    /// (parity, k) into flip signs (even N only).
    std::optional<int>    sz_parity;

    /// Stage 3 (SymmetryEngine v2): explicit OrbitTable disk-cache
    /// directory. Empty = auto (``ED_SYM_CACHE_DIR`` override, else
    /// ``<lattice_dir>/basis_cache`` for directory sources; registry-only
    /// for in-memory sources). ``ED_SYM_CACHE=0`` disables the disk layer
    /// entirely. See ed::symmetry::resolve_sym_cache_dir.
    std::string           basis_cache_dir;

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

// ---------------------------------------------------------------------------
// Eager-vs-lazy regime decision for the fixed-Sz sector set (operator-collapse
// Phase 3, Jun 2026). Mirrors the budget logic baked into
// ``FixedSzStreamingSymmetryOperator::generateSymmetrySectorsStreamingFixedSz``
// so ``make_sector_operators`` picks the SAME regime as the legacy streaming
// loop for any (N, n_up, |G|) -- the invariant the N>=28 parity check rests
// on. Estimates the all-sector resident orbit-CSR footprint as
// ``C(n_bits, n_up) * num_sectors * 40 B`` (the streaming op uses
// ``num_orbits * |G| * num_sectors * 40`` and ``num_orbits * |G| ~ dim``, so
// the two estimates agree to within the orbit-fragmentation factor). The
// fixed-Sz dimension is computed via the binomial directly -- we must NOT
// build the FixedSzSubspace just to size it (that is the ~5 GB allocation the
// lazy path exists to defer). Env overrides (identical names/semantics to the
// streaming op): ``ED_SYM_LAZY_SECTORS`` (1 force lazy / 0 force eager),
// ``ED_SYM_LAZY_SECTORS_BYTES_MAX`` (budget in bytes, default 4 GiB).
// ---------------------------------------------------------------------------
inline bool fixed_sz_sectors_should_be_lazy(std::uint64_t           n_bits,
                                            std::int64_t            n_up,
                                            const SymmetryGroupInfo& info) {
    if (const char* e = std::getenv("ED_SYM_LAZY_SECTORS")) {
        if (e[0] == '1') return true;
        if (e[0] == '0') return false;
    }
    // Default budget = 64 MiB (was 4 GiB). The "eager" orbit-CSR lane is NOT
    // actually faster for non-trivial sectors: its reverse lookup falls back to
    // an O(log dim) SortedUint64Index binary search (the O(1) dense lookup is
    // never built on this path), making its symmetry SpMV ~14x slower than the
    // matrix-free rep walk -- which is ALSO O(#reps) memory instead of O(dim).
    // So we only stay eager for genuinely tiny sectors (where construction is
    // instant); everything larger uses the rep walk. Measured (XXZ ring, |G|=N):
    // N=21 eager 44s -> rep walk 3.2s; N=24 eager >90s -> rep walk 7.5s.
    std::size_t budget = 64ULL * 1024ULL * 1024ULL;  // 64 MiB
    if (const char* e = std::getenv("ED_SYM_LAZY_SECTORS_BYTES_MAX")) {
        try { budget = std::stoull(e); } catch (...) {}
    }
    // C(n_bits, n_up) in long double (multiplicative form, no overflow).
    const int n = static_cast<int>(n_bits);
    int k = static_cast<int>(n_up);
    long double dim = 0.0L;
    if (k >= 0 && k <= n) {
        k = std::min(k, n - k);
        dim = 1.0L;
        for (int i = 0; i < k; ++i) {
            dim = dim * static_cast<long double>(n - i)
                / static_cast<long double>(i + 1);
        }
    }
    const long double num_sectors =
        static_cast<long double>(std::max<std::size_t>(1, info.sectors.size()));
    const long double est_bytes = dim * num_sectors * 40.0L;
    return est_bytes > static_cast<long double>(budget);
}

// Full-space (no Sz) twin of ``fixed_sz_sectors_should_be_lazy``: the eager full
// builder materializes an orbit CSR over the 2^N space, so route to the
// CSR-free rep-walk lane once that would exceed the budget. Same env knobs.
inline bool full_sectors_should_be_lazy(std::uint64_t            n_bits,
                                        const SymmetryGroupInfo& info) {
    if (const char* e = std::getenv("ED_SYM_LAZY_SECTORS")) {
        if (e[0] == '1') return true;
        if (e[0] == '0') return false;
    }
    std::size_t budget = 64ULL * 1024ULL * 1024ULL;  // 64 MiB
    if (const char* e = std::getenv("ED_SYM_LAZY_SECTORS_BYTES_MAX")) {
        try { budget = std::stoull(e); } catch (...) {}
    }
    long double dim = 1.0L;                            // 2^n_bits
    for (std::uint64_t i = 0; i < n_bits; ++i) dim *= 2.0L;
    const long double num_sectors =
        static_cast<long double>(std::max<std::size_t>(1, info.sectors.size()));
    const long double est_bytes = dim * num_sectors * 40.0L;
    return est_bytes > static_cast<long double>(budget);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Direct sector-set lane (operator-collapse, Jun 2026)
// ---------------------------------------------------------------------------

/// Build the symmetry sectors as a flat vector of standalone, owning
/// ``ed::symmetry::SectorOperator`` objects -- the collapse-target twin of
/// ``make_streaming_symmetry_operator``. Where the streaming lane returns a
/// single monolithic ``StreamingSymmetryOperator`` whose nested
/// ``SectorView``s back-reference the parent, this lane routes the spec
/// straight through the P5 enumerator
/// (``ed::symmetry::build_{full,fixed_sz}_sector_operators``): orbit reps and
/// per-sector ``SectorBasis`` objects are built directly from the loaded
/// ``SymmetryGroupInfo``, with no ``StreamingSymmetryOperator`` materialised
/// at all. Each returned operator owns its sector and is driven by the
/// unified ``CpuMatVecBackend<SymmetryBasisPolicy>``.
///
/// This is additive: ``make_operator`` (which must return a single
/// ``LinearOperator``) is unchanged; callers that want the collapse-target
/// multi-sector list opt in by calling this function. The production sector
/// loop continues to use ``StreamingSymmetryHandle`` (which unconditionally
/// adopts each legacy sector into a ``SectorOperator``); this entry point is
/// the orbit-enumeration-free alternative that skips the legacy operator
/// entirely.
///
/// Requirements: ``spec.streaming_symmetry == true`` and a ``DirectoryPath``
/// source carrying ``<directory>/automorphism_results/``. ``spec.fixed_sz``
/// selects the fixed-Sz lane (orbits restricted to the ``n_up`` subspace);
/// absent, the full-Hilbert lane is used.
// ---------------------------------------------------------------------------
// Tagged sector set: the sector operators paired with their ``SectorTag``s
// (raw irrep index + dim + quantum numbers + n_up). This is the
// streaming-handle-free replacement for the production sector loop -- the
// caller iterates ``set.operators`` (compacted: empty irreps dropped) and
// keys per-sector HDF5 / cross-irrep selection off ``set.tags`` exactly as it
// used ``StreamingSymmetryHandle::sector_tag(k)``. Crucially the dim is read
// from ``op->dim()`` (Pass 1.5 in the lazy regime), so building the tags never
// materialises a per-sector orbit CSR.
// ---------------------------------------------------------------------------
struct SectorOperatorSet {
    std::vector<std::unique_ptr<ed::symmetry::SectorOperator>> operators;
    std::vector<ed::SectorTag>                                 tags;
    /// Total RAW sector count (== ``symmetry_info.sectors.size()``, i.e.
    /// including irreps that fully cancel). ``operators`` / ``tags`` are
    /// COMPACTED (empty irreps dropped), so this is the upper bound on the
    /// raw irrep index ``tags[i].sector_index`` and the value the legacy
    /// ``filter_sectors(num_sectors, ...)`` semantics expect.
    std::size_t                                                num_raw_sectors = 0;
    /// Quantum-number labels for EVERY raw sector (length
    /// ``num_raw_sectors``), including the empty irreps dropped from
    /// ``operators`` / ``tags``. The cross-irrep selection-rule walker
    /// (``infer_generator_orders`` / ``resolve_target_sector``) needs the
    /// full irrep lattice -- e.g. for ``n_up == 0`` only ``k == 0`` survives,
    /// so inferring generator orders from the surviving tags alone would
    /// collapse a Z_N translation order to 1. Keyed by raw irrep index.
    std::vector<std::vector<int>>                              all_quantum_numbers;
};

namespace detail {

// |Fix(g)| -- number of basis states left invariant by site-permutation ``perm``
// (one element of ``max_clique``), restricted to popcount ``n_up`` (``n_up < 0``
// => full 2^N space). A state fixed by g is constant on each cycle of g, so it is
// chosen by one bit per cycle; restricting to popcount n_up counts the subsets of
// cycles whose lengths sum to n_up (full space: 2^#cycles).
inline std::uint64_t num_fixed_states(const std::vector<int>& perm, int n_up) {
    const int N = static_cast<int>(perm.size());
    std::vector<char> seen(static_cast<std::size_t>(N), 0);
    std::vector<int>  cyc_len;
    for (int i = 0; i < N; ++i) {
        if (seen[static_cast<std::size_t>(i)]) continue;
        int len = 0, j = i;
        while (!seen[static_cast<std::size_t>(j)]) {
            seen[static_cast<std::size_t>(j)] = 1; j = perm[static_cast<std::size_t>(j)]; ++len;
        }
        cyc_len.push_back(len);
    }
    if (n_up < 0) return std::uint64_t(1) << cyc_len.size();   // 2^#cycles
    std::vector<std::uint64_t> dp(static_cast<std::size_t>(n_up) + 1, 0);
    dp[0] = 1;
    for (int L : cyc_len)
        for (int k = n_up; k >= L; --k)
            dp[static_cast<std::size_t>(k)] += dp[static_cast<std::size_t>(k - L)];
    return dp[static_cast<std::size_t>(n_up)];
}

// Exact per-RAW-sector dimensions via the Burnside / character (Molien) formula
//   dim(chi_s) = (1/|G|) * Re sum_g chi_s(g) |Fix(g)|
// (the multiplicity of the 1-D irrep chi_s in the fixed-Sz permutation
// representation). Cheap -- O(|G|^2) total -- and needs NO orbit walk, so it can
// drive the across-sector load balance BEFORE the expensive build.
inline std::vector<std::uint64_t>
sector_dims_burnside(const ::SymmetryGroupInfo& info, int n_up) {
    const std::size_t G = info.max_clique.size();
    std::vector<std::uint64_t> fix(G);
    for (std::size_t g = 0; g < G; ++g)
        fix[g] = num_fixed_states(info.max_clique[g], n_up);
    std::vector<std::uint64_t> dims(info.sectors.size(), 0);
    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        const auto chi = ed::symmetry::sector_characters_from(
            info, info.sectors[s].phase_factors);
        double acc = 0.0;
        for (std::size_t g = 0; g < G; ++g)
            acc += (chi[g] * static_cast<double>(fix[g])).real();
        const long long d = std::llround(acc / static_cast<double>(G));
        dims[s] = d > 0 ? static_cast<std::uint64_t>(d) : 0;
    }
    return dims;
}

// Greedy longest-processing-time bin-packing: hand each raw sector (largest dim
// first) to the currently least-loaded rank. Deterministic, so every rank
// computes the IDENTICAL ``owner[raw_s] = rank`` map. Solve + construction cost
// both scale ~linearly with sector dim, so dim is the load proxy.
inline std::vector<int>
greedy_sector_owner(const std::vector<std::uint64_t>& dims, int nranks) {
    std::vector<std::size_t> order(dims.size());
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::sort(order.begin(), order.end(),
              [&](std::size_t a, std::size_t b) { return dims[a] > dims[b]; });
    std::vector<std::uint64_t> load(static_cast<std::size_t>(nranks), 0);
    std::vector<int>           owner(dims.size(), 0);
    for (std::size_t s : order) {
        int best = 0;
        for (int r = 1; r < nranks; ++r)
            if (load[static_cast<std::size_t>(r)] < load[static_cast<std::size_t>(best)]) best = r;
        owner[s] = best;
        load[static_cast<std::size_t>(best)] += (dims[s] > 0 ? dims[s] : 1);
    }
    return owner;
}

}  // namespace detail

inline SectorOperatorSet
make_sector_operators_tagged(const OperatorSpec& spec,
                             int mpi_rank = 0, int mpi_size = 1) {
    if (!spec.streaming_symmetry) {
        throw std::runtime_error(
            "ed::make_sector_operators: requires "
            "OperatorSpec::streaming_symmetry = true (this lane builds the "
            "symmetry sector set; use ed::make_operator for the plain / "
            "fixed-Sz / distributed lanes).");
    }
    const std::string& dir = detail::require_directory(spec);
    const std::string cache_dir =
        ed::symmetry::resolve_sym_cache_dir(spec.basis_cache_dir, dir);

    // Carrier operator: load the Hamiltonian term list + the symmetry group
    // metadata exactly once. The terms are copied verbatim into every sector
    // operator by the term-builder below (identical to the proven
    // ``make_sector_operator_adopt`` term-copy contract).
    auto base = detail::build_base_op(spec);
    detail::load_terms_into(*base, spec);
    base->symmetry_info.loadFromDirectory(dir);

    auto term_builder = [&base](ed::symmetry::SectorOperator& op) {
        op.transform_data_  = base->transform_data_;
        op.three_body_data_ = base->three_body_data_;
    };

    SectorOperatorSet set;
    std::vector<std::size_t> sector_ids;

    // Across-sector MPI load balance: pre-compute exact per-sector dims via the
    // Burnside/character formula (cheap, no orbit walk) and greedy-pack them onto
    // ranks. owner[raw_s] = owning rank; nullptr (single-rank) => build all.
    std::vector<int> sector_owner;
    const std::vector<int>* owner_ptr = nullptr;
    if (mpi_size > 1) {
        const int n_up_for_dims = spec.fixed_sz.has_value()
            ? static_cast<int>(*spec.fixed_sz) : -1;
        const auto dims =
            detail::sector_dims_burnside(base->symmetry_info, n_up_for_dims);
        sector_owner = detail::greedy_sector_owner(dims, mpi_size);
        owner_ptr = &sector_owner;
        if (std::getenv("ED_DEBUG_BALANCE") && mpi_rank == 0) {
            std::vector<std::uint64_t> load(static_cast<std::size_t>(mpi_size), 0);
            for (std::size_t s = 0; s < dims.size(); ++s) load[static_cast<std::size_t>(sector_owner[s])] += dims[s];
            fprintf(stderr, "[BALANCE] burnside dims:");
            for (auto d : dims) fprintf(stderr, " %llu", (unsigned long long)d);
            fprintf(stderr, "\n[BALANCE] per-rank load:");
            for (auto l : load) fprintf(stderr, " %llu", (unsigned long long)l);
            fprintf(stderr, "\n"); fflush(stderr);
        }
    }

    // Structural consolidation (Jul 2026): ONE decode of the sector-mode
    // flags with the illegal combinations rejected up front (they were
    // previously caught -- or not -- deep inside the builders).
    //   diagonal axis: fixed_sz XOR sz_parity XOR none
    //   flip split   : flip_project_half (fixed-Sz N/2 only) or
    //                  flip_sectors_full (full space / parity halves;
    //                  parity requires even N by the closure rule)
    if (spec.fixed_sz && spec.sz_parity) {
        throw std::invalid_argument(
            "OperatorSpec: fixed_sz and sz_parity are mutually exclusive "
            "diagonal axes.");
    }
    if (spec.flip_project_half && !spec.fixed_sz) {
        throw std::invalid_argument(
            "OperatorSpec: flip_project_half is the fixed-Sz N/2 variant; "
            "use flip_sectors_full for full-space / parity sectors.");
    }
    if (spec.flip_sectors_full && spec.fixed_sz) {
        throw std::invalid_argument(
            "OperatorSpec: flip_sectors_full does not apply to a fixed-Sz "
            "block (use flip_project_half at n_up == N/2).");
    }
    if (spec.flip_sectors_full && spec.sz_parity
        && spec.num_sites % 2 != 0) {
        throw std::invalid_argument(
            "OperatorSpec: the all-ones flip only preserves Sz parity for "
            "even N (closure rule).");
    }

    const bool time_ctor = std::getenv("ED_TIME_CONSTRUCTION") != nullptr;
    const auto ctor_t0 = std::chrono::steady_clock::now();
    if (spec.sz_parity.has_value()) {
        // Sz-parity halves (diagonal Z2 remnant), one RepSectorData per
        // (parity, irrep[, flip sign]); rep lanes only.
        set.operators = ed::symmetry::build_parity_sector_operators_lazy(
            static_cast<std::uint64_t>(spec.num_sites), spec.spin_l,
            (*spec.sz_parity >= 2) ? -1 : *spec.sz_parity,
            base->symmetry_info, term_builder, &sector_ids, cache_dir,
            spec.flip_sectors_full);
    } else if (spec.fixed_sz.has_value()) {
        // CSR-free lazy-rep regime (memory-bounded large systems, e.g. N=32
        // fixed-Sz mTPQ): hand out operators that know their dim up-front and
        // defer the per-sector orbit CSR / GPU RepSectorData. Small/moderate
        // systems stay eager (fast precomputed-CSR matvec). Same budget knobs
        // as the streaming operator so both production paths agree.
        if (spec.flip_project_half
            || detail::fixed_sz_sectors_should_be_lazy(
                static_cast<std::uint64_t>(spec.num_sites),
                static_cast<std::int64_t>(*spec.fixed_sz),
                base->symmetry_info)) {
            set.operators = ed::symmetry::build_fixed_sz_sector_operators_lazy(
                static_cast<std::uint64_t>(spec.num_sites), spec.spin_l,
                static_cast<std::int64_t>(*spec.fixed_sz),
                base->symmetry_info, term_builder, &sector_ids,
                mpi_rank, mpi_size, owner_ptr, cache_dir, spec.flip_project_half);
        } else {
            set.operators = ed::symmetry::build_fixed_sz_sector_operators(
                static_cast<std::uint64_t>(spec.num_sites), spec.spin_l,
                static_cast<std::int64_t>(*spec.fixed_sz),
                base->symmetry_info, term_builder, &sector_ids,
                mpi_rank, mpi_size, owner_ptr, cache_dir);
        }
    } else {
        // Pure-spatial symmetry (no Sz). Large N -> CSR-free rep-walk lazy lane
        // (memory-bounded, stabilizer-fused construction); small N stays eager.
        if (spec.flip_sectors_full
            || detail::full_sectors_should_be_lazy(
                static_cast<std::uint64_t>(spec.num_sites), base->symmetry_info)) {
            set.operators = ed::symmetry::build_full_sector_operators_lazy(
                static_cast<std::uint64_t>(spec.num_sites), spec.spin_l,
                base->symmetry_info, term_builder, &sector_ids,
                mpi_rank, mpi_size, owner_ptr, cache_dir,
                spec.flip_sectors_full);
        } else {
            set.operators = ed::symmetry::build_full_sector_operators(
                static_cast<std::uint64_t>(spec.num_sites), spec.spin_l,
                base->symmetry_info, term_builder, &sector_ids,
                mpi_rank, mpi_size, owner_ptr, cache_dir);
        }
    }
    if (time_ctor) {
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - ctor_t0).count();
        fprintf(stderr, "[CONSTRUCTION] sector build: %.1f ms  (|G|=%zu, sectors_built=%zu)\n",
                ms, base->symmetry_info.max_clique.size(), set.operators.size());
        fflush(stderr);
    }

    set.num_raw_sectors = base->symmetry_info.sectors.size();
    set.all_quantum_numbers.reserve(set.num_raw_sectors);
    for (const auto& s : base->symmetry_info.sectors) {
        set.all_quantum_numbers.push_back(s.quantum_numbers);
    }
    const int n_up = spec.fixed_sz.has_value()
        ? static_cast<int>(*spec.fixed_sz) : -1;
    set.tags.reserve(set.operators.size());
    for (std::size_t i = 0; i < set.operators.size(); ++i) {
        ed::SectorTag tag;
        tag.sector_index = sector_ids[i];
        tag.sector_dim   = static_cast<std::uint64_t>(set.operators[i]->dim());
        const std::size_t nraw = base->symmetry_info.sectors.size();
        if (sector_ids[i] < nraw) {
            tag.quantum_numbers =
                base->symmetry_info.sectors[sector_ids[i]].quantum_numbers;
        } else if (nraw > 0) {
            // Synthetic sector k + slot*num_raw: carry the raw irrep's
            // quantum numbers plus the slot labels. Slot layouts:
            //   flip only          : slot = flip sign (0 -> +1, 1 -> -1)
            //   parity only        : slot = parity     (0 -> +1 even, 1 -> -1 odd)
            //   parity x flip      : slot = 2*parity + sign
            tag.quantum_numbers =
                base->symmetry_info.sectors[sector_ids[i] % nraw]
                    .quantum_numbers;
            const std::size_t slot = sector_ids[i] / nraw;
            if (spec.sz_parity.has_value()) {
                const int n_signs = spec.flip_sectors_full ? 2 : 1;
                int p_idx = static_cast<int>(slot) / n_signs;
                if (*spec.sz_parity == 1) p_idx += 1;   // odd-only set
                tag.quantum_numbers.push_back(p_idx == 0 ? +1 : -1);
                if (spec.flip_sectors_full)
                    tag.quantum_numbers.push_back(
                        (slot % n_signs) == 0 ? +1 : -1);
            } else {
                tag.quantum_numbers.push_back(slot == 0 ? +1 : -1);
            }
        }
        tag.n_up = n_up;
        set.tags.push_back(std::move(tag));
    }
    return set;
}

/// Operators-only convenience: drops the tags. Preserves the original
/// ``make_sector_operators(spec)`` surface for callers that don't need the
/// per-sector irrep metadata (the GPU parity unit tests, the e2e check).
inline std::vector<std::unique_ptr<ed::symmetry::SectorOperator>>
make_sector_operators(const OperatorSpec& spec) {
    return make_sector_operators_tagged(spec).operators;
}

// ---------------------------------------------------------------------------
// make_all_sz_sector_operators_tagged: one-pass all-Sz sector factory.
//
// Unlike calling ``make_sector_operators_tagged`` once per n_up (which reads
// the automorphism directory, parses the symmetry JSON, and runs an orbit-rep
// scan for each Sz sector independently), this function:
//   1. Loads the Hamiltonian and symmetry group metadata ONCE.
//   2. Calls ``enumerate_full_orbit_reps`` ONCE (O(2^N × |G|)).
//   3. Partitions reps by n_up and builds all (n_up, irrep) sectors in one
//      nested loop.
//
// The returned SectorOperatorSet covers every (n_up, irrep) sector in the
// window [n_up_min, n_up_max]. tags[i].n_up carries the Sz sector; tags[i].
// sector_index carries the raw irrep index within that Sz sector. A single
// flat parallel loop over the set replaces the nested (n_up outer, irrep
// inner) double loop, removing all per-n_up cold-start overhead.
//
// Designed for the thermal use-case on many-core machines: build once, run all
// (n_up, irrep) sectors in one ``#pragma omp parallel for``.
// ---------------------------------------------------------------------------
inline SectorOperatorSet
make_all_sz_sector_operators_tagged(const OperatorSpec& spec,
                                    int n_up_min = 0,
                                    int n_up_max = -1,
                                    bool flip_project_half = false) {
    if (!spec.streaming_symmetry) {
        throw std::runtime_error(
            "ed::make_all_sz_sector_operators_tagged: requires "
            "streaming_symmetry = true.");
    }
    const std::string& dir = detail::require_directory(spec);
    const std::uint64_t n_bits = static_cast<std::uint64_t>(spec.num_sites);
    if (n_up_max < 0)
        n_up_max = static_cast<int>(n_bits);

    // Load operator terms + symmetry group info ONCE.
    auto base = detail::build_base_op(spec);
    detail::load_terms_into(*base, spec);
    base->symmetry_info.loadFromDirectory(dir);

    auto term_builder = [&base](ed::symmetry::SectorOperator& op) {
        op.transform_data_  = base->transform_data_;
        op.three_body_data_ = base->three_body_data_;
    };

    const std::string cache_dir =
        ed::symmetry::resolve_sym_cache_dir(spec.basis_cache_dir, dir);
    std::vector<std::pair<int, std::size_t>> n_up_sector_ids;
    std::vector<std::unique_ptr<ed::symmetry::SectorOperator>> operators =
        ed::symmetry::build_all_sz_sector_operators(
            n_bits, spec.spin_l, base->symmetry_info, term_builder,
            static_cast<std::int64_t>(n_up_min),
            static_cast<std::int64_t>(n_up_max),
            &n_up_sector_ids, cache_dir, flip_project_half);

    SectorOperatorSet set;
    set.num_raw_sectors = base->symmetry_info.sectors.size();
    set.all_quantum_numbers.reserve(set.num_raw_sectors);
    for (const auto& s : base->symmetry_info.sectors)
        set.all_quantum_numbers.push_back(s.quantum_numbers);
    set.operators = std::move(operators);
    set.tags.reserve(set.operators.size());
    for (std::size_t i = 0; i < set.operators.size(); ++i) {
        const auto& [n_up, raw_s] = n_up_sector_ids[i];
        ed::SectorTag tag;
        tag.n_up         = n_up;
        tag.sector_index = raw_s;
        tag.sector_dim   = static_cast<std::uint64_t>(
            set.operators[i]->dim());
        if (raw_s < base->symmetry_info.sectors.size())
            tag.quantum_numbers =
                base->symmetry_info.sectors[raw_s].quantum_numbers;
        set.tags.push_back(std::move(tag));
    }
    return set;
}

// ---------------------------------------------------------------------------
// Streaming-symmetry lane (single LinearOperator).
//
// Operator-collapse Phase 3 (Jun 2026): the StreamingSymmetryOperator /
// FixedSzStreamingSymmetryOperator carriers have been retired. Symmetry
// sectors are now built directly as standalone ``SectorOperator``s by
// ``make_sector_operators_tagged``. Since ``make_operator`` must return a
// SINGLE ``LinearOperator``, this lane returns ONE symmetry sector selected
// by ``spec.sector_index`` (raw irrep index). For the full multi-sector
// enumeration -- the production CLI / Python sector loop -- call
// ``ed::make_sector_operators_tagged(spec)`` directly.
inline std::unique_ptr<LinearOperator>
make_streaming_symmetry_operator(const OperatorSpec& spec) {
    SectorOperatorSet set = make_sector_operators_tagged(spec);
    if (set.operators.empty()) {
        throw std::runtime_error(
            "ed::make_operator: streaming-symmetry spec produced no "
            "non-empty symmetry sectors for the requested (n_up, group).");
    }
    std::size_t pos = 0;
    if (spec.sector_index.has_value()) {
        const std::size_t raw = *spec.sector_index;
        bool found = false;
        for (std::size_t i = 0; i < set.tags.size(); ++i) {
            if (set.tags[i].sector_index == raw) {
                pos = i;
                found = true;
                break;
            }
        }
        if (!found) {
            throw std::runtime_error(
                "ed::make_operator: requested streaming-symmetry "
                "sector_index is empty (orbits cancel) or out of range. Use "
                "ed::make_sector_operators_tagged(spec) for the full set.");
        }
    }
    return std::unique_ptr<LinearOperator>(set.operators[pos].release());
}

}  // namespace ed

namespace ed::core {

// ---------------------------------------------------------------------------
// SectorSetView: a copyable, handle-shaped view over a ``SectorOperatorSet``
// (operator-collapse Phase 3, Jun 2026). It exposes the SAME random-access
// surface as the legacy ``StreamingSymmetryHandle`` -- ``num_sectors()``,
// ``sector(raw_k)``, ``sector_tag(raw_k)`` -- so the per-sector loops in the
// CLI / Python bindings can switch from the monolithic
// ``FixedSzStreamingSymmetryOperator`` + handle to the direct
// ``make_sector_operators_tagged`` enumeration with a near-mechanical type
// swap.
//
// Unlike the handle (which builds a fresh sector operator per ``sector(k)``
// call), this view owns the (compacted) sector set and returns a stable,
// non-owning pointer to the persistent operator -- callers reuse the same
// object across the two-phase scan instead of rebuilding it. The set is held
// behind a ``shared_ptr`` so the view is cheaply copyable (the cross-irrep
// bindings copy their handle for the ``delta == 0`` src==dst case).
//
// ``raw_k`` is the RAW irrep index in ``[0, num_sectors())``; empty irreps
// were dropped from the set, so ``sector(raw_k)`` returns ``nullptr`` for them
// (equivalent to the handle's ``dim() == 0`` sentinel the loops already
// guard against).
// ---------------------------------------------------------------------------
class SectorSetView {
public:
    explicit SectorSetView(ed::SectorOperatorSet set)
        : set_(std::make_shared<ed::SectorOperatorSet>(std::move(set))),
          index_space_(set_->num_raw_sectors) {
        for (std::size_t i = 0; i < set_->tags.size(); ++i) {
            raw_to_pos_[set_->tags[i].sector_index] = i;
            // Stage 8c: flip-projected sets carry SYNTHETIC indices
            // k + parity * num_raw beyond the raw irrep count; widen the
            // iteration space so per-sector loops reach the (k, -) blocks.
            index_space_ = std::max(index_space_,
                                    set_->tags[i].sector_index + 1);
        }
    }

    [[nodiscard]] std::size_t num_sectors() const noexcept {
        return index_space_;
    }

    /// Non-owning pointer to the persistent per-sector operator for the RAW
    /// irrep index ``raw_k`` (``nullptr`` for an empty / dropped irrep).
    [[nodiscard]] ed::symmetry::SectorOperator*
    sector(std::size_t raw_k) const {
        auto it = raw_to_pos_.find(raw_k);
        return it == raw_to_pos_.end()
            ? nullptr
            : set_->operators[it->second].get();
    }

    /// Quantum-number tag for the RAW irrep index ``raw_k``. For a dropped
    /// (empty) irrep, returns a tag carrying ``sector_index`` + the raw
    /// ``quantum_numbers`` (dim 0). The quantum numbers are surfaced even for
    /// empty irreps so the cross-irrep selection-rule walker can infer the
    /// full generator-order lattice (matching the legacy
    /// ``StreamingSymmetryHandle::sector_tag`` contract, which reads cheap
    /// generation metadata for every raw sector).
    [[nodiscard]] ed::SectorTag sector_tag(std::size_t raw_k) const {
        auto it = raw_to_pos_.find(raw_k);
        if (it != raw_to_pos_.end()) return set_->tags[it->second];
        ed::SectorTag t;
        t.sector_index = raw_k;
        if (raw_k < set_->all_quantum_numbers.size()) {
            t.quantum_numbers = set_->all_quantum_numbers[raw_k];
        }
        return t;
    }

private:
    std::shared_ptr<ed::SectorOperatorSet>       set_;
    std::size_t                                  index_space_ = 0;
    std::unordered_map<std::size_t, std::size_t> raw_to_pos_;
};

}  // namespace ed::core

namespace ed {

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
