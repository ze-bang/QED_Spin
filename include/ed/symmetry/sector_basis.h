#pragma once
// =============================================================================
// include/ed/symmetry/sector_basis.h
//
// SectorBasis: the owning host-side description of ONE symmetry-projected
// sector. Completes the basis-owning "triplet" started by the Orthogonal
// symmetry composition wave:
//
//     FullSpaceSubspace   (subspace.h)  -> FullBasisPolicy      view
//     FixedSzSubspace     (subspace.h)  -> FixedSzBasisPolicy   view
//     SectorBasis         (this file)   -> SymmetryBasisPolicy  view
//
// A SectorBasis is the symmetry analogue of a Subspace: it OWNS the orbit
// data (one ``SymmetrySector`` plus the state->orbit lookup index) and
// hands out the non-owning POD ``SymmetryBasisPolicy`` that the unified
// matvec kernels (``ed::matvec::kernel::apply_terms`` and its GPU twin)
// consume. It is the single owning artefact the future
// ``Operator<SymmetryBasisPolicy, MemSpace>`` holds, replacing the
// per-sector state that ``StreamingSymmetryOperator`` /
// ``FixedSzStreamingSymmetryOperator`` currently bake into their class
// identity.
//
// Why this exists (operator-collapse refactor, Jun 2026):
// ------------------------------------------------------
// The four legacy operator classes (Operator / FixedSzOperator /
// StreamingSymmetryOperator / FixedSzStreamingSymmetryOperator) differ
// ONLY in their basis. The basis axis is already expressed as a POD
// policy + an owning producer for the trivial cases (Subspace). The
// symmetry case lacked a standalone owning producer -- the orbit data
// lived inside the operator and was reachable only through the nested
// ``SectorView``. SectorBasis lifts that orbit data into a free-standing,
// operator-independent value so a single operator template can hold any
// of the three host policies uniformly.
//
// Construction routes:
//   * ``SectorBasis::build(subspace, projector, sector_meta)``
//        Enumerates the sector's orbits over the chosen Subspace using
//        the shared ``compute_orbit_for_state`` host helper (bit-identical
//        to the legacy ``computeOrbitData*`` member methods), then builds
//        the state->orbit lookup index.
//   (``SectorBasis::adopt`` -- the migration bridge for already-
//   materialised sectors -- was deleted in Stage 11c-2a with the eager
//   builders; ``configureRepLazy`` + ``build`` are the two entry points.)
//
// Lookup strategy: the shipped path uses the ``SortedUint64Index``
// fallback (O(log |orbit_total|) per find), exposed through a
// ``SectorLookupHandle`` with ``dense == nullptr`` and ``lin == nullptr``
// so the policy's ``index_of`` consults the sorted keys directly. The
// dense / Lin-indirection fast paths remain available on the legacy
// operators and can be layered onto SectorBasis in a later wave without
// changing this surface.
// =============================================================================

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

#include <ed/core/sorted_uint64_index.h>
#include <ed/symmetry/symmetry_sector_data.h>      // SymmetrySector, SymBasisState,
                                             // SectorLookupHandle
#include <ed/matvec/symmetry_basis_policy.h>  // SymmetryBasisPolicy + factory
#include <ed/symmetry/projector.h>
#include <ed/symmetry/projector_chain.h>
#include <ed/symmetry/rep_sector_data.h>      // RepSectorData (CSR-free rep path)
#include <ed/symmetry/subspace.h>
#include <ed/symmetry/sym_profile.h>       // SymPhaseTimer (ED_SYM_PROFILE=1)

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// O(1) rep reverse-lookup gate ("Optimized symmetry ED" plan, Phase A).
//
// The CSR-free rep matvec can resolve ``state -> orbit index`` either by a
// binary search over the sorted ``reps`` array (O(log dim), zero extra memory)
// or by a dense combinadic rank table (O(1), C(n_sites,n_up) int32). For an
// iterative solver doing thousands of matvecs the table is reused every
// iteration, so it pays for itself -- but it costs ~2.4 GiB at N=32, so it is
// gated by a memory budget.
//
//   ED_SYM_REP_RANKTABLE = "0"  -> force OFF (always binary search)
//                          "1"  -> force ON  (build regardless of budget)
//                          unset -> build when table bytes <= budget
//   ED_SYM_REP_RANKTABLE_BUDGET_GIB (default 8) sets the budget.
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool rep_rank_table_enabled(std::uint64_t table_entries) noexcept {
    if (table_entries == 0) return false;
    const char* force = std::getenv("ED_SYM_REP_RANKTABLE");
    if (force != nullptr && force[0] == '0' && force[1] == '\0') return false;
    if (force != nullptr && force[0] == '1' && force[1] == '\0') return true;
    double budget_gib = 8.0;
    if (const char* b = std::getenv("ED_SYM_REP_RANKTABLE_BUDGET_GIB")) {
        const double parsed = std::atof(b);
        if (parsed > 0.0) budget_gib = parsed;
    }
    const long double table_bytes =
        static_cast<long double>(table_entries) * sizeof(std::int32_t);
    const long double budget_bytes =
        static_cast<long double>(budget_gib) * 1024.0L * 1024.0L * 1024.0L;
    return table_bytes <= budget_bytes;
}

// ---------------------------------------------------------------------------
// SectorBasis -- owning orbit data + lookup for one symmetry sector.
//
// Movable, non-copyable: it owns a ``SortedUint64Index`` whose addresses
// the ``SectorLookupHandle`` (and hence any live ``SymmetryBasisPolicy``)
// references. Copying would silently dangle those handles, so we delete
// the copy operations and require explicit moves.
// ---------------------------------------------------------------------------
class SectorBasis {
public:
    // Survival cutoff on a symmetrised orbit's norm^2; matches the legacy
    // StreamingSymmetryOperator (``norm_sq > 1e-10``) so the basis ordering
    // and dimension stay bit-identical to the legacy sectors.
    static constexpr double kOrbitNormSqEpsilon = 1e-10;

    SectorBasis() = default;
    SectorBasis(const SectorBasis&)            = delete;
    SectorBasis& operator=(const SectorBasis&) = delete;
    SectorBasis(SectorBasis&&)                 = default;
    SectorBasis& operator=(SectorBasis&&)      = default;

    // (Stage 11c-2a: the ``adopt()`` bridge -- take ownership of an
    // already-materialised SymmetrySector -- was deleted with its last
    // consumer, the eager sector builders. Every SectorBasis is now
    // constructed rep-lazy via ``configureRepLazy``, or through
    // ``build()`` below when a csr_provider materialises on demand.)

    // -----------------------------------------------------------------
    // build: enumerate this sector's orbits over ``subspace`` using the
    // single shared ``compute_orbit_for_state`` host helper, producing
    // orbit data bit-identical to the legacy ``computeOrbitData`` /
    // ``computeOrbitDataFixedSz`` member methods.
    //
    // ``orbit_reps`` is the deduplicated list of orbit representatives in
    // this sector (one per ``SymBasisState``); the caller computes it
    // exactly as the legacy "Pass 1" does (minimum computational state
    // over the orbit, restricted to ``subspace``). Passing it in keeps
    // SectorBasis agnostic to the rep-enumeration strategy (full scan vs
    // lazy streaming) -- it only owns the per-rep orbit expansion.
    //
    // ``sector_meta`` carries the irrep label (quantum_numbers +
    // phase_factors) verbatim from ``SectorMetadata``.
    // -----------------------------------------------------------------
    template <class SubspaceT>
    [[nodiscard]] static SectorBasis
    build(const SubspaceT&                  subspace,
          const SpatialProjector&           projector,
          const std::vector<int>&           quantum_numbers,
          const std::vector<Complex>&       phase_factors,
          const std::vector<std::uint64_t>& orbit_reps,
          std::uint64_t                     sector_id = 0)
    {
        SymPhaseTimer prof("pass2 sector-basis build (eager orbit CSR, one irrep)");
        SectorBasis sb;
        sb.group_size_ = projector.group_info().max_clique.size();

        SymmetrySector& sec = sb.sector_;
        sec.sector_id        = sector_id;
        sec.quantum_numbers  = quantum_numbers;
        sec.phase_factors    = phase_factors;
        sec.basis_states.reserve(orbit_reps.size());

        std::vector<std::uint64_t> elems;
        std::vector<Complex>       coeffs;
        double                     norm_sq = 0.0;

        for (std::uint64_t rep : orbit_reps) {
            compute_orbit_for_state(subspace, projector, rep,
                                    phase_factors, elems, coeffs, norm_sq);
            // Survival threshold matches the legacy streaming build
            // (StreamingSymmetryOperator pass 2 / pass 1.5): an orbit
            // whose symmetrised norm^2 falls at or below 1e-10 is a
            // numerically-null state in this irrep and is NOT a basis
            // state. Using the same cutoff keeps the orbit count and the
            // index ordering bit-identical to the legacy sectors.
            if (elems.empty() || norm_sq <= kOrbitNormSqEpsilon) {
                continue;  // orbit fully cancels in this irrep
            }
            SymBasisState st(rep, quantum_numbers, std::sqrt(norm_sq));
            st.orbit_elements     = elems;
            st.orbit_coefficients = coeffs;
            st.sortOrbit();  // O(log) findCoeff in the matvec hot loop
            sec.basis_states.push_back(std::move(st));
        }

        sb.rebuild_lookup_();
        prof.set_items(sec.basis_states.size());
        return sb;
    }

    // -----------------------------------------------------------------
    // build_prefiltered: Stage-2 (SymmetryEngine v2) parallel twin of
    // ``build``. The caller supplies the closed-form projected norm² per
    // rep (rep_projection.h / orbit_table.h -- |Σ_{h∈Stab}χ(h)|²/|Stab|),
    // which decides survival WITHOUT walking the orbit; only survivors
    // pay the O(|G|) ``compute_orbit_for_state`` walk, and the walk runs
    // OpenMP-parallel over reps (each survivor writes its pre-assigned
    // slot, so the output ordering -- ascending rep order -- and every
    // stored orbit/norm are bit-identical to the serial ``build``).
    //
    // PRECONDITION (same as the closed form): the full G-orbit of every
    // rep lies inside ``subspace`` (full space / popcount-preserving
    // fixed-Sz). A disagreement between the prefilter and the walk means
    // the precondition was violated; this throws rather than silently
    // producing a mis-indexed basis.
    // -----------------------------------------------------------------
    template <class SubspaceT>
    [[nodiscard]] static SectorBasis
    build_prefiltered(const SubspaceT&                  subspace,
                      const SpatialProjector&           projector,
                      const std::vector<int>&           quantum_numbers,
                      const std::vector<Complex>&       phase_factors,
                      const std::vector<std::uint64_t>& orbit_reps,
                      const std::vector<double>&        prefilter_norm_sq,
                      std::uint64_t                     sector_id = 0)
    {
        SymPhaseTimer prof("pass2 sector-basis build (prefiltered parallel, one irrep)");
        SectorBasis sb;
        sb.group_size_ = projector.group_info().max_clique.size();

        SymmetrySector& sec = sb.sector_;
        sec.sector_id       = sector_id;
        sec.quantum_numbers = quantum_numbers;
        sec.phase_factors   = phase_factors;

        // Survivor prefix-sum: slot[i] = output index of rep i (valid only
        // when the prefilter says it survives).
        const std::size_t n = orbit_reps.size();
        std::vector<std::uint32_t> slot(n);
        std::size_t n_surv = 0;
        for (std::size_t i = 0; i < n; ++i) {
            slot[i] = static_cast<std::uint32_t>(n_surv);
            if (prefilter_norm_sq[i] > kOrbitNormSqEpsilon) ++n_surv;
        }
        sec.basis_states.resize(n_surv);

        bool precondition_violated = false;
#ifdef _OPENMP
#       pragma omp parallel
#endif
        {
            std::vector<std::uint64_t> elems;
            std::vector<Complex>       coeffs;
#ifdef _OPENMP
#           pragma omp for schedule(dynamic, 256)
#endif
            for (long long ii = 0; ii < static_cast<long long>(n); ++ii) {
                const std::size_t i = static_cast<std::size_t>(ii);
                if (prefilter_norm_sq[i] <= kOrbitNormSqEpsilon) continue;
                double norm_sq = 0.0;
                compute_orbit_for_state(subspace, projector, orbit_reps[i],
                                        phase_factors, elems, coeffs, norm_sq);
                if (elems.empty() || norm_sq <= kOrbitNormSqEpsilon) {
                    precondition_violated = true;  // benign race: only ever set true
                    continue;
                }
                SymBasisState st(orbit_reps[i], quantum_numbers,
                                 std::sqrt(norm_sq));
                st.orbit_elements     = elems;
                st.orbit_coefficients = coeffs;
                st.sortOrbit();
                sec.basis_states[slot[i]] = std::move(st);
            }
        }
        if (precondition_violated) {
            throw std::runtime_error(
                "SectorBasis::build_prefiltered: closed-form prefilter and "
                "orbit walk disagree on survival -- the subspace drops orbit "
                "images (closed-form precondition violated). Use "
                "SectorBasis::build for this subspace.");
        }

        sb.rebuild_lookup_();
        prof.set_items(sec.basis_states.size());
        return sb;
    }

    // -----------------------------------------------------------------
    // Basis introspection (mirrors the Subspace surface).
    // -----------------------------------------------------------------
    [[nodiscard]] std::uint64_t dim() const noexcept {
        // CSR-free lazy regime: the sector dimension is known up-front (from
        // Pass 1.5 ``getSectorDimension``) before the host orbit CSR is
        // materialised, so report it directly while the orbit CSR is still
        // empty. After a CPU ``apply`` materialises the CSR the two agree.
        if (rep_lazy_ && sector_.basis_states.empty()) {
            return rep_dim_;
        }
        return sector_.basis_states.size();
    }

    [[nodiscard]] std::uint64_t state_of(std::uint64_t idx) const noexcept {
        return sector_.basis_states[idx].orbit_rep;
    }

    [[nodiscard]] std::int64_t index_of(std::uint64_t state) const noexcept {
        const std::size_t k = lookup_.find(state);
        return (k == ed::core::SortedUint64Index::kNotFound)
            ? std::int64_t{-1}
            : static_cast<std::int64_t>(k);
    }

    [[nodiscard]] std::size_t group_size() const noexcept {
        return group_size_;
    }

    [[nodiscard]] const SymmetrySector& sector() const noexcept {
        return sector_;
    }

    // -----------------------------------------------------------------
    // policy(): the non-owning POD view the matvec kernels consume.
    //
    // The returned ``SymmetryBasisPolicy`` holds a pointer into this
    // object's ``sector_`` and a ``SectorLookupHandle`` referencing this
    // object's ``lookup_``. It stays valid only as long as this
    // SectorBasis is alive AND not moved-from; treat it as a borrow.
    // -----------------------------------------------------------------
    [[nodiscard]] ed::matvec::basis::SymmetryBasisPolicy
    policy() const noexcept {
        return ed::matvec::basis::make_symmetry_basis(
            sector_, lookup_handle_(), static_cast<double>(group_size_));
    }

    // ABI alignment with the ed::matvec::basis::* policies + Subspace
    // surface: a SectorBasis carries the same compile-time trait flags
    // as the symmetry policy it produces.
    static constexpr bool may_leave_basis    = true;
    static constexpr bool needs_orbit_walk   = true;
    static constexpr bool has_coeff_modifier = true;
    static constexpr bool is_distributed     = false;

    // =================================================================
    // CSR-FREE lazy rep mode (operator-collapse Phase 4: folded down from
    // ``SectorOperator``). The production streaming-symmetry sector loop
    // used to fully materialise each sector's host orbit CSR (~24 GiB/sector
    // at N=32) even though the GPU on-the-fly representative matvec needs
    // none of it. ``configureRepLazy`` lets the loop hand over a SectorBasis
    // that:
    //   * knows its ``dim`` up-front (Pass 1.5 ``getSectorDimension``),
    //   * knows its real/complex character up-front (cheap |G| characters),
    //   * builds the CSR-free RepSectorData ON DEMAND (only when the GPU /
    //     CPU rep path actually engages) via ``rep_provider``, and
    //   * materialises the host orbit CSR ON DEMAND (only if a CPU orbit-CSR
    //     ``apply`` is ever invoked -- never on the GPU rep path) via
    //     ``csr_provider``.
    // =================================================================
    void configureRepLazy(std::uint64_t                   dim,
                          std::size_t                     group_size,
                          bool                            is_real,
                          std::function<RepSectorData()>  rep_provider,
                          std::function<SymmetrySector()> csr_provider,
                          bool                            csr_available = true) {
        rep_lazy_       = true;
        rep_dim_        = dim;
        group_size_     = group_size;
        rep_is_real_    = is_real;
        rep_provider_   = std::move(rep_provider);
        csr_provider_   = std::move(csr_provider);
        csr_available_  = csr_available;
    }

    [[nodiscard]] bool rep_lazy() const noexcept { return rep_lazy_; }

    /// False for rep-ONLY sectors (flip-extended / Sz-parity): the
    /// orbit-CSR form does not exist and ``csr_provider`` throws by
    /// design. Consumers with a matvec fallback (dense assembly, the
    /// FullDiag column build) must check this before ensureHostCsr().
    [[nodiscard]] bool csr_available() const noexcept {
        return csr_available_;
    }
    [[nodiscard]] bool rep_is_real() const noexcept { return rep_is_real_; }

    // True once the host orbit CSR has actually been materialised (CPU
    // fallback). On the GPU rep path this stays false for the basis's whole
    // lifetime -- the invariant the host-memory optimisation rests on.
    [[nodiscard]] bool host_csr_materialized() const noexcept {
        return sector_.basis_states.size() > 0;
    }

    // CSR-free on-the-fly rep path source. The factories either populate this
    // eagerly (``setRepData``) or supply a CSR-free provider via
    // ``configureRepLazy``; ``ensureRepData`` builds it on first use without
    // ever materialising the host orbit CSR.
    void setRepData(RepSectorData rep) noexcept { rep_data_ = std::move(rep); }
    [[nodiscard]] const RepSectorData& repData() const noexcept { return rep_data_; }

    [[nodiscard]] const RepSectorData& ensureRepData() const {
        if (!rep_data_.usable() && rep_provider_) {
            rep_data_ = rep_provider_();
        }
        // Phase A: build the O(1) dense rank table once (gated by budget). The
        // RepSymmetryBasisPolicy consumes it transparently (binary-search
        // fallback when absent), so this only ever speeds the reverse lookup.
        if (rep_data_.usable() && !rep_data_.has_rank_table()
            && !rep_data_.has_two_level()  // Stage 4: shared table already wired
            && rep_rank_table_enabled(rep_data_.rank_table_entries())) {
            rep_data_.build_rank_table();
        }
        // N≤32 apply_perm LUT: byte-decomposition table (4 lookups vs N iters).
        // No-op when n_sites > 32 or already built.
        if (rep_data_.usable() && rep_data_.perm_lut_data.empty()) {
            rep_data_.build_perm_lut();
        }
        return rep_data_;
    }

    // Lazily materialise the host orbit CSR from the deferred provider. No-op
    // when the basis is already populated (eager factories) or no provider
    // was supplied.
    void ensureHostCsr() const {
        if (sector_.basis_states.size() == 0 && csr_provider_) {
            auto* self = const_cast<SectorBasis*>(this);
            self->sector_ = csr_provider_();
            self->rebuild_lookup_();
        }
    }

private:
    // Build the state->orbit lookup index from the orbit elements of every
    // SymBasisState. Maps every computational state appearing in any orbit
    // to its owning orbit index (the array position of the SymBasisState).
    void rebuild_lookup_() {
        lookup_ = ed::core::SortedUint64Index{};
        std::size_t total = 0;
        for (const auto& st : sector_.basis_states) {
            total += st.orbit_elements.size();
        }
        lookup_.reserve(total);
        for (std::size_t i = 0; i < sector_.basis_states.size(); ++i) {
            for (std::uint64_t s : sector_.basis_states[i].orbit_elements) {
                lookup_[s] = i;
            }
        }
        lookup_.finalize();
    }

    [[nodiscard]] SectorLookupHandle lookup_handle_() const noexcept {
        SectorLookupHandle h;
        h.fallback = &lookup_;
        h.dense    = nullptr;
        h.lin      = nullptr;
        return h;
    }

    SymmetrySector             sector_{};
    ed::core::SortedUint64Index lookup_{};
    std::size_t                group_size_ = 1;

    // ---- CSR-free lazy rep mode (see configureRepLazy) ----------------
    bool                            rep_lazy_    = false;
    std::uint64_t                   rep_dim_     = 0;
    bool                            rep_is_real_ = false;
    mutable RepSectorData           rep_data_{};   // CSR-free rep path source
    mutable std::function<RepSectorData()>  rep_provider_;
    bool                            csr_available_ = true;
    mutable std::function<SymmetrySector()> csr_provider_;
};

} // namespace ed::symmetry
