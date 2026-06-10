#pragma once
// =============================================================================
// include/ed/symmetry/sector_operator.h
//
// SectorOperator: the standalone per-sector symmetry operator -- the first
// concrete embodiment of the operator-class collapse target
// ``Operator<SymmetryBasisPolicy, Host>``.
//
// Where ``StreamingSymmetryOperator`` bakes the orbit data for EVERY sector
// into one monolithic class and exposes each sector only through a nested
// ``SectorView`` that calls back into the parent, a SectorOperator is a
// free-standing ``ed::LinearOperator`` that:
//
//   * OWNS the orbit data for exactly ONE symmetry sector, as an
//     ``ed::symmetry::SectorBasis`` (the owning producer of
//     ``SymmetryBasisPolicy``); and
//   * inherits the canonical term storage + management from ``ed::Operator``
//     (``addOneBodyTerm`` / ``addTwoBodyTerm`` / ``addThreeBodyTerm`` /
//     ``commitPendingTransforms`` / ``isReal`` / ``term_view_``).
//
// The matvec is driven by the unified ``CpuMatVecBackend<SymmetryBasisPolicy>``
// (via ``make_cpu_symmetry_backend``) -- exactly the backend the production
// ``StreamingSymmetryOperator::applySymmetrizedUnified`` path now routes
// through (see src/symmetry/streaming_symmetry_unified.cpp). The CSR /
// real-fast-path branches are compiled out for the symmetry policy
// (``needs_orbit_walk == true``), so apply() always runs the complex
// matrix-free orbit walk.
//
// Lifetime: the owned ``SectorBasis`` outlives the lazily-built backend
// (both are members; the backend holds a by-value ``SymmetryBasisPolicy``
// POD that points into the SectorBasis's sector + lookup index). Standard
// member-destruction order (backend declared in the base, basis here)
// is safe because the base's ``backend_`` is reset before this object's
// ``sector_basis_`` is destroyed only if we reset it -- see the note in
// the destructor.
//
// P2c of the operator-collapse refactor (Jun 2026).
// =============================================================================

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <ed/core/operator.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_gpu_mirror.h>
#include <ed/symmetry/rep_sector_data.h>

namespace ed::symmetry {

// ---------------------------------------------------------------------------
// CPU on-the-fly representative SpMV gate ("Optimized symmetry ED + NLCE"
// plan, Jun 2026). When enabled (default ON), a fixed-Sz symmetry sector's
// CPU matvec runs the CSR-free representative kernel
// (``make_cpu_rep_symmetry_backend``) instead of materialising the per-sector
// orbit CSR (~24 GiB/sector at N=32). Set ``ED_SYM_REP=0`` to restore the
// legacy orbit-CSR path (A/B + bisection).
// ---------------------------------------------------------------------------
[[nodiscard]] inline bool cpu_rep_symmetry_enabled() {
    static const bool enabled = [] {
        const char* e = std::getenv("ED_SYM_REP");
        if (e == nullptr || e[0] == '\0') return true;   // default ON
        if (e[0] == '0' && e[1] == '\0')  return false;  // "0" -> OFF
        return true;                                      // anything else -> ON
    }();
    return enabled;
}

class SectorOperator final : public ::Operator {
public:
    // -----------------------------------------------------------------
    // Construct from an owning SectorBasis. ``n_bits`` / ``spin_l`` set
    // the lattice constants the term kernels need; ``basis`` carries the
    // orbit data + lookup for this one sector. The Hamiltonian terms are
    // added afterwards through the inherited ``addOneBodyTerm`` /
    // ``addTwoBodyTerm`` / ``addThreeBodyTerm`` setters (or copied from a
    // source operator via the term-copy helper below).
    // -----------------------------------------------------------------
    SectorOperator(std::uint64_t n_bits, float spin_l, SectorBasis basis)
        : ::Operator(n_bits, spin_l), sector_basis_(std::move(basis)) {}

    SectorOperator(const SectorOperator&)            = delete;
    SectorOperator& operator=(const SectorOperator&) = delete;
    SectorOperator(SectorOperator&&)                 = delete;
    SectorOperator& operator=(SectorOperator&&)      = delete;

    ~SectorOperator() override {
        // The base ``backend_`` holds a SymmetryBasisPolicy POD that
        // points into ``sector_basis_``. Reset it here, BEFORE
        // ``sector_basis_`` is destroyed, so no dangling backend can
        // observe a half-destroyed basis (base-class members are
        // destroyed after this derived destructor body runs).
        backend_.reset();
    }

    // -----------------------------------------------------------------
    // Basis introspection.
    // -----------------------------------------------------------------
    [[nodiscard]] std::size_t dim() const override {
        // CSR-free lazy mode: the sector dimension is known up-front (from
        // Pass 1.5 ``getSectorDimension``) without ever materialising the
        // host orbit CSR, so report it directly. Otherwise it is the owned
        // SectorBasis's orbit count.
        return rep_lazy_
            ? static_cast<std::size_t>(rep_dim_)
            : static_cast<std::size_t>(sector_basis_.dim());
    }

    [[nodiscard]] const SectorBasis& basis() const noexcept {
        return sector_basis_;
    }

    // -----------------------------------------------------------------
    // On-the-fly representative GPU path ("On-the-fly representative SpMV"
    // plan, Jun 2026). When ``ED_GPU_SYMMETRY_REP`` is set AND this CSR-free
    // RepSectorData is usable (fixed-Sz sector), ``bind_cuda()`` builds a
    // resident GpuRepSectorMirror instead of uploading the orbit CSR. The
    // factory (``make_sector_operators`` / ``make_sector_operator_adopt``)
    // populates it; it stays empty for sym-only (full-Hilbert) sectors, which
    // fall back to the orbit-CSR mirror.
    // -----------------------------------------------------------------
    void setRepSectorData(RepSectorData rep) noexcept {
        rep_data_ = std::move(rep);
    }

    [[nodiscard]] const RepSectorData& repSectorData() const noexcept {
        return rep_data_;
    }

    // -----------------------------------------------------------------
    // CSR-FREE lazy rep mode ("scan other region" optimisation, Jun 2026).
    //
    // The production streaming-symmetry sector loop used to FULLY materialise
    // each sector's host orbit CSR (orbit_elements + orbit_coefficients +
    // the state->orbit lookup) in ``StreamingSymmetryHandle::sector(k)`` --
    // ~24 GiB/sector at N=32 (14.4 GiB CSR + 9.6 GiB SortedUint64Index) --
    // EVEN THOUGH the GPU on-the-fly representative matvec needs none of it.
    //
    // ``configureRepLazy`` lets the loop hand over a SectorOperator that:
    //   * knows its ``dim`` up-front (Pass 1.5 ``getSectorDimension``),
    //   * knows its real/complex character up-front (cheap |G| characters),
    //   * builds the CSR-free RepSectorData ON DEMAND (only when ``bind_cuda``
    //     actually engages the GPU rep path) via ``rep_provider``, and
    //   * materialises the host orbit CSR ON DEMAND (only if a CPU ``apply``
    //     is ever invoked -- never on the GPU path) via ``csr_provider``.
    //
    // Net effect: a GPU-only mTPQ+symmetry run NEVER allocates the per-sector
    // host orbit CSR; the host working set drops from ~24 GiB/sector to the
    // CSR-free RepSectorData (~1.2 GiB) plus transient orbit scratch.
    // -----------------------------------------------------------------
    void configureRepLazy(std::uint64_t                   dim,
                          std::size_t                     group_size,
                          bool                            is_real,
                          std::function<RepSectorData()>  rep_provider,
                          std::function<SymmetrySector()> csr_provider) {
        rep_lazy_       = true;
        rep_dim_        = dim;
        rep_group_size_ = group_size;
        rep_is_real_    = is_real;
        rep_provider_   = std::move(rep_provider);
        csr_provider_   = std::move(csr_provider);
    }

    [[nodiscard]] bool rep_lazy() const noexcept { return rep_lazy_; }

    // True once the host orbit CSR has actually been materialised (CPU
    // fallback). On the GPU rep path this stays false for the operator's
    // whole lifetime -- the invariant the host-memory optimisation rests on.
    [[nodiscard]] bool host_csr_materialized() const noexcept {
        return sector_basis_.dim() > 0;
    }

    [[nodiscard]] std::string description() const override {
        return "SectorOperator(dim=" + std::to_string(dim()) + ")";
    }

    // -----------------------------------------------------------------
    // Matvec entry points. Identical structure to ``Operator::apply`` /
    // ``Operator::apply_real`` but the size check is against the SECTOR
    // dimension (not the full 2^N Hilbert space), and the backend is the
    // symmetry one supplied by ``make_backend_`` below.
    // -----------------------------------------------------------------
    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        if (size != dim()) {
            throw std::invalid_argument(
                "SectorOperator::apply: input/output vector size mismatch");
        }
        ensure_backend_();
        const auto tv = term_view_();  // rebuilds SoA cache if stale
        backend_->apply_complex(&tv, in, out, size);
    }

    void apply_real(const double* in, double* out, std::size_t size) const override {
        if (size != dim()) {
            throw std::invalid_argument(
                "SectorOperator::apply_real: input/output vector size mismatch");
        }
        ensure_backend_();
        const auto tv = term_view_();  // rebuilds SoA cache if stale
        backend_->apply_real(&tv, in, out, size);
    }

    // -----------------------------------------------------------------
    // Real-Hermitian fast-path eligibility.
    //
    // A symmetry sector is real-Hermitian-capable iff (i) the Hamiltonian
    // coefficients are real (``Operator::isReal``) AND (ii) every orbit
    // coefficient in THIS sector is real within tol -- i.e. the irrep
    // carries no genuine complex phase (k=0 / real-character irreps). A
    // non-trivial momentum sector (k != 0) has complex orbit phases, so
    // ``apply_real`` would silently drop the imaginary part; we therefore
    // keep such sectors on the complex path by returning false here. The
    // orchestrator only dispatches ``bind_real_cpu`` when this returns
    // true, so the guard is sufficient.
    // -----------------------------------------------------------------
    [[nodiscard]] bool is_real_hermitian() const noexcept override {
        // In CSR-free lazy mode the per-sector real/complex character is
        // precomputed from the |G| sector characters (``rep_is_real_``) so
        // we never need to scan an orbit CSR that may not be materialised.
        if (rep_lazy_) {
            return const_cast<SectorOperator*>(this)->isReal()
                && is_hermitian()
                && rep_is_real_;
        }
        return const_cast<SectorOperator*>(this)->isReal()
            && is_hermitian()
            && sector_is_real_();
    }

    // -----------------------------------------------------------------
    // Geometry: advertise device-matvec capability on WITH_CUDA builds so
    // ``ed::select_backend`` picks ``CudaBackend`` for this sector. The
    // operator stays host-resident (the SectorBasis lives on the host);
    // the capability flag is the contract that ``bind_cuda()`` lazily
    // builds a one-shot GPU mirror. Mirrors the legacy
    // ``StreamingSymmetryOperator::SectorView::geometry()`` gate
    // (``ED_GPU_SYMMETRY_MIRROR=0`` forces the CPU route for bisection).
    // -----------------------------------------------------------------
    [[nodiscard]] ed::Geometry geometry() const override {
        ed::Geometry g = ::Operator::geometry();
#ifdef WITH_CUDA
        static const bool kGpuMirrorEnabled = [] {
            const char* e = std::getenv("ED_GPU_SYMMETRY_MIRROR");
            if (e == nullptr) return true;                  // default ON
            if (e[0] == '\0') return true;                  // empty -> ON
            if (e[0] == '0' && e[1] == '\0') return false;  // "0" -> OFF
            return true;                                    // anything else -> ON
        }();
        g.supports_device_matvec = kGpuMirrorEnabled;
#endif
        return g;
    }

    // -----------------------------------------------------------------
    // GPU lane. Returns a MatvecFn that runs the unified symmetry kernel
    // on DEVICE pointers (per the ``bind_cuda()`` contract). Defined
    // out-of-line: the WITH_CUDA definition lives in
    // ``src/symmetry/sector_operator_gpu.cu`` (ed_solvers_gpu); the
    // non-CUDA throwing stub in ``sector_operator_gpu.cpp`` (ed_core).
    // -----------------------------------------------------------------
    [[nodiscard]] MatvecFn bind_cuda() const override;

protected:
    [[nodiscard]] std::unique_ptr<ed::matvec::MatVecBackendBase>
    make_backend_() const override {
        // CPU on-the-fly representative path: for a fixed-Sz symmetry sector
        // the CSR-free RepSectorData (reps + 1/norm + group perms + characters)
        // is all the matvec needs. This NEVER materialises the per-sector orbit
        // CSR (~24 GiB/sector at N=32) -- the group action + projection phase
        // are regenerated arithmetically in the kernel. It is taken ONLY in the
        // lazy regime (``rep_lazy_``: the orbit CSR was estimated too big to
        // build), because the rep kernel is ~100x slower per matvec than the
        // precomputed CSR when the CSR fits. The eager regime keeps the fast
        // CSR backend even though ``make_sector_operator_adopt`` also populated
        // ``rep_data_`` (for GPU). Falls back to CSR when the rep path is
        // disabled (``ED_SYM_REP=0``) or the RepSectorData is unusable (sym-only
        // sectors with varying popcount, where the rep reverse lookup is
        // undefined).
        if (rep_lazy_ && cpu_rep_symmetry_enabled()) {
            const RepSectorData& rd = ensure_rep_data_();
            if (rd.usable()) {
                return ed::matvec::make_cpu_rep_symmetry_backend<
                    DiagonalOneBody, OffDiagonalOneBody,
                    DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
                    ThreeBodyTransformData>(rd);
            }
        }
        // Legacy CSR path. In CSR-free lazy mode the host CSR has not been
        // built yet -- materialise it now (once).
        ensure_sector_basis_();
        return ed::matvec::make_cpu_symmetry_backend<
            DiagonalOneBody, OffDiagonalOneBody,
            DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
            ThreeBodyTransformData>(sector_basis_.policy());
    }

    // Lazily build the CSR-free RepSectorData (GPU rep path source). For the
    // eager factories this is a no-op (rep_data_ already usable or no
    // provider); for the lazy loop it runs ``getRepSectorData`` exactly once,
    // the first time ``bind_cuda`` engages the rep path. Defined here so the
    // CUDA translation unit (sector_operator_gpu.cu) can call it.
    [[nodiscard]] const RepSectorData& ensure_rep_data_() const {
        if (!rep_data_.usable() && rep_provider_) {
            rep_data_ = rep_provider_();
        }
        return rep_data_;
    }

private:
    // Lazily materialise the host orbit CSR (SectorBasis) from the deferred
    // provider. No-op when the basis is already populated (eager factories)
    // or when no provider was supplied.
    void ensure_sector_basis_() const {
        if (sector_basis_.dim() == 0 && csr_provider_) {
            sector_basis_ =
                SectorBasis::adopt(csr_provider_(), rep_group_size_);
        }
    }

    // True iff every orbit coefficient in this sector is real within tol.
    // Matches the legacy ``StreamingSymmetryOperator::isSectorReal``
    // semantics (scan of orbit coefficients), restricted to this sector.
    [[nodiscard]] bool sector_is_real_(double tol = 1e-12) const noexcept {
        for (const auto& st : sector_basis_.sector().basis_states) {
            for (const auto& c : st.orbit_coefficients) {
                if (std::abs(c.imag()) > tol) return false;
            }
        }
        return true;
    }

    // Derive the magnetization quantum number ``n_up`` for the GPU device
    // mirror's dense combinadic rank table. Returns the shared popcount of
    // the orbit representatives iff EVERY basis state in this sector has the
    // same popcount (i.e. this is a fixed-Sz sector, where every full state
    // shares that magnetization) -- in which case the GPU mirror can use the
    // O(1) rank-table lookup instead of the open-addressing hash. Returns -1
    // for a full-Hilbert (sym-only) sector, where popcounts vary and the
    // dense rank table is undefined. Self-validating: requires no external
    // convention, derives purely from the resident orbit data.
    [[nodiscard]] int sector_n_up_() const noexcept {
        const auto& states = sector_basis_.sector().basis_states;
        if (states.empty()) return -1;
        const int n_up = __builtin_popcountll(states.front().orbit_rep);
        for (const auto& st : states) {
            if (__builtin_popcountll(st.orbit_rep) != n_up) return -1;
        }
        return n_up;
    }

    // ``mutable``: both are populated lazily by const matvec entry points
    // (``make_backend_`` / ``bind_cuda``) in CSR-free lazy mode.
    mutable SectorBasis   sector_basis_;
    mutable RepSectorData rep_data_;  // CSR-free on-the-fly rep path source

    // CSR-free lazy mode state (see ``configureRepLazy``).
    bool                            rep_lazy_       = false;
    std::uint64_t                   rep_dim_        = 0;
    std::size_t                     rep_group_size_ = 0;
    bool                            rep_is_real_    = false;
    mutable std::function<RepSectorData()>  rep_provider_;
    mutable std::function<SymmetrySector()> csr_provider_;
};

} // namespace ed::symmetry
