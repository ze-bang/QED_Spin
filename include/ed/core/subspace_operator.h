#pragma once
// =============================================================================
// include/ed/core/subspace_operator.h
//
// SubspaceOperator<BasisPolicy, MemSpace>: the collapsed operator template for
// the basis-RESTRICTED lanes (fixed-Sz sector + symmetry-projected sector).
//
// Operator-collapse Phase 4 (Jun 2026)
// ------------------------------------
// Before this template, the basis axis was expressed as three hand-written
// classes that differed ONLY in their basis:
//
//     Operator                         -- full Hilbert space  (2^N)
//     FixedSzOperator : Operator       -- fixed total-Sz sector
//     ed::symmetry::SectorOperator     -- symmetry-projected sector
//
// The matvec seam was already templated (CpuMatVecBackend<BasisPolicy>,
// CudaMatVecBackend<DevicePolicy>) and the owning producers already existed
// (FullSpaceSubspace / FixedSzSubspace / SectorBasis, each emitting a
// ``policy()`` POD). This template finishes the collapse: it derives from the
// full-Hilbert ``Operator`` (inheriting the canonical term storage + matvec
// backend slot) and adds an OWNED producer member chosen by ``BasisPolicy``
// (via ``SubspaceProducerTraits``) plus the per-lane overrides
// (dim / make_backend_ / bind_cuda / ...).
//
// ``FixedSzOperator`` and ``ed::symmetry::SectorOperator`` are now
// using-aliases over this one template (see ``fixed_sz_operator.h`` /
// ``sector_operator.h``), so the ~80 construction sites, pybind classes, MPI
// wrappers, and DSSF refs compile unchanged.
//
// Why ``Operator`` (full) stays a concrete base rather than becoming
// ``SubspaceOperator<FullBasisPolicy>``: the full-Hilbert lane carries the
// delicate weak/strong GPU vtable-key-function split (operator_gpu.cu/.cpp)
// that lets CPU-only binaries link without ed_solvers_gpu. Keeping it as the
// base (the plan's sanctioned "keep name Operator as the template, alias the
// rest" option) preserves that mechanism byte-for-byte while still collapsing
// the two divergent derived classes into one template.
//
// Include-layering: this header pulls only ``operator.h`` + ``subspace.h``
// (FixedSz producer). The symmetry producer (SectorBasis) is incomplete here;
// its ``SubspaceProducerTraits`` specialization + the symmetry-lane member
// specializations live in ``sector_operator.h`` (which owns the
// symmetry_matvec_backend include), so the streaming->operator include cycle
// stays broken.
// =============================================================================

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <Eigen/Sparse>

#include <ed/core/operator.h>
#include <ed/planner/basis_policy_hook.h>   // prefer_tableless_fixed_sz()
#include <ed/matvec/basis_policy.h>
#include <ed/matvec/matvec_backend.h>
#include <ed/matvec/memory_space.h>
#include <ed/matvec/term_kernels.h>
#include <ed/matvec/term_kernels_assemble.h>
#include <ed/matvec/reduced_symmetry_csr.h>    // build_reduced_symmetry_csr (orbit-walk dense assembly)
#include <ed/symmetry/subspace.h>               // FixedSzSubspace (FixedSz producer)
#include <ed/symmetry/symmetry_sector_data.h>   // SymmetrySector (symmetry forwarders)
#include <ed/symmetry/rep_sector_data.h>        // RepSectorData (symmetry forwarders)

namespace ed {

// ---------------------------------------------------------------------------
// SubspaceProducerTraits: maps a host BasisPolicy to its owning producer type.
// Primary left undefined; specialized per policy. The FixedSz specialization
// lives here (subspace.h is a light include); the Symmetry specialization
// (Producer = ed::symmetry::SectorBasis) lives in sector_operator.h, where the
// SectorBasis type is complete.
// ---------------------------------------------------------------------------
template <class BasisPolicy>
struct SubspaceProducerTraits;  // primary: undefined

template <>
struct SubspaceProducerTraits<ed::matvec::basis::FixedSzBasisPolicy> {
    using type = ed::symmetry::FixedSzSubspace;
};

// ---------------------------------------------------------------------------
// SubspaceOperator<BasisPolicy, MS>
// ---------------------------------------------------------------------------
template <class BasisPolicy,
          ed::matvec::MemorySpace MS = ed::matvec::MemorySpace::Host>
class SubspaceOperator : public Operator {
public:
    using Producer = typename SubspaceProducerTraits<BasisPolicy>::type;

    static constexpr bool is_fixed_sz_ =
        std::is_same_v<BasisPolicy, ed::matvec::basis::FixedSzBasisPolicy>;

    // -----------------------------------------------------------------
    // Constructors. Only the constructor matching the alias's producer is
    // ever instantiated (a class-template constructor body is instantiated
    // on first use); the other parses but is never compiled.
    // -----------------------------------------------------------------

    /// Fixed-Sz lane: build the sorted basis + Lin (1990) O(1) index table
    /// for the ``n_up`` magnetisation sector -- or the tableless combinadic
    /// twin when the list would bust the byte budget (dimension-aware
    /// default; env/planner overrides win, see basis_policy_hook.h).
    SubspaceOperator(std::uint64_t n_bits, float spin_l, std::int64_t n_up)
        : Operator(n_bits, spin_l),
          producer_(ed::planner::prefer_tableless_fixed_sz(n_bits, n_up)
                        ? Producer::build_tableless(n_bits, n_up)
                        : Producer::build(n_bits, n_up)) {
        std::cout << "Fixed Sz basis: n_bits=" << n_bits
                  << ", n_up=" << n_up
                  << ", dimension=" << producer_.dim()
                  << (producer_.is_tableless() ? " (tableless combinadic)" : "")
                  << std::endl;
    }

    /// Symmetry lane: adopt an owning producer (SectorBasis) carrying this
    /// one sector's orbit data (or a CSR-free lazy-rep configuration).
    SubspaceOperator(std::uint64_t n_bits, float spin_l, Producer producer)
        : Operator(n_bits, spin_l), producer_(std::move(producer)) {}

    // The backend holds a basis-policy POD that points into ``producer_``.
    // Member destruction order destroys ``producer_`` (a derived member)
    // BEFORE the base ``backend_`` / ``cuda_backend_``; reset them here so no
    // dangling backend can observe a destroyed producer.
    ~SubspaceOperator() override {
        this->backend_.reset();
        this->cuda_backend_.reset();
    }

    // Copy/move semantics are derived automatically from the producer:
    // FixedSzSubspace is copyable (with pointer re-homing), so the FixedSz
    // alias is copyable; SectorBasis is move-only-and-here-non-copyable, so
    // the Symmetry alias is non-copyable & (via the user-declared dtor)
    // non-movable -- exactly matching the legacy FixedSzOperator /
    // SectorOperator contracts.

    // -----------------------------------------------------------------
    // Basis introspection / dimension.
    // -----------------------------------------------------------------
    [[nodiscard]] const Producer& producer() const noexcept { return producer_; }

    [[nodiscard]] std::size_t dim() const override {
        return static_cast<std::size_t>(producer_.dim());
    }

    // Fast dense assembly. Fixed-Sz: map column index -> state via the (combinadic
    // or table) basis policy, enumerate H|state> with for_each_connected_state,
    // and map each connected state back to a row index -- O(nnz), reentrant
    // (pure term reads + pure basis lookups), so parallel over columns. The
    // symmetry lanes return false (their reduced matrix elements carry SAB /
    // projection coefficients, not bare <s'|H|s>), so the caller falls back to
    // the matvec column build.
    [[nodiscard]] bool try_build_dense_columns(Complex* dense,
                                               std::size_t N) const override {
        if constexpr (is_fixed_sz_) {
            if (static_cast<std::size_t>(producer_.dim()) != N) return false;
            // Reuse the proven, matvec-consistent term assembler (the same kernel
            // buildFixedSzMatrix uses): it emits (row, col, <row|H|col>) triplets
            // in the projected fixed-Sz basis -- O(nnz), pure term reads. Scatter
            // them into the column-major dense buffer (duplicates accumulate).
            this->commitPendingTransforms();
            ed::matvec::basis::FixedSzBasisPolicy basis_pol = producer_.policy();
            std::vector<Eigen::Triplet<Complex>> triplets;
            ed::matvec::kernel::emit_term_triplets<
                ed::matvec::basis::FixedSzBasisPolicy, Complex>(
                    basis_pol, static_cast<double>(this->getSpin()),
                    this->terms_.diag_one_body, this->terms_.offdiag_one_body,
                    this->terms_.diag_two_body, this->terms_.mixed_two_body,
                    this->terms_.offdiag_two_body, this->terms_.three_body,
                    triplets);
            for (const auto& t : triplets)
                dense[static_cast<std::size_t>(t.row())
                      + static_cast<std::size_t>(t.col()) * N] += t.value();
            return true;
        } else if constexpr (Producer::needs_orbit_walk
                             && Producer::has_coeff_modifier) {
            // Orbit-walk symmetry lane (abelian spatial group, possibly without
            // fixed Sz). The reduced matrix element is NOT the bare <s'|H|s> but
            // the projection-weighted conj(base_contrib * coeff_modifier); we
            // enumerate each reduced row ONCE via ``rep_symmetry_row_for_each``
            // (the same per-row walk ``build_reduced_symmetry_csr`` and the
            // gather matvec share) and write the element straight into the
            // column-major dense buffer. This replaces the O(dim)-matvec column
            // build -- where each matvec re-walks every orbit and recomputes the
            // projection -- with a single O(|G|*nnz) assembly pass. The emitted
            // value is byte-for-byte what the gather kernel accumulates (pinned
            // by the "lazy rep-walk dense-assembly lane == full reference" case in
            // tests/integration/test_make_sector_operators_e2e.cpp: the FullDiag
            // sector union equals the full-Hilbert dense spectrum).
            if (static_cast<std::size_t>(producer_.dim()) != N) return false;
            this->commitPendingTransforms();
            // Rep-LAZY sectors -- flip/parity rep-only ones (csr_available ==
            // false) AND ordinary fixed-Sz/full-space sectors above the 64 MiB
            // budget (csr_available == true but the orbit CSR is deferred) --
            // assemble the reduced matrix DIRECTLY from the CSR-free rep policy
            // via build_reduced_symmetry_csr_rep: O(|G|*nnz), PARALLEL over
            // rows, and it NEVER materialises the orbit CSR.
            //
            // This is the full_spectrum "dense block" fix. Previously an
            // over-budget sector either (a) fell to the caller's O(dim)-matvec
            // column build (for j: col_j = H*e_j) -- serial outer loop, tiny
            // per-column payload, so OpenMP fork/join per column dominated and
            // the build collapsed to ~1 core -- or (b) paid ensureHostCsr()'s
            // multi-GB orbit-CSR materialisation storm followed by a SERIAL
            // orbit-walk assembly. The rep-CSR path avoids both. A[r, j] is
            // byte-for-byte the column build's col_j[r] (both are the rep-walk
            // gather; pinned by test_reduced_symmetry_csr.cpp and the
            // lazy-dense e2e case). Flip/parity sectors work too -- the rep
            // policy's apply_perm carries the flip masks.
            if (producer_.rep_lazy() || !producer_.csr_available()) {
                const auto& rd = producer_.ensureRepData();
                if (rd.usable()) {
                    const auto rep_pol = rd.make_policy();
                    const auto csr = ed::matvec::build_reduced_symmetry_csr_rep<
                        decltype(rep_pol), Complex>(
                            rep_pol, static_cast<double>(this->getSpin()),
                            this->terms_.diag_one_body, this->terms_.offdiag_one_body,
                            this->terms_.diag_two_body, this->terms_.mixed_two_body,
                            this->terms_.offdiag_two_body, this->terms_.three_body);
                    for (std::uint64_t r = 0; r < N; ++r)
                        for (std::uint64_t e = csr.row_ptr[r]; e < csr.row_ptr[r + 1]; ++e)
                            dense[static_cast<std::size_t>(r)
                                  + static_cast<std::size_t>(csr.col_idx[e]) * N]
                                += csr.val[e];
                    return true;
                }
                // Rep-only but rep data unusable (shouldn't happen): decline to
                // the safe matvec column build rather than risk ensureHostCsr on
                // a throwing provider.
                if (!producer_.csr_available()) return false;
            }
            // (Stage 11c-2b: the eager orbit-walk dense assembler that lived
            // here was deleted -- every production SectorBasis is rep-lazy,
            // so the rep-CSR densify above is the ONE assembly path and this
            // tail was unreachable. Declining falls to the caller's matvec
            // column build, which rides the same rep kernel.)
            return false;
        } else {
            (void)dense; (void)N;
            return false;
        }
    }

    [[nodiscard]] std::string description() const override {
        if constexpr (is_fixed_sz_) {
            return "FixedSzOperator(n_bits=" + std::to_string(this->getNumBits())
                + ", n_up=" + std::to_string(producer_.n_up())
                + ", dim=" + std::to_string(producer_.dim()) + ")";
        } else {
            return "SectorOperator(dim=" + std::to_string(dim()) + ")";
        }
    }

    // Extend the base cache-invalidation hook to clear the projected
    // sparse-matrix cache (used only by the FixedSz getFixedSzMatrix path;
    // harmless no-op for the symmetry lane).
    void invalidateMatrixCaches() override {
        Operator::invalidateMatrixCaches();
        fixed_sz_matrix_built_ = false;
    }

    // -----------------------------------------------------------------
    // Matvec entry points. Same structure as Operator::apply but the size
    // check is against the SUBSPACE dim (not 2^N); the backend supplied by
    // make_backend_() (below) reads/writes the restricted basis.
    // -----------------------------------------------------------------
    void apply(const Complex* in, Complex* out, std::size_t size) const override {
        if (size != dim()) {
            throw std::invalid_argument(
                "SubspaceOperator::apply: input/output vector size mismatch");
        }
        this->ensure_backend_();
        const auto tv = this->term_view_();  // rebuilds SoA cache if stale
        this->backend_->apply_complex(&tv, in, out, size);
    }

    void apply_real(const double* in, double* out, std::size_t size) const override {
        if (size != dim()) {
            throw std::invalid_argument(
                "SubspaceOperator::apply_real: input/output vector size mismatch");
        }
        this->ensure_backend_();
        const auto tv = this->term_view_();  // rebuilds SoA cache if stale
        this->backend_->apply_real(&tv, in, out, size);
    }

    // -----------------------------------------------------------------
    // Real-Hermitian fast-path eligibility.
    //   * FixedSz: real iff the couplings are real (no momentum phase).
    //   * Symmetry: additionally requires the sector character to carry no
    //     genuine complex phase (k=0 / real-character irreps). In CSR-free
    //     lazy mode the per-sector character is precomputed (rep_is_real()).
    // -----------------------------------------------------------------
    [[nodiscard]] bool is_real_hermitian() const noexcept override {
        const bool base_real =
            const_cast<SubspaceOperator*>(this)->isReal() && is_hermitian();
        if constexpr (is_fixed_sz_) {
            return base_real;
        } else {
            if (producer_.rep_lazy()) {
                return base_real && producer_.rep_is_real();
            }
            for (const auto& st : producer_.sector().basis_states) {
                for (const auto& c : st.orbit_coefficients) {
                    if (std::abs(c.imag()) > 1e-12) return false;
                }
            }
            return base_real;
        }
    }

    // -----------------------------------------------------------------
    // Geometry: advertise device-matvec capability on WITH_CUDA builds so
    // ed::select_backend picks the CudaBackend lane.
    //   * FixedSz: same gate as the full-Hilbert Operator
    //     (ED_GPU_OPERATOR_MIRROR, via Operator::geometry()).
    //   * Symmetry: the symmetry-specific ED_GPU_SYMMETRY_MIRROR gate.
    // -----------------------------------------------------------------
    [[nodiscard]] ed::Geometry geometry() const override {
        ed::Geometry g = Operator::geometry();
        if constexpr (!is_fixed_sz_) {
#ifdef WITH_CUDA
            static const bool kGpuMirrorEnabled = [] {
                const char* e = std::getenv("ED_GPU_SYMMETRY_MIRROR");
                if (e == nullptr) return true;                  // default ON
                if (e[0] == '\0') return true;                  // empty -> ON
                if (e[0] == '0' && e[1] == '\0') return false;  // "0" -> OFF
                return true;                                    // else -> ON
            }();
            g.supports_device_matvec = kGpuMirrorEnabled;
#endif
        }
        return g;
    }

    // -----------------------------------------------------------------
    // GPU lane. Routes through the per-policy ``bind_cuda_impl_`` (explicit
    // member specializations: strong device build in ed_solvers_gpu .cu, weak
    // CPU fallback in ed_core .cpp for the FixedSz lane). Under !WITH_CUDA the
    // FixedSz lane degrades to bind_cpu() and the Symmetry lane throws --
    // exactly matching the legacy classes.
    // -----------------------------------------------------------------
    [[nodiscard]] MatvecFn bind_cuda() const override {
#ifdef WITH_CUDA
        return bind_cuda_impl_();
#else
        if constexpr (is_fixed_sz_) {
            return bind_cpu();
        } else {
            throw std::logic_error(
                "ed::symmetry::SectorOperator::bind_cuda: built without "
                "WITH_CUDA. Rebuild with -DWITH_CUDA=ON to enable the GPU "
                "symmetry mirror, or route through CpuBackend (device='cpu').");
        }
#endif
    }

    // =================================================================
    // FixedSz lane public API (instantiated only when called on the
    // FixedSzOperator alias).
    // =================================================================
    [[nodiscard]] std::uint64_t getFixedSzDim() const { return producer_.dim(); }
    [[nodiscard]] std::uint64_t getFullDim() const { return 1ULL << this->getNumBits(); }
    [[nodiscard]] std::int64_t  getNUp() const { return producer_.n_up(); }

    [[nodiscard]] const std::vector<std::uint64_t>& getBasisStates() const {
        return producer_.basis_states();
    }
    [[nodiscard]] const LinIndexTable& lin_index_table() const noexcept {
        return producer_.lin_index();
    }

    /// Non-owning ``FixedSzSubspace`` view over this operator's tables.
    [[nodiscard]] ed::symmetry::FixedSzSubspace subspace() const noexcept {
        return ed::symmetry::FixedSzSubspace::view(
            this->getNumBits(), producer_.n_up(),
            producer_.basis_states(), producer_.lin_index());
    }

    /// O(1) Lin (1990) state -> basis-index lookup; -1 if not in this sector.
    [[nodiscard]] std::int64_t lookupState(std::uint64_t state) const {
        return producer_.lin_index().lookup(state);
    }

    std::vector<Complex> projectToReduced(const std::vector<Complex>& full_vec) const {
        return projectToFixedSz(full_vec);
    }

    std::vector<Complex> projectToFixedSz(const std::vector<Complex>& full_vec) const {
        const std::uint64_t full_dim = 1ULL << this->getNumBits();
        if (full_vec.size() != static_cast<std::size_t>(full_dim)) {
            throw std::invalid_argument(
                "projectToFixedSz: input vector size mismatch with full dimension");
        }
        // Mode-agnostic: producer_.state_of(i) is combinadic in tableless mode.
        const std::uint64_t d = producer_.dim();
        std::vector<Complex> fixed_sz_vec(d, Complex(0.0, 0.0));
        for (std::uint64_t i = 0; i < d; ++i) {
            fixed_sz_vec[i] = full_vec[producer_.state_of(i)];
        }
        return fixed_sz_vec;
    }

    std::vector<Complex> embedToFull(const std::vector<Complex>& fixed_sz_vec) const {
        const std::uint64_t d = producer_.dim();
        if (fixed_sz_vec.size() != static_cast<std::size_t>(d)) {
            throw std::invalid_argument(
                "embedToFull: input vector size mismatch with fixed-Sz dimension");
        }
        const std::uint64_t full_dim = 1ULL << this->getNumBits();
        std::vector<Complex> full_vec(full_dim, Complex(0.0, 0.0));
        for (std::uint64_t i = 0; i < d; ++i) {
            full_vec[producer_.state_of(i)] = fixed_sz_vec[i];
        }
        return full_vec;
    }

    /// Build the projected-basis sparse matrix from the canonical AoS term
    /// list (O(dim * num_terms)). Lazy; reused until invalidateMatrixCaches().
    void buildFixedSzMatrix() const {
        if (fixed_sz_matrix_built_) return;
        this->commitPendingTransforms();
        const std::uint64_t d = producer_.dim();
        fixed_sz_matrix_.resize(d, d);

        std::vector<Eigen::Triplet<Complex>> triplets;
        if (!this->transform_data_.empty() || !this->three_body_data_.empty()) {
            // producer_.policy() is combinadic in tableless mode.
            ed::matvec::basis::FixedSzBasisPolicy basis_pol = producer_.policy();
            ed::matvec::kernel::emit_term_triplets<
                ed::matvec::basis::FixedSzBasisPolicy, Complex>(
                    basis_pol, static_cast<double>(this->getSpin()),
                    this->terms_.diag_one_body, this->terms_.offdiag_one_body,
                    this->terms_.diag_two_body, this->terms_.mixed_two_body,
                    this->terms_.offdiag_two_body, this->terms_.three_body,
                    triplets);
        }
        fixed_sz_matrix_.setFromTriplets(triplets.begin(), triplets.end());
        fixed_sz_matrix_built_ = true;
    }

    Eigen::SparseMatrix<Complex> getFixedSzMatrix() const {
        buildFixedSzMatrix();
        return fixed_sz_matrix_;
    }

    // =================================================================
    // Symmetry lane public API (instantiated only when called on the
    // ed::symmetry::SectorOperator alias). Thin forwarders to the owned
    // SectorBasis producer, which holds the orbit data + lazy-rep state.
    // =================================================================
    [[nodiscard]] const Producer& basis() const noexcept { return producer_; }

    /// Force-materialise the host orbit CSR (no-op for an eager producer)
    /// and return the producer. Used by host-side consumers (e.g.
    /// CrossSectorOrbitObservable) that need the full orbit data.
    [[nodiscard]] const Producer& materialized_basis() const {
        producer_.ensureHostCsr();
        return producer_;
    }

    void configureRepLazy(std::uint64_t                   sector_dim,
                          std::size_t                     group_size,
                          bool                            is_real,
                          std::function<ed::symmetry::RepSectorData()>  rep_provider,
                          std::function<SymmetrySector()> csr_provider,
                          bool                            csr_available = true) {
        producer_.configureRepLazy(sector_dim, group_size, is_real,
                                   std::move(rep_provider),
                                   std::move(csr_provider), csr_available);
    }

    [[nodiscard]] bool csr_available() const noexcept {
        return producer_.csr_available();
    }

    [[nodiscard]] bool rep_lazy() const noexcept { return producer_.rep_lazy(); }
    [[nodiscard]] bool host_csr_materialized() const noexcept {
        return producer_.host_csr_materialized();
    }

    void setRepSectorData(ed::symmetry::RepSectorData rep) noexcept {
        producer_.setRepData(std::move(rep));
    }
    [[nodiscard]] const ed::symmetry::RepSectorData& repSectorData() const noexcept {
        return producer_.repData();
    }

protected:
    // Per-lane backend factory. Defined via explicit member specialization in
    // fixed_sz_operator.h (FixedSz) and sector_operator.h (Symmetry); the
    // primary template intentionally leaves it undefined (only the two host
    // lanes are ever instantiated).
    [[nodiscard]] std::unique_ptr<ed::matvec::MatVecBackendBase>
    make_backend_() const override;

#ifdef WITH_CUDA
    // Per-lane device matvec build. Explicit member specializations:
    //   * FixedSz: strong in operator_gpu.cu (ed_solvers_gpu), weak fallback
    //     in operator_gpu.cpp (ed_core) so CPU-only binaries still link.
    //   * Symmetry: strong in sector_operator_gpu.cu (ed_solvers_gpu); no weak
    //     fallback (a CPU-only WITH_CUDA binary that constructs a
    //     SectorOperator must link ed_solvers_gpu, matching the legacy
    //     out-of-line bind_cuda key-function behaviour).
    [[nodiscard]] MatvecFn bind_cuda_impl_() const;
#endif

private:
    mutable Producer producer_;

    // Projected-basis sparse-matrix cache (FixedSz getFixedSzMatrix path).
    mutable Eigen::SparseMatrix<Complex> fixed_sz_matrix_{};
    mutable bool                         fixed_sz_matrix_built_ = false;
};

}  // namespace ed
