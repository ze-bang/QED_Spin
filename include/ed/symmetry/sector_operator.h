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
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#ifdef WITH_CUDA
#include <cstdlib>
#endif

#include <ed/core/operator.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/sector_gpu_mirror.h>

namespace ed::symmetry {

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
        return static_cast<std::size_t>(sector_basis_.dim());
    }

    [[nodiscard]] const SectorBasis& basis() const noexcept {
        return sector_basis_;
    }

    [[nodiscard]] std::string description() const override {
        return "SectorOperator(sector="
            + std::to_string(sector_basis_.sector().sector_id)
            + ", dim=" + std::to_string(sector_basis_.dim()) + ")";
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
        return ed::matvec::make_cpu_symmetry_backend<
            DiagonalOneBody, OffDiagonalOneBody,
            DiagonalTwoBody, MixedTwoBody, OffDiagonalTwoBody,
            ThreeBodyTransformData>(sector_basis_.policy());
    }

private:
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

    SectorBasis sector_basis_;
};

} // namespace ed::symmetry
