// =============================================================================
// test_chain_extensibility
//
// Phase R5 of the "Orthogonal symmetry composition" plan (May 2026).
//
// ABI-smoke test for the future-axis seams in
// ``include/ed/symmetry/projector.h``:
//
//   * ``InternalZ2Projector``     -- placeholder for global spin-flip
//                                     / particle-hole / sublattice-Z_2
//                                     symmetries.
//   * ``AntiunitaryProjector``    -- placeholder for time-reversal-like
//                                     symmetries.
//   * ``ProjectorChain``          -- ordered list combining any of the
//                                     above with ``SpatialProjector``.
//
// We do NOT exercise the orbit/character composition with these
// placeholders here (their implementations are explicitly out of scope
// for the refactor wave). We just compile-check that:
//
//   * the placeholder types satisfy the Projector duck-type (size /
//     apply / character / quantum_numbers / sector_count /
//     is_antiunitary / preserves_sz / name);
//   * they can be pushed into a ``ProjectorChain`` heterogeneously and
//     the chain reports a coherent total group order;
//   * the ``compute_trivial_orbit`` helper (empty-chain path) returns
//     a single-element orbit of coefficient 1 / norm_sq 1.
//
// When the real spin-flip / time-reversal implementations land, the
// test file alongside them will exercise the orbit math; this test
// only guarantees that the public surface is stable enough for those
// follow-ups to drop in without revising the chain header.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/fixed_sz_operator.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/projector_chain.h>
#include <ed/symmetry/subspace.h>

#include <complex>
#include <cstdint>
#include <vector>

using namespace ed_tests;

using Complex = std::complex<double>;

TEST_CASE("orthogonal composition: InternalZ2Projector satisfies the Projector duck-type",
          "[symmetry][projector_chain][future_axes][abi_smoke]")
{
    constexpr std::uint64_t N = 6;
    const std::uint64_t mask = (1ULL << N) - 1;

    ed::symmetry::InternalZ2Projector even(mask, /*even=*/true);
    ed::symmetry::InternalZ2Projector odd(mask, /*even=*/false);

    REQUIRE(even.size() == 2);
    REQUIRE(even.sector_count() == 2);
    REQUIRE_FALSE(decltype(even)::is_antiunitary);

    // g=0 is identity; g=1 flips every bit.
    REQUIRE(even.apply(0b010101ULL, 0) == 0b010101ULL);
    REQUIRE(even.apply(0b010101ULL, 1) == (0b010101ULL ^ mask));

    // Character on (g=0) is +1 for both sectors; (g=1) is +1 / -1.
    REQUIRE(std::abs(even.character(0, 0) - Complex(1.0, 0.0)) < 1e-15);
    REQUIRE(std::abs(even.character(0, 1) - Complex(1.0, 0.0)) < 1e-15);
    REQUIRE(std::abs(even.character(1, 0) - Complex(+1.0, 0.0)) < 1e-15);
    REQUIRE(std::abs(even.character(1, 1) - Complex(-1.0, 0.0)) < 1e-15);

    // Quantum-number labels: {+1} for sector 0, {-1} for sector 1.
    REQUIRE(even.quantum_numbers(0) == std::vector<int>{+1});
    REQUIRE(even.quantum_numbers(1) == std::vector<int>{-1});

    (void)odd;  // suppress unused warning when NDEBUG strips REQUIREs.
}

TEST_CASE("orthogonal composition: AntiunitaryProjector advertises is_antiunitary=true",
          "[symmetry][projector_chain][future_axes][abi_smoke]")
{
    ed::symmetry::AntiunitaryProjector T;

    REQUIRE(T.size() == 2);
    REQUIRE(T.sector_count() == 2);
    REQUIRE(decltype(T)::is_antiunitary);
    REQUIRE(decltype(T)::preserves_sz);

    // Identity element does nothing; the placeholder treats g=1 as a
    // no-op too (real time-reversal lands in a follow-up wave).
    REQUIRE(T.apply(0xCAFEULL, 0) == 0xCAFEULL);
    REQUIRE(T.apply(0xCAFEULL, 1) == 0xCAFEULL);

    REQUIRE(std::abs(T.character(0, 0) - Complex(1.0, 0.0)) < 1e-15);
    REQUIRE(std::abs(T.character(1, 0) - Complex(+1.0, 0.0)) < 1e-15);
    REQUIRE(std::abs(T.character(1, 1) - Complex(-1.0, 0.0)) < 1e-15);
}

TEST_CASE("orthogonal composition: ProjectorChain composes heterogeneous projectors",
          "[symmetry][projector_chain][future_axes][abi_smoke]")
{
    constexpr std::uint64_t N = 6;
    const std::uint64_t mask = (1ULL << N) - 1;

    // Build a minimal SymmetryGroupInfo carrying just the identity, so
    // ``SpatialProjector`` answers ``size() == 1``. We do NOT load real
    // symmetry data here -- the test only inspects the chain wiring.
    SymmetryGroupInfo info;
    info.max_clique.push_back(std::vector<int>(N, 0));
    for (std::uint64_t i = 0; i < N; ++i) info.max_clique[0][i] = static_cast<int>(i);
    info.power_representation.push_back({});
    SectorMetadata sm;
    sm.sector_id = 0;
    sm.quantum_numbers.clear();
    sm.phase_factors.clear();
    info.sectors.push_back(sm);

    ed::symmetry::SpatialProjector       spatial(info);
    ed::symmetry::InternalZ2Projector    z2(mask);
    ed::symmetry::AntiunitaryProjector   T;

    ed::symmetry::ProjectorChain chain;
    REQUIRE(chain.empty());
    REQUIRE(chain.total_order() == 1);

    chain.push_back(spatial);     // |G| = 1 (identity-only fixture)
    chain.push_back(z2);          // |Z_2| = 2
    chain.push_back(T);           // |T|   = 2 (placeholder)

    REQUIRE(chain.size() == 3);
    REQUIRE_FALSE(chain.empty());
    // Total order = 1 * 2 * 2 = 4 (Cartesian product of component groups).
    REQUIRE(chain.total_order() == 4);
}

TEST_CASE("orthogonal composition: compute_trivial_orbit returns a unit orbit",
          "[symmetry][projector_chain][future_axes][abi_smoke]")
{
    const std::uint64_t basis = 0b101010ULL;
    const ed::symmetry::FullSpaceSubspace full(6);

    std::vector<std::uint64_t> el;
    std::vector<Complex>       co;
    double                     ns = 0.0;

    ed::symmetry::compute_trivial_orbit(full, basis, el, co, ns);

    REQUIRE(el.size() == 1);
    REQUIRE(el[0] == basis);
    REQUIRE(std::abs(co[0] - Complex(1.0, 0.0)) < 1e-15);
    REQUIRE(std::abs(ns - 1.0) < 1e-15);

    // FixedSz subspace that excludes the basis -> empty orbit, norm 0.
    auto sz = ed::symmetry::FixedSzSubspace::build(6, /*n_up=*/2);
    if (sz.index_of(basis) < 0) {
        ed::symmetry::compute_trivial_orbit(sz, basis, el, co, ns);
        REQUIRE(el.empty());
        REQUIRE(co.empty());
        REQUIRE(std::abs(ns) < 1e-15);
    }
}

TEST_CASE("orthogonal composition: subspace() view round-trips through FixedSzOperator",
          "[symmetry][projector_chain][future_axes][abi_smoke]")
{
    // FixedSzOperator -> ed::symmetry::FixedSzSubspace view -> the
    // ``ed::matvec::basis::FixedSzBasisPolicy`` POD view consumed by
    // the matvec kernels. Confirms the chain of references stays
    // valid: the operator owns the storage, the Subspace observes it,
    // and the kernel-facing policy is derived from the Subspace.
    FixedSzOperator op(6, 0.5f, 3);
    auto sub = op.subspace();
    REQUIRE(sub.n_bits() == 6);
    REQUIRE(sub.n_up()   == 3);
    REQUIRE(sub.dim()    == op.getFixedSzDim());

    auto policy = sub.policy();
    REQUIRE(policy.dim() == sub.dim());
    REQUIRE(policy.state_of(0) == sub.state_of(0));
    REQUIRE(policy.index_of(sub.state_of(0)) == 0);
}
