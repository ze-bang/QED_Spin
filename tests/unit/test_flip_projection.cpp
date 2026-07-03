// =============================================================================
// tests/unit/test_flip_projection.cpp
//
// Stage-5b guards of the SymmetryEngine v2 plan
// (docs/architecture/SYMMETRY_V2_DESIGN.md): the in-sector spin-flip
// projection of the half-filling block.
//
// At n_up = N/2 the group extends to G' = G x Z2 (the global flip F
// commutes with every site permutation), each momentum sector k splits
// into (k, +) and (k, -), and:
//
//   1. Burnside: sum of all (k, +/-) dims == C(N, N/2);
//   2. per k: dim(k,+) + dim(k,-) == dim(k) of the unprojected sector;
//   3. spectrum: eig(k,+) UNION eig(k,-) == eig(k)  (machine precision),
//      with the dense blocks assembled through the production rep-lane
//      matvec (i.e. the flip-aware policy / reduced-CSR path end-to-end).
// =============================================================================
#include "common/catch2_harness.h"

#include <ed/core/operator.h>
#include <ed/symmetry/group.h>
#include <ed/symmetry/sector_set.h>

#include <Eigen/Dense>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <memory>
#include <vector>

using Cx = std::complex<double>;

namespace {

std::unique_ptr<Operator> heisenberg_ring(std::uint64_t N, double J) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const Cx J_real(J, 0.0), J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        op->transform_data_.push_back(t);
        t.op_type = 0; t.op_type_2 = 1; t.coefficient = J_half;
        op->transform_data_.push_back(t);
        t.op_type = 1; t.op_type_2 = 0;
        op->transform_data_.push_back(t);
    }
    return op;
}

std::vector<double> dense_eigs(ed::symmetry::SectorOperator& op) {
    const std::size_t d = op.dim();
    Eigen::MatrixXcd H(d, d);
    std::vector<Cx> e(d, Cx(0, 0)), col(d);
    for (std::size_t j = 0; j < d; ++j) {
        std::fill(e.begin(), e.end(), Cx(0, 0));
        e[j] = Cx(1, 0);
        op.apply(e.data(), col.data(), d);
        for (std::size_t i = 0; i < d; ++i) H(i, j) = col[i];
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    std::vector<double> ev(es.eigenvalues().data(),
                           es.eigenvalues().data() + d);
    std::sort(ev.begin(), ev.end());
    return ev;
}

}  // namespace

TEST_CASE("flip projection: (k,+)+(k,-) tiles and reproduces each sector",
          "[spin_flip][flip_projection]") {
    const std::uint64_t N    = 8;
    const int           n_up = 4;
    const SymmetryGroupInfo info = ed::sym::translation_group_1d(
        static_cast<int>(N));
    const std::size_t num_irreps = info.sectors.size();
    REQUIRE(num_irreps == N);

    auto base = heisenberg_ring(N, 1.0);
    auto term_builder = [&base](ed::symmetry::SectorOperator& op) {
        op.transform_data_  = base->transform_data_;
        op.three_body_data_ = base->three_body_data_;
    };

    // Plain (unprojected) half-filling sectors.
    auto plain = ed::symmetry::build_fixed_sz_sector_operators_lazy(
        N, 0.5f, n_up, info, term_builder);
    REQUIRE(plain.size() == num_irreps);  // every k survives on the ring

    // Flip-projected block via the all-Sz builder restricted to n_up = N/2.
    std::vector<std::pair<int, std::size_t>> ids;
    auto flip = ed::symmetry::build_all_sz_sector_operators(
        N, 0.5f, info, term_builder,
        /*n_up_min=*/n_up, /*n_up_max=*/n_up, &ids, /*cache_dir=*/{},
        /*flip_project_half=*/true);
    REQUIRE(flip.size() > num_irreps);  // sectors split

    // Group the flip sectors by spatial k (synthetic index = k + p*num_irreps).
    std::vector<std::vector<ed::symmetry::SectorOperator*>> by_k(num_irreps);
    std::uint64_t total_dim = 0;
    for (std::size_t i = 0; i < flip.size(); ++i) {
        const std::size_t k = ids[i].second % num_irreps;
        by_k[k].push_back(flip[i].get());
        total_dim += flip[i]->dim();
    }

    // Burnside over the flip-extended sector set.
    ed::core::combinadic::BinomialTable binom(static_cast<int>(N));
    REQUIRE(total_dim == binom.at(static_cast<int>(N), n_up));

    for (std::size_t k = 0; k < num_irreps; ++k) {
        // Dim tiling per k.
        std::uint64_t dsum = 0;
        for (auto* op : by_k[k]) dsum += op->dim();
        REQUIRE(dsum == plain[k]->dim());

        // Spectrum union == plain sector spectrum.
        std::vector<double> u;
        for (auto* op : by_k[k]) {
            const auto ev = dense_eigs(*op);
            u.insert(u.end(), ev.begin(), ev.end());
        }
        std::sort(u.begin(), u.end());
        const auto ref = dense_eigs(*plain[k]);
        REQUIRE(u.size() == ref.size());
        for (std::size_t i = 0; i < u.size(); ++i) {
            INFO("k=" << k << " i=" << i);
            REQUIRE(std::abs(u[i] - ref[i]) < 1e-10);
        }
    }
}
