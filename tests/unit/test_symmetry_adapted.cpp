// =============================================================================
// tests/unit/test_symmetry_adapted.cpp
//
// End-to-end NON-ABELIAN symmetry-adapted ED: 4-site Heisenberg square with its
// full point group C4v / D4 (order 8, includes a 2-D irrep). Build the
// symmetry-adapted basis for every irrep, project H into each Γ-block, and
// recombine the eigenvalues (each ×d_Γ). Assert:
//   * completeness  Σ_Γ d_Γ · (#SAB vectors of Γ) == 2^N,
//   * the recombined spectrum matches brute-force full diagonalization to 1e-9,
//   * the 2-D irrep block actually appears (genuinely non-abelian reduction),
//     i.e. blocks shrink by the full |G|, not just an abelian subgroup.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/basis_utils.h>          // applyPermutation
#include <ed/symmetry/group.h>
#include <ed/symmetry/irreps.h>
#include <ed/symmetry/symmetry_adapted.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <complex>
#include <functional>
#include <vector>

using ed::sym::generate_group;
using ed::sym::Permutation;
using ed::symmetry::build_sab_partition0;
using ed::symmetry::decompose_irreps;
using ed::symmetry::symmetry_adapted_spectrum;
using Complex = std::complex<double>;

namespace {

// Dense Heisenberg H = Σ_bonds S_i·S_j on the periodic 4-site chain (= square).
Eigen::MatrixXcd heisenberg_square(int N) {
    const std::uint64_t dim = std::uint64_t{1} << N;
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::uint64_t b = 0; b < dim; ++b) {
        for (int i = 0; i < N; ++i) {
            const int j = (i + 1) % N;
            const double si = ((b >> i) & 1) ? 0.5 : -0.5;
            const double sj = ((b >> j) & 1) ? 0.5 : -0.5;
            H(b, b) += si * sj;                                   // Sz Sz
            if (((b >> i) & 1) != ((b >> j) & 1)) {               // (S+S- + S-S+)/2
                const std::uint64_t bp = b ^ ((1ull << i) | (1ull << j));
                H(bp, b) += 0.5;
            }
        }
    }
    return H;
}

}  // namespace

TEST_CASE("non-abelian SAB: C4v Heisenberg square == brute force (with d_Γ degeneracy)",
          "[symmetry_adapted][nonabelian]") {
    const int N = 4;
    const Permutation r{1, 2, 3, 0};   // 4-fold rotation
    const Permutation s{0, 3, 2, 1};   // reflection
    auto Gp = generate_group({r, s});
    REQUIRE(Gp.size() == 8);

    auto gi = decompose_irreps(Gp, N);
    REQUIRE_FALSE(gi.is_abelian());    // exercises the 2-D irrep path

    const Eigen::MatrixXcd H = heisenberg_square(N);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ref(H);
    std::vector<double> brute(ref.eigenvalues().data(),
                              ref.eigenvalues().data() + ref.eigenvalues().size());

    std::vector<double> recombined;
    long long completeness = 0;
    bool saw_2d_block = false;

    for (std::size_t g = 0; g < gi.irreps.size(); ++g) {
        const int d = gi.irreps[g].dim;
        const auto sab = build_sab_partition0(gi, Gp, static_cast<int>(g), N);
        if (sab.empty()) continue;
        const int nb = static_cast<int>(sab.size());
        completeness += static_cast<long long>(d) * nb;
        if (d == 2) saw_2d_block = true;

        // Φ: (2^N × nb) orthonormal columns = the partner-0 SAB vectors.
        Eigen::MatrixXcd Phi = Eigen::MatrixXcd::Zero(std::uint64_t{1} << N, nb);
        for (int j = 0; j < nb; ++j)
            for (std::size_t k = 0; k < sab[static_cast<std::size_t>(j)].states.size(); ++k)
                Phi(sab[static_cast<std::size_t>(j)].states[k], j) =
                    sab[static_cast<std::size_t>(j)].coeffs[k];

        const Eigen::MatrixXcd Hg = Phi.adjoint() * H * Phi;        // Γ-block
        // Hermiticity sanity (H commutes with the group).
        REQUIRE((Hg - Hg.adjoint()).norm() < 1e-9);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(Hg);
        for (int e = 0; e < es.eigenvalues().size(); ++e)
            for (int rep = 0; rep < d; ++rep)                       // physical d_Γ degeneracy
                recombined.push_back(es.eigenvalues()(e));
    }

    REQUIRE(saw_2d_block);
    REQUIRE(completeness == (1LL << N));                            // completeness

    std::sort(recombined.begin(), recombined.end());
    REQUIRE(recombined.size() == brute.size());
    for (std::size_t i = 0; i < brute.size(); ++i)
        REQUIRE(std::abs(recombined[i] - brute[i]) < 1e-9);        // spectrum match
}

TEST_CASE("non-abelian SAB: symmetry_adapted_spectrum on D6 ring (matvec API)",
          "[symmetry_adapted][nonabelian][matvec]") {
    // 6-site periodic Heisenberg ring with full dihedral group D6 (order 12,
    // genuinely non-abelian: TWO 2-D irreps; Σ d² = 4·1 + 2·4 = 12).
    const int N = 6;
    const Permutation t{1, 2, 3, 4, 5, 0};   // translation
    const Permutation s{0, 5, 4, 3, 2, 1};   // reflection
    auto Gp = generate_group({t, s});
    REQUIRE(Gp.size() == 12);

    auto gi = decompose_irreps(Gp, N);
    REQUIRE(gi.num_classes == 6);            // D6 has 6 classes / 6 irreps
    int n2d = 0; for (const auto& ir : gi.irreps) if (ir.dim == 2) ++n2d;
    REQUIRE(n2d == 2);

    const Eigen::MatrixXcd H = heisenberg_square(N);   // periodic ring of N sites
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ref(H);

    // Drive the production function through a matvec closure over the dense H.
    auto H_mv = [&H](const Complex* in, Complex* out, std::uint64_t dim) {
        Eigen::Map<const Eigen::VectorXcd> v(in, dim);
        Eigen::Map<Eigen::VectorXcd> o(out, dim);
        o.noalias() = H * v;
    };
    auto spec = symmetry_adapted_spectrum(H_mv, gi, Gp, N);

    // Completeness: blocks (weighted by d_Γ) span the full space.
    long long total = 0;
    for (std::size_t i = 0; i < spec.block_size.size(); ++i)
        total += static_cast<long long>(spec.block_irrep_dim[i]) * spec.block_size[i];
    REQUIRE(total == (1LL << N));
    REQUIRE(spec.eigenvalues.size() == static_cast<std::size_t>(1LL << N));

    for (int i = 0; i < ref.eigenvalues().size(); ++i)
        REQUIRE(std::abs(spec.eigenvalues[static_cast<std::size_t>(i)] - ref.eigenvalues()(i)) < 1e-9);
}

TEST_CASE("non-abelian SAB: on-the-fly term builder == brute force (D6 ring)",
          "[symmetry_adapted][nonabelian][terms]") {
    // Exercises the at-scale path: H_Γ built by applying H term-by-term over the
    // orbit support (no 2^N vector). Here the 'connect' enumerator is synthesised
    // from the dense H so we can check it against brute force.
    const int N = 6;
    const Permutation t{1, 2, 3, 4, 5, 0};
    const Permutation s{0, 5, 4, 3, 2, 1};
    auto Gp = generate_group({t, s});
    auto gi = decompose_irreps(Gp, N);

    const Eigen::MatrixXcd H = heisenberg_square(N);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ref(H);

    ed::symmetry::ConnectFn connect =
        [&H](std::uint64_t st, const std::function<void(std::uint64_t, Complex)>& emit) {
            for (int sp = 0; sp < H.rows(); ++sp) {
                const Complex h = H(sp, static_cast<Eigen::Index>(st));
                if (std::abs(h) > 1e-15) emit(static_cast<std::uint64_t>(sp), h);
            }
        };

    auto spec = ed::symmetry::symmetry_adapted_spectrum_terms(connect, gi, Gp, N);
    REQUIRE(spec.eigenvalues.size() == static_cast<std::size_t>(1LL << N));
    for (int i = 0; i < ref.eigenvalues().size(); ++i)
        REQUIRE(std::abs(spec.eigenvalues[static_cast<std::size_t>(i)] - ref.eigenvalues()(i)) < 1e-9);
}

TEST_CASE("combined Sz + non-abelian point group: D6 ring, fixed n_up == Sz-block diag",
          "[symmetry_adapted][nonabelian][sz]") {
    // Combined U(1)×D6 reduction: restrict to the popcount-n_up sector AND
    // project onto the spatial irreps. Reference = H restricted to that Sz block.
    const int N = 6;
    const Permutation t{1, 2, 3, 4, 5, 0};
    const Permutation s{0, 5, 4, 3, 2, 1};
    auto Gp = generate_group({t, s});
    auto gi = decompose_irreps(Gp, N);
    const Eigen::MatrixXcd H = heisenberg_square(N);

    ed::symmetry::ConnectFn connect =
        [&H](std::uint64_t st, const std::function<void(std::uint64_t, Complex)>& emit) {
            for (int sp = 0; sp < H.rows(); ++sp) {
                const Complex h = H(sp, static_cast<Eigen::Index>(st));
                if (std::abs(h) > 1e-15) emit(static_cast<std::uint64_t>(sp), h);
            }
        };

    long long check_total = 0;
    for (int n_up = 0; n_up <= N; ++n_up) {
        // Reference: dense H restricted to the popcount-n_up computational states.
        std::vector<int> states;
        for (int b = 0; b < (1 << N); ++b)
            if (__builtin_popcount(static_cast<unsigned>(b)) == n_up) states.push_back(b);
        const int dsz = static_cast<int>(states.size());
        Eigen::MatrixXcd Hsz(dsz, dsz);
        for (int a = 0; a < dsz; ++a)
            for (int bb = 0; bb < dsz; ++bb) Hsz(a, bb) = H(states[a], states[bb]);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> refsz(Hsz);

        auto spec = ed::symmetry::symmetry_adapted_spectrum_terms(connect, gi, Gp, N, n_up);

        long long block_total = 0;
        for (std::size_t i = 0; i < spec.block_size.size(); ++i)
            block_total += static_cast<long long>(spec.block_irrep_dim[i]) * spec.block_size[i];
        REQUIRE(block_total == dsz);                       // completeness within the Sz sector
        check_total += block_total;

        REQUIRE(spec.eigenvalues.size() == static_cast<std::size_t>(dsz));
        for (int i = 0; i < refsz.eigenvalues().size(); ++i)
            REQUIRE(std::abs(spec.eigenvalues[static_cast<std::size_t>(i)] - refsz.eigenvalues()(i)) < 1e-9);
    }
    REQUIRE(check_total == (1LL << N));                    // Σ over Sz sectors == 2^N
}

#ifdef WITH_CUDA
TEST_CASE("symmetry-adapted GPU batched eigensolve == CPU (D6 ring, all cases)",
          "[symmetry_adapted][gpu][nonabelian]") {
    const int N = 6;
    const Permutation t{1, 2, 3, 4, 5, 0};
    const Permutation s{0, 5, 4, 3, 2, 1};
    auto Gp = generate_group({t, s});
    auto gi = decompose_irreps(Gp, N);
    const Eigen::MatrixXcd H = heisenberg_square(N);
    ed::symmetry::ConnectFn connect =
        [&H](std::uint64_t st, const std::function<void(std::uint64_t, Complex)>& emit) {
            for (int sp = 0; sp < H.rows(); ++sp) {
                const Complex h = H(sp, static_cast<Eigen::Index>(st));
                if (std::abs(h) > 1e-15) emit(static_cast<std::uint64_t>(sp), h);
            }
        };
    for (int n_up : {-1, 3}) {
        auto cpu = ed::symmetry::symmetry_adapted_spectrum_terms(connect, gi, Gp, N, n_up);
        auto gpu = ed::symmetry::symmetry_adapted_spectrum_gpu(connect, gi, Gp, N, n_up);
        REQUIRE(gpu.eigenvalues.size() == cpu.eigenvalues.size());
        std::sort(cpu.eigenvalues.begin(), cpu.eigenvalues.end());
        std::sort(gpu.eigenvalues.begin(), gpu.eigenvalues.end());
        for (std::size_t i = 0; i < cpu.eigenvalues.size(); ++i)
            REQUIRE(std::abs(gpu.eigenvalues[i] - cpu.eigenvalues[i]) < 1e-9);
    }
    // finite-T GPU == CPU
    const std::vector<double> temps{0.2, 0.5, 1.0, 3.0};
    auto tc = ed::symmetry::symmetry_adapted_thermodynamics(connect, gi, Gp, N, temps);
    auto tg = ed::symmetry::symmetry_adapted_thermodynamics_gpu(connect, gi, Gp, N, temps);
    for (std::size_t i = 0; i < temps.size(); ++i) {
        REQUIRE(std::abs(tg.energy[i] - tc.energy[i]) < 1e-9);
        REQUIRE(std::abs(tg.specific_heat[i] - tc.specific_heat[i]) < 1e-9);
    }
}
#endif  // WITH_CUDA

TEST_CASE("symmetry-adapted finite-T == exact canonical thermo (D6 ring)",
          "[symmetry_adapted][thermal][nonabelian]") {
    const int N = 6;
    const Permutation t{1, 2, 3, 4, 5, 0};
    const Permutation s{0, 5, 4, 3, 2, 1};
    auto Gp = generate_group({t, s});
    auto gi = decompose_irreps(Gp, N);
    const Eigen::MatrixXcd H = heisenberg_square(N);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> ref(H);

    ed::symmetry::ConnectFn connect =
        [&H](std::uint64_t st, const std::function<void(std::uint64_t, Complex)>& emit) {
            for (int sp = 0; sp < H.rows(); ++sp) {
                const Complex h = H(sp, static_cast<Eigen::Index>(st));
                if (std::abs(h) > 1e-15) emit(static_cast<std::uint64_t>(sp), h);
            }
        };

    const std::vector<double> temps{0.1, 0.3, 0.7, 1.5, 4.0};
    auto td = ed::symmetry::symmetry_adapted_thermodynamics(connect, gi, Gp, N, temps);

    // Reference: exact canonical thermo from the full 2^N spectrum.
    const auto& ev = ref.eigenvalues();
    const double E0 = ev(0);
    for (std::size_t i = 0; i < temps.size(); ++i) {
        const double beta = 1.0 / temps[i];
        double Z = 0, E = 0, E2 = 0;
        for (int n = 0; n < ev.size(); ++n) {
            const double w = std::exp(-beta * (ev(n) - E0));
            Z += w; E += w * ev(n); E2 += w * ev(n) * ev(n);
        }
        const double Eavg = E / Z, Cv = beta * beta * (E2 / Z - Eavg * Eavg);
        REQUIRE(std::abs(td.energy[i] - Eavg) < 1e-9);
        REQUIRE(std::abs(td.specific_heat[i] - Cv) < 1e-9);
    }
}

TEST_CASE("symmetry-adapted GS DSSF == brute-force Lehmann (S^z_0 on D6 ring)",
          "[symmetry_adapted][dssf][nonabelian]") {
    const int N = 6;
    const Permutation t{1, 2, 3, 4, 5, 0};
    const Permutation s{0, 5, 4, 3, 2, 1};
    auto Gp = generate_group({t, s});
    auto gi = decompose_irreps(Gp, N);
    const Eigen::MatrixXcd H = heisenberg_square(N);

    // Observable O = S^z at site 0 (diagonal). <0|O†O|0> = 1/4 (sum rule).
    const std::uint64_t dim = std::uint64_t{1} << N;
    Eigen::MatrixXcd O = Eigen::MatrixXcd::Zero(dim, dim);
    for (std::uint64_t b = 0; b < dim; ++b) O(b, b) = ((b >> 0) & 1) ? 0.5 : -0.5;

    auto mk_connect = [](const Eigen::MatrixXcd& M) {
        return ed::symmetry::ConnectFn(
            [&M](std::uint64_t st, const std::function<void(std::uint64_t, Complex)>& emit) {
                for (int sp = 0; sp < M.rows(); ++sp) {
                    const Complex h = M(sp, static_cast<Eigen::Index>(st));
                    if (std::abs(h) > 1e-15) emit(static_cast<std::uint64_t>(sp), h);
                }
            });
    };

    const double wmin = -1.0, wmax = 8.0, eta = 0.05;
    const int nw = 200;
    auto r = ed::symmetry::symmetry_adapted_ground_state_dssf(
        mk_connect(H), mk_connect(O), gi, Gp, N, wmin, wmax, nw, eta);

    // Brute-force Lehmann reference.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H);
    const Eigen::VectorXcd gsv = es.eigenvectors().col(0);
    const double E0 = es.eigenvalues()(0);
    const Eigen::VectorXcd chi = O * gsv;
    std::vector<double> ref_S(static_cast<std::size_t>(nw), 0.0);
    double ref_weight = 0.0;
    for (int n = 0; n < es.eigenvalues().size(); ++n) {
        const Complex ov = es.eigenvectors().col(n).adjoint() * chi;
        const double w = std::norm(ov);
        ref_weight += w;
        const double de = es.eigenvalues()(n) - E0;
        for (int i = 0; i < nw; ++i) {
            const double x = (wmin + (wmax - wmin) / (nw - 1) * i) - de;
            ref_S[static_cast<std::size_t>(i)] += w * (eta / M_PI) / (x * x + eta * eta);
        }
    }

    REQUIRE(std::abs(r.ground_energy - E0) < 1e-9);
    REQUIRE(std::abs(r.total_weight - 0.25) < 1e-9);            // sum rule <(S^z)^2>=1/4
    REQUIRE(std::abs(r.total_weight - ref_weight) < 1e-9);
    for (int i = 0; i < nw; ++i)
        REQUIRE(std::abs(r.spectral[static_cast<std::size_t>(i)] - ref_S[static_cast<std::size_t>(i)]) < 1e-7);
}
