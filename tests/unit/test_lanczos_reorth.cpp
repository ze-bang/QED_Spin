// =============================================================================
// test_lanczos_reorth (Catch2 v3, Phase 3a #2)
//
// Coverage for the blocked-tile reorthogonalization helpers in
// include/ed/io/lanczos_reorth.h:
//
//   1. blocked_reorth correctness against the per-vector MGS reference on a
//      hand-rolled fixture (5 random vectors of dim 32, w with overlaps in
//      every direction).
//   2. blocked_reorth threshold filter actually skips below-threshold
//      overlaps (the resulting w preserves the small-overlap component).
//   3. blocked_reorth skip_predicate filters the requested indices.
//   4. load_basis_tile zero-copies the in-memory buffer correctly + falls
//      back to the legacy on-disk format when no buffer is registered.
//   5. End-to-end tile-size invariance: lanczos_selective_reorth on N=8 /
//      max_iter=30 Heisenberg PBC must converge to the dense reference for
//      every ED_LANCZOS_REORTH_TILE in {1, 4, 16}.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/blas_lapack_wrapper.h>
#include <ed/io/lanczos_basis_buffer.h>
#include <ed/io/lanczos_reorth.h>
#include <ed/solvers/lanczos.h>

#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>

using namespace ed_tests;
using lanczos_io::Complex;
using lanczos_io::ComplexVector;

namespace {

ComplexVector random_complex_vec(uint64_t N, uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    ComplexVector v(N);
    for (auto& c : v) c = Complex(nd(gen), nd(gen));
    return v;
}

// MGS-orthonormalize a set of vectors in place. This is the precondition
// blocked_reorth assumes about the basis tile (Lanczos basis vectors are
// orthonormal up to roundoff in practice); without it CGS and MGS
// disagree by O(||V||^2 ||w||) and the comparison test below is not
// well-posed.
void mgs_orthonormalize(std::vector<ComplexVector>& V) {
    for (std::size_t k = 0; k < V.size(); ++k) {
        // Subtract projections onto earlier vectors.
        for (std::size_t j = 0; j < k; ++j) {
            Complex overlap;
            cblas_zdotc_sub(static_cast<int>(V[j].size()), V[j].data(), 1,
                            V[k].data(), 1, &overlap);
            Complex neg = -overlap;
            cblas_zaxpy(static_cast<int>(V[j].size()), &neg, V[j].data(), 1,
                        V[k].data(), 1);
        }
        // Normalize.
        const double n = std::sqrt(std::real(
            std::inner_product(V[k].begin(), V[k].end(), V[k].begin(),
                               Complex(0.0, 0.0),
                               std::plus<Complex>(),
                               [](Complex a, Complex b) {
                                   return std::conj(a) * b;
                               })));
        REQUIRE(n > 0.0);
        const Complex inv(1.0 / n, 0.0);
        cblas_zscal(static_cast<int>(V[k].size()), &inv, V[k].data(), 1);
    }
}

// Per-vector MGS-style reference: the original semantics that lived in
// lanczos.cpp's selective re-orth loop. For an orthonormal V (which we
// enforce in every fixture below) this is exactly equivalent to a single
// CGS pass — that is the property blocked_reorth promises.
ComplexVector mgs_reference(const std::vector<ComplexVector>& V,
                            const ComplexVector& w_in,
                            double threshold) {
    ComplexVector w = w_in;
    for (const auto& v : V) {
        Complex overlap;
        cblas_zdotc_sub(static_cast<int>(v.size()), v.data(), 1, w.data(), 1,
                        &overlap);
        if (std::abs(overlap) <= threshold) continue;
        Complex neg = -overlap;
        cblas_zaxpy(static_cast<int>(v.size()), &neg, v.data(), 1, w.data(),
                    1);
    }
    return w;
}

// Pack a list of column vectors into the column-major tile layout
// blocked_reorth expects.
std::vector<Complex> pack_tile(const std::vector<ComplexVector>& V) {
    if (V.empty()) return {};
    const uint64_t N = V.front().size();
    std::vector<Complex> tile(static_cast<std::size_t>(N) * V.size());
    for (std::size_t k = 0; k < V.size(); ++k) {
        REQUIRE(V[k].size() == N);
        std::memcpy(tile.data() + k * N, V[k].data(), N * sizeof(Complex));
    }
    return tile;
}

}  // namespace

// ============================================================================
// 1. blocked_reorth correctness: matches per-vector MGS to ~1e-13
// ============================================================================
TEST_CASE("blocked_reorth matches the per-vector MGS reference",
          "[lanczos_reorth][correctness]") {
    const uint64_t N = 32;
    const uint64_t count = 5;

    std::vector<ComplexVector> V;
    V.reserve(count);
    for (uint64_t k = 0; k < count; ++k) {
        V.push_back(random_complex_vec(N, /*seed=*/100 + k));
    }
    mgs_orthonormalize(V);
    auto w_in = random_complex_vec(N, /*seed=*/777);

    auto w_blocked = w_in;
    auto tile = pack_tile(V);
    std::vector<uint64_t> indices(count);
    for (uint64_t k = 0; k < count; ++k) indices[k] = k;
    lanczos_io::blocked_reorth(N, count, tile.data(), w_blocked.data(),
                               /*threshold=*/0.0, indices.data(),
                               /*skip=*/nullptr);

    auto w_mgs = mgs_reference(V, w_in, /*threshold=*/0.0);

    const double err = l2_diff(w_blocked, w_mgs);
    INFO("CGS-vs-MGS L2 difference = " << err);
    REQUIRE(err < 1e-12);
}

// ============================================================================
// 2. Threshold filter
// ============================================================================
TEST_CASE("blocked_reorth threshold filter skips small overlaps",
          "[lanczos_reorth][threshold]") {
    const uint64_t N = 16;
    // Two clean orthonormal-like directions and one nearly-orthogonal vector
    // (overlap ~ 1e-8). The latter should be skipped by a 1e-5 threshold.
    ComplexVector v0(N, Complex(0.0, 0.0));
    ComplexVector v1(N, Complex(0.0, 0.0));
    v0[0] = Complex(1.0, 0.0);
    v1[1] = Complex(1.0, 0.0);
    ComplexVector v_tiny(N, Complex(0.0, 0.0));
    v_tiny[2] = Complex(1.0, 0.0);

    ComplexVector w(N, Complex(0.0, 0.0));
    w[0] = Complex(0.5, 0.0);                  // overlap with v0  = 0.5
    w[1] = Complex(0.3, 0.0);                  // overlap with v1  = 0.3
    w[2] = Complex(1.0e-8, 0.0);               // overlap with v_tiny = 1e-8 -> skipped
    w[3] = Complex(0.7, 0.0);                  // untouched

    std::vector<ComplexVector> V = {v0, v1, v_tiny};
    auto tile = pack_tile(V);
    std::vector<uint64_t> indices = {0, 1, 2};

    lanczos_io::blocked_reorth(N, 3, tile.data(), w.data(),
                               /*threshold=*/1.0e-5,
                               indices.data(), /*skip=*/nullptr);

    REQUIRE(std::abs(w[0]) < 1e-14);                         // subtracted
    REQUIRE(std::abs(w[1]) < 1e-14);                         // subtracted
    REQUIRE(std::abs(w[2] - Complex(1.0e-8, 0.0)) < 1e-15);  // PRESERVED
    REQUIRE(std::abs(w[3] - Complex(0.7,    0.0)) < 1e-15);  // untouched
}

// ============================================================================
// 3. Skip predicate
// ============================================================================
TEST_CASE("blocked_reorth skip predicate masks out requested indices",
          "[lanczos_reorth][skip]") {
    const uint64_t N = 8;
    ComplexVector v0(N, Complex(0.0, 0.0));
    ComplexVector v1(N, Complex(0.0, 0.0));
    v0[0] = Complex(1.0, 0.0);
    v1[1] = Complex(1.0, 0.0);

    ComplexVector w(N, Complex(0.0, 0.0));
    w[0] = Complex(0.4, 0.0);
    w[1] = Complex(0.6, 0.0);

    std::vector<ComplexVector> V = {v0, v1};
    auto tile = pack_tile(V);
    std::vector<uint64_t> indices = {17, 42};   // arbitrary basis indices

    // Skip global index 17 (= column 0 = v0). v1 should still be subtracted.
    auto skip = [](uint64_t idx) { return idx == 17; };
    lanczos_io::blocked_reorth(N, 2, tile.data(), w.data(),
                               /*threshold=*/0.0, indices.data(), skip);

    REQUIRE(std::abs(w[0] - Complex(0.4, 0.0)) < 1e-15);  // PRESERVED
    REQUIRE(std::abs(w[1])                       < 1e-14);  // subtracted
}

// ============================================================================
// 4. load_basis_tile in-memory + on-disk paths
// ============================================================================
TEST_CASE("load_basis_tile fills column-major tiles from in-memory buffer",
          "[lanczos_reorth][load_tile][in_memory]") {
    if (lanczos_io::force_disk_storage()) {
        WARN("ED_LANCZOS_DISK=1 -- in-memory tile path disabled, skipping");
        return;
    }

    const std::string key = "/tmp/ed_test_load_tile_mem";
    const uint64_t N = 6;
    const uint64_t count = 3;

    lanczos_io::register_basis_buffer(key, N, /*reserve=*/8);
    std::vector<ComplexVector> sources;
    sources.reserve(count);
    for (uint64_t k = 0; k < count; ++k) {
        auto v = random_complex_vec(N, 200 + k);
        sources.push_back(v);
        REQUIRE(lanczos_io::append_basis_vector(key, v));
    }

    std::vector<Complex> tile;
    const uint64_t loaded =
        lanczos_io::load_basis_tile(key, /*k_start=*/0, count, N, tile);
    REQUIRE(loaded == count);
    REQUIRE(tile.size() == static_cast<std::size_t>(N) * count);

    // Each column must equal the appended source vector.
    for (uint64_t k = 0; k < count; ++k) {
        ComplexVector col(tile.begin() + k * N, tile.begin() + (k + 1) * N);
        REQUIRE(l2_diff(col, sources[k]) < 1e-15);
    }

    lanczos_io::release_basis_buffer(key);
}

TEST_CASE("load_basis_tile falls back to legacy on-disk format",
          "[lanczos_reorth][load_tile][on_disk]") {
    const std::string scratch =
        ed_tests::make_scratch_dir("test_lanczos_reorth", "ondisk");
    const uint64_t N = 5;
    const uint64_t count = 4;

    // Make sure no buffer is registered for this key.
    lanczos_io::release_basis_buffer(scratch);

    // Hand-write the legacy basis_<i>.dat files.
    std::vector<ComplexVector> sources;
    sources.reserve(count);
    for (uint64_t k = 0; k < count; ++k) {
        auto v = random_complex_vec(N, 300 + k);
        sources.push_back(v);
        std::ofstream out(scratch + "/basis_" + std::to_string(k) + ".dat",
                          std::ios::binary);
        REQUIRE(static_cast<bool>(out));
        out.write(reinterpret_cast<const char*>(v.data()),
                  static_cast<std::streamsize>(N) * sizeof(Complex));
    }

    std::vector<Complex> tile;
    const uint64_t loaded =
        lanczos_io::load_basis_tile(scratch, /*k_start=*/0, count, N, tile);
    REQUIRE(loaded == count);

    for (uint64_t k = 0; k < count; ++k) {
        ComplexVector col(tile.begin() + k * N, tile.begin() + (k + 1) * N);
        REQUIRE(l2_diff(col, sources[k]) < 1e-15);
    }
}

// ============================================================================
// 5. End-to-end: lanczos_selective_reorth converges identically across
//    several tile sizes.
// ============================================================================
TEST_CASE("lanczos_selective_reorth is invariant under reorth tile size",
          "[lanczos_reorth][solver_invariance]") {
    const uint64_t N_sites = 8;
    auto op = build_heisenberg_chain(N_sites, 1.0, /*periodic=*/true);
    const uint64_t dim = 1ULL << N_sites;
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        op->apply(in, out, static_cast<size_t>(n));
    };
    auto ref = reference_from_operator(*op, dim);

    const uint64_t max_iter = 30;
    const uint64_t n_eig    = 3;

    for (const char* tile_str : {"1", "4", "16"}) {
        SECTION(std::string("ED_LANCZOS_REORTH_TILE=") + tile_str) {
            setenv("ED_LANCZOS_REORTH_TILE", tile_str, /*overwrite=*/1);
            REQUIRE(lanczos_io::reorth_tile_size() ==
                    static_cast<uint64_t>(std::stoll(tile_str)));

            std::vector<double> eigs;
            const std::string dir = ed_tests::make_scratch_dir(
                "test_lanczos_reorth",
                std::string("solver_tile_") + tile_str);
            lanczos_selective_reorth(Hv, dim, max_iter, n_eig, 1e-12, eigs,
                                     dir, /*eigenvectors=*/false);

            REQUIRE(eigs.size() >= n_eig);
            std::sort(eigs.begin(), eigs.end());

            // Same tolerance as test_lanczos_variants' "lanczos" section
            // for ground state, slightly relaxed for higher Ritz.
            REQUIRE(std::abs(eigs.front() - ref.eigs.front()) <= 1e-6);
            for (uint64_t i = 0; i < n_eig; ++i) {
                double best = std::numeric_limits<double>::infinity();
                for (double e : ref.eigs) {
                    best = std::min(best, std::abs(eigs[i] - e));
                }
                INFO("tile=" << tile_str << " Ritz #" << i << " = "
                     << eigs[i] << ", nearest dense |Δ| = " << best);
                REQUIRE(best <= 1e-3);  // selective_reorth on N=8 is loose
            }
        }
    }

    unsetenv("ED_LANCZOS_REORTH_TILE");
}

// Tile size knob: invalid values fall back to default (16).
TEST_CASE("ED_LANCZOS_REORTH_TILE clamps to [1, 256] and defaults to 16",
          "[lanczos_reorth][knob]") {
    unsetenv("ED_LANCZOS_REORTH_TILE");
    REQUIRE(lanczos_io::reorth_tile_size() == 16);

    setenv("ED_LANCZOS_REORTH_TILE", "0", 1);
    REQUIRE(lanczos_io::reorth_tile_size() == 1);  // clamped low

    setenv("ED_LANCZOS_REORTH_TILE", "9999", 1);
    REQUIRE(lanczos_io::reorth_tile_size() == 256);  // clamped high

    setenv("ED_LANCZOS_REORTH_TILE", "garbage", 1);
    REQUIRE(lanczos_io::reorth_tile_size() == 16);  // parse failure → default

    setenv("ED_LANCZOS_REORTH_TILE", "32", 1);
    REQUIRE(lanczos_io::reorth_tile_size() == 32);

    unsetenv("ED_LANCZOS_REORTH_TILE");
}
