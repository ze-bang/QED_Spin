// =============================================================================
// test_cpu_gpu_equivalence (Catch2 v3, P1.9)
//
// Bit-equivalence (well, ~1e-10 relative) checks between the CPU
// `Operator::apply()` / `lanczos()` path and the GPU `GPUOperator::matVec()`
// / `GPULanczos::run()` path.
//
// Why
// ---
// The GPU code lives in a separate static library (`ed_solvers_gpu`) and
// the only thing that catches divergence between the two paths is hand-
// running both binaries on a workstation. This test pulls the comparison
// onto every push that has a GPU runner attached.
//
// Hosted runner story
// -------------------
// * Compiles unconditionally (gated on WITH_CUDA at build time -- the
//   target is added by `ed_add_test` only when WITH_CUDA=ON).
// * At *runtime*, every TEST_CASE first checks `cudaGetDeviceCount(...)`
//   and SKIPs if no device is present, so the build-only CUDA lane in
//   `.github/workflows/ci.yml` (`linux-cuda-build`) does not fail.
//
// Coverage
// --------
// * 4-site Heisenberg OBC, H*v on a deterministic complex unit vector
//   (relative L2 < 1e-10).
// * 8-site Heisenberg PBC, ground-state Lanczos eigenvalue agreement
//   (|E_cpu - E_gpu| < 1e-8).
// =============================================================================

#include "common/catch2_harness.h"

#ifdef WITH_CUDA

#include <ed/gpu/gpu_operator.cuh>
#include <ed/gpu/gpu_lanczos.cuh>
#include <ed/gpu/gpu_solvers.h>      // Phase 4: MatVecOperator& GPU overloads
#include <ed/matvec/matvec.h>
#include <ed/solvers/lanczos.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

namespace {

using Complex = std::complex<double>;

// ----- helpers --------------------------------------------------------------

bool gpu_available() {
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // cudaGetLastError() must be called to clear the sticky error so
        // subsequent (legitimate) tests do not inherit it.
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

// Mirror the CPU Heisenberg builder (test_harness.h) in GPU land. Same
// J=1, same SoA term ordering, no field. Periodic if requested.
std::unique_ptr<GPUOperator>
build_gpu_heisenberg_chain(int N, bool periodic = false) {
    auto op = std::make_unique<GPUOperator>(N, /*spin=*/0.5f);
    const Complex J_real(1.0, 0.0);
    const Complex J_half(0.5, 0.0);
    const int last = periodic ? N : (N - 1);
    for (int i = 0; i < last; ++i) {
        const int j = (i + 1) % N;
        op->addTwoBodyTerm(/*op1=*/2, i, /*op2=*/2, j, J_real);
        op->addTwoBodyTerm(/*op1=*/0, i, /*op2=*/1, j, J_half);
        op->addTwoBodyTerm(/*op1=*/1, i, /*op2=*/0, j, J_half);
    }
    op->copyTransformDataToDevice();
    return op;
}

double l2_norm(const std::vector<Complex>& v) {
    double s = 0.0;
    for (const auto& x : v) s += std::norm(x);
    return std::sqrt(s);
}

double l2_diff(const std::vector<Complex>& a, const std::vector<Complex>& b) {
    double s = 0.0;
    const size_t n = std::min(a.size(), b.size());
    for (size_t i = 0; i < n; ++i) s += std::norm(a[i] - b[i]);
    return std::sqrt(s);
}

}  // namespace

// ============================================================================

TEST_CASE("CPU/GPU equivalence: 4-site Heisenberg H*v matches within 1e-10",
          "[cpu_gpu_eq][matvec]") {
    if (!gpu_available()) {
        SKIP("No CUDA device available -- skipping GPU equivalence test.");
    }

    const int N = 4;
    const uint64_t dim = 1ULL << N;

    auto cpu_op = ed_tests::build_heisenberg_chain(N, /*J=*/1.0,
                                                   /*periodic=*/false);
    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/false);

    // Deterministic unit vector
    auto v = ed_tests::random_unit_vector(dim, /*seed=*/12345);

    std::vector<Complex> cpu_out(dim, Complex(0.0, 0.0));
    cpu_op->apply(v.data(), cpu_out.data(), dim);

    std::vector<Complex> gpu_out(dim, Complex(0.0, 0.0));
    gpu_op->matVec(v.data(), gpu_out.data(), static_cast<int>(dim));

    const double err  = l2_diff(cpu_out, gpu_out);
    const double norm = l2_norm(cpu_out);
    INFO("dim=" << dim
         << "  ||cpu - gpu||_2 = " << err
         << "  ||cpu||_2 = "       << norm
         << "  rel = "             << err / std::max(norm, 1e-30));
    REQUIRE(err / std::max(norm, 1e-30) < 1e-10);
}

TEST_CASE("CPU/GPU equivalence: 8-site Heisenberg ground state eigenvalue "
          "matches within 1e-8",
          "[cpu_gpu_eq][lanczos]") {
    if (!gpu_available()) {
        SKIP("No CUDA device available -- skipping GPU equivalence test.");
    }

    const int N = 8;
    const uint64_t dim = 1ULL << N;

    // ------ CPU -----------------------------------------------------------
    auto cpu_op = ed_tests::build_heisenberg_chain(N, /*J=*/1.0,
                                                   /*periodic=*/true);
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        cpu_op->apply(in, out, static_cast<size_t>(n));
    };
    std::vector<double> cpu_eigs;
    lanczos(Hv, dim, /*max_iter=*/100, /*exct=*/1, /*tol=*/1e-12,
            cpu_eigs, /*dir=*/"", /*eigenvectors=*/false);
    REQUIRE(!cpu_eigs.empty());

    // ------ GPU -----------------------------------------------------------
    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);
    GPULanczos gpu_lanczos(gpu_op.get(), /*max_iter=*/100,
                           /*tolerance=*/1e-12);
    std::vector<double> gpu_eigs;
    std::vector<std::vector<Complex>> gpu_vecs;  // unused
    gpu_lanczos.run(/*num_eigenvalues=*/1, gpu_eigs, gpu_vecs,
                    /*compute_vectors=*/false);
    REQUIRE(!gpu_eigs.empty());

    INFO("dim=" << dim
         << "  E_cpu=" << cpu_eigs[0]
         << "  E_gpu=" << gpu_eigs[0]
         << "  |Δ|="   << std::abs(cpu_eigs[0] - gpu_eigs[0]));
    REQUIRE(std::abs(cpu_eigs[0] - gpu_eigs[0]) < 1e-8);
}

// ============================================================================
// Phase 4 (matvec-unification): the new ed::matvec::gpu::* MatVecOperator-
// taking overloads. Validates that:
//
//   1. ed::matvec::gpu::lanczos(GPUOperator&, ...) -- the type-safe overload
//      -- matches the legacy GPULanczos class output.
//   2. ed::matvec::gpu::lanczos(MatVecOperator&, ...) -- the unified
//      MatVecOperator& overload that dynamic_casts to GPUOperator -- also
//      matches.
//   3. The MatVecOperator& overload correctly rejects a non-GPU operator
//      (a CPU `Operator` is a MatVecOperator with MemorySpace::Host).
// ============================================================================

TEST_CASE("Phase 4 GPU: ed::matvec::gpu::lanczos(GPUOperator&) "
          "matches the GPULanczos class on 8-site Heisenberg",
          "[cpu_gpu_eq][matvec_phase4]") {
    if (!gpu_available()) {
        SKIP("No CUDA device available -- skipping Phase 4 GPU overload test.");
    }

    const int N = 8;
    const uint64_t dim = 1ULL << N;
    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);

    // Legacy class-based API (used as the reference)
    GPULanczos gpu_lanczos(gpu_op.get(), /*max_iter=*/100, /*tolerance=*/1e-12);
    std::vector<double> ref_eigs;
    std::vector<std::vector<Complex>> ref_vecs;
    gpu_lanczos.run(/*num_eigenvalues=*/1, ref_eigs, ref_vecs,
                    /*compute_vectors=*/false);
    REQUIRE(!ref_eigs.empty());

    // Phase 4 type-safe overload
    std::vector<double> typed_eigs;
    ed::matvec::gpu::lanczos(*gpu_op, /*N=*/static_cast<int>(dim),
                             /*max_iter=*/100, /*num_eigs=*/1, /*tol=*/1e-12,
                             typed_eigs, /*dir=*/"", /*eigenvectors=*/false);
    REQUIRE(!typed_eigs.empty());

    INFO("E_ref="    << ref_eigs[0]
         << "  E_typed=" << typed_eigs[0]
         << "  |Δ|="     << std::abs(ref_eigs[0] - typed_eigs[0]));
    REQUIRE(std::abs(ref_eigs[0] - typed_eigs[0]) < 1e-8);
}

TEST_CASE("Phase 4 GPU: ed::matvec::gpu::lanczos(MatVecOperator&) "
          "polymorphic overload matches the GPULanczos class",
          "[cpu_gpu_eq][matvec_phase4]") {
    if (!gpu_available()) {
        SKIP("No CUDA device available -- skipping Phase 4 GPU overload test.");
    }

    const int N = 8;
    const uint64_t dim = 1ULL << N;
    auto gpu_op = build_gpu_heisenberg_chain(N, /*periodic=*/true);

    // Reference
    GPULanczos gpu_lanczos(gpu_op.get(), /*max_iter=*/100, /*tolerance=*/1e-12);
    std::vector<double> ref_eigs;
    std::vector<std::vector<Complex>> ref_vecs;
    gpu_lanczos.run(1, ref_eigs, ref_vecs, false);
    REQUIRE(!ref_eigs.empty());

    // Phase 4 polymorphic overload -- pass through MatVecOperator&
    const ed::matvec::MatVecOperator& as_mv = *gpu_op;
    REQUIRE(as_mv.memory_space() == ed::matvec::MemorySpace::CudaDevice);
    std::vector<double> poly_eigs;
    ed::matvec::gpu::lanczos(as_mv, static_cast<int>(dim), 100, 1, 1e-12,
                             poly_eigs);
    REQUIRE(!poly_eigs.empty());

    INFO("E_ref=" << ref_eigs[0] << "  E_poly=" << poly_eigs[0]
         << "  |Δ|=" << std::abs(ref_eigs[0] - poly_eigs[0]));
    REQUIRE(std::abs(ref_eigs[0] - poly_eigs[0]) < 1e-8);
}

TEST_CASE("Phase 4 GPU: ed::matvec::gpu::lanczos(MatVecOperator&) rejects "
          "a Host-memory operator",
          "[cpu_gpu_eq][matvec_phase4]") {
    // Does NOT need a GPU device -- this is purely a runtime type check.
    const int N = 4;
    const uint64_t dim = 1ULL << N;
    auto cpu_op = ed_tests::build_heisenberg_chain(N, /*J=*/1.0,
                                                   /*periodic=*/false);
    const ed::matvec::MatVecOperator& as_mv = *cpu_op;
    REQUIRE(as_mv.memory_space() == ed::matvec::MemorySpace::Host);

    std::vector<double> eigs;
    REQUIRE_THROWS_AS(
        ed::matvec::gpu::lanczos(as_mv, static_cast<int>(dim), 50, 1, 1e-10,
                                 eigs),
        std::invalid_argument);
}

#else  // !WITH_CUDA

// When the project is configured without CUDA, the CMake target generation
// skips this file entirely (see CMakeLists.txt -- test_cpu_gpu_equivalence
// is only added when WITH_CUDA is ON), so this branch is effectively dead
// code. We keep a no-op TEST_CASE here only so a stray manual `add_executable`
// of the file in a non-CUDA build still produces a non-empty binary.
TEST_CASE("CPU/GPU equivalence: WITH_CUDA=OFF -- no-op",
          "[cpu_gpu_eq][skip]") {
    SUCCEED("Built without WITH_CUDA -- nothing to compare.");
}

#endif  // WITH_CUDA
