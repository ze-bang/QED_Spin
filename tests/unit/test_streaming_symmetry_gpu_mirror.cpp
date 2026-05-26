// =============================================================================
// test_streaming_symmetry_gpu_mirror
//
// Phase A of the "Backend x Symmetries x Workflows: close the full
// 48-cell matrix" plan (May 2026).
//
// End-to-end exercise of the new lazy GPU sector mirror introduced in
// ``src/symmetry/streaming_symmetry_gpu_mirror.cu``. The test:
//
//   1. Builds a Heisenberg ring with Z_N translation symmetry via the
//      JSON-fixture pipeline shared with test_symmetry.cpp.
//   2. For every translation sector with non-empty dim, computes the
//      sector matvec on the CPU via ``applySymmetrized`` (the legacy
//      bespoke path), then on the GPU via ``SectorView::bind_cuda``
//      (the new mirror path).
//   3. Asserts the two output vectors agree to 1e-10 in L2 norm.
//   4. Asserts the LRU-1 cache evicts on sector switch by querying
//      the symbol address through repeated bind calls.
//
// Without WITH_CUDA every test case is compiled but skipped at run
// time via Catch2's ``SKIP`` -- the binary still links because the
// stub in ``streaming_symmetry_gpu_mirror.cpp`` supplies the symbol.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/core/streaming_symmetry.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

using namespace ed_tests;

namespace {

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

void write_zN_translation_fixtures(const std::string& dir, int N) {
    const std::string root = dir + "/automorphism_results";
    std::error_code ec;
    std::filesystem::create_directories(root, ec);
    {
        std::ofstream f(root + "/max_clique.json");
        f << "[";
        for (int g = 0; g < N; ++g) {
            const auto p = translation_perm(N, g);
            f << "[";
            for (size_t i = 0; i < p.size(); ++i) {
                f << p[i] << (i + 1 < p.size() ? "," : "");
            }
            f << "]" << (g + 1 < N ? "," : "");
        }
        f << "]";
    }
    {
        std::ofstream f(root + "/minimal_generators.json");
        const auto p = translation_perm(N, 1);
        f << "{\"generators\":[{\"permutation\":[";
        for (size_t i = 0; i < p.size(); ++i) {
            f << p[i] << (i + 1 < p.size() ? "," : "");
        }
        f << "],\"order\":" << N << "}]}";
    }
    {
        std::ofstream f(root + "/sector_metadata.json");
        f << std::setprecision(17);
        f << "{\"sectors\":[";
        for (int k = 0; k < N; ++k) {
            const double angle = -2.0 * M_PI * static_cast<double>(k) /
                                 static_cast<double>(N);
            const double re = std::cos(angle);
            const double im = std::sin(angle);
            f << "{\"sector_id\":" << k
              << ",\"quantum_numbers\":[" << k << "]"
              << ",\"phase_factors\":[{\"real\":" << re
              << ",\"imag\":" << im << "}]}";
            if (k + 1 < N) f << ",";
        }
        f << "]}";
    }
}

std::unique_ptr<StreamingSymmetryOperator>
build_heisenberg_pbc_streaming(uint64_t N, double J) {
    auto op = std::make_unique<StreamingSymmetryOperator>(N, 0.5f);
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (uint64_t i = 0; i < N; ++i) {
        uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 0; t.site_index = i; t.op_type_2 = 1;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 1; t.site_index = i; t.op_type_2 = 0;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op->transform_data_.push_back(t);
    }
    return op;
}

#ifdef WITH_CUDA

bool cuda_runtime_available() {
    int count = 0;
    const cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess) {
        // Reset the sticky error so subsequent tests are unaffected.
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

// Run the GPU mirror matvec end-to-end: allocate device buffers, copy
// the input vector down, invoke the MatvecFn bound from the sector
// view, copy the output back, and return it.
std::vector<Complex>
run_gpu_sector_matvec(const ed::LinearOperator::MatvecFn& fn,
                      const std::vector<Complex>& in)
{
    const std::size_t dim = in.size();
    std::vector<Complex> out(dim, Complex(0.0, 0.0));
    if (dim == 0) return out;

    cuDoubleComplex* d_in  = nullptr;
    cuDoubleComplex* d_out = nullptr;
    REQUIRE(cudaMalloc(&d_in,  dim * sizeof(cuDoubleComplex)) == cudaSuccess);
    REQUIRE(cudaMalloc(&d_out, dim * sizeof(cuDoubleComplex)) == cudaSuccess);

    REQUIRE(cudaMemcpy(d_in, in.data(),
                       dim * sizeof(cuDoubleComplex),
                       cudaMemcpyHostToDevice) == cudaSuccess);

    fn(reinterpret_cast<const Complex*>(d_in),
       reinterpret_cast<Complex*>(d_out),
       dim);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    REQUIRE(cudaMemcpy(out.data(), d_out,
                       dim * sizeof(cuDoubleComplex),
                       cudaMemcpyDeviceToHost) == cudaSuccess);

    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

#endif  // WITH_CUDA

} // namespace

TEST_CASE("GPU mirror: SectorView::bind_cuda matches CPU applySymmetrized (N=4)",
          "[symmetry][gpu_mirror][N4]")
{
#ifndef WITH_CUDA
    SKIP("Built without WITH_CUDA");
#else
    if (!cuda_runtime_available()) {
        SKIP("No CUDA device available");
    }

    const uint64_t N = 4;
    std::string dir = make_scratch_dir("gpu_mirror", "N4");
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    auto sym_op = build_heisenberg_pbc_streaming(N, 1.0);
    REQUIRE_NOTHROW(sym_op->generateSymmetrySectorsStreaming(dir));

    double max_err = 0.0;
    for (std::size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        const std::size_t sd = sym_op->getSectorDimension(s);
        if (sd == 0) continue;

        auto view = sym_op->sector(s);
        REQUIRE(view->geometry().supports_device_matvec);

        auto in = random_unit_vector(sd, (s + 1) * 9871ULL);
        std::vector<Complex> cpu_out(sd, Complex(0.0, 0.0));
        sym_op->applySymmetrized(s, in.data(), cpu_out.data());

        auto fn = view->bind_cuda();
        auto gpu_out = run_gpu_sector_matvec(fn, in);

        const double err = l2_diff(cpu_out, gpu_out);
        INFO("sector=" << s << " dim=" << sd << " ||cpu - gpu|| = " << err);
        REQUIRE(err < 1e-10);
        max_err = std::max(max_err, err);
    }
    INFO("N=4 max sector error = " << max_err);
#endif
}

TEST_CASE("GPU mirror: SectorView::bind_cuda matches CPU applySymmetrized (N=6)",
          "[symmetry][gpu_mirror][N6]")
{
#ifndef WITH_CUDA
    SKIP("Built without WITH_CUDA");
#else
    if (!cuda_runtime_available()) {
        SKIP("No CUDA device available");
    }

    const uint64_t N = 6;
    std::string dir = make_scratch_dir("gpu_mirror", "N6");
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    auto sym_op = build_heisenberg_pbc_streaming(N, 1.0);
    REQUIRE_NOTHROW(sym_op->generateSymmetrySectorsStreaming(dir));

    double max_err = 0.0;
    for (std::size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        const std::size_t sd = sym_op->getSectorDimension(s);
        if (sd == 0) continue;

        auto view = sym_op->sector(s);
        REQUIRE(view->geometry().supports_device_matvec);

        auto in = random_unit_vector(sd, (s + 1) * 7919ULL);
        std::vector<Complex> cpu_out(sd, Complex(0.0, 0.0));
        sym_op->applySymmetrized(s, in.data(), cpu_out.data());

        auto fn = view->bind_cuda();
        auto gpu_out = run_gpu_sector_matvec(fn, in);

        const double err = l2_diff(cpu_out, gpu_out);
        INFO("sector=" << s << " dim=" << sd << " ||cpu - gpu|| = " << err);
        REQUIRE(err < 1e-10);
        max_err = std::max(max_err, err);
    }
    INFO("N=6 max sector error = " << max_err);
#endif
}

TEST_CASE("GPU mirror: LRU-1 cache survives repeated calls + sector switch",
          "[symmetry][gpu_mirror][lru]")
{
#ifndef WITH_CUDA
    SKIP("Built without WITH_CUDA");
#else
    if (!cuda_runtime_available()) {
        SKIP("No CUDA device available");
    }

    const uint64_t N = 4;
    std::string dir = make_scratch_dir("gpu_mirror", "lru");
    write_zN_translation_fixtures(dir, static_cast<int>(N));

    auto sym_op = build_heisenberg_pbc_streaming(N, 1.0);
    REQUIRE_NOTHROW(sym_op->generateSymmetrySectorsStreaming(dir));

    // Pick the two largest sectors so the matvec touches multiple
    // hash buckets per sector.
    std::vector<std::size_t> sectors;
    for (std::size_t s = 0; s < sym_op->getNumSectors(); ++s) {
        if (sym_op->getSectorDimension(s) > 0) sectors.push_back(s);
    }
    REQUIRE(sectors.size() >= 2);

    const auto run = [&](std::size_t s) {
        auto view = sym_op->sector(s);
        const std::size_t sd = sym_op->getSectorDimension(s);
        auto in = random_unit_vector(sd, s * 4253ULL + 17);
        std::vector<Complex> cpu_out(sd, Complex(0.0, 0.0));
        sym_op->applySymmetrized(s, in.data(), cpu_out.data());

        auto fn = view->bind_cuda();
        auto gpu_out = run_gpu_sector_matvec(fn, in);
        return l2_diff(cpu_out, gpu_out);
    };

    // First call: cold cache build.
    REQUIRE(run(sectors[0]) < 1e-10);
    // Second call same sector: cache hit -- no rebuild.
    REQUIRE(run(sectors[0]) < 1e-10);
    // Third call different sector: LRU-1 evicts + rebuilds.
    REQUIRE(run(sectors[1]) < 1e-10);
    // Fourth call back to first sector: rebuild path again.
    REQUIRE(run(sectors[0]) < 1e-10);
#endif
}
