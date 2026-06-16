// =============================================================================
// test_sector_operator_gpu
//
// Phase A acceptance pin for the operator-collapse GPU-parity work
// (Jun 2026): proves that a standalone ``ed::symmetry::SectorOperator``
// (built straight from the directory loader via
// ``ed::make_sector_operators`` -> the P5 sector-set enumerator, with NO
// ``StreamingSymmetryOperator`` materialised) is backend-complete on the
// GPU lane, exactly matching its own CPU ``apply``.
//
// For the Heisenberg Z_N ring (N=6, J=1) the test:
//   * Builds the SectorOperators for BOTH the full-Hilbert lane and the
//     fixed-Sz (n_up=3) lane.
//   * For every sector asserts geometry().supports_device_matvec is set
//     (the contract that routes select_backend to CudaBackend).
//   * Runs the sector matvec on the CPU (``apply``) and on the GPU
//     (``bind_cuda()`` -> make_sector_matvec_gpu -> unified device kernel)
//     for a constant probe and a random complex probe, and asserts the
//     two outputs agree to < 1e-9.
//
// Standalone (plain main, no Catch2): nvcc compiles the host symmetry
// machinery as host code, exactly as test_cuda_symmetry_matvec_backend.cu
// and streaming_symmetry_gpu_mirror.cu do. Skips with exit 0 when no CUDA
// device is present.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/core/make_operator.h>
#include <ed/symmetry/sector_operator.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <system_error>
#include <vector>

namespace {

using Complex = std::complex<double>;

int g_failures = 0;

std::vector<int> translation_perm(int N, int shift) {
    std::vector<int> p(N);
    for (int i = 0; i < N; ++i) p[i] = ((i - shift) % N + N) % N;
    return p;
}

// Z_N translation automorphism metadata consumed by
// SymmetryGroupInfo::loadFromDirectory (mirrors the e2e fixture writer).
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

// 6-site periodic Heisenberg chain directory deck (Trans.dat empty +
// InterAll.dat three terms per periodic bond).
void write_heisenberg_directory(const std::string& dir, int N, double J = 1.0) {
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    {
        std::ofstream f(dir + "/Trans.dat");
        f << "===================\n";
        f << "num       0\n";
        f << "===================\n";
        f << "===================\n";
        f << "===================\n";
    }
    {
        std::ofstream f(dir + "/InterAll.dat");
        const std::uint64_t nlines = 3ULL * static_cast<std::uint64_t>(N);
        f << "===================\n";
        f << "num       " << nlines << "\n";
        f << "===================\n";
        f << "===================\n";
        f << "===================\n";
        for (int i = 0; i < N; ++i) {
            const int j = (i + 1) % N;
            f << "        2         " << i
              << "           2         " << j
              << "    " << J << "    0.000000\n";
            f << "        0         " << i
              << "           1         " << j
              << "    " << (0.5 * J) << "    0.000000\n";
            f << "        1         " << i
              << "           0         " << j
              << "    " << (0.5 * J) << "    0.000000\n";
        }
    }
}

std::vector<Complex> random_complex_vec(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> v(n);
    for (auto& z : v) z = Complex(dist(rng), dist(rng));
    return v;
}

void expect_close(const std::vector<Complex>& a, const std::vector<Complex>& b,
                  const std::string& what, double tol = 1e-9) {
    double max_diff = 0.0, ref_scale = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
        ref_scale = std::max(ref_scale, std::abs(b[i]));
    }
    if (max_diff >= tol * (1.0 + ref_scale)) {
        std::cerr << "  FAIL " << what << " max_diff=" << max_diff
                  << " ref_scale=" << ref_scale << "\n";
        ++g_failures;
    } else {
        std::cout << "  ok   " << what << " max_diff=" << max_diff << "\n";
    }
}

// Run a bind_cuda() MatvecFn (DEVICE-pointer contract) end-to-end:
// allocate device buffers, copy the input down, invoke, copy back.
std::vector<Complex>
run_gpu_matvec(const ed::LinearOperator::MatvecFn& fn,
               const std::vector<Complex>& in) {
    const std::size_t dim = in.size();
    std::vector<Complex> out(dim, Complex(0.0, 0.0));
    if (dim == 0) return out;

    cuDoubleComplex* d_in  = nullptr;
    cuDoubleComplex* d_out = nullptr;
    if (cudaMalloc(&d_in,  dim * sizeof(cuDoubleComplex)) != cudaSuccess ||
        cudaMalloc(&d_out, dim * sizeof(cuDoubleComplex)) != cudaSuccess) {
        std::cerr << "  FAIL cudaMalloc\n";
        ++g_failures;
        return out;
    }
    cudaMemcpy(d_in, in.data(), dim * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    fn(reinterpret_cast<const Complex*>(d_in),
       reinterpret_cast<Complex*>(d_out),
       dim);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, dim * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

ed::OperatorSpec heisenberg_spec(const std::string& dir, int N) {
    ed::OperatorSpec spec;
    spec.source             = ed::DirectoryPath{dir};
    spec.num_sites          = static_cast<std::uint64_t>(N);
    spec.spin_l             = 0.5f;
    spec.streaming_symmetry = true;
    return spec;
}

// Exercise one lane (full or fixed-Sz): for every SectorOperator assert
// the GPU bind_cuda() matvec matches the CPU apply on two probes.
void check_lane(const std::string& lane,
                std::vector<std::unique_ptr<ed::symmetry::SectorOperator>>& ops) {
    if (ops.empty()) {
        std::cerr << "  FAIL " << lane << ": no sector operators built\n";
        ++g_failures;
        return;
    }
    for (std::size_t s = 0; s < ops.size(); ++s) {
        const auto& op = ops[s];
        const std::size_t sd = op->dim();
        if (sd == 0) continue;

        if (!op->geometry().supports_device_matvec) {
            std::cerr << "  FAIL " << lane << " sector " << s
                      << ": geometry().supports_device_matvec is false\n";
            ++g_failures;
            continue;
        }

        auto fn = op->bind_cuda();
        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) {
                std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            } else {
                x = random_complex_vec(sd, (s + 5) * 60013ULL + 7);
            }

            std::vector<Complex> y_cpu(sd, Complex(0.0, 0.0));
            op->apply(x.data(), y_cpu.data(), sd);
            std::vector<Complex> y_gpu = run_gpu_matvec(fn, x);

            expect_close(y_gpu, y_cpu,
                         lane + " sector " + std::to_string(s) +
                         " probe " + std::to_string(probe) +
                         " (dim " + std::to_string(sd) + ")");
        }
    }
}

// Operator-collapse Phase 3 (Jun 2026): the CSR-free lazy-rep lane of
// ``make_sector_operators``. Builds the SAME fixed-Sz system twice -- once
// forced eager (orbit-CSR matvec, the proven reference) and once forced lazy
// (``build_fixed_sz_sector_operators_lazy``) -- and asserts the lazy lane:
//   * returns rep-lazy operators whose dims match the eager lane sector-for-
//     sector WITHOUT materialising the host orbit CSR at construction;
//   * keeps the host orbit CSR un-materialised after ``bind_cuda`` (GPU path);
//   * reproduces the eager CSR CPU reference matvec on the GPU to < 1e-9.
// This pins that the direct-enumeration lazy path (no streaming carrier)
// builds bit-identical physics to the eager path.
void check_lazy_lane(const std::string& dir, int N, int n_up) {
    const std::string lane = "lazy(n_up=" + std::to_string(n_up) + ")";
    ed::OperatorSpec spec = heisenberg_spec(dir, N);
    spec.fixed_sz = n_up;

    setenv("ED_SYM_LAZY_SECTORS", "0", /*overwrite=*/1);
    auto eager = ed::make_sector_operators(spec);
    setenv("ED_SYM_LAZY_SECTORS", "1", /*overwrite=*/1);
    auto lazy = ed::make_sector_operators(spec);
    unsetenv("ED_SYM_LAZY_SECTORS");

    if (eager.size() != lazy.size()) {
        std::cerr << "  FAIL " << lane << ": sector count eager="
                  << eager.size() << " lazy=" << lazy.size() << "\n";
        ++g_failures;
        return;
    }
    for (std::size_t s = 0; s < lazy.size(); ++s) {
        auto& lop = lazy[s];
        auto& eop = eager[s];
        const std::string tg = lane + " sector " + std::to_string(s);

        if (!lop->rep_lazy()) {
            std::cerr << "  FAIL " << tg << ": rep_lazy()\n";
            ++g_failures;
        }
        if (lop->dim() != eop->dim()) {
            std::cerr << "  FAIL " << tg << ": dim " << lop->dim()
                      << " vs eager " << eop->dim() << "\n";
            ++g_failures;
            continue;
        }
        if (lop->host_csr_materialized()) {
            std::cerr << "  FAIL " << tg
                      << ": host CSR materialised at construction\n";
            ++g_failures;
        }
        const std::size_t sd = lop->dim();
        if (sd == 0) continue;

        auto fn = lop->bind_cuda();
        if (lop->host_csr_materialized()) {
            std::cerr << "  FAIL " << tg
                      << ": host CSR materialised after bind_cuda (GPU path)\n";
            ++g_failures;
        }
        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            else            x = random_complex_vec(sd, (s + 9) * 70001ULL + 3);

            std::vector<Complex> y_eager_cpu(sd, Complex(0.0, 0.0));
            eop->apply(x.data(), y_eager_cpu.data(), sd);  // orbit-CSR reference
            std::vector<Complex> y_lazy_gpu = run_gpu_matvec(fn, x);

            expect_close(y_lazy_gpu, y_eager_cpu,
                         tg + " probe " + std::to_string(probe) +
                         " (dim " + std::to_string(sd) + ")");
        }
    }
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "[skip] no CUDA device available\n";
        return 0;
    }

    const int N = 6;
    std::string dir = (std::filesystem::temp_directory_path() /
                       "ed_sector_operator_gpu_test").string();
    {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
    }
    write_zN_translation_fixtures(dir, N);
    write_heisenberg_directory(dir, N, 1.0);

    std::cout << "[full lane] make_sector_operators (full Hilbert)\n";
    {
        auto ops = ed::make_sector_operators(heisenberg_spec(dir, N));
        check_lane("full", ops);
    }

    std::cout << "[fixed-Sz lane] make_sector_operators (n_up=3)\n";
    {
        ed::OperatorSpec spec = heisenberg_spec(dir, N);
        spec.fixed_sz = 3;
        auto ops = ed::make_sector_operators(spec);
        check_lane("fixedsz", ops);
    }

    std::cout << "[lazy lane] make_sector_operators (CSR-free lazy-rep, n_up=3)\n";
    check_lazy_lane(dir, N, /*n_up=*/3);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    if (g_failures == 0) {
        std::cout << "ALL SECTOR-OPERATOR GPU CHECKS PASSED\n";
        return 0;
    }
    std::cerr << g_failures << " sector-operator GPU check(s) FAILED\n";
    return 1;
}

#else  // !WITH_CUDA

int main() { return 0; }

#endif  // WITH_CUDA
