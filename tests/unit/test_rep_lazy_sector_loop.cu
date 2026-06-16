// =============================================================================
// test_rep_lazy_sector_loop
//
// Acceptance pin for the CSR-free lazy sector-loop optimisation
// ("scan other region for a 32 sites mtpq run with symm", Jun 2026).
//
// The production streaming-symmetry sector loop
// (``ed::core::StreamingSymmetryHandle::sector(k)``) used to FULLY materialise
// each sector's host orbit CSR (orbit_elements + orbit_coefficients + the
// state->orbit ``SortedUint64Index`` lookup) -- ~24 GiB/sector at N=32 -- even
// though the GPU on-the-fly representative matvec needs NONE of it.
//
// With ``ED_GPU_SYMMETRY_REP`` on (default), the fixed-Sz handle now returns a
// CSR-free *lazy* ``SectorOperator`` (``make_rep_sector_operator_lazy``) that:
//   * knows its ``dim`` up-front (Pass 1.5, no materialisation),
//   * builds the CSR-free RepSectorData on demand for ``bind_cuda`` (GPU), and
//   * materialises the host orbit CSR ONLY if a CPU ``apply`` is invoked.
//
// This test proves, across every momentum sector of the Heisenberg Z_6 (n_up=3)
// and Z_8 (n_up=4, |G|=8 complex characters) rings driven THROUGH the handle:
//   1. the handle returns a rep-lazy operator whose ``dim`` matches the
//      sector tag WITHOUT materialising the host orbit CSR;
//   2. ``bind_cuda`` engages the GPU rep path and STILL does not materialise
//      the host orbit CSR (the host-memory invariant the optimisation rests on);
//   3. the GPU rep matvec matches the CPU ``apply`` reference to < 1e-10; and
//   4. the CPU ``apply`` fallback lazily materialises the host orbit CSR
//      (so the path stays correct on CPU-only runs).
//
// Standalone (plain main, no Catch2). Skips with exit 0 when no CUDA device.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/core/make_operator.h>
#include <ed/core/sector_loop.h>
#include <ed/symmetry/sector_operator.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
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
                  const std::string& what, double tol = 1e-10) {
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

void check(bool cond, const std::string& what) {
    if (cond) {
        std::cout << "  ok   " << what << "\n";
    } else {
        std::cerr << "  FAIL " << what << "\n";
        ++g_failures;
    }
}

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
       reinterpret_cast<Complex*>(d_out), dim);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), d_out, dim * sizeof(cuDoubleComplex),
               cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

ed::OperatorSpec heisenberg_spec(const std::string& dir, int N, int n_up) {
    ed::OperatorSpec spec;
    spec.source             = ed::DirectoryPath{dir};
    spec.num_sites          = static_cast<std::uint64_t>(N);
    spec.spin_l             = 0.5f;
    spec.streaming_symmetry = true;
    spec.fixed_sz           = n_up;
    return spec;
}

void run_ring(int N, int n_up) {
    const std::string lane = "Z" + std::to_string(N) + " n_up=" +
                             std::to_string(n_up);
    std::cout << "[" << lane << "] CSR-free lazy sector loop\n";
    std::string dir = (std::filesystem::temp_directory_path() /
                       ("ed_rep_lazy_loop_test_N" + std::to_string(N)))
                          .string();
    {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
    }
    write_zN_translation_fixtures(dir, N);
    write_heisenberg_directory(dir, N, 1.0);

    // Build the fixed-Sz symmetry sector set and drive it THROUGH the
    // carrier-free SectorSetView -- exactly the production sector-loop path
    // the workflows use after the operator-collapse Phase 3 carrier removal.
    ed::core::SectorSetView view(
        ed::make_sector_operators_tagged(heisenberg_spec(dir, N, n_up)));

    if (!ed::core::rep_lazy_sector_path_enabled()) {
        std::cerr << "  FAIL " << lane
                  << ": rep_lazy_sector_path_enabled() is false\n";
        ++g_failures;
    }

    const auto sectors =
        ed::core::filter_sectors(view.num_sectors(), /*selected=*/{});

    for (std::size_t k : sectors) {
        const auto tag = view.sector_tag(k);              // CSR-free dim
        if (tag.sector_dim == 0) continue;

        ed::symmetry::SectorOperator* op = view.sector(k);
        const std::string tg = lane + " sector " + std::to_string(k);

        if (!op) {
            std::cerr << "  FAIL " << tg << ": not a SectorOperator\n";
            ++g_failures;
            continue;
        }

        // (1) Rep-lazy operator, dim known, NO host CSR materialised yet.
        check(op->rep_lazy(), tg + ": rep_lazy()");
        check(op->dim() == tag.sector_dim,
              tg + ": dim==tag (" + std::to_string(op->dim()) + ")");
        check(!op->host_csr_materialized(),
              tg + ": host CSR not materialised at construction");
        check(op->geometry().supports_device_matvec,
              tg + ": supports_device_matvec");

        const std::size_t sd = op->dim();

        // (2) GPU rep path must NOT materialise the host orbit CSR.
        auto fn = op->bind_cuda();
        check(!op->host_csr_materialized(),
              tg + ": host CSR not materialised after bind_cuda (GPU path)");

        // (3) GPU rep matvec == CPU apply reference.
        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            else            x = random_complex_vec(sd, (k + 5) * 90011ULL + 7);

            std::vector<Complex> y_gpu = run_gpu_matvec(fn, x);

            std::vector<Complex> y_cpu(sd, Complex(0.0, 0.0));
            op->apply(x.data(), y_cpu.data(), sd);  // (4) lazily builds CSR

            expect_close(y_gpu, y_cpu,
                         tg + " probe " + std::to_string(probe) +
                         " (dim " + std::to_string(sd) + ")");
        }

        // (4) CPU apply must have lazily materialised the host orbit CSR.
        check(op->host_csr_materialized(),
              tg + ": host CSR materialised by CPU apply fallback");
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "[skip] no CUDA device available\n";
        return 0;
    }

    // Pin the full CSR-free lazy configuration this acceptance test models.
    // Three independent knobs all have to line up; the tiny Z_6 / Z_8 rings
    // below would otherwise NOT trigger lazy mode on their own:
    //
    //   * ED_SYM_LAZY_SECTORS=1 forces build_fixed_sz_sector_operators_lazy
    //     into lazy generation (Pass 1.5 dims, no eager orbit CSR). Without
    //     it the ~4 GiB memory-budget heuristic keeps these small systems
    //     eager, so SectorSetView::sector(k) would hand out a CSR-backed
    //     adopt operator and checks (1)-(3) (rep_lazy / no host CSR) would
    //     fail.
    //   * ED_GPU_SYMMETRY_REP=1 engages the GPU on-the-fly representative
    //     matvec in bind_cuda() (check (2): no host CSR on the GPU path).
    //   * ED_SYM_REP=0 keeps the CPU `apply` on the orbit-CSR backend so the
    //     CPU fallback lazily MATERIALISES the host CSR (check (4)). With the
    //     CPU rep kernel on (its default) `apply` would also run CSR-free and
    //     `host_csr_materialized()` would stay false, contradicting (4).
    //
    // All three are latched via function-local statics / read at generation
    // time, so they must be set before the first operator is built.
    setenv("ED_SYM_LAZY_SECTORS", "1", /*overwrite=*/1);
    setenv("ED_GPU_SYMMETRY_REP", "1", /*overwrite=*/1);
    setenv("ED_SYM_REP", "0", /*overwrite=*/1);

    run_ring(/*N=*/6, /*n_up=*/3);
    run_ring(/*N=*/8, /*n_up=*/4);  // |G|=8 fixture, complex characters

    if (g_failures == 0) {
        std::cout << "ALL REP-LAZY SECTOR-LOOP CHECKS PASSED\n";
        return 0;
    }
    std::cerr << g_failures << " rep-lazy sector-loop check(s) FAILED\n";
    return 1;
}

#else  // !WITH_CUDA

int main() { return 0; }

#endif  // WITH_CUDA
