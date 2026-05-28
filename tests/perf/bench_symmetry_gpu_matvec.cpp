// =============================================================================
// tests/perf/bench_symmetry_gpu_matvec.cpp
//
// Phase I of the "Close CPU / GPU Gaps Across Workflows" plan
// (May 2026): targeted microbench for the streaming-symmetry GPU
// mirror matvec.
//
// What this measures
// ------------------
//   * Per-sector ``launch_symmetry_matvec`` wall-clock (ms / matvec)
//   * Per-sector CPU ``StreamingSymmetryOperator::applySymmetrized``
//     wall-clock (ms / matvec, reference)
// for representative (N, |G|, dim) tuples on Heisenberg + Z_N
// translation symmetry, with and without the V2 small-win bundle
// (env ``ED_GPU_SYMMETRY_MIRROR_V2``).
//
// Acceptance bar (from the plan):
//   GPU >= 2x CPU on sector dim >= 8k.
//
// Output
// ------
// Tab-separated rows on stdout so the wrapper script in
// docs/perf/ can ingest them directly. Each row is:
//     N \t sector_idx \t dim \t backend \t v2 \t reps \t ms_per_matvec
//
// Usage
// -----
//   bench_symmetry_gpu_matvec [--reps R] [--N N1,N2,...]
//
// Defaults: reps=50, N=12,14
// =============================================================================

#include <ed/core/streaming_symmetry.h>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include <cuComplex.h>
#endif

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <system_error>
#include <unistd.h>
#include <vector>

using Complex = std::complex<double>;

namespace {

// -----------------------------------------------------------------
// Z_N translation fixture writer (identical layout to
// tests/unit/test_streaming_symmetry_gpu_mirror.cpp -- duplicated
// here to keep the bench harness self-contained).
// -----------------------------------------------------------------
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

std::vector<Complex> random_unit_vector(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    std::vector<Complex> v(n);
    double norm2 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = Complex(nd(rng), nd(rng));
        norm2 += std::norm(v[i]);
    }
    const double inv = 1.0 / std::sqrt(norm2);
    for (auto& z : v) z *= inv;
    return v;
}

std::string make_scratch_dir(const std::string& tag, const std::string& sub) {
    const char* root = std::getenv("ED_BENCH_TMP_DIR");
    std::string base = root ? root : "/tmp";
    base += "/bench_symmetry_gpu_matvec/" + tag + "_" + sub + "_" +
            std::to_string(static_cast<std::uint64_t>(::getpid()));
    std::error_code ec;
    std::filesystem::create_directories(base, ec);
    return base;
}

double bench_cpu_ms_per_matvec(StreamingSymmetryOperator& op,
                               std::size_t sector_idx,
                               int reps,
                               std::uint64_t seed)
{
    const std::size_t sd = op.getSectorDimension(sector_idx);
    if (sd == 0) return 0.0;
    auto in  = random_unit_vector(sd, seed);
    std::vector<Complex> out(sd, Complex(0.0, 0.0));

    // Warmup.
    for (int w = 0; w < std::max(1, reps / 20); ++w) {
        op.applySymmetrized(sector_idx, in.data(), out.data());
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        op.applySymmetrized(sector_idx, in.data(), out.data());
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();
    return ms_total / static_cast<double>(reps);
}

#ifdef WITH_CUDA

double bench_gpu_ms_per_matvec(StreamingSymmetryOperator& op,
                               std::size_t sector_idx,
                               int reps,
                               std::uint64_t seed)
{
    const std::size_t sd = op.getSectorDimension(sector_idx);
    if (sd == 0) return 0.0;
    auto in  = random_unit_vector(sd, seed);

    cuDoubleComplex* d_in  = nullptr;
    cuDoubleComplex* d_out = nullptr;
    cudaMalloc(&d_in,  sd * sizeof(cuDoubleComplex));
    cudaMalloc(&d_out, sd * sizeof(cuDoubleComplex));
    cudaMemcpy(d_in, in.data(), sd * sizeof(cuDoubleComplex),
               cudaMemcpyHostToDevice);

    auto view = op.sector(sector_idx);
    auto fn   = view->bind_cuda();

    // Warmup.
    for (int w = 0; w < std::max(1, reps / 20); ++w) {
        fn(reinterpret_cast<const Complex*>(d_in),
           reinterpret_cast<Complex*>(d_out), sd);
    }
    cudaDeviceSynchronize();

    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
        fn(reinterpret_cast<const Complex*>(d_in),
           reinterpret_cast<Complex*>(d_out), sd);
    }
    cudaDeviceSynchronize();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms_total = std::chrono::duration<double, std::milli>(t1 - t0).count();

    cudaFree(d_in);
    cudaFree(d_out);
    return ms_total / static_cast<double>(reps);
}

bool cuda_runtime_available() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

#endif  // WITH_CUDA

} // namespace

int main(int argc, char** argv)
{
    int reps = 50;
    std::vector<int> Ns = {12, 14};
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--reps" && i + 1 < argc) {
            reps = std::atoi(argv[++i]);
        } else if (arg == "--N" && i + 1 < argc) {
            Ns.clear();
            std::string s = argv[++i];
            std::stringstream ss(s);
            std::string item;
            while (std::getline(ss, item, ',')) {
                if (!item.empty()) Ns.push_back(std::atoi(item.c_str()));
            }
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: bench_symmetry_gpu_matvec [--reps R] [--N N1,N2,...]\n"
                << "Defaults: --reps 50 --N 12,14\n"
                << "Env: ED_GPU_SYMMETRY_MIRROR_V2=1 enables the Phase I\n"
                << "     small-win bundle (threads_per_block + memset stream).\n";
            return 0;
        }
    }

    const char* v2_env = std::getenv("ED_GPU_SYMMETRY_MIRROR_V2");
    const bool v2 = v2_env != nullptr && v2_env[0] != '\0' && v2_env[0] != '0';

    std::cout << "# bench_symmetry_gpu_matvec\n"
              << "# reps=" << reps
              << " v2=" << (v2 ? 1 : 0) << "\n"
              << "# N\tsector\tdim\tbackend\tv2\treps\tms_per_matvec\n";

    for (int N : Ns) {
        std::string dir = make_scratch_dir("Z_N", "N" + std::to_string(N));
        write_zN_translation_fixtures(dir, N);
        auto op = build_heisenberg_pbc_streaming(static_cast<uint64_t>(N), 1.0);
        try {
            op->generateSymmetrySectorsStreaming(dir);
        } catch (const std::exception& e) {
            std::cerr << "skip N=" << N << ": " << e.what() << "\n";
            continue;
        }

        for (std::size_t s = 0; s < op->getNumSectors(); ++s) {
            const std::size_t sd = op->getSectorDimension(s);
            if (sd == 0) continue;
            const std::uint64_t seed = static_cast<std::uint64_t>(N) * 1000ULL +
                                       static_cast<std::uint64_t>(s) * 31ULL +
                                       17;

            const double cpu_ms = bench_cpu_ms_per_matvec(*op, s, reps, seed);
            std::cout << N << "\t" << s << "\t" << sd << "\tcpu\t"
                      << (v2 ? 1 : 0) << "\t" << reps << "\t"
                      << std::setprecision(6) << cpu_ms << "\n";

#ifdef WITH_CUDA
            if (cuda_runtime_available()) {
                const double gpu_ms = bench_gpu_ms_per_matvec(*op, s, reps, seed);
                std::cout << N << "\t" << s << "\t" << sd << "\tgpu\t"
                          << (v2 ? 1 : 0) << "\t" << reps << "\t"
                          << std::setprecision(6) << gpu_ms << "\n";
            }
#endif
            std::cout.flush();
        }
    }
    return 0;
}
