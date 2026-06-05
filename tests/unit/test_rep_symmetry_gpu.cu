// =============================================================================
// test_rep_symmetry_gpu
//
// Acceptance pin for the "On-the-fly representative SpMV for streaming
// symmetry" plan (Jun 2026).
//
// With ``ED_GPU_SYMMETRY_REP=1`` a fixed-Sz ``ed::symmetry::SectorOperator``'s
// ``bind_cuda()`` takes the RESIDENT on-the-fly representative path
// (``apply_terms_rep_symmetry_scatter`` driven by
// ``DeviceRepSymmetryBasisPolicy`` + ``make_sector_matvec_gpu_rep``). That
// path stores NO orbit CSR and NO O(full-Sz-dim) projection table: it applies
// H to the single representative of each row and regenerates the destination
// orbit index + projection phase arithmetically from the |G| group
// permutations + the per-sector characters.
//
// This test proves the rep matvec matches the CPU ``applySymmetrized``
// reference (the orbit-CSR symmetry backend reached through ``apply()``) to
// < 1e-10 across every momentum sector of:
//   * the Heisenberg Z_6 ring, n_up = 3  (real + complex characters), and
//   * the Heisenberg Z_8 ring, n_up = 4  (|G| = 8, non-trivial complex
//     characters -- the pyrochlore-flavoured fixture).
//
// It also pins the CSR-free ``RepSectorData`` extraction: every sector's
// ``repSectorData()`` must be ``usable()`` and its reps / inv_norms must equal
// the sector basis's ``orbit_rep`` / ``1/norm`` in order.
//
// Standalone (plain main, no Catch2), mirroring test_sector_operator_gpu.cu.
// Skips with exit 0 when no CUDA device is present.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/core/make_operator.h>
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

ed::OperatorSpec heisenberg_spec(const std::string& dir, int N, int n_up) {
    ed::OperatorSpec spec;
    spec.source             = ed::DirectoryPath{dir};
    spec.num_sites          = static_cast<std::uint64_t>(N);
    spec.spin_l             = 0.5f;
    spec.streaming_symmetry = true;
    spec.fixed_sz           = n_up;
    return spec;
}

// Pin the CSR-free RepSectorData extraction against the materialised sector
// basis: usable() must hold, and reps / inv_norms must match orbit_rep / 1/norm
// in order (this is the array index the solver's in/out vectors use).
void check_rep_data(const std::string& lane,
                    const ed::symmetry::SectorOperator& op,
                    std::size_t s) {
    const auto& rd  = op.repSectorData();
    const auto& sec = op.basis().sector();
    const std::string tag = lane + " sector " + std::to_string(s);

    if (!rd.usable()) {
        std::cerr << "  FAIL " << tag << ": RepSectorData not usable()\n";
        ++g_failures;
        return;
    }
    if (rd.reps.size() != sec.basis_states.size()) {
        std::cerr << "  FAIL " << tag << ": rep count "
                  << rd.reps.size() << " != sector dim "
                  << sec.basis_states.size() << "\n";
        ++g_failures;
        return;
    }
    double max_norm_err = 0.0;
    bool   reps_match   = true;
    for (std::size_t i = 0; i < rd.reps.size(); ++i) {
        if (rd.reps[i] != sec.basis_states[i].orbit_rep) reps_match = false;
        const double inv = (sec.basis_states[i].norm > 0.0)
                               ? 1.0 / sec.basis_states[i].norm : 0.0;
        max_norm_err = std::max(max_norm_err,
                                std::abs(rd.inv_norms[i] - inv));
    }
    if (!reps_match) {
        std::cerr << "  FAIL " << tag << ": reps disagree with sector basis\n";
        ++g_failures;
    } else if (max_norm_err > 1e-12) {
        std::cerr << "  FAIL " << tag << ": inv_norm err " << max_norm_err << "\n";
        ++g_failures;
    } else {
        std::cout << "  ok   " << tag << " rep-data (dim "
                  << rd.reps.size() << ")\n";
    }
}

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

        check_rep_data(lane, *op, s);

        if (!op->geometry().supports_device_matvec) {
            std::cerr << "  FAIL " << lane << " sector " << s
                      << ": geometry().supports_device_matvec is false\n";
            ++g_failures;
            continue;
        }

        // ED_GPU_SYMMETRY_REP=1 (set in main) routes this to the on-the-fly
        // representative kernel.
        auto fn = op->bind_cuda();
        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) {
                std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            } else {
                x = random_complex_vec(sd, (s + 11) * 70001ULL + 3);
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

void run_ring(int N, int n_up) {
    const std::string lane = "Z" + std::to_string(N) + " n_up=" +
                             std::to_string(n_up);
    std::cout << "[" << lane << "] on-the-fly representative GPU matvec\n";
    std::string dir = (std::filesystem::temp_directory_path() /
                       ("ed_rep_symmetry_gpu_test_N" + std::to_string(N)))
                          .string();
    {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
    }
    write_zN_translation_fixtures(dir, N);
    write_heisenberg_directory(dir, N, 1.0);

    auto ops = ed::make_sector_operators(heisenberg_spec(dir, N, n_up));
    check_lane(lane, ops);

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

    // Engage the on-the-fly representative path (must be set before the first
    // bind_cuda(), which latches the gate via a function-local static).
    setenv("ED_GPU_SYMMETRY_REP", "1", /*overwrite=*/1);

    run_ring(/*N=*/6, /*n_up=*/3);
    run_ring(/*N=*/8, /*n_up=*/4);  // |G|=8 fixture, complex characters

    if (g_failures == 0) {
        std::cout << "ALL REP-SYMMETRY GPU CHECKS PASSED\n";
        return 0;
    }
    std::cerr << g_failures << " rep-symmetry GPU check(s) FAILED\n";
    return 1;
}

#else  // !WITH_CUDA

int main() { return 0; }

#endif  // WITH_CUDA
