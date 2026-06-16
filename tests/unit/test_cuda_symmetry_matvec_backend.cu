// =============================================================================
// test_cuda_symmetry_matvec_backend
//
// Validation pin for P3c of the operator-collapse refactor: the CUDA
// symmetry-projected device lane,
// ``ed::matvec::make_cuda_symmetry_backend`` +
// ``CudaMatVecBackend<DeviceSymmetryBasisPolicy>``, backed by
// ``DeviceSymmetryBasisPolicyHolder`` (orbit CSR + pre-baked projection
// hash uploaded to the GPU).
//
// Proves the GPU symmetry matvec reproduces the host
// ``make_cpu_symmetry_backend`` (itself pinned against the legacy
// ``StreamingSymmetryOperator::applySymmetrized`` to 1e-12) across every
// momentum sector of the Heisenberg Z_N ring (N=6), including the complex
// k != 0, pi sectors that exercise the symmetry weighting's imaginary part.
//
// Standalone (plain main, no Catch2): nvcc compiles the host symmetry
// machinery (projector / sector_basis) as host code, exactly as
// src/symmetry/streaming_symmetry_gpu_mirror.cu does. The test skips with
// exit 0 if no CUDA device is present.
// =============================================================================

#ifdef WITH_CUDA

#include <ed/core/operator.h>
#include <ed/matvec/symmetry_matvec_backend.h>
#include <ed/matvec/cuda_matvec_backend.cuh>
#include <ed/matvec/term_storage.h>
#include <ed/symmetry/projector.h>
#include <ed/symmetry/sector_basis.h>
#include <ed/symmetry/subspace.h>

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

std::vector<std::uint64_t>
enumerate_orbit_reps(const SymmetryGroupInfo& info, int N) {
    std::vector<std::uint64_t> reps;
    const std::uint64_t dim = (1ULL << N);
    for (std::uint64_t s = 0; s < dim; ++s) {
        std::uint64_t mn = s;
        for (std::size_t g = 0; g < info.max_clique.size(); ++g) {
            mn = std::min(mn, applyPermutation(s, info.max_clique[g]));
        }
        if (mn == s) reps.push_back(s);
    }
    return reps;
}

// Full-Hilbert Operator carrying just the Heisenberg term list; we only use
// its ``transform_data_`` / ``three_body_data_`` to feed TermStorage. The
// per-sector basis is built independently via SectorBasis::build.
std::unique_ptr<Operator>
build_heisenberg_pbc_full(std::uint64_t N, double J) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op->addTwoBodyTerm(2, i, 2, j, J_real);
        op->addTwoBodyTerm(0, i, 1, j, J_half);
        op->addTwoBodyTerm(1, i, 0, j, J_half);
    }
    return op;
}

using TermView_t = ed::matvec::TermViewT<
    ed::matvec::DiagOneBody, ed::matvec::OffDiagOneBody,
    ed::matvec::DiagTwoBody, ed::matvec::MixedTwoBody,
    ed::matvec::OffDiagTwoBody, ed::matvec::ThreeBodyTerm>;

TermView_t make_term_view(const ed::matvec::TermStorage& soa,
                          double spin_l, bool is_real) {
    TermView_t tv;
    tv.diag_one    = &soa.diag_one_body;
    tv.offdiag_one = &soa.offdiag_one_body;
    tv.diag_two    = &soa.diag_two_body;
    tv.mixed_two   = &soa.mixed_two_body;
    tv.offdiag_two = &soa.offdiag_two_body;
    tv.three_body  = &soa.three_body;
    tv.spin_l      = spin_l;
    tv.is_real     = is_real;
    return tv;
}

std::vector<Complex> random_complex_vec(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<Complex> v(n);
    for (auto& z : v) z = Complex(dist(rng), dist(rng));
    return v;
}

// Extract the orbit CSR a SymmetryBasisPolicy exposes into the plain
// host arrays make_cuda_symmetry_backend consumes.
void extract_orbit_csr(const ed::matvec::basis::SymmetryBasisPolicy& policy,
                       std::vector<std::uint32_t>&        offsets,
                       std::vector<std::uint64_t>&        elements,
                       std::vector<std::complex<double>>& coefficients,
                       std::vector<double>&               norms) {
    const auto& states = policy.sector->basis_states;
    const std::size_t dim = states.size();
    offsets.assign(dim + 1, 0u);
    elements.clear();
    coefficients.clear();
    norms.assign(dim, 0.0);
    for (std::size_t k = 0; k < dim; ++k) {
        norms[k] = states[k].norm;
        const std::size_t m = states[k].orbit_elements.size();
        for (std::size_t j = 0; j < m; ++j) {
            elements.push_back(states[k].orbit_elements[j]);
            coefficients.push_back(states[k].orbit_coefficients[j]);
        }
        offsets[k + 1] = static_cast<std::uint32_t>(elements.size());
    }
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

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::cout << "[skip] no CUDA device available\n";
        return 0;
    }

    const int N = 6;
    std::string dir = (std::filesystem::temp_directory_path() /
                       "ed_cuda_symmetry_test").string();
    {
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
    }
    write_zN_translation_fixtures(dir, N);

    auto full_op = build_heisenberg_pbc_full(N, 1.0);

    ed::matvec::TermStorage soa;
    ed::matvec::TermStorage::classify_route(
        soa, full_op->transform_data_, full_op->three_body_data_,
        [](const Complex& c) { return c; });
    const TermView_t tv = make_term_view(soa, /*spin_l=*/0.5, /*is_real=*/true);

    SymmetryGroupInfo info;
    info.loadFromDirectory(dir);
    const ed::symmetry::FullSpaceSubspace full(static_cast<std::uint64_t>(N));
    const ed::symmetry::SpatialProjector  spatial(info);
    const std::vector<std::uint64_t> reps = enumerate_orbit_reps(info, N);

    for (std::size_t s = 0; s < info.sectors.size(); ++s) {
        ed::symmetry::SectorBasis sb = ed::symmetry::SectorBasis::build(
            full, spatial,
            info.sectors[s].quantum_numbers,
            info.sectors[s].phase_factors,
            reps, /*sector_id=*/s);

        const std::size_t sd = sb.policy().sector->basis_states.size();
        if (sd == 0) continue;

        auto cpu_backend = ed::matvec::make_cpu_symmetry_backend<
            ed::matvec::DiagOneBody, ed::matvec::OffDiagOneBody,
            ed::matvec::DiagTwoBody, ed::matvec::MixedTwoBody,
            ed::matvec::OffDiagTwoBody, ed::matvec::ThreeBodyTerm>(sb.policy());

        std::vector<std::uint32_t>        offsets;
        std::vector<std::uint64_t>        elements;
        std::vector<std::complex<double>> coefficients;
        std::vector<double>               norms;
        extract_orbit_csr(sb.policy(), offsets, elements, coefficients, norms);

        auto gpu_backend = ed::matvec::make_cuda_symmetry_backend<
            ed::matvec::DiagOneBody, ed::matvec::OffDiagOneBody,
            ed::matvec::DiagTwoBody, ed::matvec::MixedTwoBody,
            ed::matvec::OffDiagTwoBody, ed::matvec::ThreeBodyTerm>(
            offsets, elements, coefficients, norms,
            sb.policy().group_norm, /*spin_l=*/0.5);

        for (int probe = 0; probe < 2; ++probe) {
            std::vector<Complex> x(sd);
            if (probe == 0) {
                std::fill(x.begin(), x.end(), Complex(1.0, 0.0));
            } else {
                x = random_complex_vec(sd, (s + 7) * 99991ULL + 13);
            }

            std::vector<Complex> y_cpu(sd, Complex(0.0, 0.0));
            std::vector<Complex> y_gpu(sd, Complex(0.0, 0.0));

            cpu_backend->apply_complex(&tv, x.data(), y_cpu.data(), sd);
            gpu_backend->apply_complex(&tv, x.data(), y_gpu.data(), sd);

            expect_close(y_gpu, y_cpu,
                         "sector " + std::to_string(s) +
                         " probe " + std::to_string(probe) +
                         " (dim " + std::to_string(sd) + ")");
        }
    }

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);

    if (g_failures == 0) {
        std::cout << "ALL CUDA SYMMETRY CHECKS PASSED\n";
        return 0;
    }
    std::cerr << g_failures << " CUDA symmetry check(s) FAILED\n";
    return 1;
}

#else  // !WITH_CUDA

int main() { return 0; }

#endif  // WITH_CUDA
