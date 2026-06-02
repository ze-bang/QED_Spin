// =============================================================================
// test_cuda_matvec_backend.cu
//
// Validation pin for P3a of the operator-collapse refactor (Jun 2026):
// ``ed::matvec::CudaMatVecBackend<DeviceFullBasisPolicy>`` -- the CUDA-device
// SpMV strategy (GPU twin of CpuMatVecBackend) for the Full Hilbert lane.
//
// Strategy: feed the SAME ``TermView`` (six canonical SoA bins) to both the
// already-validated host ``CpuMatVecBackend<FullBasisPolicy>`` and the new
// ``CudaMatVecBackend`` and require the matvec outputs to agree to 1e-9 over
// random probe vectors. This exercises every term bin (one/two/three-body,
// diagonal + off-diagonal + mixed) and BOTH scalar lanes (complex apply +
// real apply_real). Atomic-reduction ordering makes the GPU result differ
// from the CPU result only at the ~1e-13 round-off level, so 1e-9 is a tight
// physical-equivalence bar.
//
// Standalone (no Catch2, no operator.h): builds the TermStorage by hand via
// the typed setters so the only dependencies are header-only matvec headers
// + the CUDA backend, keeping the nvcc compile self-contained and fast.
//
// Returns 0 on success, non-zero on the first failed check.
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include <ed/matvec/matvec_backend.h>        // CpuMatVecBackend, make_cpu_full_basis_backend, TermViewT
#include <ed/matvec/term_storage.h>          // TermStorage + canonical bins
#include <ed/matvec/cuda_matvec_backend.cuh>  // CudaMatVecBackend, make_cuda_full_backend
#include <ed/core/basis_utils.h>             // generateFixedSzBasis, LinIndexTable

using Complex = std::complex<double>;
using namespace ed::matvec;

namespace {

// The six canonical bin types (== Operator::DiagonalOneBody et al.).
using TV = TermViewT<DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
                     OffDiagTwoBody, ThreeBodyTerm>;

int g_failures = 0;

void expect_close(const std::vector<Complex>& a, const std::vector<Complex>& b,
                  double tol, const char* what) {
    if (a.size() != b.size()) {
        std::printf("FAIL [%s]: size mismatch %zu vs %zu\n", what, a.size(), b.size());
        ++g_failures;
        return;
    }
    double max_err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    if (max_err > tol) {
        std::printf("FAIL [%s]: max abs err %.3e > tol %.3e\n", what, max_err, tol);
        ++g_failures;
    } else {
        std::printf("PASS [%s]: max abs err %.3e\n", what, max_err);
    }
}

void expect_close_real(const std::vector<double>& a, const std::vector<double>& b,
                       double tol, const char* what) {
    double max_err = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
        max_err = std::max(max_err, std::abs(a[i] - b[i]));
    if (max_err > tol) {
        std::printf("FAIL [%s]: max abs err %.3e > tol %.3e\n", what, max_err, tol);
        ++g_failures;
    } else {
        std::printf("PASS [%s]: max abs err %.3e\n", what, max_err);
    }
}

TV make_view(const TermStorage& ts, bool is_real, double spin_l = 0.5) {
    TV tv;
    tv.diag_one    = &ts.diag_one_body;
    tv.offdiag_one = &ts.offdiag_one_body;
    tv.diag_two    = &ts.diag_two_body;
    tv.mixed_two   = &ts.mixed_two_body;
    tv.offdiag_two = &ts.offdiag_two_body;
    tv.three_body  = &ts.three_body;
    tv.spin_l      = spin_l;
    tv.is_real     = is_real;
    return tv;
}

// Heisenberg PBC ring (J=1): Sz_i Sz_j (diag two-body) + 0.5 (S+_i S-_j +
// S-_i S+_j) (off-diag two-body). Purely real => exercises apply_real too.
void add_heisenberg_pbc(TermStorage& ts, int N) {
    for (int i = 0; i < N; ++i) {
        const std::uint64_t a = static_cast<std::uint64_t>(i);
        const std::uint64_t b = static_cast<std::uint64_t>((i + 1) % N);
        ts.add_diag_two_body(a, b, Complex(1.0, 0.0));
        ts.add_offdiag_two_body(a, b, /*S+*/0, /*S-*/1, Complex(0.5, 0.0));
        ts.add_offdiag_two_body(a, b, /*S-*/1, /*S+*/0, Complex(0.5, 0.0));
    }
}

// A richer COMPLEX term set exercising every bin (transverse + DM-like +
// three-body), with complex couplings so the real fast path is bypassed.
void add_complex_mixed(TermStorage& ts, int N) {
    for (int i = 0; i < N; ++i) {
        const std::uint64_t a = static_cast<std::uint64_t>(i);
        const std::uint64_t b = static_cast<std::uint64_t>((i + 1) % N);
        const std::uint64_t c = static_cast<std::uint64_t>((i + 2) % N);
        ts.add_diag_one_body(a, Complex(0.3, 0.0));                       // field h Sz
        ts.add_offdiag_one_body(a, /*S+*/0, Complex(0.2, 0.1));           // transverse
        ts.add_offdiag_one_body(a, /*S-*/1, Complex(0.2, -0.1));          // h.c.
        ts.add_diag_two_body(a, b, Complex(0.7, 0.0));
        ts.add_mixed_two_body(a, b, /*flip S+*/0, Complex(0.15, 0.05));   // Sz_a S+_b
        ts.add_mixed_two_body(a, b, /*flip S-*/1, Complex(0.15, -0.05));
        ts.add_offdiag_two_body(a, b, 0, 1, Complex(0.4, 0.2));           // S+_a S-_b
        ts.add_offdiag_two_body(a, b, 1, 0, Complex(0.4, -0.2));          // h.c.
        ts.add_three_body(/*Sz*/2, a, /*S+*/0, b, /*S-*/1, c, Complex(0.1, 0.03));
    }
}

std::vector<Complex> random_complex_vec(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<Complex> v(n);
    for (auto& z : v) z = Complex(d(rng), d(rng));
    return v;
}

std::vector<double> random_real_vec(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> v(n);
    for (auto& x : v) x = d(rng);
    return v;
}

}  // namespace

int main() {
    int dev_count = 0;
    if (cudaGetDeviceCount(&dev_count) != cudaSuccess || dev_count == 0) {
        std::printf("SKIP: no CUDA device available\n");
        return 0;  // not a failure on CPU-only CI
    }

    const int N = 8;                       // full Hilbert dim = 256
    const std::size_t dim = 1ULL << N;
    const double spin_l = 0.5;

    // --- Case 1: real Heisenberg, complex apply -----------------------------
    {
        TermStorage ts;
        add_heisenberg_pbc(ts, N);
        TV tv = make_view(ts, /*is_real=*/true, spin_l);

        auto cpu = make_cpu_full_basis_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(N);
        auto gpu = make_cuda_full_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(N, spin_l);

        for (int p = 0; p < 3; ++p) {
            auto x = random_complex_vec(dim, 1000 + p);
            std::vector<Complex> yc(dim), yg(dim);
            cpu->apply_complex(&tv, x.data(), yc.data(), dim);
            gpu->apply_complex(&tv, x.data(), yg.data(), dim);
            expect_close(yc, yg, 1e-9, "heisenberg complex apply");
        }
    }

    // --- Case 2: real Heisenberg, REAL apply_real ---------------------------
    {
        TermStorage ts;
        add_heisenberg_pbc(ts, N);
        TV tv = make_view(ts, /*is_real=*/true, spin_l);

        auto cpu = make_cpu_full_basis_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(N);
        auto gpu = make_cuda_full_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(N, spin_l);

        for (int p = 0; p < 3; ++p) {
            auto x = random_real_vec(dim, 2000 + p);
            std::vector<double> yc(dim), yg(dim);
            cpu->apply_real(&tv, x.data(), yc.data(), dim);
            gpu->apply_real(&tv, x.data(), yg.data(), dim);
            expect_close_real(yc, yg, 1e-9, "heisenberg real apply_real");
        }
    }

    // --- Case 3: complex mixed Hamiltonian, complex apply (all bins) --------
    {
        TermStorage ts;
        add_complex_mixed(ts, N);
        TV tv = make_view(ts, /*is_real=*/false, spin_l);

        auto cpu = make_cpu_full_basis_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(N);
        auto gpu = make_cuda_full_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(N, spin_l);

        for (int p = 0; p < 3; ++p) {
            auto x = random_complex_vec(dim, 3000 + p);
            std::vector<Complex> yc(dim), yg(dim);
            cpu->apply_complex(&tv, x.data(), yc.data(), dim);
            gpu->apply_complex(&tv, x.data(), yg.data(), dim);
            expect_close(yc, yg, 1e-9, "complex-mixed complex apply");
        }
    }

    // ======================================================================
    // P3b: Fixed-Sz lane. Half-filled sector of N=8 (dim = C(8,4) = 70).
    // The CUDA backend uploads a sorted basis + open-addressing hash via
    // DeviceFixedSzBasisPolicyHolder; compare against the host
    // CpuMatVecBackend<FixedSzBasisPolicy> (Lin-index lookup). Sz-conserving
    // terms (SzSz + S+S-/S-S+) stay in the sector; any sector-leaving term
    // is dropped identically by both backends' index_of(), so CPU==GPU is a
    // valid equivalence check either way.
    // ======================================================================
    const int Nf = 8;
    const auto fixed_basis = generateFixedSzBasis(Nf, Nf / 2);  // sorted, dim 70
    const std::size_t fdim = fixed_basis.size();
    LinIndexTable lin;
    lin.build(Nf, Nf / 2, fixed_basis);

    // --- Case 4: real Heisenberg on the Sz sector, complex apply ------------
    {
        TermStorage ts;
        add_heisenberg_pbc(ts, Nf);
        TV tv = make_view(ts, /*is_real=*/true, spin_l);

        auto cpu = make_cpu_fixed_sz_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(fixed_basis, lin);
        auto gpu = make_cuda_fixed_sz_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(fixed_basis, spin_l);

        for (int p = 0; p < 3; ++p) {
            auto x = random_complex_vec(fdim, 4000 + p);
            std::vector<Complex> yc(fdim), yg(fdim);
            cpu->apply_complex(&tv, x.data(), yc.data(), fdim);
            gpu->apply_complex(&tv, x.data(), yg.data(), fdim);
            expect_close(yc, yg, 1e-9, "fixed-sz heisenberg complex apply");
        }
    }

    // --- Case 5: real Heisenberg on the Sz sector, REAL apply_real ----------
    {
        TermStorage ts;
        add_heisenberg_pbc(ts, Nf);
        TV tv = make_view(ts, /*is_real=*/true, spin_l);

        auto cpu = make_cpu_fixed_sz_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(fixed_basis, lin);
        auto gpu = make_cuda_fixed_sz_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(fixed_basis, spin_l);

        for (int p = 0; p < 3; ++p) {
            auto x = random_real_vec(fdim, 5000 + p);
            std::vector<double> yc(fdim), yg(fdim);
            cpu->apply_real(&tv, x.data(), yc.data(), fdim);
            gpu->apply_real(&tv, x.data(), yg.data(), fdim);
            expect_close_real(yc, yg, 1e-9, "fixed-sz heisenberg real apply_real");
        }
    }

    // --- Case 6: complex mixed on the Sz sector, complex apply --------------
    {
        TermStorage ts;
        add_complex_mixed(ts, Nf);
        TV tv = make_view(ts, /*is_real=*/false, spin_l);

        auto cpu = make_cpu_fixed_sz_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(fixed_basis, lin);
        auto gpu = make_cuda_fixed_sz_backend<
            DiagOneBody, OffDiagOneBody, DiagTwoBody, MixedTwoBody,
            OffDiagTwoBody, ThreeBodyTerm>(fixed_basis, spin_l);

        for (int p = 0; p < 3; ++p) {
            auto x = random_complex_vec(fdim, 6000 + p);
            std::vector<Complex> yc(fdim), yg(fdim);
            cpu->apply_complex(&tv, x.data(), yc.data(), fdim);
            gpu->apply_complex(&tv, x.data(), yg.data(), fdim);
            expect_close(yc, yg, 1e-9, "fixed-sz complex-mixed complex apply");
        }
    }

    if (g_failures == 0) {
        std::printf("ALL CUDA MATVEC BACKEND TESTS PASSED\n");
        return 0;
    }
    std::printf("%d CHECK(S) FAILED\n", g_failures);
    return 1;
}
