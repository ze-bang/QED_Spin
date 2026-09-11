// =============================================================================
// benchmarks/bench_audit_solve.cpp
//
// Wall-clock benchmark of the orchestrator ground-state lanes that the
// 2026-09 performance audit targets:
//
//   (a) eigenvalues only   (LocalDGKS3 K=1, no kept basis)
//   (b) with eigenvectors  (kept-basis FullCGS2 before the audit; two-pass
//                           Lanczos after)
//   (c) num_eigs = 2, eigenvalues only
//
// on the periodic spin-1/2 Heisenberg chain in the Sz = 0 sector.
// Prints one JSON line per case so before/after runs can be diffed.
//
//   ./build/benchmarks/bench_audit_solve 18 20 22 24
// =============================================================================

#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/construct_ham.h>
#include <ed/core/linear_operator.h>
#include <ed/core/results.h>
#include <ed/orchestrator.h>

#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

namespace {

using Complex = std::complex<double>;

std::unique_ptr<FixedSzOperator> chain(std::uint64_t N) {
    auto op = std::make_unique<FixedSzOperator>(N, 0.5f, static_cast<std::int64_t>(N / 2));
    for (std::uint64_t i = 0; i < N; ++i) {
        const std::uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.is_two_body = true;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2; t.site_index_2 = j; t.coefficient = Complex(1.0, 0.0);
        op->transform_data_.push_back(t);
        t.op_type = 0; t.site_index = i; t.op_type_2 = 1; t.site_index_2 = j; t.coefficient = Complex(0.5, 0.0);
        op->transform_data_.push_back(t);
        t.op_type = 1; t.site_index = i; t.op_type_2 = 0; t.site_index_2 = j; t.coefficient = Complex(0.5, 0.0);
        op->transform_data_.push_back(t);
    }
    return op;
}

double residual_of(const ed::LinearOperator& H, const std::vector<Complex>& v, double E) {
    std::vector<Complex> hv(v.size());
    H.apply(v.data(), hv.data(), v.size());
    double num = 0.0, den = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) {
        num += std::norm(hv[i] - E * v[i]);
        den += std::norm(v[i]);
    }
    return std::sqrt(num / den);
}

void run_case(const char* label, const ed::LinearOperator& H, std::uint64_t N,
              std::size_t num_eigs, bool vectors, int repeats) {
    double best = 1e300, first = -1.0; ed::GroundStateResult res;
    for (int r = 0; r < repeats; ++r) {
        ed::SolveOptions opts;
        opts.num_eigs = num_eigs;
        opts.tolerance = 1e-10;
        opts.compute_vectors = vectors;
        opts.method = ed::SolveMethod::Lanczos;
        opts.backend.allow_gpu = (std::getenv("BENCH_GPU") != nullptr);  // BENCH_GPU=1 => CudaBackend lane
        opts.backend.allow_mpi = false;
        const auto t0 = std::chrono::steady_clock::now();
        res = ed::workflows::solve(H, opts);
        const double ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t0).count();
        if (r == 0) first = ms;
        if (ms < best) best = ms;
    }
    double resid = -1.0;
    if (vectors && res.eigenvectors && !res.eigenvectors->host.empty())
        resid = residual_of(H, res.eigenvectors->host[0], res.eigenvalues[0]);
    std::printf("{\"case\": \"%s\", \"N\": %llu, \"dim\": %zu, \"num_eigs\": %zu, \"vectors\": %s, "
                "\"first_ms\": %.2f, \"ms\": %.2f, \"iters\": %zu, \"E0\": %.12f, \"residual\": %.3e}\n",
                label, static_cast<unsigned long long>(N), H.dim(), num_eigs, vectors ? "true" : "false",
                first, best, res.krylov.iters_done, res.eigenvalues.empty() ? 0.0 : res.eigenvalues[0], resid);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::uint64_t> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(static_cast<std::uint64_t>(std::atoll(argv[i])));
    if (sizes.empty()) sizes = {16, 18, 20, 22};
    const int repeats = std::getenv("BENCH_REPEATS") ? std::atoi(std::getenv("BENCH_REPEATS")) : 2;
    for (auto N : sizes) {
        auto op = chain(N);
        run_case("eigvals", *op, N, 1, false, repeats);
        run_case("eigvecs", *op, N, 1, true, repeats);
        run_case("eigvals2", *op, N, 2, false, repeats);
    }
    return 0;
}
