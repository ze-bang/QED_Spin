#pragma once
// =============================================================================
// examples/_shared/common.h
//
// Tiny header-only helpers shared by every cell in the
// `examples/{solve,thermal,spectral}/` tree. The point is that each
// example body can stay at ~30 lines and read line-for-line like its
// Python twin. The helpers here are:
//
//   * `heisenberg_chain(N, pbc)` -- build a J=1 spin-1/2 Heisenberg
//     Operator on N sites (PBC / OBC). Mirrors
//     `examples/_shared/common.py::heisenberg_chain`.
//
//   * `bethe_E0(N)` -- exact ground-state energy of the J=1 spin-1/2
//     Heisenberg ring (PBC) for the small N where the Bethe ansatz
//     gives a closed form. NaN for N outside the table. Used in
//     "|E_0 - E_0_Bethe|" sanity-check lines of the example bodies.
//
//   * `rank0_print(...)` -- variadic print helper that prints only on
//     MPI rank 0. Boils down to `std::cout << ...` on non-MPI builds
//     so the same example body works for CPU / GPU / MPI / MPI+GPU.
//
//   * `OperatorSpec` builder helper that wraps an in-memory Operator
//     into a fresh spec; the four `OperatorSpec` source variants
//     (InMemory / FilePaths / DirectoryPath) all flow through
//     `ed::make_operator`.
//
// Author: ed-collapse, Phase B of the "mirror examples" plan (May 2026).
// =============================================================================

#include <cmath>
#include <complex>
#include <cstdint>
#include <iostream>
#include <memory>
#include <string>
#include <utility>

#include <ed/core/make_operator.h>
#include <ed/core/operator.h>

#ifdef WITH_MPI
#  include <mpi.h>
#endif

namespace ed_example {

/// Build a J=1 spin-1/2 Heisenberg chain (`H = J * sum_<ij> S_i . S_j`).
/// `periodic` toggles PBC / OBC.
[[nodiscard]] inline std::unique_ptr<Operator>
heisenberg_chain(std::uint64_t N, bool periodic = true, double J = 1.0) {
    using Complex = std::complex<double>;
    auto op = std::make_unique<Operator>(N, /*spin=*/0.5f);
    const Complex Jz(J,      0.0);
    const Complex Jpm(0.5 * J, 0.0);
    const std::uint64_t last = periodic ? N : (N - 1);
    for (std::uint64_t i = 0; i < last; ++i) {
        const std::uint64_t j = (i + 1) % N;
        op->addTwoBodyTerm(2, i, 2, j, Jz);  // Sz Sz
        op->addTwoBodyTerm(0, i, 1, j, Jpm); // S+ S-
        op->addTwoBodyTerm(1, i, 0, j, Jpm); // S- S+
    }
    return op;
}

/// Exact ground-state energy of the J=1 spin-1/2 Heisenberg ring (PBC).
/// Hard-coded for the canonical small-N example sizes. Returns NaN for
/// values not in the table (the caller can still print the computed
/// number; the comparison just silently no-ops).
[[nodiscard]] inline double bethe_E0(std::uint64_t N) {
    // H = J * sum_<ij> S_i . S_j with J = 1, PBC. Values cross-checked
    // against `python/qed/feasibility.py::_bethe_E0` and the standard
    // Bethe-ansatz roots tabulated in Mattis (1981).
    switch (N) {
        case 2:  return -0.75;
        case 4:  return -2.0;
        case 6:  return -2.8027756377319946;
        case 8:  return -3.6510934089371783;
        case 10: return -4.515446354492385;
        case 12: return -5.387390917796387;
        case 14: return -6.263728685245183;
        case 16: return -7.142296361092491;
        default: return std::numeric_limits<double>::quiet_NaN();
    }
}

/// `true` on the rank that should produce stdout (rank 0 on MPI builds;
/// always true on serial builds).
[[nodiscard]] inline bool is_rank0() noexcept {
#ifdef WITH_MPI
    int initialized = 0;
    MPI_Initialized(&initialized);
    if (initialized) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank == 0;
    }
#endif
    return true;
}

/// Variadic rank-0 print. Cheaper than building a stringstream upstream.
template <typename... Ts>
inline void rank0_print(Ts&&... args) {
    if (!is_rank0()) return;
    (std::cout << ... << std::forward<Ts>(args));
}

/// Wrap an in-memory `Operator` into a fresh `OperatorSpec`. Saves the
/// 3-line boilerplate in every cell.
[[nodiscard]] inline ed::OperatorSpec
in_memory_spec(std::unique_ptr<Operator> op, std::uint64_t N) {
    ed::OperatorSpec spec;
    spec.source    = ed::InMemoryOperator{std::move(op)};
    spec.num_sites = N;
    spec.spin_l    = 0.5f;
    return spec;
}

/// Initialise MPI (if compiled in) and ensure MPI_Finalize on scope
/// exit. Use as `auto guard = mpi_guard(argc, argv);` at the top of
/// `main()`. No-op on non-MPI builds.
struct MpiGuard {
    bool inited = false;
    ~MpiGuard() {
#ifdef WITH_MPI
        if (inited) {
            int finalized = 0;
            MPI_Finalized(&finalized);
            if (!finalized) MPI_Finalize();
        }
#endif
    }
};

[[nodiscard]] inline MpiGuard mpi_guard([[maybe_unused]] int argc,
                                         [[maybe_unused]] char** argv) {
    MpiGuard g;
#ifdef WITH_MPI
    int already = 0;
    MPI_Initialized(&already);
    if (!already) {
        MPI_Init(&argc, &argv);
        g.inited = true;
    }
#endif
    return g;
}

}  // namespace ed_example
