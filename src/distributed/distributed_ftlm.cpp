// =============================================================================
// src/distributed/distributed_ftlm.cpp
//
// Phase 3b #3 implementation. See header for the design.
// =============================================================================

#ifdef WITH_MPI

#include <ed/distributed/distributed_ftlm.h>
#include <ed/distributed/distributed_lanczos.h>
#include <ed/distributed/distributed_operator.h>

#include <ed/core/construct_ham.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace ed::distributed {

namespace {

// Not constexpr: OpenMPI predefined handles cast through (void*) at runtime.
const MPI_Datatype kComplexDatatype = MPI_C_DOUBLE_COMPLEX;

}  // namespace

DistributedFtlmResult distributed_ftlm(
    std::shared_ptr<class ::Operator> op,
    const DistributedFtlmOptions& options,
    MPI_Comm world_comm) {

    int world_rank = 0, world_size = 0;
    MPI_Comm_rank(world_comm, &world_rank);
    MPI_Comm_size(world_comm, &world_size);

    int n_groups = std::max(1, options.n_groups);
    if (n_groups > world_size) {
        n_groups = world_size;
    }
    if (world_size % n_groups != 0) {
        throw std::invalid_argument(
            "distributed_ftlm: n_groups (" + std::to_string(n_groups)
            + ") must divide world_size (" + std::to_string(world_size) + ")");
    }
    const int ranks_per_group = world_size / n_groups;
    const int my_group        = world_rank / ranks_per_group;

    // Split world communicator into per-group subcomms. Each subcomm holds
    // `ranks_per_group` ranks (a slice of the row decomposition for that
    // group's DistributedOperator).
    MPI_Comm group_comm;
    MPI_Comm_split(world_comm, my_group, world_rank, &group_comm);

    DistributedOperator dop(op, group_comm);

    // Distribute samples round-robin across groups: sample s belongs to
    // group s % n_groups.
    const int n_samples = std::max(1, options.n_samples);
    std::vector<int> my_samples;
    my_samples.reserve(n_samples / n_groups + 1);
    for (int s = 0; s < n_samples; ++s) {
        if ((s % n_groups) == my_group) my_samples.push_back(s);
    }

    std::vector<double> betas = options.betas;
    if (betas.empty()) betas.push_back(1.0);

    // Per-group accumulator of Z(beta) across this group's samples.
    std::vector<double> Z_local(betas.size(), 0.0);

    DistributedLanczosOptions lopts;
    lopts.max_iter   = options.lanczos_max_iter;
    lopts.exct       = options.lanczos_max_iter;  // keep all Ritz values
    lopts.full_reorth = false;

    for (int s : my_samples) {
        lopts.seed = options.seed_offset + static_cast<unsigned long>(s);

        // We need both Ritz values AND the squared first-component weights
        // for the FTLM trace estimator
        //
        //   Z(beta) = sum_k |<v0 | psi_k>|^2 exp(-beta E_k)
        //           = sum_k weights[k] * exp(-beta E_k)
        //
        // The DistributedLanczosResult struct currently only carries
        // eigenvalues. For the first cut, we replicate the partial trace:
        // the leading-order FTLM kernel uses *only* the ground-state
        // contribution. A full FTLM with Ritz weights is wired in here as
        // soon as DistributedLanczos exposes them (currently a Phase
        // 3b #2.5 follow-up).
        //
        // Until then, we contribute exp(-beta E_min) per sample, which is
        // a strict-from-below partition-function estimator that converges
        // to the true Z at beta -> infinity. Documented honestly in
        // PHASE_3_SUMMARY.md.
        DistributedLanczosResult lres = distributed_lanczos(dop, lopts);

        if (lres.eigenvalues.empty()) continue;
        const double E_min = lres.eigenvalues.front();
        for (std::size_t b = 0; b < betas.size(); ++b) {
            Z_local[b] += std::exp(-betas[b] * E_min);
        }

        if (options.verbose && world_rank == 0) {
            std::cout << "  [dist-ftlm] sample s=" << s
                      << " group=" << my_group
                      << " E0=" << E_min
                      << " iters=" << lres.iterations << std::endl;
        }
    }

    // Reduce across groups via the world communicator.
    // Each rank in a group holds the SAME Z_local (DistributedLanczos is
    // collective on group_comm and produces a replicated eigenvalue
    // vector).  To avoid double-counting, only rank 0 of each group
    // contributes to the world-level reduction, which we implement as a
    // broadcast within each group followed by a per-group-rank-0 sum.
    if (world_rank % ranks_per_group != 0) {
        std::fill(Z_local.begin(), Z_local.end(), 0.0);
    }
    std::vector<double> Z(betas.size(), 0.0);
    MPI_Allreduce(Z_local.data(), Z.data(),
                  static_cast<int>(betas.size()),
                  MPI_DOUBLE, MPI_SUM, world_comm);
    const double inv = 1.0 / static_cast<double>(n_samples);
    for (auto& z : Z) z *= inv;

    MPI_Comm_free(&group_comm);

    DistributedFtlmResult result;
    result.Z = std::move(Z);
    result.samples_used = n_samples;
    return result;
}

}  // namespace ed::distributed

#endif  // WITH_MPI
