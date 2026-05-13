// ftlm_dist.h - MPI-aware overloads for FTLM dynamical correlation kernels.
//
// Audit item #4 (distributed DSSF orchestration):
//   The legacy public APIs in <ed/solvers/ftlm.h> hardcode MPI_COMM_WORLD
//   for sample distribution and reductions. When the workflow has many
//   independent operator pairs and many ranks, a better topology is to
//   split MPI_COMM_WORLD into sub-communicators, with each subgroup
//   handling a slice of the operator pairs and running its own FTLM
//   sample reduction on the subgroup. This file declares overloads that
//   accept an explicit MPI_Comm so the workflow layer can drive that
//   topology.
//
//   The legacy entry points become thin wrappers that forward to these
//   overloads with comm = MPI_COMM_WORLD; behavior is bit-identical for
//   single-comm callers.
//
// Header is intentionally separate from <ed/solvers/ftlm.h> so consumers
// that don't need MPI don't pay for <mpi.h>.

#pragma once

#include <ed/solvers/ftlm.h>

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace ed {
namespace dssf {

#ifdef WITH_MPI

// Multi-sample multi-temperature FTLM with explicit communicator.
// All samples are split across ranks of `comm`; reductions and the
// initial Barrier use `comm` instead of MPI_COMM_WORLD.
std::map<double, DynamicalResponseResults>
compute_dynamical_correlation_multi_sample_multi_temperature_comm(
    std::function<void(const Complex*, Complex*, int)> H,
    std::function<void(const Complex*, Complex*, int)> O1,
    std::function<void(const Complex*, Complex*, int)> O2,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift,
    const std::string& output_dir,
    MPI_Comm comm);

// Multi-operator multi-temperature FTLM with explicit communicator.
std::vector<std::map<double, DynamicalResponseResults>>
compute_dynamical_correlation_multi_operator_multi_temperature_comm(
    std::function<void(const Complex*, Complex*, int)> H,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O1_list,
    const std::vector<std::function<void(const Complex*, Complex*, int)>>& O2_list,
    uint64_t N,
    const DynamicalResponseParameters& params,
    double omega_min,
    double omega_max,
    uint64_t num_omega_bins,
    const std::vector<double>& temperatures,
    double energy_shift,
    const std::string& output_dir,
    MPI_Comm comm);

#endif  // WITH_MPI

}  // namespace dssf
}  // namespace ed
