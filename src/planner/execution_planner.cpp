// =============================================================================
// src/planner/execution_planner.cpp
//
// Decision logic for the capability-aware ED execution planner. See header.
// =============================================================================

#include <ed/planner/execution_planner.h>

#include <ed/planner/csr_policy_hook.h>

#include <algorithm>
#include <cmath>
#include <sstream>

namespace ed::planner {

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

[[nodiscard]] bool gpu_lane(DeviceLane d) noexcept {
    return d == DeviceLane::Gpu || d == DeviceLane::MpiGpu;
}

[[nodiscard]] bool mpi_lane(DeviceLane d) noexcept {
    return d == DeviceLane::Mpi || d == DeviceLane::MpiGpu;
}

// Resolve the device lane from the user request + capabilities + dimension.
// "auto" mirrors select_backend's ordering but gates on probed fit.
DeviceLane resolve_device(const std::string& req,
                          const SystemCapabilities& caps,
                          std::uint64_t dim,
                          std::vector<std::string>& notes) {
    auto cuda_ok = caps.has_cuda_build && caps.n_gpus > 0;
    auto mpi_ok  = caps.has_mpi_build && caps.n_mpi_ranks > 1;
    auto nccl_ok = caps.has_nccl_build && cuda_ok && mpi_ok;

    if (req == "cpu")     return DeviceLane::Cpu;
    if (req == "gpu") {
        if (!cuda_ok) notes.push_back("device='gpu' requested but no CUDA GPU; "
                                      "falling back to CPU");
        return cuda_ok ? DeviceLane::Gpu : DeviceLane::Cpu;
    }
    if (req == "mpi")     return mpi_ok ? DeviceLane::Mpi : DeviceLane::Cpu;
    if (req == "mpi_gpu") return nccl_ok ? DeviceLane::MpiGpu
                                         : (mpi_ok ? DeviceLane::Mpi : DeviceLane::Cpu);

    // auto: prefer the cheapest lane that can hold the working set. Big problems
    // that exceed single-node RAM/VRAM escalate to MPI; small ones stay CPU/GPU.
    const double per_complex = 16.0;
    const double one_vec_gb = per_complex * static_cast<double>(dim) / kGiB;
    // Heuristic: a single GPU helps once dim is large enough to amortize launch
    // overhead but still fits VRAM; otherwise CPU. Distributed when nothing fits.
    if (nccl_ok && one_vec_gb * 4 > caps.vram_avail_gb()) {
        notes.push_back("auto: working set exceeds single-GPU VRAM -> mpi_gpu");
        return DeviceLane::MpiGpu;
    }
    if (cuda_ok && dim >= (std::uint64_t{1} << 14)
        && one_vec_gb * 4 <= caps.vram_avail_gb()) {
        return DeviceLane::Gpu;
    }
    if (mpi_ok && one_vec_gb * 4 > caps.ram_avail_gb() * 0.8) {
        notes.push_back("auto: working set exceeds host RAM -> mpi");
        return DeviceLane::Mpi;
    }
    return DeviceLane::Cpu;
}

}  // namespace

const char* ExecutionPlan::matvec_str() const noexcept {
    return matvec == MatvecStrategy::Csr ? "csr" : "matrix_free";
}
const char* ExecutionPlan::device_str() const noexcept {
    switch (device) {
        case DeviceLane::Cpu:    return "cpu";
        case DeviceLane::Gpu:    return "gpu";
        case DeviceLane::Mpi:    return "mpi";
        case DeviceLane::MpiGpu: return "mpi_gpu";
    }
    return "cpu";
}
const char* ExecutionPlan::basis_str() const noexcept {
    switch (basis) {
        case BasisStrategy::FullDense:        return "full_dense";
        case BasisStrategy::BinarySearchReps: return "binary_search_reps";
        case BasisStrategy::DenseRankTable:   return "dense_rank_table";
        case BasisStrategy::Distributed:      return "distributed";
    }
    return "full_dense";
}
const char* ExecutionPlan::reorth_str() const noexcept {
    using R = ed::krylov::ReorthPolicy;
    switch (reorth) {
        case R::None:         return "none";
        case R::FullCGS2:     return "full_cgs2";
        case R::PeriodicCGS2: return "periodic_cgs2";
        case R::LocalDGKS3:   return "local_dgks3";
    }
    return "local_dgks3";
}

std::string ExecutionPlan::summary() const {
    std::ostringstream os;
    os << "[ed::planner] " << (feasible ? "FEASIBLE" : "INFEASIBLE (" + bottleneck + ")")
       << "\n  matvec  : " << matvec_str()
       << "\n  device  : " << device_str() << " (n_ranks=" << n_ranks << ")"
       << "\n  reorth  : " << reorth_str()
       << (krylov_dim_cap > 0 ? "  krylov_cap=" + std::to_string(krylov_dim_cap) : "")
       << "\n  basis   : " << basis_str()
       << "\n  memory  : ~" << est_memory_gb << " GB"
       << "\n  time    : ~" << est_seconds << " s\n";
    for (const auto& n : notes) os << "  note: " << n << "\n";
    for (const auto& s : suggestions) os << "  suggestion: " << s << "\n";
    return os.str();
}

ExecutionPlan plan_execution(const TaskDescriptor&     task,
                             const SystemCapabilities& caps,
                             const UserConstraints&    uc) {
    using R = ed::krylov::ReorthPolicy;
    ExecutionPlan p;

    // ---- Device lane --------------------------------------------------------
    p.device = resolve_device(uc.device, caps, task.basis_dim, p.notes);
    const bool on_gpu = gpu_lane(p.device);
    p.n_ranks = mpi_lane(p.device)
        ? (uc.n_ranks > 0 ? uc.n_ranks : std::max(1, caps.n_mpi_ranks))
        : 1;

    const TaskCost cost = estimate_task_cost(task, on_gpu);

    // ---- Memory budget (per rank, on the chosen lane) -----------------------
    double budget_gb;
    if (on_gpu) {
        budget_gb = std::max(0.0, caps.vram_avail_gb() - 0.5);
    } else {
        const int local_ranks = (p.device == DeviceLane::Mpi) ? p.n_ranks : 1;
        budget_gb = std::max(0.0,
            (caps.ram_avail_gb() - 2.0 /* OS/runtime baseline */) /
            std::max(1, local_ranks));
    }
    const double budget = budget_gb * uc.memory_safety;

    const double working_gb = static_cast<double>(cost.working_set_bytes) / kGiB
                              / std::max(1, p.n_ranks);
    const double csr_gb     = static_cast<double>(cost.csr_bytes) / kGiB
                              / std::max(1, p.n_ranks);
    p.est_memory_gb = working_gb;

    // ---- Reorth (per-method default, downgrade if full-reorth won't fit) ----
    switch (task.method) {
        case Method::LTLM:          p.reorth = R::FullCGS2;   break;
        case Method::Lanczos:
        case Method::KrylovSchur:
        case Method::FTLM:
        case Method::TPQ:
        case Method::KPM:
        case Method::BlockLanczos:
        case Method::Full:          p.reorth = R::LocalDGKS3; break;
    }
    if (p.reorth == R::FullCGS2) {
        // Full reorth stores the whole Krylov basis (krylov * dim vectors).
        const std::uint64_t ms = (task.krylov_dim > 0)
            ? static_cast<std::uint64_t>(task.krylov_dim) : 100;
        const double basis_gb =
            static_cast<double>(ms) * static_cast<double>(task.basis_dim)
            * 16.0 / kGiB / std::max(1, p.n_ranks);
        if (basis_gb > budget) {
            p.reorth = R::PeriodicCGS2;
            p.notes.push_back("full-reorth basis (" + std::to_string(basis_gb)
                + " GB) exceeds budget; downgraded to periodic CGS2");
        }
    }

    // ---- Matvec strategy: CSR iff it fits AND amortizes (NO hard cap) -------
    if (uc.matvec == "csr") {
        p.matvec = MatvecStrategy::Csr;
        p.notes.push_back("matvec=csr forced by user");
    } else if (uc.matvec == "matrix_free") {
        p.matvec = MatvecStrategy::MatrixFree;
        p.notes.push_back("matvec=matrix_free forced by user");
    } else {
        const bool csr_fits = (csr_gb + working_gb) <= budget;
        // amortization: build cost must be repaid by the per-matvec saving over
        // the expected number of applications.
        const double saving_per_matvec =
            std::max(0.0, cost.matvec_seconds_mf - cost.matvec_seconds_csr);
        const double amortized =
            cost.csr_build_seconds <=
            static_cast<double>(cost.matvec_count) * saving_per_matvec;
        if (csr_fits && amortized && cost.matvec_count > 0) {
            p.matvec = MatvecStrategy::Csr;
            p.notes.push_back("CSR chosen: footprint " + std::to_string(csr_gb)
                + " GB + working " + std::to_string(working_gb)
                + " GB <= budget " + std::to_string(budget) + " GB, and assembly amortizes");
        } else {
            p.matvec = MatvecStrategy::MatrixFree;
            if (!csr_fits) {
                p.notes.push_back("CSR footprint " + std::to_string(csr_gb)
                    + " GB would not fit budget " + std::to_string(budget)
                    + " GB -> matrix-free");
            } else {
                p.notes.push_back("CSR assembly would not amortize over "
                    + std::to_string(cost.matvec_count) + " matvecs -> matrix-free");
            }
        }
    }

    // ---- Basis strategy + construction guard --------------------------------
    if (task.kind == BasisKind::Full) {
        p.basis = BasisStrategy::FullDense;
    } else {
        const double reps_gb = static_cast<double>(cost.reps_array_bytes) / kGiB;
        const double table_gb = static_cast<double>(cost.rank_table_bytes) / kGiB;
        // The reps array itself must fit (it backs both lookup paths).
        if (reps_gb > budget) {
            if (mpi_lane(p.device)) {
                p.basis = BasisStrategy::Distributed;
                p.feasible = false;            // not yet implemented -> guard
                p.bottleneck = "basis_construction";
                p.notes.push_back("reps array " + std::to_string(reps_gb)
                    + " GB exceeds per-rank budget; distributed reps construction "
                      "is not yet implemented");
                p.suggestions.push_back("reduce the sector (smaller N, more "
                    "symmetry) or add RAM/ranks until the reps array fits one rank");
            } else {
                p.basis = BasisStrategy::BinarySearchReps;
                p.feasible = false;
                p.bottleneck = "basis_construction";
                p.notes.push_back("reps array " + std::to_string(reps_gb)
                    + " GB exceeds memory budget " + std::to_string(budget) + " GB");
                p.suggestions.push_back("use device='mpi' or a bigger-memory node, "
                    "or project onto more symmetry");
            }
        } else if (table_gb > 0.0 && table_gb <= budget * 0.25) {
            // Dense O(1) rank table affordable -> faster reverse lookup.
            p.basis = BasisStrategy::DenseRankTable;
            p.notes.push_back("dense combinadic rank table (" + std::to_string(table_gb)
                + " GB) fits -> O(1) rep lookup");
        } else {
            p.basis = BasisStrategy::BinarySearchReps;
            if (table_gb > 0.0) {
                p.notes.push_back("rank table " + std::to_string(table_gb)
                    + " GB too large -> tableless O(log dim) binary-search reps");
            }
        }
    }

    // ---- Memory feasibility verdict (working set) ---------------------------
    if (p.feasible && working_gb > budget) {
        p.feasible = false;
        p.bottleneck = "memory";
        p.suggestions.push_back("working set " + std::to_string(working_gb)
            + " GB exceeds budget " + std::to_string(budget)
            + " GB; raise n_ranks / reduce the sector / drop eigenvectors");
    }

    // ---- Time estimate ------------------------------------------------------
    const double per_matvec = (p.matvec == MatvecStrategy::Csr)
        ? cost.matvec_seconds_csr : cost.matvec_seconds_mf;
    p.est_seconds = static_cast<double>(cost.matvec_count) *
                    (per_matvec / std::max(1, p.n_ranks))
                    + (p.matvec == MatvecStrategy::Csr ? cost.csr_build_seconds : 0.0)
                    + cost.enumeration_seconds;

    return p;
}

void apply_csr_decision(const ExecutionPlan& plan) {
    set_csr_override(plan.matvec == MatvecStrategy::Csr
                         ? CsrOverride::Csr
                         : CsrOverride::MatrixFree);
}

}  // namespace ed::planner
