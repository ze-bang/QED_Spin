// =============================================================================
// python/qed/_bindings/workflow_bindings.cpp
//
// pybind11 surface for the Minimalist ED Collapse entry points:
//
//   * ed::workflows::solve    -> qed._core.workflows_solve(op, opts)
//   * ed::workflows::thermal  -> qed._core.workflows_thermal(op, opts)
//   * ed::workflows::spectral -> qed._core.workflows_spectral(op, obs, opts)
//
// Plus the option / result / enum / diagnostics types.
//
// The legacy dispatcher surface that used to live in
// `dispatcher_bindings.cpp` (the `exact_diagonalization_*` family) was
// hard-removed in the surface-unification collapse (May 2026); all
// Python entry points (`qed.solve` / `qed.thermal` / `qed.spectral`)
// now route through the bindings below.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <ed/core/hdf5_io.h>             // isDisabledOutputPath
#include <ed/core/fixed_sz_operator.h>   // FixedSzOperator (bound pybind type)
#include <ed/core/linear_operator.h>
#include <ed/core/make_operator.h>
#include <ed/symmetry/spin_flip.h>
#include <ed/symmetry/sector_plan.h>
#include <ed/symmetry/time_reversal.h>
#include <ed/symmetry/sym_profile.h>
#include <ed/core/operator.h>
#include <ed/core/results.h>
#include <ed/core/sector_loop.h>          // filter_sectors + resolve_target_sector
#include <ed/core/sector_thermo.h>        // combine_sector_thermodynamics (SOTA)
#include <ed/core/select_backend.h>
#include <ed/dssf/cross_sector_orbit_observable.h>  // SOTA cross-irrep observable
#include <ed/matvec/backends/cpu_backend.h>          // CpuBackend for cf_spectral_from_vector
#include <ed/observables/cf_spectral_kernel.h>      // cf_spectral_from_vector
#include <ed/observables/ftlm_cross_irrep_kernel.h>  // SOTA finite-T cross-irrep
#include <ed/orchestrator.h>
#include <ed/solvers/kpm_dos.h>                      // Wave B3: estimate_spectral_bounds

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#ifdef WITH_MPI
#include <mpi.h>
#endif

namespace py = pybind11;

namespace {

using Complex = std::complex<double>;

// (rank, size) on MPI_COMM_WORLD, or (0,1) when MPI is unavailable / not
// initialized. Lets the in-process symmetry sector loops distribute their
// independent per-sector work across ranks when launched under
// ``mpirun -n N python ...`` (mpi4py / a launcher initializes MPI).
inline std::pair<int, int> binding_mpi_rank_size() {
    int rank = 0, size = 1;
#ifdef WITH_MPI
    int inited = 0;
    MPI_Initialized(&inited);
    if (inited) {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);
    }
#endif
    return {rank, size};
}

#ifdef WITH_MPI
// Across-sector finite-T recombination: every rank holds the per-sector
// ThermodynamicData for ITS sectors only; Allgather the combine-relevant
// arrays (temperatures, energy, specific_heat, entropy, free_energy -- the
// fields ed::core::combine_sector_thermodynamics reads) so every rank ends with
// the FULL per-sector list and computes an identical combined result. gs_E is
// min-reduced. No-op when single-rank. Returns the gathered full list in-place.
inline void mpi_allgather_sector_thermo(
    std::vector<ThermodynamicData>&  per_sector_thermo,
    std::vector<std::uint64_t>&      per_sector_dims,
    const std::vector<std::uint64_t>& raw_indices,  // parallel to per_sector_thermo
    double&                          gs_E,
    int                              mpi_size) {
    if (mpi_size <= 1) return;

    // Agree on the temperature-grid length (a rank that owns no sector has 0).
    int nT = per_sector_thermo.empty()
                 ? 0 : static_cast<int>(per_sector_thermo.front().temperatures.size());
    int nT_global = nT;
    MPI_Allreduce(&nT, &nT_global, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (nT_global <= 0) {  // no rank produced any thermo
        double gmin = gs_E;
        MPI_Allreduce(&gs_E, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        gs_E = gmin;
        return;
    }

    // Flatten local sectors: [raw_index, temps, energy, Cv, S, F] = 1 + 5*nT
    // doubles each. The raw index travels with the block so the gathered list
    // can be re-sorted into the canonical (single-node) order -- otherwise the
    // F-based combine sums sectors in rank order and the result differs by FP
    // rounding from the single-rank run.
    const int per_block = 1 + 5 * nT_global;
    std::vector<double> send;
    send.reserve(per_sector_thermo.size() * static_cast<std::size_t>(per_block));
    auto put = [&](const std::vector<double>& v) {
        for (int t = 0; t < nT_global; ++t)
            send.push_back(t < static_cast<int>(v.size()) ? v[static_cast<std::size_t>(t)] : 0.0);
    };
    for (std::size_t s = 0; s < per_sector_thermo.size(); ++s) {
        send.push_back(static_cast<double>(s < raw_indices.size() ? raw_indices[s] : s));
        const auto& th = per_sector_thermo[s];
        put(th.temperatures); put(th.energy); put(th.specific_heat);
        put(th.entropy);      put(th.free_energy);
    }

    const int sendcount = static_cast<int>(send.size());
    std::vector<int> counts(static_cast<std::size_t>(mpi_size));
    MPI_Allgather(&sendcount, 1, MPI_INT, counts.data(), 1, MPI_INT, MPI_COMM_WORLD);
    std::vector<int> displs(static_cast<std::size_t>(mpi_size));
    int total = 0;
    for (int i = 0; i < mpi_size; ++i) { displs[i] = total; total += counts[i]; }
    std::vector<double> recv(static_cast<std::size_t>(total));
    MPI_Allgatherv(send.data(), sendcount, MPI_DOUBLE,
                   recv.data(), counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

    // Sort the gathered blocks by raw sector index (canonical order == the
    // single-node sector loop order) so the combine is bit-identical.
    std::vector<int> block_off;
    for (int off = 0; off + per_block <= total; off += per_block) block_off.push_back(off);
    std::sort(block_off.begin(), block_off.end(),
              [&](int a, int b) { return recv[a] < recv[b]; });

    per_sector_thermo.clear();
    per_sector_dims.clear();
    for (int off : block_off) {
        ThermodynamicData th;
        auto take = [&](int slot) {  // slot 0 is raw_index; arrays start at 1
            const int base = off + 1 + slot * nT_global;
            return std::vector<double>(recv.begin() + base, recv.begin() + base + nT_global);
        };
        th.temperatures  = take(0);
        th.energy        = take(1);
        th.specific_heat = take(2);
        th.entropy       = take(3);
        th.free_energy   = take(4);
        per_sector_thermo.push_back(std::move(th));
        per_sector_dims.push_back(1);   // dims are unused by the F-based combine
    }

    double gmin = gs_E;
    MPI_Allreduce(&gs_E, &gmin, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    gs_E = gmin;
}
#endif  // WITH_MPI

// Pybind11 cannot move a captured `std::unique_ptr<LinearOperator>` out of
// a Python-owned Operator easily; we instead accept the raw `Operator&` /
// `FixedSzOperator&` and rely on the fact that both inherit from
// `ed::LinearOperator`. The orchestrator calls take `const LinearOperator&`,
// so a simple reference upcast is enough.

// ---------------------------------------------------------------------------
// GPU lane (operator-collapse Phase 2a, Jun 2026).
//
// ``qed.solve/thermal/spectral(device='gpu')`` flips
// ``opts.backend.allow_gpu = true`` in the Python wrappers. The plain
// ``ed::Operator`` / ``ed::FixedSzOperator`` now advertise
// ``geometry().supports_device_matvec=true`` (WITH_CUDA), and their
// ``bind_cuda()`` lazily builds a ``CudaMatVecBackend`` device mirror (the
// SOTA no-atomic gather kernel). So ``ed::select_backend`` picks the
// ``CudaBackend`` lane straight off the host operator's capability flag --
// no bespoke ``GPUOperator`` / ``GPUFixedSzOperator`` promotion needed
// (that path is retired here; the legacy classes go away in Phase 2b).
//
// The streaming-symmetry directory binding is likewise capability-driven:
// the per-sector operators advertise the flag and wire their own lazy GPU
// mirror.
//
// What remains below: a runtime probe + a Python ``RuntimeWarning`` for the
// few lanes that genuinely have no GPU implementation (FullDiag for solve;
// any future host-only thermal/spectral method), so a ``device='gpu'``
// request that silently runs on CPU is visible at the call site.
// ---------------------------------------------------------------------------
inline bool gpu_runtime_available() noexcept {
#ifdef WITH_CUDA
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return n > 0;
#else
    return false;
#endif
}

/// FullDiag in ``solve_on<Backend>`` builds the dense matrix by calling
/// the bound matvec with HOST ``std::vector<Complex>`` storage and runs
/// the (host-only) LAPACK dense eigensolver; it explicitly pins the
/// column-build to ``bind_cpu()``. FullDiag is the auto-selected method
/// for ``global_dim <= 2^12 = 4096`` (see ``auto_solve_method`` in
/// orchestrator.cpp). At those dimensions the matvec is negligible
/// compared to the O(N^3) LAPACK solve and the GPU lane offers no
/// measurable win, so the binding pins ``allow_gpu=false`` for FullDiag
/// to avoid spinning up an unused CudaBackend (and a misleading "gpu"
/// lane label).
inline bool will_use_full_diag(const ed::LinearOperator& op,
                               const ed::workflows::SolveOptions& opts) noexcept {
    if (opts.method == ed::workflows::SolveMethod::FullDiag) return true;
    if (opts.method != ed::workflows::SolveMethod::Auto)     return false;
    const auto geom = op.geometry();
    return geom.global_dim <= (1ULL << 12);
}

/// The thermal lane has uneven GPU coverage:
///   * FTLM         : CPU only (orchestrator throws on CUDA).
///   * LTLM, KpmDos : CPU or CUDA.
///   * mTPQ         : any backend.
///
/// As of Phase E of the "Close CPU/GPU Gaps" plan (May 2026), the
/// FTLM facade dispatches on Backend internally (see
/// ``include/ed/thermal/ftlm_kernel.h``) and accepts both ``CpuBackend``
/// and ``CudaBackend``, so it joins the GPU-eligible set.
inline bool thermal_method_supports_gpu(
    ed::workflows::ThermalOptions::Method m) noexcept {
    using M = ed::workflows::ThermalOptions::Method;
    return m == M::FTLM
        || m == M::LTLM
        || m == M::KpmDos
        || m == M::mTPQ;
}

/// Spectral lanes after Phases F + G of the "Close CPU/GPU Gaps"
/// plan (May 2026): every method now dispatches on Backend
/// internally, so the entire spectral lane is GPU-eligible.
///   * GroundStateCF : runs the inner solve + CF kernel through
///                     ``H.template bind<B>()``.
///   * FtlmDynamical : routes through
///                     ``detail::ftlm_dynamical_kernel_via_backend``.
///   * KpmDynamical  : routes through
///                     ``detail::kpm_dynamical_kernel_via_backend``.
inline bool spectral_method_supports_gpu(
    ed::workflows::SpectralOptions::Method /*m*/) noexcept {
    return true;
}

/// Emit a Python ``RuntimeWarning`` so the caller sees the silent
/// demotion ``device='gpu' -> CPU lane`` instead of finding out through
/// a profiler. The warning fires only when the GPU was actually
/// reachable (``allow_gpu=true`` AND ``gpu_runtime_available()``) -- if
/// the build is CPU-only or no NVIDIA device is visible there is no
/// "demotion" to report. Uses ``stacklevel=2`` so the warning blame
/// points at the user's ``qed.solve / qed.thermal / qed.spectral``
/// call site rather than at this binding.
inline void warn_silent_cpu_fallback(const char* what,
                                     const ed::BackendConstraints& c) {
    if (!c.allow_gpu) return;
#ifdef WITH_CUDA
    if (!gpu_runtime_available()) return;
#else
    return;
#endif
    try {
        py::module_::import("warnings").attr("warn")(
            std::string(what)
                + " requested device='gpu' but the chosen method has no GPU "
                  "implementation in the orchestrator. Falling back to the "
                  "CPU lane. Pass device='cpu' to silence this warning, or "
                  "switch to a GPU-clean method (Lanczos/BlockLanczos/"
                  "KrylovSchur for solve; LTLM/KPM_DOS/mTPQ for "
                  "thermal; GroundStateCF for spectral).",
            py::module_::import("builtins").attr("RuntimeWarning"),
            py::arg("stacklevel") = 2);
    } catch (const py::error_already_set&) {
        // Best-effort -- never let the warning machinery break the
        // workflow call. The caller still gets the correct CPU result.
    }
}

}  // namespace

void bind_workflows(py::module_& m) {
    // -----------------------------------------------------------------
    // BackendConstraints (Geometry-side knobs the caller can override).
    // -----------------------------------------------------------------
    py::class_<ed::BackendConstraints>(m, "BackendConstraints")
        .def(py::init<>())
        .def_readwrite("allow_gpu",     &ed::BackendConstraints::allow_gpu)
        .def_readwrite("allow_mpi",     &ed::BackendConstraints::allow_mpi)
        .def_readwrite("allow_mpi_gpu", &ed::BackendConstraints::allow_mpi_gpu)
        .def_readwrite("fudge_factor",  &ed::BackendConstraints::fudge_factor);

    // -----------------------------------------------------------------
    // SolveMethod / SolveOptions / GroundStateResult.
    // -----------------------------------------------------------------
    py::enum_<ed::workflows::SolveMethod>(m, "SolveMethod")
        .value("Auto",        ed::workflows::SolveMethod::Auto)
        .value("Lanczos",     ed::workflows::SolveMethod::Lanczos)
        .value("BlockLanczos",ed::workflows::SolveMethod::BlockLanczos)
        .value("BlockKrylovSchur", ed::workflows::SolveMethod::BlockKrylovSchur)
        .value("KrylovSchur", ed::workflows::SolveMethod::KrylovSchur)
        .value("FullDiag",    ed::workflows::SolveMethod::FullDiag)
        .export_values();

    py::class_<ed::workflows::SolveOptions>(m, "SolveOptions")
        .def(py::init<>())
        .def_readwrite("num_eigs",        &ed::workflows::SolveOptions::num_eigs)
        .def_readwrite("max_iter",        &ed::workflows::SolveOptions::max_iter)
        .def_readwrite("block_size",      &ed::workflows::SolveOptions::block_size)
        .def_readwrite("tolerance",       &ed::workflows::SolveOptions::tolerance)
        .def_readwrite("compute_vectors", &ed::workflows::SolveOptions::compute_vectors)
        .def_readwrite("output_dir",      &ed::workflows::SolveOptions::output_dir)
        .def_readwrite("method",          &ed::workflows::SolveOptions::method)
        .def_readwrite("allow_infeasible",&ed::workflows::SolveOptions::allow_infeasible)
        .def_readwrite("backend",         &ed::workflows::SolveOptions::backend)
        // Wave A5 (Full unified-interface collapse, May 2026): CLI parity knobs.
        .def_readwrite("use_fixed_sz",
                       &ed::workflows::SolveOptions::use_fixed_sz)
        .def_readwrite("use_symmetry",
                       &ed::workflows::SolveOptions::use_symmetry)
        .def_readwrite("n_up",
                       &ed::workflows::SolveOptions::n_up)
        .def_readwrite("basis_cache_dir",
                       &ed::workflows::SolveOptions::basis_cache_dir)
        .def_readwrite("spin_flip",
                       &ed::workflows::SolveOptions::spin_flip)
        .def_readwrite("time_reversal",
                       &ed::workflows::SolveOptions::time_reversal)
        .def_readwrite("star_maps", &ed::workflows::SolveOptions::star_maps)
        .def_readwrite("sz_parity", &ed::workflows::SolveOptions::sz_parity)
        .def_readwrite("precompute_basis_only",
                       &ed::workflows::SolveOptions::precompute_basis_only)
        // SOTA streaming-symmetry filter (May 2026).
        .def_readwrite("selected_sectors",
                       &ed::workflows::SolveOptions::selected_sectors);

    py::class_<ed::BackendMetadata>(m, "BackendMetadata")
        .def(py::init<>())
        .def_readonly("lane",         &ed::BackendMetadata::lane)
        .def_readonly("mpi_size",     &ed::BackendMetadata::mpi_size)
        .def_readonly("cuda_devices", &ed::BackendMetadata::cuda_devices)
        .def_readonly("wall_seconds", &ed::BackendMetadata::wall_seconds);

    py::class_<ed::KrylovDiagnostics>(m, "KrylovDiagnostics")
        .def(py::init<>())
        .def_readonly("alpha",          &ed::KrylovDiagnostics::alpha)
        .def_readonly("beta",           &ed::KrylovDiagnostics::beta)
        .def_readonly("iters_done",     &ed::KrylovDiagnostics::iters_done)
        .def_readonly("residual_norm",  &ed::KrylovDiagnostics::residual_norm)
        .def_readonly("ritz_residuals", &ed::KrylovDiagnostics::ritz_residuals)
        .def_readonly("n_converged",    &ed::KrylovDiagnostics::n_converged)
        .def_readonly("resid_history",  &ed::KrylovDiagnostics::resid_history)
        .def_readonly("converged",      &ed::KrylovDiagnostics::converged);

    py::class_<ed::EigenvectorRef>(m, "EigenvectorRef")
        .def(py::init<>())
        .def_readonly("host",       &ed::EigenvectorRef::host)
        .def_readonly("hdf5_path",  &ed::EigenvectorRef::hdf5_path)
        .def_readonly("on_backend", &ed::EigenvectorRef::on_backend);

    // SOTA streaming-symmetry quantum-number tag (May 2026).
    py::class_<ed::SectorTag>(m, "SectorTag")
        .def(py::init<>())
        .def_readwrite("sector_index",    &ed::SectorTag::sector_index)
        .def_readwrite("sector_dim",      &ed::SectorTag::sector_dim)
        .def_readwrite("quantum_numbers", &ed::SectorTag::quantum_numbers)
        .def_readwrite("n_up",            &ed::SectorTag::n_up)
        .def("__repr__", [](const ed::SectorTag& t) {
            std::string s = "SectorTag(index=" + std::to_string(t.sector_index)
                          + ", dim=" + std::to_string(t.sector_dim);
            if (!t.quantum_numbers.empty()) {
                s += ", QN=[";
                for (std::size_t i = 0; i < t.quantum_numbers.size(); ++i) {
                    s += (i ? "," : "")
                       + std::to_string(t.quantum_numbers[i]);
                }
                s += "]";
            }
            if (t.n_up >= 0) s += ", n_up=" + std::to_string(t.n_up);
            s += ")";
            return s;
        });

    py::class_<ed::GroundStateResult>(m, "GroundStateResult")
        .def(py::init<>())
        .def_readonly("eigenvalues",  &ed::GroundStateResult::eigenvalues)
        .def_readonly("eigenvectors", &ed::GroundStateResult::eigenvectors)
        .def_readonly("krylov",       &ed::GroundStateResult::krylov)
        .def_readonly("backend",      &ed::GroundStateResult::backend)
        .def_readonly("hdf5_path",    &ed::GroundStateResult::hdf5_path)
        // SOTA streaming-symmetry attribution (May 2026).
        .def_readonly("sector_tags",
                      &ed::GroundStateResult::sector_tags)
        .def_readonly("eigenvalues_per_sector",
                      &ed::GroundStateResult::eigenvalues_per_sector)
        .def_readonly("sector_index_of_eigenvalue",
                      &ed::GroundStateResult::sector_index_of_eigenvalue);

    // -----------------------------------------------------------------
    // ThermalOptions / ThermalResult.
    // -----------------------------------------------------------------
    py::enum_<ed::workflows::ThermalOptions::Method>(m, "ThermalMethod")
        .value("FTLM",   ed::workflows::ThermalOptions::Method::FTLM)
        .value("LTLM",   ed::workflows::ThermalOptions::Method::LTLM)
        .value("mTPQ",   ed::workflows::ThermalOptions::Method::mTPQ)
        .value("KpmDos", ed::workflows::ThermalOptions::Method::KpmDos)
        .value("OFTLM",  ed::workflows::ThermalOptions::Method::OFTLM)
        .export_values();

    py::class_<ed::workflows::ThermalOptions>(m, "ThermalOptions")
        .def(py::init<>())
        .def_readwrite("method",       &ed::workflows::ThermalOptions::method)
        .def_readwrite("allow_infeasible", &ed::workflows::ThermalOptions::allow_infeasible)
        .def_readwrite("num_samples",  &ed::workflows::ThermalOptions::num_samples)
        .def_readwrite("krylov_dim",   &ed::workflows::ThermalOptions::krylov_dim)
        .def_readwrite("num_exact",    &ed::workflows::ThermalOptions::num_exact)
        .def_readwrite("taylor_order", &ed::workflows::ThermalOptions::taylor_order)
        .def_readwrite("betas",        &ed::workflows::ThermalOptions::betas)
        .def_readwrite("delta_beta",   &ed::workflows::ThermalOptions::delta_beta)
        .def_readwrite("beta_max",     &ed::workflows::ThermalOptions::beta_max)
        .def_readwrite("random_seed",  &ed::workflows::ThermalOptions::random_seed)
        .def_readwrite("spin_flip",
                       &ed::workflows::ThermalOptions::spin_flip)
        .def_readwrite("time_reversal",
                       &ed::workflows::ThermalOptions::time_reversal)
        .def_readwrite("star_maps", &ed::workflows::ThermalOptions::star_maps)
        .def_readwrite("sz_parity", &ed::workflows::ThermalOptions::sz_parity)
        .def_readwrite("output_dir",   &ed::workflows::ThermalOptions::output_dir)
        .def_readwrite("backend",      &ed::workflows::ThermalOptions::backend)
        // Wave A5: CLI parity knobs (temperature scan + KPM broadening).
        .def_readwrite("temp_min",
                       &ed::workflows::ThermalOptions::temp_min)
        .def_readwrite("temp_max",
                       &ed::workflows::ThermalOptions::temp_max)
        .def_readwrite("num_temp_bins",
                       &ed::workflows::ThermalOptions::num_temp_bins)
        .def_readwrite("broadening",
                       &ed::workflows::ThermalOptions::broadening)
        // SOTA streaming-symmetry sector filter (May 2026).
        .def_readwrite("selected_sectors",
                       &ed::workflows::ThermalOptions::selected_sectors)
        // Wave B3 follow-up (May 2026): KPM-DOS spectral-bound
        // overrides (NaN sentinel) plus the Hutchinson/moment knobs
        // that the orchestrator forwards into ``KpmDosOptions``.
        // ``0`` for the int knobs means "kernel default" which keeps
        // legacy call sites unchanged.
        .def_readwrite("e_min_override",
                       &ed::workflows::ThermalOptions::e_min_override)
        .def_readwrite("e_max_override",
                       &ed::workflows::ThermalOptions::e_max_override)
        // mTPQ expert override of the (L*I - H) large value. ``0.0`` ->
        // auto-tune from a Lanczos spectral-bound estimate.
        .def_readwrite("energy_shift",
                       &ed::workflows::ThermalOptions::energy_shift)
        // fp32 single-GPU mTPQ (memory-halving lane): complex<float> state
        // vectors so the full 2^32 Hilbert space runs mTPQ on one 80 GB H100.
        .def_readwrite("mtpq_fp32",
                       &ed::workflows::ThermalOptions::mtpq_fp32)
        .def_readwrite("kpm_num_moments",
                       &ed::workflows::ThermalOptions::kpm_num_moments)
        .def_readwrite("kpm_num_random_vectors",
                       &ed::workflows::ThermalOptions::kpm_num_random_vectors)
        // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
        // user-supplied probe-betas for mTPQ state-vector
        // snapshots. Empty list (default) means "no state vectors are
        // saved"; the trajectory is always saved when ``output_dir``
        // is set.
        .def_readwrite("probe_betas",
                       &ed::workflows::ThermalOptions::probe_betas);

    py::class_<ed::ThermalResult>(m, "ThermalResult")
        .def(py::init<>())
        // Full Unified-Interface Collapse, Wave E2 (May 2026): expose
        // the `thermo` (ThermodynamicData) field so qed.thermal can read
        // back the recombined temperature scan from
        // `_core.workflows_thermal`. `ThermodynamicData` is already bound
        // via `dispatcher_bindings.cpp`, and the per-sector entries are
        // also surfaced for the Sz-iteration consumer.
        .def_readonly("thermo",              &ed::ThermalResult::thermo)
        .def_readonly("per_sector",          &ed::ThermalResult::per_sector)
        .def_readonly("ground_state_energy", &ed::ThermalResult::ground_state_energy)
        .def_readonly("krylov",              &ed::ThermalResult::krylov)
        .def_readonly("backend",             &ed::ThermalResult::backend)
        .def_readonly("hdf5_path",           &ed::ThermalResult::hdf5_path)
        // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
        // TPQ trajectory + state-snapshot surface. Mirror-images of
        // the kernel result so the user can introspect what landed
        // in HDF5 (or post-process in memory) without re-running the
        // sample.
        .def_readonly("tpq_sample_betas",
                      &ed::ThermalResult::tpq_sample_betas)
        .def_readonly("tpq_sample_energies",
                      &ed::ThermalResult::tpq_sample_energies)
        .def_readonly("tpq_sample_variances",
                      &ed::ThermalResult::tpq_sample_variances)
        .def_readonly("tpq_state_snapshots",
                      &ed::ThermalResult::tpq_state_snapshots);

    py::class_<ed::TpqStateSnapshot>(m, "TpqStateSnapshot")
        .def(py::init<>())
        .def_readonly("sample_index",   &ed::TpqStateSnapshot::sample_index)
        .def_readonly("requested_beta", &ed::TpqStateSnapshot::requested_beta)
        .def_readonly("effective_beta", &ed::TpqStateSnapshot::effective_beta)
        .def_readonly("psi",            &ed::TpqStateSnapshot::psi);

    // ThermalSectorEntry binding (needed for ThermalResult.per_sector).
    py::class_<ed::ThermalSectorEntry>(m, "ThermalSectorEntry")
        .def(py::init<>())
        .def_readonly("sz_index",            &ed::ThermalSectorEntry::sz_index)
        .def_readonly("ground_state_energy", &ed::ThermalSectorEntry::ground_state_energy)
        .def_readonly("thermo",              &ed::ThermalSectorEntry::thermo)
        // SOTA: streaming-symmetry attribution (May 2026).
        .def_readonly("tag",                 &ed::ThermalSectorEntry::tag);

    // -----------------------------------------------------------------
    // SpectralOptions / SpectralResult.
    // -----------------------------------------------------------------
    py::enum_<ed::workflows::SpectralOptions::Method>(m, "SpectralMethod")
        .value("GroundStateCF",  ed::workflows::SpectralOptions::Method::GroundStateCF)
        .value("FtlmDynamical",  ed::workflows::SpectralOptions::Method::FtlmDynamical)
        // Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026):
        // KpmDynamical -- Chebyshev expansion of `delta(omega - H)`.
        .value("KpmDynamical",   ed::workflows::SpectralOptions::Method::KpmDynamical)
        .export_values();

    py::enum_<ed::workflows::SpectralOptions::KpmKernel>(m, "SpectralKpmKernel")
        .value("Jackson", ed::workflows::SpectralOptions::KpmKernel::Jackson)
        .value("Lorentz", ed::workflows::SpectralOptions::KpmKernel::Lorentz)
        .export_values();

    py::class_<ed::workflows::SpectralOptions>(m, "SpectralOptions")
        .def(py::init<>())
        .def_readwrite("method",       &ed::workflows::SpectralOptions::method)
        .def_readwrite("allow_infeasible", &ed::workflows::SpectralOptions::allow_infeasible)
        .def_readwrite("krylov_dim",   &ed::workflows::SpectralOptions::krylov_dim)
        .def_readwrite("broadening",   &ed::workflows::SpectralOptions::broadening)
        .def_readwrite("omega_min",    &ed::workflows::SpectralOptions::omega_min)
        .def_readwrite("omega_max",    &ed::workflows::SpectralOptions::omega_max)
        .def_readwrite("num_omega",    &ed::workflows::SpectralOptions::num_omega)
        .def_readwrite("energy_shift", &ed::workflows::SpectralOptions::energy_shift)
        .def_readwrite("output_dir",   &ed::workflows::SpectralOptions::output_dir)
        .def_readwrite("backend",      &ed::workflows::SpectralOptions::backend)
        // SOTA streaming-symmetry knobs (May 2026).
        .def_readwrite("momentum_transfer",
                       &ed::workflows::SpectralOptions::momentum_transfer)
        .def_readwrite("momentum_tolerance",
                       &ed::workflows::SpectralOptions::momentum_tolerance)
        .def_readwrite("selected_sectors",
                       &ed::workflows::SpectralOptions::selected_sectors)
        // Wave A5: CLI parity knobs (FtlmDynamical sample/temperature
        // controls and the observable-type discriminator).
        .def_readwrite("num_samples",
                       &ed::workflows::SpectralOptions::num_samples)
        .def_readwrite("temperatures",
                       &ed::workflows::SpectralOptions::temperatures)
        .def_readwrite("observable_type",
                       &ed::workflows::SpectralOptions::observable_type)
        // Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026):
        // user-supplied seed state for the GroundStateCF lane.
        .def_readwrite("initial_state",
                       &ed::workflows::SpectralOptions::initial_state)
        // Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026):
        // KpmDynamical knobs.
        .def_readwrite("kpm_moments",
                       &ed::workflows::SpectralOptions::kpm_moments)
        .def_readwrite("kpm_kernel",
                       &ed::workflows::SpectralOptions::kpm_kernel)
        .def_readwrite("kpm_lorentz_lambda",
                       &ed::workflows::SpectralOptions::kpm_lorentz_lambda)
        .def_readwrite("kpm_spectral_bounds",
                       &ed::workflows::SpectralOptions::kpm_spectral_bounds);

    // SOTA cross-sector spectral contribution (May 2026).
    py::class_<ed::SpectralSectorEntry>(m, "SpectralSectorEntry")
        .def(py::init<>())
        .def_readonly("initial",   &ed::SpectralSectorEntry::initial)
        .def_readonly("final",     &ed::SpectralSectorEntry::final_)
        .def_readonly("S_real",    &ed::SpectralSectorEntry::S_real)
        .def_readonly("S_imag",    &ed::SpectralSectorEntry::S_imag)
        .def_readonly("static_sf", &ed::SpectralSectorEntry::static_sf);

    py::class_<ed::SpectralResult>(m, "SpectralResult", py::dynamic_attr())
        .def(py::init<>())
        .def_readonly("omega",        &ed::SpectralResult::omega)
        .def_readonly("S_real",       &ed::SpectralResult::S_real)
        .def_readonly("S_imag",       &ed::SpectralResult::S_imag)
        .def_readonly("errors_real",  &ed::SpectralResult::errors_real)
        .def_readonly("errors_imag",  &ed::SpectralResult::errors_imag)
        .def_readonly("krylov",       &ed::SpectralResult::krylov)
        .def_readonly("backend",      &ed::SpectralResult::backend)
        .def_readonly("hdf5_path",    &ed::SpectralResult::hdf5_path)
        // SOTA streaming-symmetry attribution (May 2026).
        .def_readonly("per_sector_pair",
                      &ed::SpectralResult::per_sector_pair)
        .def_readonly("selection_rule_label",
                      &ed::SpectralResult::selection_rule_label);

    // -----------------------------------------------------------------
    // The three entry points.
    // -----------------------------------------------------------------
    // -----------------------------------------------------------------
    // Symmetry detection for the Python toggle surface (Stage 8e):
    // term-level checks of the discrete symmetries the composition
    // layer can exploit, so qed.solve/thermal/spectral can REPORT
    // which symmetries the Hamiltonian actually carries and degrade
    // gracefully when a requested one is absent.
    // -----------------------------------------------------------------
    m.def("detect_hamiltonian_symmetries",
          [](const Operator& op) {
              ed::matvec::TermStorage soa;
              ed::matvec::TermStorage::classify_route(
                  soa, op.transform_data_, op.three_body_data_,
                  [](const std::complex<double>& c) { return c; });
              py::dict out;
              out["spin_flip"] =
                  ed::symmetry::hamiltonian_is_spin_flip_symmetric(soa);
              out["time_reversal"] = ed::symmetry::hamiltonian_is_real(soa);
              const auto ax = ed::symmetry::sz_axis_of(soa);
              out["u1"] = (ax == ed::symmetry::SzAxis::U1);
              out["sz_parity"] =
                  (ax != ed::symmetry::SzAxis::None);  // U1 implies parity
              return out;
          },
          py::arg("op"),
          R"pbdoc(
            Term-level discrete-symmetry detection.

            Returns ``{"spin_flip": bool, "time_reversal": bool}``:
            whether ``[H, prod_i sigma^x_i] == 0`` (global spin flip)
            and whether every coefficient is real in the computational
            basis (time reversal / conjugation pairing). These are the
            same checks the C++ composition layer
            (``ed::symmetry::resolve_symmetry_composition``) runs; the
            Python binding exists so the toggle surface can *report*
            the detection outcome and warn on a requested-but-absent
            symmetry instead of failing later.
          )pbdoc");

    m.def("workflows_solve",
          [](Operator& op, ed::workflows::SolveOptions opts) {
              // GPU lane (operator-collapse Phase 2a): the host Operator /
              // FixedSzOperator advertise ``supports_device_matvec`` and
              // ``bind_cuda()`` builds a CudaMatVecBackend device mirror, so
              // ``ed::select_backend`` picks the CudaBackend lane straight
              // off the capability flag -- no GPUOperator promotion needed.
              //
              // Exception: FullDiag (small-dim O(N^3) dense LAPACK solve) has
              // no GPU implementation and its column-build runs on host
              // pointers, so pin it to the CPU lane (avoids an unused
              // CudaBackend + a misleading "gpu" label). Surface the demotion
              // as a Python RuntimeWarning when device='gpu' was requested.
              if (will_use_full_diag(op, opts)) {
                  warn_silent_cpu_fallback(
                      "qed.solve (FullDiag)", opts.backend);
                  opts.backend.allow_gpu = false;
              }
              return ed::workflows::solve(
                  static_cast<const ed::LinearOperator&>(op), std::move(opts));
          },
          py::arg("op"),
          py::arg("opts") = ed::workflows::SolveOptions{},
          "Run the unified ground-state Krylov solver (Phase 4.2 collapse). "
          "Backend (CPU/GPU/MPI/MPI+GPU) is chosen via `ed::select_backend`. "
          "When ``opts.backend.allow_gpu`` is true and a CUDA device is "
          "available, the host operator's lazy CudaMatVecBackend device "
          "mirror runs the matvec on the GPU (no manual conversion). "
          "FullDiag stays on the CPU lane (no GPU implementation) and the "
          "binding emits a Python RuntimeWarning so the demotion is "
          "visible at the call site.");

    m.def("workflows_thermal",
          [](Operator& op, ed::workflows::ThermalOptions opts) {
              // Phase E of the "Close CPU/GPU Gaps" plan (May 2026):
              // every thermal method (FTLM / LTLM / mTPQ /
              // KpmDos) dispatches on Backend internally and accepts both
              // ``CpuBackend`` and ``CudaBackend``, so the host operator's
              // lazy CudaMatVecBackend mirror (operator-collapse Phase 2a)
              // serves the GPU lane directly. The ``supports_gpu`` gate
              // stays as a defensive future-proofing hook: a new host-only
              // method opts out via ``thermal_method_supports_gpu`` and the
              // binding demotes loudly with a Python ``RuntimeWarning``.
              if (!thermal_method_supports_gpu(opts.method)) {
                  warn_silent_cpu_fallback(
                      "qed.thermal (host-only method)", opts.backend);
                  opts.backend.allow_gpu = false;
              }
              return ed::workflows::thermal(
                  static_cast<const ed::LinearOperator&>(op), std::move(opts));
          },
          py::arg("op"),
          py::arg("opts") = ed::workflows::ThermalOptions{},
          "Run the unified finite-temperature workflow (FTLM / LTLM / mTPQ / "
          "KPM-DOS) over the auto-selected Backend. ``allow_gpu`` "
          "routes the matvec through the host operator's lazy "
          "CudaMatVecBackend device mirror without manual conversion.");

    m.def("workflows_spectral",
          [](Operator& op,
             std::vector<Operator*> observables,
             ed::workflows::SpectralOptions opts) {
              // GPU lane (operator-collapse Phase 2a): every spectral method
              // dispatches on Backend internally and consumes H + the
              // observables through ``bind<Backend>()``. The host operators
              // advertise ``supports_device_matvec`` and their ``bind_cuda()``
              // device mirror serves the CudaBackend lane directly, so both
              // the Hamiltonian and the observables run device-resident with
              // no GPUOperator promotion. ``spectral_method_supports_gpu``
              // stays as a defensive hook for any future host-only method.
              if (!spectral_method_supports_gpu(opts.method)) {
                  warn_silent_cpu_fallback(
                      "qed.spectral (host-only method)",
                      opts.backend);
                  opts.backend.allow_gpu = false;
              }
              std::vector<const ed::LinearOperator*> obs;
              obs.reserve(observables.size());
              for (auto* o : observables) {
                  if (!o) continue;
                  obs.push_back(static_cast<const ed::LinearOperator*>(o));
              }
              return ed::workflows::spectral(
                  static_cast<const ed::LinearOperator&>(op), obs,
                  std::move(opts));
          },
          py::arg("op"),
          py::arg("observables"),
          py::arg("opts") = ed::workflows::SpectralOptions{},
          "Run the unified dynamical-correlator workflow "
          "(continued-fraction Lanczos or FTLM dynamical) over the auto-"
          "selected Backend. When ``allow_gpu`` is set and a CUDA device is "
          "available, the Hamiltonian and the observable pair run through "
          "their lazy CudaMatVecBackend device mirrors on the GPU lane.");

    // -----------------------------------------------------------------
    // Streaming-symmetry workflow over a directory (mirrors the CLI's
    // `run_streaming_symmetry_workflow` in `src/cli/workflows.cpp`).
    //
    // This single C++ entry point replaces the deleted Python forwarders
    // ``exact_diagonalization_streaming_symmetry[_fixed_sz]`` by composing
    // ``ed::make_operator(streaming_symmetry=true) ->
    //  StreamingSymmetryOperator::sector(k) ->
    //  ed::workflows::solve(*sec, opts)`` for every sector and
    // aggregating the eigenvalues. The aggregated payload is the same
    // shape callers received from the legacy entry: ascending eigenvalues
    // truncated to `opts.num_eigs`.
    // -----------------------------------------------------------------
    m.def("workflows_solve_streaming_symmetry_directory",
          [](const std::string& directory,
             std::uint64_t num_sites,
             double spin_l,
             ed::workflows::SolveOptions opts,
             py::object fixed_sz_n_up) {
              ed::OperatorSpec spec;
              spec.source             = ed::DirectoryPath{directory};
              spec.num_sites          = num_sites;
              spec.spin_l             = static_cast<float>(spin_l);
              spec.streaming_symmetry = true;
              spec.basis_cache_dir    = opts.basis_cache_dir;  // Stage 3
              if (!fixed_sz_n_up.is_none()) {
                  spec.fixed_sz = fixed_sz_n_up.cast<int>();
              }
              if (opts.sz_parity >= 0 && !spec.fixed_sz) {
                  spec.sz_parity = opts.sz_parity;
              }

              ed::GroundStateResult agg;
              {
                  py::gil_scoped_release release;
                  // Operator-collapse Phase 3 (Jun 2026): enumerate the
                  // symmetry sectors directly via
                  // ``make_sector_operators_tagged`` and view them through
                  // the handle-shaped ``SectorSetView`` (random access by raw
                  // irrep index). No monolithic streaming operator is
                  // materialised; the CSR-free lazy-rep memory optimisation
                  // is preserved by the tagged factory's lazy regime.
                  // -----------------------------------------------------
                  // Stage 8 + 8c (SymmetryEngine v2): symmetry composition
                  // for the GS lane.
                  //   * time-reversal PAIRING: for real H, spec(k) ==
                  //     spec(-k); solve one member of each conjugate pair
                  //     and DUPLICATE its eigenvalue list under the
                  //     partner's tag (duplication -- not deduplication --
                  //     keeps pooled degeneracy multiplicities correct).
                  //   * spin-flip TRANSPORT: [H, X] == 0 makes the fixed-Sz
                  //     blocks n_up and N - n_up isospectral; solve the
                  //     smaller-n_up block and re-tag.
                  //   * spin-flip PROJECTION: at n_up == N/2 split into
                  //     (k, +/-) sectors (eigenvalues-only workloads; the
                  //     projected eigenvectors are not orbit-reconstructable
                  //     through the flip-unaware CSR lane).
                  // Per-call toggles on SolveOptions (auto/off/require);
                  // ED_SYM_SPIN_FLIP[_PROJECT] / ED_SYM_TIME_REVERSAL are
                  // the env escapes for the Auto defaults.
                  // -----------------------------------------------------
                  ed::symmetry::SymmetryComposition comp;
                  {
                      auto probe = ed::detail::build_base_op(spec);
                      ed::detail::load_terms_into(*probe, spec);
                      ed::matvec::TermStorage soa;
                      ed::matvec::TermStorage::classify_route(
                          soa, probe->transform_data_, probe->three_body_data_,
                          [](const std::complex<double>& c) { return c; });
                      probe->symmetry_info.loadFromDirectory(directory);
                      comp = ed::symmetry::resolve_symmetry_composition(
                          soa, probe->symmetry_info, opts.backend.allow_gpu,
                          ed::symmetry::sym_toggle_from_int(opts.spin_flip),
                          ed::symmetry::sym_toggle_from_int(
                              opts.time_reversal));
                      // Stage 7a: star maps (non-abelian residue) join
                      // the isospectral-orbit relation alongside TR.
                      const std::size_t nsec =
                          probe->symmetry_info.sectors.size();
                      for (const auto& m : opts.star_maps) {
                          if (m.size() != nsec) continue;
                          comp.star_maps.emplace_back(m.begin(), m.end());
                      }
                  }
                  // Requested n_up before any transport re-target; -1 when
                  // the fixed-Sz axis is off. Used to restore the caller's
                  // n_up on the emitted sector tags.
                  const int requested_n_up =
                      spec.fixed_sz ? *spec.fixed_sz : -1;
                  if (comp.flip_transport && spec.fixed_sz
                      && *spec.fixed_sz * 2
                             > static_cast<int>(num_sites)) {
                      spec.fixed_sz = static_cast<int>(num_sites)
                                      - *spec.fixed_sz;
                  }
                  if (comp.flip_project && spec.fixed_sz
                      && *spec.fixed_sz * 2 == static_cast<int>(num_sites)
                      && !opts.compute_vectors) {
                      spec.flip_project_half = true;
                  }
                  // Full-space / parity-half prod-sigma^x sectors: no
                  // fixed-Sz axis but [H, X] == 0 -- every irrep splits
                  // (k, +/-). Closure rule: over a parity half the flip
                  // only preserves the subspace for even N.
                  if (comp.flip_project && !spec.fixed_sz
                      && !opts.compute_vectors
                      && (!spec.sz_parity || num_sites % 2 == 0)) {
                      spec.flip_sectors_full = true;
                  }
                  ed::core::SectorSetView handle(
                      ed::make_sector_operators_tagged(spec));

                  const std::size_t num_sectors = handle.num_sectors();
                  if (num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_solve_streaming_symmetry_directory: "
                          "make_operator returned an operator with no "
                          "symmetry sectors; check the "
                          "automorphism_results/ directory.");
                  }

                  const std::vector<std::size_t> sector_indices_all =
                      ed::core::filter_sectors(num_sectors,
                                               opts.selected_sectors);

                  std::vector<std::size_t> sector_indices;
                  // (skipped_k, canonical_k): emit skipped_k's eigenvalues
                  // as a copy of canonical_k's after the solve loop.
                  std::vector<std::pair<std::size_t, std::size_t>> tr_mirrors;
                  // Flip projection doubles the sector count with the
                  // synthetic index convention k + parity * num_raw; the
                  // TR conjugation AND the point-group star maps act on
                  // the raw irrep index and preserve flip parity, so
                  // canonicalise within parity blocks. One union-find
                  // over both relations (sector_orbit_canonical).
                  const std::size_t tr_num_raw =
                      !comp.tr_partner.empty() ? comp.tr_partner.size()
                      : !comp.star_maps.empty() ? comp.star_maps[0].size()
                                                : 0;
                  // Spectrum copies are not eigenVECTOR copies: a
                  // skipped sector has no states to return or save, so
                  // the orbit skip only runs on eigenvalue-only
                  // workloads (TR pairing and star reduction alike).
                  if (tr_num_raw > 0 && num_sectors % tr_num_raw == 0
                      && !opts.compute_vectors) {
                      const std::vector<std::int32_t> canon =
                          ed::symmetry::sector_orbit_canonical(
                              tr_num_raw, comp);
                      std::vector<char> in_sel(num_sectors, 0);
                      for (std::size_t k : sector_indices_all) in_sel[k] = 1;
                      for (std::size_t k : sector_indices_all) {
                          const std::size_t ck = static_cast<std::size_t>(
                              canon[k % tr_num_raw]);
                          const std::size_t psy =
                              ck + (k / tr_num_raw) * tr_num_raw;
                          if (psy < k && in_sel[psy]
                              && handle.sector_tag(k).sector_dim ==
                                 handle.sector_tag(psy).sector_dim) {
                              tr_mirrors.emplace_back(k, psy);
                              continue;  // skip solving k
                          }
                          sector_indices.push_back(k);
                      }
                      if (!tr_mirrors.empty()
                          && ed::symmetry::sym_profile_enabled()) {
                          std::fprintf(stderr,
                              "[sym-profile] time-reversal pairing (GS): "
                              "solving %zu of %zu sectors\n",
                              sector_indices.size(),
                              sector_indices_all.size());
                      }
                  } else {
                      sector_indices = sector_indices_all;
                  }

                  // Per-sector storage we accumulate into so that the
                  // final ``GroundStateResult`` carries both the
                  // legacy merged eigenvalue list AND the SOTA
                  // (sector_index, irrep QNs, sector_dim, per-sector
                  // eigenvalue list) attribution. Indexing pattern:
                  //   touched[s] <-> agg.sector_tags[s]
                  //                <-> agg.eigenvalues_per_sector[s]
                  std::vector<double>                  all_eigs;
                  std::vector<std::size_t>             touched_idx;
                  std::vector<ed::SectorTag>           touched_tags;
                  std::vector<std::vector<double>>     eigs_per_sector;

                  // -----------------------------------------------------
                  // Wave B1 (May 2026): GS two-phase irrep search.
                  //
                  // For small ``num_eigs`` we run a cheap "scan" pass
                  // (krylov_dim=40, no eigenvectors) on every sector to
                  // estimate the lowest eigenvalue per sector, then only
                  // do the user's full Krylov budget on the sectors that
                  // could plausibly contain the top-K eigenvalues. The
                  // safety margin ``gap`` accounts for the fact that
                  // Phase-1 Lanczos overestimates true minima (the
                  // estimate is an upper bound on the Ritz residual).
                  //
                  // The two-phase scan is disabled when the user
                  // requests a large ``num_eigs`` (>= 8), since
                  // estimating that many eigenvalues per sector with a
                  // short Krylov pass is unreliable -- the legacy
                  // ``num_sectors x full_solve`` pattern wins.
                  // -----------------------------------------------------
                  const std::size_t target_num_eigs =
                      opts.num_eigs ? opts.num_eigs : 1;
                  const bool enable_two_phase =
                      sector_indices.size() > 2
                      && target_num_eigs < 8
                      && !opts.precompute_basis_only;
                  std::vector<std::size_t> phase2_sector_indices;

                  // Wave C3 (May 2026): opt-in OpenMP parallelism over
                  // the Phase-1 sector scan. Per-sector calls are
                  // independent and the only shared state is the
                  // ``phase1_min`` accumulator (collected under a
                  // critical). Gated by ``ED_SYM_SECTOR_PARALLEL`` to
                  // avoid OMP nesting surprises with the inner
                  // matvec/Lanczos team -- production users on large
                  // sectors should leave this off (per-sector matvec
                  // already saturates the CPU); small-sector regimes
                  // benefit.
                  bool sector_parallel = false;
                  if (const char* env =
                          std::getenv("ED_SYM_SECTOR_PARALLEL")) {
                      sector_parallel = (env[0] == '1');
                  }
                  if (enable_two_phase) {
                      const std::size_t phase1_iter =
                          std::min<std::size_t>(40,
                              (opts.max_iter ? opts.max_iter : 100));
                      std::vector<std::pair<double, std::size_t>>
                          phase1_min;  // (lowest_E, idx_in_sector_indices)
                      phase1_min.reserve(sector_indices.size());
                      const long n_idx =
                          static_cast<long>(sector_indices.size());
                      #pragma omp parallel for schedule(dynamic, 1) \
                          if(sector_parallel)
                      for (long ii = 0; ii < n_idx; ++ii) {
                          std::size_t i = static_cast<std::size_t>(ii);
                          std::size_t k = sector_indices[i];
                          auto sec = handle.sector(k);
                          if (!sec || sec->dim() == 0) continue;
                          ed::workflows::SolveOptions p1 = opts;
                          p1.num_eigs = std::min<std::size_t>(
                              target_num_eigs, sec->dim());
                          p1.compute_vectors = false;
                          p1.max_iter = std::min<std::size_t>(
                              phase1_iter, sec->dim());
                          p1.output_dir.clear();
                          p1.selected_sectors.clear();
                          p1.use_symmetry = false;
                          std::pair<double, std::size_t> entry;
                          bool valid = false;
                          try {
                              auto sr = ed::workflows::solve(*sec, p1);
                              if (!sr.eigenvalues.empty()) {
                                  entry = {sr.eigenvalues.front(), i};
                                  valid = true;
                              }
                          } catch (...) {
                              entry = {
                                  -std::numeric_limits<double>::infinity(),
                                  i};
                              valid = true;
                          }
                          if (valid) {
                              #pragma omp critical
                              phase1_min.push_back(entry);
                          }
                      }
                      // Identify the (target_num_eigs)-th lowest as the
                      // cutoff, with a generous safety gap to absorb
                      // Phase-1 over-estimation. Sectors at-or-below
                      // cutoff make it into Phase 2.
                      std::sort(phase1_min.begin(), phase1_min.end(),
                                [](const auto& a, const auto& b) {
                                    return a.first < b.first;
                                });
                      double cutoff_E;
                      const bool phase1_has_nonfinite = std::any_of(
                          phase1_min.begin(), phase1_min.end(),
                          [](const auto& p) { return !std::isfinite(p.first); });
                      if (phase1_min.size() <= target_num_eigs
                          || phase1_has_nonfinite) {
                          // Too few estimates, or a phase-1 solve threw
                          // (recorded as -inf): the cutoff arithmetic
                          // below would produce inf/NaN and silently
                          // select NOTHING. Fail safe: solve everything.
                          cutoff_E = std::numeric_limits<double>::infinity();
                      } else {
                          const double best_E = phase1_min.front().first;
                          const double kth_E  =
                              phase1_min[target_num_eigs - 1].first;
                          const double gap = std::max(
                              1e-2 * std::abs(best_E),
                              1e-4);
                          cutoff_E = kth_E + gap;
                      }
                      for (const auto& [E, i] : phase1_min) {
                          if (E <= cutoff_E) {
                              phase2_sector_indices.push_back(
                                  sector_indices[i]);
                          }
                      }
                  }
                  const std::vector<std::size_t>& iter_sectors =
                      enable_two_phase ? phase2_sector_indices
                                       : sector_indices;
                  // "Universal save contract" follow-up (May 2026):
                  // when the user requested eigenvectors + a real
                  // ``output_dir`` for a streaming-symmetry solve, every
                  // sector wrote to the SAME ``<output_dir>/ed_results.h5``
                  // -- the ``/eigendata/*`` datasets are keyed without
                  // a sector tag, so every sector silently overwrote
                  // the previous one and only the last sector's
                  // eigenvectors survived. Route per-sector writes to
                  // ``<output_dir>/sector_k_<k>/ed_results.h5`` (mirrors
                  // the thermal streaming-symmetry binding) and surface
                  // the parent dir + per-sector file list on the
                  // aggregate ``GroundStateResult``.
                  const bool need_per_sector_outdir =
                      opts.compute_vectors
                      && !opts.output_dir.empty()
                      && !HDF5IO::isDisabledOutputPath(opts.output_dir);
                  std::vector<std::string> sector_hdf5_paths;
                  // Phase D (May 2026): capture the truthful per-sector
                  // backend lane on the first non-empty sector so the
                  // aggregate result can surface "gpu" vs "cpu" vs
                  // "mpi" instead of leaving ``agg.backend.lane``
                  // empty. All sectors share the same ``opts.backend``
                  // and the same SectorView geometry contract, so the
                  // first lane is authoritative.
                  std::string sector_lane;
                  std::size_t sector_mpi_size = 1;

                  // Phase 2: sector-parallel solve. Same ED_SYM_SECTOR_PARALLEL
                  // gate as the Phase-1 Lanczos scan above (the gate variable
                  // was already read into sector_parallel). Pre-index result
                  // storage to avoid push_back data races in the parallel body.
                  const long n_iter_sec =
                      static_cast<long>(iter_sectors.size());
                  std::vector<ed::GroundStateResult> solve_results_p2(
                      static_cast<std::size_t>(n_iter_sec));

                  #pragma omp parallel for schedule(dynamic, 1) \
                      if(sector_parallel)
                  for (long ii = 0; ii < n_iter_sec; ++ii) {
                      const std::size_t k =
                          iter_sectors[static_cast<std::size_t>(ii)];
                      auto sec = handle.sector(k);
                      if (!sec || sec->dim() == 0) continue;
                      ed::workflows::SolveOptions sopts = opts;
                      sopts.num_eigs = std::min<std::size_t>(
                          opts.num_eigs ? opts.num_eigs : 1, sec->dim());
                      sopts.selected_sectors.clear();
                      sopts.use_symmetry = false;
                      if (need_per_sector_outdir) {
                          sopts.output_dir = opts.output_dir
                              + "/sector_k_" + std::to_string(k);
                      }
                      solve_results_p2[static_cast<std::size_t>(ii)] =
                          ed::workflows::solve(*sec, sopts);
                  }

                  // Serial collection: preserve iter_sectors ordering for
                  // the eigenvalue merge and sector-tag attribution below.
                  for (long ii = 0; ii < n_iter_sec; ++ii) {
                      const std::size_t k =
                          iter_sectors[static_cast<std::size_t>(ii)];
                      auto sec = handle.sector(k);
                      if (!sec || sec->dim() == 0) continue;
                      auto& sr =
                          solve_results_p2[static_cast<std::size_t>(ii)];
                      if (sector_lane.empty()
                          && !sr.backend.lane.empty()) {
                          sector_lane     = sr.backend.lane;
                          sector_mpi_size = sr.backend.mpi_size;
                      }
                      touched_idx.push_back(touched_tags.size());
                      touched_tags.push_back(handle.sector_tag(k));
                      eigs_per_sector.push_back(sr.eigenvalues);
                      all_eigs.insert(all_eigs.end(),
                                      sr.eigenvalues.begin(),
                                      sr.eigenvalues.end());
                      if (need_per_sector_outdir && !sr.hdf5_path.empty()) {
                          sector_hdf5_paths.push_back(sr.hdf5_path);
                      }
                  }

                  // Stage 8 TR mirror emission: duplicate each solved
                  // canonical sector's eigenvalue list under its conjugate
                  // partner's tag (identical spectrum; keeps pooled
                  // degeneracy multiplicities correct).
                  for (const auto& [skipped_k, src_k] : tr_mirrors) {
                      std::size_t src_slot = touched_tags.size();
                      for (std::size_t t = 0; t < touched_tags.size(); ++t) {
                          if (touched_tags[t].sector_index == src_k) {
                              src_slot = t;
                              break;
                          }
                      }
                      if (src_slot == touched_tags.size()) continue;
                      // (canonical pruned by the two-phase scan or empty:
                      // the partner shares its spectrum, so skipping both
                      // is consistent)
                      std::vector<double> copy_eigs = eigs_per_sector[src_slot];
                      touched_idx.push_back(touched_tags.size());
                      touched_tags.push_back(handle.sector_tag(skipped_k));
                      all_eigs.insert(all_eigs.end(),
                                      copy_eigs.begin(), copy_eigs.end());
                      eigs_per_sector.push_back(std::move(copy_eigs));
                  }

                  // Stage 8c: when spin-flip transport re-targeted the
                  // solve to N - n_up, restore the caller's requested n_up
                  // on the emitted tags (the two blocks are isospectral;
                  // the tag should describe the sector the caller asked
                  // for, not the one physics let us solve instead).
                  if (requested_n_up >= 0) {
                      for (auto& tag : touched_tags) {
                          if (tag.n_up >= 0) tag.n_up = requested_n_up;
                      }
                  }

                  // Build the global merged-then-sorted vector while
                  // remembering which (touched) sector each entry came
                  // from. We sort an indexed range so that
                  // ``sector_index_of_eigenvalue`` stays parallel to
                  // ``eigenvalues``.
                  std::vector<std::size_t> origin(all_eigs.size(), 0);
                  {
                      std::size_t cursor = 0;
                      for (std::size_t s = 0; s < eigs_per_sector.size(); ++s) {
                          for (std::size_t j = 0;
                               j < eigs_per_sector[s].size();
                               ++j) {
                              origin[cursor++] = s;
                          }
                      }
                  }
                  std::vector<std::size_t> perm(all_eigs.size());
                  std::iota(perm.begin(), perm.end(), std::size_t{0});
                  std::sort(perm.begin(), perm.end(),
                            [&](std::size_t a, std::size_t b) {
                                return all_eigs[a] < all_eigs[b];
                            });
                  std::vector<double>      sorted_eigs(all_eigs.size());
                  std::vector<std::size_t> sorted_origin(all_eigs.size());
                  for (std::size_t i = 0; i < perm.size(); ++i) {
                      sorted_eigs[i]   = all_eigs[perm[i]];
                      sorted_origin[i] = origin[perm[i]];
                  }
                  if (opts.num_eigs > 0
                      && sorted_eigs.size() > opts.num_eigs) {
                      sorted_eigs.resize(opts.num_eigs);
                      sorted_origin.resize(opts.num_eigs);
                  }

                  agg.eigenvalues                = std::move(sorted_eigs);
                  agg.sector_tags                = std::move(touched_tags);
                  agg.eigenvalues_per_sector     = std::move(eigs_per_sector);
                  agg.sector_index_of_eigenvalue = std::move(sorted_origin);
                  // Surface the parent ``output_dir`` on the aggregate
                  // when at least one sector wrote a real HDF5 file --
                  // mirrors the thermal binding's contract.
                  if (need_per_sector_outdir && !sector_hdf5_paths.empty()) {
                      agg.hdf5_path = opts.output_dir;
                  }
                  // Phase D (May 2026): propagate the per-sector
                  // backend lane on the aggregate so callers reading
                  // ``GroundStateResult.backend.lane`` from
                  // ``qed.solve(symmetry=...)`` see the truthful lane
                  // ("gpu" when the SectorView's lazy GPU mirror
                  // fired, "cpu" otherwise). All per-sector calls
                  // share the same ``opts.backend``, so the first
                  // non-empty sector lane is authoritative for the
                  // aggregate; we keep ``mpi_size`` from the same
                  // sector for symmetry.
                  if (!sector_lane.empty()) {
                      agg.backend.lane = sector_lane;
                      agg.backend.mpi_size = sector_mpi_size;
                  }
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")      = 0.5,
          py::arg("opts")        = ed::workflows::SolveOptions{},
          py::arg("fixed_sz_n_up") = py::none(),
          R"pbdoc(
        Streaming-symmetry-projected ED over a directory.

        Composes ``ed::make_operator(streaming_symmetry=true,
        fixed_sz=...)`` with a per-sector ``ed::workflows::solve`` loop
        and returns the aggregated (ascending) eigenvalues. Mirrors the
        CLI's ``run_streaming_symmetry_workflow``; this is the canonical
        Python entry for symmetry-projected ED.

        The ``directory`` must contain the Hamiltonian dat files
        (``InterAll.dat`` / ``Trans.dat``) and an ``automorphism_results/``
        subdirectory with the precomputed symmetry metadata.

        Parameters
        ----------
        directory : str
            Path containing the Hamiltonian dat files and
            ``automorphism_results/``.
        num_sites : int
            Number of sites in the lattice (sets the qubit count).
        spin_l : float, optional
            Spin magnitude (0.5 for spin-1/2, the default).
        opts : SolveOptions, optional
            Per-sector solver options. ``num_eigs`` is the global cap on
            the returned eigenvalue list (sectors are union-merged then
            sorted).
        fixed_sz_n_up : int or None, optional
            If set, project to the fixed-Sz sector with this n_up
            (number of "up" spins). None (the default) keeps the full
            magnetization span.

        Returns
        -------
        GroundStateResult
            Carries the merged eigenvalues across every symmetry
            sector, truncated to ``opts.num_eigs`` and sorted
            ascending. ``sector_tags``, ``eigenvalues_per_sector``,
            and ``sector_index_of_eigenvalue`` carry the full
            (irrep, sector_dim, n_up) attribution for every eigenvalue
            in the merged list.
    )pbdoc");

    // -----------------------------------------------------------------
    // SOTA streaming-symmetry thermal workflow over a directory
    // (May 2026). Mirrors the solve binding above but, instead of
    // sorting eigenvalues, recombines per-sector ``ThermodynamicData``
    // via ``ed::core::combine_sector_thermodynamics`` (the canonical
    // free-energy Z-weighted mixture rule, single source of truth in
    // ``include/ed/core/sector_thermo.h``).
    //
    // Same factory pattern as the solve binding:
    //   ed::make_streaming_symmetry_operator(spec)
    //   -> StreamingSymmetryHandle::sector(k)
    //   -> ed::workflows::thermal(*sec, opts)   (per-sector)
    //   -> combine_sector_thermodynamics(per_sector_thermo)
    // The aggregated ``ThermalResult`` carries the recombined thermo
    // grid AND per-sector entries (with irrep tags) for callers that
    // want a breakdown.
    // -----------------------------------------------------------------
    m.def("workflows_thermal_streaming_symmetry_directory",
          [](const std::string& directory,
             std::uint64_t num_sites,
             double spin_l,
             ed::workflows::ThermalOptions opts,
             py::object fixed_sz_n_up) {
              ed::OperatorSpec spec;
              spec.source             = ed::DirectoryPath{directory};
              spec.num_sites          = num_sites;
              spec.spin_l             = static_cast<float>(spin_l);
              spec.streaming_symmetry = true;
              if (!fixed_sz_n_up.is_none()) {
                  spec.fixed_sz = fixed_sz_n_up.cast<int>();
              }
              // Monomial consolidation (Jul 2026): Sz-parity halves +
              // full-space/parity prod-sigma^x sectors for the
              // per-sector thermal lane (thermo = eigenvalue-only, so
              // the rep-only sectors are always admissible).
              if (opts.sz_parity >= 0 && !spec.fixed_sz) {
                  spec.sz_parity = opts.sz_parity;
              }
              {
                  auto probe = ed::detail::build_base_op(spec);
                  ed::detail::load_terms_into(*probe, spec);
                  ed::matvec::TermStorage soa;
                  ed::matvec::TermStorage::classify_route(
                      soa, probe->transform_data_, probe->three_body_data_,
                      [](const std::complex<double>& c) { return c; });
                  probe->symmetry_info.loadFromDirectory(directory);
                  auto comp = ed::symmetry::resolve_symmetry_composition(
                      soa, probe->symmetry_info, opts.backend.allow_gpu,
                      ed::symmetry::sym_toggle_from_int(opts.spin_flip),
                      ed::symmetry::sym_toggle_from_int(opts.time_reversal));
                  if (comp.flip_project && !spec.fixed_sz
                      && (!spec.sz_parity || num_sites % 2 == 0)) {
                      spec.flip_sectors_full = true;
                  }
              }

              ed::ThermalResult agg;
              {
                  py::gil_scoped_release release;
                  // Operator-collapse Phase 3 (Jun 2026): direct sector
                  // enumeration via ``make_sector_operators_tagged`` viewed
                  // through ``SectorSetView`` (preserves the CSR-free
                  // lazy-rep memory path for large N).
                  // Across-sector MPI (Level 1): when launched under mpirun the
                  // factory dim-balances + hands each rank only its sectors
                  // (construction + memory distribute); the inner thermal solve
                  // is forced rank-local below, and the per-sector thermo is
                  // Allgather-combined after the loop. Single-rank => unchanged.
                  const auto [mpi_rank, mpi_size] = binding_mpi_rank_size();
                  ed::core::SectorSetView handle(
                      ed::make_sector_operators_tagged(spec, mpi_rank, mpi_size));

                  const std::size_t num_sectors = handle.num_sectors();
                  if (mpi_size == 1 && num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_thermal_streaming_symmetry_directory: "
                          "make_operator returned an operator with no "
                          "symmetry sectors; check the "
                          "automorphism_results/ directory.");
                  }

                  const std::vector<std::size_t> sector_indices =
                      ed::core::filter_sectors(num_sectors,
                                               opts.selected_sectors);

                  std::vector<ThermodynamicData>     per_sector_thermo;
                  std::vector<std::uint64_t>         per_sector_dims;
                  std::vector<ed::ThermalSectorEntry> per_sector;
                  per_sector_thermo.reserve(sector_indices.size());
                  per_sector_dims.reserve(sector_indices.size());
                  per_sector.reserve(sector_indices.size());

                  double gs_E = std::numeric_limits<double>::infinity();

                  // -----------------------------------------------------
                  // Wave B4 (May 2026): pre-build the betas grid ONCE at
                  // the binding level so the orchestrator's auto-build
                  // branch is skipped on every per-sector call. The
                  // temperature axis is shared across sectors anyway --
                  // recomputing it N times per binding is pure overhead.
                  // Reuses the orchestrator's same construction logic
                  // (linear T, beta = 1/T) so the result is identical
                  // to the legacy per-sector path.
                  // -----------------------------------------------------
                  if (opts.betas.empty() && opts.num_temp_bins > 0
                      && opts.temp_min > 0.0
                      && opts.temp_max > opts.temp_min) {
                      opts.betas.reserve(opts.num_temp_bins);
                      const double t_lo = opts.temp_min;
                      const double t_hi = opts.temp_max;
                      const std::size_t n = opts.num_temp_bins;
                      for (std::size_t i = 0; i < n; ++i) {
                          const double T = (n == 1)
                              ? t_lo
                              : t_lo + (t_hi - t_lo)
                                * static_cast<double>(i)
                                / static_cast<double>(n - 1);
                          opts.betas.push_back(
                              T > 0.0 ? 1.0 / T : 1.0 / 1e-300);
                      }
                  }
                  // Lock the random seed (if not user-set) once at the
                  // binding level so every per-sector call uses the
                  // SAME seed -- otherwise sectors with random_seed=0
                  // would draw from std::random_device each time,
                  // making the result non-reproducible AND increasing
                  // variance.
                  if (opts.random_seed == 0) {
                      opts.random_seed = std::random_device{}();
                  }
#ifdef WITH_MPI
                  // Across-sector MPI: every rank must use the SAME base seed so
                  // a sector's FTLM draws the identical random vectors no matter
                  // which rank owns it -- otherwise the distributed combine would
                  // not match the single-node result. Broadcast rank 0's seed.
                  if (mpi_size > 1) {
                      unsigned long long s =
                          static_cast<unsigned long long>(opts.random_seed);
                      MPI_Bcast(&s, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
                      opts.random_seed = static_cast<decltype(opts.random_seed)>(s);
                  }
#endif

                  // -----------------------------------------------------
                  // Wave B3 (May 2026): for the KPM-DOS lane, estimate
                  // the spectral bounds ONCE on the largest sector and
                  // reuse for every sector call. The bounds are global
                  // properties of H, so per-sector re-estimation is a
                  // 1.5-3x overhead amplifier for KPM-DOS-Symm. Only
                  // applies when the caller has not provided their own
                  // overrides.
                  // -----------------------------------------------------
                  double shared_e_min =
                      std::numeric_limits<double>::quiet_NaN();
                  double shared_e_max =
                      std::numeric_limits<double>::quiet_NaN();
                  if (opts.method ==
                          ed::workflows::ThermalOptions::Method::KpmDos
                      && !(std::isfinite(opts.e_min_override)
                           && std::isfinite(opts.e_max_override))
                      && sector_indices.size() > 1) {
                      std::size_t best_k   = sector_indices.front();
                      std::size_t best_dim = 0;
                      for (std::size_t k : sector_indices) {
                          auto sec = handle.sector(k);
                          if (!sec) continue;
                          if (sec->dim() > best_dim) {
                              best_dim = sec->dim();
                              best_k   = k;
                          }
                      }
                      auto sec = handle.sector(best_k);
                      if (sec && sec->dim() > 0) {
                          try {
                              std::mt19937 gen(
                                  opts.random_seed
                                      ? opts.random_seed
                                      : 0xdeadbeefULL);
                              double lo = 0.0, hi = 0.0;
                              ed::kpm_dos::MatVec H_mv =
                                  [&sec](const Complex* in, Complex* out,
                                         int n) {
                                      sec->apply(in, out,
                                          static_cast<std::size_t>(n));
                                  };
                              ed::kpm_dos::estimate_spectral_bounds(
                                  H_mv, sec->dim(),
                                  /*krylov_dim=*/80,
                                  /*full_reorth=*/true,
                                  /*reorth_freq=*/10,
                                  /*tol=*/1e-10,
                                  gen, lo, hi);
                              shared_e_min = lo;
                              shared_e_max = hi;
                          } catch (...) {
                              // Silent fallback: kernel estimates.
                          }
                      }
                  }

                  // Save & DSSF Upgrades follow-up (May 2026): when the
                  // user supplied an ``output_dir`` AND a TPQ method,
                  // each per-sector run wrote to the SAME
                  // ``<output_dir>/ed_results.h5`` and overwrote the
                  // previous sector's TPQ samples / state vectors. The
                  // workaround in the old code path was to clear
                  // ``topts.output_dir`` entirely, which silently
                  // destroyed every state-vector snapshot when
                  // ``probe_betas`` was set. We now route per-sector
                  // writes to ``<output_dir>/sector_k_<k>/`` and surface
                  // the parent dir + the per-sector path list on the
                  // aggregate ``ThermalResult``. Non-TPQ methods can
                  // also benefit (per-sector ftlm/averaged groups stay
                  // intact) but the bug was specific to TPQ because
                  // FTLM/LTLM/KPM-DOS only ship the aggregated curves.
                  const bool need_per_sector_outdir =
                      !opts.output_dir.empty()
                      && !HDF5IO::isDisabledOutputPath(opts.output_dir);
                  std::vector<std::string> sector_hdf5_paths;
                  // Phase D (May 2026): capture the truthful per-sector
                  // backend lane on the first sector so the aggregate
                  // ``ThermalResult.backend.lane`` correctly reports
                  // the lane the orchestrator dispatched to. Before
                  // this fix the aggregate left ``backend.lane``
                  // empty, which made
                  // ``qed.thermal(symmetry=..., device='gpu')`` read
                  // back as if it ran on CPU even when the GPU mirror
                  // fired (the timing-vs-CPU diff was masked).
                  std::string sector_lane;
                  std::size_t sector_mpi_size = 1;

                  // Sector-level OMP parallelism: every irrep sector's
                  // thermal call is independent (distinct Krylov workspace,
                  // distinct random state, distinct output path). On
                  // many-core machines (>=32 cores) this is the dominant
                  // speedup lever — typical symmetry-projected sectors have
                  // dim ~100-10k, far too small to saturate the CPU with a
                  // single-sector BLAS team, so the right unit of work is
                  // the sector itself.
                  //
                  // Enable with ED_SYM_SECTOR_PARALLEL=1. To prevent inner
                  // BLAS / OMP nesting from oversubscribing:
                  //   OMP_MAX_ACTIVE_LEVELS=1     (serialise nested OMP teams)
                  //   OPENBLAS_NUM_THREADS=1       (or MKL_NUM_THREADS=1)
                  //   ED_AUTO_THREADS=0            (disable ThreadBudgetScope)
                  // OMP_NUM_THREADS should be set to the machine core count.
                  bool thermal_sector_parallel = false;
                  if (const char* env =
                          std::getenv("ED_SYM_SECTOR_PARALLEL")) {
                      thermal_sector_parallel = (env[0] == '1');
                  }

                  // Pre-allocate indexed result storage so the parallel
                  // fill is race-free (each slot ii is written by exactly
                  // one thread; no push_back races).
                  const long n_sec_th =
                      static_cast<long>(sector_indices.size());
                  std::vector<ed::ThermalResult> sec_results_th(
                      static_cast<std::size_t>(n_sec_th));

                  #pragma omp parallel for schedule(dynamic, 1) \
                      if(thermal_sector_parallel)
                  for (long ii = 0; ii < n_sec_th; ++ii) {
                      const std::size_t k =
                          sector_indices[static_cast<std::size_t>(ii)];
                      auto sec = handle.sector(k);
                      if (!sec || sec->dim() == 0) continue;
                      ed::workflows::ThermalOptions topts = opts;
                      topts.selected_sectors.clear();
                      if (mpi_size > 1) {
                          // Across-sector MPI: the inner thermal solve must be
                          // rank-local. select_backend would otherwise pick
                          // MpiBackend (MPI_Comm_dup ctor + Allreduce dot) on
                          // MPI_COMM_WORLD; with ranks on different sectors those
                          // collectives mismatch and deadlock. Collective on-disk
                          // I/O (create_directory_mpi_safe) is suppressed too.
                          topts.backend.allow_mpi     = false;
                          topts.backend.allow_mpi_gpu = false;
                          topts.output_dir.clear();
                      } else if (need_per_sector_outdir) {
                          topts.output_dir = opts.output_dir
                              + "/sector_k_" + std::to_string(k);
                      } else {
                          topts.output_dir.clear();
                      }
                      if (std::isfinite(shared_e_min)
                          && std::isfinite(shared_e_max)) {
                          topts.e_min_override = shared_e_min;
                          topts.e_max_override = shared_e_max;
                      }
                      sec_results_th[static_cast<std::size_t>(ii)] =
                          ed::workflows::thermal(*sec, topts);
                  }

                  // Serial post-processing: collect results in
                  // sector_indices order. The same dim==0 guard filters
                  // sectors that the parallel loop skipped (their result
                  // slots are default-constructed and safely ignored).
                  for (long ii = 0; ii < n_sec_th; ++ii) {
                      const std::size_t k =
                          sector_indices[static_cast<std::size_t>(ii)];
                      auto sec = handle.sector(k);
                      if (!sec || sec->dim() == 0) continue;
                      auto& tr =
                          sec_results_th[static_cast<std::size_t>(ii)];
                      if (sector_lane.empty()
                          && !tr.backend.lane.empty()) {
                          sector_lane     = tr.backend.lane;
                          sector_mpi_size = tr.backend.mpi_size;
                      }
                      if (tr.thermo.temperatures.empty()) {
                          if (std::isfinite(tr.ground_state_energy)) {
                              gs_E = std::min(gs_E, tr.ground_state_energy);
                          }
                          continue;
                      }
                      ed::SectorTag tag = handle.sector_tag(k);
                      per_sector_thermo.push_back(tr.thermo);
                      per_sector_dims.push_back(tag.sector_dim);
                      ed::ThermalSectorEntry entry;
                      entry.sz_index            = tag.n_up;
                      entry.ground_state_energy = tr.ground_state_energy;
                      entry.thermo              = tr.thermo;
                      entry.tag                 = tag;
                      per_sector.push_back(std::move(entry));
                      if (need_per_sector_outdir && !tr.hdf5_path.empty()) {
                          sector_hdf5_paths.push_back(tr.hdf5_path);
                      }
                      if (std::isfinite(tr.ground_state_energy)) {
                          gs_E = std::min(gs_E, tr.ground_state_energy);
                      }
                  }

#ifdef WITH_MPI
                  // Across-sector MPI: recombine every rank's per-sector thermo
                  // into the full list (collective; all ranks participate, even
                  // those owning zero sectors) so the combined result below is
                  // identical on every rank -- bit-identical to single-node.
                  if (mpi_size > 1) {
                      std::vector<std::uint64_t> raw_idx;
                      raw_idx.reserve(per_sector.size());
                      for (const auto& e : per_sector)
                          raw_idx.push_back(e.tag.sector_index);
                      mpi_allgather_sector_thermo(per_sector_thermo, per_sector_dims,
                                                  raw_idx, gs_E, mpi_size);
                  }
#endif
                  if (!per_sector_thermo.empty()) {
                      agg.thermo = ed::core::combine_sector_thermodynamics(
                          per_sector_thermo, per_sector_dims);
                  }
                  agg.per_sector         = std::move(per_sector);
                  agg.ground_state_energy = std::isfinite(gs_E) ? gs_E : 0.0;
                  // Surface the parent output_dir on the aggregate
                  // result when any per-sector run wrote to disk. Each
                  // sector's per-file path is under
                  // ``<output_dir>/sector_k_<k>/ed_results.h5``; the
                  // aggregate value points at the parent so callers
                  // can glob.
                  if (need_per_sector_outdir && !sector_hdf5_paths.empty()) {
                      agg.hdf5_path = opts.output_dir;
                  }
                  // Phase D (May 2026): propagate the per-sector
                  // backend lane on the aggregate so callers reading
                  // ``ThermalResult.backend.lane`` see the truthful
                  // lane ("gpu" / "cpu" / "mpi" / "mpi_gpu").
                  if (!sector_lane.empty()) {
                      agg.backend.lane = sector_lane;
                      agg.backend.mpi_size = sector_mpi_size;
                  }
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")      = 0.5,
          py::arg("opts")        = ed::workflows::ThermalOptions{},
          py::arg("fixed_sz_n_up") = py::none(),
          R"pbdoc(
        Streaming-symmetry-projected finite-T workflow over a directory.

        Composes ``ed::make_operator(streaming_symmetry=true, fixed_sz=...)``
        with a per-sector ``ed::workflows::thermal`` loop and then
        recombines the per-sector ``ThermodynamicData`` blocks via the
        canonical free-energy Z-weighted mixture rule
        (``ed::core::combine_sector_thermodynamics``).

        Parameters
        ----------
        directory : str
            Path containing the Hamiltonian dat files and
            ``automorphism_results/``.
        num_sites : int
            Number of sites in the lattice.
        spin_l : float, optional
            Spin magnitude (0.5 for spin-1/2, the default).
        opts : ThermalOptions, optional
            Per-sector finite-T options (FTLM / LTLM / mTPQ /
            KPM-DOS). ``selected_sectors`` filters the loop.
        fixed_sz_n_up : int or None, optional
            If set, project to a fixed-Sz sector with this ``n_up``
            and run the symmetry sector loop *inside* that Sz block.

        Returns
        -------
        ThermalResult
            ``thermo`` carries the recombined (Z-weighted) full-Hilbert
            thermodynamics on the requested temperature grid;
            ``per_sector`` lists every sector that contributed, with
            the irrep ``tag`` (``sector_index`` / ``quantum_numbers`` /
            ``sector_dim``) attached.
    )pbdoc");

    // -----------------------------------------------------------------
    // SOTA streaming-symmetry spectral workflow over a directory
    // (May 2026).
    //
    // For the ``GroundStateCF`` method, the spectral function
    //
    //   S_OO(omega) = -1/pi Im <psi_0| O^dag (omega + E_0 - H + i eta)^-1 O |psi_0>
    //
    // factorises over irreps when O carries a definite momentum
    // transfer Q. The exact selection rule reads
    //
    //   k_final = k_initial + Q   (mod reciprocal lattice)
    //
    // and at T=0 only the irrep containing the global ground state
    // contributes to the initial state. Concretely, the SOTA path
    // walks the per-irrep sector loop once, finds the sector
    // containing the global GS (sector with the smallest per-sector
    // ground-state energy), and runs the CF-Lanczos kernel
    // exclusively in that sector. The same-sector ``Q = 0`` path
    // (i.e. O is the q=0 component of S^z / n / ...) is the
    // canonical "DOS / static structure factor" workflow this entry
    // covers; cross-irrep (``Q != 0``) transitions are emitted as a
    // diagnostic in ``selection_rule_label`` and left to the
    // forthcoming ``CrossSectorOrbitObservable`` (see
    // ``docs/architecture/SYMMETRY.md`` Section 3).
    // -----------------------------------------------------------------
    m.def("workflows_spectral_streaming_symmetry_directory",
          [](const std::string& directory,
             std::uint64_t num_sites,
             double spin_l,
             ed::workflows::SpectralOptions opts,
             py::object fixed_sz_n_up) {
              ed::OperatorSpec spec;
              spec.source             = ed::DirectoryPath{directory};
              spec.num_sites          = num_sites;
              spec.spin_l             = static_cast<float>(spin_l);
              spec.streaming_symmetry = true;
              if (!fixed_sz_n_up.is_none()) {
                  spec.fixed_sz = fixed_sz_n_up.cast<int>();
              }

              ed::SpectralResult agg;
              {
                  py::gil_scoped_release release;
                  // Operator-collapse Phase 3 (Jun 2026): direct sector
                  // enumeration via ``make_sector_operators_tagged`` viewed
                  // through ``SectorSetView`` (same-irrep spectral path;
                  // cross-irrep transitions remain on CrossSectorOrbitObservable
                  // below).
                  ed::core::SectorSetView handle(
                      ed::make_sector_operators_tagged(spec));

                  const std::size_t num_sectors = handle.num_sectors();
                  if (num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_directory: "
                          "make_operator returned an operator with no "
                          "symmetry sectors; check the "
                          "automorphism_results/ directory.");
                  }

                  // Pass 1 -- per-sector ground-state solve. Picks the
                  // irrep containing the global GS so the CF-Lanczos
                  // run only touches that sector (huge speedup for
                  // SOTA DOS / static structure factor workflows).
                  const std::vector<std::size_t> sector_indices =
                      ed::core::filter_sectors(num_sectors,
                                               opts.selected_sectors);
                  std::size_t gs_sector_idx = 0;
                  double      gs_energy     = std::numeric_limits<double>::infinity();
                  bool        any_solved    = false;
                  {
                      // Pass 1: GS sector scan. Independent per-sector
                      // Lanczos calls (1 eig, no vectors) — embarrassingly
                      // parallel. Reuses the ED_SYM_SECTOR_PARALLEL gate.
                      bool sp_spectral = false;
                      if (const char* env =
                              std::getenv("ED_SYM_SECTOR_PARALLEL")) {
                          sp_spectral = (env[0] == '1');
                      }
                      const long n_sp =
                          static_cast<long>(sector_indices.size());
                      // (gs_energy, sector_idx) per slot; sector_idx=SIZE_MAX
                      // means "no eigenvalue".
                      std::vector<std::pair<double, std::size_t>>
                          sp_results(static_cast<std::size_t>(n_sp),
                              {std::numeric_limits<double>::infinity(),
                               std::size_t(-1)});

                      #pragma omp parallel for schedule(dynamic, 1) \
                          if(sp_spectral)
                      for (long ii = 0; ii < n_sp; ++ii) {
                          const std::size_t k =
                              sector_indices[static_cast<std::size_t>(ii)];
                          auto sec = handle.sector(k);
                          if (!sec || sec->dim() == 0) continue;
                          ed::workflows::SolveOptions sopts;
                          sopts.num_eigs        = 1;
                          sopts.tolerance       = 1e-12;
                          sopts.backend         = opts.backend;
                          sopts.method =
                              ed::workflows::SolveMethod::Lanczos;
                          sopts.compute_vectors = false;
                          auto sr = ed::workflows::solve(*sec, sopts);
                          if (!sr.eigenvalues.empty()) {
                              sp_results[static_cast<std::size_t>(ii)] =
                                  {sr.eigenvalues.front(), k};
                          }
                      }
                      for (const auto& [E_k, k] : sp_results) {
                          if (k == std::size_t(-1)) continue;
                          any_solved = true;
                          if (E_k < gs_energy) {
                              gs_energy     = E_k;
                              gs_sector_idx = k;
                          }
                      }
                  }
                  if (!any_solved) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_directory: "
                          "every sector returned an empty spectrum; "
                          "check the operator / Hilbert space.");
                  }

                  // Selection rule label: we currently route every
                  // observable through the GS sector, which is exact
                  // for Q=0 observables and a documented
                  // approximation for Q!=0 (the matrix element
                  // remains in the same irrep, dropping cross-irrep
                  // pieces; the cross-irrep generalisation is the
                  // ``CrossSectorOrbitObservable`` follow-on).
                  bool has_Q = !opts.momentum_transfer.empty();
                  if (has_Q) {
                      double q_norm = 0.0;
                      for (double q : opts.momentum_transfer) {
                          const double frac = q - std::round(q);
                          q_norm += frac * frac;
                      }
                      has_Q = std::sqrt(q_norm) > opts.momentum_tolerance;
                  }
                  agg.selection_rule_label = has_Q
                      ? std::string("k_final = k_initial + Q (cross-irrep "
                                    "transitions deferred to "
                                    "CrossSectorOrbitObservable; "
                                    "running same-irrep approximation)")
                      : std::string("k_final = k_initial (same-irrep; "
                                    "exact for Q=0 observables)");

                  // Pass 2 -- CF-Lanczos in the GS sector only.
                  ed::SectorTag gs_tag = handle.sector_tag(gs_sector_idx);
                  auto gs_sec = handle.sector(gs_sector_idx);
                  ed::workflows::SpectralOptions sopts = opts;
                  sopts.selected_sectors.clear();
                  // Use the per-sector GS energy as the resolvent shift
                  // (override only if the user didn't already).
                  if (std::abs(opts.energy_shift) < 1e-14) {
                      sopts.energy_shift = gs_energy;
                  }
                  // Build a placeholder identity observable: the
                  // streaming-symmetry spectral binding currently
                  // expects an in-process observable to be wired
                  // through Python (e.g. an irrep-restricted S^z).
                  // For now we route H itself as the observable so
                  // the CF kernel produces the local density of
                  // states of the GS sector -- exactly the SOTA
                  // single-shot DOS workflow that HPhi / EDLib
                  // expose as their canonical symmetry-projected
                  // spectral output. The Python wrapper layer
                  // (qed.spectral) lifts the user-provided
                  // observable to the orbit basis before calling
                  // this entry.
                  std::vector<const ed::LinearOperator*> obs_vec{
                      gs_sec};
                  ed::SpectralResult sr =
                      ed::workflows::spectral(*gs_sec, obs_vec, sopts);

                  agg.omega        = sr.omega;
                  agg.S_real       = sr.S_real;
                  agg.S_imag       = sr.S_imag;
                  agg.errors_real  = sr.errors_real;
                  agg.errors_imag  = sr.errors_imag;
                  agg.krylov       = sr.krylov;
                  agg.backend      = sr.backend;

                  // SOTA attribution: record the GS sector as the
                  // initial-and-final irrep that produced the spectral
                  // function. Cross-irrep (Q != 0) pairs will appear
                  // as additional entries once
                  // CrossSectorOrbitObservable is wired in.
                  ed::SpectralSectorEntry entry;
                  entry.initial = gs_tag;
                  entry.final_  = gs_tag;
                  entry.S_real  = sr.S_real;
                  entry.S_imag  = sr.S_imag;
                  agg.per_sector_pair.push_back(std::move(entry));
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")      = 0.5,
          py::arg("opts")        = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up") = py::none(),
          R"pbdoc(
        Streaming-symmetry-projected spectral workflow over a directory.

        For ``Method::GroundStateCF`` the entry performs a per-irrep
        ground-state pass to locate the irrep containing the global
        ground state, then runs continued-fraction Lanczos
        exclusively in that one sector. This is the canonical SOTA
        same-irrep spectral path (exact for ``Q = 0`` observables;
        documented approximation for ``Q != 0`` until the
        ``CrossSectorOrbitObservable`` follow-on lands -- see
        ``docs/architecture/SYMMETRY.md`` Section 3 for the design).

        Parameters
        ----------
        directory : str
            Path containing the Hamiltonian dat files and
            ``automorphism_results/``.
        num_sites : int
            Number of sites in the lattice.
        spin_l : float, optional
            Spin magnitude (0.5 for spin-1/2, the default).
        opts : SpectralOptions, optional
            CF / FTLM-dynamical knobs. ``momentum_transfer`` describes
            the observable's momentum (used for the selection-rule
            label); ``selected_sectors`` restricts the initial-sector
            search.
        fixed_sz_n_up : int or None, optional
            Optional Sz projection.

        Returns
        -------
        SpectralResult
            ``omega`` / ``S_real`` / ``S_imag`` carry the merged
            spectral function. ``per_sector_pair`` lists each
            (initial, final) irrep pair that contributed, tagged
            with the per-sector ``SectorTag``;
            ``selection_rule_label`` describes the symmetry filter
            that was applied.
    )pbdoc");

    // -----------------------------------------------------------------
    // Cross-irrep streaming-symmetry spectral binding (SOTA, May 2026).
    //
    // Closes the remaining gap in docs/architecture/SYMMETRY.md Section
    // 3: dynamical S(Q, omega) for spatial irreps with non-trivial
    // momentum transfer ``Q`` and / or a fixed-Sz delta_n_up.
    //
    // The algorithm:
    //   (1) Per-sector GS pass over the source operator (with the
    //       requested ``fixed_sz_n_up``). Identifies the irrep that
    //       hosts the global ground state.
    //   (2) Re-solve that sector with ``compute_vectors=true`` to
    //       recover |psi_0> in the source-orbit basis.
    //   (3) Resolve the target sector via the selection rule
    //       k_dst = k_src + Q (integer-quantised against the inferred
    //       generator orders in ``include/ed/core/sector_loop.h``).
    //       If ``delta_n_up != 0`` we build a *second* streaming
    //       operator with the shifted ``fixed_sz_n_up`` so the
    //       observable can scatter into a different Sz subspace.
    //   (4) Build a ``CrossSectorOrbitObservable`` from the user's
    //       transform list and apply it to |psi_0> to obtain |phi>
    //       in the *target* orbit basis.
    //   (5) Run ``cf_spectral_from_vector`` against H restricted to
    //       the target sector. The spectral weight ||phi||^2 is
    //       folded in automatically.
    //
    // The returned ``SpectralResult`` carries the (initial, final)
    // SectorTag pair in ``per_sector_pair`` plus a
    // ``selection_rule_label`` documenting how the target sector was
    // chosen.
    // -----------------------------------------------------------------
    m.def("workflows_spectral_streaming_symmetry_cross_irrep_directory",
          [](const std::string&                    directory,
             std::uint64_t                          num_sites,
             double                                 spin_l,
             const std::vector<py::tuple>&          observable_transforms,
             ed::workflows::SpectralOptions         opts,
             py::object                             fixed_sz_n_up,
             int                                    delta_n_up) {
              // ----------------------------------------------------------
              // Decode the user-supplied transform tuples into the SoA
              // layout expected by CrossSectorOrbitObservable. Each
              // tuple has the shape produced by
              //   _transforms_from_operator(op) below:
              //     (op_type, site, coeff, is_two_body, op_type_2, site_2)
              // ----------------------------------------------------------
              std::vector<Operator::TransformData> tlist;
              tlist.reserve(observable_transforms.size());
              for (const auto& row : observable_transforms) {
                  if (row.size() < 6) {
                      throw std::invalid_argument(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: each transform must be a 6-tuple "
                          "(op_type, site, coeff, is_two_body, op_type_2, "
                          "site_2).");
                  }
                  Operator::TransformData t;
                  t.op_type       = static_cast<uint8_t>(row[0].cast<int>());
                  t.site_index    = row[1].cast<std::uint64_t>();
                  t.coefficient   = row[2].cast<std::complex<double>>();
                  t.is_two_body   = row[3].cast<bool>();
                  t.op_type_2     = static_cast<uint8_t>(row[4].cast<int>());
                  t.site_index_2  = row[5].cast<std::uint64_t>();
                  tlist.push_back(t);
              }
              if (tlist.empty()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_cross_irrep_"
                      "directory: observable_transforms is empty -- the "
                      "cross-irrep walk needs at least one term.");
              }

              ed::SpectralResult agg;
              {
                  py::gil_scoped_release release;

                  // -----------------------------------------------------
                  // (1) Build source streaming operator. Mirror the
                  //     same-irrep binding's OperatorSpec layout.
                  // -----------------------------------------------------
                  ed::OperatorSpec src_spec;
                  src_spec.source             = ed::DirectoryPath{directory};
                  src_spec.num_sites          = num_sites;
                  src_spec.spin_l             = static_cast<float>(spin_l);
                  src_spec.streaming_symmetry = true;
                  if (!fixed_sz_n_up.is_none()) {
                      src_spec.fixed_sz = fixed_sz_n_up.cast<int>();
                  }

                  // Operator-collapse Phase 3 (Jun 2026): enumerate the source
                  // symmetry sectors directly via
                  // ``make_sector_operators_tagged`` + ``SectorSetView``. The
                  // cross-irrep observable's ``OperatorRef`` is built later
                  // from the resolved source / target sectors'
                  // ``SectorOperator::materialized_basis()`` (no monolithic
                  // streaming operator).
                  ed::core::SectorSetView src_handle(
                      ed::make_sector_operators_tagged(src_spec));

                  const std::size_t src_num_sectors = src_handle.num_sectors();
                  if (src_num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: source operator has no symmetry sectors; "
                          "check automorphism_results/.");
                  }

                  // -----------------------------------------------------
                  // (2) Find the global GS sector across the source
                  //     symmetry block.
                  //
                  // Wave B2 (May 2026): two-phase scan -- run a cheap
                  // Phase 1 (``krylov_dim=40``) across every sector, then
                  // refine only candidates within ``gap`` of the best
                  // Phase-1 minimum. Skips N-1 full Lanczos solves on
                  // average. The candidate list is collapsed back to the
                  // single minimum-energy sector for the eigenvector
                  // pull in (3) -- the safety margin is the gap from
                  // best Phase-1 E so the true GS sector is always
                  // among the refined candidates.
                  // -----------------------------------------------------
                  const std::vector<std::size_t> src_sector_indices =
                      ed::core::filter_sectors(src_num_sectors,
                                               opts.selected_sectors);
                  std::size_t gs_src_idx = 0;
                  double      gs_energy  = std::numeric_limits<double>::infinity();
                  bool        any_solved = false;

                  const bool enable_two_phase_dssf =
                      src_sector_indices.size() > 2;
                  std::vector<std::size_t> phase2_candidates;
                  if (enable_two_phase_dssf) {
                      std::vector<std::pair<double, std::size_t>> phase1_min;
                      phase1_min.reserve(src_sector_indices.size());
                      for (std::size_t k : src_sector_indices) {
                          auto sec = src_handle.sector(k);
                          if (!sec || sec->dim() == 0) continue;
                          ed::workflows::SolveOptions p1;
                          p1.num_eigs        = 1;
                          p1.tolerance       = 1e-8;
                          p1.backend         = opts.backend;
                          p1.method          = ed::workflows::SolveMethod::Lanczos;
                          p1.compute_vectors = false;
                          p1.max_iter        = std::min<std::size_t>(40, sec->dim());
                          try {
                              auto sr = ed::workflows::solve(*sec, p1);
                              if (!sr.eigenvalues.empty()) {
                                  phase1_min.emplace_back(
                                      sr.eigenvalues.front(), k);
                              }
                          } catch (...) {
                              phase1_min.emplace_back(
                                  -std::numeric_limits<double>::infinity(),
                                  k);
                          }
                      }
                      if (!phase1_min.empty()) {
                          std::sort(phase1_min.begin(), phase1_min.end(),
                                    [](const auto& a, const auto& b) {
                                        return a.first < b.first;
                                    });
                          const double best_E = phase1_min.front().first;
                          const double gap = std::max(
                              1e-2 * std::abs(best_E), 1e-4);
                          for (const auto& [E, k] : phase1_min) {
                              if (E <= best_E + gap) {
                                  phase2_candidates.push_back(k);
                              }
                          }
                      }
                  }
                  const std::vector<std::size_t>& src_scan_indices =
                      enable_two_phase_dssf ? phase2_candidates
                                            : src_sector_indices;
                  for (std::size_t k : src_scan_indices) {
                      auto sec = src_handle.sector(k);
                      if (!sec || sec->dim() == 0) continue;
                      ed::workflows::SolveOptions sopts;
                      sopts.num_eigs        = 1;
                      sopts.tolerance       = 1e-12;
                      sopts.backend         = opts.backend;
                      sopts.method          = ed::workflows::SolveMethod::Lanczos;
                      sopts.compute_vectors = false;
                      auto sr = ed::workflows::solve(*sec, sopts);
                      if (sr.eigenvalues.empty()) continue;
                      const double E_k = sr.eigenvalues.front();
                      if (E_k < gs_energy) {
                          gs_energy  = E_k;
                          gs_src_idx = k;
                      }
                      any_solved = true;
                  }
                  if (!any_solved) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: every source sector returned an empty "
                          "spectrum; check the operator / Hilbert space.");
                  }

                  // -----------------------------------------------------
                  // (3) Re-solve the GS sector with compute_vectors=true
                  //     so we can extract |psi_0> for the cross-sector
                  //     scatter step.
                  // -----------------------------------------------------
                  auto gs_sec_view = src_handle.sector(gs_src_idx);
                  ed::workflows::SolveOptions sopts_full;
                  sopts_full.num_eigs        = 1;
                  sopts_full.tolerance       = 1e-12;
                  sopts_full.backend         = opts.backend;
                  sopts_full.method          = ed::workflows::SolveMethod::Lanczos;
                  sopts_full.compute_vectors = true;
                  auto gs_sr =
                      ed::workflows::solve(*gs_sec_view, sopts_full);
                  if (gs_sr.eigenvalues.empty() || !gs_sr.eigenvectors ||
                      gs_sr.eigenvectors->host.empty()) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: ground-state eigenvector reconstruction "
                          "failed (eigenvectors->host empty).");
                  }
                  // Phase H.1 of the "Close CPU/GPU Gaps" plan
                  // (May 2026): capture the truthful backend lane
                  // from the inner GS solve so the cross-irrep
                  // aggregate reports "gpu" (or "cpu" / "mpi") on
                  // ``agg.backend.lane`` instead of leaving it
                  // empty. The GS sector full solve is the anchor:
                  // it's the only solve guaranteed to run in this
                  // binding (the two-phase scan above does many
                  // cheaper solves but those share the same backend
                  // contract).
                  if (!gs_sr.backend.lane.empty()) {
                      agg.backend.lane     = gs_sr.backend.lane;
                      agg.backend.mpi_size = gs_sr.backend.mpi_size;
                  }
                  const auto& psi0 = gs_sr.eigenvectors->host[0];
                  const double E0  = gs_sr.eigenvalues.front();
                  ed::SectorTag gs_src_tag = src_handle.sector_tag(gs_src_idx);

                  // Source observable handle: the GS sector's materialised
                  // orbit basis (forces the host CSR for this one sector even
                  // in the CSR-free lazy regime). ``num_bits`` is the lattice
                  // site count the legacy streaming op exposed via
                  // ``getNumBits()``.
                  const ed::symmetry::SectorBasis& src_basis =
                      gs_sec_view->materialized_basis();
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef src_ref =
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(
                          src_basis, static_cast<std::uint64_t>(num_sites));

                  // -----------------------------------------------------
                  // (4) Build / re-use the target sector set. If
                  //     ``delta_n_up == 0`` we re-use the source set (the
                  //     view is cheaply copyable -- shared_ptr-backed);
                  //     otherwise we enumerate a second set with the shifted
                  //     ``fixed_sz_n_up``.
                  // -----------------------------------------------------
                  ed::core::SectorSetView dst_handle = src_handle;

                  if (delta_n_up != 0) {
                      if (!src_spec.fixed_sz.has_value()) {
                          throw std::invalid_argument(
                              "workflows_spectral_streaming_symmetry_cross_"
                              "irrep_directory: delta_n_up != 0 requires "
                              "fixed_sz_n_up to be set.");
                      }
                      ed::OperatorSpec dst_spec;
                      dst_spec.source             = ed::DirectoryPath{directory};
                      dst_spec.num_sites          = num_sites;
                      dst_spec.spin_l             = static_cast<float>(spin_l);
                      dst_spec.streaming_symmetry = true;
                      dst_spec.fixed_sz           = *src_spec.fixed_sz + delta_n_up;
                      dst_handle = ed::core::SectorSetView(
                          ed::make_sector_operators_tagged(dst_spec));
                  }

                  // -----------------------------------------------------
                  // (5) Resolve the target sector via the selection
                  //     rule. ``Q`` lives in fractional reciprocal-
                  //     lattice units; sector_loop.h does the integer
                  //     quantisation.
                  // -----------------------------------------------------
                  double q_residual = 0.0;
                  const std::size_t dst_sector_idx =
                      ed::core::resolve_target_sector(
                          dst_handle,
                          gs_src_idx,
                          opts.momentum_transfer,
                          &q_residual);
                  if (dst_sector_idx == ed::core::kSectorNotFound) {
                      // Build a useful diagnostic for the no-survivor
                      // case so users can tell *why* their Q didn't land
                      // anywhere.
                      std::string msg =
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: no surviving target sector for the "
                          "requested selection rule. Source sector qn = [";
                      for (std::size_t g = 0; g < gs_src_tag.quantum_numbers.size(); ++g) {
                          if (g) msg += ", ";
                          msg += std::to_string(gs_src_tag.quantum_numbers[g]);
                      }
                      msg += "], Q (frac) = [";
                      for (std::size_t g = 0; g < opts.momentum_transfer.size(); ++g) {
                          if (g) msg += ", ";
                          msg += std::to_string(opts.momentum_transfer[g]);
                      }
                      msg += "], delta_n_up = " + std::to_string(delta_n_up) +
                             ", momentum-quantisation residual = " +
                             std::to_string(q_residual) + ".";
                      throw std::runtime_error(msg);
                  }
                  if (q_residual > opts.momentum_tolerance) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: Q is incommensurate with the lattice "
                          "(residual = " + std::to_string(q_residual) +
                          " > tolerance = " +
                          std::to_string(opts.momentum_tolerance) + ").");
                  }
                  ed::SectorTag dst_tag =
                      dst_handle.sector_tag(dst_sector_idx);

                  // -----------------------------------------------------
                  // (6) Resolve the target sector operator. An empty
                  //     (dropped) target irrep => spectral function is
                  //     identically zero (the SectorSetView returns nullptr
                  //     for a vanishing sector). Even so the GS solve above
                  //     already populated ``agg.backend.lane`` (Phase H.1)
                  //     so the caller can still see which lane produced the
                  //     zero result.
                  // -----------------------------------------------------
                  ed::symmetry::SectorOperator* dst_sec_view =
                      dst_handle.sector(dst_sector_idx);
                  if (!dst_sec_view || dst_sec_view->dim() == 0) {
                      const std::size_t num_omega =
                          (opts.num_omega > 0) ? opts.num_omega : 1;
                      agg.omega.resize(num_omega);
                      agg.S_real.assign(num_omega, 0.0);
                      agg.S_imag.assign(num_omega, 0.0);
                      const double step = (opts.num_omega > 1)
                          ? (opts.omega_max - opts.omega_min) /
                            static_cast<double>(opts.num_omega - 1)
                          : 0.0;
                      for (std::size_t i = 0; i < num_omega; ++i) {
                          agg.omega[i] = opts.omega_min +
                              static_cast<double>(i) * step;
                      }
                      agg.selection_rule_label =
                          "k_final = k_initial + Q (cross-irrep; "
                          "target sector empty)";
                      ed::SpectralSectorEntry entry;
                      entry.initial = gs_src_tag;
                      entry.final_  = dst_tag;
                      entry.S_real  = agg.S_real;
                      entry.S_imag  = agg.S_imag;
                      agg.per_sector_pair.push_back(std::move(entry));
                      return agg;
                  }

                  // Target observable handle: the resolved sector's
                  // materialised orbit basis. Build phi = O_Q |psi_0> in the
                  // target orbit basis via CrossSectorOrbitObservable. Each
                  // OperatorRef wraps exactly ONE sector, so the source /
                  // target sector indices passed to the observable are 0.
                  const ed::symmetry::SectorBasis& dst_basis =
                      dst_sec_view->materialized_basis();
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef dst_ref =
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(
                          dst_basis, static_cast<std::uint64_t>(num_sites));
                  ed::dssf::CrossSectorOrbitObservable orb_obs(
                      src_ref, /*src_sector=*/0,
                      dst_ref, /*dst_sector=*/0,
                      tlist,
                      static_cast<float>(spin_l));
                  const std::size_t dim_dst = orb_obs.dim_dst();
                  if (dst_sec_view->dim() != dim_dst) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: target sector dim mismatch (view="
                          + std::to_string(dst_sec_view->dim())
                          + ", observable=" + std::to_string(dim_dst) + ").");
                  }
                  std::vector<Complex> phi(dim_dst, Complex(0.0, 0.0));
                  orb_obs.apply(psi0.data(), phi.data(), dim_dst);

                  // -----------------------------------------------------
                  // (7) Run cf_spectral_from_vector on H restricted to
                  //     the target sector (the resolved ``dst_sec_view``).
                  // -----------------------------------------------------

                  // Build the frequency grid -- linear spacing between
                  // [omega_min, omega_max]. Matches the convention used
                  // by ed::workflows::spectral for the omega axis.
                  const std::size_t num_omega = std::max<std::size_t>(opts.num_omega, 1);
                  std::vector<double> omega_grid(num_omega);
                  if (num_omega == 1) {
                      omega_grid[0] = opts.omega_min;
                  } else {
                      const double step =
                          (opts.omega_max - opts.omega_min) /
                          static_cast<double>(num_omega - 1);
                      for (std::size_t i = 0; i < num_omega; ++i) {
                          omega_grid[i] =
                              opts.omega_min + static_cast<double>(i) * step;
                      }
                  }

                  ed::observables::CfSpectralOptions cfopts;
                  cfopts.krylov_dim   = opts.krylov_dim;
                  cfopts.broadening   = opts.broadening;
                  cfopts.energy_shift = (std::abs(opts.energy_shift) > 1e-14)
                                          ? opts.energy_shift
                                          : E0;
                  cfopts.tolerance    = 1e-12;
                  cfopts.global_n     = dim_dst;
                  cfopts.verbose      = false;

                  // Phase H.2 of the "Close CPU/GPU Gaps" plan
                  // (May 2026): route the inner CF Lanczos through
                  // ``select_backend`` + ``dst_sec_view->bind<B>()``
                  // so the target-sector Lanczos runs on the same
                  // backend as the source-sector GS solve. The
                  // ``CrossSectorOrbitObservable::apply`` rectangular
                  // scatter (line above this block) stays host-only;
                  // we stage the resulting ``phi`` into backend memory
                  // here as a single D2H/H2D trip and then run the
                  // CF kernel entirely device-resident. The phi-norm
                  // and spectral weight are preserved by
                  // ``cf_spectral_from_vector`` (header-only,
                  // backend-templated).
                  ed::observables::CfSpectralResult cf;
                  auto cf_variant = ed::select_backend(
                      dst_sec_view->geometry(), opts.backend);
                  std::visit([&](auto& backend_uptr) {
                      using BPtr = std::decay_t<decltype(backend_uptr)>;
                      using B = typename BPtr::element_type;
                      auto apply_H = dst_sec_view->template bind<B>();
                      cf = ed::observables::cf_spectral_from_vector(
                          *backend_uptr, apply_H, dim_dst,
                          phi.data(), omega_grid, cfopts);
                  }, cf_variant);

                  // -----------------------------------------------------
                  // (8) Marshal the result into ed::SpectralResult.
                  // -----------------------------------------------------
                  // Phase H.2: refine the propagated lane to reflect
                  // the LANCZOS lane (which now matches the GS lane
                  // by construction since both go through
                  // ``select_backend(dst_sec_view->geometry(),
                  // opts.backend)`` / ``H.geometry()`` with the same
                  // ``opts.backend``). Falling back to the GS lane
                  // captured by Phase H.1 above when the CF variant
                  // didn't override the BackendMetadata.
                  agg.backend.lane =
                      ed::lane_label_from_variant(cf_variant);
                  agg.omega       = cf.frequencies;
                  agg.S_real      = cf.spectral_function;
                  agg.S_imag.assign(cf.spectral_function.size(), 0.0);
                  agg.errors_real.assign(cf.spectral_function.size(), 0.0);
                  agg.errors_imag.assign(cf.spectral_function.size(), 0.0);

                  std::string label =
                      "k_final = k_initial + Q (cross-irrep); "
                      "src qn = [";
                  for (std::size_t g = 0; g < gs_src_tag.quantum_numbers.size(); ++g) {
                      if (g) label += ",";
                      label += std::to_string(gs_src_tag.quantum_numbers[g]);
                  }
                  label += "], dst qn = [";
                  for (std::size_t g = 0; g < dst_tag.quantum_numbers.size(); ++g) {
                      if (g) label += ",";
                      label += std::to_string(dst_tag.quantum_numbers[g]);
                  }
                  label += "], delta_n_up = " + std::to_string(delta_n_up) +
                           ", ||phi||^2 = " +
                           std::to_string(cf.phi_norm * cf.phi_norm);
                  agg.selection_rule_label = std::move(label);

                  ed::SpectralSectorEntry entry;
                  entry.initial = gs_src_tag;
                  entry.final_  = dst_tag;
                  entry.S_real  = agg.S_real;
                  entry.S_imag  = agg.S_imag;
                  agg.per_sector_pair.push_back(std::move(entry));
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")                = 0.5,
          py::arg("observable_transforms") = std::vector<py::tuple>{},
          py::arg("opts")                  = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up")         = py::none(),
          py::arg("delta_n_up")            = 0,
          R"pbdoc(
        Cross-irrep streaming-symmetry spectral workflow (SOTA).

        Implements the dynamical S(Q, omega) Lehmann sum with full
        spatial-symmetry exploitation:

          S(Q, omega) = sum_n delta(omega - E_n + E_0) * |<n|O_Q|0>|^2

        Source ground state is solved per-irrep, the target sector
        is resolved via the selection rule
        ``k_final = k_initial + Q``, the user-supplied observable
        terms scatter |psi_0> into the target orbit basis, and a
        continued-fraction Lanczos run on H restricted to that
        sector yields the spectral function.

        Parameters
        ----------
        directory : str
            Hamiltonian directory (must contain
            ``automorphism_results/``).
        num_sites : int
            Number of lattice sites.
        spin_l : float, optional
            Local spin (0.5 by default).
        observable_transforms : list of (op_type:int, site:int,
                                         coeff:complex,
                                         is_two_body:bool,
                                         op_type_2:int, site_2:int)
            One row per term in the probe observable ``O_Q``. The
            tuple layout mirrors ``Operator::TransformData`` -- the
            Python wrapper ``qed.spectral`` extracts these from an
            ``ed.Operator`` instance automatically.
        opts : SpectralOptions, optional
            CF / FTLM knobs. ``momentum_transfer`` (fractional
            reciprocal-lattice units) is the selection-rule shift Q,
            ``momentum_tolerance`` controls commensurability.
        fixed_sz_n_up : int or None, optional
            Source-sector ``n_up``. Required when ``delta_n_up != 0``.
        delta_n_up : int, optional
            Change in ``n_up`` produced by the observable (0 for
            Sz-conserving probes, +1 for S+, -1 for S-). When non-zero
            the workflow builds a second streaming operator for the
            target subspace.

        Returns
        -------
        SpectralResult
            ``omega`` / ``S_real`` carry the cross-irrep spectral
            function (S_imag is zero in the current build).
            ``per_sector_pair`` records the (initial, final)
            SectorTag pair; ``selection_rule_label`` documents the
            resolved transition.
    )pbdoc");

    // -----------------------------------------------------------------
    // SOTA cross-irrep MULTI-Q streaming-symmetry spectral binding.
    //
    // Amortised sister of
    // ``workflows_spectral_streaming_symmetry_cross_irrep_directory``.
    // The single-Q entry re-pays the (dominant) per-sector ground-state
    // scan + GS eigenvector solve on every call -- but |psi_0> and the
    // GS irrep ``k_src`` are Q-INDEPENDENT. This entry solves the
    // ground state ONCE and then loops the requested momentum-transfer
    // points internally; each Q only costs one ``resolve_target_sector``
    // + one ``CrossSectorOrbitObservable`` scatter + one inner
    // CF-Lanczos. For a Q-path sweep at 32-36 sites this turns N_Q
    // expensive ground-state solves into a single one.
    //
    // Each Q lands as one ``SpectralSectorEntry`` in
    // ``per_sector_pair`` (positionally aligned with
    // ``momentum_points``):
    //   - ``S_real``     : dynamical S(Q, omega) on the shared grid
    //   - ``static_sf``  : equal-time S(Q) = ||O_Q|psi_0>||^2 (free
    //                      from the CF pivot norm -- this is the SSSF)
    // ``agg.S_real`` mirrors the first Q for back-compat; the per-Q
    // payload always lives in ``per_sector_pair``.
    //
    // Per-Q robustness: a Q that is incommensurate, lands on no
    // surviving sector, or maps to an empty target sector records a
    // zero entry (with a diagnostic note) and the sweep CONTINUES,
    // rather than aborting every other Q-point.
    // -----------------------------------------------------------------
    m.def("workflows_spectral_streaming_symmetry_cross_irrep_multiq_directory",
          [](const std::string&                       directory,
             std::uint64_t                             num_sites,
             double                                    spin_l,
             const std::vector<std::vector<py::tuple>>& observable_transforms_per_q,
             const std::vector<std::vector<double>>&   momentum_points,
             ed::workflows::SpectralOptions            opts,
             py::object                                fixed_sz_n_up,
             int                                       delta_n_up) {
              // Decode the per-Q transform tuples. Each Q-point carries
              // its OWN observable O_Q (the e^{-iQ.r} Fourier phase is
              // baked into the transform coefficients), so the outer
              // list MUST be aligned 1:1 with ``momentum_points``.
              if (momentum_points.empty()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_cross_irrep_"
                      "multiq_directory: momentum_points is empty -- pass at "
                      "least one Q vector.");
              }
              if (observable_transforms_per_q.size() != momentum_points.size()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_cross_irrep_"
                      "multiq_directory: observable_transforms_per_q must be "
                      "aligned 1:1 with momentum_points (got " +
                      std::to_string(observable_transforms_per_q.size()) +
                      " transform lists for " +
                      std::to_string(momentum_points.size()) + " Q-points).");
              }
              std::vector<std::vector<Operator::TransformData>> tlist_per_q;
              tlist_per_q.reserve(observable_transforms_per_q.size());
              for (const auto& rows : observable_transforms_per_q) {
                  std::vector<Operator::TransformData> tlist;
                  tlist.reserve(rows.size());
                  for (const auto& row : rows) {
                      if (row.size() < 6) {
                          throw std::invalid_argument(
                              "workflows_spectral_streaming_symmetry_cross_"
                              "irrep_multiq_directory: each transform must be "
                              "a 6-tuple (op_type, site, coeff, is_two_body, "
                              "op_type_2, site_2).");
                      }
                      Operator::TransformData t;
                      t.op_type      = static_cast<uint8_t>(row[0].cast<int>());
                      t.site_index   = row[1].cast<std::uint64_t>();
                      t.coefficient  = row[2].cast<std::complex<double>>();
                      t.is_two_body  = row[3].cast<bool>();
                      t.op_type_2    = static_cast<uint8_t>(row[4].cast<int>());
                      t.site_index_2 = row[5].cast<std::uint64_t>();
                      tlist.push_back(t);
                  }
                  if (tlist.empty()) {
                      throw std::invalid_argument(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory: one of the per-Q observable "
                          "transform lists is empty.");
                  }
                  tlist_per_q.push_back(std::move(tlist));
              }

              ed::SpectralResult agg;
              {
                  py::gil_scoped_release release;

                  // (1) Source streaming operator + OperatorRef.
                  ed::OperatorSpec src_spec;
                  src_spec.source             = ed::DirectoryPath{directory};
                  src_spec.num_sites          = num_sites;
                  src_spec.spin_l             = static_cast<float>(spin_l);
                  src_spec.streaming_symmetry = true;
                  if (!fixed_sz_n_up.is_none()) {
                      src_spec.fixed_sz = fixed_sz_n_up.cast<int>();
                  }
                  // Operator-collapse Phase 3 (Jun 2026): direct sector
                  // enumeration via ``make_sector_operators_tagged`` +
                  // ``SectorSetView``. ``src_ref`` is built once after the GS
                  // solve from the GS sector's materialised orbit basis;
                  // ``dst_ref`` is rebuilt per-Q from the resolved target
                  // sector (the target sector varies with Q).
                  ed::core::SectorSetView src_handle(
                      ed::make_sector_operators_tagged(src_spec));

                  const std::size_t src_num_sectors = src_handle.num_sectors();
                  if (src_num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory: source operator has no symmetry "
                          "sectors; check automorphism_results/.");
                  }

                  // (2) Locate the global GS sector -- two-phase scan
                  //     (cheap krylov=40 pass, refine within gap), run
                  //     ONCE and reused across all Q.
                  const std::vector<std::size_t> src_sector_indices =
                      ed::core::filter_sectors(src_num_sectors,
                                               opts.selected_sectors);
                  std::size_t gs_src_idx = 0;
                  double      gs_energy  = std::numeric_limits<double>::infinity();
                  bool        any_solved = false;

                  const bool enable_two_phase =
                      src_sector_indices.size() > 2;
                  std::vector<std::size_t> phase2_candidates;
                  if (enable_two_phase) {
                      std::vector<std::pair<double, std::size_t>> phase1_min;
                      phase1_min.reserve(src_sector_indices.size());
                      for (std::size_t k : src_sector_indices) {
                          auto sec = src_handle.sector(k);
                          if (!sec || sec->dim() == 0) continue;
                          ed::workflows::SolveOptions p1;
                          p1.num_eigs        = 1;
                          p1.tolerance       = 1e-8;
                          p1.backend         = opts.backend;
                          p1.method          = ed::workflows::SolveMethod::Lanczos;
                          p1.compute_vectors = false;
                          p1.max_iter        = std::min<std::size_t>(40, sec->dim());
                          try {
                              auto sr = ed::workflows::solve(*sec, p1);
                              if (!sr.eigenvalues.empty()) {
                                  phase1_min.emplace_back(
                                      sr.eigenvalues.front(), k);
                              }
                          } catch (...) {
                              phase1_min.emplace_back(
                                  -std::numeric_limits<double>::infinity(), k);
                          }
                      }
                      if (!phase1_min.empty()) {
                          std::sort(phase1_min.begin(), phase1_min.end(),
                                    [](const auto& a, const auto& b) {
                                        return a.first < b.first;
                                    });
                          const double best_E = phase1_min.front().first;
                          const double gap =
                              std::max(1e-2 * std::abs(best_E), 1e-4);
                          for (const auto& [E, k] : phase1_min) {
                              if (E <= best_E + gap) {
                                  phase2_candidates.push_back(k);
                              }
                          }
                      }
                  }
                  const std::vector<std::size_t>& src_scan_indices =
                      enable_two_phase ? phase2_candidates
                                       : src_sector_indices;
                  for (std::size_t k : src_scan_indices) {
                      auto sec = src_handle.sector(k);
                      if (!sec || sec->dim() == 0) continue;
                      ed::workflows::SolveOptions sopts;
                      sopts.num_eigs        = 1;
                      sopts.tolerance       = 1e-12;
                      sopts.backend         = opts.backend;
                      sopts.method          = ed::workflows::SolveMethod::Lanczos;
                      sopts.compute_vectors = false;
                      auto sr = ed::workflows::solve(*sec, sopts);
                      if (sr.eigenvalues.empty()) continue;
                      const double E_k = sr.eigenvalues.front();
                      if (E_k < gs_energy) {
                          gs_energy  = E_k;
                          gs_src_idx = k;
                      }
                      any_solved = true;
                  }
                  if (!any_solved) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory: every source sector returned an "
                          "empty spectrum.");
                  }

                  // (3) Re-solve GS sector with eigenvectors -> |psi_0>.
                  auto gs_sec_view = src_handle.sector(gs_src_idx);
                  ed::workflows::SolveOptions sopts_full;
                  sopts_full.num_eigs        = 1;
                  sopts_full.tolerance       = 1e-12;
                  sopts_full.backend         = opts.backend;
                  sopts_full.method          = ed::workflows::SolveMethod::Lanczos;
                  sopts_full.compute_vectors = true;
                  auto gs_sr = ed::workflows::solve(*gs_sec_view, sopts_full);
                  if (gs_sr.eigenvalues.empty() || !gs_sr.eigenvectors ||
                      gs_sr.eigenvectors->host.empty()) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory: GS eigenvector reconstruction "
                          "failed.");
                  }
                  if (!gs_sr.backend.lane.empty()) {
                      agg.backend.lane     = gs_sr.backend.lane;
                      agg.backend.mpi_size = gs_sr.backend.mpi_size;
                  }
                  const auto& psi0 = gs_sr.eigenvectors->host[0];
                  const double E0  = gs_sr.eigenvalues.front();
                  ed::SectorTag gs_src_tag = src_handle.sector_tag(gs_src_idx);

                  // Source observable handle: GS sector's materialised orbit
                  // basis (Q-independent, built once). Each OperatorRef wraps
                  // exactly ONE sector.
                  const ed::symmetry::SectorBasis& src_basis =
                      gs_sec_view->materialized_basis();
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef src_ref =
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(
                          src_basis, static_cast<std::uint64_t>(num_sites));

                  // (4) Build / re-use the target sector set ONCE (depends
                  //     only on delta_n_up, not on Q). The view is cheaply
                  //     copyable (shared_ptr-backed), so the ``delta == 0``
                  //     case shares the source set.
                  ed::core::SectorSetView dst_handle = src_handle;
                  if (delta_n_up != 0) {
                      if (!src_spec.fixed_sz.has_value()) {
                          throw std::invalid_argument(
                              "workflows_spectral_streaming_symmetry_cross_"
                              "irrep_multiq_directory: delta_n_up != 0 "
                              "requires fixed_sz_n_up to be set.");
                      }
                      ed::OperatorSpec dst_spec;
                      dst_spec.source             = ed::DirectoryPath{directory};
                      dst_spec.num_sites          = num_sites;
                      dst_spec.spin_l             = static_cast<float>(spin_l);
                      dst_spec.streaming_symmetry = true;
                      dst_spec.fixed_sz           = *src_spec.fixed_sz + delta_n_up;
                      dst_handle = ed::core::SectorSetView(
                          ed::make_sector_operators_tagged(dst_spec));
                  }

                  // (5) Shared omega grid (built once).
                  const std::size_t num_omega =
                      std::max<std::size_t>(opts.num_omega, 1);
                  std::vector<double> omega_grid(num_omega);
                  if (num_omega == 1) {
                      omega_grid[0] = opts.omega_min;
                  } else {
                      const double step =
                          (opts.omega_max - opts.omega_min) /
                          static_cast<double>(num_omega - 1);
                      for (std::size_t i = 0; i < num_omega; ++i) {
                          omega_grid[i] =
                              opts.omega_min + static_cast<double>(i) * step;
                      }
                  }
                  agg.omega = omega_grid;

                  const double resolvent_shift =
                      (std::abs(opts.energy_shift) > 1e-14) ? opts.energy_shift
                                                            : E0;

                  // (6) Loop the requested Q-points. The GS solve above is
                  //     NOT repeated -- only the per-Q scatter + inner CF.
                  std::string lane_label = agg.backend.lane;
                  for (std::size_t qi = 0; qi < momentum_points.size(); ++qi) {
                      const std::vector<double>& Q = momentum_points[qi];

                      auto push_zero_entry = [&](const std::string& why) {
                          ed::SpectralSectorEntry e;
                          e.initial   = gs_src_tag;
                          e.final_    = gs_src_tag;
                          e.S_real.assign(num_omega, 0.0);
                          e.S_imag.assign(num_omega, 0.0);
                          e.static_sf = 0.0;
                          e.notes.emplace_back("status", why);
                          agg.per_sector_pair.push_back(std::move(e));
                      };

                      double q_residual = 0.0;
                      const std::size_t dst_sector_idx =
                          ed::core::resolve_target_sector(
                              dst_handle, gs_src_idx, Q, &q_residual);
                      if (dst_sector_idx == ed::core::kSectorNotFound) {
                          push_zero_entry(
                              "no surviving target sector (residual=" +
                              std::to_string(q_residual) + ")");
                          continue;
                      }
                      if (q_residual > opts.momentum_tolerance) {
                          push_zero_entry(
                              "Q incommensurate (residual=" +
                              std::to_string(q_residual) + " > tol=" +
                              std::to_string(opts.momentum_tolerance) + ")");
                          continue;
                      }
                      ed::SectorTag dst_tag =
                          dst_handle.sector_tag(dst_sector_idx);

                      // Resolve the target sector operator; an empty (dropped)
                      // irrep => zero entry. Build ``dst_ref`` from the
                      // resolved sector's materialised orbit basis (the target
                      // sector varies with Q, so this is per-Q).
                      ed::symmetry::SectorOperator* dst_sec_view =
                          dst_handle.sector(dst_sector_idx);
                      if (!dst_sec_view || dst_sec_view->dim() == 0) {
                          push_zero_entry("target sector empty");
                          continue;
                      }
                      const ed::symmetry::SectorBasis& dst_basis =
                          dst_sec_view->materialized_basis();
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef dst_ref =
                          ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(
                              dst_basis, static_cast<std::uint64_t>(num_sites));

                      ed::dssf::CrossSectorOrbitObservable orb_obs(
                          src_ref, /*src_sector=*/0,
                          dst_ref, /*dst_sector=*/0,
                          tlist_per_q[qi], static_cast<float>(spin_l));
                      const std::size_t dim_dst = orb_obs.dim_dst();
                      if (dst_sec_view->dim() != dim_dst) {
                          push_zero_entry("target sector dim mismatch");
                          continue;
                      }
                      std::vector<Complex> phi(dim_dst, Complex(0.0, 0.0));
                      orb_obs.apply(psi0.data(), phi.data(), dim_dst);

                      ed::observables::CfSpectralOptions cfopts;
                      cfopts.krylov_dim   = opts.krylov_dim;
                      cfopts.broadening   = opts.broadening;
                      cfopts.energy_shift = resolvent_shift;
                      cfopts.tolerance    = 1e-12;
                      cfopts.global_n     = dim_dst;
                      cfopts.verbose      = false;

                      ed::observables::CfSpectralResult cf;
                      auto cf_variant = ed::select_backend(
                          dst_sec_view->geometry(), opts.backend);
                      std::visit([&](auto& backend_uptr) {
                          using BPtr = std::decay_t<decltype(backend_uptr)>;
                          using B = typename BPtr::element_type;
                          auto apply_H = dst_sec_view->template bind<B>();
                          cf = ed::observables::cf_spectral_from_vector(
                              *backend_uptr, apply_H, dim_dst,
                              phi.data(), omega_grid, cfopts);
                      }, cf_variant);
                      if (lane_label.empty()) {
                          lane_label = ed::lane_label_from_variant(cf_variant);
                      }

                      ed::SpectralSectorEntry entry;
                      entry.initial   = gs_src_tag;
                      entry.final_    = dst_tag;
                      entry.S_real    = cf.spectral_function;
                      entry.S_imag.assign(cf.spectral_function.size(), 0.0);
                      entry.static_sf = cf.phi_norm * cf.phi_norm;
                      entry.notes.emplace_back("status", "ok");
                      agg.per_sector_pair.push_back(std::move(entry));

                      // Mirror the FIRST resolved Q into the top-level
                      // S_real for back-compat with single-Q callers.
                      if (agg.S_real.empty()) {
                          agg.S_real = cf.spectral_function;
                          agg.S_imag.assign(cf.spectral_function.size(), 0.0);
                          agg.errors_real.assign(cf.spectral_function.size(), 0.0);
                          agg.errors_imag.assign(cf.spectral_function.size(), 0.0);
                      }
                  }

                  if (agg.S_real.empty()) {
                      agg.S_real.assign(num_omega, 0.0);
                      agg.S_imag.assign(num_omega, 0.0);
                      agg.errors_real.assign(num_omega, 0.0);
                      agg.errors_imag.assign(num_omega, 0.0);
                  }
                  if (!lane_label.empty()) agg.backend.lane = lane_label;
                  agg.selection_rule_label =
                      "k_final = k_initial + Q (cross-irrep, multi-Q "
                      "amortised; " +
                      std::to_string(momentum_points.size()) +
                      " Q-points, single GS solve)";
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")                       = 0.5,
          py::arg("observable_transforms_per_q")  = std::vector<std::vector<py::tuple>>{},
          py::arg("momentum_points")              = std::vector<std::vector<double>>{},
          py::arg("opts")                         = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up")                = py::none(),
          py::arg("delta_n_up")                   = 0,
          R"pbdoc(
        Amortised multi-Q cross-irrep streaming-symmetry spectral
        workflow (SOTA).

        Identical physics to
        ``workflows_spectral_streaming_symmetry_cross_irrep_directory``
        but the per-sector ground-state scan and the GS eigenvector
        solve -- the dominant cost -- run ONCE and are reused across
        every momentum-transfer point in ``momentum_points``. Each Q
        carries its OWN observable O_Q (its e^{-iQ.r} Fourier phase is
        baked into the transform coefficients), so
        ``observable_transforms_per_q`` is a list of transform lists
        aligned 1:1 with ``momentum_points``. Per Q the only work is a
        selection-rule lookup, a ``CrossSectorOrbitObservable`` scatter,
        and one inner CF-Lanczos in the (reduced) target sector.

        Results land in ``per_sector_pair`` positionally aligned with
        ``momentum_points``: ``S_real`` is the dynamical S(Q, omega)
        and ``static_sf`` is the equal-time S(Q) = ||O_Q|psi_0>||^2
        (the static / equal-time structure factor, obtained for free
        from the CF pivot norm). Incommensurate or empty-target Q
        points record a zero entry with a ``status`` note and the
        sweep continues.
    )pbdoc");

    // -----------------------------------------------------------------
    // SOTA FINITE-T cross-irrep dynamical spectral binding (FTLM).
    //
    // Sister entry point to the GS cross-irrep binding above. The
    // selection rule, observable transforms, and (src, dst) operator
    // pair are resolved identically, but instead of a single GS
    // solve we loop over ALL source sectors and run an FTLM-style
    // multi-sample / multi-temperature dynamical kernel per
    // (k_src, k_dst) pair, then Z-weight-recombine across source
    // sectors. This closes the DYNAMICAL_THERMAL spatial-irrep gap.
    //
    // Math (per source sector k_src, target k_dst = k_src + Q):
    //
    //   S_{k_src}(omega, T) = (dim_src / R)
    //       sum_r sum_m e^{-beta E_m} |c_m|^2 * S_m(omega)
    //
    // where S_m(omega) = ||phi_m||^2 sum_n V_S[0,n]^2 *
    //                       Lorentzian(omega - (lambda_n - E_m))
    // and ``phi_m = O_Q |m^{(r)}>`` is the cross-irrep scatter of
    // the m-th outer Ritz state into the target orbit basis.
    //
    // Aggregation:
    //
    //   S_total(omega, T) = sum_{k_src} S_{k_src}(omega, T)
    //                       / sum_{k_src} Z_{k_src}(T).
    //
    // Returns a SpectralResult shaped exactly like the GS cross-irrep
    // path, with one extra omega-axis copy per temperature exposed
    // via ``S_by_T_real[T]`` / ``S_by_T_imag[T]`` (carried inside
    // ``per_sector_pair`` for now; the GS entry sits at index 0 for
    // backwards compat with consumers that read agg.S_real).
    // -----------------------------------------------------------------
    m.def("workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory",
          [](const std::string&                    directory,
             std::uint64_t                          num_sites,
             double                                 spin_l,
             const std::vector<py::tuple>&          observable_transforms,
             ed::workflows::SpectralOptions         opts,
             py::object                             fixed_sz_n_up,
             int                                    delta_n_up,
             std::vector<double>                    temperatures,
             std::uint64_t                          num_samples,
             std::uint64_t                          random_seed) {
              // ----------------------------------------------------------
              // Same observable-transform decoder as the GS path.
              // ----------------------------------------------------------
              std::vector<Operator::TransformData> tlist;
              tlist.reserve(observable_transforms.size());
              for (const auto& row : observable_transforms) {
                  if (row.size() < 6) {
                      throw std::invalid_argument(
                          "workflows_spectral_streaming_symmetry_ftlm_"
                          "cross_irrep_directory: each transform must be "
                          "a 6-tuple (op_type, site, coeff, is_two_body, "
                          "op_type_2, site_2).");
                  }
                  Operator::TransformData t;
                  t.op_type       = static_cast<uint8_t>(row[0].cast<int>());
                  t.site_index    = row[1].cast<std::uint64_t>();
                  t.coefficient   = row[2].cast<std::complex<double>>();
                  t.is_two_body   = row[3].cast<bool>();
                  t.op_type_2     = static_cast<uint8_t>(row[4].cast<int>());
                  t.site_index_2  = row[5].cast<std::uint64_t>();
                  tlist.push_back(t);
              }
              if (tlist.empty()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_ftlm_cross_"
                      "irrep_directory: observable_transforms is empty.");
              }
              if (temperatures.empty()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_ftlm_cross_"
                      "irrep_directory: temperatures is empty.");
              }

              ed::SpectralResult agg;
              {
                  py::gil_scoped_release release;

                  // -----------------------------------------------
                  // (1) Build source streaming operator.
                  // -----------------------------------------------
                  ed::OperatorSpec src_spec;
                  src_spec.source             = ed::DirectoryPath{directory};
                  src_spec.num_sites          = num_sites;
                  src_spec.spin_l             = static_cast<float>(spin_l);
                  src_spec.streaming_symmetry = true;
                  if (!fixed_sz_n_up.is_none()) {
                      src_spec.fixed_sz = fixed_sz_n_up.cast<int>();
                  }
                  // Operator-collapse Phase 3 (Jun 2026): direct sector
                  // enumeration via ``make_sector_operators_tagged`` +
                  // ``SectorSetView``. ``src_ref`` / ``dst_ref`` are built
                  // per (k_src, k_dst) pair inside the loop from each sector
                  // operator's materialised orbit basis.
                  ed::core::SectorSetView src_handle(
                      ed::make_sector_operators_tagged(src_spec));

                  const std::size_t src_num_sectors = src_handle.num_sectors();
                  if (src_num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_ftlm_cross_"
                          "irrep_directory: source operator has no symmetry "
                          "sectors; check automorphism_results/.");
                  }

                  // -----------------------------------------------
                  // (2) Build / re-use target operator + ref.
                  // -----------------------------------------------
                  ed::core::SectorSetView dst_handle = src_handle;
                  if (delta_n_up != 0) {
                      if (!src_spec.fixed_sz.has_value()) {
                          throw std::invalid_argument(
                              "workflows_spectral_streaming_symmetry_ftlm_"
                              "cross_irrep_directory: delta_n_up != 0 "
                              "requires fixed_sz_n_up to be set.");
                      }
                      ed::OperatorSpec dst_spec;
                      dst_spec.source             = ed::DirectoryPath{directory};
                      dst_spec.num_sites          = num_sites;
                      dst_spec.spin_l             = static_cast<float>(spin_l);
                      dst_spec.streaming_symmetry = true;
                      dst_spec.fixed_sz           = *src_spec.fixed_sz + delta_n_up;
                      dst_handle = ed::core::SectorSetView(
                          ed::make_sector_operators_tagged(dst_spec));
                  }

                  // -----------------------------------------------
                  // (3) Frequency grid (same convention as GS path).
                  // -----------------------------------------------
                  const std::size_t num_omega =
                      std::max<std::size_t>(opts.num_omega, 1);
                  std::vector<double> omega_grid(num_omega);
                  if (num_omega == 1) {
                      omega_grid[0] = opts.omega_min;
                  } else {
                      const double step =
                          (opts.omega_max - opts.omega_min) /
                          static_cast<double>(num_omega - 1);
                      for (std::size_t i = 0; i < num_omega; ++i) {
                          omega_grid[i] = opts.omega_min +
                              static_cast<double>(i) * step;
                      }
                  }

                  // -----------------------------------------------
                  // (4) Filter source sectors via selected_sectors.
                  // -----------------------------------------------
                  const std::vector<std::size_t> src_sector_indices =
                      ed::core::filter_sectors(src_num_sectors,
                                               opts.selected_sectors);

                  // -----------------------------------------------
                  // (5) For each source sector, resolve target,
                  //     build cross-irrep observable, and run the
                  //     FTLM cross-irrep kernel. Accumulate per-
                  //     sector results in a vector for the combiner.
                  // -----------------------------------------------
                  ed::matvec::CpuBackend cpu_be;
                  std::vector<ed::observables::FtlmCrossIrrepSectorResult>
                      sector_results;
                  std::vector<std::pair<ed::SectorTag, ed::SectorTag>>
                      sector_tags;
                  sector_results.reserve(src_sector_indices.size());
                  sector_tags.reserve(src_sector_indices.size());

                  for (std::size_t k_src : src_sector_indices) {
                      ed::symmetry::SectorOperator* src_view =
                          src_handle.sector(k_src);
                      if (!src_view || src_view->dim() == 0) continue;

                      double q_residual = 0.0;
                      const std::size_t k_dst =
                          ed::core::resolve_target_sector(
                              dst_handle,
                              k_src,
                              opts.momentum_transfer,
                              &q_residual);
                      if (k_dst == ed::core::kSectorNotFound) continue;
                      if (q_residual > opts.momentum_tolerance) {
                          throw std::runtime_error(
                              "workflows_spectral_streaming_symmetry_ftlm_"
                              "cross_irrep_directory: Q incommensurate with "
                              "the lattice (residual = " +
                              std::to_string(q_residual) +
                              " > tolerance = " +
                              std::to_string(opts.momentum_tolerance) + ").");
                      }
                      ed::symmetry::SectorOperator* dst_view =
                          dst_handle.sector(k_dst);
                      if (!dst_view || dst_view->dim() == 0) continue;

                      // Cross-irrep rectangular observable for THIS
                      // (k_src, k_dst) pair. Each OperatorRef wraps one
                      // sector's materialised orbit basis (src_sector ==
                      // dst_sector == 0).
                      const ed::symmetry::SectorBasis& src_basis =
                          src_view->materialized_basis();
                      const ed::symmetry::SectorBasis& dst_basis =
                          dst_view->materialized_basis();
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef src_ref =
                          ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(
                              src_basis, static_cast<std::uint64_t>(num_sites));
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef dst_ref =
                          ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(
                              dst_basis, static_cast<std::uint64_t>(num_sites));
                      ed::dssf::CrossSectorOrbitObservable orb_obs(
                          src_ref, /*src_sector=*/0,
                          dst_ref, /*dst_sector=*/0,
                          tlist,
                          static_cast<float>(spin_l));
                      const std::size_t dim_src = src_view->dim();
                      const std::size_t dim_dst = orb_obs.dim_dst();
                      if (dim_dst != dst_view->dim()) {
                          throw std::runtime_error(
                              "workflows_spectral_streaming_symmetry_ftlm_"
                              "cross_irrep_directory: dim_dst mismatch "
                              "(observable=" + std::to_string(dim_dst) +
                              ", view=" + std::to_string(dst_view->dim()) +
                              ").");
                      }

                      // Three lambdas: H_src, H_dst, O.
                      auto apply_H_src = [&src_view](const Complex* x,
                                                     Complex*       y,
                                                     int            n) {
                          src_view->apply(x, y,
                              static_cast<std::size_t>(n));
                      };
                      auto apply_H_dst = [&dst_view](const Complex* x,
                                                     Complex*       y,
                                                     int            n) {
                          dst_view->apply(x, y,
                              static_cast<std::size_t>(n));
                      };
                      auto apply_O = [&orb_obs](const Complex* x,
                                                Complex*       y,
                                                int            n) {
                          orb_obs.apply(x, y,
                              static_cast<std::size_t>(n));
                      };

                      ed::observables::FtlmCrossIrrepOptions kopts;
                      kopts.krylov_dim       = opts.krylov_dim;
                      kopts.num_samples      = num_samples;
                      kopts.broadening       = opts.broadening;
                      kopts.tolerance        = 1e-12;
                      kopts.random_seed      = random_seed;
                      kopts.verbose          = false;
                      kopts.full_reorthogonalization = false;
                      kopts.reorth_frequency = 1;

                      auto sec_res =
                          ed::observables::ftlm_cross_irrep_kernel_one_sector(
                              apply_H_src, apply_H_dst, apply_O,
                              dim_src, dim_dst,
                              temperatures, omega_grid, kopts);

                      sector_results.push_back(std::move(sec_res));
                      sector_tags.emplace_back(
                          src_handle.sector_tag(k_src),
                          dst_handle.sector_tag(k_dst));
                  }

                  // -----------------------------------------------
                  // (6) Z-weighted recombine across source sectors.
                  // -----------------------------------------------
                  auto merged = ed::observables::combine_sector_dynamical_spectra(
                      sector_results, temperatures, num_omega);

                  // -----------------------------------------------
                  // (7) Marshal into ed::SpectralResult. The aggregate
                  //     ``S_real`` / ``S_imag`` arrays carry the
                  //     first temperature; the full T-resolved data
                  //     is exposed via ``per_sector_pair`` entries
                  //     whose initial/final SectorTag pair and
                  //     selection-rule label record EACH source
                  //     sector's contribution along with the
                  //     temperatures encoded into the label.
                  // -----------------------------------------------
                  agg.omega = omega_grid;
                  const double T_primary = temperatures.front();
                  agg.S_real = merged.S_real[T_primary];
                  agg.S_imag = merged.S_imag[T_primary];
                  agg.errors_real.assign(num_omega, 0.0);
                  agg.errors_imag.assign(num_omega, 0.0);

                  // Phase H.1 of the "Close CPU/GPU Gaps" plan
                  // (May 2026): surface the truthful backend lane on
                  // the aggregate. The FTLM cross-irrep kernel
                  // (``ftlm_cross_irrep_kernel_one_sector``) is
                  // host-only -- it consumes raw ``apply_H_src/dst/O``
                  // lambdas and runs the inner Lanczos / observable
                  // contraction on the host. GPU rectangular scatter
                  // for ``CrossSectorOrbitObservable`` is tracked as
                  // a deferred follow-up. Until that lands, this
                  // binding always reports ``lane='cpu'``.
                  agg.backend.lane     = "cpu";
                  agg.backend.mpi_size = 1;

                  std::string label =
                      "k_final = k_initial + Q (cross-irrep, FTLM, "
                      "finite-T); source-sectors swept = " +
                      std::to_string(sector_results.size()) +
                      "; T = [";
                  for (std::size_t i = 0; i < temperatures.size(); ++i) {
                      if (i) label += ", ";
                      label += std::to_string(temperatures[i]);
                  }
                  label += "]; num_samples = " + std::to_string(num_samples);
                  agg.selection_rule_label = std::move(label);

                  // Per-sector pair entries: one per (k_src, k_dst)
                  // pair, holding that sector's PRIMARY-T S(omega).
                  // For the multi-T payload, also expose
                  // ``S_by_T_real`` / ``S_by_T_imag`` (full grid),
                  // packed into the entry's S_real / S_imag fields
                  // when there is only one sector. For now we ship
                  // the first-T slice in S_real / S_imag and surface
                  // all temperatures through the multi-T payload
                  // map on the agg structure.
                  for (std::size_t i = 0; i < sector_results.size(); ++i) {
                      ed::SpectralSectorEntry entry;
                      entry.initial = sector_tags[i].first;
                      entry.final_  = sector_tags[i].second;
                      auto sr = sector_results[i].S_real.find(T_primary);
                      auto si = sector_results[i].S_imag.find(T_primary);
                      auto sz = sector_results[i].Z.find(T_primary);
                      if (sr != sector_results[i].S_real.end()) {
                          double Z_local = (sz != sector_results[i].Z.end() && sz->second > 1e-300)
                                            ? sz->second : 1.0;
                          entry.S_real.resize(num_omega);
                          entry.S_imag.resize(num_omega);
                          for (std::size_t iw = 0; iw < num_omega; ++iw) {
                              entry.S_real[iw] = sr->second[iw] / Z_local;
                              entry.S_imag[iw] = (si != sector_results[i].S_imag.end())
                                                  ? si->second[iw] / Z_local
                                                  : 0.0;
                          }
                      }
                      agg.per_sector_pair.push_back(std::move(entry));
                  }

                  // Multi-T payload for the Python wrapper to read:
                  // pack the full {T: S(omega)} map by appending one
                  // synthetic SpectralSectorEntry per temperature
                  // with a sentinel label. The Python wrapper
                  // distinguishes them by their `selection_rule_label`
                  // and the index offset. (The simple, structured
                  // alternative -- adding a new C++ field to
                  // SpectralResult -- is intentionally avoided here
                  // to keep this PR strictly additive.)
                  for (double T : temperatures) {
                      ed::SpectralSectorEntry mt;
                      ed::SectorTag dummy;
                      dummy.sector_index = 0;
                      dummy.sector_dim   = 0;
                      mt.initial = dummy;
                      mt.final_  = dummy;
                      mt.S_real  = merged.S_real[T];
                      mt.S_imag  = merged.S_imag[T];
                      agg.per_sector_pair.push_back(std::move(mt));
                  }
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")                = 0.5,
          py::arg("observable_transforms") = std::vector<py::tuple>{},
          py::arg("opts")                  = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up")         = py::none(),
          py::arg("delta_n_up")            = 0,
          py::arg("temperatures")          = std::vector<double>{},
          py::arg("num_samples")           = std::uint64_t{30},
          py::arg("random_seed")           = std::uint64_t{0},
          R"pbdoc(
        Cross-irrep streaming-symmetry **finite-T** spectral workflow
        (SOTA FTLM).

        Computes S(Q, omega, T) for a user-supplied probe observable
        ``O_Q`` whose lattice symmetry character implies a sector
        transition

            k_final = k_initial + Q

        via the streaming-symmetry selection rule (and an optional
        Sz shift ``delta_n_up``). The implementation is a per-source-
        sector Finite-Temperature Lanczos Method (FTLM): for each
        source sector ``k_src`` we draw ``num_samples`` Gaussian
        random vectors in the source orbit basis, build an outer
        Lanczos basis on ``H`` restricted to ``k_src``, reconstruct
        each Ritz state, scatter it into ``k_dst`` via the
        rectangular ``CrossSectorOrbitObservable``, run a second,
        target-sector Lanczos starting from ``phi = O_Q |m>``, and
        accumulate a Lorentzian-broadened Lehmann sum weighted by
        ``exp(-beta * E_m) * |c_m|^2``. Sectors are recombined via
        the F-shifted Z-weighted combiner so disparate per-sector
        E_min values do not destabilise the floating-point exponent.

        Parameters
        ----------
        directory : str
            Hamiltonian directory (must contain
            ``automorphism_results/``).
        num_sites : int
            Number of lattice sites.
        spin_l : float, optional
            Local spin (0.5 by default).
        observable_transforms : list of (op_type:int, site:int,
                                         coeff:complex,
                                         is_two_body:bool,
                                         op_type_2:int, site_2:int)
            One row per term in ``O_Q``. ``qed.spectral`` extracts
            these automatically from an ``ed.Operator`` argument.
        opts : SpectralOptions, optional
            ``krylov_dim``, ``broadening``, ``omega_*``,
            ``momentum_transfer``, ``selected_sectors``, etc.
        fixed_sz_n_up : int or None, optional
            Source-sector n_up. Required when ``delta_n_up != 0``.
        delta_n_up : int, optional
            Change in n_up produced by ``O_Q``.
        temperatures : list of float
            Temperatures to evaluate (must be non-empty; values are
            energy-axis units).
        num_samples : int, optional
            FTLM random samples per source sector.
        random_seed : int, optional
            Seed base; per-sample seed is
            ``random_seed + sample_idx * 12345``.

        Returns
        -------
        SpectralResult
            ``omega`` / ``S_real`` carry the recombined S(Q,omega) at
            ``temperatures[0]``. The per-T data set is stuffed into
            ``per_sector_pair`` entries (one per source sector pair
            for the **primary** T, followed by one synthetic entry
            per temperature whose ``initial.sector_dim == 0`` flag
            distinguishes the multi-T payload). The Python wrapper
            ``qed.spectral`` unpacks this and surfaces a clean
            ``{T -> S(omega)}`` dict to the user.
    )pbdoc");

    // -----------------------------------------------------------------
    // All-Sz flat-pool thermal binding (Jun 2026).
    //
    // Replaces the Python ThreadPoolExecutor n_up outer loop with a
    // single C++ call that:
    //   1. Loads the Hamiltonian + symmetry group info ONCE.
    //   2. Enumerates orbit reps ONCE via enumerate_full_orbit_reps
    //      (O(2^N × |G|)), partitions by popcount into per-n_up buckets.
    //   3. Builds ALL (n_up, irrep) sector operators in one nested loop.
    //   4. Runs a single flat #pragma omp parallel for over the entire
    //      (n_up × irrep) sector set simultaneously (ED_SYM_SECTOR_PARALLEL=1).
    //   5. Does one combine_sector_thermodynamics call across all sectors.
    //
    // Eliminates N+1 cold-start overhead (JSON loads + orbit rep scans)
    // present when calling workflows_thermal_streaming_symmetry_directory
    // once per n_up from a Python ThreadPoolExecutor.
    // -----------------------------------------------------------------
    m.def("workflows_thermal_all_sz_streaming_symmetry_directory",
          [](const std::string& directory,
             std::uint64_t num_sites,
             double spin_l,
             ed::workflows::ThermalOptions opts,
             int n_up_min,
             int n_up_max) {
              ed::ThermalResult agg;
              {
                  py::gil_scoped_release release;

                  ed::OperatorSpec spec;
                  spec.source             = ed::DirectoryPath{directory};
                  spec.num_sites          = num_sites;
                  spec.spin_l             = static_cast<float>(spin_l);
                  spec.streaming_symmetry = true;
                  // No fixed_sz: build_all_sz_sector_operators covers all n_up.

                  // ---------------------------------------------------------
                  // Stage 8 (SymmetryEngine v2): unified symmetry composition.
                  // One SectorPlan resolution drives all three mechanisms
                  // (spin-flip transport across Sz, spin-flip projection of
                  // the half-filling block, time-reversal k <-> -k pairing);
                  // see include/ed/symmetry/sector_plan.h. Per-call toggles
                  // on ThermalOptions (auto/off/require) override the env
                  // gates.
                  // ---------------------------------------------------------
                  const int N_sites = static_cast<int>(num_sites);
                  const int req_lo  = std::max(0, n_up_min);
                  const int req_hi  = (n_up_max < 0) ? N_sites
                                                     : std::min(n_up_max, N_sites);
                  ed::symmetry::SymmetryComposition comp;
                  {
                      auto probe = ed::detail::build_base_op(spec);
                      ed::detail::load_terms_into(*probe, spec);
                      ed::matvec::TermStorage soa;
                      ed::matvec::TermStorage::classify_route(
                          soa, probe->transform_data_, probe->three_body_data_,
                          [](const std::complex<double>& c) { return c; });
                      probe->symmetry_info.loadFromDirectory(directory);
                      comp = ed::symmetry::resolve_symmetry_composition(
                          soa, probe->symmetry_info, opts.backend.allow_gpu,
                          ed::symmetry::sym_toggle_from_int(opts.spin_flip),
                          ed::symmetry::sym_toggle_from_int(opts.time_reversal));
                      // Stage 7a: point-group star maps join the
                      // isospectral-orbit relation alongside TR.
                      const std::size_t nsec =
                          probe->symmetry_info.sectors.size();
                      for (const auto& m : opts.star_maps) {
                          if (m.size() != nsec) continue;
                          comp.star_maps.emplace_back(m.begin(), m.end());
                      }
                  }
                  const ed::symmetry::BuildWindow win =
                      ed::symmetry::plan_build_window(
                          comp, req_lo, req_hi, N_sites);
                  const bool flip_transport = win.flip_transport;
                  const bool flip_project   = win.flip_project_half;
                  if (ed::symmetry::sym_profile_enabled()) {
                      if (flip_transport)
                          std::fprintf(stderr,
                              "[sym-profile] spin-flip transport: solving "
                              "n_up [%d, %d] for requested [%d, %d]\n",
                              win.lo, win.hi, req_lo, req_hi);
                      if (flip_project)
                          std::fprintf(stderr,
                              "[sym-profile] spin-flip projection: half-filling "
                              "block split into (k, +/-) sectors\n");
                  }

                  // Build the (n_up, irrep) sector operators in a single pass.
                  ed::SectorOperatorSet set =
                      ed::make_all_sz_sector_operators_tagged(
                          spec, win.lo, win.hi, flip_project);

                  const long n_ops =
                      static_cast<long>(set.operators.size());
                  if (n_ops == 0) {
                      throw std::runtime_error(
                          "workflows_thermal_all_sz_streaming_symmetry_"
                          "directory: no sectors found; check n_up range "
                          "and automorphism_results/ directory.");
                  }

                  const std::size_t n_raw = set.num_raw_sectors;
                  const ed::symmetry::TrActionPlan tr_plan =
                      ed::symmetry::plan_tr_actions(set.tags, n_raw, comp);
                  if (tr_plan.active()
                      && ed::symmetry::sym_profile_enabled()) {
                      std::fprintf(stderr,
                          "[sym-profile] time-reversal pairing: skipping "
                          "%zu of %ld conjugate sectors\n",
                          tr_plan.n_skipped, n_ops);
                  }

                  // Pre-build the beta grid once (same logic as the
                  // per-n_up binding but shared across ALL sectors).
                  if (opts.betas.empty() && opts.num_temp_bins > 0
                      && opts.temp_min > 0.0
                      && opts.temp_max > opts.temp_min) {
                      opts.betas.reserve(opts.num_temp_bins);
                      const double t_lo = opts.temp_min;
                      const double t_hi = opts.temp_max;
                      const std::size_t n = opts.num_temp_bins;
                      for (std::size_t i = 0; i < n; ++i) {
                          const double T = (n == 1)
                              ? t_lo
                              : t_lo + (t_hi - t_lo)
                                * static_cast<double>(i)
                                / static_cast<double>(n - 1);
                          opts.betas.push_back(
                              T > 0.0 ? 1.0 / T : 1.0 / 1e-300);
                      }
                  }

                  // Lock random seed once: deterministic + lower variance.
                  if (opts.random_seed == 0) {
                      opts.random_seed = std::random_device{}();
                  }

                  // Wave B3: for KPM-DOS, estimate spectral bounds on the
                  // globally largest sector and share across all sectors.
                  double shared_e_min =
                      std::numeric_limits<double>::quiet_NaN();
                  double shared_e_max =
                      std::numeric_limits<double>::quiet_NaN();
                  if (opts.method ==
                          ed::workflows::ThermalOptions::Method::KpmDos
                      && !(std::isfinite(opts.e_min_override)
                           && std::isfinite(opts.e_max_override))
                      && n_ops > 1) {
                      std::size_t best_i   = 0;
                      std::uint64_t best_dim = 0;
                      for (std::size_t i = 0;
                           i < static_cast<std::size_t>(n_ops); ++i) {
                          if (set.operators[i] &&
                              set.operators[i]->dim() > best_dim) {
                              best_dim = set.operators[i]->dim();
                              best_i   = i;
                          }
                      }
                      auto* best_sec = set.operators[best_i].get();
                      if (best_sec && best_sec->dim() > 0) {
                          try {
                              std::mt19937 gen(opts.random_seed
                                                   ? opts.random_seed
                                                   : 0xdeadbeefULL);
                              double lo = 0.0, hi = 0.0;
                              ed::kpm_dos::MatVec H_mv =
                                  [best_sec](const Complex* in, Complex* out,
                                              int n) {
                                      best_sec->apply(in, out,
                                          static_cast<std::size_t>(n));
                                  };
                              ed::kpm_dos::estimate_spectral_bounds(
                                  H_mv, best_sec->dim(),
                                  /*krylov_dim=*/80,
                                  /*full_reorth=*/true,
                                  /*reorth_freq=*/10,
                                  /*tol=*/1e-10,
                                  gen, lo, hi);
                              shared_e_min = lo;
                              shared_e_max = hi;
                          } catch (...) {}
                      }
                  }

                  const bool need_per_sector_outdir =
                      !opts.output_dir.empty()
                      && !HDF5IO::isDisabledOutputPath(opts.output_dir);

                  // Sector-level OMP parallelism gate (same as the per-n_up
                  // binding). With this binding, the parallel region covers
                  // ALL (n_up, irrep) sectors simultaneously, giving better
                  // load balancing than two nested loops.
                  bool sector_parallel = false;
                  if (const char* env =
                          std::getenv("ED_SYM_SECTOR_PARALLEL")) {
                      sector_parallel = (env[0] == '1');
                  }

                  // Pre-indexed result storage: each OMP thread writes to its
                  // own slot ii -- no push_back races.
                  std::vector<ed::ThermalResult> all_results(
                      static_cast<std::size_t>(n_ops));

                  #pragma omp parallel for schedule(dynamic, 1) \
                      if(sector_parallel)
                  for (long ii = 0; ii < n_ops; ++ii) {
                      const std::size_t i = static_cast<std::size_t>(ii);
                      auto* op = set.operators[i].get();
                      if (!op || op->dim() == 0) continue;
                      if (tr_plan.skip[i]) continue;  // TR: copy from partner
                      ed::workflows::ThermalOptions topts = opts;
                      topts.selected_sectors.clear();
                      if (need_per_sector_outdir) {
                          const auto& tag = set.tags[i];
                          topts.output_dir = opts.output_dir
                              + "/sz_" + std::to_string(tag.n_up)
                              + "_sector_k_"
                              + std::to_string(tag.sector_index);
                      } else {
                          topts.output_dir.clear();
                      }
                      if (std::isfinite(shared_e_min)
                          && std::isfinite(shared_e_max)) {
                          topts.e_min_override = shared_e_min;
                          topts.e_max_override = shared_e_max;
                      }
                      all_results[i] = ed::workflows::thermal(*op, topts);
                  }

                  // Serial collection: flat combine across ALL (n_up, irrep).
                  std::vector<ThermodynamicData>      per_sector_thermo;
                  std::vector<std::uint64_t>          per_sector_dims;
                  std::vector<ed::ThermalSectorEntry> per_sector;
                  double gs_E = std::numeric_limits<double>::infinity();
                  std::string sector_lane;
                  std::size_t sector_mpi_size = 1;

                  for (long ii = 0; ii < n_ops; ++ii) {
                      const std::size_t i = static_cast<std::size_t>(ii);
                      auto* op = set.operators[i].get();
                      if (!op || op->dim() == 0) continue;
                      if (tr_plan.skip[i]) {
                          // TR pairing: source the conjugate partner's result.
                          all_results[i] = all_results[tr_plan.source[i]];
                      }
                      auto& tr = all_results[i];
                      if (sector_lane.empty()
                          && !tr.backend.lane.empty()) {
                          sector_lane     = tr.backend.lane;
                          sector_mpi_size = tr.backend.mpi_size;
                      }
                      if (tr.thermo.temperatures.empty()) {
                          if (std::isfinite(tr.ground_state_energy))
                              gs_E = std::min(gs_E, tr.ground_state_energy);
                          continue;
                      }
                      const auto& tag = set.tags[i];
                      per_sector_thermo.push_back(tr.thermo);
                      per_sector_dims.push_back(tag.sector_dim);
                      ed::ThermalSectorEntry entry;
                      entry.sz_index            = tag.n_up;
                      entry.ground_state_energy = tr.ground_state_energy;
                      entry.thermo              = tr.thermo;
                      entry.tag                 = tag;
                      per_sector.push_back(std::move(entry));
                      if (std::isfinite(tr.ground_state_energy))
                          gs_E = std::min(gs_E, tr.ground_state_energy);
                  }

                  // Stage 5 mirror pass: duplicate every solved n_up < N/2
                  // block into its flip partner N - n_up (same irrep, same
                  // spectrum, same dim), then keep only entries inside the
                  // originally requested window (the clamp may have solved
                  // mirror-only blocks below req_lo).
                  if (flip_transport) {
                      const std::size_t solved = per_sector.size();
                      for (std::size_t e = 0; e < solved; ++e) {
                          const int n = per_sector[e].tag.n_up;
                          const int m = N_sites - n;
                          if (m == n || m < req_lo || m > req_hi) continue;
                          ed::ThermalSectorEntry mirror = per_sector[e];
                          mirror.sz_index  = m;
                          mirror.tag.n_up  = m;
                          per_sector_thermo.push_back(mirror.thermo);
                          per_sector_dims.push_back(mirror.tag.sector_dim);
                          per_sector.push_back(std::move(mirror));
                      }
                      std::vector<ThermodynamicData>      f_thermo;
                      std::vector<std::uint64_t>          f_dims;
                      std::vector<ed::ThermalSectorEntry> f_sector;
                      for (std::size_t e = 0; e < per_sector.size(); ++e) {
                          const int n = per_sector[e].tag.n_up;
                          if (n < req_lo || n > req_hi) continue;
                          f_thermo.push_back(std::move(per_sector_thermo[e]));
                          f_dims.push_back(per_sector_dims[e]);
                          f_sector.push_back(std::move(per_sector[e]));
                      }
                      per_sector_thermo = std::move(f_thermo);
                      per_sector_dims   = std::move(f_dims);
                      per_sector        = std::move(f_sector);
                  }

                  if (!per_sector_thermo.empty()) {
                      agg.thermo = ed::core::combine_sector_thermodynamics(
                          per_sector_thermo, per_sector_dims);
                  }
                  agg.per_sector          = std::move(per_sector);
                  agg.ground_state_energy = std::isfinite(gs_E) ? gs_E : 0.0;
                  if (!sector_lane.empty()) {
                      agg.backend.lane     = sector_lane;
                      agg.backend.mpi_size = sector_mpi_size;
                  }
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")    = 0.5,
          py::arg("opts")      = ed::workflows::ThermalOptions{},
          py::arg("n_up_min")  = 0,
          py::arg("n_up_max")  = -1,
          R"pbdoc(
        All-Sz flat-pool finite-T workflow (Jun 2026 architecture).

        Replaces the Python-level ``ThreadPoolExecutor`` n_up outer loop with
        a single C++ call. Internally:

        1. Loads the Hamiltonian and symmetry group info **once**.
        2. Calls ``enumerate_full_orbit_reps`` **once** (O(2^N × |G|)),
           partitions reps by popcount to get per-n_up fixed-Sz reps.
        3. Builds ALL ``(n_up, irrep)`` sector operators in one pass over
           the range ``[n_up_min, n_up_max]``.
        4. Runs a single ``#pragma omp parallel for`` (gate:
           ``ED_SYM_SECTOR_PARALLEL=1``) over the full flat sector list,
           giving the scheduler unobstructed access to all (n_up × irrep)
           sectors simultaneously.
        5. Combines all per-sector thermodynamics in one
           ``combine_sector_thermodynamics`` call.

        Eliminates the N+1 cold-start penalty (JSON + orbit rep scan per
        n_up) that the per-n_up Python loop incurs.

        Parameters
        ----------
        directory : str
            Hamiltonian directory (must contain ``automorphism_results/``).
        num_sites : int
            Number of lattice sites.
        spin_l : float, optional
            Local spin magnitude (default 0.5).
        opts : ThermalOptions, optional
            Shared finite-T options for all sectors.
        n_up_min : int, optional
            Minimum n_up (default 0).
        n_up_max : int, optional
            Maximum n_up (-1 = num_sites, the default).

        Returns
        -------
        ThermalResult
            ``thermo`` carries Z-weighted combined thermodynamics over ALL
            (n_up, irrep) sectors; ``per_sector`` lists every sector with
            ``tag.n_up`` and ``tag.sector_index`` set.
    )pbdoc");
}
