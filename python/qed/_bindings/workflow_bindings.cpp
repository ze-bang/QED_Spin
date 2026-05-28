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
#include <ed/core/fixed_sz_operator.h>   // FixedSzOperator (for device='gpu' promotion)
#include <ed/core/linear_operator.h>
#include <ed/core/make_operator.h>
#include <ed/core/operator.h>
#include <ed/core/results.h>
#include <ed/core/sector_loop.h>          // StreamingSymmetryHandle (SOTA)
#include <ed/core/sector_thermo.h>        // combine_sector_thermodynamics (SOTA)
#include <ed/core/select_backend.h>
#include <ed/core/streaming_symmetry.h>
#include <ed/dssf/cross_sector_orbit_observable.h>  // SOTA cross-irrep observable
#include <ed/matvec/backends/cpu_backend.h>          // CpuBackend for cf_spectral_from_vector
#include <ed/observables/cf_spectral_kernel.h>      // cf_spectral_from_vector
#include <ed/observables/ftlm_cross_irrep_kernel.h>  // SOTA finite-T cross-irrep
#include <ed/orchestrator.h>
#include <ed/solvers/kpm_dos.h>                      // Wave B3: estimate_spectral_bounds
#ifdef WITH_CUDA
#  include <ed/gpu/gpu_operator.cuh>     // GPUOperator + convertOperatorToGPU helper
#endif

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

namespace py = pybind11;

namespace {

using Complex = std::complex<double>;

// Pybind11 cannot move a captured `std::unique_ptr<LinearOperator>` out of
// a Python-owned Operator easily; we instead accept the raw `Operator&` /
// `FixedSzOperator&` and rely on the fact that both inherit from
// `ed::LinearOperator`. The orchestrator calls take `const LinearOperator&`,
// so a simple reference upcast is enough.

// ---------------------------------------------------------------------------
// GPU promotion helper (May 2026 follow-up to the "Universal save contract").
//
// ``qed.solve/thermal/spectral(device='gpu')`` flips
// ``opts.backend.allow_gpu = true`` in the Python wrappers, but the
// orchestrator's ``ed::select_backend(geom, c)`` will only pick the
// ``CudaBackend`` lane when the operator's ``Geometry`` either lives in
// device memory (``MemorySpace::CudaDevice``) or advertises lazy device
// matvec support (``supports_device_matvec=true``). The plain
// ``ed::Operator`` and ``ed::FixedSzOperator`` (no symmetry) classes
// advertise neither, so ``device='gpu'`` was silently downgraded to
// ``CpuBackend`` -- the user got CPU performance while paying for the
// GPU runtime check.
//
// The streaming-symmetry binding ``workflows_solve_streaming_symmetry_
// directory`` is unaffected: ``StreamingSymmetryOperator::SectorView``
// already advertises ``supports_device_matvec=true`` via the lazy GPU
// mirror that ``bind_cuda_for_sector`` materialises.
//
// This helper bridges the gap for the plain ``Operator`` /
// ``FixedSzOperator`` lanes by lazily constructing a ``GPUOperator``
// (or ``GPUFixedSzOperator``) from the host operator's term list when
// the caller actually wants the GPU lane. When the build does not
// have CUDA, or no NVIDIA device is visible, or the operator already
// supplies a device matvec path, we return ``nullptr`` -- the caller
// uses the original host operator.
//
// The returned ``unique_ptr`` OWNS the GPU operator; callers MUST keep
// it alive for the duration of the workflow call. The pattern in
// every binding is:
//
//     auto gpu_owned = maybe_promote_to_gpu(op, opts.backend);
//     const ed::LinearOperator& H =
//         gpu_owned ? static_cast<ed::LinearOperator&>(*gpu_owned) : op;
//     return ed::workflows::solve(H, std::move(opts));
//
// (Operator -> GPUOperator, FixedSzOperator -> GPUFixedSzOperator.)
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

inline bool needs_gpu_promotion(const ed::LinearOperator& op,
                                const ed::BackendConstraints& c) noexcept {
    if (!c.allow_gpu) return false;
    if (!gpu_runtime_available()) return false;
    const auto geom = op.geometry();
    // Already a device-resident operator OR advertises lazy device
    // matvec (streaming-symmetry SectorView, etc.) -> orchestrator
    // picks CudaBackend natively.
    if (ed::matvec::is_device(geom.memory_space)) return false;
    if (geom.supports_device_matvec) return false;
    // Distributed lanes pick MPI / MPI+CUDA via their own backends;
    // never promote a DistributedOperator here.
    if (ed::matvec::is_distributed(geom.memory_space)) return false;
    return true;
}

/// FullDiag fallback in ``solve_on<Backend>`` calls the bound matvec
/// with HOST ``std::vector<Complex>`` storage (the LAPACK dense
/// eigensolver is host-only, so the column-build runs on host
/// pointers). A GPUOperator's ``bind_cpu()`` throws because the
/// operator is device-pointer-only; promoting a host Operator to a
/// GPUOperator and then routing FullDiag through it would crash.
///
/// FullDiag is the auto-selected method for ``global_dim <= 2^12 =
/// 4096`` (see ``auto_solve_method`` in orchestrator.cpp). At those
/// dimensions the matvec is negligible compared to the O(N^3) LAPACK
/// solve and the GPU lane offers no measurable win, so silently
/// keeping the operator on the host is both safe and good for
/// throughput. The promoter therefore declines promotion whenever
/// the resolved method is (or would be) FullDiag.
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
///   * mTPQ, cTPQ   : any backend.
///
/// When the chosen method is FTLM we must NOT promote: routing a
/// GPUOperator into ``ed::thermal::ftlm`` would land on
/// ``std::is_same_v<B, ed::matvec::CpuBackend>`` failing and the
/// orchestrator would raise a runtime error. The promoter therefore
/// declines promotion for FTLM. (Before this helper landed, plain
/// ``Operator`` was silently demoted to CpuBackend in
/// ``select_backend``, so FTLM-with-``allow_gpu=true`` was a no-op
/// rather than a crash; preserving that contract avoids a regression.)
inline bool thermal_method_supports_gpu(
    ed::workflows::ThermalOptions::Method m) noexcept {
    using M = ed::workflows::ThermalOptions::Method;
    return m == M::LTLM
        || m == M::KpmDos
        || m == M::mTPQ
        || m == M::cTPQ;
}

/// Spectral lanes split along the same "backend-aware vs host-only"
/// boundary:
///   * GroundStateCF : runs the inner solve + CF kernel through
///                     ``H.template bind<B>()`` -- GPU-clean.
///   * FtlmDynamical : drives ``compute_dynamical_correlation`` which
///                     calls ``H.apply(host_in, host_out)``. A
///                     GPUOperator's ``apply`` reinterprets the host
///                     pointers as device pointers, which crashes.
///   * KpmDynamical  : hands a stack-allocated ``CpuBackend`` and host
///                     buffers to ``kpm_dynamical_correlator``. Same
///                     host-only constraint as FtlmDynamical.
inline bool spectral_method_supports_gpu(
    ed::workflows::SpectralOptions::Method m) noexcept {
    using M = ed::workflows::SpectralOptions::Method;
    return m == M::GroundStateCF;
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
                  "KrylovSchur for solve; LTLM/KPM_DOS/mTPQ/cTPQ for "
                  "thermal; GroundStateCF for spectral).",
            py::module_::import("builtins").attr("RuntimeWarning"),
            py::arg("stacklevel") = 2);
    } catch (const py::error_already_set&) {
        // Best-effort -- never let the warning machinery break the
        // workflow call. The caller still gets the correct CPU result.
    }
}

inline std::unique_ptr<ed::LinearOperator>
maybe_promote_to_gpu(Operator& host_op,
                     const ed::BackendConstraints& c) {
#ifdef WITH_CUDA
    if (!needs_gpu_promotion(host_op, c)) return nullptr;
    // FixedSzOperator dispatches to GPUFixedSzOperator (preserves the
    // n_up projection); plain Operator -> GPUOperator. The dynamic_cast
    // chain runs in derived-first order so the fixed-Sz subclass wins.
    if (auto* fsz = dynamic_cast<FixedSzOperator*>(&host_op)) {
        auto gpu = std::make_unique<GPUFixedSzOperator>(
            static_cast<int>(fsz->getNumBits()),
            static_cast<int>(fsz->getNUp()),
            fsz->getSpin());
        if (!convertOperatorToGPU(*fsz, *gpu)) {
            return nullptr;
        }
        return gpu;
    }
    auto gpu = std::make_unique<GPUOperator>(
        static_cast<int>(host_op.getNumBits()),
        host_op.getSpin());
    if (!convertOperatorToGPU(host_op, *gpu)) {
        return nullptr;
    }
    return gpu;
#else
    (void)host_op; (void)c;
    return nullptr;
#endif
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
        .value("cTPQ",   ed::workflows::ThermalOptions::Method::cTPQ)
        .value("KpmDos", ed::workflows::ThermalOptions::Method::KpmDos)
        .export_values();

    py::class_<ed::workflows::ThermalOptions>(m, "ThermalOptions")
        .def(py::init<>())
        .def_readwrite("method",       &ed::workflows::ThermalOptions::method)
        .def_readwrite("num_samples",  &ed::workflows::ThermalOptions::num_samples)
        .def_readwrite("krylov_dim",   &ed::workflows::ThermalOptions::krylov_dim)
        .def_readwrite("taylor_order", &ed::workflows::ThermalOptions::taylor_order)
        .def_readwrite("betas",        &ed::workflows::ThermalOptions::betas)
        .def_readwrite("delta_beta",   &ed::workflows::ThermalOptions::delta_beta)
        .def_readwrite("beta_max",     &ed::workflows::ThermalOptions::beta_max)
        .def_readwrite("random_seed",  &ed::workflows::ThermalOptions::random_seed)
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
        .def_readwrite("kpm_num_moments",
                       &ed::workflows::ThermalOptions::kpm_num_moments)
        .def_readwrite("kpm_num_random_vectors",
                       &ed::workflows::ThermalOptions::kpm_num_random_vectors)
        // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
        // user-supplied probe-betas for mTPQ/cTPQ state-vector
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
        .def_readonly("initial", &ed::SpectralSectorEntry::initial)
        .def_readonly("final",   &ed::SpectralSectorEntry::final_)
        .def_readonly("S_real",  &ed::SpectralSectorEntry::S_real)
        .def_readonly("S_imag",  &ed::SpectralSectorEntry::S_imag);

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
    m.def("workflows_solve",
          [](Operator& op, ed::workflows::SolveOptions opts) {
              // GPU promotion (May 2026): when the caller requested the
              // GPU lane (``opts.backend.allow_gpu==true``) but the
              // operator does not natively supply a device matvec, lazily
              // construct a GPU mirror (GPUOperator / GPUFixedSzOperator)
              // from the host operator's term list. See
              // ``maybe_promote_to_gpu`` for the full contract.
              //
              // Exception: FullDiag (small-dim O(N^3) dense LAPACK
              // solve) keeps the host operator since its column-build
              // calls the matvec with host pointers; routing through a
              // GPUOperator would crash inside ``solve_on<Backend>``'s
              // ``H.bind_cpu()`` call.
              std::unique_ptr<ed::LinearOperator> gpu_owned;
              if (will_use_full_diag(op, opts)) {
                  // Silent demotion is the legacy contract for small-dim
                  // FullDiag (GPU offers nothing over LAPACK at dim <
                  // 2^12 and the column-build needs host pointers
                  // anyway). Surface it as a Python RuntimeWarning so
                  // the caller can audit the demotion instead of
                  // discovering it via profiling.
                  warn_silent_cpu_fallback(
                      "qed.solve (FullDiag)", opts.backend);
              } else {
                  gpu_owned = maybe_promote_to_gpu(op, opts.backend);
              }
              const ed::LinearOperator& H =
                  gpu_owned ? static_cast<const ed::LinearOperator&>(*gpu_owned)
                            : static_cast<const ed::LinearOperator&>(op);
              return ed::workflows::solve(H, std::move(opts));
          },
          py::arg("op"),
          py::arg("opts") = ed::workflows::SolveOptions{},
          "Run the unified ground-state Krylov solver (Phase 4.2 collapse). "
          "Backend (CPU/GPU/MPI/MPI+GPU) is chosen via `ed::select_backend`. "
          "When ``opts.backend.allow_gpu`` is true and the operator does not "
          "natively expose a device matvec, a transient GPUOperator mirror "
          "is built from the host term list so the GPU lane actually runs. "
          "FullDiag stays on the CPU lane (no GPU implementation) and the "
          "binding emits a Python RuntimeWarning so the demotion is "
          "visible at the call site.");

    m.def("workflows_thermal",
          [](Operator& op, ed::workflows::ThermalOptions opts) {
              // Skip promotion for thermal methods whose orchestrator
              // dispatch is host-only (FTLM). Preserves the legacy
              // silent CPU lane for ``device='gpu'`` callers running
              // FTLM and prevents the runtime throw inside
              // ``solve_on<CudaBackend>``. Surface the demotion as a
              // Python RuntimeWarning so the caller can audit it.
              std::unique_ptr<ed::LinearOperator> gpu_owned;
              if (thermal_method_supports_gpu(opts.method)) {
                  gpu_owned = maybe_promote_to_gpu(op, opts.backend);
              } else {
                  // For ``allow_gpu=true`` with a host-only method we
                  // also need to clear ``allow_gpu`` so ``select_backend``
                  // doesn't try to route through ``CudaBackend`` on
                  // operators that DO advertise device matvec (e.g.
                  // streaming-symmetry SectorView).
                  warn_silent_cpu_fallback(
                      "qed.thermal (FTLM)", opts.backend);
                  opts.backend.allow_gpu = false;
              }
              const ed::LinearOperator& H =
                  gpu_owned ? static_cast<const ed::LinearOperator&>(*gpu_owned)
                            : static_cast<const ed::LinearOperator&>(op);
              return ed::workflows::thermal(H, std::move(opts));
          },
          py::arg("op"),
          py::arg("opts") = ed::workflows::ThermalOptions{},
          "Run the unified finite-temperature workflow (FTLM / LTLM / mTPQ / "
          "cTPQ / KPM-DOS) over the auto-selected Backend. ``allow_gpu`` "
          "transparently lifts a host operator to a GPUOperator mirror so "
          "the device-matvec lane runs without manual conversion.");

    m.def("workflows_spectral",
          [](Operator& op,
             std::vector<Operator*> observables,
             ed::workflows::SpectralOptions opts) {
              // Promotion is only safe for the GroundStateCF lane (which
              // routes through the backend matvec abstraction). The
              // FtlmDynamical / KpmDynamical lanes call ``H.apply`` with
              // HOST pointers, so a GPUOperator's device-pointer apply
              // would crash. Skip promotion in those cases; the
              // orchestrator falls through to CpuBackend (and we surface
              // the silent demotion as a Python RuntimeWarning).
              const bool can_promote =
                  spectral_method_supports_gpu(opts.method);
              std::unique_ptr<ed::LinearOperator> gpu_owned_H;
              if (can_promote) {
                  gpu_owned_H = maybe_promote_to_gpu(op, opts.backend);
              } else {
                  warn_silent_cpu_fallback(
                      "qed.spectral (FtlmDynamical/KpmDynamical)",
                      opts.backend);
                  opts.backend.allow_gpu = false;
              }
              // Observables share the Hamiltonian's backend lane: when
              // H is promoted to GPU, the matching observables must
              // run on the device too (CF / KPM / FTLM kernels apply
              // observables via the same Backend pointers as H).
              std::vector<std::unique_ptr<ed::LinearOperator>> gpu_owned_obs;
              gpu_owned_obs.reserve(observables.size());
              std::vector<const ed::LinearOperator*> obs;
              obs.reserve(observables.size());
              for (auto* o : observables) {
                  if (!o) continue;
                  std::unique_ptr<ed::LinearOperator> o_owned;
                  if (can_promote) {
                      o_owned = maybe_promote_to_gpu(*o, opts.backend);
                  }
                  if (o_owned) {
                      obs.push_back(
                          static_cast<const ed::LinearOperator*>(o_owned.get()));
                      gpu_owned_obs.push_back(std::move(o_owned));
                  } else {
                      obs.push_back(static_cast<const ed::LinearOperator*>(o));
                  }
              }
              const ed::LinearOperator& H =
                  gpu_owned_H ? static_cast<const ed::LinearOperator&>(*gpu_owned_H)
                              : static_cast<const ed::LinearOperator&>(op);
              return ed::workflows::spectral(H, obs, std::move(opts));
          },
          py::arg("op"),
          py::arg("observables"),
          py::arg("opts") = ed::workflows::SpectralOptions{},
          "Run the unified dynamical-correlator workflow "
          "(continued-fraction Lanczos or FTLM dynamical) over the auto-"
          "selected Backend. When ``allow_gpu`` is set and the Hamiltonian "
          "is a host operator, both the Hamiltonian and the observable "
          "pair are mirrored onto the device so the GPU lane actually runs.");

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
              if (!fixed_sz_n_up.is_none()) {
                  spec.fixed_sz = fixed_sz_n_up.cast<int>();
              }

              ed::GroundStateResult agg;
              {
                  py::gil_scoped_release release;
                  // Call the streaming-symmetry sub-factory directly
                  // instead of the top-level `ed::make_operator` so the
                  // bind never references `make_distributed_operator`
                  // (whose constructor lives in `ed_distributed`, not
                  // linked against `_core.so`).
                  auto base_op = ed::make_streaming_symmetry_operator(spec);
                  ed::core::StreamingSymmetryHandle handle(base_op.get());

                  const std::size_t num_sectors = handle.num_sectors();
                  if (num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_solve_streaming_symmetry_directory: "
                          "make_operator returned an operator with no "
                          "symmetry sectors; check the "
                          "automorphism_results/ directory.");
                  }

                  const std::vector<std::size_t> sector_indices =
                      ed::core::filter_sectors(num_sectors,
                                               opts.selected_sectors);

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
                      if (phase1_min.size() <= target_num_eigs) {
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
                  for (std::size_t k : iter_sectors) {
                      auto sec = handle.sector(k);
                      if (!sec || sec->dim() == 0) continue;
                      ed::workflows::SolveOptions sopts = opts;
                      sopts.num_eigs = std::min<std::size_t>(
                          opts.num_eigs ? opts.num_eigs : 1, sec->dim());
                      // Avoid recursing into the streaming loop when
                      // the orchestrator is invoked on a SectorView.
                      sopts.selected_sectors.clear();
                      sopts.use_symmetry = false;
                      if (need_per_sector_outdir) {
                          sopts.output_dir = opts.output_dir
                              + "/sector_k_" + std::to_string(k);
                      }
                      auto sr = ed::workflows::solve(*sec, sopts);
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

              ed::ThermalResult agg;
              {
                  py::gil_scoped_release release;
                  auto base_op = ed::make_streaming_symmetry_operator(spec);
                  ed::core::StreamingSymmetryHandle handle(base_op.get());

                  const std::size_t num_sectors = handle.num_sectors();
                  if (num_sectors == 0) {
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

                  for (std::size_t k : sector_indices) {
                      auto sec = handle.sector(k);
                      if (!sec || sec->dim() == 0) continue;
                      ed::workflows::ThermalOptions topts = opts;
                      topts.selected_sectors.clear();
                      if (need_per_sector_outdir) {
                          topts.output_dir = opts.output_dir
                              + "/sector_k_" + std::to_string(k);
                      } else {
                          // No user-supplied output_dir -- keep the
                          // per-sector call silent on disk.
                          topts.output_dir.clear();
                      }
                      // FTLM is host-only in the orchestrator (see the
                      // ``ed::thermal: FTLM lane requires a CpuBackend
                      // today`` guard in ``orchestrator.cpp``). The
                      // streaming-symmetry SectorView advertises
                      // ``supports_device_matvec=true`` (lazy GPU
                      // mirror), so without this pin
                      // ``select_backend`` would pick the CudaBackend
                      // lane and the orchestrator would raise mid-loop.
                      // Force the CPU lane explicitly for FTLM; the
                      // other methods (LTLM / KpmDos / mTPQ / cTPQ)
                      // are GPU-clean and stay on the caller's chosen
                      // backend. Warn the FIRST time we demote so the
                      // caller can audit the choice; per-sector
                      // demotion is the same decision repeated.
                      if (topts.method
                          == ed::workflows::ThermalOptions::Method::FTLM) {
                          if (k == sector_indices.front()) {
                              py::gil_scoped_acquire gil;
                              warn_silent_cpu_fallback(
                                  "qed.thermal (FTLM, streaming-symmetry)",
                                  topts.backend);
                          }
                          topts.backend.allow_gpu = false;
                          topts.backend.allow_mpi = false;
                      }
                      // Wave B3: inject shared spectral bounds when
                      // the binding-level estimator succeeded.
                      if (std::isfinite(shared_e_min)
                          && std::isfinite(shared_e_max)) {
                          topts.e_min_override = shared_e_min;
                          topts.e_max_override = shared_e_max;
                      }
                      auto tr = ed::workflows::thermal(*sec, topts);

                      if (tr.thermo.temperatures.empty()) {
                          // Method (e.g. raw TPQ) did not populate
                          // ThermodynamicData; skip the sector for
                          // recombination but record the GS energy.
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
            Per-sector finite-T options (FTLM / LTLM / mTPQ / cTPQ /
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
                  auto base_op = ed::make_streaming_symmetry_operator(spec);
                  ed::core::StreamingSymmetryHandle handle(base_op.get());

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
                  for (std::size_t k : sector_indices) {
                      auto sec = handle.sector(k);
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
                          gs_energy     = E_k;
                          gs_sector_idx = k;
                      }
                      any_solved = true;
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
                      gs_sec.get()};
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

                  auto src_base = ed::make_streaming_symmetry_operator(src_spec);
                  ed::core::StreamingSymmetryHandle src_handle(src_base.get());
                  auto* src_fsz =
                      dynamic_cast<FixedSzStreamingSymmetryOperator*>(src_base.get());
                  auto* src_sym =
                      dynamic_cast<StreamingSymmetryOperator*>(src_base.get());

                  ed::dssf::CrossSectorOrbitObservable::OperatorRef src_ref;
                  if (src_fsz)      src_ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(*src_fsz);
                  else if (src_sym) src_ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(*src_sym);
                  else {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: streaming-symmetry operator dynamic_cast "
                          "failed (neither fixed-Sz nor full streaming op).");
                  }

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
                  const auto& psi0 = gs_sr.eigenvectors->host[0];
                  const double E0  = gs_sr.eigenvalues.front();
                  ed::SectorTag gs_src_tag = src_handle.sector_tag(gs_src_idx);

                  // -----------------------------------------------------
                  // (4) Build / re-use the target streaming operator. If
                  //     ``delta_n_up == 0`` we re-use the source object;
                  //     otherwise we build a second operator with the
                  //     shifted ``fixed_sz_n_up``.
                  // -----------------------------------------------------
                  std::unique_ptr<ed::LinearOperator> dst_base;
                  ed::core::StreamingSymmetryHandle* dst_handle_ptr = nullptr;
                  ed::core::StreamingSymmetryHandle dst_handle_storage =
                      src_handle;
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef dst_ref =
                      src_ref;

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
                      dst_base = ed::make_streaming_symmetry_operator(dst_spec);
                      dst_handle_storage =
                          ed::core::StreamingSymmetryHandle(dst_base.get());
                      auto* dst_fsz =
                          dynamic_cast<FixedSzStreamingSymmetryOperator*>(dst_base.get());
                      auto* dst_sym =
                          dynamic_cast<StreamingSymmetryOperator*>(dst_base.get());
                      if (dst_fsz)      dst_ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(*dst_fsz);
                      else if (dst_sym) dst_ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(*dst_sym);
                      else {
                          throw std::runtime_error(
                              "workflows_spectral_streaming_symmetry_cross_"
                              "irrep_directory: target operator dynamic_cast "
                              "failed.");
                      }
                  }
                  dst_handle_ptr = &dst_handle_storage;

                  // -----------------------------------------------------
                  // (5) Resolve the target sector via the selection
                  //     rule. ``Q`` lives in fractional reciprocal-
                  //     lattice units; sector_loop.h does the integer
                  //     quantisation.
                  // -----------------------------------------------------
                  double q_residual = 0.0;
                  const std::size_t dst_sector_idx =
                      ed::core::resolve_target_sector(
                          *dst_handle_ptr,
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
                      dst_handle_ptr->sector_tag(dst_sector_idx);

                  // -----------------------------------------------------
                  // (6) Build phi = O_Q |psi_0> in the target orbit
                  //     basis via CrossSectorOrbitObservable.
                  // -----------------------------------------------------
                  ed::dssf::CrossSectorOrbitObservable orb_obs(
                      src_ref, gs_src_idx,
                      dst_ref, dst_sector_idx,
                      tlist,
                      static_cast<float>(spin_l));
                  const std::size_t dim_dst = orb_obs.dim_dst();
                  if (dim_dst == 0) {
                      // Target sector is empty -- spectral function is
                      // identically zero.
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
                  std::vector<Complex> phi(dim_dst, Complex(0.0, 0.0));
                  orb_obs.apply(psi0.data(), phi.data(), dim_dst);

                  // -----------------------------------------------------
                  // (7) Run cf_spectral_from_vector on H restricted to
                  //     the target sector. Uses CpuBackend (host memory)
                  //     and the SectorView matvec.
                  // -----------------------------------------------------
                  auto dst_sec_view = dst_handle_ptr->sector(dst_sector_idx);
                  if (!dst_sec_view || dst_sec_view->dim() != dim_dst) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: target sector dim mismatch (view="
                          + std::to_string(dst_sec_view ? dst_sec_view->dim() : 0)
                          + ", observable=" + std::to_string(dim_dst) + ").");
                  }
                  ed::matvec::CpuBackend cpu_be;

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

                  auto apply_H = [&dst_sec_view](const Complex* x,
                                                 Complex*       y,
                                                 std::size_t    n) {
                      dst_sec_view->apply(x, y, n);
                  };
                  auto cf = ed::observables::cf_spectral_from_vector(
                      cpu_be, apply_H, dim_dst,
                      phi.data(), omega_grid, cfopts);

                  // -----------------------------------------------------
                  // (8) Marshal the result into ed::SpectralResult.
                  // -----------------------------------------------------
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
                  auto src_base = ed::make_streaming_symmetry_operator(src_spec);
                  ed::core::StreamingSymmetryHandle src_handle(src_base.get());
                  auto* src_fsz = dynamic_cast<FixedSzStreamingSymmetryOperator*>(src_base.get());
                  auto* src_sym = dynamic_cast<StreamingSymmetryOperator*>(src_base.get());
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef src_ref;
                  if (src_fsz)      src_ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(*src_fsz);
                  else if (src_sym) src_ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(*src_sym);
                  else throw std::runtime_error(
                      "workflows_spectral_streaming_symmetry_ftlm_cross_"
                      "irrep_directory: streaming-symmetry operator "
                      "dynamic_cast failed.");

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
                  std::unique_ptr<ed::LinearOperator> dst_base;
                  ed::core::StreamingSymmetryHandle dst_handle_storage =
                      src_handle;
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef dst_ref =
                      src_ref;
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
                      dst_base = ed::make_streaming_symmetry_operator(dst_spec);
                      dst_handle_storage =
                          ed::core::StreamingSymmetryHandle(dst_base.get());
                      auto* dst_fsz = dynamic_cast<FixedSzStreamingSymmetryOperator*>(dst_base.get());
                      auto* dst_sym = dynamic_cast<StreamingSymmetryOperator*>(dst_base.get());
                      if (dst_fsz)      dst_ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(*dst_fsz);
                      else if (dst_sym) dst_ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef::from(*dst_sym);
                      else throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_ftlm_cross_"
                          "irrep_directory: target dynamic_cast failed.");
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
                      auto src_view = src_handle.sector(k_src);
                      if (!src_view || src_view->dim() == 0) continue;

                      double q_residual = 0.0;
                      const std::size_t k_dst =
                          ed::core::resolve_target_sector(
                              dst_handle_storage,
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
                      auto dst_view = dst_handle_storage.sector(k_dst);
                      if (!dst_view || dst_view->dim() == 0) continue;

                      // Cross-irrep rectangular observable for THIS
                      // (k_src, k_dst) pair.
                      ed::dssf::CrossSectorOrbitObservable orb_obs(
                          src_ref, k_src,
                          dst_ref, k_dst,
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
                          dst_handle_storage.sector_tag(k_dst));
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
}
