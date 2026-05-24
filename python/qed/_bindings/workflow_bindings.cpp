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
// The legacy dispatcher surface in `dispatcher_bindings.cpp`
// (`exact_diagonalization_core` / `_from_directory` /
// `_streaming_symmetry*`) remains for the migration window. Phase 5 of
// the cleanup sweep deletes those once `workflow.py` / `thermal.py` /
// the Python test suite all route through the bindings below.
//
// Phase 1 of the ED Cleanup Sweep (May 2026).
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>

#include <ed/core/linear_operator.h>
#include <ed/core/operator.h>
#include <ed/core/results.h>
#include <ed/core/select_backend.h>
#include <ed/orchestrator.h>

#include <complex>
#include <cstdint>
#include <memory>
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
                       &ed::workflows::SolveOptions::precompute_basis_only);

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

    py::class_<ed::GroundStateResult>(m, "GroundStateResult")
        .def(py::init<>())
        .def_readonly("eigenvalues",  &ed::GroundStateResult::eigenvalues)
        .def_readonly("eigenvectors", &ed::GroundStateResult::eigenvectors)
        .def_readonly("krylov",       &ed::GroundStateResult::krylov)
        .def_readonly("backend",      &ed::GroundStateResult::backend)
        .def_readonly("hdf5_path",    &ed::GroundStateResult::hdf5_path);

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
                       &ed::workflows::ThermalOptions::broadening);

    py::class_<ed::ThermalResult>(m, "ThermalResult")
        .def(py::init<>())
        .def_readonly("ground_state_energy", &ed::ThermalResult::ground_state_energy)
        .def_readonly("krylov",              &ed::ThermalResult::krylov)
        .def_readonly("backend",             &ed::ThermalResult::backend)
        .def_readonly("hdf5_path",           &ed::ThermalResult::hdf5_path);

    // -----------------------------------------------------------------
    // SpectralOptions / SpectralResult.
    // -----------------------------------------------------------------
    py::enum_<ed::workflows::SpectralOptions::Method>(m, "SpectralMethod")
        .value("GroundStateCF",  ed::workflows::SpectralOptions::Method::GroundStateCF)
        .value("FtlmDynamical",  ed::workflows::SpectralOptions::Method::FtlmDynamical)
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
        // Wave A5: CLI parity knobs (FtlmDynamical sample/temperature
        // controls and the observable-type discriminator).
        .def_readwrite("num_samples",
                       &ed::workflows::SpectralOptions::num_samples)
        .def_readwrite("temperatures",
                       &ed::workflows::SpectralOptions::temperatures)
        .def_readwrite("observable_type",
                       &ed::workflows::SpectralOptions::observable_type);

    py::class_<ed::SpectralResult>(m, "SpectralResult")
        .def(py::init<>())
        .def_readonly("omega",        &ed::SpectralResult::omega)
        .def_readonly("S_real",       &ed::SpectralResult::S_real)
        .def_readonly("S_imag",       &ed::SpectralResult::S_imag)
        .def_readonly("errors_real",  &ed::SpectralResult::errors_real)
        .def_readonly("errors_imag",  &ed::SpectralResult::errors_imag)
        .def_readonly("krylov",       &ed::SpectralResult::krylov)
        .def_readonly("backend",      &ed::SpectralResult::backend)
        .def_readonly("hdf5_path",    &ed::SpectralResult::hdf5_path);

    // -----------------------------------------------------------------
    // The three entry points.
    // -----------------------------------------------------------------
    m.def("workflows_solve",
          [](Operator& op, ed::workflows::SolveOptions opts) {
              const ed::LinearOperator& H = op;
              return ed::workflows::solve(H, std::move(opts));
          },
          py::arg("op"),
          py::arg("opts") = ed::workflows::SolveOptions{},
          "Run the unified ground-state Krylov solver (Phase 4.2 collapse). "
          "Backend (CPU/GPU/MPI/MPI+GPU) is chosen via `ed::select_backend`.");

    m.def("workflows_thermal",
          [](Operator& op, ed::workflows::ThermalOptions opts) {
              const ed::LinearOperator& H = op;
              return ed::workflows::thermal(H, std::move(opts));
          },
          py::arg("op"),
          py::arg("opts") = ed::workflows::ThermalOptions{},
          "Run the unified finite-temperature workflow (FTLM / LTLM / mTPQ / "
          "cTPQ / KPM-DOS) over the auto-selected Backend.");

    m.def("workflows_spectral",
          [](Operator& op,
             std::vector<Operator*> observables,
             ed::workflows::SpectralOptions opts) {
              const ed::LinearOperator& H = op;
              std::vector<const ed::LinearOperator*> obs;
              obs.reserve(observables.size());
              for (auto* o : observables) {
                  obs.push_back(static_cast<const ed::LinearOperator*>(o));
              }
              return ed::workflows::spectral(H, obs, std::move(opts));
          },
          py::arg("op"),
          py::arg("observables"),
          py::arg("opts") = ed::workflows::SpectralOptions{},
          "Run the unified dynamical-correlator workflow "
          "(continued-fraction Lanczos or FTLM dynamical) over the auto-"
          "selected Backend.");
}
