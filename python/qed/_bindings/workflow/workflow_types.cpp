// =============================================================================
// python/qed/_bindings/workflow/workflow_types.cpp
//
// The workflow VALUE types: BackendConstraints, the SolveMethod /
// ThermalMethod / SpectralMethod enums, SolveOptions / ThermalOptions /
// SpectralOptions, GroundStateResult / ThermalResult / SpectralResult and
// their satellites (BackendMetadata, KrylovDiagnostics, EigenvectorRef,
// SectorTag, TpqStateSnapshot, ThermalSectorEntry, SpectralSectorEntry),
// plus `combine_thermo_weighted`.
//
// Registered FIRST: pybind11 renders a later binding's signature with the
// C++ type name unless the type is already known to the module.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_types(py::module_& m) {
    // -----------------------------------------------------------------
    // BackendConstraints (Geometry-side knobs the caller can override).
    // -----------------------------------------------------------------
    py::class_<ed::BackendConstraints>(m, "BackendConstraints")
        .def(py::init<>())
        .def_readwrite("allow_gpu",     &ed::BackendConstraints::allow_gpu)
        .def_readwrite("gpu_dim_floor", &ed::BackendConstraints::gpu_dim_floor)
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
                       &ed::workflows::SolveOptions::selected_sectors)
        // Stage 12 (SU(2) rollout): total-spin axis.
        .def_readwrite("two_total_spin",
                       &ed::workflows::SolveOptions::two_total_spin)
        .def_readwrite("label_total_spin",
                       &ed::workflows::SolveOptions::label_total_spin);

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
        .def_readwrite("two_S",           &ed::SectorTag::two_S)
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
            if (t.two_S >= 0) s += ", 2S=" + std::to_string(t.two_S);
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
                      &ed::GroundStateResult::sector_index_of_eigenvalue)
        // Stage 12 (SU(2) rollout): total-spin labels, parallel to
        // ``eigenvalues`` when present.
        .def_readonly("s2_of_eigenvalue",
                      &ed::GroundStateResult::s2_of_eigenvalue)
        .def_readonly("two_S_of_eigenvalue",
                      &ed::GroundStateResult::two_S_of_eigenvalue);

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
        .def_readwrite("random_seed",  &ed::workflows::ThermalOptions::random_seed)
        .def_readwrite("spin_flip",
                       &ed::workflows::ThermalOptions::spin_flip)
        .def_readwrite("time_reversal",
                       &ed::workflows::ThermalOptions::time_reversal)
        .def_readwrite("star_maps", &ed::workflows::ThermalOptions::star_maps)
        .def_readwrite("sz_parity", &ed::workflows::ThermalOptions::sz_parity)
        // Stage 12f (SU(2) rollout): per-tower stochastic sampling.
        .def_readwrite("two_total_spin",
                       &ed::workflows::ThermalOptions::two_total_spin)
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

    // Stage 12f (SU(2) rollout): degeneracy-weighted recombination for
    // the per-tower thermal driver -- Z = sum_S (2S+1) Z_S, implemented
    // by the shifted-F mixture with F_S -> F_S - T ln(2S+1).
    m.def("combine_thermo_weighted",
          [](const std::vector<::ThermodynamicData>& blocks,
             const std::vector<double>& degeneracy) {
              std::vector<std::uint64_t> dims(blocks.size(), 1);
              return ed::core::combine_sector_thermodynamics(
                  blocks, dims, degeneracy);
          },
          py::arg("blocks"), py::arg("degeneracy"),
          "Recombine per-block ThermodynamicData with positive weights "
          "g_s (Z = sum_s g_s Z_s). Used by qed.thermal(total_spin=...) "
          "to merge per-spin-tower curves with (2S+1) multiplicities.");

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
                      &ed::ThermalResult::tpq_state_snapshots)
        .def_readonly("dos_energies", &ed::ThermalResult::dos_energies)
        .def_readonly("dos_values",   &ed::ThermalResult::dos_values);

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
        .def_readonly("static_sf", &ed::SpectralSectorEntry::static_sf)
        .def_readonly("notes",     &ed::SpectralSectorEntry::notes);

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
                      &ed::SpectralResult::selection_rule_label)
        // Stage 12g (SU(2) rollout): total-spin label of the CF source
        // state (-1 = unlabeled). Wigner-Eckart: a rank-1 spin probe
        // reaches only final states with S' in {S-1, S, S+1}.
        .def_readonly("gs_two_S", &ed::SpectralResult::gs_two_S)
        .def_readonly("gs_s2",    &ed::SpectralResult::gs_s2);

}
