// =============================================================================
// python/qed/_bindings/dispatcher_bindings.cpp
//
// Surface unification (May 2026): this file is the residual of the
// legacy "dispatcher" binding surface. Every
// ``exact_diagonalization_*`` Python forwarder was deleted in lockstep
// with the C++ ``ed::exact_diagonalization_*`` family removal; the
// canonical entry points now live in `workflow_bindings.cpp`
// (``_core.workflows_solve / thermal / spectral`` and the streaming-
// symmetry directory helper) and the Python module
// `qed.{solve,thermal,spectral}` re-exports a kwargs-only surface
// over them.
//
// What remains in this translation unit is the "type-level" plumbing
// that the surviving public API still needs:
//
//   * ``DiagonalizationMethod`` enum -- consumed by the legacy
//     ``EDParameters`` parameter-carrier and by Python helpers that
//     bridge user-facing solver names onto the orchestrator's
//     ``SolveOptions::method`` enum.
//   * ``EDParameters`` mutable parameter bag -- still used internally
//     by ``qed.workflow._diag_via_workflows_solve`` to carry the
//     legacy knobs (FTLM krylov_dim, TPQ taylor_order, KPM moments,
//     etc.) across the Python <-> C++ boundary before they get
//     translated into ``SolveOptions`` / ``ThermalOptions``.
//   * ``EDResults`` + ``ThermodynamicData`` -- the result envelope
//     every legacy consumer (including the MPI Python wrapper that
//     reads ``ed_distributed_main`` HDF5 dumps) expects.
//   * Symmetry attribute setters/getters on ``Operator`` /
//     ``FixedSzOperator`` -- the in-process bridge between
//     ``qed.symmetry.group_from_generators(...)`` and the streaming
//     kernel's expected ``SymmetryGroupInfo`` shape.
//   * Build introspection (``has_cuda_build`` / ``has_mpi_build``).
//
// The five ``exact_diagonalization_*`` ``m.def`` forwarders that used
// to live here (``_core``, ``_streaming_symmetry``,
// ``_streaming_symmetry_fixed_sz``, ``_from_directory``, and the
// ``Operator``/``FixedSzOperator`` overloads of ``_core``) are gone:
// the equivalent behaviour is reachable through
// ``_core.workflows_solve_streaming_symmetry_directory`` (streaming-
// symmetry) and ``_core.workflows_solve`` / ``workflows_thermal``
// (everything else).
// =============================================================================

#include "dispatcher_bindings.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <ed/core/ed_legacy_types.h>  // EDResults envelope (slim residue of ed_wrapper.h)
#include <ed/core/ed_parameters.h>
#include <ed/core/ed_types.h>
#include <ed/core/operator.h>
#include <ed/core/fixed_sz_operator.h>
#include <ed/core/results.h>          // ThermodynamicData + FTLMResults (legacy envelope)

#include <complex>
#include <cstdint>
#include <string>
#include <algorithm>
#include <cctype>
#include <vector>

namespace py = pybind11;

namespace {

using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// Convert a Python dict (matching the schema returned by
// qed.symmetry.group_from_generators / translation_group_1d) into a
// SymmetryGroupInfo. Both endpoints round-trip:
//   d = qed.symmetry.group_from_generators(...)
//   op.set_symmetry_info_from_dict(d)
//   d2 = op.get_symmetry_info_as_dict()
//   assert d2 == d  # up to ordering of generators
// -----------------------------------------------------------------------------
SymmetryGroupInfo dict_to_symmetry_info(const py::dict& d) {
    SymmetryGroupInfo info;

    auto get_or_default = [&](const char* key, auto default_) {
        return d.contains(key) ? d[key].cast<decltype(default_)>() : default_;
    };

    info.num_generators       = get_or_default("num_generators", uint64_t{0});
    info.generator_orders     = get_or_default("generator_orders", std::vector<int>{});
    info.generators           = get_or_default("generators", std::vector<std::vector<int>>{});
    info.max_clique           = get_or_default("max_clique", std::vector<std::vector<int>>{});
    info.power_representation = get_or_default("power_representation",
                                               std::vector<std::vector<int>>{});

    if (d.contains("sectors")) {
        for (auto handle : d["sectors"].cast<py::list>()) {
            auto sd = handle.cast<py::dict>();
            SectorMetadata s;
            s.sector_id        = sd.contains("sector_id")
                                     ? sd["sector_id"].cast<uint64_t>() : 0;
            s.quantum_numbers  = sd.contains("quantum_numbers")
                                     ? sd["quantum_numbers"].cast<std::vector<int>>()
                                     : std::vector<int>{};
            if (sd.contains("phase_factors")) {
                for (auto z_handle : sd["phase_factors"].cast<py::list>()) {
                    s.phase_factors.push_back(z_handle.cast<Complex>());
                }
            }
            // dimension is computed during basis generation -- leave at 0.
            info.sectors.push_back(std::move(s));
        }
    }

    return info;
}

py::dict symmetry_info_to_dict(const SymmetryGroupInfo& info) {
    py::dict d;
    d["num_generators"]       = info.num_generators;
    d["generator_orders"]     = info.generator_orders;
    d["generators"]           = info.generators;
    d["max_clique"]           = info.max_clique;
    d["power_representation"] = info.power_representation;
    py::list sectors;
    for (const auto& s : info.sectors) {
        py::dict sd;
        sd["sector_id"]       = s.sector_id;
        sd["quantum_numbers"] = s.quantum_numbers;
        py::list pf;
        for (const auto& z : s.phase_factors) {
            pf.append(z);
        }
        sd["phase_factors"] = pf;
        sd["dimension"]     = s.dimension;
        sectors.append(sd);
    }
    d["sectors"] = sectors;
    return d;
}

py::dict ftlm_results_to_dict(const FTLMResults& res) {
    py::dict d;
    d["ground_state_estimate"] = res.ground_state_estimate;
    d["total_samples"]         = res.total_samples;
    d["energy_error"]          = res.energy_error;
    d["specific_heat_error"]   = res.specific_heat_error;
    d["entropy_error"]         = res.entropy_error;
    d["free_energy_error"]     = res.free_energy_error;
    return d;
}

py::dict thermo_data_to_dict(const ThermodynamicData& t) {
    py::dict d;
    d["temperatures"]  = t.temperatures;
    d["energy"]        = t.energy;
    d["specific_heat"] = t.specific_heat;
    d["entropy"]       = t.entropy;
    d["free_energy"]   = t.free_energy;
    return d;
}

py::dict ed_results_to_dict(const EDResults& r) {
    py::dict d;
    d["eigenvalues"]           = r.eigenvalues;
    d["eigenvectors_computed"] = r.eigenvectors_computed;
    d["eigenvectors_path"]     = r.eigenvectors_path;
    d["thermo_data"]           = thermo_data_to_dict(r.thermo_data);
    d["ftlm_results"]          = ftlm_results_to_dict(r.ftlm_results);
    return d;
}

}  // anonymous namespace

void bind_dispatcher(py::module_& m) {
    // ------------------------------------------------------------------------
    // 1. DiagonalizationMethod enum -- every value the C++ enum carries.
    //    The numeric values are part of the public ABI (see ed_types.h);
    //    pybind11 preserves them, so cross-language IO via int casts works.
    // ------------------------------------------------------------------------
    py::enum_<DiagonalizationMethod>(m, "DiagonalizationMethod", R"pbdoc(
        The 9 solver algorithms the unified orchestrator exposes.

        These tags are consumed by the legacy ``EDParameters``
        parameter-carrier; the orchestrator's own knob is
        ``_core.SolveMethod`` (mapped 1:1 by the Python bridge).
    )pbdoc")
        // Ground-state / spectrum
        .value("LANCZOS",                  DiagonalizationMethod::LANCZOS)
        .value("BLOCK_LANCZOS",            DiagonalizationMethod::BLOCK_LANCZOS)
        .value("KRYLOV_SCHUR",             DiagonalizationMethod::KRYLOV_SCHUR)
        .value("BLOCK_KRYLOV_SCHUR",       DiagonalizationMethod::BLOCK_KRYLOV_SCHUR)
        .value("FULL",                     DiagonalizationMethod::FULL)
        // Thermal
        .value("mTPQ",                     DiagonalizationMethod::mTPQ)
        .value("cTPQ",                     DiagonalizationMethod::cTPQ)
        .value("FTLM",                     DiagonalizationMethod::FTLM)
        .value("LTLM",                     DiagonalizationMethod::LTLM)
        .value("KPM_DOS",                  DiagonalizationMethod::KPM_DOS)
        .value("OFTLM",                    DiagonalizationMethod::OFTLM)
        .export_values();

    // (``HamiltonianFileFormat`` enum was deleted alongside the
    // legacy file-loader entry points; no surviving Python entry
    // consumes it.)

    // ------------------------------------------------------------------------
    // 3. EDParameters -- the legacy parameter bag still used internally by
    //    ``qed.workflow._diag_via_workflows_solve`` to carry knobs the
    //    orchestrator's ``SolveOptions`` / ``ThermalOptions`` don't carry
    //    natively (FTLM krylov_dim, TPQ taylor_order, KPM moments, etc.).
    // ------------------------------------------------------------------------
    py::class_<EDParameters>(m, "EDParameters", R"pbdoc(
        Parameter bag used internally by ``qed.workflow`` to carry the
        legacy knobs across the Python <-> C++ boundary before they get
        translated into ``_core.SolveOptions`` / ``_core.ThermalOptions``.

        Mirrors ``include/ed/core/ed_parameters.h`` 1:1. End users should
        prefer the kwargs-only ``qed.solve(H, ...)`` /
        ``qed.thermal(H, ...)`` / ``qed.spectral(H, ...)`` API, which
        wraps this struct internally.
    )pbdoc")
        .def(py::init<>())
        // General
        .def_readwrite("max_iterations",       &EDParameters::max_iterations)
        .def_readwrite("num_eigenvalues",      &EDParameters::num_eigenvalues)
        .def_readwrite("tolerance",            &EDParameters::tolerance)
        .def_readwrite("compute_eigenvectors", &EDParameters::compute_eigenvectors)
        .def_readwrite("output_dir",           &EDParameters::output_dir)
        // Method-specific
        .def_readwrite("block_size",           &EDParameters::block_size)
        // Thermal common
        .def_readwrite("num_samples",          &EDParameters::num_samples)
        .def_readwrite("temp_min",             &EDParameters::temp_min)
        .def_readwrite("temp_max",             &EDParameters::temp_max)
        .def_readwrite("num_temp_bins",        &EDParameters::num_temp_bins)
        // TPQ
        // Pillar 1 of the "Save and DSSF Upgrades" plan (May 2026):
        // probe-beta list for TPQ state-vector snapshots.
        .def_readwrite("tpq_probe_betas",          &EDParameters::tpq_probe_betas)
        .def_readwrite("tpq_max_steps",            &EDParameters::tpq_max_steps)
        .def_readwrite("tpq_measurement_interval", &EDParameters::tpq_measurement_interval)
        .def_readwrite("tpq_energy_shift",         &EDParameters::tpq_energy_shift)
        .def_readwrite("tpq_fp32",                 &EDParameters::tpq_fp32)
        .def_readwrite("tpq_beta_max",             &EDParameters::tpq_beta_max)
        .def_readwrite("tpq_delta_beta",           &EDParameters::tpq_delta_beta)
        .def_readwrite("tpq_taylor_order",         &EDParameters::tpq_taylor_order)
        .def_readwrite("tpq_continue",             &EDParameters::tpq_continue)
        .def_readwrite("tpq_continue_sample",      &EDParameters::tpq_continue_sample)
        .def_readwrite("tpq_continue_beta",        &EDParameters::tpq_continue_beta)
        .def_readwrite("tpq_target_beta",          &EDParameters::tpq_target_beta)
        .def_readwrite("tpq_num_measure_points",   &EDParameters::tpq_num_measure_points)
        .def_readwrite("tpq_measure_beta_min",     &EDParameters::tpq_measure_beta_min)
        .def_readwrite("tpq_measure_beta_max",     &EDParameters::tpq_measure_beta_max)
        // FTLM
        .def_readwrite("ftlm_krylov_dim",      &EDParameters::ftlm_krylov_dim)
        .def_readwrite("ftlm_full_reorth",     &EDParameters::ftlm_full_reorth)
        .def_readwrite("ftlm_reorth_freq",     &EDParameters::ftlm_reorth_freq)
        .def_readwrite("ftlm_seed",            &EDParameters::ftlm_seed)
        .def_readwrite("ftlm_store_samples",   &EDParameters::ftlm_store_samples)
        .def_readwrite("ftlm_error_bars",      &EDParameters::ftlm_error_bars)
        // LTLM
        .def_readwrite("ltlm_krylov_dim",      &EDParameters::ltlm_krylov_dim)
        .def_readwrite("ltlm_ground_krylov",   &EDParameters::ltlm_ground_krylov)
        .def_readwrite("ltlm_full_reorth",     &EDParameters::ltlm_full_reorth)
        .def_readwrite("ltlm_reorth_freq",     &EDParameters::ltlm_reorth_freq)
        .def_readwrite("ltlm_seed",            &EDParameters::ltlm_seed)
        .def_readwrite("ltlm_store_data",      &EDParameters::ltlm_store_data)
        // KPM-DOS (see include/ed/solvers/kpm_dos.h)
        .def_readwrite("kpm_num_moments",            &EDParameters::kpm_num_moments)
        .def_readwrite("kpm_num_random_vectors",     &EDParameters::kpm_num_random_vectors)
        .def_readwrite("kpm_num_quadrature_nodes",   &EDParameters::kpm_num_quadrature_nodes)
        .def_readwrite("kpm_spectral_bounds_krylov", &EDParameters::kpm_spectral_bounds_krylov)
        .def_readwrite("kpm_spectral_bound_buffer",  &EDParameters::kpm_spectral_bound_buffer)
        .def_readwrite("kpm_use_jackson_kernel",     &EDParameters::kpm_use_jackson_kernel)
        .def_readwrite("kpm_lorentz_lambda",         &EDParameters::kpm_lorentz_lambda)
        .def_readwrite("kpm_full_reorth",            &EDParameters::kpm_full_reorth)
        .def_readwrite("kpm_reorth_freq",            &EDParameters::kpm_reorth_freq)
        .def_readwrite("kpm_seed",                   &EDParameters::kpm_seed)
        // Observables (DSSF window etc.)
        .def_readwrite("omega_min",   &EDParameters::omega_min)
        .def_readwrite("omega_max",   &EDParameters::omega_max)
        .def_readwrite("num_points",  &EDParameters::num_points)
        .def_readwrite("t_end",       &EDParameters::t_end)
        .def_readwrite("dt",          &EDParameters::dt)
        // Lattice
        .def_readwrite("num_sites",        &EDParameters::num_sites)
        .def_readwrite("spin_length",      &EDParameters::spin_length)
        .def_readwrite("sublattice_size",  &EDParameters::sublattice_size)
        .def_readwrite("selected_sectors", &EDParameters::selected_sectors)
        // TPQ observable controls
        .def_readwrite("save_thermal_states",        &EDParameters::save_thermal_states)
        .def_readwrite("compute_spin_correlations",  &EDParameters::compute_spin_correlations)
        // Fixed-Sz
        .def_readwrite("use_fixed_sz",   &EDParameters::use_fixed_sz)
        .def_readwrite("n_up",           &EDParameters::n_up)
        .def_readwrite("full_sz_split",  &EDParameters::full_sz_split)
        // Orthogonal device / parallelism axes (matches use_fixed_sz):
        //   use_gpu  : route through CUDA backend when WITH_CUDA=ON
        //   use_mpi  : use the MPI backend (distributed-memory)
        .def_readwrite("use_gpu",          &EDParameters::use_gpu)
        .def_readwrite("use_mpi",          &EDParameters::use_mpi)
        // 5th orthogonal axis -- spatial-symmetry projection. Consumed
        // by ``_core.workflows_solve_streaming_symmetry_directory``.
        .def_readwrite("use_symmetry",     &EDParameters::use_symmetry)
        .def_readwrite("translation_only", &EDParameters::translation_only)
        .def("__repr__", [](const EDParameters& p) {
            return "<qed.EDParameters num_eigenvalues=" +
                   std::to_string(p.num_eigenvalues) +
                   " max_iterations=" + std::to_string(p.max_iterations) +
                   " tolerance=" + std::to_string(p.tolerance) +
                   " num_sites=" + std::to_string(p.num_sites) +
                   " use_fixed_sz=" + (p.use_fixed_sz ? "True" : "False") +
                   ">";
        });

    // ------------------------------------------------------------------------
    // 4. EDResults -- read-only result envelope returned by the legacy
    //    Python adapter (``qed.workflow._ed_result_from_*``).
    // ------------------------------------------------------------------------
    py::class_<ThermodynamicData>(m, "ThermodynamicData",
        "Thermodynamic observables on the temperature grid the orchestrator "
        "computed. Populated by FTLM/LTLM/TPQ/KPM_DOS; otherwise empty.")
        .def(py::init<>())
        .def_readwrite("temperatures",  &ThermodynamicData::temperatures)
        .def_readwrite("energy",        &ThermodynamicData::energy)
        .def_readwrite("specific_heat", &ThermodynamicData::specific_heat)
        .def_readwrite("entropy",       &ThermodynamicData::entropy)
        .def_readwrite("free_energy",   &ThermodynamicData::free_energy)
        .def("to_dict", &thermo_data_to_dict);

    py::class_<EDResults>(m, "EDResults", py::dynamic_attr(), R"pbdoc(
        Legacy result envelope returned by the Python adapter
        (``qed.workflow._diag_via_workflows_solve``). Carries the
        ground-state eigenvalues plus (for thermal lanes) a
        ``thermo_data`` block with the recombined temperature scan.

        ``py::dynamic_attr`` is enabled so the Python layer can attach
        optional per-sector diagnostics (``eigenvalues_per_sector``,
        ``sector_tags``) populated by the symmetry-decomposed solve /
        full-spectrum paths without growing the C++ POD.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("eigenvalues",           &EDResults::eigenvalues)
        .def_readwrite("eigenvectors_computed", &EDResults::eigenvectors_computed)
        .def_readwrite("eigenvectors_path",     &EDResults::eigenvectors_path)
        .def_readwrite("thermo_data",           &EDResults::thermo_data)
        .def("to_dict", &ed_results_to_dict);

    // ------------------------------------------------------------------------
    // 5. Symmetry projection: attach the Operator.symmetry_info setter /
    //    getter so callers can wire ``qed.symmetry.group_from_generators(...)``
    //    output straight onto an in-process Operator without going through
    //    the on-disk automorphism_results/ JSON detour.
    //
    //    The Operator and FixedSzOperator pybind11 classes were already
    //    declared in qed_bindings.cpp before bind_dispatcher() runs,
    //    so we attach the new methods by looking them up via attr() and
    //    binding cpp_functions as instance methods. SymmetryGroupInfo lives
    //    on Operator (FixedSzOperator inherits it), so a single Operator-
    //    typed setter handles both.
    // ------------------------------------------------------------------------
    {
        py::object op_cls    = m.attr("Operator");
        py::object fixed_cls = m.attr("FixedSzOperator");

        const char* setter_doc = R"pbdoc(
            Attach a symmetry group to the operator from a dict.

            The dict schema matches what
            ``qed.symmetry.group_from_generators(...)`` and
            ``qed.symmetry.translation_group_1d(...)`` return:
            ``num_generators`` (int), ``generator_orders`` (list[int]),
            ``generators`` (list[list[int]]), ``max_clique``
            (list[list[int]]), ``power_representation`` (list[list[int]]),
            and ``sectors`` (list of dicts with ``sector_id``,
            ``quantum_numbers``, ``phase_factors``).
        )pbdoc";

        const char* getter_doc =
            "Round-trip the operator's symmetry group as a Python dict. "
            "The dict is bit-compatible with the schema produced by "
            "``qed.symmetry.group_from_generators(...)``. Returns "
            "an empty-defaults dict if no symmetry info has been set.";

        op_cls.attr("set_symmetry_info_from_dict") = py::cpp_function(
            [](Operator& self, const py::dict& d) {
                self.symmetry_info = dict_to_symmetry_info(d);
            },
            py::is_method(op_cls), py::arg("info"), setter_doc);
        op_cls.attr("get_symmetry_info_as_dict") = py::cpp_function(
            [](const Operator& self) {
                return symmetry_info_to_dict(self.symmetry_info);
            },
            py::is_method(op_cls), getter_doc);
        fixed_cls.attr("set_symmetry_info_from_dict") = py::cpp_function(
            [](Operator& self, const py::dict& d) {
                self.symmetry_info = dict_to_symmetry_info(d);
            },
            py::is_method(fixed_cls), py::arg("info"), setter_doc);
        fixed_cls.attr("get_symmetry_info_as_dict") = py::cpp_function(
            [](const Operator& self) {
                return symmetry_info_to_dict(self.symmetry_info);
            },
            py::is_method(fixed_cls), getter_doc);
    }

    // ------------------------------------------------------------------------
    // 6. Build introspection. Lets callers gate GPU / MPI codepaths in
    //    pure Python without trying a method and catching the warning.
    // ------------------------------------------------------------------------
    m.def("has_cuda_build", []() {
#ifdef WITH_CUDA
        return true;
#else
        return false;
#endif
    },
        "True iff this build of ``qed._core`` was compiled with "
        "``WITH_CUDA=ON``. When False, setting ``device='gpu'`` on "
        "``qed.solve(...)`` falls back to CPU with a runtime warning.");

    m.def("has_mpi_build", []() {
#ifdef WITH_MPI
        return true;
#else
        return false;
#endif
    },
        "True iff this build was compiled with ``WITH_MPI=ON``. The "
        "single-process ``qed._core`` does not call MPI directly; "
        "use the standalone ``mpiexec ed_distributed_main ...`` binary "
        "(see ``qed.mpi.run_distributed(...)``) to drive the MPI "
        "solvers.");

    // (capability-aware execution planner removed: sensible defaults +
    //  env-override leaf hooks; no probe_system / plan_execution surface)
}
