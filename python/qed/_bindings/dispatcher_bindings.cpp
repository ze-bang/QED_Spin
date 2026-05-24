// =============================================================================
// python/qed/_bindings/dispatcher_bindings.cpp
//
// Phase 5 Python interface expansion (Apr 2026): bind the high-level C++
// dispatcher and the directory- / streaming-symmetry drivers so the entire
// Krylov + dense + thermal stack is reachable from a
// single Python entry point. See the header for the full feature list.
// =============================================================================

#include "dispatcher_bindings.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <ed/core/ed_wrapper.h>            // EDResults / EDParameters / legacy free functions
#include <ed/core/ed_wrapper_streaming.h>  // streaming-symmetry kernel
#include <ed/core/ed_method_traits.h>      // canonicalize_method_and_flags
#include <ed/core/ed_parameters.h>
#include <ed/core/ed_types.h>
#include <ed/core/construct_ham.h>

#include <filesystem>

#include <complex>
#include <cstdint>
#include <cstring>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

using Complex = std::complex<double>;

// -----------------------------------------------------------------------------
// Full Unified-Interface Collapse, Wave E4 (May 2026): the
// `exact_diagonalization_*` Python entry points exposed by this file are
// legacy aliases. The canonical surface is now `qed.workflows.solve /
// thermal / spectral` (which delegates to the C++
// `ed::workflows::{solve,thermal,spectral}` orchestrator). The aliases
// here keep working for out-of-tree consumers but emit a
// `DeprecationWarning` to nudge migration.
//
// Internal in-tree callers should NOT use these aliases directly --
// they should use `qed.workflows.*` or the C++ `_core.workflows_*`
// bindings instead.
// -----------------------------------------------------------------------------
inline void emit_deprecation_warning(const char* legacy_name,
                                     const char* replacement) {
    // Use stacklevel = 2 so the warning points at the caller, not at
    // the lambda inside the binding.
    std::string msg = std::string(legacy_name) +
        " is a legacy alias since the Full Unified-Interface Collapse "
        "(May 2026); use " + replacement + " instead. "
        "This alias will be removed in a future release.";
    if (PyErr_WarnEx(PyExc_DeprecationWarning, msg.c_str(), 2) < 0) {
        // PyErr_WarnEx returns -1 if the warning was promoted to an
        // exception (e.g. via `-W error::DeprecationWarning`). The
        // pybind11 trampoline will see PyErr_Occurred() and convert
        // it into a Python exception on return.
        throw py::error_already_set();
    }
}

// -----------------------------------------------------------------------------
// Adapter: build an `apply(in, out, n)` callback closing over an Operator-like
// object. `Op` is either `Operator` or `FixedSzOperator`; both provide the
// same `apply(const Complex*, Complex*, size_t)` signature.
// -----------------------------------------------------------------------------
template <typename Op>
std::function<void(const Complex*, Complex*, int)>
make_matvec(const Op& op) {
    const Op* p = &op;
    return [p](const Complex* in, Complex* out, int n) {
        p->apply(in, out, static_cast<std::size_t>(n));
    };
}

// -----------------------------------------------------------------------------
// Convert a Python dict (matching the schema returned by
// qed.symmetry.group_from_generators / translation_group_1d) into a
// SymmetryGroupInfo. Both endpoints round-trip:
//   d = qed.symmetry.group_from_generators(...)
//   op.set_symmetry_info_from_dict(d)
//   d2 = op.get_symmetry_info_as_dict()
//   assert d2 == d  # up to ordering of generators
//
// The expected keys are:
//   "num_generators"       -> int
//   "generator_orders"     -> list[int]
//   "generators"           -> list[list[int]]
//   "max_clique"           -> list[list[int]]
//   "power_representation" -> list[list[int]]
//   "sectors"              -> list[dict] with keys
//                             "sector_id" (int),
//                             "quantum_numbers" (list[int]),
//                             "phase_factors" (list[complex]).
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

// -----------------------------------------------------------------------------
// FTLM "per-sector" trace data on EDResults.ftlm_results is a vector of small
// structs; pybind11 needs an opaque wrapper or an explicit converter. We just
// re-export the headline fields (eigenvalues / thermo_data) for now and let
// callers that need per-sector internals reach into the C++ struct directly
// via `EDResults.ftlm_eigenvalues_per_sector` / etc. (added below).
// -----------------------------------------------------------------------------

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
        The 9 solver algorithms the dispatcher supports.

        Device (CPU vs GPU) and parallelism (single-process vs MPI) are
        flags on :class:`EDParameters`, not enum values:

            params = EDParameters()
            params.use_gpu = True       # run on GPU when WITH_CUDA=ON
            params.use_mpi = True       # use MPI (distributed-memory)
            params.use_fixed_sz = True  # restrict to one Sz sector

        Setting ``use_gpu`` on a build without CUDA falls back to CPU
        with a runtime warning (see ``has_cuda_build()``).
    )pbdoc")
        // Ground-state / spectrum
        .value("LANCZOS",                  DiagonalizationMethod::LANCZOS)
        .value("BLOCK_LANCZOS",            DiagonalizationMethod::BLOCK_LANCZOS)
        .value("KRYLOV_SCHUR",             DiagonalizationMethod::KRYLOV_SCHUR)
        .value("FULL",                     DiagonalizationMethod::FULL)
        // Thermal
        .value("mTPQ",                     DiagonalizationMethod::mTPQ)
        .value("cTPQ",                     DiagonalizationMethod::cTPQ)
        .value("FTLM",                     DiagonalizationMethod::FTLM)
        .value("LTLM",                     DiagonalizationMethod::LTLM)
        .value("KPM_DOS",                  DiagonalizationMethod::KPM_DOS)
        .export_values();

    // ------------------------------------------------------------------------
    // 2. HamiltonianFileFormat (used by the directory dispatchers).
    // ------------------------------------------------------------------------
    py::enum_<HamiltonianFileFormat>(m, "HamiltonianFileFormat",
        "On-disk format for the Hamiltonian files passed to "
        "`exact_diagonalization_from_directory(...)`. The default "
        "STANDARD value is the mVMC InterAll/Trans tuple that "
        "`qed.input.HamiltonianBuilder.write_files(...)` "
        "emits.")
        .value("STANDARD",      HamiltonianFileFormat::STANDARD)
        .value("SPARSE_MATRIX", HamiltonianFileFormat::SPARSE_MATRIX)
        .value("CUSTOM",        HamiltonianFileFormat::CUSTOM)
        .export_values();

    // ------------------------------------------------------------------------
    // 3. EDParameters -- the legacy parameter bag the CPU dispatcher reads.
    //    Every field of the C++ struct (excluding deprecated accessors) is
    //    exposed read/write so callers can drive every solver.
    // ------------------------------------------------------------------------
    py::class_<EDParameters>(m, "EDParameters", R"pbdoc(
        Parameter bag consumed by ``exact_diagonalization_core(...)``,
        ``exact_diagonalization_from_directory[_symmetrized](...)``, and
        ``exact_diagonalization_streaming_symmetry[_fixed_sz](...)``.

        Construct an instance, set the fields you care about (everything
        defaults to a sensible scalar), then pass it to one of the
        dispatchers. Mirrors ``include/ed/core/ed_parameters.h`` 1:1.

        Example
        -------
        >>> import qed as qed
        >>> params = qed.EDParameters()
        >>> params.num_sites = 6
        >>> params.num_eigenvalues = 4
        >>> params.tolerance = 1e-12
        >>> result = qed.exact_diagonalization_core(
        ...     op, qed.DiagonalizationMethod.KRYLOV_SCHUR, params)
        >>> result.eigenvalues
        [-2.802..., -2.118..., -1.732..., -1.000...]
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
        .def_readwrite("tpq_max_steps",            &EDParameters::tpq_max_steps)
        .def_readwrite("tpq_measurement_interval", &EDParameters::tpq_measurement_interval)
        .def_readwrite("tpq_energy_shift",         &EDParameters::tpq_energy_shift)
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
        // 5th orthogonal axis -- spatial-symmetry projection. Routes
        // through the streaming-symmetry kernel (per-sector, matrix-free,
        // GPU-capable).
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
    // 4. EDResults -- read-only result envelope returned by every dispatcher.
    // ------------------------------------------------------------------------
    py::class_<ThermodynamicData>(m, "ThermodynamicData",
        "Thermodynamic observables on the temperature grid the dispatcher "
        "computed. Populated by FTLM/LTLM/TPQ/KPM_DOS; otherwise empty.")
        .def(py::init<>())
        .def_readwrite("temperatures",  &ThermodynamicData::temperatures)
        .def_readwrite("energy",        &ThermodynamicData::energy)
        .def_readwrite("specific_heat", &ThermodynamicData::specific_heat)
        .def_readwrite("entropy",       &ThermodynamicData::entropy)
        .def_readwrite("free_energy",   &ThermodynamicData::free_energy)
        .def("to_dict", &thermo_data_to_dict);

    py::class_<EDResults>(m, "EDResults", R"pbdoc(
        Result envelope returned by ``exact_diagonalization_core(...)`` and
        the directory / streaming-symmetry dispatchers.

        Attributes
        ----------
        eigenvalues : list[float]
            Computed eigenvalues, ascending. For thermal methods this is
            empty; consult ``thermo_data`` instead.
        eigenvectors_computed : bool
            True iff ``params.compute_eigenvectors`` was set; eigenvectors
            are persisted to ``eigenvectors_path`` (HDF5).
        eigenvectors_path : str
            Directory holding the eigenvector HDF5 file (matches
            ``params.output_dir`` when present).
        thermo_data : ThermodynamicData
            Per-temperature ``E(T)``, ``Cv(T)``, ``S(T)``, ``F(T)`` for
            FTLM/LTLM/TPQ/KPM_DOS runs.

        The fields are read-write so external code (notably the MPI
        Python wrapper that reads ``ed_distributed_main`` HDF5 dumps)
        can construct a fully-populated EDResults from disk and return
        it from ``qed.diag(H, device='mpi', ...)``.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("eigenvalues",           &EDResults::eigenvalues)
        .def_readwrite("eigenvectors_computed", &EDResults::eigenvectors_computed)
        .def_readwrite("eigenvectors_path",     &EDResults::eigenvectors_path)
        .def_readwrite("thermo_data",           &EDResults::thermo_data)
        .def("to_dict", &ed_results_to_dict);

    // ------------------------------------------------------------------------
    // 5. The single high-level dispatcher: exact_diagonalization_core.
    //
    // Two overloads: one accepting an `Operator` (full-Hilbert), one
    // accepting a `FixedSzOperator` (reduced sector). Both build the
    // matrix-free apply callback internally; the GIL is released for the
    // long-running C++ call.
    // ------------------------------------------------------------------------
    auto run_core_op = [](const Operator& op,
                          DiagonalizationMethod method,
                          const EDParameters& params) {
        emit_deprecation_warning("exact_diagonalization_core",
                                 "qed.workflows.solve / qed.workflows.thermal");
        const std::uint64_t dim = (std::uint64_t{1} << op.getNumBits());
        EDResults results;
        auto matvec = make_matvec(op);
        {
            py::gil_scoped_release release;
            results = exact_diagonalization_core(matvec, dim, method, params);
        }
        return results;
    };

    auto run_core_fixed = [](const FixedSzOperator& op,
                             DiagonalizationMethod method,
                             EDParameters params) {
        emit_deprecation_warning("exact_diagonalization_core",
                                 "qed.workflows.solve / qed.workflows.thermal");
        const std::uint64_t dim = op.getFixedSzDim();
        // Mirror the C++ convention: when called with a FixedSzOperator the
        // dispatcher needs use_fixed_sz=True to dispatch the fixed-Sz
        // observables / eigenvector layout. We mutate a local copy so
        // callers can keep reusing their template params object.
        params.use_fixed_sz = true;
        params.fixed_sz_op  = const_cast<FixedSzOperator*>(&op);
        EDResults results;
        auto matvec = make_matvec(op);
        {
            py::gil_scoped_release release;
            results = exact_diagonalization_core(matvec, dim, method, params);
        }
        return results;
    };

    // IMPORTANT: pybind11 dispatches overloads in *registration order*.
    // FixedSzOperator inherits from Operator, so we must register the
    // FixedSzOperator overload FIRST; otherwise pybind11 binds a
    // FixedSzOperator argument to the base ``Operator`` overload, the
    // dim is computed as ``1 << getNumBits()`` (the full Hilbert space),
    // and the solver silently runs on the unprojected operator. This
    // was the matvec-unification audit follow-up bug surfaced by the
    // ``qed.thermal(...)`` end-to-end smoke test where every Sz sector
    // came back with the global ground state energy.
    m.def("exact_diagonalization_core", run_core_fixed,
          py::arg("operator_"),
          py::arg("method") = DiagonalizationMethod::LANCZOS,
          py::arg("params") = EDParameters(),
          "Same as the Operator overload but for a FixedSzOperator. "
          "Internally sets `params.use_fixed_sz = True` and points "
          "`params.fixed_sz_op` at the supplied operator so the "
          "dispatcher reaches the fixed-Sz observable code paths.");

    m.def("exact_diagonalization_core", run_core_op,
          py::arg("operator_"),
          py::arg("method") = DiagonalizationMethod::LANCZOS,
          py::arg("params") = EDParameters(),
          R"pbdoc(
        Run the C++ high-level dispatcher on a matrix-free ``Operator``.

        This single function routes through the same backend table the
        ``./ED`` CLI uses, so you get one entry point for every
        supported algorithm: the Krylov family (LANCZOS, BLOCK_LANCZOS,
        KRYLOV_SCHUR), the dense LAPACK back-end (FULL), and every
        thermal solver (FTLM, LTLM, mTPQ, cTPQ, KPM_DOS).

        GPU execution is requested by setting ``params.use_gpu = True``
        on a CUDA-enabled build. Streaming-symmetry / per-sector GPU
        dispatch uses
        :func:`exact_diagonalization_streaming_symmetry` or the
        directory dispatcher.

        Parameters
        ----------
        operator_ : Operator
            Matrix-free Hamiltonian.
        method : DiagonalizationMethod
            Backend selector. Defaults to ``LANCZOS``.
        params : EDParameters
            Parameter bag. Use the defaults plus the field(s) you care
            about; ``num_eigenvalues`` is the most common knob.

        Returns
        -------
        EDResults
            Eigenvalues, eigenvectors_path, and (for thermal methods)
            ``thermo_data``.
    )pbdoc");

    // ------------------------------------------------------------------------
    // 6. Symmetry projection: attach the Operator.symmetry_info setter /
    //    getter so callers can wire `qed.symmetry.group_from_generators(...)`
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

            Once set the operator carries the symmetry group; callers
            then run ``exact_diagonalization_streaming_symmetry(...)`` on
            the corresponding directory of dat files (which is where
            the C++ engine actually consumes the symmetry data).

            Example
            -------
            >>> import qed as qed
            >>> N = 6
            >>> g = qed.symmetry.translation(N, 1)
            >>> info = qed.symmetry.group_from_generators(N, [g])
            >>> op = qed.Operator(N)
            >>> op.set_symmetry_info_from_dict(info)
            >>> op.get_symmetry_info_as_dict()["num_generators"]
            1
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
    // 7. Streaming symmetry: directory-driven wrapper around
    //    exact_diagonalization_streaming_symmetry[_fixed_sz]. This is the
    //    canonical Python entry point for symmetry-projected ED with
    //    optional GPU per-sector dispatch (set ``params.use_gpu``).
    //
    //    The C++ functions take a directory containing InterAll.dat /
    //    Trans.dat / automorphism_results/, run the streaming-symmetry
    //    pipeline (orbit basis on the fly, per-sector matrix-free apply,
    //    block-by-block diagonalization), and return aggregated EDResults.
    // ------------------------------------------------------------------------
    m.def("exact_diagonalization_streaming_symmetry",
          [](const std::string& directory,
             DiagonalizationMethod method,
             const EDParameters& params,
             const std::string& interaction_filename,
             const std::string& single_site_filename,
             const std::string& basis_cache_dir,
             bool precompute_basis_only) {
              emit_deprecation_warning(
                  "exact_diagonalization_streaming_symmetry",
                  "qed.workflows.solve with an OperatorSpec carrying "
                  "streaming_symmetry=true (per-sector iteration via "
                  "StreamingSymmetryOperator::sector(k))");
              EDResults res;
              {
                  py::gil_scoped_release release;
                  res = exact_diagonalization_streaming_symmetry(
                      directory, method, params,
                      interaction_filename, single_site_filename,
                      basis_cache_dir, precompute_basis_only);
              }
              return res;
          },
          py::arg("directory"),
          py::arg("method") = DiagonalizationMethod::LANCZOS,
          py::arg("params") = EDParameters(),
          py::arg("interaction_filename") = "InterAll.dat",
          py::arg("single_site_filename") = "Trans.dat",
          py::arg("basis_cache_dir") = "",
          py::arg("precompute_basis_only") = false,
          R"pbdoc(
        Symmetry-projected exact diagonalization, streaming variant.

        Reads ``InterAll.dat`` / ``Trans.dat`` and the
        ``automorphism_results/`` directory under ``directory``,
        computes orbit representatives on the fly (no disk basis
        materialisation), and dispatches each symmetry sector to the
        chosen ``method``.

        Set ``params.use_gpu = True`` to run each sector on the GPU
        via the streaming-symmetry dispatch in
        ``include/ed/core/ed_wrapper_streaming.h``.

        Parameters
        ----------
        directory : str
            Path containing the Hamiltonian dat files and
            ``automorphism_results/``.
        method : DiagonalizationMethod
            Per-sector solver. Setting ``params.use_gpu = True`` runs
            each sector via the cuSPARSE / per-sector kernels in
            ``gpu_symmetrized_operator.cu``.
        params : EDParameters
            ``num_sites`` / ``spin_length`` MUST be set; everything else
            inherits the dispatcher defaults.
        basis_cache_dir : str, optional
            HDF5 directory to cache orbit basis between runs.
        precompute_basis_only : bool, optional
            If True, materialise + cache the orbit basis and return
            without solving.

        Returns
        -------
        EDResults
            Aggregated eigenvalues across every sector (sorted ascending).
    )pbdoc");

    m.def("exact_diagonalization_streaming_symmetry_fixed_sz",
          [](const std::string& directory,
             std::int64_t n_up,
             DiagonalizationMethod method,
             const EDParameters& params,
             const std::string& interaction_filename,
             const std::string& single_site_filename,
             const std::string& basis_cache_dir,
             bool precompute_basis_only) {
              emit_deprecation_warning(
                  "exact_diagonalization_streaming_symmetry_fixed_sz",
                  "qed.workflows.solve with an OperatorSpec carrying "
                  "streaming_symmetry=true and fixed_sz=<n_up>");
              EDResults res;
              {
                  py::gil_scoped_release release;
                  res = exact_diagonalization_streaming_symmetry_fixed_sz(
                      directory, n_up, method, params,
                      interaction_filename, single_site_filename,
                      basis_cache_dir, precompute_basis_only);
              }
              return res;
          },
          py::arg("directory"),
          py::arg("n_up"),
          py::arg("method") = DiagonalizationMethod::LANCZOS,
          py::arg("params") = EDParameters(),
          py::arg("interaction_filename") = "InterAll.dat",
          py::arg("single_site_filename") = "Trans.dat",
          py::arg("basis_cache_dir") = "",
          py::arg("precompute_basis_only") = false,
          "Streaming-symmetry ED restricted to the fixed-Sz sector with "
          "``n_up`` up spins. Otherwise behaves exactly like "
          ":func:`exact_diagonalization_streaming_symmetry`. The fixed-Sz "
          "sector cuts both the orbit count and the per-sector dim, so "
          "this is the right entry point for the largest tractable "
          "clusters (32-site spin-1/2 with C2 + translations etc.).");

    // ------------------------------------------------------------------------
    // 8. Directory dispatchers: from_directory and the symmetrized variants.
    //    These mirror the C++ inline wrappers in ed_wrapper.h.
    // ------------------------------------------------------------------------
    m.def("exact_diagonalization_from_directory",
          [](const std::string& directory,
             DiagonalizationMethod method,
             const EDParameters& params,
             HamiltonianFileFormat format,
             const std::string& interaction_filename,
             const std::string& single_site_filename,
             const std::string& counterterm_filename,
             const std::string& three_body_filename) {
              emit_deprecation_warning(
                  "exact_diagonalization_from_directory",
                  "qed.workflows.solve / qed.workflows.thermal with an "
                  "OperatorSpec carrying source=DirectoryPath(...)");
              EDResults res;
              {
                  py::gil_scoped_release release;
                  // Full Unified-Interface Collapse, Wave F-partial
                  // (May 2026): the legacy `<ed/core/dispatch.h>` is
                  // gone. The routing it used to do --
                  // canonicalize_method_and_flags + auto-promote
                  // use_symmetry when `automorphism_results/` is
                  // present + route to streaming when use_symmetry is
                  // set -- is inlined here so the deprecation alias
                  // keeps the same observable behaviour for
                  // out-of-tree callers.
                  EDParameters resolved = params;
                  {
                      const auto canon = ed::canonicalize_method_and_flags(
                          method,
                          resolved.use_fixed_sz, resolved.use_gpu, resolved.use_mpi);
                      method                = canon.method;
                      resolved.use_fixed_sz = canon.use_fixed_sz;
                      resolved.use_gpu      = canon.use_gpu;
                      resolved.use_mpi      = canon.use_mpi;
                  }
                  if (!resolved.use_symmetry) {
                      namespace fs = std::filesystem;
                      const fs::path ar =
                          fs::path(directory) / "automorphism_results";
                      if (fs::exists(ar) && fs::is_directory(ar)) {
                          for (const char* name : {
                                  "automorphisms.json",
                                  "max_clique.json",
                                  "sector_metadata.json",
                                  "minimal_generators.json",
                                  "sectors.json",
                                  "generators.json"}) {
                              if (fs::exists(ar / name)) {
                                  resolved.use_symmetry = true;
                                  break;
                              }
                          }
                      }
                  }
                  if (!resolved.use_symmetry) {
                      res = ::exact_diagonalization_from_directory(
                          directory, method, resolved, format,
                          interaction_filename, single_site_filename,
                          counterterm_filename, three_body_filename);
                  } else if (resolved.use_fixed_sz) {
                      const std::int64_t n_up = (resolved.n_up >= 0)
                          ? resolved.n_up
                          : static_cast<std::int64_t>(resolved.num_sites / 2);
                      res = ::exact_diagonalization_streaming_symmetry_fixed_sz(
                          directory, n_up, method, resolved,
                          interaction_filename, single_site_filename,
                          resolved.basis_cache_dir,
                          resolved.precompute_basis_only);
                  } else {
                      res = ::exact_diagonalization_streaming_symmetry(
                          directory, method, resolved,
                          interaction_filename, single_site_filename,
                          resolved.basis_cache_dir,
                          resolved.precompute_basis_only);
                  }
              }
              return res;
          },
          py::arg("directory"),
          py::arg("method") = DiagonalizationMethod::LANCZOS,
          py::arg("params") = EDParameters(),
          py::arg("format") = HamiltonianFileFormat::STANDARD,
          py::arg("interaction_filename") = "InterAll.dat",
          py::arg("single_site_filename") = "Trans.dat",
          py::arg("counterterm_filename") = "CounterTerm.dat",
          py::arg("three_body_filename") = "ThreeBodyG.dat",
          R"pbdoc(
        Run ED on a directory of Hamiltonian files (Phase 7.1: 5-axis dispatcher).

        Reads ``InterAll.dat`` / ``Trans.dat`` (plus optional
        ``CounterTerm.dat`` / ``ThreeBodyG.dat``) and dispatches on
        the orthogonal axes carried by ``params``:

        ====================  ================================================
        Flag                  Effect
        ====================  ================================================
        ``use_fixed_sz``      restrict to the ``n_up`` Sz-sector
        ``use_gpu``           use GPU kernels (requires WITH_CUDA=ON)
        ``use_mpi``           use MPI parallelism (distributed-memory)
        ``use_symmetry``      project onto symmetry-adapted basis (streaming)
        ====================  ================================================

        Setting ``params.use_symmetry = True`` is the canonical way to
        request symmetry-projected ED -- it routes internally through
        :func:`exact_diagonalization_streaming_symmetry` (or its
        fixed-Sz cousin), which is the only symmetry kernel that
        supports GPU per-sector dispatch and avoids materialising the
        orbit basis on disk.
    )pbdoc");

    // Phase 9 cleanup: the explicit-block ``*_symmetrized`` entry points
    // (``exact_diagonalization_from_directory_symmetrized`` and
    // ``exact_diagonalization_fixed_sz_symmetrized``) were removed.
    // They were already ``[[deprecated]]`` in Phase 7.1 -- they
    // materialised block matrices on disk, were strictly slower than
    // the streaming kernel, and had no GPU support. Anyone hitting
    // ``AttributeError`` on those names should switch to
    // ``exact_diagonalization_from_directory(...)`` with
    // ``params.use_symmetry = True`` (and optionally
    // ``params.use_fixed_sz = True`` + ``params.n_up = ...``), which is
    // the canonical 5-axis dispatcher and is faster on every problem
    // size we have benchmarked.

    // ------------------------------------------------------------------------
    // 9. Build introspection. Lets callers gate GPU / MPI codepaths in
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
        "``WITH_CUDA=ON``. When False, setting ``params.use_gpu = True`` "
        "falls back to the CPU code path with a runtime warning.");

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

}
