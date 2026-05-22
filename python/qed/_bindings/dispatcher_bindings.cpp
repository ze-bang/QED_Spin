// =============================================================================
// python/qed/_bindings/dispatcher_bindings.cpp
//
// Phase 5 Python interface expansion (Apr 2026): bind the high-level C++
// dispatcher and the directory- / streaming-symmetry drivers so the entire
// CPU iterative + dense + thermal + ARPACK + TPQ stack is reachable from a
// single Python entry point. See the header for the full feature list.
// =============================================================================

#include "dispatcher_bindings.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>

#include <ed/core/ed_wrapper.h>            // EDResults / EDParameters / legacy free functions
#include <ed/core/ed_wrapper_streaming.h>  // streaming-symmetry kernel (transitive via dispatch.h)
#include <ed/core/dispatch.h>              // Phase 6: ed::exact_diagonalization(...) -- canonical entry
#include <ed/core/ed_parameters.h>
#include <ed/core/ed_types.h>
#include <ed/core/construct_ham.h>

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
    // Phase 7: pybind11's `.value(...)` helper takes the enum value as a
    // template argument, which trips the deprecation warning even though
    // we *want* to keep the legacy bindings live for backwards compatibility.
    // Suppress the diagnostic for this single block.
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    py::enum_<DiagonalizationMethod>(m, "DiagonalizationMethod", R"pbdoc(
        Every solver the C++ dispatcher knows about.

        The canonical surface is **algorithmic_solver** only.  Device
        (CPU vs GPU) and parallelism (single-process vs MPI) are now
        flags on :class:`EDParameters`:

            params = EDParameters()
            params.use_gpu = True   # canonical replacement for LANCZOS_GPU
            params.use_mpi = True   # canonical replacement for mTPQ_MPI
            params.use_fixed_sz = True  # always the right way to pick fixed-Sz

        The legacy ``_GPU`` / ``_CUDA`` / ``_MPI`` / ``_FIXED_SZ`` enum
        values are still recognised (the dispatcher canonicalizes them
        on entry via ``ed::canonicalize_method_and_flags()``) but new
        code is encouraged to use the base solver + flag form.

        GPU values silently fall back to CPU when the build does not
        have ``WITH_CUDA=ON`` (see ``has_cuda_build()``).
    )pbdoc")
        // CPU iterative
        .value("LANCZOS",                  DiagonalizationMethod::LANCZOS)
        .value("LANCZOS_SELECTIVE",        DiagonalizationMethod::LANCZOS_SELECTIVE)
        .value("LANCZOS_NO_ORTHO",         DiagonalizationMethod::LANCZOS_NO_ORTHO)
        .value("BLOCK_LANCZOS",            DiagonalizationMethod::BLOCK_LANCZOS)
        .value("CHEBYSHEV_FILTERED",       DiagonalizationMethod::CHEBYSHEV_FILTERED)
        .value("SHIFT_INVERT",             DiagonalizationMethod::SHIFT_INVERT)
        .value("SHIFT_INVERT_ROBUST",      DiagonalizationMethod::SHIFT_INVERT_ROBUST)
        .value("DAVIDSON",                 DiagonalizationMethod::DAVIDSON)
        .value("BICG",                     DiagonalizationMethod::BICG)
        .value("LOBPCG",                   DiagonalizationMethod::LOBPCG)
        .value("KRYLOV_SCHUR",             DiagonalizationMethod::KRYLOV_SCHUR)
        .value("BLOCK_KRYLOV_SCHUR",       DiagonalizationMethod::BLOCK_KRYLOV_SCHUR)
        .value("IMPLICIT_RESTART_LANCZOS", DiagonalizationMethod::IMPLICIT_RESTART_LANCZOS)
        .value("THICK_RESTART_LANCZOS",    DiagonalizationMethod::THICK_RESTART_LANCZOS)
        // Dense / distributed dense
        .value("FULL",                     DiagonalizationMethod::FULL)
        .value("OSS",                      DiagonalizationMethod::OSS)
        .value("SCALAPACK",                DiagonalizationMethod::SCALAPACK)
        .value("SCALAPACK_MIXED",          DiagonalizationMethod::SCALAPACK_MIXED)
        // Thermal
        .value("mTPQ",                     DiagonalizationMethod::mTPQ)
        .value("mTPQ_MPI",                 DiagonalizationMethod::mTPQ_MPI)
        .value("cTPQ",                     DiagonalizationMethod::cTPQ)
        .value("mTPQ_CUDA",                DiagonalizationMethod::mTPQ_CUDA)
        .value("FTLM",                     DiagonalizationMethod::FTLM)
        .value("LTLM",                     DiagonalizationMethod::LTLM)
        .value("HYBRID",                   DiagonalizationMethod::HYBRID)
        .value("KPM_DOS",                  DiagonalizationMethod::KPM_DOS)
        // ARPACK
        .value("ARPACK_SM",                DiagonalizationMethod::ARPACK_SM)
        .value("ARPACK_LM",                DiagonalizationMethod::ARPACK_LM)
        .value("ARPACK_SHIFT_INVERT",      DiagonalizationMethod::ARPACK_SHIFT_INVERT)
        .value("ARPACK_ADVANCED",          DiagonalizationMethod::ARPACK_ADVANCED)
        // GPU (silently fall back to CPU when WITH_CUDA=OFF; see has_cuda_build)
        .value("LANCZOS_GPU",              DiagonalizationMethod::LANCZOS_GPU)
        .value("BLOCK_LANCZOS_GPU",        DiagonalizationMethod::BLOCK_LANCZOS_GPU)
        .value("DAVIDSON_GPU",             DiagonalizationMethod::DAVIDSON_GPU)
        .value("LOBPCG_GPU",               DiagonalizationMethod::LOBPCG_GPU)
        .value("KRYLOV_SCHUR_GPU",         DiagonalizationMethod::KRYLOV_SCHUR_GPU)
        .value("BLOCK_KRYLOV_SCHUR_GPU",   DiagonalizationMethod::BLOCK_KRYLOV_SCHUR_GPU)
        .value("mTPQ_GPU",                 DiagonalizationMethod::mTPQ_GPU)
        .value("cTPQ_GPU",                 DiagonalizationMethod::cTPQ_GPU)
        .value("FTLM_GPU",                 DiagonalizationMethod::FTLM_GPU)
        .value("FULL_GPU",                 DiagonalizationMethod::FULL_GPU)
        // Deprecated combined GPU + FIXED_SZ variants (Phase 7: use base
        // method + use_gpu=true + use_fixed_sz=true).
        .value("LANCZOS_GPU_FIXED_SZ",       DiagonalizationMethod::LANCZOS_GPU_FIXED_SZ)
        .value("BLOCK_LANCZOS_GPU_FIXED_SZ", DiagonalizationMethod::BLOCK_LANCZOS_GPU_FIXED_SZ)
        .value("FTLM_GPU_FIXED_SZ",          DiagonalizationMethod::FTLM_GPU_FIXED_SZ)
        .export_values();
    #pragma GCC diagnostic pop  // legacy _GPU / _CUDA / _MPI / _FIXED_SZ enum bindings

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
        .def_readwrite("shift",                &EDParameters::shift)
        .def_readwrite("block_size",           &EDParameters::block_size)
        .def_readwrite("max_subspace",         &EDParameters::max_subspace)
        .def_readwrite("target_lower",         &EDParameters::target_lower)
        .def_readwrite("target_upper",         &EDParameters::target_upper)
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
        // Hybrid
        .def_readwrite("hybrid_crossover",      &EDParameters::hybrid_crossover)
        .def_readwrite("hybrid_auto_crossover", &EDParameters::hybrid_auto_crossover)
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
        // Phase 7: orthogonal device / parallelism axes. Setting these is
        // equivalent to picking the deprecated `_GPU` / `_MPI` enum value.
        // E.g. ``DiagonalizationMethod.LANCZOS`` + ``params.use_gpu = True``
        // is the canonical replacement for ``DiagonalizationMethod.LANCZOS_GPU``.
        // The dispatcher canonicalizes both forms via
        // ``ed::canonicalize_method_and_flags()`` at entry.
        .def_readwrite("use_gpu",        &EDParameters::use_gpu)
        .def_readwrite("use_mpi",        &EDParameters::use_mpi)
        // Phase 7.1: 5th orthogonal axis -- symmetry projection.
        // Setting ``use_symmetry = True`` and calling
        // ``exact_diagonalization_from_directory(...)`` is the canonical
        // (and only non-deprecated) way to request symmetry-projected ED.
        // Internally routes through the streaming symmetry kernel
        // (per-sector, matrix-free, GPU-capable).
        .def_readwrite("use_symmetry",   &EDParameters::use_symmetry)
        // Symmetry sub-options
        .def_readwrite("translation_only", &EDParameters::translation_only)
        // ScaLAPACK
        .def_readwrite("scalapack_nprow",                &EDParameters::scalapack_nprow)
        .def_readwrite("scalapack_npcol",                &EDParameters::scalapack_npcol)
        .def_readwrite("scalapack_block_size",           &EDParameters::scalapack_block_size)
        // Phase 8 #5: when true (default), ``scalapack_block_size`` is
        // overridden by a dim/grid-aware heuristic at solve time. Setting
        // ``scalapack_block_size`` from Python (or via the CLI) implicitly
        // disables auto -- see ``run_diagonalization`` Python wrapper.
        .def_readwrite("scalapack_block_size_auto",      &EDParameters::scalapack_block_size_auto)
        .def_readwrite("scalapack_mixed_precision",      &EDParameters::scalapack_mixed_precision)
        .def_readwrite("scalapack_refinement_tol",       &EDParameters::scalapack_refinement_tol)
        .def_readwrite("scalapack_max_refinement_iter",  &EDParameters::scalapack_max_refinement_iter)
        .def_readwrite("scalapack_verbose",              &EDParameters::scalapack_verbose)
        // ARPACK
        .def_readwrite("arpack_advanced_verbose",         &EDParameters::arpack_advanced_verbose)
        .def_readwrite("arpack_which",                    &EDParameters::arpack_which)
        .def_readwrite("arpack_ncv",                      &EDParameters::arpack_ncv)
        .def_readwrite("arpack_max_restarts",             &EDParameters::arpack_max_restarts)
        .def_readwrite("arpack_ncv_growth",               &EDParameters::arpack_ncv_growth)
        .def_readwrite("arpack_auto_enlarge_ncv",         &EDParameters::arpack_auto_enlarge_ncv)
        .def_readwrite("arpack_two_phase_refine",         &EDParameters::arpack_two_phase_refine)
        .def_readwrite("arpack_relaxed_tol",              &EDParameters::arpack_relaxed_tol)
        .def_readwrite("arpack_shift_invert",             &EDParameters::arpack_shift_invert)
        .def_readwrite("arpack_sigma",                    &EDParameters::arpack_sigma)
        .def_readwrite("arpack_auto_switch_shift_invert", &EDParameters::arpack_auto_switch_shift_invert)
        .def_readwrite("arpack_switch_sigma",             &EDParameters::arpack_switch_sigma)
        .def_readwrite("arpack_adaptive_inner_tol",       &EDParameters::arpack_adaptive_inner_tol)
        .def_readwrite("arpack_inner_tol_factor",         &EDParameters::arpack_inner_tol_factor)
        .def_readwrite("arpack_inner_tol_min",            &EDParameters::arpack_inner_tol_min)
        .def_readwrite("arpack_inner_max_iter",           &EDParameters::arpack_inner_max_iter)
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
        "computed. Populated by FTLM/LTLM/HYBRID/TPQ; otherwise empty.")
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
            FTLM/LTLM/HYBRID/TPQ runs.

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

    m.def("exact_diagonalization_core", run_core_op,
          py::arg("operator_"),
          py::arg("method") = DiagonalizationMethod::LANCZOS,
          py::arg("params") = EDParameters(),
          R"pbdoc(
        Run the C++ high-level dispatcher on a matrix-free ``Operator``.

        This single function routes through the same backend table the
        ``./ED`` CLI uses, so you get one entry point for every CPU
        iterative method (LANCZOS family, BLOCK_LANCZOS, KRYLOV_SCHUR,
        BLOCK_KRYLOV_SCHUR, DAVIDSON, LOBPCG, CHEBYSHEV_FILTERED,
        SHIFT_INVERT[_ROBUST], IRL, TRL, BICG, ARPACK_*), the dense
        backend (FULL, OSS, SCALAPACK / SCALAPACK_MIXED), and every
        thermal solver (FTLM, LTLM, HYBRID, mTPQ, cTPQ).

        For GPU methods (``LANCZOS_GPU`` and friends) and per-sector
        symmetry-projected GPU dispatch use
        :func:`exact_diagonalization_streaming_symmetry` or the
        directory dispatcher: ``exact_diagonalization_core`` itself
        rejects GPU methods because they need a `GPUOperator` handle.

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

    m.def("exact_diagonalization_core", run_core_fixed,
          py::arg("operator_"),
          py::arg("method") = DiagonalizationMethod::LANCZOS,
          py::arg("params") = EDParameters(),
          "Same as the Operator overload but for a FixedSzOperator. "
          "Internally sets `params.use_fixed_sz = True` and points "
          "`params.fixed_sz_op` at the supplied operator so the "
          "dispatcher reaches the fixed-Sz observable code paths.");

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
    //    optional GPU per-sector dispatch (just pass `LANCZOS_GPU` etc.).
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

        Pass any GPU-flavoured method (``LANCZOS_GPU``,
        ``BLOCK_LANCZOS_GPU``, ``DAVIDSON_GPU``, ``LOBPCG_GPU``,
        ``KRYLOV_SCHUR_GPU``, ``BLOCK_KRYLOV_SCHUR_GPU``) to run each
        sector on the GPU via the streaming-symmetry dispatch in
        ``include/ed/core/ed_wrapper_streaming.h``.

        Parameters
        ----------
        directory : str
            Path containing the Hamiltonian dat files and
            ``automorphism_results/``.
        method : DiagonalizationMethod
            Per-sector solver. CPU values run in single-process Python;
            GPU values use cuSPARSE / per-sector kernels in
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
              EDResults res;
              {
                  py::gil_scoped_release release;
                  // Phase 6 (matvec-unification): canonical entry point
                  // ed::exact_diagonalization(...) in <ed/core/dispatch.h>.
                  // Dispatches on the four orthogonal axes recorded in
                  // params (use_symmetry, use_fixed_sz, use_gpu, use_mpi);
                  // see ed/core/dispatch.h for the full contract.
                  res = ed::exact_diagonalization(
                      directory, method, params, format,
                      interaction_filename, single_site_filename,
                      counterterm_filename, three_body_filename);
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
        ``use_gpu``           use GPU kernels (legacy ``LANCZOS_GPU`` etc.)
        ``use_mpi``           use MPI parallelism (ScaLAPACK, mTPQ_MPI, ...)
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
        "``WITH_CUDA=ON``. GPU-flavoured methods (``LANCZOS_GPU`` etc.) "
        "fall back to their CPU equivalents on builds where this is "
        "False.");

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

    m.def("has_scalapack_build", []() {
#ifdef WITH_SCALAPACK
        return true;
#else
        return false;
#endif
    },
        "True iff this build was compiled with both ``WITH_MPI=ON`` and "
        "ScaLAPACK linkage. When False the SCALAPACK / SCALAPACK_MIXED "
        "method values silently fall back to FULL diagonalization.");

    // ------------------------------------------------------------------------
    // 10. Phase 7 introspection: expose the canonicalize() helper so
    //     Python-side tooling and tests can verify that the orthogonal
    //     (SOLVER, use_fixed_sz, use_gpu, use_mpi) decomposition is
    //     correct without having to round-trip through the dispatcher.
    // ------------------------------------------------------------------------
    m.def("canonicalize_method",
          [](DiagonalizationMethod method,
             bool use_fixed_sz,
             bool use_gpu,
             bool use_mpi) {
              const auto canon = ed::canonicalize_method_and_flags(
                  method, use_fixed_sz, use_gpu, use_mpi);
              py::dict d;
              d["method"]       = canon.method;
              d["use_fixed_sz"] = canon.use_fixed_sz;
              d["use_gpu"]      = canon.use_gpu;
              d["use_mpi"]      = canon.use_mpi;
              return d;
          },
          py::arg("method"),
          py::arg("use_fixed_sz") = false,
          py::arg("use_gpu")      = false,
          py::arg("use_mpi")      = false,
          R"pbdoc(
Collapse the deprecated ``_GPU`` / ``_CUDA`` / ``_MPI`` / ``_FIXED_SZ``
enum variants of :class:`DiagonalizationMethod` onto the canonical
``(method, use_fixed_sz, use_gpu, use_mpi)`` tuple.

Returns a dict with keys ``method`` (the base solver), ``use_fixed_sz``,
``use_gpu``, ``use_mpi``. The flags are *OR-ed* with the input values so
calling with ``use_fixed_sz=True`` plus a deprecated ``LANCZOS_GPU_FIXED_SZ``
enum value still returns ``use_fixed_sz=True``.

Idempotent: feeding the output back in produces the same dict.

>>> canon = canonicalize_method(DiagonalizationMethod.LANCZOS_GPU)
>>> canon["method"], canon["use_gpu"]
(DiagonalizationMethod.LANCZOS, True)
>>> canonicalize_method(DiagonalizationMethod.SCALAPACK)["use_mpi"]
True
)pbdoc");
}
