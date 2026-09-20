// =============================================================================
// src/cli/workflows/wf_kpm.cpp
//
// `compute_kpm_thermodynamics_workflow` (audit item #3): operator-free
// thermodynamics from the Chebyshev-expanded density of states, plus the
// WITH_CUDA `device_matvec_from` adapter whose only call site is the KPM GPU
// lane.
//
// Moved verbatim out of src/cli/workflows.cpp (WP14). See
// workflows_internal.h for the file map.
// =============================================================================

#include "workflows_internal.h"

#ifdef WITH_CUDA
namespace {
// Operator-collapse Phase 2b (Jun 2026): wrap a unified host operator's device
// matvec (CudaMatVecBackend via Operator/FixedSzOperator::bind_cuda) as the
// cuDoubleComplex-typed callable the GPU FTLM / KPM drivers expect.
// std::complex<double> and cuDoubleComplex are layout-compatible, so the
// reinterpret_cast is well-defined. Replaces the legacy GPUOperator mirror +
// convertOperatorToGPU round-trip for the CLI DSSF / KPM GPU lanes.
inline std::function<void(const cuDoubleComplex*, cuDoubleComplex*, int)>
device_matvec_from(ed::LinearOperator& op) {
    auto fn = op.bind_cuda();
    return [fn = std::move(fn)](const cuDoubleComplex* in,
                                cuDoubleComplex* out, int n) {
        fn(reinterpret_cast<const ed::Complex*>(in),
           reinterpret_cast<ed::Complex*>(out),
           static_cast<std::size_t>(n));
    };
}
}  // namespace
#endif

// ============================================================================
// KPM thermodynamics workflow (audit item #3)
//
// Operator-free thermodynamics from the Chebyshev-expanded density of
// states: Z(beta), E(beta), C(beta), S(beta), F(beta) (and an optional
// reconstructed DOS) for the configured Hamiltonian. Persists results
// under `/kpm_thermodynamics/...` in the run's HDF5 file.
//
// Driven by the existing `ed::kpm_dos::compute_kpm_dos` solver. The
// dispatch wiring lives in `src/cli/dssf_engine.cpp`; this is the body.
// ============================================================================
void compute_kpm_thermodynamics_workflow(const EDConfig& config) {
    // Plain locals (not a structured binding): C++17 forbids capturing
    // structured bindings in lambdas (clang enforces; gcc extension).
    int rank = 0, size = 1;
    std::tie(rank, size) = get_mpi_rank_size_safe();
    (void)size;

    if (rank == 0) {
        std::cout << "\n==========================================\n";
        std::cout << "KPM Thermodynamics (Chebyshev DOS)\n";
        std::cout << "==========================================\n";
    }

    auto wh = build_workflow_hamiltonian(config, rank, /*verbose_label=*/nullptr);
    const bool use_fixed_sz = wh.use_fixed_sz;
    const int64_t n_up      = wh.n_up;
    const uint64_t N        = wh.N;
    auto& ham_full          = wh.ham_full;
    auto& ham_fs            = wh.ham_fs;
    Operator& ham           = wh.ham_ref();
    auto& H_func            = wh.H_func;
    (void)ham_full; (void)ham_fs;

    // Temperature grid: prefer the dynamical-block grid when a sweep is
    // configured (num_temp_bins > 1); else fall back to the thermal block
    // (which the standard thermodynamics workflow uses) so the user gets
    // something sensible without having to repopulate dssf-specific knobs.
    std::vector<double> temperatures;
    {
        double tmin = config.dynamical.temp_min;
        double tmax = config.dynamical.temp_max;
        uint64_t nT = config.dynamical.num_temp_bins;
        if (nT <= 1) {
            tmin = config.thermal.temp_min;
            tmax = config.thermal.temp_max;
            nT   = std::max<uint64_t>(1, config.thermal.num_temp_bins);
        }
        temperatures.resize(nT);
        if (nT == 1) {
            temperatures[0] = tmin;
        } else {
            const double log_tmin = std::log(tmin);
            const double log_tmax = std::log(tmax);
            const double dl = (log_tmax - log_tmin) / (nT - 1);
            for (uint64_t i = 0; i < nT; ++i) {
                temperatures[i] = std::exp(log_tmin + i * dl);
            }
        }
    }

    std::vector<double> betas(temperatures.size());
    for (size_t i = 0; i < temperatures.size(); ++i) {
        if (!(temperatures[i] > 0.0)) {
            throw std::invalid_argument(
                "compute_kpm_thermodynamics_workflow: temperature must be > 0");
        }
        betas[i] = 1.0 / temperatures[i];
    }

    ed::kpm_dos::KPMDOSParameters kpm_params;
    if (config.dynamical.krylov_dim > 0) {
        // Reuse the user's Krylov budget for the spectral-bound Lanczos.
        kpm_params.spectral_bounds_krylov =
            static_cast<int>(std::min<uint64_t>(
                config.dynamical.krylov_dim, 500));
    }
    if (config.dynamical.num_random_states > 0) {
        kpm_params.num_random_vectors =
            static_cast<int>(config.dynamical.num_random_states);
    }
    if (config.dynamical.random_seed != 0) {
        kpm_params.random_seed =
            static_cast<std::uint64_t>(config.dynamical.random_seed);
    }
    // Tunables not yet exposed through ed_config.cpp — pick them up from env
    // so production scripts can sweep without rebuilding.  Defaults are
    // documented next to KPMDOSParameters in include/ed/solvers/kpm_dos.h.
    if (const char* env_M = ed::env::raw("ED_KPM_NUM_MOMENTS")) {
        const int v = std::atoi(env_M);
        if (v >= 4) kpm_params.num_moments = v;
    }
    if (const char* env_Nq = ed::env::raw("ED_KPM_NUM_QUAD")) {
        const int v = std::atoi(env_Nq);
        if (v > 0) kpm_params.num_quadrature_nodes = v;
    }
    if (const char* env_buf = ed::env::raw("ED_KPM_BOUND_BUFFER")) {
        const double v = std::atof(env_buf);
        if (v > 0.0) kpm_params.spectral_bound_buffer = v;
    }
    if (const char* env_kern = ed::env::raw("ED_KPM_KERNEL")) {
        const std::string s(env_kern);
        if (s == "lorentz" || s == "Lorentz" || s == "LORENTZ") {
            kpm_params.use_jackson_kernel = false;
        }
    }
    if (const char* env_lambda = ed::env::raw("ED_KPM_LORENTZ_LAMBDA")) {
        const double v = std::atof(env_lambda);
        if (v > 0.0) kpm_params.lorentz_lambda = v;
    }

    if (rank == 0) {
        std::cout << "  dim          = " << N << "\n";
        std::cout << "  num_moments  = " << kpm_params.num_moments << "\n";
        std::cout << "  num_samples  = " << kpm_params.num_random_vectors << "\n";
        std::cout << "  temperatures = " << temperatures.size() << "\n";
        std::cout << "  jackson      = "
                  << (kpm_params.use_jackson_kernel ? "yes" : "no") << "\n";
    }

    create_directory_mpi_safe(config.workflow.output_dir);

    // Single-process driver: no MPI distribution at this point.  GPU dispatch
    // is per rank when --use-gpu is enabled (and CUDA is built); otherwise we
    // fall back to the CPU operator-free implementation.
    if (rank != 0) {
        return;
    }

    ed::kpm_dos::KPMDOSResult kpm;
#ifdef WITH_CUDA
    const bool kpm_use_gpu =
        config.dynamical.use_gpu || config.system.use_gpu;
    if (kpm_use_gpu) {
        // Operator-collapse Phase 2b (Jun 2026): drive the GPU KPM
        // Chebyshev/Hutchinson loop straight off the unified host operator's
        // device matvec (CudaMatVecBackend via Operator/FixedSzOperator::
        // bind_cuda) -- no bespoke GPUOperator mirror / convertOperatorToGPU
        // round-trip. `ham` is a FixedSzOperator in the fixed-Sz sector and a
        // full-Hilbert Operator otherwise; bind_cuda() dispatches virtually.
        std::cout << "  backend      = GPU (SOTA gather matvec"
                  << (use_fixed_sz ? ", fixed-Sz" : "") << ")\n";
        kpm = ed::kpm_dos::compute_kpm_dos_gpu_with_matvec(
            device_matvec_from(ham), N, betas, /*dos_grid=*/{}, kpm_params);
    } else {
        std::cout << "  backend      = CPU (operator-free)\n";
        kpm = ed::kpm_dos::compute_kpm_dos(
            H_func, N, betas, /*dos_grid=*/{}, kpm_params);
    }
#else
    std::cout << "  backend      = CPU (operator-free; no CUDA build)\n";
    kpm = ed::kpm_dos::compute_kpm_dos(
        H_func, N, betas, /*dos_grid=*/{}, kpm_params);
#endif

    // Persist under the standard thermodynamics group so existing readers
    // (Python `qed.workflow`, the analysis scripts) just work, and stamp
    // a small KPM_THERMODYNAMICS provenance under /kpm_thermodynamics.
    const std::string h5_file =
        HDF5IO::createOrOpenFile(config.workflow.output_dir);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "energy",        kpm.energy);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "specific_heat", kpm.specific_heat);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "entropy",       kpm.entropy);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "free_energy",   kpm.free_energy);
    HDF5IO::saveThermodynamics(h5_file, temperatures, "partition_function",
                               kpm.partition_function);

    try {
        H5::H5File file(h5_file, H5F_ACC_RDWR);
        if (!file.nameExists("/kpm_thermodynamics")) {
            file.createGroup("/kpm_thermodynamics");
        }
        H5::Group g = file.openGroup("/kpm_thermodynamics");
        const auto write_dbl = [&](const char* name, double v) {
            H5::DataSpace s(H5S_SCALAR);
            if (g.attrExists(name)) g.removeAttr(name);
            auto a = g.createAttribute(name, H5::PredType::NATIVE_DOUBLE, s);
            a.write(H5::PredType::NATIVE_DOUBLE, &v);
        };
        const auto write_int = [&](const char* name, int v) {
            H5::DataSpace s(H5S_SCALAR);
            if (g.attrExists(name)) g.removeAttr(name);
            auto a = g.createAttribute(name, H5::PredType::NATIVE_INT, s);
            a.write(H5::PredType::NATIVE_INT, &v);
        };
        write_int("num_moments_used",        kpm.num_moments_used);
        write_int("num_random_vectors_used", kpm.num_random_vectors_used);
        write_dbl("kpm_a",                   kpm.kpm_a);
        write_dbl("kpm_b",                   kpm.kpm_b);
        write_dbl("e_min_estimate",          kpm.e_min_estimate);
        write_dbl("e_max_estimate",          kpm.e_max_estimate);
        write_dbl("energy_shift_used",       kpm.energy_shift_used);

        if (!kpm.moments_weighted.empty()) {
            const std::string ds = "/kpm_thermodynamics/moments_weighted";
            if (file.nameExists(ds)) file.unlink(ds);
            hsize_t dims[1] = { kpm.moments_weighted.size() };
            H5::DataSpace ms(1, dims);
            auto d = file.createDataSet(
                ds, H5::PredType::NATIVE_DOUBLE, ms);
            d.write(kpm.moments_weighted.data(), H5::PredType::NATIVE_DOUBLE);
        }
    } catch (const H5::Exception& e) {
        std::cerr << "  Warning: KPM provenance write failed: "
                  << e.getDetailMsg() << "\n";
    }

    std::cout << "\nKPM thermodynamics complete.\n";
    std::cout << "Results: " << config.workflow.output_dir
              << "/ed_results.h5  (groups /thermodynamics, /kpm_thermodynamics)\n";
}
