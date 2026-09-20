// =============================================================================
// python/qed/_bindings/workflow/workflow_thermal.cpp
//
// The thermal lane: the plain `workflows_thermal` entry point, the Stage
// 12f SU(2) tower entry (`workflows_thermal_su2_tower`) and the
// streaming-symmetry thermal pair, which share one body lambda.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_thermal(py::module_& m) {
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

    // Stage 12f (SU(2) rollout): ONE spin tower's thermodynamics on the
    // plain fixed-Sz lane. Highest-weight formulation: restrict to the
    // Sz = +S sector (n_up = (N+2S)/2), Lowdin-project every FTLM/mTPQ
    // sample seed onto the spin-S tower, drift-scrub the Krylov walk,
    // and re-normalise the stochastic trace from the sector dim to the
    // tower dim M(N,S). The qed.thermal driver loops towers and
    // recombines with (2S+1) weights via ``combine_thermo_weighted``.
    m.def("workflows_thermal_su2_tower",
          [](Operator& op, ed::workflows::ThermalOptions opts) {
              // Hoisted (audit 2026-07-31): the tower driver lives on the
              // ed::workflows surface now; this is a thin GIL-releasing
              // wrapper.
              py::gil_scoped_release release;
              return ed::workflows::thermal_su2_tower(op, std::move(opts));
          },
          py::arg("op"),
          py::arg("opts"),
          "Thermodynamics of ONE spin-S tower (opts.two_total_spin = 2S) "
          "via Lowdin-projected sampling in the highest-weight Sz sector. "
          "Free energy/entropy re-normalised to the tower dimension "
          "M(N,S); recombine towers with (2S+1) weights via "
          "combine_thermo_weighted.");

}

void bind_workflows_thermal_streaming(py::module_& m) {
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
    //
    // WP9: one body for the directory binding and its in-memory twin
    // ``workflows_thermal_streaming_symmetry``.
    // -----------------------------------------------------------------
    const auto thermal_streaming_symmetry_body =
          [](const SymmetricSource& source,
             std::uint64_t num_sites,
             double spin_l,
             ed::workflows::ThermalOptions opts,
             py::object fixed_sz_n_up) {
              ed::OperatorSpec spec;
              set_symmetric_source(spec, source);
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
              const DirectoryProbe probe =
                  load_probe(spec);
              {
                  const auto comp = ed::symmetry::resolve_symmetry_composition(
                      probe.soa, probe.base->symmetry_info,
                      opts.backend.allow_gpu,
                      ed::symmetry::sym_toggle_from_int(opts.spin_flip),
                      ed::symmetry::sym_toggle_from_int(opts.time_reversal));
                  if (comp.flip_project && !spec.fixed_sz
                      && ed::symmetry::flip_subspace_admissible(
                             -1, spec.sz_parity ? *spec.sz_parity : -1,
                             static_cast<int>(num_sites))) {
                      spec.flip_sectors_full = true;
                  }
              }
              // Stage 12f (SU(2) rollout): per-tower sampling. The Python
              // driver loops the towers; this binding sees ONE tower per
              // call (two_total_spin >= 0) and restricts every sample
              // seed + Krylov walk to it via the Lowdin projector on the
              // sector's own rep basis.
              std::shared_ptr<::Operator> su2t_carrier;
              std::vector<std::uint64_t> su2_tower_dims;
              if (opts.two_total_spin >= 0) {
                  (void)resolve_su2_engagement(
                      *probe.base, opts.two_total_spin, /*label=*/0,
                      "qed.thermal[symmetry]");
                  if (!spec.fixed_sz
                      || *spec.fixed_sz != ed::symmetry::n_up_of_highest_weight(
                             static_cast<int>(num_sites),
                             opts.two_total_spin)) {
                      throw std::runtime_error(
                          "qed.thermal[symmetry]: two_total_spin sampling "
                          "uses the highest-weight formulation -- call with "
                          "fixed_sz = (N + 2S)/2 (the qed.thermal driver "
                          "does this per tower).");
                  }
                  su2t_carrier = ed::ops::make_S2_carrier(num_sites);
                  spec.two_total_spin = opts.two_total_spin;
                  // Exact per-raw-sector tower dims (Burnside differencing):
                  // the sampled estimator averages over the TOWER, but the
                  // kernel's free energy bakes in ln(D_sector) -- correct
                  // to ln(D_tower) after each per-sector call.
                  su2_tower_dims = ed::detail::sector_dims_s_resolved(
                      probe.base->symmetry_info,
                      static_cast<int>(num_sites), opts.two_total_spin);
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
                      ed::make_sector_operators_tagged(spec, mpi_rank, mpi_size,
                                                       probe.base));

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
                  // Audit 2026-07-31 (M2): under across-sector MPI the
                  // gate must be RANK-UNIFORM (sector_indices is this
                  // rank's owned subset -- a rank owning one sector
                  // still has to join the Allreduce below), and the
                  // locally estimated bounds must be MIN/MAX-combined
                  // across ranks: each rank used to run KPM with bounds
                  // from its LOCAL largest sector, breaking the
                  // bit-identical-to-single-node combine and clipping a
                  // rank's other sectors when its local largest bounded
                  // them poorly.
                  if (opts.method ==
                          ed::workflows::ThermalOptions::Method::KpmDos
                      && !(std::isfinite(opts.e_min_override)
                           && std::isfinite(opts.e_max_override))
                      && (sector_indices.size() > 1 || mpi_size > 1)) {
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
#ifdef WITH_MPI
                      if (mpi_size > 1) {
                          // Global bounds: MIN/MAX over every rank's
                          // local estimate (non-finite locals contribute
                          // inert sentinels; every rank participates).
                          const double inf =
                              std::numeric_limits<double>::infinity();
                          double lo_s = std::isfinite(shared_e_min)
                                            ? shared_e_min : inf;
                          double hi_s = std::isfinite(shared_e_max)
                                            ? shared_e_max : -inf;
                          double lo_g = 0.0, hi_g = 0.0;
                          MPI_Allreduce(&lo_s, &lo_g, 1, MPI_DOUBLE,
                                        MPI_MIN, MPI_COMM_WORLD);
                          MPI_Allreduce(&hi_s, &hi_g, 1, MPI_DOUBLE,
                                        MPI_MAX, MPI_COMM_WORLD);
                          if (std::isfinite(lo_g) && std::isfinite(hi_g)
                              && hi_g > lo_g) {
                              shared_e_min = lo_g;
                              shared_e_max = hi_g;
                          } else {
                              shared_e_min = std::numeric_limits<
                                  double>::quiet_NaN();
                              shared_e_max = std::numeric_limits<
                                  double>::quiet_NaN();
                          }
                      }
#endif
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
                  // B6: auto-parallel across many tiny sectors.
                  const bool thermal_sector_parallel = resolve_sector_parallel(
                      sector_indices.size(),
                      max_sector_dim(handle, sector_indices),
                      opts.backend.allow_gpu);

                  // Pre-allocate indexed result storage so the parallel
                  // fill is race-free (each slot ii is written by exactly
                  // one thread; no push_back races).
                  const long n_sec_th =
                      static_cast<long>(sector_indices.size());
                  std::vector<ed::ThermalResult> sec_results_th(
                      static_cast<std::size_t>(n_sec_th));

                  // Audit 2026-07-31 (M1): a throw escaping an OpenMP
                  // parallel region is UB (std::terminate in practice),
                  // and under across-sector MPI a rank dying here skips
                  // the Allgather below and hangs every peer in the
                  // collective. Capture per-sector exceptions, then
                  // coordinate a consistent failure across ranks BEFORE
                  // rethrowing, so either every rank raises or none does.
                  std::exception_ptr sec_eptr;
                  #pragma omp parallel for schedule(dynamic, 1) \
                      if(thermal_sector_parallel)
                  for (long ii = 0; ii < n_sec_th; ++ii) {
                      try {
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
                      // Stage 12f: restrict this sector's stochastic trace
                      // to the spin-S tower -- Lowdin-projected seeds +
                      // drift-scrubbed matvec, S^2 on the SAME rep basis.
                      std::shared_ptr<const
                          ed::symmetry::CasimirProjectedOperator> su2_wrap;
                      if (su2t_carrier) {
                          const auto& rd = sec->producer().ensureRepData();
                          if (!rd.usable()) {
                              throw std::runtime_error(
                                  "qed.thermal[symmetry]: total_spin "
                                  "sampling needs the rep-sector lane "
                                  "(sector " + std::to_string(k)
                                  + " has no usable rep data)");
                          }
                          std::shared_ptr<const
                              ed::matvec::MatVecOperator> s2_mv(
                              ed::solvers::make_rep_sector_matvec(
                                  *su2t_carrier, rd));
                          std::shared_ptr<const
                              ed::matvec::MatVecOperator> h_alias(
                              sec,
                              [](const ed::matvec::MatVecOperator*) {});
                          auto t = make_su2_targeting(
                              std::move(h_alias), std::move(s2_mv),
                              static_cast<int>(num_sites),
                              handle.sector_tag(k).n_up,
                              opts.two_total_spin);
                          if (!t.wrapped) continue;  // tower not admissible
                          auto proj = t.projector;
                          topts.seed_transform =
                              [proj](Complex* v, std::size_t n) {
                                  proj->project(v, n);
                              };
                          su2_wrap = t.wrapped;
                      }
                      try {
                          sec_results_th[static_cast<std::size_t>(ii)] =
                              ed::workflows::thermal(
                                  su2_wrap
                                      ? static_cast<const
                                            ed::LinearOperator&>(*su2_wrap)
                                      : static_cast<const
                                            ed::LinearOperator&>(*sec),
                                  topts);
                      } catch (const std::runtime_error& e) {
                          // A tower with zero weight in this sector
                          // annihilates every seed: skip the sector (its
                          // default-constructed slot is ignored) instead
                          // of failing the whole sweep.
                          if (!su2_wrap
                              || std::string(e.what()).find("annihilated")
                                     == std::string::npos) {
                              throw;
                          }
                      }
                      } catch (...) {   // M1: never escape the OMP region
                          #pragma omp critical(qed_thermal_sector_eptr)
                          { if (!sec_eptr) sec_eptr = std::current_exception(); }
                      }
                  }
#ifdef WITH_MPI
                  if (mpi_size > 1) {
                      // Every rank reaches this collective whether its own
                      // loop failed or not; a failure anywhere then raises
                      // everywhere (peers get a descriptive error instead
                      // of hanging in the Allgather below).
                      int local_fail = sec_eptr ? 1 : 0;
                      int any_fail   = 0;
                      MPI_Allreduce(&local_fail, &any_fail, 1, MPI_INT,
                                    MPI_MAX, MPI_COMM_WORLD);
                      if (any_fail) {
                          if (sec_eptr) std::rethrow_exception(sec_eptr);
                          throw std::runtime_error(
                              "workflows_thermal_streaming_symmetry_"
                              "directory: a peer MPI rank failed inside "
                              "its sector loop (see its stderr); raising "
                              "on every rank instead of deadlocking in "
                              "the thermo Allgather.");
                      }
                  } else if (sec_eptr) {
                      std::rethrow_exception(sec_eptr);
                  }
#else
                  if (sec_eptr) std::rethrow_exception(sec_eptr);
#endif

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
                      if (su2t_carrier) {
                          // Stage 12f: re-normalise the stochastic trace
                          // from the full sector dim (baked into the
                          // kernel's F as ln D_k) to the tower dim:
                          //   Z_tower = Z_est * D_S/D_k
                          //   => F -= T ln(D_S/D_k), S += ln(D_S/D_k).
                          const std::uint64_t d_tower =
                              (tag.sector_index < su2_tower_dims.size())
                                  ? su2_tower_dims[tag.sector_index]
                                  : 0;
                          if (d_tower == 0) continue;  // empty tower here
                          const double lnr =
                              std::log(static_cast<double>(d_tower)
                                       / static_cast<double>(
                                             tag.sector_dim));
                          for (std::size_t t = 0;
                               t < tr.thermo.temperatures.size(); ++t) {
                              tr.thermo.free_energy[t] -=
                                  tr.thermo.temperatures[t] * lnr;
                              tr.thermo.entropy[t] += lnr;
                          }
                          tag.two_S = opts.two_total_spin;
                          tag.sector_dim = d_tower;
                      }
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
          };
    m.def("workflows_thermal_streaming_symmetry_directory",
          [thermal_streaming_symmetry_body](
              const std::string& directory,
              std::uint64_t num_sites,
              double spin_l,
              ed::workflows::ThermalOptions opts,
              py::object fixed_sz_n_up) {
              return thermal_streaming_symmetry_body(
                  ed::DirectoryPath{directory}, num_sites, spin_l,
                  std::move(opts), std::move(fixed_sz_n_up));
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
    m.def("workflows_thermal_streaming_symmetry",
          [thermal_streaming_symmetry_body](
              const Operator& H,
              const py::dict& group,
              std::uint64_t num_sites,
              double spin_l,
              ed::workflows::ThermalOptions opts,
              py::object fixed_sz_n_up) {
              return thermal_streaming_symmetry_body(
                  in_memory_symmetric_source(
                      H, group, "workflows_thermal_streaming_symmetry"),
                  num_sites, spin_l, std::move(opts),
                  std::move(fixed_sz_n_up));
          },
          py::arg("H"),
          py::arg("group"),
          py::arg("num_sites"),
          py::arg("spin_l")      = 0.5,
          py::arg("opts")        = ed::workflows::ThermalOptions{},
          py::arg("fixed_sz_n_up") = py::none(),
          R"pbdoc(
        In-memory twin of ``workflows_thermal_streaming_symmetry_directory``.

        Takes the Hamiltonian ``H`` (its terms are copied) and the group
        info dict the directory writer consumes in place of the
        directory; every other argument and the result are identical.
    )pbdoc");

}
