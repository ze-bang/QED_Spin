// =============================================================================
// python/qed/_bindings/workflow/workflow_thermal_all_sz.cpp
//
// The all-Sz flat-pool thermal binding (Jun 2026):
// `workflows_thermal_all_sz_streaming_symmetry[_directory]`, one body
// lambda shared by the directory binding and its in-memory twin.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_thermal_all_sz(py::module_& m) {
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
    //
    // WP9: one body for the directory binding and its in-memory twin
    // ``workflows_thermal_all_sz_streaming_symmetry``.
    // -----------------------------------------------------------------
    const auto thermal_all_sz_streaming_symmetry_body =
          [](const SymmetricSource& source,
             std::uint64_t num_sites,
             double spin_l,
             ed::workflows::ThermalOptions opts,
             int n_up_min,
             int n_up_max) {
              ed::ThermalResult agg;
              {
                  py::gil_scoped_release release;

                  ed::OperatorSpec spec;
                  set_symmetric_source(spec, source);
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
                  DirectoryProbe probe =
                      load_probe(spec);
                  ed::symmetry::SymmetryComposition comp =
                      resolve_comp_with_stars(probe, opts);
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
                          spec, win.lo, win.hi, flip_project, probe.base);

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
                  // B6: auto-parallel across the flat (n_up x irrep) set
                  // when it is many tiny sectors.
                  std::uint64_t fp_max_dim = 0;
                  for (const auto& fp_op : set.operators)
                      if (fp_op) fp_max_dim = std::max<std::uint64_t>(
                          fp_max_dim, fp_op->dim());
                  const bool sector_parallel = resolve_sector_parallel(
                      static_cast<std::size_t>(n_ops), fp_max_dim,
                      opts.backend.allow_gpu);

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
          };
    m.def("workflows_thermal_all_sz_streaming_symmetry_directory",
          [thermal_all_sz_streaming_symmetry_body](
              const std::string& directory,
              std::uint64_t num_sites,
              double spin_l,
              ed::workflows::ThermalOptions opts,
              int n_up_min,
              int n_up_max) {
              return thermal_all_sz_streaming_symmetry_body(
                  ed::DirectoryPath{directory}, num_sites, spin_l,
                  std::move(opts), n_up_min, n_up_max);
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
    m.def("workflows_thermal_all_sz_streaming_symmetry",
          [thermal_all_sz_streaming_symmetry_body](
              const Operator& H,
              const py::dict& group,
              std::uint64_t num_sites,
              double spin_l,
              ed::workflows::ThermalOptions opts,
              int n_up_min,
              int n_up_max) {
              return thermal_all_sz_streaming_symmetry_body(
                  in_memory_symmetric_source(
                      H, group,
                      "workflows_thermal_all_sz_streaming_symmetry"),
                  num_sites, spin_l, std::move(opts), n_up_min, n_up_max);
          },
          py::arg("H"),
          py::arg("group"),
          py::arg("num_sites"),
          py::arg("spin_l")    = 0.5,
          py::arg("opts")      = ed::workflows::ThermalOptions{},
          py::arg("n_up_min")  = 0,
          py::arg("n_up_max")  = -1,
          R"pbdoc(
        In-memory twin of
        ``workflows_thermal_all_sz_streaming_symmetry_directory``.

        Takes the Hamiltonian ``H`` (its terms are copied) and the group
        info dict the directory writer consumes in place of the
        directory; every other argument and the result are identical.
    )pbdoc");
}
