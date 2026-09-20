// =============================================================================
// python/qed/_bindings/workflow/workflow_solve.cpp
//
// The solve lane: the plain `workflows_solve` entry point and the
// streaming-symmetry solve pair (`workflows_solve_streaming_symmetry`
// + its `_directory` twin), which share one body lambda.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_solve(py::module_& m) {
    m.def("workflows_solve",
          [](Operator& op, ed::workflows::SolveOptions opts) {
              // Stage 12 (SU(2) rollout): total-spin axis for the plain /
              // fixed-Sz lanes. Targeting wraps the operator in the Lowdin
              // projector (Krylov lane, projected seed); auto labeling
              // fills certified two_S when eigenvectors come back.
              const bool su2_on = resolve_su2_engagement(
                  op, opts.two_total_spin, opts.label_total_spin,
                  "qed.solve");
              int su2_n_up = -1;
              if (su2_on) {
                  if (const auto* fsz =
                          dynamic_cast<const FixedSzOperator*>(&op)) {
                      su2_n_up = static_cast<int>(fsz->producer().n_up());
                  }
              }
              if (su2_on && opts.two_total_spin >= 0) {
                  const int two_S = opts.two_total_spin;
                  // Non-owning alias: the Python caller owns `op` for the
                  // duration of the call.
                  std::shared_ptr<const ed::matvec::MatVecOperator> h(
                      &op, [](const ed::matvec::MatVecOperator*) {});
                  auto t = make_su2_targeting(
                      h, make_s2_like(op),
                      static_cast<int>(op.getNumBits()), su2_n_up, two_S);
                  if (!t.wrapped) {
                      return ed::GroundStateResult{};  // inadmissible tower
                  }
                  ed::GroundStateResult res;
                  {
                      py::gil_scoped_release release;
                      res = solve_su2_targeted(*t.wrapped, t.projector,
                                               std::move(opts));
                  }
                  res.s2_of_eigenvalue.assign(
                      res.eigenvalues.size(),
                      ed::ops::s2_eigenvalue_of_two_S(two_S));
                  res.two_S_of_eigenvalue.assign(res.eigenvalues.size(),
                                                 two_S);
                  return res;
              }
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
              auto res = ed::workflows::solve(
                  static_cast<const ed::LinearOperator&>(op), std::move(opts));
              if (su2_on && res.eigenvectors
                  && !res.eigenvectors->host.empty()) {
                  auto s2 = make_s2_like(op);
                  label_vectors_with_s2(
                      *s2, res.eigenvectors->host,
                      static_cast<int>(op.getNumBits()), su2_n_up,
                      /*flip_parity=*/-1,
                      res.s2_of_eigenvalue, res.two_S_of_eigenvalue, &res.eigenvalues);
              }
              return res;
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

}

void bind_workflows_solve_streaming(py::module_& m) {
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
    //
    // WP9: ONE body shared by the directory binding and its in-memory twin
    // ``workflows_solve_streaming_symmetry`` (H + group dict); the two
    // m.def's below differ only in how they build the source.
    // -----------------------------------------------------------------
    const auto solve_streaming_symmetry_body =
          [](const SymmetricSource& source,
             std::uint64_t num_sites,
             double spin_l,
             ed::workflows::SolveOptions opts,
             py::object fixed_sz_n_up) {
              ed::OperatorSpec spec;
              set_symmetric_source(spec, source);
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
                  DirectoryProbe probe =
                      load_probe(spec);
                  ed::symmetry::SymmetryComposition comp =
                      resolve_comp_with_stars(probe, opts);
                  // Stage 12 (SU(2) rollout): engagement + ONE carrier for
                  // every sector. Targeting (two_total_spin >= 0) wraps each
                  // sector matvec in the Lowdin projector and projects the
                  // Krylov seed; labeling fills certified two_S per
                  // eigenvalue when eigenvectors are available.
                  const bool su2_on = resolve_su2_engagement(
                      *probe.base, opts.two_total_spin,
                      opts.label_total_spin, "qed.solve[symmetry]");
                  const bool su2_target =
                      su2_on && opts.two_total_spin >= 0;
                  std::shared_ptr<::Operator> s2_carrier =
                      su2_on ? ed::ops::make_S2_carrier(num_sites)
                             : nullptr;
                  spec.two_total_spin =
                      su2_target ? opts.two_total_spin : -1;
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
                      && ed::symmetry::flip_subspace_admissible(
                             *spec.fixed_sz, -1,
                             static_cast<int>(num_sites))
                      && !opts.compute_vectors) {
                      spec.flip_project_half = true;
                  }
                  // Full-space / parity-half prod-sigma^x sectors: no
                  // fixed-Sz axis but [H, X] == 0 -- every irrep splits
                  // (k, +/-). Closure rule: over a parity half the flip
                  // only preserves the subspace for even N.
                  if (comp.flip_project && !spec.fixed_sz
                      && !opts.compute_vectors
                      && ed::symmetry::flip_subspace_admissible(
                             -1, spec.sz_parity ? *spec.sz_parity : -1,
                             static_cast<int>(num_sites))) {
                      spec.flip_sectors_full = true;
                  }
                  ed::core::SectorSetView handle(
                      ed::make_sector_operators_tagged(spec, 0, 1,
                                                       probe.base));

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
                      // Structural consolidation (Jul 2026): the GS lane
                      // consumes the SAME orbit plan the thermal flat
                      // pool uses (plan_tr_actions over the selected
                      // tags) instead of a hand-rolled twin of it.
                      std::vector<ed::SectorTag> sel_tags;
                      sel_tags.reserve(sector_indices_all.size());
                      for (std::size_t k : sector_indices_all)
                          sel_tags.push_back(handle.sector_tag(k));
                      const auto plan = ed::symmetry::plan_tr_actions(
                          sel_tags, tr_num_raw, comp);
                      for (std::size_t i = 0;
                           i < sector_indices_all.size(); ++i) {
                          if (plan.skip[i]) {
                              tr_mirrors.emplace_back(
                                  sector_indices_all[i],
                                  sector_indices_all[plan.source[i]]);
                          } else {
                              sector_indices.push_back(
                                  sector_indices_all[i]);
                          }
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
                  // Stage 12: per-sector label arrays parallel to
                  // ``eigs_per_sector`` (placeholders -1 / NaN where a
                  // vector was unavailable or failed certification).
                  const bool su2_label_arrays = su2_on;
                  std::vector<std::vector<double>>     s2_per_sector;
                  std::vector<std::vector<int>>        twoS_per_sector;

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
                      && !opts.precompute_basis_only
                      // Stage 12: the phase-1 scan solves the UNPROJECTED
                      // sector and would prune by the wrong (all-tower)
                      // minima under total-spin targeting.
                      && !su2_target;
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
                  // B6: auto-parallel the outer sector loop in the
                  // many-tiny-sectors regime (composed sectors); explicit
                  // ED_SYM_SECTOR_PARALLEL wins.
                  const bool sector_parallel = resolve_sector_parallel(
                      sector_indices.size(),
                      max_sector_dim(handle, sector_indices),
                      opts.backend.allow_gpu);
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

                  // A per-sector solve can now legitimately THROW (e.g.
                  // full_diagonalization refuses a full-spectrum request
                  // above its dense window since the Eigen-sparse fallback
                  // was retired, 2026-07-20). An exception escaping an OMP
                  // worker is std::terminate, so latch the first one and
                  // rethrow after the region -- the caller gets a clean
                  // Python RuntimeError instead of a dead process.
                  std::exception_ptr sector_solve_err = nullptr;
                  #pragma omp parallel for schedule(dynamic, 1) \
                      if(sector_parallel)
                  for (long ii = 0; ii < n_iter_sec; ++ii) {
                      try {
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
                          if (su2_target) {
                              // Stage 12d: solve the spin-S tower of this
                              // sector through the Lowdin-wrapped matvec.
                              // S^2 rides the SAME rep-sector data as H
                              // ([S^2, g] = 0 for every group element).
                              const auto& rd =
                                  sec->producer().ensureRepData();
                              if (!rd.usable()) {
                                  throw std::runtime_error(
                                      "qed.solve[symmetry]: total_spin "
                                      "targeting needs the rep-sector "
                                      "lane (sector " + std::to_string(k)
                                      + " has no usable rep data)");
                              }
                              std::shared_ptr<const
                                  ed::matvec::MatVecOperator> s2_mv(
                                  ed::solvers::make_rep_sector_matvec(
                                      *s2_carrier, rd));
                              // Non-owning alias: `handle` outlives the
                              // sector loop.
                              std::shared_ptr<const
                                  ed::matvec::MatVecOperator> h_alias(
                                  sec,
                                  [](const ed::matvec::MatVecOperator*) {});
                              auto t = make_su2_targeting(
                                  std::move(h_alias),
                                  std::move(s2_mv),
                                  static_cast<int>(num_sites),
                                  handle.sector_tag(k).n_up,
                                  opts.two_total_spin);
                              if (t.wrapped) {
                                  solve_results_p2[
                                      static_cast<std::size_t>(ii)] =
                                      solve_su2_targeted(*t.wrapped,
                                                         t.projector,
                                                         sopts);
                              }
                              // inadmissible tower: leave the empty result
                          } else {
                              solve_results_p2[static_cast<std::size_t>(ii)] =
                                  ed::workflows::solve(*sec, sopts);
                          }
                      } catch (...) {
                          #pragma omp critical(sector_solve_err_latch)
                          {
                              if (!sector_solve_err) {
                                  sector_solve_err = std::current_exception();
                              }
                          }
                      }
                  }
                  if (sector_solve_err) {
                      std::rethrow_exception(sector_solve_err);
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
                      if (su2_target) {
                          touched_tags.back().two_S = opts.two_total_spin;
                      }
                      // Stage 12a: post-hoc <S^2> labeling when the sector
                      // returned host eigenvectors (auto/require toggle).
                      // Under targeting the label is known by construction.
                      if (su2_label_arrays && !su2_target
                          && opts.label_total_spin != 0
                          && sr.eigenvectors
                          && !sr.eigenvectors->host.empty()) {
                          const auto& rd = sec->producer().ensureRepData();
                          if (rd.usable()) {
                              std::shared_ptr<const
                                  ed::matvec::MatVecOperator> s2_mv(
                                  ed::solvers::make_rep_sector_matvec(
                                      *s2_carrier, rd));
                              sr.s2_of_eigenvalue.clear();
                              sr.two_S_of_eigenvalue.clear();
                              label_vectors_with_s2(
                                  *s2_mv, sr.eigenvectors->host,
                                  static_cast<int>(num_sites),
                                  handle.sector_tag(k).n_up,
                                  /*flip_parity=*/-1,
                                  sr.s2_of_eigenvalue,
                                  sr.two_S_of_eigenvalue, &sr.eigenvalues);
                          }
                      }
                      // GAP-10 v2: the MERGED window is where an
                      // uncertified interior Ritz value does damage
                      // (a stalled value from one sector can shadow
                      // another sector's true lowest). Filter on the
                      // per-value residual bound the orchestrator now
                      // reports; each sector's LOWEST is always kept
                      // (it is convergence-guarded by the stall
                      // checker). Direct single-operator calls keep
                      // their full num_eigs window + diagnostics.
                      std::vector<double> kept;
                      std::vector<double> kept_s2;
                      std::vector<int>    kept_twoS;
                      kept.reserve(sr.eigenvalues.size());
                      const auto& rb = sr.krylov.ritz_residuals;
                      const double s2_target_val =
                          su2_target ? ed::ops::s2_eigenvalue_of_two_S(
                                           opts.two_total_spin)
                                     : std::numeric_limits<double>::quiet_NaN();
                      for (std::size_t vi = 0;
                           vi < sr.eigenvalues.size(); ++vi) {
                          if (vi > 0 && vi < rb.size()
                              && rb[vi] > 1e-6)
                              continue;
                          kept.push_back(sr.eigenvalues[vi]);
                          if (su2_label_arrays) {
                              if (su2_target) {
                                  kept_s2.push_back(s2_target_val);
                                  kept_twoS.push_back(opts.two_total_spin);
                              } else if (vi < sr.two_S_of_eigenvalue.size()) {
                                  kept_s2.push_back(sr.s2_of_eigenvalue[vi]);
                                  kept_twoS.push_back(
                                      sr.two_S_of_eigenvalue[vi]);
                              } else {
                                  kept_s2.push_back(
                                      std::numeric_limits<double>::quiet_NaN());
                                  kept_twoS.push_back(-1);
                              }
                          }
                      }
                      eigs_per_sector.push_back(kept);
                      if (su2_label_arrays) {
                          s2_per_sector.push_back(std::move(kept_s2));
                          twoS_per_sector.push_back(std::move(kept_twoS));
                      }
                      all_eigs.insert(all_eigs.end(),
                                      kept.begin(), kept.end());
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
                      if (su2_target) {
                          touched_tags.back().two_S = opts.two_total_spin;
                      }
                      all_eigs.insert(all_eigs.end(),
                                      copy_eigs.begin(), copy_eigs.end());
                      eigs_per_sector.push_back(std::move(copy_eigs));
                      // Stage 12: S is basis-independent, so labels copy
                      // verbatim under the k <-> -k fold.
                      if (su2_label_arrays) {
                          s2_per_sector.push_back(s2_per_sector[src_slot]);
                          twoS_per_sector.push_back(
                              twoS_per_sector[src_slot]);
                      }
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
                  // Stage 12: flatten the per-sector label arrays in the
                  // same per-sector order all_eigs used, then apply the
                  // same sort permutation so the labels stay parallel to
                  // the merged eigenvalue list.
                  std::vector<double> sorted_s2;
                  std::vector<int>    sorted_twoS;
                  if (su2_label_arrays) {
                      std::vector<double> flat_s2;
                      std::vector<int>    flat_twoS;
                      flat_s2.reserve(all_eigs.size());
                      flat_twoS.reserve(all_eigs.size());
                      for (std::size_t s = 0; s < s2_per_sector.size();
                           ++s) {
                          flat_s2.insert(flat_s2.end(),
                                         s2_per_sector[s].begin(),
                                         s2_per_sector[s].end());
                          flat_twoS.insert(flat_twoS.end(),
                                           twoS_per_sector[s].begin(),
                                           twoS_per_sector[s].end());
                      }
                      if (flat_s2.size() == all_eigs.size()) {
                          sorted_s2.resize(all_eigs.size());
                          sorted_twoS.resize(all_eigs.size());
                          for (std::size_t i = 0; i < perm.size(); ++i) {
                              sorted_s2[i]   = flat_s2[perm[i]];
                              sorted_twoS[i] = flat_twoS[perm[i]];
                          }
                      }
                  }
                  if (opts.num_eigs > 0
                      && sorted_eigs.size() > opts.num_eigs) {
                      sorted_eigs.resize(opts.num_eigs);
                      sorted_origin.resize(opts.num_eigs);
                      if (!sorted_s2.empty()) {
                          sorted_s2.resize(opts.num_eigs);
                          sorted_twoS.resize(opts.num_eigs);
                      }
                  }

                  agg.eigenvalues                = std::move(sorted_eigs);
                  agg.sector_tags                = std::move(touched_tags);
                  agg.eigenvalues_per_sector     = std::move(eigs_per_sector);
                  agg.sector_index_of_eigenvalue = std::move(sorted_origin);
                  // Attach the label arrays only when something was
                  // actually labeled (targeting counts); an auto run
                  // with no eigenvectors leaves them empty rather than
                  // all-placeholder.
                  const bool any_label = su2_target
                      || std::any_of(sorted_twoS.begin(), sorted_twoS.end(),
                                     [](int t) { return t >= 0; });
                  if (any_label) {
                      agg.s2_of_eigenvalue    = std::move(sorted_s2);
                      agg.two_S_of_eigenvalue = std::move(sorted_twoS);
                  }
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
          };
    m.def("workflows_solve_streaming_symmetry_directory",
          [solve_streaming_symmetry_body](
              const std::string& directory,
              std::uint64_t num_sites,
              double spin_l,
              ed::workflows::SolveOptions opts,
              py::object fixed_sz_n_up) {
              return solve_streaming_symmetry_body(
                  ed::DirectoryPath{directory}, num_sites, spin_l,
                  std::move(opts), std::move(fixed_sz_n_up));
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
    m.def("workflows_solve_streaming_symmetry",
          [solve_streaming_symmetry_body](
              const Operator& H,
              const py::dict& group,
              std::uint64_t num_sites,
              double spin_l,
              ed::workflows::SolveOptions opts,
              py::object fixed_sz_n_up) {
              return solve_streaming_symmetry_body(
                  in_memory_symmetric_source(
                      H, group, "workflows_solve_streaming_symmetry"),
                  num_sites, spin_l, std::move(opts),
                  std::move(fixed_sz_n_up));
          },
          py::arg("H"),
          py::arg("group"),
          py::arg("num_sites"),
          py::arg("spin_l")      = 0.5,
          py::arg("opts")        = ed::workflows::SolveOptions{},
          py::arg("fixed_sz_n_up") = py::none(),
          R"pbdoc(
        In-memory twin of ``workflows_solve_streaming_symmetry_directory``.

        Takes the Hamiltonian ``H`` (its terms are copied) and the group
        info dict the directory writer consumes (``max_clique``,
        ``generators``, ``generator_orders``, ``sectors`` with
        ``sector_id`` / ``quantum_numbers``) in place of the directory;
        every other argument and the result are identical.
    )pbdoc");

}
