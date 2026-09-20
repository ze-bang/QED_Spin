// =============================================================================
// python/qed/_bindings/workflow/workflow_spectral_multiq.cpp
//
// The amortised MULTI-Q cross-irrep streaming-symmetry spectral binding
// (`workflows_spectral_streaming_symmetry_cross_irrep_multiq_directory`):
// one ground-state scan, N_Q continued fractions.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_spectral_multiq(py::module_& m) {
    // -----------------------------------------------------------------
    // SOTA cross-irrep MULTI-Q streaming-symmetry spectral binding.
    //
    // Amortised sister of
    // ``workflows_spectral_streaming_symmetry_cross_irrep_directory``.
    // The single-Q entry re-pays the (dominant) per-sector ground-state
    // scan + GS eigenvector solve on every call -- but |psi_0> and the
    // GS irrep ``k_src`` are Q-INDEPENDENT. This entry solves the
    // ground state ONCE and then loops the requested momentum-transfer
    // points internally; each Q only costs one ``resolve_target_sector``
    // + one ``CrossSectorOrbitObservable`` scatter + one inner
    // CF-Lanczos. For a Q-path sweep at 32-36 sites this turns N_Q
    // expensive ground-state solves into a single one.
    //
    // Each Q lands as one ``SpectralSectorEntry`` in
    // ``per_sector_pair`` (positionally aligned with
    // ``momentum_points``):
    //   - ``S_real``     : dynamical S(Q, omega) on the shared grid
    //   - ``static_sf``  : equal-time S(Q) = ||O_Q|psi_0>||^2 (free
    //                      from the CF pivot norm -- this is the SSSF)
    // ``agg.S_real`` mirrors the first Q for back-compat; the per-Q
    // payload always lives in ``per_sector_pair``.
    //
    // Per-Q robustness: a Q that is incommensurate, lands on no
    // surviving sector, or maps to an empty target sector records a
    // zero entry (with a diagnostic note) and the sweep CONTINUES,
    // rather than aborting every other Q-point.
    // -----------------------------------------------------------------
    m.def("workflows_spectral_streaming_symmetry_cross_irrep_multiq_directory",
          [](const std::string&                       directory,
             std::uint64_t                             num_sites,
             double                                    spin_l,
             const std::vector<std::vector<py::tuple>>& observable_transforms_per_q,
             const std::vector<std::vector<double>>&   momentum_points,
             ed::workflows::SpectralOptions            opts,
             py::object                                fixed_sz_n_up,
             int                                       delta_n_up,
             int                                       sz_parity,
             bool                                      flip_sectors) {
              // Decode the per-Q transform tuples. Each Q-point carries
              // its OWN observable O_Q (the e^{-iQ.r} Fourier phase is
              // baked into the transform coefficients), so the outer
              // list MUST be aligned 1:1 with ``momentum_points``.
              if (momentum_points.empty()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_cross_irrep_"
                      "multiq_directory: momentum_points is empty -- pass at "
                      "least one Q vector.");
              }
              if (observable_transforms_per_q.size() != momentum_points.size()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_cross_irrep_"
                      "multiq_directory: observable_transforms_per_q must be "
                      "aligned 1:1 with momentum_points (got " +
                      std::to_string(observable_transforms_per_q.size()) +
                      " transform lists for " +
                      std::to_string(momentum_points.size()) + " Q-points).");
              }
              std::vector<std::vector<Operator::TransformData>> tlist_per_q;
              tlist_per_q.reserve(observable_transforms_per_q.size());
              for (const auto& rows : observable_transforms_per_q) {
                  std::vector<Operator::TransformData> tlist;
                  tlist.reserve(rows.size());
                  for (const auto& row : rows) {
                      if (row.size() < 6) {
                          throw std::invalid_argument(
                              "workflows_spectral_streaming_symmetry_cross_"
                              "irrep_multiq_directory: each transform must be "
                              "a 6-tuple (op_type, site, coeff, is_two_body, "
                              "op_type_2, site_2).");
                      }
                      Operator::TransformData t;
                      t.op_type      = static_cast<uint8_t>(row[0].cast<int>());
                      t.site_index   = row[1].cast<std::uint64_t>();
                      t.coefficient  = row[2].cast<std::complex<double>>();
                      t.is_two_body  = row[3].cast<bool>();
                      t.op_type_2    = static_cast<uint8_t>(row[4].cast<int>());
                      t.site_index_2 = row[5].cast<std::uint64_t>();
                      tlist.push_back(t);
                  }
                  if (tlist.empty()) {
                      throw std::invalid_argument(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory: one of the per-Q observable "
                          "transform lists is empty.");
                  }
                  tlist_per_q.push_back(std::move(tlist));
              }

              // Decode Python arguments BEFORE dropping the GIL.
              const std::optional<int> fixed_sz_opt = decode_optional_n_up(fixed_sz_n_up);
              ed::SpectralResult agg;
              {
                  py::gil_scoped_release release;

                  // (1) Source streaming operator + OperatorRef.
                  ed::OperatorSpec src_spec = make_cross_irrep_src_spec(
                      ed::DirectoryPath{directory}, num_sites, spin_l,
                      fixed_sz_opt, sz_parity, flip_sectors);
                  // Stage 8d TR panel gate: for a REAL H, the -Q panel of an
                  // adjoint probe pair equals the +Q panel (S(-Q, omega) =
                  // S(Q, omega)^* with a real spectral function) -- detected
                  // once here, applied per-Q below.
                  const DirectoryProbe mq_probe =
                      load_probe(src_spec);
                  const bool h_real =
                      ed::symmetry::hamiltonian_is_real(mq_probe.soa);

                  // Operator-collapse Phase 3 (Jun 2026): direct sector
                  // enumeration via ``make_sector_operators_tagged`` +
                  // ``SectorSetView``. ``src_ref`` is built once after the GS
                  // solve; ``dst_ref`` is rebuilt per-Q from the resolved
                  // target sector (the target sector varies with Q).
                  ed::core::SectorSetView src_handle(
                      ed::make_sector_operators_tagged(src_spec, 0, 1,
                                                       mq_probe.base));

                  const std::size_t src_num_sectors = src_handle.num_sectors();
                  if (src_num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory: source operator has no symmetry "
                          "sectors; check automorphism_results/.");
                  }

                  // (2) Locate the global GS sector -- two-phase scan
                  //     (cheap krylov=40 pass, refine within gap), run
                  //     ONCE and reused across all Q.
                  const std::vector<std::size_t> src_sector_indices =
                      ed::core::filter_sectors(src_num_sectors,
                                               opts.selected_sectors);
                  const GsScanResult gs_scan = find_gs_sector_two_phase(
                      src_handle, src_sector_indices, opts);
                  const std::size_t gs_src_idx = gs_scan.gs_idx;
                  const double      gs_energy  = gs_scan.gs_energy;
                  std::fprintf(stderr, "[multiq_gs_energy] %.12f\n", gs_energy);
                  std::fflush(stderr);
                  const bool        any_solved = gs_scan.any_solved;
                  if (!any_solved) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory: every source sector returned an "
                          "empty spectrum.");
                  }

                  // (3) Re-solve GS sector with eigenvectors -> |psi_0>.
                  auto gs_sec_view = src_handle.sector(gs_src_idx);
                  ed::workflows::SolveOptions sopts_full;
                  sopts_full.num_eigs        = 1;
                  sopts_full.tolerance       = 1e-12;
                  sopts_full.backend         = opts.backend;
                  sopts_full.method          = ed::workflows::SolveMethod::Lanczos;
                  sopts_full.compute_vectors = true;
                  // GS eigenVECTOR quality gates every scattered weight and
                  // the CF resolvent; give the solve enough iterations that
                  // ensure_gs_residual's CGS2 rescue (which materialises a
                  // full Krylov basis -- unaffordable at frontier dims)
                  // stays a no-op.
                  sopts_full.max_iter        = std::min<std::size_t>(
                      gs_sec_view->dim(), 600);
                  auto gs_sr = ed::workflows::solve(*gs_sec_view, sopts_full);
                  if (gs_sr.eigenvalues.empty() || !gs_sr.eigenvectors ||
                      gs_sr.eigenvectors->host.empty()) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory: GS eigenvector reconstruction "
                          "failed.");
                  }
                  if (!gs_sr.backend.lane.empty()) {
                      agg.backend.lane     = gs_sr.backend.lane;
                      agg.backend.mpi_size = gs_sr.backend.mpi_size;
                  }
                  std::vector<Complex> psi0 = gs_sr.eigenvectors->host[0];
                  double E0 = gs_sr.eigenvalues.front();
                  ensure_gs_residual(*gs_sec_view, psi0, E0);
                  ed::SectorTag gs_src_tag = src_handle.sector_tag(gs_src_idx);

                  // Source observable handle (Q-independent, built once).
                  // Stage 8d: rep-lazy / flip / parity sectors ride the
                  // CSR-free RepSectorData ref.
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef src_ref =
                      make_cross_sector_ref(
                          gs_sec_view, static_cast<std::uint64_t>(num_sites));

                  // (4) Build / re-use the target sector set ONCE (depends
                  //     only on delta_n_up, not on Q). The view is cheaply
                  //     copyable (shared_ptr-backed), so the ``delta == 0``
                  //     case shares the source set.
                  ed::core::SectorSetView dst_handle = src_handle;
                  if (delta_n_up != 0) {
                      if (!src_spec.fixed_sz.has_value()) {
                          throw std::invalid_argument(
                              "workflows_spectral_streaming_symmetry_cross_"
                              "irrep_multiq_directory: delta_n_up != 0 "
                              "requires fixed_sz_n_up to be set.");
                      }
                      ed::OperatorSpec dst_spec;
                      dst_spec.source             = ed::DirectoryPath{directory};
                      dst_spec.num_sites          = num_sites;
                      dst_spec.spin_l             = static_cast<float>(spin_l);
                      dst_spec.streaming_symmetry = true;
                      dst_spec.fixed_sz           = *src_spec.fixed_sz + delta_n_up;
                      // Same source as the probe: reuse its carrier (the
                      // factory reads only its terms + symmetry_info, which
                      // do not depend on n_up) instead of re-parsing.
                      dst_handle = ed::core::SectorSetView(
                          ed::make_sector_operators_tagged(dst_spec, 0, 1,
                                                           mq_probe.base));
                  }

                  // (5) Shared omega grid (built once).
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
                          omega_grid[i] =
                              opts.omega_min + static_cast<double>(i) * step;
                      }
                  }
                  agg.omega = omega_grid;

                  const double resolvent_shift =
                      (std::abs(opts.energy_shift) > 1e-14) ? opts.energy_shift
                                                            : E0;

                  // (6) Loop the requested Q-points. The GS solve above is
                  //     NOT repeated -- only the per-Q scatter + inner CF.
                  std::string lane_label = agg.backend.lane;
                  for (std::size_t qi = 0; qi < momentum_points.size(); ++qi) {
                      const std::vector<double>& Q = momentum_points[qi];

                      auto push_zero_entry = [&](const std::string& why) {
                          ed::SpectralSectorEntry e;
                          e.initial   = gs_src_tag;
                          e.final_    = gs_src_tag;
                          e.S_real.assign(num_omega, 0.0);
                          e.S_imag.assign(num_omega, 0.0);
                          e.static_sf = 0.0;
                          e.notes.emplace_back("status", why);
                          agg.per_sector_pair.push_back(std::move(e));
                      };

                      // Stage 8d TR panel copy: an earlier Q_j with
                      // Q_j == -Q_i (mod 1) whose probe is THIS probe's
                      // adjoint already computed this panel (real H).
                      if (h_real) {
                          bool copied = false;
                          for (std::size_t qj = 0;
                               qj < qi && qj < agg.per_sector_pair.size()
                               && !copied; ++qj) {
                              const auto& Qj = momentum_points[qj];
                              if (Qj.size() != Q.size()) continue;
                              bool neg = true;
                              for (std::size_t c = 0; c < Q.size(); ++c) {
                                  double s = Q[c] + Qj[c];
                                  s -= std::round(s);
                                  if (std::abs(s) > 1e-9) { neg = false; break; }
                              }
                              if (!neg) continue;
                              const auto& prev = agg.per_sector_pair[qj];
                              bool prev_ok = false;
                              for (const auto& n : prev.notes) {
                                  if (n.first == "status"
                                      && n.second.rfind("ok", 0) == 0) {
                                      prev_ok = true;
                                      break;
                                  }
                              }
                              if (!prev_ok) continue;
                              if (!ed::symmetry::transforms_are_conjugate(
                                      tlist_per_q[qj], tlist_per_q[qi]))
                                  continue;
                              ed::SpectralSectorEntry e;
                              e.initial   = prev.initial;
                              e.final_    = prev.final_;
                              e.S_real    = prev.S_real;
                              e.S_imag    = prev.S_imag;
                              e.static_sf = prev.static_sf;
                              e.notes.emplace_back(
                                  "status",
                                  "ok (TR panel copy of Q#"
                                  + std::to_string(qj) + ")");
                              agg.per_sector_pair.push_back(std::move(e));
                              copied = true;
                          }
                          if (copied) continue;
                      }

                      // Stage 8d: per-Q slot routing (each Q has its own
                      // probe; classifier failures are caller errors, not
                      // zero-weight physics, so they throw).
                      const SlottedSelection slots = slotted_selection_for(
                          src_spec, tlist_per_q[qi],
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "multiq_directory");

                      double q_residual = 0.0;
                      const std::size_t dst_sector_idx =
                          ed::core::resolve_target_sector_slotted(
                              dst_handle, gs_src_idx, Q,
                              slots.n_slots, slots.signs, &q_residual);
                      if (dst_sector_idx == ed::core::kSectorNotFound) {
                          push_zero_entry(
                              "no surviving target sector (residual=" +
                              std::to_string(q_residual) + ")");
                          continue;
                      }
                      if (q_residual > opts.momentum_tolerance) {
                          push_zero_entry(
                              "Q incommensurate (residual=" +
                              std::to_string(q_residual) + " > tol=" +
                              std::to_string(opts.momentum_tolerance) + ")");
                          continue;
                      }
                      ed::SectorTag dst_tag =
                          dst_handle.sector_tag(dst_sector_idx);

                      // Resolve the target sector operator; an empty (dropped)
                      // irrep => zero entry. Build ``dst_ref`` from the
                      // resolved sector's materialised orbit basis (the target
                      // sector varies with Q, so this is per-Q).
                      ed::symmetry::SectorOperator* dst_sec_view =
                          dst_handle.sector(dst_sector_idx);
                      if (!dst_sec_view || dst_sec_view->dim() == 0) {
                          push_zero_entry("target sector empty");
                          continue;
                      }
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef dst_ref =
                          make_cross_sector_ref(
                              dst_sec_view,
                              static_cast<std::uint64_t>(num_sites));

                      ed::dssf::CrossSectorOrbitObservable orb_obs(
                          src_ref, /*src_sector=*/0,
                          dst_ref, /*dst_sector=*/0,
                          tlist_per_q[qi], static_cast<float>(spin_l));
                      const std::size_t dim_dst = orb_obs.dim_dst();
                      if (dst_sec_view->dim() != dim_dst) {
                          push_zero_entry("target sector dim mismatch");
                          continue;
                      }
                      std::vector<Complex> phi(dim_dst, Complex(0.0, 0.0));
                      orb_obs.apply(psi0.data(), phi.data(), dim_dst);

                      ed::observables::CfSpectralOptions cfopts;
                      cfopts.krylov_dim   = opts.krylov_dim;
                      cfopts.broadening   = opts.broadening;
                      cfopts.energy_shift = resolvent_shift;
                      cfopts.tolerance    = 1e-12;
                      cfopts.global_n     = dim_dst;
                      cfopts.verbose      = false;

                      ed::observables::CfSpectralResult cf;
                      auto cf_variant = ed::select_backend(
                          dst_sec_view->geometry(), opts.backend);
                      std::visit([&](auto& backend_uptr) {
                          using BPtr = std::decay_t<decltype(backend_uptr)>;
                          using B = typename BPtr::element_type;
                          auto apply_H = dst_sec_view->template bind<B>();
                          cf = ed::observables::cf_spectral_from_vector(
                              *backend_uptr, apply_H, dim_dst,
                              phi.data(), omega_grid, cfopts);
                      }, cf_variant);
                      if (lane_label.empty()) {
                          lane_label = ed::lane_label_from_variant(cf_variant);
                      }

                      ed::SpectralSectorEntry entry;
                      entry.initial   = gs_src_tag;
                      entry.final_    = dst_tag;
                      entry.S_real    = cf.spectral_function;
                      entry.S_imag.assign(cf.spectral_function.size(), 0.0);
                      entry.static_sf = cf.phi_norm * cf.phi_norm;
                      entry.notes.emplace_back("status", "ok");
                      agg.per_sector_pair.push_back(std::move(entry));

                      // Mirror the FIRST resolved Q into the top-level
                      // S_real for back-compat with single-Q callers.
                      if (agg.S_real.empty()) {
                          agg.S_real = cf.spectral_function;
                          agg.S_imag.assign(cf.spectral_function.size(), 0.0);
                          agg.errors_real.assign(cf.spectral_function.size(), 0.0);
                          agg.errors_imag.assign(cf.spectral_function.size(), 0.0);
                      }
                  }

                  if (agg.S_real.empty()) {
                      agg.S_real.assign(num_omega, 0.0);
                      agg.S_imag.assign(num_omega, 0.0);
                      agg.errors_real.assign(num_omega, 0.0);
                      agg.errors_imag.assign(num_omega, 0.0);
                  }
                  if (!lane_label.empty()) agg.backend.lane = lane_label;
                  agg.selection_rule_label =
                      "k_final = k_initial + Q (cross-irrep, multi-Q "
                      "amortised; " +
                      std::to_string(momentum_points.size()) +
                      " Q-points, single GS solve)";
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")                       = 0.5,
          py::arg("observable_transforms_per_q")  = std::vector<std::vector<py::tuple>>{},
          py::arg("momentum_points")              = std::vector<std::vector<double>>{},
          py::arg("opts")                         = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up")                = py::none(),
          py::arg("delta_n_up")                   = 0,
          py::arg("sz_parity")                    = -1,
          py::arg("flip_sectors")                 = false,
          R"pbdoc(
        Amortised multi-Q cross-irrep streaming-symmetry spectral
        workflow (SOTA).

        Identical physics to
        ``workflows_spectral_streaming_symmetry_cross_irrep_directory``
        but the per-sector ground-state scan and the GS eigenvector
        solve -- the dominant cost -- run ONCE and are reused across
        every momentum-transfer point in ``momentum_points``. Each Q
        carries its OWN observable O_Q (its e^{-iQ.r} Fourier phase is
        baked into the transform coefficients), so
        ``observable_transforms_per_q`` is a list of transform lists
        aligned 1:1 with ``momentum_points``. Per Q the only work is a
        selection-rule lookup, a ``CrossSectorOrbitObservable`` scatter,
        and one inner CF-Lanczos in the (reduced) target sector.

        Results land in ``per_sector_pair`` positionally aligned with
        ``momentum_points``: ``S_real`` is the dynamical S(Q, omega)
        and ``static_sf`` is the equal-time S(Q) = ||O_Q|psi_0>||^2
        (the static / equal-time structure factor, obtained for free
        from the CF pivot norm). Incommensurate or empty-target Q
        points record a zero entry with a ``status`` note and the
        sweep continues.
    )pbdoc");

}
