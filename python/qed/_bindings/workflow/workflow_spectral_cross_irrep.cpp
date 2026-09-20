// =============================================================================
// python/qed/_bindings/workflow/workflow_spectral_cross_irrep.cpp
//
// The single-Q cross-irrep streaming-symmetry spectral binding
// (`workflows_spectral_streaming_symmetry_cross_irrep[_directory]`): one
// body lambda shared by the directory binding and its in-memory twin.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_spectral_cross_irrep(py::module_& m) {
    // -----------------------------------------------------------------
    // Cross-irrep streaming-symmetry spectral binding (SOTA, May 2026).
    //
    // Closes the remaining gap in docs/architecture/SYMMETRY.md Section
    // 3: dynamical S(Q, omega) for spatial irreps with non-trivial
    // momentum transfer ``Q`` and / or a fixed-Sz delta_n_up.
    //
    // The algorithm:
    //   (1) Per-sector GS pass over the source operator (with the
    //       requested ``fixed_sz_n_up``). Identifies the irrep that
    //       hosts the global ground state.
    //   (2) Re-solve that sector with ``compute_vectors=true`` to
    //       recover |psi_0> in the source-orbit basis.
    //   (3) Resolve the target sector via the selection rule
    //       k_dst = k_src + Q (integer-quantised against the inferred
    //       generator orders in ``include/ed/core/sector_loop.h``).
    //       If ``delta_n_up != 0`` we build a *second* streaming
    //       operator with the shifted ``fixed_sz_n_up`` so the
    //       observable can scatter into a different Sz subspace.
    //   (4) Build a ``CrossSectorOrbitObservable`` from the user's
    //       transform list and apply it to |psi_0> to obtain |phi>
    //       in the *target* orbit basis.
    //   (5) Run ``cf_spectral_from_vector`` against H restricted to
    //       the target sector. The spectral weight ||phi||^2 is
    //       folded in automatically.
    //
    // The returned ``SpectralResult`` carries the (initial, final)
    // SectorTag pair in ``per_sector_pair`` plus a
    // ``selection_rule_label`` documenting how the target sector was
    // chosen.
    // -----------------------------------------------------------------
    //
    // WP9: one body for the directory binding and its in-memory twin
    // ``workflows_spectral_streaming_symmetry_cross_irrep``; the shifted-Sz
    // target set (delta_n_up != 0) is seeded from the SAME source.
    // -----------------------------------------------------------------
    const auto spectral_cross_irrep_body =
          [](const SymmetricSource&                source,
             std::uint64_t                          num_sites,
             double                                 spin_l,
             const std::vector<py::tuple>&          observable_transforms,
             ed::workflows::SpectralOptions         opts,
             py::object                             fixed_sz_n_up,
             int                                    delta_n_up,
             int                                    sz_parity,
             bool                                   flip_sectors) {
              // ----------------------------------------------------------
              // Decode the user-supplied transform tuples into the SoA
              // layout expected by CrossSectorOrbitObservable. Each
              // tuple has the shape produced by
              //   _transforms_from_operator(op) below:
              //     (op_type, site, coeff, is_two_body, op_type_2, site_2)
              // ----------------------------------------------------------
              std::vector<Operator::TransformData> tlist =
                  decode_probe_transforms(
                      observable_transforms,
                      "workflows_spectral_streaming_symmetry_cross_irrep_"
                      "directory");
              if (tlist.empty()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_cross_irrep_"
                      "directory: observable_transforms is empty -- the "
                      "cross-irrep walk needs at least one term.");
              }

              // Decode Python arguments BEFORE dropping the GIL.
              const std::optional<int> fixed_sz_opt = decode_optional_n_up(fixed_sz_n_up);
              ed::SpectralResult agg;
              {
                  py::gil_scoped_release release;

                  // -----------------------------------------------------
                  // (1) Build source streaming operator. Mirror the
                  //     same-irrep binding's OperatorSpec layout.
                  // -----------------------------------------------------
                  ed::OperatorSpec src_spec = make_cross_irrep_src_spec(
                      source, num_sites, spin_l, fixed_sz_opt,
                      sz_parity, flip_sectors);
                  const SlottedSelection slots = slotted_selection_for(
                      src_spec, tlist,
                      "workflows_spectral_streaming_symmetry_cross_irrep_"
                      "directory");

                  // Operator-collapse Phase 3 (Jun 2026): enumerate the source
                  // symmetry sectors directly via
                  // ``make_sector_operators_tagged`` + ``SectorSetView``. The
                  // cross-irrep observable's ``OperatorRef`` is built later
                  // from the resolved source / target sectors (Stage 8d: the
                  // CSR-free RepSectorData on the rep-lazy lane, the
                  // materialised orbit basis otherwise).
                  ed::core::SectorSetView src_handle(
                      ed::make_sector_operators_tagged(src_spec));

                  const std::size_t src_num_sectors = src_handle.num_sectors();
                  if (src_num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: source operator has no symmetry sectors; "
                          "check automorphism_results/.");
                  }

                  // -----------------------------------------------------
                  // (2) Find the global GS sector across the source
                  //     symmetry block.
                  //
                  // Wave B2 (May 2026): two-phase scan -- run a cheap
                  // Phase 1 (``krylov_dim=40``) across every sector, then
                  // refine only candidates within ``gap`` of the best
                  // Phase-1 minimum. Skips N-1 full Lanczos solves on
                  // average. The candidate list is collapsed back to the
                  // single minimum-energy sector for the eigenvector
                  // pull in (3) -- the safety margin is the gap from
                  // best Phase-1 E so the true GS sector is always
                  // among the refined candidates.
                  // -----------------------------------------------------
                  const std::vector<std::size_t> src_sector_indices =
                      ed::core::filter_sectors(src_num_sectors,
                                               opts.selected_sectors);
                  const GsScanResult gs_scan = find_gs_sector_two_phase(
                      src_handle, src_sector_indices, opts);
                  const std::size_t gs_src_idx = gs_scan.gs_idx;
                  const double      gs_energy  = gs_scan.gs_energy;
                  const bool        any_solved = gs_scan.any_solved;
                  if (!any_solved) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: every source sector returned an empty "
                          "spectrum; check the operator / Hilbert space.");
                  }

                  // -----------------------------------------------------
                  // (3) Re-solve the GS sector with compute_vectors=true
                  //     so we can extract |psi_0> for the cross-sector
                  //     scatter step.
                  // -----------------------------------------------------
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
                  auto gs_sr =
                      ed::workflows::solve(*gs_sec_view, sopts_full);
                  if (gs_sr.eigenvalues.empty() || !gs_sr.eigenvectors ||
                      gs_sr.eigenvectors->host.empty()) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: ground-state eigenvector reconstruction "
                          "failed (eigenvectors->host empty).");
                  }
                  // Phase H.1 of the "Close CPU/GPU Gaps" plan
                  // (May 2026): capture the truthful backend lane
                  // from the inner GS solve so the cross-irrep
                  // aggregate reports "gpu" (or "cpu" / "mpi") on
                  // ``agg.backend.lane`` instead of leaving it
                  // empty. The GS sector full solve is the anchor:
                  // it's the only solve guaranteed to run in this
                  // binding (the two-phase scan above does many
                  // cheaper solves but those share the same backend
                  // contract).
                  if (!gs_sr.backend.lane.empty()) {
                      agg.backend.lane     = gs_sr.backend.lane;
                      agg.backend.mpi_size = gs_sr.backend.mpi_size;
                  }
                  std::vector<Complex> psi0 = gs_sr.eigenvectors->host[0];
                  double E0 = gs_sr.eigenvalues.front();
                  ensure_gs_residual(*gs_sec_view, psi0, E0);
                  ed::SectorTag gs_src_tag = src_handle.sector_tag(gs_src_idx);

                  // Source observable handle. Stage 8d: rep-lazy sectors
                  // (including flip / parity sectors, which have no orbit
                  // CSR at all) ride the CSR-free RepSectorData ref; eager
                  // sectors keep the materialised orbit basis.
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef src_ref =
                      make_cross_sector_ref(
                          gs_sec_view, static_cast<std::uint64_t>(num_sites));

                  // -----------------------------------------------------
                  // (4) Build / re-use the target sector set. If
                  //     ``delta_n_up == 0`` we re-use the source set (the
                  //     view is cheaply copyable -- shared_ptr-backed);
                  //     otherwise we enumerate a second set with the shifted
                  //     ``fixed_sz_n_up``.
                  // -----------------------------------------------------
                  ed::core::SectorSetView dst_handle = src_handle;

                  if (delta_n_up != 0) {
                      if (!src_spec.fixed_sz.has_value()) {
                          throw std::invalid_argument(
                              "workflows_spectral_streaming_symmetry_cross_"
                              "irrep_directory: delta_n_up != 0 requires "
                              "fixed_sz_n_up to be set.");
                      }
                      ed::OperatorSpec dst_spec;
                      set_symmetric_source(dst_spec, source);
                      dst_spec.num_sites          = num_sites;
                      dst_spec.spin_l             = static_cast<float>(spin_l);
                      dst_spec.streaming_symmetry = true;
                      dst_spec.fixed_sz           = *src_spec.fixed_sz + delta_n_up;
                      dst_handle = ed::core::SectorSetView(
                          ed::make_sector_operators_tagged(dst_spec));
                  }

                  // -----------------------------------------------------
                  // (5) Resolve the target sector via the selection
                  //     rule. ``Q`` lives in fractional reciprocal-
                  //     lattice units; sector_loop.h does the integer
                  //     quantisation.
                  // -----------------------------------------------------
                  double q_residual = 0.0;
                  const std::size_t dst_sector_idx =
                      ed::core::resolve_target_sector_slotted(
                          dst_handle,
                          gs_src_idx,
                          opts.momentum_transfer,
                          slots.n_slots,
                          slots.signs,
                          &q_residual);
                  if (dst_sector_idx == ed::core::kSectorNotFound) {
                      // Build a useful diagnostic for the no-survivor
                      // case so users can tell *why* their Q didn't land
                      // anywhere.
                      std::string msg =
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: no surviving target sector for the "
                          "requested selection rule. Source sector qn = [";
                      for (std::size_t g = 0; g < gs_src_tag.quantum_numbers.size(); ++g) {
                          if (g) msg += ", ";
                          msg += std::to_string(gs_src_tag.quantum_numbers[g]);
                      }
                      msg += "], Q (frac) = [";
                      for (std::size_t g = 0; g < opts.momentum_transfer.size(); ++g) {
                          if (g) msg += ", ";
                          msg += std::to_string(opts.momentum_transfer[g]);
                      }
                      msg += "], delta_n_up = " + std::to_string(delta_n_up) +
                             ", momentum-quantisation residual = " +
                             std::to_string(q_residual) + ".";
                      throw std::runtime_error(msg);
                  }
                  if (q_residual > opts.momentum_tolerance) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: Q is incommensurate with the lattice "
                          "(residual = " + std::to_string(q_residual) +
                          " > tolerance = " +
                          std::to_string(opts.momentum_tolerance) + ").");
                  }
                  ed::SectorTag dst_tag =
                      dst_handle.sector_tag(dst_sector_idx);

                  // -----------------------------------------------------
                  // (6) Resolve the target sector operator. An empty
                  //     (dropped) target irrep => spectral function is
                  //     identically zero (the SectorSetView returns nullptr
                  //     for a vanishing sector). Even so the GS solve above
                  //     already populated ``agg.backend.lane`` (Phase H.1)
                  //     so the caller can still see which lane produced the
                  //     zero result.
                  // -----------------------------------------------------
                  ed::symmetry::SectorOperator* dst_sec_view =
                      dst_handle.sector(dst_sector_idx);
                  if (!dst_sec_view || dst_sec_view->dim() == 0) {
                      const std::size_t num_omega =
                          (opts.num_omega > 0) ? opts.num_omega : 1;
                      agg.omega.resize(num_omega);
                      agg.S_real.assign(num_omega, 0.0);
                      agg.S_imag.assign(num_omega, 0.0);
                      const double step = (opts.num_omega > 1)
                          ? (opts.omega_max - opts.omega_min) /
                            static_cast<double>(opts.num_omega - 1)
                          : 0.0;
                      for (std::size_t i = 0; i < num_omega; ++i) {
                          agg.omega[i] = opts.omega_min +
                              static_cast<double>(i) * step;
                      }
                      agg.selection_rule_label =
                          "k_final = k_initial + Q (cross-irrep; "
                          "target sector empty)";
                      ed::SpectralSectorEntry entry;
                      entry.initial = gs_src_tag;
                      entry.final_  = dst_tag;
                      entry.S_real  = agg.S_real;
                      entry.S_imag  = agg.S_imag;
                      agg.per_sector_pair.push_back(std::move(entry));
                      return agg;
                  }

                  // Target observable handle: build phi = O_Q |psi_0> in the
                  // target sector basis via CrossSectorOrbitObservable. Each
                  // OperatorRef wraps exactly ONE sector, so the source /
                  // target sector indices passed to the observable are 0.
                  ed::dssf::CrossSectorOrbitObservable::OperatorRef dst_ref =
                      make_cross_sector_ref(
                          dst_sec_view, static_cast<std::uint64_t>(num_sites));
                  ed::dssf::CrossSectorOrbitObservable orb_obs(
                      src_ref, /*src_sector=*/0,
                      dst_ref, /*dst_sector=*/0,
                      tlist,
                      static_cast<float>(spin_l));
                  const std::size_t dim_dst = orb_obs.dim_dst();
                  if (dst_sec_view->dim() != dim_dst) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_cross_irrep_"
                          "directory: target sector dim mismatch (view="
                          + std::to_string(dst_sec_view->dim())
                          + ", observable=" + std::to_string(dim_dst) + ").");
                  }
                  std::vector<Complex> phi(dim_dst, Complex(0.0, 0.0));
                  orb_obs.apply(psi0.data(), phi.data(), dim_dst);

                  // -----------------------------------------------------
                  // (7) Run cf_spectral_from_vector on H restricted to
                  //     the target sector (the resolved ``dst_sec_view``).
                  // -----------------------------------------------------

                  // Build the frequency grid -- linear spacing between
                  // [omega_min, omega_max]. Matches the convention used
                  // by ed::workflows::spectral for the omega axis.
                  const std::size_t num_omega = std::max<std::size_t>(opts.num_omega, 1);
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

                  ed::observables::CfSpectralOptions cfopts;
                  cfopts.krylov_dim   = opts.krylov_dim;
                  cfopts.broadening   = opts.broadening;
                  cfopts.energy_shift = (std::abs(opts.energy_shift) > 1e-14)
                                          ? opts.energy_shift
                                          : E0;
                  cfopts.tolerance    = 1e-12;
                  cfopts.global_n     = dim_dst;
                  cfopts.verbose      = false;

                  // Phase H.2 of the "Close CPU/GPU Gaps" plan
                  // (May 2026): route the inner CF Lanczos through
                  // ``select_backend`` + ``dst_sec_view->bind<B>()``
                  // so the target-sector Lanczos runs on the same
                  // backend as the source-sector GS solve. The
                  // ``CrossSectorOrbitObservable::apply`` rectangular
                  // scatter (line above this block) stays host-only;
                  // we stage the resulting ``phi`` into backend memory
                  // here as a single D2H/H2D trip and then run the
                  // CF kernel entirely device-resident. The phi-norm
                  // and spectral weight are preserved by
                  // ``cf_spectral_from_vector`` (header-only,
                  // backend-templated).
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

                  // -----------------------------------------------------
                  // (8) Marshal the result into ed::SpectralResult.
                  // -----------------------------------------------------
                  // Phase H.2: refine the propagated lane to reflect
                  // the LANCZOS lane (which now matches the GS lane
                  // by construction since both go through
                  // ``select_backend(dst_sec_view->geometry(),
                  // opts.backend)`` / ``H.geometry()`` with the same
                  // ``opts.backend``). Falling back to the GS lane
                  // captured by Phase H.1 above when the CF variant
                  // didn't override the BackendMetadata.
                  agg.backend.lane =
                      ed::lane_label_from_variant(cf_variant);
                  agg.omega       = cf.frequencies;
                  agg.S_real      = cf.spectral_function;
                  agg.S_imag.assign(cf.spectral_function.size(), 0.0);
                  agg.errors_real.assign(cf.spectral_function.size(), 0.0);
                  agg.errors_imag.assign(cf.spectral_function.size(), 0.0);

                  std::string label =
                      "k_final = k_initial + Q (cross-irrep); "
                      "src qn = [";
                  for (std::size_t g = 0; g < gs_src_tag.quantum_numbers.size(); ++g) {
                      if (g) label += ",";
                      label += std::to_string(gs_src_tag.quantum_numbers[g]);
                  }
                  label += "], dst qn = [";
                  for (std::size_t g = 0; g < dst_tag.quantum_numbers.size(); ++g) {
                      if (g) label += ",";
                      label += std::to_string(dst_tag.quantum_numbers[g]);
                  }
                  label += "], delta_n_up = " + std::to_string(delta_n_up) +
                           ", ||phi||^2 = " +
                           std::to_string(cf.phi_norm * cf.phi_norm);
                  agg.selection_rule_label = std::move(label);

                  ed::SpectralSectorEntry entry;
                  entry.initial = gs_src_tag;
                  entry.final_  = dst_tag;
                  entry.S_real  = agg.S_real;
                  entry.S_imag  = agg.S_imag;
                  agg.per_sector_pair.push_back(std::move(entry));
              }
              return agg;
          };
    m.def("workflows_spectral_streaming_symmetry_cross_irrep_directory",
          [spectral_cross_irrep_body](
              const std::string&                    directory,
              std::uint64_t                          num_sites,
              double                                 spin_l,
              const std::vector<py::tuple>&          observable_transforms,
              ed::workflows::SpectralOptions         opts,
              py::object                             fixed_sz_n_up,
              int                                    delta_n_up,
              int                                    sz_parity,
              bool                                   flip_sectors) {
              return spectral_cross_irrep_body(
                  ed::DirectoryPath{directory}, num_sites, spin_l,
                  observable_transforms, std::move(opts),
                  std::move(fixed_sz_n_up), delta_n_up, sz_parity,
                  flip_sectors);
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")                = 0.5,
          py::arg("observable_transforms") = std::vector<py::tuple>{},
          py::arg("opts")                  = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up")         = py::none(),
          py::arg("delta_n_up")            = 0,
          py::arg("sz_parity")             = -1,
          py::arg("flip_sectors")          = false,
          R"pbdoc(
        Cross-irrep streaming-symmetry spectral workflow (SOTA).

        Implements the dynamical S(Q, omega) Lehmann sum with full
        spatial-symmetry exploitation:

          S(Q, omega) = sum_n delta(omega - E_n + E_0) * |<n|O_Q|0>|^2

        Source ground state is solved per-irrep, the target sector
        is resolved via the selection rule
        ``k_final = k_initial + Q``, the user-supplied observable
        terms scatter |psi_0> into the target orbit basis, and a
        continued-fraction Lanczos run on H restricted to that
        sector yields the spectral function.

        Parameters
        ----------
        directory : str
            Hamiltonian directory (must contain
            ``automorphism_results/``).
        num_sites : int
            Number of lattice sites.
        spin_l : float, optional
            Local spin (0.5 by default).
        observable_transforms : list of (op_type:int, site:int,
                                         coeff:complex,
                                         is_two_body:bool,
                                         op_type_2:int, site_2:int)
            One row per term in the probe observable ``O_Q``. The
            tuple layout mirrors ``Operator::TransformData`` -- the
            Python wrapper ``qed.spectral`` extracts these from an
            ``ed.Operator`` instance automatically.
        opts : SpectralOptions, optional
            CF / FTLM knobs. ``momentum_transfer`` (fractional
            reciprocal-lattice units) is the selection-rule shift Q,
            ``momentum_tolerance`` controls commensurability.
        fixed_sz_n_up : int or None, optional
            Source-sector ``n_up``. Required when ``delta_n_up != 0``.
        delta_n_up : int, optional
            Change in ``n_up`` produced by the observable (0 for
            Sz-conserving probes, +1 for S+, -1 for S-). When non-zero
            the workflow builds a second streaming operator for the
            target subspace.

        Returns
        -------
        SpectralResult
            ``omega`` / ``S_real`` carry the cross-irrep spectral
            function (S_imag is zero in the current build).
            ``per_sector_pair`` records the (initial, final)
            SectorTag pair; ``selection_rule_label`` documents the
            resolved transition.
    )pbdoc");
    m.def("workflows_spectral_streaming_symmetry_cross_irrep",
          [spectral_cross_irrep_body](
              const Operator&                        H,
              const py::dict&                        group,
              std::uint64_t                          num_sites,
              double                                 spin_l,
              const std::vector<py::tuple>&          observable_transforms,
              ed::workflows::SpectralOptions         opts,
              py::object                             fixed_sz_n_up,
              int                                    delta_n_up,
              int                                    sz_parity,
              bool                                   flip_sectors) {
              return spectral_cross_irrep_body(
                  in_memory_symmetric_source(
                      H, group,
                      "workflows_spectral_streaming_symmetry_cross_irrep"),
                  num_sites, spin_l, observable_transforms, std::move(opts),
                  std::move(fixed_sz_n_up), delta_n_up, sz_parity,
                  flip_sectors);
          },
          py::arg("H"),
          py::arg("group"),
          py::arg("num_sites"),
          py::arg("spin_l")                = 0.5,
          py::arg("observable_transforms") = std::vector<py::tuple>{},
          py::arg("opts")                  = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up")         = py::none(),
          py::arg("delta_n_up")            = 0,
          py::arg("sz_parity")             = -1,
          py::arg("flip_sectors")          = false,
          R"pbdoc(
        In-memory twin of
        ``workflows_spectral_streaming_symmetry_cross_irrep_directory``.

        Takes the Hamiltonian ``H`` (its terms are copied) and the group
        info dict the directory writer consumes in place of the
        directory; every other argument and the result are identical.
    )pbdoc");

}
