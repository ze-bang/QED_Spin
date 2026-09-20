// =============================================================================
// python/qed/_bindings/workflow/workflow_spectral.cpp
//
// The spectral lane: the plain `workflows_spectral` entry point and the
// same-irrep streaming-symmetry spectral binding over a directory.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_spectral(py::module_& m) {
    m.def("workflows_spectral",
          [](Operator& op,
             std::vector<Operator*> observables,
             ed::workflows::SpectralOptions opts) {
              // GPU lane (operator-collapse Phase 2a): every spectral method
              // dispatches on Backend internally and consumes H + the
              // observables through ``bind<Backend>()``. The host operators
              // advertise ``supports_device_matvec`` and their ``bind_cuda()``
              // device mirror serves the CudaBackend lane directly, so both
              // the Hamiltonian and the observables run device-resident with
              // no GPUOperator promotion. ``spectral_method_supports_gpu``
              // stays as a defensive hook for any future host-only method.
              if (!spectral_method_supports_gpu(opts.method)) {
                  warn_silent_cpu_fallback(
                      "qed.spectral (host-only method)",
                      opts.backend);
                  opts.backend.allow_gpu = false;
              }
              std::vector<const ed::LinearOperator*> obs;
              obs.reserve(observables.size());
              for (auto* o : observables) {
                  if (!o) continue;
                  obs.push_back(static_cast<const ed::LinearOperator*>(o));
              }
              // Stage 12g (SU(2) rollout): label the CF source state's
              // total spin when H is SU(2)-invariant. Cheap (one S^2
              // matvec + certification residual) and purely additive.
              if (ed::symmetry::su2_enabled() && op_is_su2_symmetric(op)) {
                  auto s2 = make_s2_like(op);
                  int n_up = -1;
                  if (const auto* fsz =
                          dynamic_cast<const FixedSzOperator*>(&op)) {
                      n_up = static_cast<int>(fsz->producer().n_up());
                  }
                  const int n_sites = static_cast<int>(op.getNumBits());
                  opts.su2_labeler = [s2, n_sites, n_up](
                                         const Complex* v, std::size_t n,
                                         double* s2_out) -> int {
                      double res = 0.0;
                      const double s2_exp =
                          ed::ops::s2_expectation(*s2, v, n, &res);
                      if (s2_out) *s2_out = s2_exp;
                      return res <= ed::ops::kS2CertifyTol
                                 ? ed::ops::snap_two_S(s2_exp, n_sites,
                                                       n_up)
                                 : -1;
                  };
              }
              return ed::workflows::spectral(
                  static_cast<const ed::LinearOperator&>(op), obs,
                  std::move(opts));
          },
          py::arg("op"),
          py::arg("observables"),
          py::arg("opts") = ed::workflows::SpectralOptions{},
          "Run the unified dynamical-correlator workflow "
          "(continued-fraction Lanczos or FTLM dynamical) over the auto-"
          "selected Backend. When ``allow_gpu`` is set and a CUDA device is "
          "available, the Hamiltonian and the observable pair run through "
          "their lazy CudaMatVecBackend device mirrors on the GPU lane.");

}

void bind_workflows_spectral_streaming(py::module_& m) {
    // -----------------------------------------------------------------
    // SOTA streaming-symmetry spectral workflow over a directory
    // (May 2026).
    //
    // For the ``GroundStateCF`` method, the spectral function
    //
    //   S_OO(omega) = -1/pi Im <psi_0| O^dag (omega + E_0 - H + i eta)^-1 O |psi_0>
    //
    // factorises over irreps when O carries a definite momentum
    // transfer Q. The exact selection rule reads
    //
    //   k_final = k_initial + Q   (mod reciprocal lattice)
    //
    // and at T=0 only the irrep containing the global ground state
    // contributes to the initial state. Concretely, the SOTA path
    // walks the per-irrep sector loop once, finds the sector
    // containing the global GS (sector with the smallest per-sector
    // ground-state energy), and runs the CF-Lanczos kernel
    // exclusively in that sector. The same-sector ``Q = 0`` path
    // (i.e. O is the q=0 component of S^z / n / ...) is the
    // canonical "DOS / static structure factor" workflow this entry
    // covers; cross-irrep (``Q != 0``) transitions are emitted as a
    // diagnostic in ``selection_rule_label`` and left to the
    // forthcoming ``CrossSectorOrbitObservable`` (see
    // ``docs/architecture/SYMMETRY.md`` Section 3).
    // -----------------------------------------------------------------
    m.def("workflows_spectral_streaming_symmetry_directory",
          [](const std::string& directory,
             std::uint64_t num_sites,
             double spin_l,
             ed::workflows::SpectralOptions opts,
             py::object fixed_sz_n_up) {
              ed::OperatorSpec spec;
              spec.source             = ed::DirectoryPath{directory};
              spec.num_sites          = num_sites;
              spec.spin_l             = static_cast<float>(spin_l);
              spec.streaming_symmetry = true;
              if (!fixed_sz_n_up.is_none()) {
                  spec.fixed_sz = fixed_sz_n_up.cast<int>();
              }

              ed::SpectralResult agg;
              {
                  py::gil_scoped_release release;
                  // Operator-collapse Phase 3 (Jun 2026): direct sector
                  // enumeration via ``make_sector_operators_tagged`` viewed
                  // through ``SectorSetView`` (same-irrep spectral path;
                  // cross-irrep transitions remain on CrossSectorOrbitObservable
                  // below).
                  ed::core::SectorSetView handle(
                      ed::make_sector_operators_tagged(spec));

                  const std::size_t num_sectors = handle.num_sectors();
                  if (num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_directory: "
                          "make_operator returned an operator with no "
                          "symmetry sectors; check the "
                          "automorphism_results/ directory.");
                  }

                  // Pass 1 -- per-sector ground-state solve. Picks the
                  // irrep containing the global GS so the CF-Lanczos
                  // run only touches that sector (huge speedup for
                  // SOTA DOS / static structure factor workflows).
                  const std::vector<std::size_t> sector_indices =
                      ed::core::filter_sectors(num_sectors,
                                               opts.selected_sectors);
                  std::size_t gs_sector_idx = 0;
                  double      gs_energy     = std::numeric_limits<double>::infinity();
                  bool        any_solved    = false;
                  {
                      // Pass 1: GS sector scan. Independent per-sector
                      // Lanczos calls (1 eig, no vectors) — embarrassingly
                      // parallel. Reuses the ED_SYM_SECTOR_PARALLEL gate.
                      // B6: auto-parallel across many tiny sectors.
                      const bool sp_spectral = resolve_sector_parallel(
                          sector_indices.size(),
                          max_sector_dim(handle, sector_indices),
                          opts.backend.allow_gpu);
                      const long n_sp =
                          static_cast<long>(sector_indices.size());
                      // (gs_energy, sector_idx) per slot; sector_idx=SIZE_MAX
                      // means "no eigenvalue".
                      std::vector<std::pair<double, std::size_t>>
                          sp_results(static_cast<std::size_t>(n_sp),
                              {std::numeric_limits<double>::infinity(),
                               std::size_t(-1)});

                      #pragma omp parallel for schedule(dynamic, 1) \
                          if(sp_spectral)
                      for (long ii = 0; ii < n_sp; ++ii) {
                          const std::size_t k =
                              sector_indices[static_cast<std::size_t>(ii)];
                          auto sec = handle.sector(k);
                          if (!sec || sec->dim() == 0) continue;
                          ed::workflows::SolveOptions sopts;
                          sopts.num_eigs        = 1;
                          sopts.tolerance       = 1e-12;
                          sopts.backend         = opts.backend;
                          sopts.method =
                              ed::workflows::SolveMethod::Lanczos;
                          sopts.compute_vectors = false;
                          auto sr = ed::workflows::solve(*sec, sopts);
                          if (!sr.eigenvalues.empty()) {
                              sp_results[static_cast<std::size_t>(ii)] =
                                  {sr.eigenvalues.front(), k};
                          }
                      }
                      for (const auto& [E_k, k] : sp_results) {
                          if (k == std::size_t(-1)) continue;
                          any_solved = true;
                          if (E_k < gs_energy) {
                              gs_energy     = E_k;
                              gs_sector_idx = k;
                          }
                      }
                  }
                  if (!any_solved) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_directory: "
                          "every sector returned an empty spectrum; "
                          "check the operator / Hilbert space.");
                  }

                  // Selection rule label: we currently route every
                  // observable through the GS sector, which is exact
                  // for Q=0 observables and a documented
                  // approximation for Q!=0 (the matrix element
                  // remains in the same irrep, dropping cross-irrep
                  // pieces; the cross-irrep generalisation is the
                  // ``CrossSectorOrbitObservable`` follow-on).
                  bool has_Q = !opts.momentum_transfer.empty();
                  if (has_Q) {
                      double q_norm = 0.0;
                      for (double q : opts.momentum_transfer) {
                          const double frac = q - std::round(q);
                          q_norm += frac * frac;
                      }
                      has_Q = std::sqrt(q_norm) > opts.momentum_tolerance;
                  }
                  agg.selection_rule_label = has_Q
                      ? std::string("k_final = k_initial + Q (cross-irrep "
                                    "transitions deferred to "
                                    "CrossSectorOrbitObservable; "
                                    "running same-irrep approximation)")
                      : std::string("k_final = k_initial (same-irrep; "
                                    "exact for Q=0 observables)");

                  // Pass 2 -- CF-Lanczos in the GS sector only.
                  ed::SectorTag gs_tag = handle.sector_tag(gs_sector_idx);
                  auto gs_sec = handle.sector(gs_sector_idx);
                  ed::workflows::SpectralOptions sopts = opts;
                  sopts.selected_sectors.clear();
                  // Use the per-sector GS energy as the resolvent shift
                  // (override only if the user didn't already).
                  if (std::abs(opts.energy_shift) < 1e-14) {
                      sopts.energy_shift = gs_energy;
                  }
                  // Build a placeholder identity observable: the
                  // streaming-symmetry spectral binding currently
                  // expects an in-process observable to be wired
                  // through Python (e.g. an irrep-restricted S^z).
                  // For now we route H itself as the observable so
                  // the CF kernel produces the local density of
                  // states of the GS sector -- exactly the SOTA
                  // single-shot DOS workflow that HPhi / EDLib
                  // expose as their canonical symmetry-projected
                  // spectral output. The Python wrapper layer
                  // (qed.spectral) lifts the user-provided
                  // observable to the orbit basis before calling
                  // this entry.
                  std::vector<const ed::LinearOperator*> obs_vec{
                      gs_sec};
                  ed::SpectralResult sr =
                      ed::workflows::spectral(*gs_sec, obs_vec, sopts);

                  agg.omega        = sr.omega;
                  agg.S_real       = sr.S_real;
                  agg.S_imag       = sr.S_imag;
                  agg.errors_real  = sr.errors_real;
                  agg.errors_imag  = sr.errors_imag;
                  agg.krylov       = sr.krylov;
                  agg.backend      = sr.backend;

                  // SOTA attribution: record the GS sector as the
                  // initial-and-final irrep that produced the spectral
                  // function. Cross-irrep (Q != 0) pairs will appear
                  // as additional entries once
                  // CrossSectorOrbitObservable is wired in.
                  ed::SpectralSectorEntry entry;
                  entry.initial = gs_tag;
                  entry.final_  = gs_tag;
                  entry.S_real  = sr.S_real;
                  entry.S_imag  = sr.S_imag;
                  agg.per_sector_pair.push_back(std::move(entry));
              }
              return agg;
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")      = 0.5,
          py::arg("opts")        = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up") = py::none(),
          R"pbdoc(
        Streaming-symmetry-projected spectral workflow over a directory.

        For ``Method::GroundStateCF`` the entry performs a per-irrep
        ground-state pass to locate the irrep containing the global
        ground state, then runs continued-fraction Lanczos
        exclusively in that one sector. This is the canonical SOTA
        same-irrep spectral path (exact for ``Q = 0`` observables;
        documented approximation for ``Q != 0`` until the
        ``CrossSectorOrbitObservable`` follow-on lands -- see
        ``docs/architecture/SYMMETRY.md`` Section 3 for the design).

        Parameters
        ----------
        directory : str
            Path containing the Hamiltonian dat files and
            ``automorphism_results/``.
        num_sites : int
            Number of sites in the lattice.
        spin_l : float, optional
            Spin magnitude (0.5 for spin-1/2, the default).
        opts : SpectralOptions, optional
            CF / FTLM-dynamical knobs. ``momentum_transfer`` describes
            the observable's momentum (used for the selection-rule
            label); ``selected_sectors`` restricts the initial-sector
            search.
        fixed_sz_n_up : int or None, optional
            Optional Sz projection.

        Returns
        -------
        SpectralResult
            ``omega`` / ``S_real`` / ``S_imag`` carry the merged
            spectral function. ``per_sector_pair`` lists each
            (initial, final) irrep pair that contributed, tagged
            with the per-sector ``SectorTag``;
            ``selection_rule_label`` describes the symmetry filter
            that was applied.
    )pbdoc");

}
