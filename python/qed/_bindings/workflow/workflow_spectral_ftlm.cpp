// =============================================================================
// python/qed/_bindings/workflow/workflow_spectral_ftlm.cpp
//
// The finite-T (FTLM) spectral bindings: `workflows_spectral_ftlm_plain`
// and the cross-irrep FTLM pair
// (`workflows_spectral_streaming_symmetry_ftlm_cross_irrep[_directory]`),
// which share one body lambda.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_spectral_ftlm(py::module_& m) {
    // -----------------------------------------------------------------
    // SOTA FINITE-T cross-irrep dynamical spectral binding (FTLM).
    //
    // Sister entry point to the GS cross-irrep binding above. The
    // selection rule, observable transforms, and (src, dst) operator
    // pair are resolved identically, but instead of a single GS
    // solve we loop over ALL source sectors and run an FTLM-style
    // multi-sample / multi-temperature dynamical kernel per
    // (k_src, k_dst) pair, then Z-weight-recombine across source
    // sectors. This closes the DYNAMICAL_THERMAL spatial-irrep gap.
    //
    // Math (per source sector k_src, target k_dst = k_src + Q):
    //
    //   S_{k_src}(omega, T) = (dim_src / R)
    //       sum_r sum_m e^{-beta E_m} |c_m|^2 * S_m(omega)
    //
    // where S_m(omega) = ||phi_m||^2 sum_n V_S[0,n]^2 *
    //                       Lorentzian(omega - (lambda_n - E_m))
    // and ``phi_m = O_Q |m^{(r)}>`` is the cross-irrep scatter of
    // the m-th outer Ritz state into the target orbit basis.
    //
    // Aggregation:
    //
    //   S_total(omega, T) = sum_{k_src} S_{k_src}(omega, T)
    //                       / sum_{k_src} Z_{k_src}(T).
    //
    // Returns a SpectralResult shaped exactly like the GS cross-irrep
    // path, with one extra omega-axis copy per temperature exposed
    // via ``S_by_T_real[T]`` / ``S_by_T_imag[T]`` (carried inside
    // ``per_sector_pair`` for now; the GS entry sits at index 0 for
    // backwards compat with consumers that read agg.S_real).
    // -----------------------------------------------------------------
    // 2026-09-11: finite-T dynamical spectra WITHOUT spatial symmetry. Source
    // and target blocks are the operator itself (it may be a FixedSzOperator
    // when the probe conserves Sz); the estimator is the verified
    // ftlm_cross_irrep_kernel_one_sector, so the plain lane no longer refuses.
    m.def("workflows_spectral_ftlm_plain",
          [](const Operator& op, const Operator& obs,
             std::vector<double> temperatures, std::vector<double> omega,
             double eta, int num_samples, int krylov_dim, std::uint64_t seed) {
              const std::size_t dim = static_cast<std::size_t>(op.dim());
              if (static_cast<std::size_t>(obs.dim()) != dim)
                  throw std::invalid_argument("workflows_spectral_ftlm_plain: observable and "
                                              "Hamiltonian act on different spaces");
              for (double T : temperatures)
                  if (!(T > 0.0)) throw std::invalid_argument("temperatures must be > 0");
              auto Hmv = op.bind_cpu();
              auto Omv = obs.bind_cpu();
              auto apply_H = [&Hmv](const Complex* x, Complex* y, int n) { Hmv(x, y, static_cast<std::size_t>(n)); };
              auto apply_O = [&Omv](const Complex* x, Complex* y, int n) { Omv(x, y, static_cast<std::size_t>(n)); };
              ed::observables::FtlmCrossIrrepOptions kopts;
              kopts.krylov_dim  = static_cast<std::size_t>(std::max(krylov_dim, 2));
              kopts.num_samples = static_cast<std::size_t>(std::max(num_samples, 1));
              kopts.broadening  = eta;
              kopts.tolerance   = 1e-12;
              kopts.random_seed = seed;
              kopts.verbose     = false;
              kopts.full_reorthogonalization = true;   // 2026-09-11: ghosts on degenerate spectra biased the trace by 15 % (XY chain)
              kopts.reorth_frequency = 1;
              std::vector<ed::observables::FtlmCrossIrrepSectorResult> secs;
              {
                  py::gil_scoped_release release;
                  secs.push_back(ed::observables::ftlm_cross_irrep_kernel_one_sector(
                      apply_H, apply_H, apply_O, dim, dim, temperatures, omega, kopts));
              }
              auto merged = ed::observables::combine_sector_dynamical_spectra(
                  secs, temperatures, omega.size());
              py::dict out;
              out["omega"] = omega;
              out["temperatures"] = temperatures;
              py::list S;
              for (double T : temperatures) S.append(merged.S_real[T]);
              out["S_real"] = S;
              return out;
          },
          py::arg("op"), py::arg("observable"), py::arg("temperatures"), py::arg("omega"),
          py::arg("eta"), py::arg("num_samples") = 30, py::arg("krylov_dim") = 100,
          py::arg("seed") = 0,
          "Finite-temperature S(omega, T) of one observable on the operator's own block "
          "via the FTLM cross-irrep estimator with source = target = this block.");

    // WP9: one body for the directory binding and its in-memory twin
    // ``workflows_spectral_streaming_symmetry_ftlm_cross_irrep``; the
    // shifted-Sz target set is seeded from the SAME source.
    const auto spectral_ftlm_cross_irrep_body =
          [](const SymmetricSource&                source,
             std::uint64_t                          num_sites,
             double                                 spin_l,
             const std::vector<py::tuple>&          observable_transforms,
             ed::workflows::SpectralOptions         opts,
             py::object                             fixed_sz_n_up,
             int                                    delta_n_up,
             std::vector<double>                    temperatures,
             std::uint64_t                          num_samples,
             std::uint64_t                          random_seed,
             int                                    sz_parity,
             bool                                   flip_sectors) {
              // ----------------------------------------------------------
              // Same observable-transform decoder as the GS path.
              // ----------------------------------------------------------
              std::vector<Operator::TransformData> tlist =
                  decode_probe_transforms(
                      observable_transforms,
                      "workflows_spectral_streaming_symmetry_ftlm_cross_"
                      "irrep_directory");
              if (tlist.empty()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_ftlm_cross_"
                      "irrep_directory: observable_transforms is empty.");
              }
              if (temperatures.empty()) {
                  throw std::invalid_argument(
                      "workflows_spectral_streaming_symmetry_ftlm_cross_"
                      "irrep_directory: temperatures is empty.");
              }

              // Decode Python arguments BEFORE dropping the GIL.
              const std::optional<int> fixed_sz_opt = decode_optional_n_up(fixed_sz_n_up);
              ed::SpectralResult agg;
              {
                  py::gil_scoped_release release;

                  // -----------------------------------------------
                  // (1) Build source streaming operator.
                  // -----------------------------------------------
                  ed::OperatorSpec src_spec = make_cross_irrep_src_spec(
                      source, num_sites, spin_l, fixed_sz_opt,
                      sz_parity, flip_sectors);
                  const SlottedSelection slots = slotted_selection_for(
                      src_spec, tlist,
                      "workflows_spectral_streaming_symmetry_ftlm_cross_"
                      "irrep_directory");
                  // Operator-collapse Phase 3 (Jun 2026): direct sector
                  // enumeration via ``make_sector_operators_tagged`` +
                  // ``SectorSetView``. ``src_ref`` / ``dst_ref`` are built
                  // per (k_src, k_dst) pair inside the loop (Stage 8d:
                  // CSR-free rep refs on the rep-lazy lane).
                  ed::core::SectorSetView src_handle(
                      ed::make_sector_operators_tagged(src_spec));

                  const std::size_t src_num_sectors = src_handle.num_sectors();
                  if (src_num_sectors == 0) {
                      throw std::runtime_error(
                          "workflows_spectral_streaming_symmetry_ftlm_cross_"
                          "irrep_directory: source operator has no symmetry "
                          "sectors; check automorphism_results/.");
                  }

                  // -----------------------------------------------
                  // (2) Build / re-use target operator + ref.
                  // -----------------------------------------------
                  ed::core::SectorSetView dst_handle = src_handle;
                  if (delta_n_up != 0) {
                      if (!src_spec.fixed_sz.has_value()) {
                          throw std::invalid_argument(
                              "workflows_spectral_streaming_symmetry_ftlm_"
                              "cross_irrep_directory: delta_n_up != 0 "
                              "requires fixed_sz_n_up to be set.");
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

                  // -----------------------------------------------
                  // (3) Frequency grid (same convention as GS path).
                  // -----------------------------------------------
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
                          omega_grid[i] = opts.omega_min +
                              static_cast<double>(i) * step;
                      }
                  }

                  // -----------------------------------------------
                  // (4) Filter source sectors via selected_sectors.
                  // -----------------------------------------------
                  const std::vector<std::size_t> src_sector_indices =
                      ed::core::filter_sectors(src_num_sectors,
                                               opts.selected_sectors);

                  // -----------------------------------------------
                  // (5) For each source sector, resolve target,
                  //     build cross-irrep observable, and run the
                  //     FTLM cross-irrep kernel. Accumulate per-
                  //     sector results in a vector for the combiner.
                  // -----------------------------------------------
                  ed::matvec::CpuBackend cpu_be;
                  std::vector<ed::observables::FtlmCrossIrrepSectorResult>
                      sector_results;
                  std::vector<std::pair<ed::SectorTag, ed::SectorTag>>
                      sector_tags;
                  sector_results.reserve(src_sector_indices.size());
                  sector_tags.reserve(src_sector_indices.size());

                  for (std::size_t k_src : src_sector_indices) {
                      ed::symmetry::SectorOperator* src_view =
                          src_handle.sector(k_src);
                      if (!src_view || src_view->dim() == 0) continue;

                      double q_residual = 0.0;
                      const std::size_t k_dst =
                          ed::core::resolve_target_sector_slotted(
                              dst_handle,
                              k_src,
                              opts.momentum_transfer,
                              slots.n_slots,
                              slots.signs,
                              &q_residual);
                      if (k_dst == ed::core::kSectorNotFound) continue;
                      if (q_residual > opts.momentum_tolerance) {
                          throw std::runtime_error(
                              "workflows_spectral_streaming_symmetry_ftlm_"
                              "cross_irrep_directory: Q incommensurate with "
                              "the lattice (residual = " +
                              std::to_string(q_residual) +
                              " > tolerance = " +
                              std::to_string(opts.momentum_tolerance) + ").");
                      }
                      ed::symmetry::SectorOperator* dst_view =
                          dst_handle.sector(k_dst);
                      if (!dst_view || dst_view->dim() == 0) continue;

                      // Cross-irrep rectangular observable for THIS
                      // (k_src, k_dst) pair. Each OperatorRef wraps one
                      // sector (src_sector == dst_sector == 0); Stage 8d:
                      // rep-lazy / flip / parity sectors ride the CSR-free
                      // RepSectorData ref.
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef src_ref =
                          make_cross_sector_ref(
                              src_view, static_cast<std::uint64_t>(num_sites));
                      ed::dssf::CrossSectorOrbitObservable::OperatorRef dst_ref =
                          make_cross_sector_ref(
                              dst_view, static_cast<std::uint64_t>(num_sites));
                      ed::dssf::CrossSectorOrbitObservable orb_obs(
                          src_ref, /*src_sector=*/0,
                          dst_ref, /*dst_sector=*/0,
                          tlist,
                          static_cast<float>(spin_l));
                      const std::size_t dim_src = src_view->dim();
                      const std::size_t dim_dst = orb_obs.dim_dst();
                      if (dim_dst != dst_view->dim()) {
                          throw std::runtime_error(
                              "workflows_spectral_streaming_symmetry_ftlm_"
                              "cross_irrep_directory: dim_dst mismatch "
                              "(observable=" + std::to_string(dim_dst) +
                              ", view=" + std::to_string(dst_view->dim()) +
                              ").");
                      }

                      // Three lambdas: H_src, H_dst, O.
                      auto apply_H_src = [&src_view](const Complex* x,
                                                     Complex*       y,
                                                     int            n) {
                          src_view->apply(x, y,
                              static_cast<std::size_t>(n));
                      };
                      auto apply_H_dst = [&dst_view](const Complex* x,
                                                     Complex*       y,
                                                     int            n) {
                          dst_view->apply(x, y,
                              static_cast<std::size_t>(n));
                      };
                      auto apply_O = [&orb_obs](const Complex* x,
                                                Complex*       y,
                                                int            n) {
                          orb_obs.apply(x, y,
                              static_cast<std::size_t>(n));
                      };

                      ed::observables::FtlmCrossIrrepOptions kopts;
                      kopts.krylov_dim       = opts.krylov_dim;
                      kopts.num_samples      = num_samples;
                      kopts.broadening       = opts.broadening;
                      kopts.tolerance        = 1e-12;
                      kopts.random_seed      = random_seed;
                      kopts.verbose          = false;
                      kopts.full_reorthogonalization = true;   // 2026-09-11: ghosts on degenerate spectra biased the trace by 15 % (XY chain)
                      kopts.reorth_frequency = 1;

                      auto sec_res =
                          ed::observables::ftlm_cross_irrep_kernel_one_sector(
                              apply_H_src, apply_H_dst, apply_O,
                              dim_src, dim_dst,
                              temperatures, omega_grid, kopts);

                      sector_results.push_back(std::move(sec_res));
                      sector_tags.emplace_back(
                          src_handle.sector_tag(k_src),
                          dst_handle.sector_tag(k_dst));
                  }

                  // -----------------------------------------------
                  // (6) Z-weighted recombine across source sectors.
                  // -----------------------------------------------
                  auto merged = ed::observables::combine_sector_dynamical_spectra(
                      sector_results, temperatures, num_omega);

                  // -----------------------------------------------
                  // (7) Marshal into ed::SpectralResult. The aggregate
                  //     ``S_real`` / ``S_imag`` arrays carry the
                  //     first temperature; the full T-resolved data
                  //     is exposed via ``per_sector_pair`` entries
                  //     whose initial/final SectorTag pair and
                  //     selection-rule label record EACH source
                  //     sector's contribution along with the
                  //     temperatures encoded into the label.
                  // -----------------------------------------------
                  agg.omega = omega_grid;
                  const double T_primary = temperatures.front();
                  agg.S_real = merged.S_real[T_primary];
                  agg.S_imag = merged.S_imag[T_primary];
                  agg.errors_real.assign(num_omega, 0.0);
                  agg.errors_imag.assign(num_omega, 0.0);

                  // Phase H.1 of the "Close CPU/GPU Gaps" plan
                  // (May 2026): surface the truthful backend lane on
                  // the aggregate. The FTLM cross-irrep kernel
                  // (``ftlm_cross_irrep_kernel_one_sector``) is
                  // host-only -- it consumes raw ``apply_H_src/dst/O``
                  // lambdas and runs the inner Lanczos / observable
                  // contraction on the host. GPU rectangular scatter
                  // for ``CrossSectorOrbitObservable`` is tracked as
                  // a deferred follow-up. Until that lands, this
                  // binding always reports ``lane='cpu'``.
                  agg.backend.lane     = "cpu";
                  agg.backend.mpi_size = 1;

                  std::string label =
                      "k_final = k_initial + Q (cross-irrep, FTLM, "
                      "finite-T); source-sectors swept = " +
                      std::to_string(sector_results.size()) +
                      "; T = [";
                  for (std::size_t i = 0; i < temperatures.size(); ++i) {
                      if (i) label += ", ";
                      label += std::to_string(temperatures[i]);
                  }
                  label += "]; num_samples = " + std::to_string(num_samples);
                  agg.selection_rule_label = std::move(label);

                  // Per-sector pair entries: one per (k_src, k_dst)
                  // pair, holding that sector's PRIMARY-T S(omega).
                  // For the multi-T payload, also expose
                  // ``S_by_T_real`` / ``S_by_T_imag`` (full grid),
                  // packed into the entry's S_real / S_imag fields
                  // when there is only one sector. For now we ship
                  // the first-T slice in S_real / S_imag and surface
                  // all temperatures through the multi-T payload
                  // map on the agg structure.
                  for (std::size_t i = 0; i < sector_results.size(); ++i) {
                      ed::SpectralSectorEntry entry;
                      entry.initial = sector_tags[i].first;
                      entry.final_  = sector_tags[i].second;
                      auto sr = sector_results[i].S_real.find(T_primary);
                      auto si = sector_results[i].S_imag.find(T_primary);
                      auto sz = sector_results[i].Z.find(T_primary);
                      if (sr != sector_results[i].S_real.end()) {
                          double Z_local = (sz != sector_results[i].Z.end() && sz->second > 1e-300)
                                            ? sz->second : 1.0;
                          entry.S_real.resize(num_omega);
                          entry.S_imag.resize(num_omega);
                          for (std::size_t iw = 0; iw < num_omega; ++iw) {
                              entry.S_real[iw] = sr->second[iw] / Z_local;
                              entry.S_imag[iw] = (si != sector_results[i].S_imag.end())
                                                  ? si->second[iw] / Z_local
                                                  : 0.0;
                          }
                      }
                      agg.per_sector_pair.push_back(std::move(entry));
                  }

                  // Multi-T payload for the Python wrapper to read:
                  // pack the full {T: S(omega)} map by appending one
                  // synthetic SpectralSectorEntry per temperature
                  // with a sentinel label. The Python wrapper
                  // distinguishes them by their `selection_rule_label`
                  // and the index offset. (The simple, structured
                  // alternative -- adding a new C++ field to
                  // SpectralResult -- is intentionally avoided here
                  // to keep this PR strictly additive.)
                  for (double T : temperatures) {
                      ed::SpectralSectorEntry mt;
                      ed::SectorTag dummy;
                      dummy.sector_index = 0;
                      dummy.sector_dim   = 0;
                      mt.initial = dummy;
                      mt.final_  = dummy;
                      mt.S_real  = merged.S_real[T];
                      mt.S_imag  = merged.S_imag[T];
                      agg.per_sector_pair.push_back(std::move(mt));
                  }
              }
              return agg;
          };
    m.def("workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory",
          [spectral_ftlm_cross_irrep_body](
              const std::string&                    directory,
              std::uint64_t                          num_sites,
              double                                 spin_l,
              const std::vector<py::tuple>&          observable_transforms,
              ed::workflows::SpectralOptions         opts,
              py::object                             fixed_sz_n_up,
              int                                    delta_n_up,
              std::vector<double>                    temperatures,
              std::uint64_t                          num_samples,
              std::uint64_t                          random_seed,
              int                                    sz_parity,
              bool                                   flip_sectors) {
              return spectral_ftlm_cross_irrep_body(
                  ed::DirectoryPath{directory}, num_sites, spin_l,
                  observable_transforms, std::move(opts),
                  std::move(fixed_sz_n_up), delta_n_up,
                  std::move(temperatures), num_samples, random_seed,
                  sz_parity, flip_sectors);
          },
          py::arg("directory"),
          py::arg("num_sites"),
          py::arg("spin_l")                = 0.5,
          py::arg("observable_transforms") = std::vector<py::tuple>{},
          py::arg("opts")                  = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up")         = py::none(),
          py::arg("delta_n_up")            = 0,
          py::arg("temperatures")          = std::vector<double>{},
          py::arg("num_samples")           = std::uint64_t{30},
          py::arg("random_seed")           = std::uint64_t{0},
          py::arg("sz_parity")             = -1,
          py::arg("flip_sectors")          = false,
          R"pbdoc(
        Cross-irrep streaming-symmetry **finite-T** spectral workflow
        (SOTA FTLM).

        Computes S(Q, omega, T) for a user-supplied probe observable
        ``O_Q`` whose lattice symmetry character implies a sector
        transition

            k_final = k_initial + Q

        via the streaming-symmetry selection rule (and an optional
        Sz shift ``delta_n_up``). The implementation is a per-source-
        sector Finite-Temperature Lanczos Method (FTLM): for each
        source sector ``k_src`` we draw ``num_samples`` Gaussian
        random vectors in the source orbit basis, build an outer
        Lanczos basis on ``H`` restricted to ``k_src``, reconstruct
        each Ritz state, scatter it into ``k_dst`` via the
        rectangular ``CrossSectorOrbitObservable``, run a second,
        target-sector Lanczos starting from ``phi = O_Q |m>``, and
        accumulate a Lorentzian-broadened Lehmann sum weighted by
        ``exp(-beta * E_m) * |c_m|^2``. Sectors are recombined via
        the F-shifted Z-weighted combiner so disparate per-sector
        E_min values do not destabilise the floating-point exponent.

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
            One row per term in ``O_Q``. ``qed.spectral`` extracts
            these automatically from an ``ed.Operator`` argument.
        opts : SpectralOptions, optional
            ``krylov_dim``, ``broadening``, ``omega_*``,
            ``momentum_transfer``, ``selected_sectors``, etc.
        fixed_sz_n_up : int or None, optional
            Source-sector n_up. Required when ``delta_n_up != 0``.
        delta_n_up : int, optional
            Change in n_up produced by ``O_Q``.
        temperatures : list of float
            Temperatures to evaluate (must be non-empty; values are
            energy-axis units).
        num_samples : int, optional
            FTLM random samples per source sector.
        random_seed : int, optional
            Seed base; per-sample seed is
            ``random_seed + sample_idx * 12345``.

        Returns
        -------
        SpectralResult
            ``omega`` / ``S_real`` carry the recombined S(Q,omega) at
            ``temperatures[0]``. The per-T data set is stuffed into
            ``per_sector_pair`` entries (one per source sector pair
            for the **primary** T, followed by one synthetic entry
            per temperature whose ``initial.sector_dim == 0`` flag
            distinguishes the multi-T payload). The Python wrapper
            ``qed.spectral`` unpacks this and surfaces a clean
            ``{T -> S(omega)}`` dict to the user.
    )pbdoc");
    m.def("workflows_spectral_streaming_symmetry_ftlm_cross_irrep",
          [spectral_ftlm_cross_irrep_body](
              const Operator&                        H,
              const py::dict&                        group,
              std::uint64_t                          num_sites,
              double                                 spin_l,
              const std::vector<py::tuple>&          observable_transforms,
              ed::workflows::SpectralOptions         opts,
              py::object                             fixed_sz_n_up,
              int                                    delta_n_up,
              std::vector<double>                    temperatures,
              std::uint64_t                          num_samples,
              std::uint64_t                          random_seed,
              int                                    sz_parity,
              bool                                   flip_sectors) {
              return spectral_ftlm_cross_irrep_body(
                  in_memory_symmetric_source(
                      H, group,
                      "workflows_spectral_streaming_symmetry_ftlm_cross_"
                      "irrep"),
                  num_sites, spin_l, observable_transforms, std::move(opts),
                  std::move(fixed_sz_n_up), delta_n_up,
                  std::move(temperatures), num_samples, random_seed,
                  sz_parity, flip_sectors);
          },
          py::arg("H"),
          py::arg("group"),
          py::arg("num_sites"),
          py::arg("spin_l")                = 0.5,
          py::arg("observable_transforms") = std::vector<py::tuple>{},
          py::arg("opts")                  = ed::workflows::SpectralOptions{},
          py::arg("fixed_sz_n_up")         = py::none(),
          py::arg("delta_n_up")            = 0,
          py::arg("temperatures")          = std::vector<double>{},
          py::arg("num_samples")           = std::uint64_t{30},
          py::arg("random_seed")           = std::uint64_t{0},
          py::arg("sz_parity")             = -1,
          py::arg("flip_sectors")          = false,
          R"pbdoc(
        In-memory twin of
        ``workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory``.

        Takes the Hamiltonian ``H`` (its terms are copied) and the group
        info dict the directory writer consumes in place of the
        directory; every other argument and the result are identical.
    )pbdoc");

}
