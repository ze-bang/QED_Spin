// =============================================================================
// python/qed/_bindings/little_group_bindings.cpp
//
// The little-group (factorized non-abelian) engine exposed to Python as the
// _core.little_group_* verbs. G = A x| P: solve one momentum per star, project the
// star representative's matrix-free k-sector with the little co-group's
// numerically decomposed irreps -- memory O(#reps(k)), never O(2^N). Every
// refinement step degrades gracefully to the plain k0 block.
//
// Invoked from PYBIND11_MODULE(_core, ...) in qed_bindings.cpp, after the Operator
// types are registered.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include <ed/core/construct_ham.h>
#include <ed/dssf/operator_spec.h>
#include <ed/planner/basis_policy_hook.h>   // ScopedBasisRepr / prefer_tableless_fixed_sz (leaf)
#include <ed/solvers/ftlm.h>
#include <ed/solvers/lanczos.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/observables.h>
#include <ed/symmetry/group.h>
#include <ed/symmetry/irreps.h>
#include <ed/solvers/little_group_solve.h>  // Stage 7 factorized non-abelian
#include <ed/solvers/little_group_blocks.h> // U1b: little_group_thermal
#include <ed/core/select_backend.h>         // have_cuda (sweep GPU cell)
#include <ed/core/hdf5_io.h>                // r2b: canonical eigenvector save
#include <ed/symmetry/spin_flip.h>
#include <ed/symmetry/env_gates.h>  // Stage 10b: gate inventory + dump  // sz_axis_of (Stage 8d diagonal-axis compose)
#include <ed/dssf/cross_sector_orbit_observable.h>  // 9d: rectangular rep apply
#include <ed/observables/cf_spectral_kernel.h>      // 9d: cf_spectral_from_vector
#include <ed/matvec/backends/cpu_backend.h>         // 9d: CF backend
#include <ed/thermal/ftlm_kernel.h>                 // Family-2 front door

#include "little_group_bindings.h"

namespace py = pybind11;

namespace {
using Complex = std::complex<double>;
}  // namespace

void bind_little_group(py::module_& m) {
    auto lg_opts = [](int n_up, int sz_parity, int dense_max_dim,
                      bool use_gpu = false, int spin_flip = -1,
                      int time_reversal = -1,
                      const std::vector<int>& only_k0 = {},
                      bool plan_only = false,
                      const std::vector<int>& only_irrep = {}) {
        ed::solvers::LittleGroupOptions o;
        o.n_up          = n_up;
        o.sz_parity     = sz_parity;
        o.dense_max_dim = dense_max_dim;
        o.spin_flip     = spin_flip;
        o.time_reversal = time_reversal;
        o.only_k0       = only_k0;
        o.plan_only     = plan_only;
        o.only_irrep    = only_irrep;
#ifdef WITH_CUDA
        o.use_gpu       = use_gpu;
#else
        (void)use_gpu;
#endif
        return o;
    };
    // Audit 2026-08-01: subspace argmin for the little-group GS verbs
    // (gs_dssf / gs_static_sf / gs_correlators). The old per-verb loops
    // called little_group_lowest_eigenvalues (k=1) and treated an EMPTY
    // return as "nothing here" -- but the flattened API also returns
    // empty when the E0 scan did NOT CONVERGE (the honest-refusal
    // convention), so at frontier dims the true GS subspace could
    // silently lose the argmin and the verb would then certify a
    // beautiful eigenpair of the WRONG Sz subspace. Scan the LABELED
    // spectrum instead: an unconverged label, or an empty spectrum over
    // stars that hold states (dim_k0 > 0), is a refusal and throws --
    // mirroring little_group_ground_state's star-scan contract. Only a
    // subspace whose stars are all dimension-0 is genuinely empty.
    auto scan_gs_subspace = [](const Operator& op_h,
                               const std::vector<std::vector<int>>& ag,
                               const std::vector<std::vector<int>>& rp,
                               int n_sites_,
                               const std::vector<std::pair<int, int>>& subs,
                               const auto& lg_o,
                               const char* who) -> std::pair<int, int> {
        if (subs.size() == 1) return subs[0];
        int gs_nu = -1, gs_par = -1;
        double e_best = 0.0;
        bool have = false;
        std::size_t n_refused = 0;
        for (const auto& [nu, par] : subs) {
            const auto spec = ed::solvers::little_group_lowest_spectrum(
                op_h, ag, rp, n_sites_, 1, lg_o(nu, par));
            bool conv = true;
            for (const auto& lab : spec.labels)
                if (!lab.converged) conv = false;
            if (!conv) { ++n_refused; continue; }
            if (spec.eigenvalues.empty()) {
                bool has_states = false;
                for (const auto& st : spec.stars)
                    if (st.dim_k0 > 0) has_states = true;
                if (has_states) ++n_refused;
                continue;                        // else genuinely empty
            }
            const double e0 = *std::min_element(spec.eigenvalues.begin(),
                                                spec.eigenvalues.end());
            if (!have || e0 < e_best) {
                have = true; e_best = e0; gs_nu = nu; gs_par = par;
            }
        }
        if (n_refused > 0)
            throw std::runtime_error(
                std::string(who) + ": the subspace E0 scan failed to "
                "converge on " + std::to_string(n_refused) + " subspace(s)"
                " -- the winner would be chosen among the remainder, i.e."
                " the certified ground state could belong to the WRONG"
                " subspace. Raise ED_SYM_LG_LOWEST_MAX_ITER, or pin the"
                " ground-state subspace (n_up / sz_parity) if it is"
                " known.");
        if (!have)
            throw std::runtime_error(std::string(who)
                                     + ": no non-empty subspace.");
        return {gs_nu, gs_par};
    };
    // Stage 9f: per-BLOCK quantum-number labels, parallel to
    // block_values / multiplicities. k_raw is the star REPRESENTATIVE's
    // abelian irrep (fold partners share it; membership is in "stars").
    // block_size >= 2 routes above-crossover blocks through block Krylov-Schur,
    // which resolves within-block degeneracies up to the block width.
    auto with_block_size = [](ed::solvers::LittleGroupOptions o, int block_size) {
        o.block_size = block_size;
        return o;
    };
    auto lg_label_arrays = [](const ed::solvers::LittleGroupSpectrum& s,
                              py::dict& d) {
        std::vector<int> kraw, fpar, irr, irrd;
        kraw.reserve(s.labels.size()); fpar.reserve(s.labels.size());
        irr.reserve(s.labels.size());  irrd.reserve(s.labels.size());
        for (const auto& L : s.labels) {
            kraw.push_back(L.k_raw);
            fpar.push_back(L.flip_parity);
            irr.push_back(L.irrep);
            irrd.push_back(L.irrep_dim);
        }
        d["block_k_raw"]       = kraw;
        d["block_flip_parity"] = fpar;
        d["block_irrep"]       = irr;
        d["block_irrep_dim"]   = irrd;
    };
    auto lg_stars_dict = [](const ed::solvers::LittleGroupSpectrum& s) {
        py::list stars;
        for (const auto& st : s.stars) {
            py::dict d;
            d["k0"]           = st.k0;
            d["star_size"]    = st.star_size;
            d["members"]      = st.members;
            // P_k0's character table -- name an irrep by its CHARACTER, not by
            // decompose_irreps' internal index. Columns are identified by
            // little_elems (residue indices into the caller's residue_perms;
            // -1 = identity). Empty when the star was not projected.
            d["little_elems"]      = st.little_elems;
            d["little_characters"] = st.little_characters;
            d["little_irrep_dims"] = st.little_irrep_dims;
            d["little_order"] = st.little_order;
            d["projected"]    = st.projected;
            d["dim_k0"]       = st.dim_k0;
            d["flip_parity"]  = st.flip_parity;
            d["tr_pairs"]     = st.tr_pairs;
            d["gpu_engaged"]  = st.gpu_engaged;
            d["csr_engaged"]  = st.csr_engaged;
            stars.append(d);
        }
        return stars;
    };

    m.def("little_group_full_spectrum",
          [lg_opts, lg_stars_dict, lg_label_arrays](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int n_up, int sz_parity, bool use_gpu, int spin_flip,
             int time_reversal, int dense_max_dim,
             const std::vector<int>& only_k0, bool plan_only,
             const std::vector<int>& only_irrep) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto s = ed::solvers::little_group_full_spectrum(
                  op, abelian_group, residue_perms, n_sites,
                  lg_opts(n_up, sz_parity, dense_max_dim, use_gpu, spin_flip,
                          time_reversal, only_k0, plan_only, only_irrep));
              py::dict d;
              d["eigenvalues"]    = s.expanded();
              d["block_values"]   = s.eigenvalues;
              d["multiplicities"] = s.multiplicities;
              lg_label_arrays(s, d);
              d["irrep_characters"] = s.irrep_characters;
              d["stars"]          = lg_stars_dict(s);
              d["flip_engaged"]   = s.flip_engaged;
              d["tr_engaged"]     = s.tr_engaged;
              d["unconverged_blocks"] = s.unconverged_blocks;
              d["gpu_engaged"]    = s.gpu_engaged;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("n_up") = -1,
          py::arg("sz_parity") = -1, py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          py::arg("only_k0") = std::vector<int>{},
          py::arg("plan_only") = false,
          py::arg("only_irrep") = std::vector<int>{},
          "Full spectrum via the FACTORIZED little-co-group reduction "
          "(one momentum per star, per-irrep blocks inside the star "
          "representative's matrix-free k-sector). only_k0=[...] solves ONLY "
          "those star representatives (extended irrep indices, as reported by "
          "stars[].k0); the covering sum rule is skipped for a restricted "
          "call because a subset cannot tile the subspace. plan_only=True builds "
          "every star's sector and returns stars[] + irrep_characters WITHOUT "
          "solving -- read the star membership and decode a momentum from "
          "chi_k before naming only_k0 (k_raw is an internal irrep index, not "
          "the momentum).");

    m.def("little_group_lowest_eigenvalues",
          [lg_opts, with_block_size](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int k, int n_up, int sz_parity, int dense_max_dim,
             int spin_flip, int time_reversal, bool use_gpu,
             const std::vector<int>& only_k0,
             const std::vector<int>& only_irrep, int block_size) {
              const int n_sites = static_cast<int>(op.getNumBits());
              return ed::solvers::little_group_lowest_eigenvalues(
                  op, abelian_group, residue_perms, n_sites, k,
                  with_block_size(lg_opts(n_up, sz_parity, dense_max_dim, use_gpu,
                          spin_flip, time_reversal, only_k0,
                          /*plan_only=*/false, only_irrep), block_size));
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("k") = 1,
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("dense_max_dim") = 64,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("use_gpu") = false,
          py::arg("only_k0") = std::vector<int>{},
          py::arg("only_irrep") = std::vector<int>{},
          py::arg("block_size") = 1,
          "Lowest-k eigenvalues via the factorized little-co-group "
          "reduction (dense on small blocks, Lanczos on the projected "
          "matrix-free matvec otherwise); multiplicities expanded.");

    m.def("little_group_lowest_eigenvalues_labeled",
          [lg_opts, lg_stars_dict, with_block_size](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int k, int n_up, int sz_parity, int dense_max_dim,
             bool use_gpu, int spin_flip, int time_reversal,
             const std::vector<int>& only_k0,
             const std::vector<int>& only_irrep, int block_size) {
              // Stage 9f: the labeled twin of little_group_lowest_eigenvalues.
              // Aligned per-eigenvalue arrays (expanded by multiplicity,
              // sorted ascending, truncated to k): momentum k_raw is the star
              // REPRESENTATIVE's abelian irrep (fold partners share it; the
              // membership is in "stars"), irrep indexes the little co-group
              // decomposition at that star (-1 = plain block), flip_parity is
              // the (k, +/-) slot when A' = A x Z2 engaged.
              const int n_sites = static_cast<int>(op.getNumBits());
              const auto s = ed::solvers::little_group_lowest_spectrum(
                  op, abelian_group, residue_perms, n_sites, k,
                  with_block_size(lg_opts(n_up, sz_parity, dense_max_dim, use_gpu,
                          spin_flip, time_reversal, only_k0, false,
                          only_irrep), block_size));
              std::vector<std::size_t> order(s.eigenvalues.size());
              std::iota(order.begin(), order.end(), std::size_t{0});
              std::sort(order.begin(), order.end(),
                        [&](std::size_t a, std::size_t b) {
                            return s.eigenvalues[a] < s.eigenvalues[b];
                        });
              std::vector<double> ev;
              std::vector<int> kraw, fpar, irr, irrd, mult;
              std::vector<bool> conv;
              const std::size_t want = static_cast<std::size_t>(
                  std::max(k, 1));
              for (std::size_t idx : order) {
                  const auto& L = s.labels[idx];
                  for (int r = 0; r < s.multiplicities[idx]
                                  && ev.size() < want; ++r) {
                      ev.push_back(s.eigenvalues[idx]);
                      kraw.push_back(L.k_raw);
                      fpar.push_back(L.flip_parity);
                      irr.push_back(L.irrep);
                      irrd.push_back(L.irrep_dim);
                      mult.push_back(s.multiplicities[idx]);
                      conv.push_back(L.converged);
                  }
                  if (ev.size() >= want) break;
              }
              py::dict d;
              d["eigenvalues"]  = ev;
              d["k_raw"]        = kraw;
              d["flip_parity"]  = fpar;
              d["irrep"]        = irr;
              d["irrep_dim"]    = irrd;
              d["multiplicity"] = mult;
              d["converged"]    = conv;
              d["irrep_characters"] = s.irrep_characters;
              d["stars"]        = lg_stars_dict(s);
              d["flip_engaged"] = s.flip_engaged;
              d["tr_engaged"]   = s.tr_engaged;
              d["unconverged_blocks"] = s.unconverged_blocks;
              d["gpu_engaged"]  = s.gpu_engaged;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("k") = 1,
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("dense_max_dim") = 64, py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("only_k0") = std::vector<int>{},
          py::arg("only_irrep") = std::vector<int>{},
          py::arg("block_size") = 1,
          "Stage 9f: lowest-k eigenvalues WITH aligned per-eigenvalue "
          "quantum-number labels (k_raw = star representative's momentum "
          "irrep, little-group irrep index + dimension, flip parity, "
          "multiplicity) -- the labels the engine always computed and "
          "previously discarded at this boundary. only_k0=[...] restricts the "
          "walk to those star representatives, so NAMING a momentum costs only "
          "that star's work instead of forcing the call off the projection "
          "lane onto the (larger) abelian block.");

    m.def("little_group_block_grounds",
          [lg_opts, lg_stars_dict](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int n_up, int sz_parity, int dense_max_dim,
             bool use_gpu, int spin_flip, int time_reversal,
             const std::vector<int>& only_k0,
             const std::vector<int>& only_irrep) {
              // Lowest eigenvalue of EVERY (momentum, little-co-group irrep)
              // block -- k=1 PER BLOCK, so the no-reorth Lanczos early-exits
              // the instant the lowest Ritz value converges, BEFORE the
              // three-term recurrence loses orthogonality. (The k>1 path of
              // little_group_lowest_spectrum runs to ~400 iters and returns a
              // ghost at N=36 -- a bit-identical spurious constant across all
              // stars; this lane avoids it entirely by never asking a block
              // for more than its ground.) Returns ALL blocks, no global
              // truncation: the per-(k,irrep) minima ARE the Anderson tower's
              // momentum/irrep-resolved low-energy structure -- collect the
              // lowest-few across blocks for the tower multiplet, each entry
              // already carrying (k_raw, irrep, flip_parity).
              const int n_sites = static_cast<int>(op.getNumBits());
              const auto s = ed::solvers::little_group_lowest_spectrum(
                  op, abelian_group, residue_perms, n_sites, /*k=*/1,
                  lg_opts(n_up, sz_parity, dense_max_dim, use_gpu,
                          spin_flip, time_reversal, only_k0, false,
                          only_irrep));
              std::vector<double> ev;
              std::vector<int> kraw, fpar, irr, irrd, mult;
              std::vector<bool> conv;
              for (std::size_t idx = 0; idx < s.eigenvalues.size(); ++idx) {
                  const auto& L = s.labels[idx];
                  ev.push_back(s.eigenvalues[idx]);
                  kraw.push_back(L.k_raw);
                  fpar.push_back(L.flip_parity);
                  irr.push_back(L.irrep);
                  irrd.push_back(L.irrep_dim);
                  mult.push_back(s.multiplicities[idx]);
                  conv.push_back(L.converged);
              }
              py::dict d;
              d["eigenvalues"]  = ev;
              d["k_raw"]        = kraw;
              d["flip_parity"]  = fpar;
              d["irrep"]        = irr;
              d["irrep_dim"]    = irrd;
              d["multiplicity"] = mult;
              d["converged"]    = conv;
              d["irrep_characters"] = s.irrep_characters;
              d["stars"]        = lg_stars_dict(s);
              d["flip_engaged"] = s.flip_engaged;
              d["tr_engaged"]   = s.tr_engaged;
              d["unconverged_blocks"] = s.unconverged_blocks;
              d["gpu_engaged"]  = s.gpu_engaged;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("n_up") = -1,
          py::arg("sz_parity") = -1, py::arg("dense_max_dim") = 64,
          py::arg("use_gpu") = false, py::arg("spin_flip") = -1,
          py::arg("time_reversal") = -1,
          py::arg("only_k0") = std::vector<int>{},
          py::arg("only_irrep") = std::vector<int>{},
          "Lowest eigenvalue of EVERY (momentum, irrep) block -- k=1 per "
          "block (reliable early-exit, unlike the k>1 spectrum lane which "
          "ghosts at N=36). Aligned per-block arrays {eigenvalues, k_raw, "
          "irrep, irrep_dim, flip_parity, multiplicity, converged} plus "
          "stars, irrep_characters. The momentum/irrep-resolved tower.");

    m.def("little_group_thermodynamics",
          [lg_opts](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             const std::vector<double>& temperatures,
             int n_up, int sz_parity, bool use_gpu, int spin_flip,
             int time_reversal, int dense_max_dim) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto td = ed::solvers::little_group_thermodynamics(
                  op, abelian_group, residue_perms, n_sites, temperatures,
                  lg_opts(n_up, sz_parity, dense_max_dim, use_gpu, spin_flip,
                          time_reversal));
              py::dict d;
              d["temperatures"]  = td.temperatures;
              d["energy"]        = td.energy;
              d["specific_heat"] = td.specific_heat;
              d["entropy"]       = td.entropy;
              d["free_energy"]   = td.free_energy;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("temperatures"),
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          "Exact canonical thermodynamics from the factorized "
          "little-co-group full spectrum.");

    // U1b (lane unification): SAMPLED thermodynamics inside the projected
    // blocks -- FTLM/LTLM/mTPQ/OFTLM per (n_up, k, +/-, sigma) block via
    // ed::workflows::thermal(block.op(), ...), Z-recombined with the block
    // multiplicity folded in as an F-shift. KPM_DOS raises (full-spectrum
    // DOS deliverable; use the abelian lane). Returns the combined thermo
    // plus parallel per-block tag arrays -- the engagement signal the
    // dimension-reduction matrix asserts block structure against.
    m.def("little_group_thermal",
          [lg_opts](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             const std::string& method, double t_min, double t_max,
             std::size_t num_t, std::size_t num_samples,
             std::size_t krylov_dim, std::uint64_t random_seed,
             int n_up, int sz_parity, bool use_gpu, int spin_flip,
             int time_reversal, int dense_max_dim) {
              using Method = ed::workflows::ThermalOptions::Method;
              ed::workflows::ThermalOptions topts;
              if      (method == "FTLM")  topts.method = Method::FTLM;
              else if (method == "LTLM")  topts.method = Method::LTLM;
              else if (method == "mTPQ")  topts.method = Method::mTPQ;
              else if (method == "OFTLM") topts.method = Method::OFTLM;
              else
                  throw std::invalid_argument(
                      "little_group_thermal: method must be one of "
                      "FTLM/LTLM/mTPQ/OFTLM (KPM_DOS recombines on the "
                      "abelian lane only), got '" + method + "'");
              topts.temp_min      = t_min;
              topts.temp_max      = t_max;
              topts.num_temp_bins = num_t;
              topts.num_samples   = num_samples;
              topts.krylov_dim    = krylov_dim;
              topts.random_seed   = random_seed;
              topts.backend.allow_gpu = use_gpu;
              const int n_sites = static_cast<int>(op.getNumBits());
              ed::solvers::LittleGroupThermalResult r;
              {
                  py::gil_scoped_release release;
                  r = ed::solvers::little_group_thermal(
                      op, abelian_group, residue_perms, n_sites, topts,
                      lg_opts(n_up, sz_parity, dense_max_dim, use_gpu,
                              spin_flip, time_reversal));
              }
              py::dict d;
              d["temperatures"]  = r.thermo.temperatures;
              d["energy"]        = r.thermo.energy;
              d["specific_heat"] = r.thermo.specific_heat;
              d["entropy"]       = r.thermo.entropy;
              d["free_energy"]   = r.thermo.free_energy;
              d["ground_state_energy"] = r.ground_state_energy;
              d["projected_any"] = r.projected_any;
              d["gpu_engaged"]   = r.gpu_engaged;
              std::vector<int> b_nup, b_kraw, b_flip, b_irrep, b_idim,
                               b_star;
              std::vector<std::uint64_t> b_dim, b_mult, b_weight;
              for (std::size_t i = 0; i < r.block_tags.size(); ++i) {
                  const auto& t = r.block_tags[i];
                  b_nup.push_back(t.n_up);
                  b_kraw.push_back(t.k_raw);
                  b_flip.push_back(t.flip_parity);
                  b_irrep.push_back(t.irrep);
                  b_idim.push_back(t.irrep_dim);
                  b_star.push_back(t.star_size);
                  b_dim.push_back(t.dim);
                  b_mult.push_back(t.multiplicity);
                  b_weight.push_back(r.weights[i]);
              }
              d["block_n_up"]        = b_nup;
              d["block_k_raw"]       = b_kraw;
              d["block_flip_parity"] = b_flip;
              d["block_irrep"]       = b_irrep;
              d["block_irrep_dim"]   = b_idim;
              d["block_star_size"]   = b_star;
              d["block_dim"]         = b_dim;
              d["block_multiplicity"] = b_mult;
              d["block_weight"]      = b_weight;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("method") = "FTLM",
          py::arg("t_min") = 0.1, py::arg("t_max") = 10.0,
          py::arg("num_t") = 24, py::arg("num_samples") = 40,
          py::arg("krylov_dim") = 100, py::arg("random_seed") = 0,
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          "Sampled (FTLM/LTLM/mTPQ/OFTLM) thermodynamics inside the "
          "factorized little-group blocks, Z-recombined with block "
          "multiplicities.");

    // U2b-r2: certified lowest-k eigenpairs on the projection lane.
    // Vectors are returned as COMPUTATIONAL-basis amplitude arrays
    // (flip-aware expansion; moderate-N contract, n_sites <= 30). A row
    // with multiplicity > 1 carries ONE representative vector (fold
    // partners need U3 transport).
    m.def("little_group_lowest_vectors",
          [lg_opts](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int k, int n_up, int sz_parity, int spin_flip,
             int time_reversal, int dense_max_dim,
             const std::string& output_dir,
             const std::vector<int>& only_k0,
             const std::vector<int>& only_irrep) {
              const int n_sites = static_cast<int>(op.getNumBits());
              ed::solvers::LittleGroupVectors lv;
              {
                  py::gil_scoped_release release;
                  lv = ed::solvers::little_group_lowest_vectors(
                      op, abelian_group, residue_perms, n_sites, k,
                      lg_opts(n_up, sz_parity, dense_max_dim,
                              /*use_gpu=*/false, spin_flip,
                              time_reversal, only_k0,
                              /*plan_only=*/false, only_irrep));
              }
              py::dict d;
              std::vector<double> evals;
              std::vector<int> kraw, flip, irrep, idim;
              std::vector<std::uint64_t> mult;
              py::list vecs;
              std::vector<std::vector<std::complex<double>>> psis;
              for (const auto& row : lv.rows) {
                  evals.push_back(row.eigenvalue);
                  kraw.push_back(row.tag.k_raw);
                  flip.push_back(row.tag.flip_parity);
                  irrep.push_back(row.tag.irrep);
                  idim.push_back(row.tag.irrep_dim);
                  mult.push_back(row.tag.multiplicity);
                  auto psi =
                      ed::solvers::expand_rep_vector_to_computational(
                          lv.sectors[row.sector_slot], row.vec);
                  py::array_t<std::complex<double>> arr(
                      static_cast<py::ssize_t>(psi.size()));
                  std::copy(psi.begin(), psi.end(),
                            arr.mutable_data());
                  vecs.append(std::move(arr));
                  if (!output_dir.empty()) psis.push_back(std::move(psi));
              }
              // r2b: persist through the canonical writer (the same
              // /eigendata layout every solver emits). COMPUTATIONAL
              // basis -- directly consumable, unlike the abelian lane's
              // per-sector sector-basis files. Row i's eigenvector pairs
              // with eigenvalue i.
              if (!output_dir.empty()
                  && !HDF5IO::isDisabledOutputPath(output_dir)) {
                  HDF5IO::saveDiagonalizationResults(
                      output_dir, evals, psis, "LITTLE_GROUP_VECTORS");
                  d["hdf5_path"] = output_dir + "/ed_results.h5";
              } else {
                  d["hdf5_path"] = std::string();
              }
              d["eigenvalues"]   = evals;
              d["k_raw"]         = kraw;
              d["flip_parity"]   = flip;
              d["irrep"]         = irrep;
              d["irrep_dim"]     = idim;
              d["multiplicity"]  = mult;
              d["vectors"]       = vecs;
              d["flip_engaged"]  = lv.flip_engaged;
              d["tr_engaged"]    = lv.tr_engaged;
              // Audit 2026-08-01: refusal accounting (a full window with
              // nonzero counts may still start above a refused level).
              d["refused_blocks"] = lv.refused_blocks;
              d["refused_rows"]   = lv.refused_rows;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("k") = 1,
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          py::arg("output_dir") = std::string(),
          py::arg("only_k0") = std::vector<int>{},
          py::arg("only_irrep") = std::vector<int>{},
          "Certified lowest-k eigenpairs of the factorized block "
          "decomposition, vectors expanded to the computational basis "
          "(persisted to <output_dir>/ed_results.h5 when given).");

    // -------------------------------------------------------------------------
    // I4: ground state in the SYMMETRY-REDUCED representative basis.
    //
    // Why this exists. `little_group_lowest_vectors` expands through
    // expand_rep_vector_to_computational, i.e. to 2^n_sites amplitudes. At the
    // N=36 production point that is 2^36 * 16 B = 1.1 TB PER VECTOR, so the
    // only vector lane the engine had was unusable exactly where the campaign
    // needs vectors. The rep basis is 3.78e8 amplitudes (6.05 GB) for the same
    // state -- a factor 182 -- and it is what every diagonal observable already
    // consumes (see little_group_gs_correlators, which reads gs.rd.reps and
    // gs.vec and never expands).
    //
    // Returns everything needed to reconstruct or measure WITHOUT this process:
    // reps + inv_norms + characters + perms_flat + flip_masks fully determine
    // the orbit expansion, so a saved vector is self-contained.
    //
    // Pin one block per job with ED_SYM_LG_ONLY_K0; with no filter the engine
    // scans the stars and returns the GLOBAL ground state. One state per call:
    // a manifold spread over several blocks is one pinned call per block.
    // -------------------------------------------------------------------------
    m.def("little_group_gs_rep_vector",
          [lg_opts](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int n_up, int sz_parity, int spin_flip,
             int time_reversal, int dense_max_dim,
             const std::vector<int>& only_irrep,
             const std::vector<int>& only_k0) {
              const int n_sites = static_cast<int>(op.getNumBits());
              ed::solvers::LittleGroupGroundState gs;
              {
                  py::gil_scoped_release release;
                  // only_irrep names ONE little-co-group irrep, which is what
                  // makes 36d tractable: the Gamma star carries 12 irreps, and
                  // solving them in one process is why the first gate attempt
                  // ran 19 h. One irrep per array task parallelises that 12x.
                  auto o = lg_opts(n_up, sz_parity, dense_max_dim,
                                   /*use_gpu=*/false, spin_flip,
                                   time_reversal);
                  o.only_irrep = only_irrep;
                  o.only_k0    = only_k0;
                  gs = ed::solvers::little_group_ground_state(
                      op, abelian_group, residue_perms, n_sites, o);
              }
              const auto& rd = gs.rd;
              const std::size_t nr = rd.reps.size();
              // A length mismatch means the vec contract broke; refuse rather
              // than hand back a silently misaligned 6 GB array.
              if (gs.vec.size() != nr)
                  throw std::runtime_error(
                      "little_group_gs_rep_vector: vec/reps length mismatch ("
                      + std::to_string(gs.vec.size()) + " vs "
                      + std::to_string(nr) + ").");
              if (nr == 0)
                  throw std::runtime_error(
                      "little_group_gs_rep_vector: empty sector -- the star "
                      "filter selected nothing solvable.");
              auto to_np = [](const auto& v) {
                  using T = typename std::decay_t<decltype(v)>::value_type;
                  py::array_t<T> a(static_cast<py::ssize_t>(v.size()));
                  std::copy(v.begin(), v.end(), a.mutable_data());
                  return a;
              };
              py::dict d;
              d["energy"]      = gs.energy;
              d["k0"]          = gs.k0;
              d["irrep"]       = gs.irrep;
              d["flip_parity"] = gs.flip_parity;
              d["group_size"]  = rd.group_size;
              d["n_sites"]     = rd.n_sites;
              d["n_up"]        = rd.n_up;
              d["dim"]         = static_cast<std::uint64_t>(nr);
              d["vec"]         = to_np(gs.vec);
              d["reps"]        = to_np(rd.reps);
              d["inv_norms"]   = to_np(rd.inv_norms);
              d["characters"]  = to_np(rd.characters);
              d["perms_flat"]  = to_np(rd.perms_flat);
              d["flip_masks"]  = to_np(rd.flip_masks);
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("n_up") = -1,
          py::arg("sz_parity") = -1, py::arg("spin_flip") = -1,
          py::arg("time_reversal") = -1, py::arg("dense_max_dim") = 256,
          py::arg("only_irrep") = std::vector<int>{},
          py::arg("only_k0") = std::vector<int>{},
          "Ground state in the representative basis of its own momentum "
          "sector (NOT expanded to 2^N), with the orbit data needed to "
          "expand or measure it elsewhere. Pin a block with only_k0 / only_irrep "
          "(or ED_SYM_LG_ONLY_K0 when only_k0 is empty); unfiltered returns the "
          "global ground state.");

    m.def("little_group_gs_dssf",
          [scan_gs_subspace](const Operator& op_h, const Operator& op_o,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             double omega_min, double omega_max, int n_omega,
             double broadening, int krylov_dim, int dense_max_dim,
             bool use_gpu, int time_reversal) {
              // Stage 9d: FACTORIZED GS-DSSF. The ground state is
              // localized by the star walk (folds shrink the search) and
              // solved PLAIN in its momentum sector; O|0> is scattered
              // into every RAW destination sector by the Stage-8d
              // CrossSectorOrbitObservable rep lane (matrix elements are
              // never folded -- ||phi|| decides every selection rule);
              // one continued-fraction Lanczos per receiving sector.
              // Memory O(#reps) throughout -- this replaces the
              // monolithic SAB DSSF, which materialised the FULL
              // eigenbasis in the computational basis.
              using Complex = std::complex<double>;
              const int n_sites = static_cast<int>(op_h.getNumBits());
              if (!op_o.three_body_data_.empty())
                  throw std::runtime_error(
                      "little_group_gs_dssf: three-body probes are not "
                      "supported (one/two-body TransformData only).");

              // Diagonal axis (same detection as everywhere else).
              ed::matvec::TermStorage soa;
              ed::matvec::TermStorage::classify_route(
                  soa, op_h.transform_data_, op_h.three_body_data_,
                  [](const std::complex<double>& c) { return c; });
              const auto ax = ed::symmetry::sz_axis_of(soa);
              std::vector<std::pair<int, int>> gs_subspaces{{-1, -1}};
              if (ax == ed::symmetry::SzAxis::U1) {
                  gs_subspaces.clear();
                  for (int k = 0; k <= n_sites; ++k)
                      gs_subspaces.emplace_back(k, -1);
              } else if (ax == ed::symmetry::SzAxis::Parity) {
                  gs_subspaces = {{-1, 0}, {-1, 1}};
              }

              // (1) Locate the GS subspace by star-walk lowest-1 solves.
              auto lg_o = [&](int nu, int par) {
                  ed::solvers::LittleGroupOptions o;
                  o.n_up          = nu;
                  o.sz_parity     = par;
                  o.dense_max_dim = dense_max_dim;
#ifdef WITH_CUDA
                  o.use_gpu       = use_gpu;   // batched eigensolve on the
                                               // GS-subspace scan
#endif
                  // U2b-r1b: the SOURCE may be flip-extended (auto). The
                  // rep-lane scatter is flip-aware (Stage-5b masks ride
                  // the policy) and the cross-sector normalization
                  // handles |G_src| != |G_dst| (1/sqrt(Gs*Gd)); the
                  // destinations below stay RAW. Pinned by the
                  // flip-engaged Lehmann test in test_little_group_dssf.
                  o.spin_flip     = -1;
                  o.time_reversal = time_reversal;
                  return o;
              };
              const auto [gs_nu, gs_par] = scan_gs_subspace(
                  op_h, abelian_group, residue_perms, n_sites,
                  gs_subspaces, lg_o, "little_group_gs_dssf");

              // (2) The GS eigenvector in its momentum sector.
              const auto gs = ed::solvers::little_group_ground_state(
                  op_h, abelian_group, residue_perms, n_sites,
                  lg_o(gs_nu, gs_par));

              // (3) Destination sweep: 1/2-body probes reach at most
              // n_up +- 2 (U(1)) / both parity halves / the full space.
              std::vector<std::pair<int, int>> dst_subspaces;
              if (ax == ed::symmetry::SzAxis::U1) {
                  for (int d = -2; d <= 2; ++d) {
                      const int nu = gs_nu + d;
                      if (nu >= 0 && nu <= n_sites)
                          dst_subspaces.emplace_back(nu, -1);
                  }
              } else if (ax == ed::symmetry::SzAxis::Parity) {
                  dst_subspaces = {{-1, 0}, {-1, 1}};
              } else {
                  dst_subspaces = {{-1, -1}};
              }

              std::vector<double> omega_grid(
                  static_cast<std::size_t>(std::max(n_omega, 1)));
              const double dw = (n_omega > 1)
                  ? (omega_max - omega_min) / (n_omega - 1) : 0.0;
              for (int i = 0; i < std::max(n_omega, 1); ++i)
                  omega_grid[static_cast<std::size_t>(i)] =
                      omega_min + dw * i;

              std::vector<double> s_omega(omega_grid.size(), 0.0);
              double total_weight = 0.0;
              bool dssf_gpu = false;   // truthful: any receiving sector's
                                       // CF matvec ran the device gather
              ed::matvec::CpuBackend be;
              using Ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef;
              const auto src_ref = Ref::from_rep(
                  gs.rd, static_cast<std::uint64_t>(n_sites));

              for (const auto& [dnu, dpar] : dst_subspaces) {
                  auto sectors = ed::solvers::little_group_k_sectors(
                      op_h, abelian_group, n_sites, dnu, dpar);
                  for (const auto& rd_dst : sectors) {
                      const std::size_t dim_dst = rd_dst.reps.size();
                      ed::dssf::CrossSectorOrbitObservable obs(
                          src_ref, 0,
                          Ref::from_rep(rd_dst,
                                        static_cast<std::uint64_t>(n_sites)),
                          0, op_o.transform_data_,
                          static_cast<float>(op_h.getSpin()));
                      std::vector<Complex> phi(dim_dst, Complex(0, 0));
                      obs.apply(gs.vec.data(), phi.data(), dim_dst);
                      double n2 = 0.0;
                      for (const Complex& c : phi) n2 += std::norm(c);
                      if (n2 < 1e-24) continue;   // selection rule says no
                      total_weight += n2;

                      // GS-DSSF GPU lane (2026-07-20): an explicit GPU
                      // request forces the device rep-gather on the
                      // receiving sector's CF matvec (dimension floor
                      // dropped, reduced CSR demoted to fallback).
                      auto mv = ed::solvers::make_rep_sector_matvec(
                          op_h, rd_dst, /*force_gpu=*/use_gpu);
                      ed::observables::CfSpectralOptions cfopts;
                      cfopts.krylov_dim   = static_cast<std::size_t>(
                          std::max(krylov_dim, 2));
                      cfopts.broadening   = broadening;
                      cfopts.energy_shift = gs.energy;
                      cfopts.tolerance    = 1e-12;
                      cfopts.global_n     = dim_dst;
                      auto apply_H = [&mv](const Complex* in, Complex* out,
                                           std::size_t nn) {
                          mv->apply(in, out, nn);
                      };
                      const auto cf =
                          ed::observables::cf_spectral_from_vector(
                              be, apply_H, dim_dst, phi.data(),
                              omega_grid, cfopts);
                      for (std::size_t i = 0; i < s_omega.size(); ++i)
                          s_omega[i] += cf.spectral_function[i];
                      dssf_gpu = dssf_gpu ||
                          ed::solvers::rep_sector_matvec_gpu_engaged(*mv);
                  }
              }

              py::dict d;
              d["omega"]        = omega_grid;
              d["s_omega"]      = s_omega;
              d["gs_energy"]    = gs.energy;
              d["gs_k0"]        = gs.k0;
              d["total_weight"] = total_weight;
              d["gpu_engaged"]  = dssf_gpu;
              return d;
          },
          py::arg("op"), py::arg("observable"),
          py::arg("abelian_group"), py::arg("residue_perms"),
          py::arg("omega_min"), py::arg("omega_max"),
          py::arg("n_omega"), py::arg("broadening") = 0.1,
          py::arg("krylov_dim") = 200, py::arg("dense_max_dim") = 512,
          py::arg("use_gpu") = false,
          py::arg("time_reversal") = -1,
          "Stage 9d: factorized ground-state DSSF -- GS localized by the "
          "star walk and solved matrix-free in its momentum sector; O|0> "
          "scattered into every raw destination sector "
          "(CrossSectorOrbitObservable rep lane); one continued-fraction "
          "Lanczos per receiving sector. Memory O(#reps): the scalable "
          "replacement for symmetry_adapted_gs_dssf.");

    // STATIC transverse structure factor S^{+-}(q) for a LIST of probes,
    // amortising the ONE (n_up-pinned) little-group GS solve across every q.
    //
    // The static structure factor is the ZEROTH frequency moment of the DSSF:
    //   S_O = <GS| O^dagger O |GS> = || O|GS> ||^2
    // (the ``total_weight`` little_group_gs_dssf already computes as a
    // by-product before its continued fraction). For O_q = N^{-1/2} sum_j
    // e^{-i q.r_j} S^-_j this is exactly S^{+-}(q). No omega grid, no CF --
    // just the scatter norm per probe, so a whole q-mesh costs ONE GS solve.
    //
    // n_up pins the GS magnetisation (the AFM GS is a singlet at N/2); the
    // unpinned U(1) sweep is ~sqrt(N)x more work for the same answer (mirrors
    // little_group_gs_correlators 0461db3). Memory O(#reps): one source + one
    // destination sector resident at a time.
    m.def("little_group_gs_static_sf",
          [scan_gs_subspace](const Operator& op_h,
             const std::vector<Operator>& observables,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int n_up, int delta_n_up, int dense_max_dim, bool use_gpu,
             int time_reversal, const std::vector<int>& only_k0) {
              using Complex = std::complex<double>;
              const int n_sites = static_cast<int>(op_h.getNumBits());
              for (const Operator& o : observables)
                  if (!o.three_body_data_.empty())
                      throw std::runtime_error(
                          "little_group_gs_static_sf: probes must be "
                          "one/two-body operators.");

              ed::matvec::TermStorage soa;
              ed::matvec::TermStorage::classify_route(
                  soa, op_h.transform_data_, op_h.three_body_data_,
                  [](const std::complex<double>& c) { return c; });
              const auto ax = ed::symmetry::sz_axis_of(soa);

              std::vector<std::pair<int, int>> gs_subspaces;
              if (n_up >= 0) {
                  gs_subspaces = {{n_up, -1}};
              } else if (ax == ed::symmetry::SzAxis::U1) {
                  for (int k = 0; k <= n_sites; ++k)
                      gs_subspaces.emplace_back(k, -1);
              } else if (ax == ed::symmetry::SzAxis::Parity) {
                  gs_subspaces = {{-1, 0}, {-1, 1}};
              } else {
                  gs_subspaces = {{-1, -1}};
              }
              auto lg_o = [&](int nu, int par) {
                  ed::solvers::LittleGroupOptions o;
                  o.n_up = nu; o.sz_parity = par;
                  o.dense_max_dim = dense_max_dim;
#ifdef WITH_CUDA
                  o.use_gpu = use_gpu;
#endif
                  o.spin_flip = -1; o.time_reversal = time_reversal;
                  // Pin the GS to a named momentum star. Needed to keep both
                  // structure-factor channels on the SAME ground state at a
                  // near-degeneracy: two stars within ~1e-5 (e.g. the 4x3
                  // kagome k0=16/20 crossing near Jpm=-0.13) are ranked
                  // inconsistently by the two-phase scan across builds, so
                  // one channel can land on a slightly-excited star.
                  o.only_k0 = only_k0;
                  return o;
              };
              const auto [gs_nu, gs_par] = scan_gs_subspace(
                  op_h, abelian_group, residue_perms, n_sites,
                  gs_subspaces, lg_o, "little_group_gs_static_sf");

              const auto gs = ed::solvers::little_group_ground_state(
                  op_h, abelian_group, residue_perms, n_sites, lg_o(gs_nu, gs_par));

              (void)delta_n_up;

              // IN-SECTOR expectation <GS|O_q|GS>. Each observable O_q is an
              // Sz- and momentum-CONSERVING operator (e.g. the transverse
              // O_q = sum_{i!=j} e^{iq(r_i-r_j)} S^+_i S^-_j, Hermitian), so it
              // maps the GS momentum sector to ITSELF -- only ONE sector is
              // ever resident (memory O(#reps), no destination sector). The
              // cross-sector norm route needed BOTH the GS source and a
              // destination sector built at once: two ~40 GB sectors OOM a
              // 128 GB node at N=36 (task 50220971 died building the 716M-orbit
              // dst table). Here the rep-sector matvec (GPU rep-gather) applies
              // O_q on the GS vector and we dot with <GS|.
              const std::size_t dim = gs.vec.size();
              std::vector<double> static_sf;
              static_sf.reserve(observables.size());
              std::vector<Complex> ov(dim);
              for (const Operator& o : observables) {
                  // rd is consumed by the matvec factory; copy so the GS sector
                  // survives for the next probe (cheap vs. the GS solve).
                  ed::symmetry::RepSectorData rd_copy = gs.rd;
                  auto mv = ed::solvers::make_rep_sector_matvec(
                      o, std::move(rd_copy), /*force_gpu=*/use_gpu);
                  mv->apply(gs.vec.data(), ov.data(), dim);
                  Complex c(0.0, 0.0);
                  for (std::size_t i = 0; i < dim; ++i)
                      c += std::conj(gs.vec[i]) * ov[i];
                  static_sf.push_back(c.real());   // <GS|O_q|GS> (real: Hermitian)
              }

              py::dict d;
              d["gs_energy"] = gs.energy;
              d["gs_k0"]     = gs.k0;
              d["gs_n_up"]   = gs_nu;
              d["n_reps"]    = static_cast<std::uint64_t>(gs.rd.reps.size());
              d["static_sf"] = static_sf;
              return d;
          },
          py::arg("op"), py::arg("observables"),
          py::arg("abelian_group"), py::arg("residue_perms"),
          py::arg("n_up") = -1, py::arg("delta_n_up") = 1,
          py::arg("dense_max_dim") = 512,
          py::arg("use_gpu") = false, py::arg("time_reversal") = -1,
          py::arg("only_k0") = std::vector<int>{},
          "IN-SECTOR ground-state expectations <GS|O_q|GS> for a LIST of "
          "Sz- and momentum-conserving operators O_q, amortising ONE "
          "n_up-pinned little-group GS solve. Each O_q is applied on the GS "
          "vector by the rep-sector matvec (GPU rep-gather) and dotted with "
          "<GS| -- only the GS momentum sector is ever resident (memory "
          "O(#reps), no destination sector). For the transverse static "
          "structure factor pass O_q = sum_{i!=j} e^{iq(r_i-r_j)} S^+_i S^-_j "
          "(Hermitian); then S^{+-}(q) = <GS|O_q|GS>/N + n_up/N (the i=j "
          "self-term is exactly n_up on the fixed-Sz sector). ``delta_n_up`` "
          "is ignored (kept for call-site compatibility). Returns "
          "{gs_energy, gs_k0, gs_n_up, n_reps, static_sf[]} with static_sf = "
          "the raw <GS|O_q|GS>.");

    m.def("little_group_gs_correlators",
          [scan_gs_subspace](const Operator& op_h,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             const std::vector<std::vector<int>>& translations,
             int n_up, int dense_max_dim, bool use_gpu,
             int time_reversal) {
              // STATIC structure factor without the DSSF machinery.
              //
              // S^zz(Q) needs only DIAGONAL, G-INVARIANT correlators:
              //   O_d = sum_i S^z_i S^z_{i+d}
              // is diagonal in the computational basis, translation
              // invariant (the sum over i) and flip invariant
              // (S^z -> -S^z gives (-)(-)). For such an operator the
              // orbit cross-terms vanish and the expectation collapses to
              // a weighted sum over REPRESENTATIVES:
              //     <Psi|O_d|Psi> = sum_r |c_r|^2 O_d(rep_r)
              // with O_d(s) = [N - 2*popcount(s ^ T_d s)] / 4.
              //
              // So ONE ground-state solve yields the energy AND every
              // C(d); the caller Fourier transforms C(d) -> S(Q) at ALL
              // momenta for free. No destination sectors, no
              // continued-fraction chains: memory is ONE sector instead of
              // the ~N_Q resident mirrors the multi-Q spectral lane needs,
              // and the flip Z2 is exploited (half the reps).
              using Complex = std::complex<double>;
              const int n_sites = static_cast<int>(op_h.getNumBits());

              ed::matvec::TermStorage soa;
              ed::matvec::TermStorage::classify_route(
                  soa, op_h.transform_data_, op_h.three_body_data_,
                  [](const std::complex<double>& c) { return c; });
              const auto ax = ed::symmetry::sz_axis_of(soa);
              std::vector<std::pair<int, int>> gs_subspaces{{-1, -1}};
              if (n_up >= 0) {
                  // PINNED: the caller knows the GS magnetisation (a
                  // Heisenberg AFM ground state is a total-spin singlet, so
                  // n_up = N/2). Scanning every n_up costs ~7.6x the
                  // half-filled sector alone at N=36 and cannot find a lower
                  // state -- measured: 96 star solves in 10 h, none of them
                  // in the sector that holds the GS.
                  gs_subspaces = {{n_up, -1}};
              } else if (ax == ed::symmetry::SzAxis::U1) {
                  gs_subspaces.clear();
                  for (int k = 0; k <= n_sites; ++k)
                      gs_subspaces.emplace_back(k, -1);
              } else if (ax == ed::symmetry::SzAxis::Parity) {
                  gs_subspaces = {{-1, 0}, {-1, 1}};
              }
              auto lg_o = [&](int nu, int par) {
                  ed::solvers::LittleGroupOptions o;
                  o.n_up          = nu;
                  o.sz_parity     = par;
                  o.dense_max_dim = dense_max_dim;
#ifdef WITH_CUDA
                  o.use_gpu       = use_gpu;   // batched eigensolve on the
                                               // GS-subspace scan
#endif
                  o.spin_flip     = -1;
                  o.time_reversal = time_reversal;
                  return o;
              };
              // (single-candidate fast path lives inside the helper: the
              // localization scan would solve the very same blocks that
              // little_group_ground_state re-solves below, DOUBLING the
              // cost of the whole call.)
              const auto [gs_nu, gs_par] = scan_gs_subspace(
                  op_h, abelian_group, residue_perms, n_sites,
                  gs_subspaces, lg_o, "little_group_gs_correlators");

              const auto gs = ed::solvers::little_group_ground_state(
                  op_h, abelian_group, residue_perms, n_sites,
                  lg_o(gs_nu, gs_par));

              const auto& reps = gs.rd.reps;
              const std::size_t nr = reps.size();
              if (gs.vec.size() != nr)
                  throw std::runtime_error(
                      "little_group_gs_correlators: vec/reps length mismatch.");
              std::vector<double> w(nr);
              double nrm = 0.0;
              for (std::size_t r = 0; r < nr; ++r) {
                  w[r] = std::norm(gs.vec[r]);
                  nrm += w[r];
              }
              if (!(nrm > 0.0))
                  throw std::runtime_error(
                      "little_group_gs_correlators: zero-norm GS vector.");
              const double inv_nrm = 1.0 / nrm;

              const std::size_t nd = translations.size();
              std::vector<double> C(nd, 0.0);
              for (std::size_t d = 0; d < nd; ++d) {
                  const auto& perm = translations[d];
                  if (static_cast<int>(perm.size()) != n_sites)
                      throw std::runtime_error(
                          "little_group_gs_correlators: translation length "
                          "!= n_sites.");
                  double acc = 0.0;
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
                  for (long long r = 0; r < static_cast<long long>(nr); ++r) {
                      const std::uint64_t s = reps[static_cast<std::size_t>(r)];
                      std::uint64_t t = 0;
                      for (int i = 0; i < n_sites; ++i)
                          t |= ((s >> perm[i]) & 1ULL) << i;
                      const int pc = __builtin_popcountll(s ^ t);
                      acc += w[static_cast<std::size_t>(r)]
                             * static_cast<double>(n_sites - 2 * pc) * 0.25;
                  }
                  C[d] = acc * inv_nrm / static_cast<double>(n_sites);
              }

              py::dict out;
              out["gs_energy"]   = gs.energy;
              out["gs_k0"]       = gs.k0;
              out["gs_n_up"]     = gs_nu;
              out["correlators"] = C;
              out["n_reps"]      = static_cast<std::uint64_t>(nr);
              return out;
          },
          py::arg("op"), py::arg("abelian_group"), py::arg("residue_perms"),
          py::arg("translations"), py::arg("n_up") = -1,
          py::arg("dense_max_dim") = 512, py::arg("use_gpu") = false,
          py::arg("time_reversal") = -1,
          "ONE ground-state solve -> energy AND the translation-averaged "
          "diagonal correlators C(d) = <sum_i S^z_i S^z_{i+d}>/N. Fourier "
          "transform C(d) for S^zz(Q) at EVERY momentum. Memory is one "
          "sector (no destination mirrors, no continued fractions) and the "
          "flip Z2 is exploited -- the cheap replacement for the multi-Q "
          "spectral lane when only the STATIC structure factor is wanted.");


    // -------------------------------------------------------------------------
    // Expectation values <n|O_i|n> of the lowest k levels of every block, in the
    // representative basis (no 2^N expansion): the excited-state observable lane.
    // With O_i = dH/dlambda_i these are Hellmann-Feynman derivatives.
    // -------------------------------------------------------------------------
    m.def("little_group_block_expectations",
          [lg_opts, lg_stars_dict, with_block_size](const Operator& op,
             const std::vector<const Operator*>& observables,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int k, int n_up, int sz_parity, int dense_max_dim, bool use_gpu,
             int spin_flip, int time_reversal,
             const std::vector<int>& only_k0,
             const std::vector<int>& only_irrep, int block_size) {
              const int n_sites = static_cast<int>(op.getNumBits());
              ed::solvers::LittleGroupExpectations r;
              {
                  py::gil_scoped_release release;
                  r = ed::solvers::little_group_block_expectations(
                      op, observables, abelian_group, residue_perms, n_sites, k,
                      with_block_size(lg_opts(n_up, sz_parity, dense_max_dim, use_gpu,
                                              spin_flip, time_reversal, only_k0,
                                              /*plan_only=*/false, only_irrep),
                                      block_size));
              }
              std::vector<int> kraw, fpar, irr, idim;
              std::vector<bool> conv;
              for (const auto& L : r.labels) {
                  kraw.push_back(L.k_raw);
                  fpar.push_back(L.flip_parity);
                  irr.push_back(L.irrep);
                  idim.push_back(L.irrep_dim);
                  conv.push_back(L.converged);
              }
              const std::size_t n_obs = observables.size();
              py::array_t<double> vals({static_cast<py::ssize_t>(r.values.size()),
                                        static_cast<py::ssize_t>(n_obs)});
              auto vm = vals.mutable_unchecked<2>();
              for (std::size_t i = 0; i < r.values.size(); ++i)
                  for (std::size_t j = 0; j < n_obs; ++j)
                      vm(static_cast<py::ssize_t>(i), static_cast<py::ssize_t>(j)) =
                          r.values[i][j];
              ed::solvers::LittleGroupSpectrum star_view;   // lg_stars_dict's input shape
              star_view.stars = r.stars;
              py::dict d;
              d["energies"]           = r.energies;
              d["values"]             = vals;
              d["k_raw"]              = kraw;
              d["flip_parity"]        = fpar;
              d["irrep"]              = irr;
              d["irrep_dim"]          = idim;
              d["level"]              = r.level;
              d["multiplicity"]       = r.multiplicity;
              d["converged"]          = conv;
              d["residuals"]          = r.residuals;
              d["irrep_characters"]   = r.irrep_characters;
              d["stars"]              = lg_stars_dict(star_view);
              d["flip_engaged"]       = r.flip_engaged;
              d["tr_engaged"]         = r.tr_engaged;
              d["unconverged_blocks"] = r.unconverged_blocks;
              return d;
          },
          py::arg("operator"), py::arg("observables"),
          py::arg("abelian_group"), py::arg("residue_perms"),
          py::arg("k") = 1, py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("dense_max_dim") = 256, py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("only_k0") = std::vector<int>{},
          py::arg("only_irrep") = std::vector<int>{},
          py::arg("block_size") = 1,
          "<n|O_i|n> for the lowest k levels of every (star, irrep, flip) block, "
          "computed in the momentum sector's representative basis (never expanded "
          "to 2^N). One row per (block, level): energies, values[row, i], labels "
          "aligned like little_group_lowest_eigenvalues_labeled, the rep-basis "
          "residual, and unconverged_blocks. Observables must commute with every "
          "abelian element and residue (and the spin flip / be real when those are "
          "folded), else ValueError. With O_i = dH/dlambda_i the values are "
          "Hellmann-Feynman derivatives.");
}
