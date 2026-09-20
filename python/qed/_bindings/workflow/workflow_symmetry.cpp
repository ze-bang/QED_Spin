// =============================================================================
// python/qed/_bindings/workflow/workflow_symmetry.cpp
//
// Term-level symmetry surface for the Python toggle layer:
// `detect_hamiltonian_symmetries`, the Stage 8d probe classifiers
// (`probe_spin_flip_character` / `probe_delta_n_up_parity`) and the A2
// generator commutation check.
//
// Split out of the former monolithic `workflow_bindings.cpp` (WP11, Sep
// 2026). The binding bodies are unchanged; the shared helpers now live in
// `workflow_bindings_internal.h`.
// =============================================================================

#include "workflow_bindings_internal.h"

using namespace workflow_bindings_detail;  // NOLINT(build/namespaces)

void bind_workflows_symmetry(py::module_& m) {
    // -----------------------------------------------------------------
    // The three entry points.
    // -----------------------------------------------------------------
    // -----------------------------------------------------------------
    // Symmetry detection for the Python toggle surface (Stage 8e):
    // term-level checks of the discrete symmetries the composition
    // layer can exploit, so qed.solve/thermal/spectral can REPORT
    // which symmetries the Hamiltonian actually carries and degrade
    // gracefully when a requested one is absent.
    // -----------------------------------------------------------------
    m.def("detect_hamiltonian_symmetries",
          [](const Operator& op) {
              ed::matvec::TermStorage soa;
              ed::matvec::TermStorage::classify_route(
                  soa, op.transform_data_, op.three_body_data_,
                  [](const std::complex<double>& c) { return c; });
              py::dict out;
              out["spin_flip"] =
                  ed::symmetry::hamiltonian_is_spin_flip_symmetric(soa);
              out["time_reversal"] = ed::symmetry::hamiltonian_is_real(soa);
              const auto ax = ed::symmetry::sz_axis_of(soa);
              out["u1"] = (ax == ed::symmetry::SzAxis::U1);
              out["sz_parity"] =
                  (ax != ed::symmetry::SzAxis::None);  // U1 implies parity
              // Stage 12 (SU(2) rollout): full spin-rotation invariance
              // (per-bond isotropic exchange, no fields / DM / 3-body).
              // Audit 2026-09: term test plus the numerical [H, S^-] fallback
              // (SU(2)-invariant three-body terms such as scalar chirality).
              out["su2"] = ed::workflows::op_is_su2_symmetric(op);
              return out;
          },
          py::arg("op"),
          R"pbdoc(
            Term-level discrete-symmetry detection.

            Returns ``{"spin_flip": bool, "time_reversal": bool}``:
            whether ``[H, prod_i sigma^x_i] == 0`` (global spin flip)
            and whether every coefficient is real in the computational
            basis (time reversal / conjugation pairing). These are the
            same checks the C++ composition layer
            (``ed::symmetry::resolve_symmetry_composition``) runs; the
            Python binding exists so the toggle surface can *report*
            the detection outcome and warn on a requested-but-absent
            symmetry instead of failing later.
          )pbdoc");

    // Stage 8d probe classifiers (Python decides whether the projected
    // spectral lanes can route a given observable BEFORE engaging them).
    m.def("probe_spin_flip_character",
          [](const std::vector<py::tuple>& rows) {
              return ed::symmetry::spin_flip_character(
                  decode_probe_transforms(rows, "probe_spin_flip_character"));
          },
          py::arg("observable_transforms"),
          "Spin-flip character of a probe's transform tuples: +1 when "
          "X O X == +O (flip-even), -1 when X O X == -O (flip-odd; e.g. "
          "any S^z_Q probe), 0 when O has no definite character (e.g. a "
          "lone S^+ probe) and cannot ride the flip-projected DSSF lane.");
    m.def("probe_delta_n_up_parity",
          [](const std::vector<py::tuple>& rows) {
              return ed::symmetry::delta_n_up_parity(
                  decode_probe_transforms(rows, "probe_delta_n_up_parity"));
          },
          py::arg("observable_transforms"),
          "Set-bit-parity selection rule of a probe: 0 = parity-even "
          "(stays in its Sz-parity half), 1 = parity-odd (crosses "
          "halves), -1 = mixed (cannot ride the parity DSSF lane).");

    // A2: term-level [H, U_g] = 0 validation for EXPLICIT generator sets. The
    // abelian rep lane trusts its generators; an auto automorphism commutes by
    // construction, but a hand-supplied permutation list is unchecked and a
    // wrong one yields silently-wrong spectra. Returns one bool per generator.
    m.def("check_generators_commute",
          [](const Operator& op,
             const std::vector<std::vector<int>>& generators) {
              std::vector<bool> out;
              out.reserve(generators.size());
              for (const auto& g : generators)
                  out.push_back(
                      ed::symmetry::hamiltonian_commutes_with_permutation(
                          op.transform_data_, op.three_body_data_, g));
              return out;
          },
          py::arg("op"), py::arg("generators"),
          "Per-generator [H, U_g] = 0 check (term-level, exact, no matvec): "
          "True iff relabelling H's term sites by the permutation leaves the "
          "term multiset invariant. Used to validate explicit / bridge-"
          "supplied generator sets before the rep lane trusts them.");

}
