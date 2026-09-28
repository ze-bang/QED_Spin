// =============================================================================
// python/qed/_bindings/workflow/workflow_spectral.cpp
//
// The spectral lane: the plain `workflows_spectral` entry point.
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
              // the Hamiltonian and the observables run device-resident.
              // ``spectral_method_supports_gpu``
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
