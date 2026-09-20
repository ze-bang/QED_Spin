"""``qed._solve``: the internals of :func:`qed.solve` / :func:`qed.full_spectrum`.

``workflow.py`` grew into a 3000-line module holding the whole auto-pilot
(method picker, device picker, symmetry normalisation, the projection /
abelian / plain lanes, the dense full-spectrum sweep and the legacy
directory writers). It is split here by concern:

* :mod:`~qed._solve.methods`        -- method taxonomy, ``solver=``
  canonicalisation, the total-spin context, the orchestrator bridges
* :mod:`~qed._solve.device`         -- the ``device=`` picker and the
  (solver x device) capability matrix
* :mod:`~qed._solve.parameters`     -- ``sz=`` normalisation and the
  ``EDParameters`` builders + catalogue
* :mod:`~qed._solve.symmetry_input` -- ``symmetry=`` normalisation into the
  group ``info`` dict the kernels read
* :mod:`~qed._solve.symmetry_lane`  -- the in-memory streaming-symmetry lane
* :mod:`~qed._solve.results`        -- result shaping (project-lane backend
  envelope, SU(2) spectral-differencing labels)
* :mod:`~qed._solve.spectrum`       -- :func:`qed.full_spectrum`
* :mod:`~qed._solve.entry`          -- :func:`qed.solve`
* :mod:`~qed._solve.directories`    -- the ``.dat`` / ``automorphism_results``
  writers the directory-form bindings consume

``qed.workflow`` re-exports every name these modules define, so
``from qed.workflow import X`` keeps working unchanged.
"""
