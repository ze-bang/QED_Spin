"""Stage 11a-tail contract: ONE EDParameters -> SolveOptions converter.

The Python surface and the C++ CLI used to carry independent copies of
this mapping which silently drifted in three fields. These tests pin
the shared converter's semantics -- in particular the explicit flags
that encode the two callers' DELIBERATE differences -- so any future
re-fork shows up as a failing contract, not as a campaign surprise.
"""

import qed
from qed import _core
from qed._params import ed_params_to_solve_options


def _bag(**kw):
    p = _core.EDParameters()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def test_core_converter_is_bound():
    assert hasattr(_core, "ed_params_to_solve_options")


def test_scalar_fields_round_trip():
    p = _bag(num_eigenvalues=7, max_iterations=321, tolerance=1e-9,
             compute_eigenvectors=True, block_size=4,
             use_fixed_sz=True, n_up=5, output_dir="/tmp/x")
    opts = ed_params_to_solve_options(p, _core.DiagonalizationMethod.LANCZOS)
    assert opts.num_eigs == 7
    assert opts.max_iter == 321
    assert abs(opts.tolerance - 1e-9) < 1e-30
    assert opts.compute_vectors is True
    assert opts.block_size == 4
    assert opts.use_fixed_sz is True
    assert opts.n_up == 5
    assert opts.output_dir == "/tmp/x"


def test_python_surface_pins_backend():
    # wire_backend=True (the qed._params delegation): device axes become
    # backend CONSTRAINTS -- use_gpu=False must FORBID the GPU promoter.
    p = _bag(use_gpu=False, use_mpi=False)
    opts = ed_params_to_solve_options(p, _core.DiagonalizationMethod.LANCZOS)
    assert opts.backend.allow_gpu is False
    assert opts.backend.allow_mpi is False

    p2 = _bag(use_gpu=True, use_mpi=True)
    opts2 = ed_params_to_solve_options(p2, _core.DiagonalizationMethod.LANCZOS)
    assert opts2.backend.allow_gpu is True
    assert opts2.backend.allow_mpi is True


def test_cli_semantics_leave_backend_open():
    # wire_backend=False (the CLI's historical semantics): the flags do
    # NOT constrain the backend; select_backend keeps auto-promoting.
    p = _bag(use_gpu=False, use_mpi=False)
    opts = _core.ed_params_to_solve_options(
        params=p, method=_core.DiagonalizationMethod.LANCZOS,
        auto_method=False, wire_backend=False, allow_infeasible=False)
    assert opts.backend.allow_gpu is True
    assert opts.backend.allow_mpi is True


def test_method_map_and_auto_override():
    p = _bag()
    full = ed_params_to_solve_options(p, _core.DiagonalizationMethod.FULL)
    assert full.method == _core.SolveMethod.FullDiag
    auto = ed_params_to_solve_options(
        p, _core.DiagonalizationMethod.FULL, auto_method=True)
    assert auto.method == _core.SolveMethod.Auto


def test_allow_infeasible_passthrough():
    p = _bag()
    assert ed_params_to_solve_options(
        p, _core.DiagonalizationMethod.LANCZOS).allow_infeasible is False
    assert ed_params_to_solve_options(
        p, _core.DiagonalizationMethod.LANCZOS,
        allow_infeasible=True).allow_infeasible is True


def test_selected_sectors_now_mapped_on_python_path():
    # Previously the Python twin dropped this field (CLI-only mapping).
    p = _bag(selected_sectors=[0, 3, -1, 5])
    opts = ed_params_to_solve_options(p, _core.DiagonalizationMethod.LANCZOS)
    assert list(opts.selected_sectors) == [0, 3, 5]  # negatives filtered
