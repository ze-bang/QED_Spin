"""``qed.spectral``: canonical spectral / structure-factor entry point.

This module implements :func:`qed.spectral`, the one Python verb for
zero- and finite-temperature dynamical / static structure factors and
related spectral functions.

Two call shapes are supported:

* **In-memory:** ``qed.spectral(H, observables, ...)`` -- runs the
  unified C++ orchestrator (``ed::workflows::spectral``) directly on
  the supplied operator and observable list. Used for programmatic
  workflows and notebook prototyping.

* **Directory form:** ``qed.spectral(directory, ...)`` -- shells out to
  the canonical ``./ED dssf <method> <directory>`` CLI workflow, which
  handles the full DSSF / SSSF / static-response pipeline with HDF5
  outputs. Used for production runs.

The CLI form is the same code path the old ``qed.dssf.compute`` /
``qed.dssf.run_from_directory`` helpers exercised; those names were
removed during the May-2026 surface unification.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from typing import Iterable, Optional, Sequence, Union, Any

from . import _core
from ._core import (  # type: ignore[attr-defined]
    Operator,
    FixedSzOperator,
)
from .auto_tune import TunedDSSFKnobs, tune_dssf as _tune_dssf

__all__ = [
    "spectral",
    "TunedDSSFKnobs",
]


_VALID_CLI_METHODS: tuple[str, ...] = (
    "dynamical_thermal",
    "static_thermal",
    "ground_state_dssf",
    "single_expectation",
    "kpm_thermodynamics",
)


def _pick_cli_method(
    *,
    T: Optional[Union[float, Iterable[float]]],
    omega: Optional[Iterable[float]],
) -> str:
    """Choose the ``./ED dssf`` method token from the (T, omega) tuple."""
    has_T = T is not None
    has_w = omega is not None
    if has_T and has_w:
        return "dynamical_thermal"
    if has_T:
        return "static_thermal"
    if has_w:
        return "ground_state_dssf"
    return "single_expectation"


def _resolve_ed_binary(ed_binary: Optional[str]) -> str:
    if ed_binary:
        if not os.path.isfile(ed_binary):
            raise FileNotFoundError(
                f"ed_binary={ed_binary!r} does not exist; pass an absolute path"
            )
        return ed_binary
    on_path = shutil.which("ED")
    if on_path is not None:
        return on_path
    raise FileNotFoundError(
        "Could not find the `ED` binary. Either build it (cmake --build "
        "<build> --target ED), put the build directory on $PATH, or pass "
        "ed_binary=/abs/path/to/ED to qed.spectral(directory, ...)."
    )


def _spectral_directory(
    directory: str,
    *,
    T: Optional[Union[float, Iterable[float]]],
    omega: Optional[Iterable[float]],
    method: Optional[str],
    eta: Optional[float],
    krylov_dim: Optional[int],
    num_random_vectors: Optional[int],
    kpm_moments: Optional[int],
    bandwidth: Optional[float],
    device: Optional[str],
    level: str,
    sector_dim: Optional[int],
    operator: Optional[object],
    auto_tune: bool,
    ed_binary: Optional[str],
    extra_args: Sequence[str],
    env: Optional[dict[str, str]],
    check: bool,
    capture_output: bool,
    verbose: bool,
) -> subprocess.CompletedProcess:
    """Shell out to ``./ED dssf <method> <directory>``."""
    if method is None:
        chosen = _pick_cli_method(T=T, omega=omega)
    else:
        if method not in _VALID_CLI_METHODS:
            raise ValueError(
                f"method={method!r} is not a recognised DSSF method token. "
                f"Valid tokens: {_VALID_CLI_METHODS}."
            )
        chosen = method

    auto_args: list[str] = []
    tuned: Optional[TunedDSSFKnobs] = None
    if auto_tune and chosen != "single_expectation":
        try:
            from . import has_cuda_build, has_mpi_build  # local import
            cuda_ok = bool(has_cuda_build())
            mpi_ok = bool(has_mpi_build())
        except Exception:
            cuda_ok = mpi_ok = False
        tuned = _tune_dssf(
            operator=operator,
            sector_dim=sector_dim,
            bandwidth=bandwidth,
            omega=omega,
            eta=eta,
            krylov_dim=krylov_dim,
            num_random_vectors=num_random_vectors,
            kpm_moments=kpm_moments,
            device=device,
            has_cuda_build=cuda_ok,
            has_mpi_build=mpi_ok,
            level=level,
        )
        auto_args = tuned.to_cli_args(method=chosen)

    if verbose:
        has_T = T is not None
        has_w = omega is not None
        msg = (f"[qed.spectral] method={chosen!r} "
               f"(T given: {has_T}, omega given: {has_w})")
        if tuned is not None:
            msg += (f" | auto-tuned [{tuned.level}]: "
                    f"eta={tuned.eta:.4g}, "
                    f"krylov={tuned.krylov_dim}, "
                    f"R={tuned.num_random_vectors}, "
                    f"omega=[{tuned.omega_min:.4g},{tuned.omega_max:.4g}]"
                    f" x {tuned.num_omega_points}, "
                    f"device={tuned.device}")
        print(msg)

    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"directory={directory!r} does not exist or is not a directory"
        )

    binary = _resolve_ed_binary(ed_binary)
    cmd = [binary, "dssf", chosen, directory,
           *tuple(auto_args), *tuple(extra_args)]
    return subprocess.run(
        cmd,
        check=check,
        env=env,
        capture_output=capture_output,
        text=True,
    )


def _extract_transforms(observable: Any) -> list:
    """Extract the ``Operator::TransformData`` list from a Python-side
    observable spec for the cross-irrep spectral path.

    Accepts an ``_core.Operator`` instance directly (we read its
    ``transform_data_`` -- in the absence of a Python accessor we
    rely on the SoA layout being mirrored via the dedicated helper
    below) or a pre-built list of 6-tuples shaped like the C++
    binding expects: ``(op_type, site, coeff, is_two_body, op_type_2,
    site_2)``.
    """
    if observable is None:
        return []
    if isinstance(observable, list):
        return [tuple(row) for row in observable]
    if hasattr(observable, "transform_tuples"):
        return [tuple(t) for t in observable.transform_tuples()]
    raise TypeError(
        "qed.spectral cross-irrep symmetry: `observable` must be either "
        "(a) an _core.Operator with a transform_tuples() helper, or "
        "(b) an explicit list of 6-tuples "
        "(op_type:int, site:int, coeff:complex, is_two_body:bool, "
        "op_type_2:int, site_2:int) describing the probe O_Q. "
        f"Got: {type(observable).__name__}"
    )


def _infer_delta_n_up(transforms: list) -> int:
    """Infer the Sz selection rule of a probe from its transform tuples.

    In this codebase the fixed-Sz axis counts SET bits, and a set bit
    is the state S- creates (verified against the plain full-Hilbert
    lane at 1e-11): an S- probe (op_type 1) RAISES the sector''s n_up
    by 1, S+ (op_type 0) lowers it, Sz / paired two-body combinations
    conserve it. A probe mixing S+ and S- one-body terms has no single
    selection rule; the caller must pass ``delta_n_up`` explicitly
    (dict form) in that case.
    """
    deltas = set()
    for (op1, _s1, _c, is2, op2, _s2) in transforms:
        if is2:
            d = (-1 if op1 == 0 else 1 if op1 == 1 else 0) \
                + (-1 if op2 == 0 else 1 if op2 == 1 else 0)
        else:
            d = -1 if op1 == 0 else 1 if op1 == 1 else 0
        deltas.add(d)
    if len(deltas) == 1:
        return deltas.pop()
    raise ValueError(
        "qed.spectral symmetry: the observable mixes terms with "
        f"different Sz selection rules ({sorted(deltas)}); pass the "
        "directory form with symmetry={'delta_n_up': ...} to pick one."
    )


def _spectral_in_memory_with_symmetry(
    H,
    observables,
    *,
    symmetry,
    sz,
    T,
    omega,
    method,
    eta,
    krylov_dim,
    num_random_vectors,
    energy_shift,
    momentum_transfer,
    momentum_tolerance,
    selected_sectors,
    output_dir,
    observable_type,
    spin_l,
    verbose,
):
    """Route an IN-MEMORY spectral call through the streaming-symmetry
    machinery: resolve ``symmetry`` (including ``"auto"``), export the
    operator + automorphism metadata to a temp directory, extract each
    observable's transform tuples, and dispatch to the cross-irrep
    GS-CF (T is None) or FTLM (finite T) C++ binding.

    Returns ``NotImplemented`` when the request cannot be routed (no
    spatial symmetry found, an observable without ``transform_tuples``,
    an unsupported method) -- the caller then falls back to the plain
    in-memory lane.
    """
    import shutil as _shutil
    import tempfile as _tempfile

    from .workflow import (
        resolve_auto_symmetry as _resolve_auto,
        _write_operator_directory as _write_op_dir,
        _write_symmetry_directory as _write_sym_dir,
        _normalize_symmetry_info as _norm_sym_info,
    )

    if isinstance(symmetry, dict):
        raise TypeError(
            "qed.spectral(H, observables, symmetry=dict) is a "
            "directory-form spec; for an in-memory operator pass "
            "symmetry='auto', a GeneratorSet, or a permutation list."
        )
    gen = _resolve_auto(H, symmetry, verbose=verbose)
    if gen is None:
        return NotImplemented          # trivial group: plain lane
    m_norm = (method or "ground_state_cf").lower().replace("-", "_")
    if T is None and m_norm not in ("ground_state_cf",
                                    "ground_state_dssf"):
        if verbose:
            print(f"[qed.spectral] symmetry with method={method!r} is "
                  "not routed through the sector machinery yet; "
                  "running the plain in-memory lane.")
        return NotImplemented
    if omega is None:
        return NotImplemented
    if momentum_transfer is None:
        if verbose:
            print("[qed.spectral] symmetry= with an in-memory operator "
                  "needs momentum_transfer=[...] to pick the destination "
                  "irrep of the probe (e.g. the Q you built into the "
                  "observable's phases); running the plain in-memory "
                  "lane instead.")
        return NotImplemented

    transforms_per_obs = []
    for obs in observables:
        try:
            tuples = _extract_transforms(obs)
        except TypeError:
            return NotImplemented
        if not tuples:
            return NotImplemented
        transforms_per_obs.append(tuples)

    tmpdir = _tempfile.mkdtemp(prefix="qed_spectral_sym_")
    try:
        _write_op_dir(H, tmpdir)
        info = _norm_sym_info(H, gen)
        if info is None:
            return NotImplemented
        _write_sym_dir(tmpdir, info)

        results = []
        for tuples in transforms_per_obs:
            delta = _infer_delta_n_up(tuples)
            if T is None:
                results.append(
                    _spectral_streaming_symmetry_cross_irrep_directory(
                        tmpdir,
                        num_sites=int(H.num_sites),
                        spin_l=float(spin_l),
                        fixed_sz_n_up=(int(sz) if sz is not None else None),
                        omega=omega, eta=eta, krylov_dim=krylov_dim,
                        energy_shift=energy_shift,
                        momentum_transfer=momentum_transfer,
                        momentum_tolerance=momentum_tolerance,
                        selected_sectors=selected_sectors,
                        observable_transforms=tuples,
                        delta_n_up=delta,
                        output_dir=output_dir,
                        observable_type=observable_type,
                        verbose=verbose,
                    ))
            else:
                Ts_list = T if isinstance(T, (list, tuple)) else [T]
                results.append(
                    _spectral_streaming_symmetry_ftlm_cross_irrep_directory(
                        tmpdir,
                        num_sites=int(H.num_sites),
                        spin_l=float(spin_l),
                        fixed_sz_n_up=(int(sz) if sz is not None else None),
                        omega=omega, eta=eta, krylov_dim=krylov_dim,
                        momentum_transfer=momentum_transfer,
                        momentum_tolerance=momentum_tolerance,
                        selected_sectors=selected_sectors,
                        observable_transforms=tuples,
                        delta_n_up=delta,
                        temperatures=list(Ts_list),
                        num_samples=int(num_random_vectors or 30),
                        random_seed=0,
                        output_dir=output_dir,
                        observable_type=observable_type,
                        verbose=verbose,
                    ))
        return results[0] if len(results) == 1 else results
    finally:
        _shutil.rmtree(tmpdir, ignore_errors=True)


def _spectral_streaming_symmetry_cross_irrep_directory(
    directory: str,
    *,
    num_sites: int,
    spin_l: float,
    fixed_sz_n_up: Optional[int],
    omega: Optional[Iterable[float]],
    eta: Optional[float],
    krylov_dim: Optional[int],
    energy_shift: Optional[float],
    momentum_transfer: Optional[Sequence[float]],
    momentum_tolerance: float,
    selected_sectors: Optional[Sequence[int]],
    observable_transforms: list,
    delta_n_up: int,
    output_dir: str,
    observable_type: str,
    verbose: bool,
) -> Any:
    """SOTA cross-irrep streaming-symmetry spectral routing the call to
    ``_core.workflows_spectral_streaming_symmetry_cross_irrep_directory``.
    """
    opts = _core.SpectralOptions()
    opts.method = _core.SpectralMethod.GroundStateCF
    if krylov_dim is not None:
        opts.krylov_dim = int(krylov_dim)
    if eta is not None:
        opts.broadening = float(eta)
    if energy_shift is not None:
        opts.energy_shift = float(energy_shift)
    if output_dir:
        opts.output_dir = str(output_dir)
    if observable_type:
        opts.observable_type = str(observable_type)
    if omega is not None:
        ws = list(omega)
        if len(ws) >= 2:
            opts.omega_min = float(min(ws))
            opts.omega_max = float(max(ws))
            opts.num_omega = int(len(ws))
    if momentum_transfer is not None:
        opts.momentum_transfer = [float(q) for q in momentum_transfer]
    opts.momentum_tolerance = float(momentum_tolerance)
    if selected_sectors is not None:
        opts.selected_sectors = [int(k) for k in selected_sectors]

    if verbose:
        print(
            f"[qed.spectral] cross-irrep streaming-symmetry: "
            f"directory={directory!r}  N={num_sites}  "
            f"fixed_sz_n_up={fixed_sz_n_up}  delta_n_up={delta_n_up}  "
            f"Q={list(opts.momentum_transfer)}  "
            f"terms={len(observable_transforms)}"
        )
    return _core.workflows_spectral_streaming_symmetry_cross_irrep_directory(
        directory,
        int(num_sites),
        float(spin_l),
        observable_transforms,
        opts,
        fixed_sz_n_up,
        int(delta_n_up),
    )


def _spectral_streaming_symmetry_cross_irrep_multiq_directory(
    directory: str,
    *,
    num_sites: int,
    spin_l: float,
    fixed_sz_n_up: Optional[int],
    omega: Optional[Iterable[float]],
    eta: Optional[float],
    krylov_dim: Optional[int],
    energy_shift: Optional[float],
    momentum_points: Sequence[Sequence[float]],
    momentum_tolerance: float,
    selected_sectors: Optional[Sequence[int]],
    observable_transforms_per_q: list,
    delta_n_up: int,
    output_dir: str,
    observable_type: str,
    verbose: bool,
) -> Any:
    """SOTA amortised multi-Q cross-irrep streaming-symmetry spectral
    routing to
    ``_core.workflows_spectral_streaming_symmetry_cross_irrep_multiq_directory``.

    The ground state is solved ONCE and reused across every Q in
    ``momentum_points``; each Q only pays a selection-rule lookup, a
    ``CrossSectorOrbitObservable`` scatter, and one inner CF-Lanczos in
    the reduced target sector. ``observable_transforms_per_q`` is a list
    of transform lists aligned 1:1 with ``momentum_points`` (each Q owns
    its phased observable O_Q). Results land in ``per_sector_pair``
    positionally aligned with ``momentum_points``: ``.S_real`` is the
    dynamical S(Q, omega) and ``.static_sf`` is the equal-time S(Q)
    (the SSSF, obtained for free from the CF pivot norm).
    """
    opts = _core.SpectralOptions()
    opts.method = _core.SpectralMethod.GroundStateCF
    if krylov_dim is not None:
        opts.krylov_dim = int(krylov_dim)
    if eta is not None:
        opts.broadening = float(eta)
    if energy_shift is not None:
        opts.energy_shift = float(energy_shift)
    if output_dir:
        opts.output_dir = str(output_dir)
    if observable_type:
        opts.observable_type = str(observable_type)
    if omega is not None:
        ws = list(omega)
        if len(ws) >= 2:
            opts.omega_min = float(min(ws))
            opts.omega_max = float(max(ws))
            opts.num_omega = int(len(ws))
    opts.momentum_tolerance = float(momentum_tolerance)
    if selected_sectors is not None:
        opts.selected_sectors = [int(k) for k in selected_sectors]

    q_points = [[float(c) for c in q] for q in momentum_points]
    if not q_points:
        raise ValueError(
            "qed.spectral multi-Q cross-irrep: momentum_points is empty."
        )
    if len(observable_transforms_per_q) != len(q_points):
        raise ValueError(
            "qed.spectral multi-Q cross-irrep: observable list must be "
            f"aligned 1:1 with momentum_points (got "
            f"{len(observable_transforms_per_q)} observables for "
            f"{len(q_points)} Q-points)."
        )
    if verbose:
        print(
            f"[qed.spectral] multi-Q cross-irrep streaming-symmetry: "
            f"directory={directory!r}  N={num_sites}  "
            f"fixed_sz_n_up={fixed_sz_n_up}  delta_n_up={delta_n_up}  "
            f"n_Q={len(q_points)} (single amortised GS solve)"
        )
    return _core.workflows_spectral_streaming_symmetry_cross_irrep_multiq_directory(
        directory,
        int(num_sites),
        float(spin_l),
        observable_transforms_per_q,
        q_points,
        opts,
        fixed_sz_n_up,
        int(delta_n_up),
    )


def _spectral_streaming_symmetry_ftlm_cross_irrep_directory(
    directory: str,
    *,
    num_sites: int,
    spin_l: float,
    fixed_sz_n_up: Optional[int],
    omega: Optional[Iterable[float]],
    eta: Optional[float],
    krylov_dim: Optional[int],
    momentum_transfer: Optional[Sequence[float]],
    momentum_tolerance: float,
    selected_sectors: Optional[Sequence[int]],
    observable_transforms: list,
    delta_n_up: int,
    temperatures: Sequence[float],
    num_samples: int,
    random_seed: int,
    output_dir: str,
    observable_type: str,
    verbose: bool,
) -> Any:
    """SOTA finite-T cross-irrep streaming-symmetry spectral routing to
    ``_core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory``.

    The C++ binding performs the per-source-sector FTLM walk:
    draw random samples in each source orbit basis, outer-Lanczos on
    H restricted to the sector, scatter Ritz states into the target
    sector via ``CrossSectorOrbitObservable``, inner-Lanczos in the
    target sector, Lehmann sum + F-shifted Z-weighted recombination
    across source sectors. Result carries the recombined S(omega, T)
    at the *first* temperature in ``agg.S_real``; the full T-resolved
    payload sits inside ``agg.per_sector_pair`` and is unpacked into
    a clean dict by this wrapper before returning to the caller.
    """
    opts = _core.SpectralOptions()
    opts.method = _core.SpectralMethod.FtlmDynamical
    if krylov_dim is not None:
        opts.krylov_dim = int(krylov_dim)
    if eta is not None:
        opts.broadening = float(eta)
    if output_dir:
        opts.output_dir = str(output_dir)
    if observable_type:
        opts.observable_type = str(observable_type)
    if omega is not None:
        ws = list(omega)
        if len(ws) >= 2:
            opts.omega_min = float(min(ws))
            opts.omega_max = float(max(ws))
            opts.num_omega = int(len(ws))
    if momentum_transfer is not None:
        opts.momentum_transfer = [float(q) for q in momentum_transfer]
    opts.momentum_tolerance = float(momentum_tolerance)
    if selected_sectors is not None:
        opts.selected_sectors = [int(k) for k in selected_sectors]
    Ts = [float(t) for t in temperatures]
    if not Ts:
        raise ValueError(
            "qed.spectral finite-T cross-irrep: temperatures is empty."
        )
    if verbose:
        print(
            f"[qed.spectral] FTLM cross-irrep streaming-symmetry: "
            f"directory={directory!r}  N={num_sites}  "
            f"fixed_sz_n_up={fixed_sz_n_up}  delta_n_up={delta_n_up}  "
            f"Q={list(opts.momentum_transfer)}  T={Ts}  "
            f"num_samples={num_samples}  terms={len(observable_transforms)}"
        )
    agg = _core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep_directory(
        directory,
        int(num_sites),
        float(spin_l),
        observable_transforms,
        opts,
        fixed_sz_n_up,
        int(delta_n_up),
        Ts,
        int(num_samples),
        int(random_seed),
    )

    # The binding stuffs the multi-T payload into per_sector_pair as
    # extra synthetic entries (one per T) with
    # ``initial.sector_dim == 0`` as the sentinel. We unpack into a
    # clean ``{T: list[float]}`` dict and attach via Python's
    # dynamic-attribute support (the SpectralResult pybind11 class is
    # declared with ``py::dynamic_attr()`` so this works directly).
    # The genuine per-(k_src, k_dst) entries remain inside
    # ``agg.per_sector_pair``; we additionally surface them through
    # ``agg.sector_pairs`` so downstream code does not have to filter
    # by the sentinel flag.
    num_T = len(Ts)
    entries = list(agg.per_sector_pair)
    if num_T > 0 and len(entries) >= num_T:
        multi_T_slice = entries[-num_T:]
        all_sentinel = all(
            getattr(e.initial, "sector_dim", 0) == 0
            for e in multi_T_slice
        )
        if all_sentinel:
            agg.S_by_T_real = {
                T: list(e.S_real)
                for T, e in zip(Ts, multi_T_slice)
            }
            agg.S_by_T_imag = {
                T: list(e.S_imag)
                for T, e in zip(Ts, multi_T_slice)
            }
            agg.temperatures = list(Ts)
            agg.sector_pairs = entries[:-num_T]
    return agg


def _spectral_streaming_symmetry_directory(
    directory: str,
    *,
    num_sites: int,
    spin_l: float,
    fixed_sz_n_up: Optional[int],
    omega: Optional[Iterable[float]],
    method: Optional[str],
    eta: Optional[float],
    krylov_dim: Optional[int],
    num_random_vectors: Optional[int],
    energy_shift: Optional[float],
    momentum_transfer: Optional[Sequence[float]],
    momentum_tolerance: float,
    selected_sectors: Optional[Sequence[int]],
    output_dir: str,
    observable_type: str,
    verbose: bool,
) -> Any:
    """SOTA streaming-symmetry spectral via the C++ binding
    ``_core.workflows_spectral_streaming_symmetry_directory``.

    Walks per-irrep ground-state Lanczos in pass 1 to find the irrep
    containing the global GS, then runs continued-fraction Lanczos
    inside that single sector. The Δk selection rule is annotated on
    the result (``selection_rule_label``); cross-irrep transitions
    are deferred to the orbit-basis cross-sector observable
    (``docs/architecture/SYMMETRY.md`` Section 3).
    """
    opts = _core.SpectralOptions()
    if method is not None:
        key = method.upper().replace("-", "_")
        if key in ("GROUND_STATE_CF", "GROUND_STATE_DSSF", "GROUNDSTATECF"):
            opts.method = _core.SpectralMethod.GroundStateCF
        elif key in ("FTLM_DYNAMICAL", "DYNAMICAL_THERMAL", "FTLMDYNAMICAL"):
            opts.method = _core.SpectralMethod.FtlmDynamical
        else:
            raise ValueError(
                f"method={method!r} not supported by the streaming-"
                f"symmetry spectral path; use 'ground_state_cf'."
            )
    if krylov_dim is not None:
        opts.krylov_dim = int(krylov_dim)
    if eta is not None:
        opts.broadening = float(eta)
    if num_random_vectors is not None:
        opts.num_samples = int(num_random_vectors)
    if energy_shift is not None:
        opts.energy_shift = float(energy_shift)
    if output_dir:
        opts.output_dir = str(output_dir)
    if observable_type:
        opts.observable_type = str(observable_type)
    if omega is not None:
        ws = list(omega)
        if len(ws) >= 2:
            opts.omega_min = float(min(ws))
            opts.omega_max = float(max(ws))
            opts.num_omega = int(len(ws))
    if momentum_transfer is not None:
        opts.momentum_transfer = [float(q) for q in momentum_transfer]
    opts.momentum_tolerance = float(momentum_tolerance)
    if selected_sectors is not None:
        opts.selected_sectors = [int(k) for k in selected_sectors]

    if verbose:
        print(
            f"[qed.spectral] streaming-symmetry: "
            f"directory={directory!r}  N={num_sites}  "
            f"fixed_sz_n_up={fixed_sz_n_up}  method={opts.method}  "
            f"selected_sectors={list(opts.selected_sectors)}"
        )
    return _core.workflows_spectral_streaming_symmetry_directory(
        directory,
        int(num_sites),
        float(spin_l),
        opts,
        fixed_sz_n_up,
    )


def _spectral_in_memory(
    H: Operator,
    observables: Sequence[Operator],
    *,
    omega: Optional[Iterable[float]],
    method: Optional[str],
    eta: Optional[float],
    krylov_dim: Optional[int],
    num_random_vectors: Optional[int],
    energy_shift: Optional[float],
    output_dir: str,
    T: Optional[Union[float, Iterable[float]]],
    observable_type: str,
    verbose: bool,
    # Phase D of the "Backend x Symmetries x Workflows" plan
    # (May 2026): explicit device selector for the in-memory spectral
    # path. ``None`` / ``"auto"`` lets the C++ ``select_backend`` pick;
    # ``"cpu"`` / ``"gpu"`` pins the choice via
    # ``SpectralOptions.backend.allow_gpu``.
    device: Optional[str] = None,
    # Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026):
    # caller-supplied seed state for the GroundStateCF lane. Accepts a
    # numpy array, list, or anything h5py reads as a 1D complex vector.
    initial_state: Optional[Any] = None,
    # Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026):
    # KpmDynamical knobs.
    kpm_moments: Optional[int] = None,
    kpm_kernel: Optional[str] = None,
    kpm_lorentz_lambda: Optional[float] = None,
) -> Any:
    """Build a `_core.SpectralOptions` and call `_core.workflows_spectral`."""
    opts = _core.SpectralOptions()
    if method is not None:
        key = method.upper().replace("-", "_")
        if key in ("GROUND_STATE_CF", "GROUND_STATE_DSSF", "GROUNDSTATECF"):
            opts.method = _core.SpectralMethod.GroundStateCF
        elif key in ("FTLM_DYNAMICAL", "DYNAMICAL_THERMAL", "FTLMDYNAMICAL"):
            opts.method = _core.SpectralMethod.FtlmDynamical
        elif key in ("KPM_DYNAMICAL", "KPM_DYN", "KPMDYNAMICAL"):
            opts.method = _core.SpectralMethod.KpmDynamical
        else:
            raise ValueError(
                f"method={method!r} not supported by the in-memory "
                f"spectral path. Use 'ground_state_cf', "
                f"'ftlm_dynamical', or 'kpm_dynamical', or pass a "
                f"directory path to use the CLI form."
            )
    if krylov_dim is not None:
        opts.krylov_dim = int(krylov_dim)
    if eta is not None:
        opts.broadening = float(eta)
    if num_random_vectors is not None:
        opts.num_samples = int(num_random_vectors)
    if energy_shift is not None:
        opts.energy_shift = float(energy_shift)
    if output_dir:
        opts.output_dir = str(output_dir)
    if observable_type:
        opts.observable_type = str(observable_type)
    if omega is not None:
        ws = list(omega)
        if len(ws) >= 2:
            opts.omega_min = float(min(ws))
            opts.omega_max = float(max(ws))
            opts.num_omega = int(len(ws))
    if T is not None:
        if hasattr(T, "__iter__"):
            opts.temperatures = [float(t) for t in T]
        else:
            opts.temperatures = [float(T)]

    # Phase D of the "Backend x Symmetries x Workflows" plan
    # (May 2026): map ``device=`` to ``opts.backend.allow_gpu``. The
    # in-memory path used to silently drop ``device=`` from the
    # dispatcher; this closes that gap so qed.spectral(H, device='gpu')
    # actually lands on CudaBackend.
    if device is not None:
        dev_lc = str(device).lower()
        if dev_lc not in ("auto", "cpu", "gpu", "mpi", "mpi_gpu"):
            raise ValueError(
                f"qed.spectral: device={device!r} not in "
                "{'auto', 'cpu', 'gpu', 'mpi', 'mpi_gpu'}."
            )
        if dev_lc == "gpu":
            if not _core.has_cuda_build():
                raise RuntimeError(
                    "qed.spectral(device='gpu') requested but this "
                    "build of qed._core does not have WITH_CUDA=ON.")
            opts.backend.allow_gpu = True
            opts.backend.allow_mpi = False
        elif dev_lc == "cpu":
            opts.backend.allow_gpu = False
            opts.backend.allow_mpi = False
        elif dev_lc in ("mpi", "mpi_gpu"):
            raise NotImplementedError(
                f"qed.spectral(device={device!r}): MPI spectral is "
                "not wired in the in-memory binding yet -- use the "
                "directory form with a launched MPI binary.")
        # "auto" => let select_backend decide (default behaviour).

    # Pillar 3 (May 2026): user-supplied seed for the GroundStateCF
    # lane. The C++ binding accepts a Python list of complex; numpy
    # arrays are coerced via ``.tolist()`` to keep the conversion
    # robust across structured (real/imag) and native complex dtypes.
    if initial_state is not None:
        import numpy as _np
        arr = _np.asarray(initial_state)
        if arr.dtype.names is not None and {"real", "imag"}.issubset(arr.dtype.names):
            arr = arr["real"] + 1j * arr["imag"]
        arr = arr.astype(_np.complex128, copy=False).ravel()
        opts.initial_state = [complex(z) for z in arr.tolist()]

    # Pillar 4 (May 2026): KPM-dynamical knobs. Only the moments knob
    # is forwarded unconditionally (it is meaningful for any future
    # KPM expansion lane). The kernel + lambda knobs are forwarded
    # opaquely; the C++ binding stays the source of truth for
    # validation.
    if kpm_moments is not None:
        opts.kpm_moments = int(kpm_moments)
    if kpm_kernel is not None:
        k_lc = str(kpm_kernel).lower()
        if k_lc in ("jackson", "j"):
            opts.kpm_kernel = _core.SpectralKpmKernel.Jackson
        elif k_lc in ("lorentz", "l"):
            opts.kpm_kernel = _core.SpectralKpmKernel.Lorentz
        else:
            raise ValueError(
                f"qed.spectral: kpm_kernel={kpm_kernel!r} not in "
                "{'Jackson', 'Lorentz'}."
            )
    if kpm_lorentz_lambda is not None:
        opts.kpm_lorentz_lambda = float(kpm_lorentz_lambda)

    obs_list = list(observables)
    if verbose:
        print(
            f"[qed.spectral] in-memory: dim={H.dimension}, "
            f"observables={len(obs_list)}, method={opts.method}"
        )
    return _core.workflows_spectral(H, obs_list, opts)


def spectral(
    H_or_directory: Union[Operator, FixedSzOperator, str],
    observables: Optional[Sequence[Operator]] = None,
    *,
    T: Optional[Union[float, Iterable[float]]] = None,
    omega: Optional[Iterable[float]] = None,
    method: Optional[str] = None,
    # In-memory + CLI shared knobs ---------------------------------------
    eta: Optional[float] = None,
    krylov_dim: Optional[int] = None,
    num_random_vectors: Optional[int] = None,
    # In-memory only -----------------------------------------------------
    energy_shift: Optional[float] = None,
    output_dir: str = "",
    observable_type: str = "",
    # Shared between CLI and in-memory KpmDynamical (Pillar 4 of the
    # "Save and DSSF Upgrades" plan, May 2026). When the in-memory
    # ``method="kpm_dynamical"`` lane is selected, these are forwarded
    # to ``_core.SpectralOptions.kpm_*``; otherwise they only affect
    # the CLI ``kpm_thermodynamics`` shell-out.
    kpm_moments: Optional[int] = None,
    kpm_kernel: Optional[str] = None,
    kpm_lorentz_lambda: Optional[float] = None,
    bandwidth: Optional[float] = None,
    device: Optional[str] = None,
    level: str = "balanced",
    sector_dim: Optional[int] = None,
    operator: Optional[object] = None,
    auto_tune: bool = True,
    ed_binary: Optional[str] = None,
    extra_args: Sequence[str] = (),
    env: Optional[dict[str, str]] = None,
    check: bool = True,
    capture_output: bool = False,
    verbose: bool = True,
    # SOTA streaming-symmetry knobs (May 2026) ---------------------------
    symmetry: Union[bool, str, dict, None] = None,
    num_sites: Optional[int] = None,
    spin_l: float = 0.5,
    sz: Optional[int] = None,
    momentum_transfer: Optional[Sequence[float]] = None,
    momentum_tolerance: float = 1e-6,
    selected_sectors: Optional[Sequence[int]] = None,
    # Stage 8e (SymmetryEngine v2): per-symmetry toggles. The spectral
    # solver exploits the U(1) x spatial sector machinery; the discrete
    # spin-flip / time-reversal mechanisms are NOT consumed by the
    # spectral lanes yet (Stage 8d), so these kwargs only DETECT and
    # REPORT: 'on' confirms or warns, 'require' throws when absent.
    spin_flip: Union[str, bool, int, None] = "auto",
    time_reversal: Union[str, bool, int, None] = "auto",
    # Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026) --------
    initial_state: Optional[Any] = None,
):
    """Spectral / structure-factor calculation. Auto-routes by input shape.

    Polymorphic over the first positional argument:

    * If ``H_or_directory`` is an :class:`Operator` (or
      :class:`FixedSzOperator`), the call runs the in-memory C++
      orchestrator (``_core.workflows_spectral``) on ``H`` and
      ``observables``. ``observables`` is required in this form.

    * If ``H_or_directory`` is a string, it is interpreted as a
      directory containing ``parameters.def`` plus the Hamiltonian
      deck, and the call shells out to ``./ED dssf <method>
      <directory>``. Observables are assembled by the CLI from
      ``parameters.def`` in this form; the ``observables=`` kwarg is
      ignored.

    Parameters common to both forms
    -------------------------------
    T : float, sequence of floats, or None
        Temperature axis (None means T=0).
    omega : sequence of floats or None
        Frequency grid (None means no omega axis).
    method : str, optional
        Explicit method token. For in-memory: ``"ground_state_cf"`` or
        ``"ftlm_dynamical"``. For CLI: ``"dynamical_thermal"``,
        ``"static_thermal"``, ``"ground_state_dssf"``,
        ``"single_expectation"``, ``"kpm_thermodynamics"``. When
        omitted, the method is auto-picked from ``T`` / ``omega``.
    eta : float, optional
        Lorentzian broadening (passes to the C++ ``broadening`` knob
        in-memory or to ``--dyn-broadening`` for the CLI).
    krylov_dim : int, optional
        Continued-fraction / FTLM Krylov subspace dimension.
    num_random_vectors : int, optional
        FTLM number of random initial vectors.
    verbose : bool, optional
        Print one-line progress.

    Parameters specific to the in-memory form
    -----------------------------------------
    energy_shift : float, optional
        Spectral energy shift (e.g. subtract ground-state energy).
    output_dir : str, optional
        Where the C++ engine writes HDF5 artifacts.
    observable_type : str, optional
        Label used in the HDF5 group naming (e.g. "Sz").

    Parameters specific to the directory form
    -----------------------------------------
    kpm_moments : int, optional
        Number of KPM Chebyshev moments (for ``kpm_thermodynamics``).
    bandwidth : float, optional
        Spectral bandwidth W for default omega/eta picking.
    device : str, optional
        Backend selector for the CLI (``"cpu"``, ``"gpu"``,
        ``"mpi"``, ``"mpi_gpu"``).
    level : {"conservative", "balanced", "aggressive"}, optional
        Auto-tune aggressiveness.
    sector_dim, operator : optional
        Inputs for the auto-tuner.
    auto_tune : bool, optional
        If True, auto-tune the omega grid / Krylov dim / etc.
    ed_binary, extra_args, env, check, capture_output : see
        :func:`subprocess.run`.

    Returns
    -------
    * In-memory form: a :class:`_core.SpectralResult`.
    * Directory form with ``symmetry=``: a :class:`_core.SpectralResult`
      (the streaming-symmetry C++ binding returns the same result type).
    * Directory form without ``symmetry=``: a
      :class:`subprocess.CompletedProcess` from the ``./ED dssf`` CLI.

    SOTA streaming-symmetry kwargs (directory form, May 2026)
    ---------------------------------------------------------
    symmetry : bool, str, or dict, optional
        Enable the streaming-symmetry path. ``True`` / ``"auto"`` /
        ``"streaming"`` engages it; a ``dict`` lets you pass
        ``momentum_transfer`` / ``momentum_tolerance`` /
        ``selected_sectors`` in one bundle. The streaming-symmetry
        binding currently covers the ground-state CF method on a
        directory; other workflows fall through to the legacy CLI
        path (which has its own per-sector treatment).
    num_sites : int
        Number of lattice sites; required when ``symmetry=True``.
    spin_l : float, optional
        Spin magnitude; defaults to 0.5.
    sz : int, optional
        Fixed-Sz projection (``n_up``).
    momentum_transfer : sequence of floats, optional
        Momentum transfer Q of the probe observable, in fractional
        reciprocal-lattice units. Drives the
        ``k_final = k_initial + Q`` selection-rule annotation.
    momentum_tolerance : float, optional
        Tolerance for the Q match.
    selected_sectors : sequence of ints, optional
        Restrict the initial-sector search to a subset of irreps.

    Examples
    --------
    Ground-state continued-fraction S^z(omega) for an in-memory chain:

    .. code-block:: python

        H  = qed.input.HamiltonianBuilder(8).heisenberg(...).to_operator()
        Sz = qed.input.HamiltonianBuilder(8).build_sz_operator()
        res = qed.spectral(H, [Sz],
                           omega=np.linspace(-2, 2, 200),
                           eta=0.05)

    Directory-form dynamical S(Q,omega) at finite T:

    .. code-block:: python

        qed.spectral("runs/heisenberg6",
                     T=0.5,
                     omega=np.linspace(-2, 2, 200),
                     eta=0.05)

    SOTA streaming-symmetry GS-CF (directory + automorphism_results/):

    .. code-block:: python

        res = qed.spectral("runs/heisenberg16",
                           omega=np.linspace(-2, 2, 200), eta=0.05,
                           symmetry={"momentum_transfer": [0.0]},
                           num_sites=16)
        print(res.selection_rule_label)         # selection-rule annotation
        print(res.per_sector_pair[0].initial)   # SectorTag of the GS irrep
    """
    if isinstance(H_or_directory, str):
        # SOTA streaming-symmetry path (May 2026). Engaged when the
        # caller asks for it via ``symmetry=`` and the workflow is
        # one the streaming-symmetry C++ binding currently covers
        # (ground-state CF on a directory). For everything else we
        # fall back to the CLI ``./ED dssf`` workflow, which has its
        # own per-sector treatment.
        wants_streaming_sym = (
            symmetry is True
            or isinstance(symmetry, dict)
            or (isinstance(symmetry, str)
                and symmetry.lower() in ("auto", "streaming", "true"))
        )
        # Pick up momentum_transfer + cross-irrep observable from a
        # dict spec if provided. ``observable`` / ``transforms`` are
        # the SOTA cross-irrep knob (May 2026); when present we route
        # through the cross-irrep C++ binding below instead of the
        # same-irrep DOS path.
        cross_irrep_observable = None
        cross_irrep_delta_n_up = 0
        cross_irrep_momentum_points = None
        cross_irrep_observables_list = None
        if isinstance(symmetry, dict):
            momentum_transfer = symmetry.get(
                "momentum_transfer", momentum_transfer)
            momentum_tolerance = float(symmetry.get(
                "momentum_tolerance", momentum_tolerance))
            selected_sectors = symmetry.get(
                "selected_sectors", selected_sectors)
            cross_irrep_observable = (
                symmetry.get("observable")
                if "observable" in symmetry
                else symmetry.get("transforms")
            )
            cross_irrep_delta_n_up = int(symmetry.get("delta_n_up", 0))
            cross_irrep_momentum_points = symmetry.get(
                "momentum_points", None)
            cross_irrep_observables_list = symmetry.get("observables", None)

        # The streaming-symmetry C++ binding is available for the
        # GS-CF method (T is None) AND for the FTLM-cross-irrep
        # method (T is a non-empty list of finite temperatures
        # combined with a cross-irrep observable). The two routes
        # are mutually exclusive on the `T` keyword.
        ground_state_cf_compatible = (
            (method is None or method.lower().replace("-", "_")
             in ("ground_state_cf", "ground_state_dssf"))
            and T is None
            and omega is not None
        )
        # SOTA finite-T cross-irrep path. Triggers when the caller
        # supplies BOTH a cross-irrep observable AND a finite T (or
        # list of T's). The C++ binding does the FTLM sample loop +
        # per-source-sector recombination internally.
        finite_T_cross_irrep_compatible = (
            T is not None
            and omega is not None
            and cross_irrep_observable is not None
        )
        if wants_streaming_sym and finite_T_cross_irrep_compatible:
            if num_sites is None:
                raise TypeError(
                    "qed.spectral(directory, ..., symmetry={'observable': "
                    "..., 'momentum_transfer': ...}, T=..., omega=...) "
                    "requires `num_sites=`."
                )
            obs_transforms = _extract_transforms(cross_irrep_observable)
            if not obs_transforms:
                raise ValueError(
                    "qed.spectral cross-irrep finite-T: observable "
                    "expanded to zero transforms; check the Operator "
                    "has at least one one-body / two-body term."
                )
            Ts_list = T if isinstance(T, (list, tuple)) else [T]
            return _spectral_streaming_symmetry_ftlm_cross_irrep_directory(
                H_or_directory,
                num_sites=int(num_sites),
                spin_l=float(spin_l),
                fixed_sz_n_up=(int(sz) if sz is not None else None),
                omega=omega,
                eta=eta,
                krylov_dim=krylov_dim,
                momentum_transfer=momentum_transfer,
                momentum_tolerance=momentum_tolerance,
                selected_sectors=selected_sectors,
                observable_transforms=obs_transforms,
                delta_n_up=cross_irrep_delta_n_up,
                temperatures=list(Ts_list),
                num_samples=int(num_random_vectors or 30),
                random_seed=0,
                output_dir=output_dir,
                observable_type=observable_type,
                verbose=verbose,
            )
        if (wants_streaming_sym
                and ground_state_cf_compatible
                and cross_irrep_momentum_points is not None
                and cross_irrep_observables_list is not None):
            # ----------------------------------------------------------
            # SOTA amortised multi-Q cross-irrep path. The GS solve is
            # done once and reused across every Q in ``momentum_points``.
            # Each Q owns its phased observable O_Q via the parallel
            # ``observables`` list. Routes through
            # ``workflows_spectral_streaming_symmetry_cross_irrep_multiq_directory``.
            # ----------------------------------------------------------
            if num_sites is None:
                raise TypeError(
                    "qed.spectral(directory, ..., symmetry={'observables': "
                    "[...], 'momentum_points': [...]}) requires `num_sites=`."
                )
            obs_list = list(cross_irrep_observables_list)
            q_pts = list(cross_irrep_momentum_points)
            if len(obs_list) != len(q_pts):
                raise ValueError(
                    "qed.spectral multi-Q cross-irrep: symmetry['observables'] "
                    f"must be aligned 1:1 with symmetry['momentum_points'] "
                    f"(got {len(obs_list)} observables for {len(q_pts)} "
                    f"Q-points)."
                )
            transforms_per_q = []
            for idx, obs_q in enumerate(obs_list):
                tq = _extract_transforms(obs_q)
                if not tq:
                    raise ValueError(
                        f"qed.spectral multi-Q cross-irrep: observable #{idx} "
                        "expanded to zero transforms; check the Operator has "
                        "at least one one-body / two-body term."
                    )
                transforms_per_q.append(tq)
            return _spectral_streaming_symmetry_cross_irrep_multiq_directory(
                H_or_directory,
                num_sites=int(num_sites),
                spin_l=float(spin_l),
                fixed_sz_n_up=(int(sz) if sz is not None else None),
                omega=omega,
                eta=eta,
                krylov_dim=krylov_dim,
                energy_shift=energy_shift,
                momentum_points=q_pts,
                momentum_tolerance=momentum_tolerance,
                selected_sectors=selected_sectors,
                observable_transforms_per_q=transforms_per_q,
                delta_n_up=cross_irrep_delta_n_up,
                output_dir=output_dir,
                observable_type=observable_type,
                verbose=verbose,
            )
        if (wants_streaming_sym
                and ground_state_cf_compatible
                and cross_irrep_observable is not None):
            # ----------------------------------------------------------
            # SOTA cross-irrep path. Routes through
            # ``workflows_spectral_streaming_symmetry_cross_irrep_directory``.
            # ----------------------------------------------------------
            if num_sites is None:
                raise TypeError(
                    "qed.spectral(directory, ..., symmetry={'observable': "
                    "...}) requires `num_sites=`."
                )
            obs_transforms = _extract_transforms(cross_irrep_observable)
            if not obs_transforms:
                raise ValueError(
                    "qed.spectral cross-irrep symmetry: observable "
                    "expanded to zero transforms; check the Operator "
                    "has at least one one-body / two-body term."
                )
            return _spectral_streaming_symmetry_cross_irrep_directory(
                H_or_directory,
                num_sites=int(num_sites),
                spin_l=float(spin_l),
                fixed_sz_n_up=(int(sz) if sz is not None else None),
                omega=omega,
                eta=eta,
                krylov_dim=krylov_dim,
                energy_shift=energy_shift,
                momentum_transfer=momentum_transfer,
                momentum_tolerance=momentum_tolerance,
                selected_sectors=selected_sectors,
                observable_transforms=obs_transforms,
                delta_n_up=cross_irrep_delta_n_up,
                output_dir=output_dir,
                observable_type=observable_type,
                verbose=verbose,
            )
        if wants_streaming_sym and ground_state_cf_compatible:
            if num_sites is None:
                raise TypeError(
                    "qed.spectral(directory, ..., symmetry=...) requires "
                    "an explicit `num_sites=` kwarg so the streaming-"
                    "symmetry binding knows the qubit count. Pass the "
                    "same N you used when generating the deck "
                    "(`HamiltonianBuilder(N).write_directory(...)`)."
                )
            return _spectral_streaming_symmetry_directory(
                H_or_directory,
                num_sites=int(num_sites),
                spin_l=float(spin_l),
                fixed_sz_n_up=(int(sz) if sz is not None else None),
                omega=omega, method=method,
                eta=eta, krylov_dim=krylov_dim,
                num_random_vectors=num_random_vectors,
                energy_shift=energy_shift,
                momentum_transfer=momentum_transfer,
                momentum_tolerance=momentum_tolerance,
                selected_sectors=selected_sectors,
                output_dir=output_dir,
                observable_type=observable_type,
                verbose=verbose,
            )
        if wants_streaming_sym and verbose:
            print(
                "[qed.spectral] symmetry= requested but the requested "
                "workflow (method/T/omega combination) has no "
                "streaming-symmetry binding yet; falling back to the "
                "CLI ./ED dssf path which has its own per-sector "
                "treatment."
            )
        return _spectral_directory(
            H_or_directory,
            T=T, omega=omega, method=method,
            eta=eta, krylov_dim=krylov_dim,
            num_random_vectors=num_random_vectors,
            kpm_moments=kpm_moments, bandwidth=bandwidth,
            device=device, level=level,
            sector_dim=sector_dim, operator=operator,
            auto_tune=auto_tune,
            ed_binary=ed_binary, extra_args=extra_args,
            env=env, check=check, capture_output=capture_output,
            verbose=verbose,
        )
    if not isinstance(H_or_directory, Operator):
        raise TypeError(
            f"qed.spectral first argument must be a directory path "
            f"(str) or an Operator / FixedSzOperator; got "
            f"{type(H_or_directory).__name__}"
        )
    if observables is None:
        raise TypeError(
            "qed.spectral(H, observables, ...) requires an "
            "``observables`` list when H is an in-memory operator. "
            "For directory-form runs, pass the directory path as the "
            "first argument."
        )
    if spin_flip not in (None, "auto", -1) or \
            time_reversal not in (None, "auto", -1):
        from .workflow import resolve_discrete_toggle
        sf_i = resolve_discrete_toggle(
            H_or_directory, spin_flip, "spin_flip", verbose=verbose)
        tr_i = resolve_discrete_toggle(
            H_or_directory, time_reversal, "time_reversal",
            verbose=verbose)
        if sf_i == 1 or tr_i == 1:
            det = _core.detect_hamiltonian_symmetries(H_or_directory)
            if sf_i == 1 and not det["spin_flip"]:
                raise RuntimeError(
                    "qed.spectral: spin_flip='require' but "
                    "[H, prod sigma^x] != 0 at the term level.")
            if tr_i == 1 and not det["time_reversal"]:
                raise RuntimeError(
                    "qed.spectral: time_reversal='require' but H has "
                    "complex matrix elements.")
        if verbose and (sf_i != 0 or tr_i != 0):
            print("[qed.spectral] note: spin-flip / time-reversal are "
                  "not exploited by the spectral lanes yet (they only "
                  "affect solve/thermal); the U(1) x spatial sector "
                  "machinery is used when symmetry= is given.")
    if symmetry is not None:
        routed = _spectral_in_memory_with_symmetry(
            H_or_directory, observables,
            symmetry=symmetry, sz=sz, T=T, omega=omega, method=method,
            eta=eta, krylov_dim=krylov_dim,
            num_random_vectors=num_random_vectors,
            energy_shift=energy_shift,
            momentum_transfer=momentum_transfer,
            momentum_tolerance=momentum_tolerance,
            selected_sectors=selected_sectors,
            output_dir=output_dir, observable_type=observable_type,
            spin_l=spin_l, verbose=verbose)
        if routed is not NotImplemented:
            return routed
        if verbose:
            print("[qed.spectral] symmetry= requested but the call "
                  "could not be routed through the sector machinery "
                  "(trivial group / non-transform observable / "
                  "unsupported method); running the plain in-memory "
                  "lane.")
    return _spectral_in_memory(
        H_or_directory,
        observables,
        omega=omega, method=method,
        eta=eta, krylov_dim=krylov_dim,
        num_random_vectors=num_random_vectors,
        energy_shift=energy_shift, output_dir=output_dir,
        T=T, observable_type=observable_type, verbose=verbose,
        # Phase D of the "Backend x Symmetries x Workflows" plan
        # (May 2026): forward ``device=`` to the in-memory binding.
        device=device,
        # Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026).
        initial_state=initial_state,
        # Pillar 4 of the "Save and DSSF Upgrades" plan (May 2026).
        kpm_moments=kpm_moments,
        kpm_kernel=kpm_kernel,
        kpm_lorentz_lambda=kpm_lorentz_lambda,
    )
