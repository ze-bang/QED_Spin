"""``qed.spectral``: canonical spectral / structure-factor entry point.

This module implements :func:`qed.spectral`, the one Python verb for
zero- and finite-temperature dynamical / static structure factors and
related spectral functions: ``qed.spectral(H, observables, ...)`` runs
the unified C++ orchestrator (``ed::workflows::spectral``) on the
supplied operator and observable list, or, with ``symmetry=``, the
cross-irrep sector lanes. (The former directory form, which shelled
out to the removed ``ED`` command-line program, lives on in the
``pre-simplify-2026-09`` tag.)
"""

from __future__ import annotations

import numpy as np

from typing import Iterable, Optional, Sequence, Union, Any

from . import _core
from ._core import (  # type: ignore[attr-defined]
    Operator,
    FixedSzOperator,
)

__all__ = [
    "spectral",
]


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


from dataclasses import dataclass as _dataclass


@_dataclass
class FiniteTSpectralResult:
    """Plain-lane finite-temperature S(omega, T) (2026-09-11): ``S_real`` is the
    spectrum at the FIRST temperature (mirrors the symmetry lane's convention),
    ``S_real_by_T`` holds every requested temperature."""
    omega: "np.ndarray"
    temperatures: list
    S_real_by_T: dict

    @property
    def S_real(self):
        return self.S_real_by_T[float(self.temperatures[0])]



@_dataclass
class _GsDssfResult:
    """Return shape of the full-group GS-DSSF lane (Stage 9d: the
    factorized little-group engine)."""
    omega: "np.ndarray"
    S_real: "np.ndarray"
    gs_energy: float
    total_weight: float
    gpu_engaged: bool = False   # truthful: the CF matvecs ran the device
                                # rep-gather (2026-07-20 GS-DSSF GPU lane)


def _infer_delta_n_up(transforms: list) -> int:
    """Infer the Sz selection rule of a probe from its transform tuples.

    In this codebase the fixed-Sz axis counts SET bits, and a set bit
    is the state S- creates (verified against the plain full-Hilbert
    lane at 1e-11): an S- probe (op_type 1) RAISES the sector''s n_up
    by 1, S+ (op_type 0) lowers it, Sz / paired two-body combinations
    conserve it. A probe mixing S+ and S- one-body terms has no single
    selection rule and is refused; split it into one probe per rule.
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
        f"different Sz selection rules ({sorted(deltas)}); split it into "
        "probes with one selection rule each."
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
    spin_flip=-1,
):
    """Route an IN-MEMORY spectral call through the streaming-symmetry
    machinery: resolve ``symmetry`` (including ``"auto"``) to the closed
    group info dict, extract each observable's transform tuples, and
    dispatch ``(H, info)`` to the cross-irrep GS-CF (T is None) or
    FTLM (finite T) C++ binding.

    Stage 8d (SymmetryEngine v2): when ``sz`` is None the diagonal /
    flip axes compose automatically -- Sz-parity halves when H carries
    the Z2 remnant and the probe has a definite parity selection rule,
    and full-space prod-sigma^x flip sectors when [H, X] == 0 and the
    probe has a definite flip character (X O X == +-O; note S^z probes
    are flip-ODD). ``spin_flip`` follows the toggle ints (-1 auto /
    0 off / 1 require).

    Returns ``NotImplemented`` when the request cannot be routed (no
    spatial symmetry found, an observable without ``transform_tuples``,
    an unsupported method) -- the caller then falls back to the plain
    in-memory lane.
    """
    from .workflow import (
        resolve_auto_symmetry as _resolve_auto,
        _closed_symmetry_info as _closed_sym_info,
        _normalize_symmetry_info as _norm_sym_info,
    )

    if isinstance(symmetry, dict):
        raise TypeError(
            "qed.spectral(H, observables, symmetry=dict) is not supported; "
            "pass symmetry='auto', a GeneratorSet, or a permutation list, "
            "and momentum_transfer=[...] for the probe's Q."
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

    info = _norm_sym_info(H, gen)
    if info is None:
        return NotImplemented
    # The cross-irrep bindings receive this dict verbatim; close a raw
    # generators-only group first (fixing the group, sector ids and
    # quantum numbers).
    source = (H, _closed_sym_info(info))

    # Stage 8d: diagonal / flip axis composition for the sz=None
    # lanes. Detection is term-level (same walk solve/thermal use);
    # engagement is PER PROBE, since routability depends on the
    # probe's selection rules.
    det = None
    if sz is None:
        try:
            det = dict(_core.detect_hamiltonian_symmetries(H))
        except Exception:
            det = None

    def _probe_lanes(tuples):
        """(sz_parity, flip_sectors) for one probe's transforms."""
        szp, flip = -1, False
        if sz is not None or det is None:
            return szp, flip
        n = int(H.num_sites)
        if det.get("sz_parity") and not det.get("u1"):
            if _core.probe_delta_n_up_parity(tuples) >= 0:
                szp = 2                     # pool both parity halves
        if spin_flip != 0 and det.get("spin_flip"):
            if (_core.probe_spin_flip_character(tuples) != 0
                    and (szp < 0 or n % 2 == 0)):
                flip = True
        if spin_flip == 1 and not flip:
            raise RuntimeError(
                "qed.spectral: spin_flip='require' but the flip lane "
                "cannot route this call (H lacks the symmetry, the "
                "probe has no definite flip character, or the parity "
                "closure rule excludes it).")
        return szp, flip

    results = []
    for tuples in transforms_per_obs:
        delta = _infer_delta_n_up(tuples) if sz is not None else 0
        szp, flip = _probe_lanes(tuples)
        if T is None:
            results.append(
                _spectral_streaming_symmetry_cross_irrep(
                    source,
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
                    sz_parity=szp,
                    flip_sectors=flip,
                    output_dir=output_dir,
                    observable_type=observable_type,
                    verbose=verbose,
                ))
        else:
            Ts_list = T if isinstance(T, (list, tuple)) else [T]
            results.append(
                _spectral_streaming_symmetry_ftlm_cross_irrep(
                    source,
                    num_sites=int(H.num_sites),
                    spin_l=float(spin_l),
                    fixed_sz_n_up=(int(sz) if sz is not None else None),
                    omega=omega, eta=eta, krylov_dim=krylov_dim,
                    momentum_transfer=momentum_transfer,
                    momentum_tolerance=momentum_tolerance,
                    selected_sectors=selected_sectors,
                    observable_transforms=tuples,
                    delta_n_up=delta,
                    sz_parity=szp,
                    flip_sectors=flip,
                    temperatures=list(Ts_list),
                    num_samples=int(num_random_vectors or 30),
                    random_seed=0,
                    output_dir=output_dir,
                    observable_type=observable_type,
                    verbose=verbose,
                ))
    return results[0] if len(results) == 1 else results


# The cross-irrep sector lanes take an ``(Operator, info)`` pair whose ``info``
# is the CLOSED group dict (``_closed_symmetry_info``).
SymmetricSource = tuple


def _spectral_streaming_symmetry_cross_irrep(
    source: SymmetricSource,
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
    sz_parity: int = -1,
    flip_sectors: bool = False,
) -> Any:
    """Cross-irrep streaming-symmetry spectral: the call goes to
    ``_core.workflows_spectral_streaming_symmetry_cross_irrep``
    (``(Operator, info)`` source).

    Stage 8d: ``sz_parity`` / ``flip_sectors`` engage the Sz-parity /
    prod-sigma^x synthetic sector lanes (sz=None only).
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
            f"source=in-memory  N={num_sites}  "
            f"fixed_sz_n_up={fixed_sz_n_up}  delta_n_up={delta_n_up}  "
            f"Q={list(opts.momentum_transfer)}  "
            f"terms={len(observable_transforms)}"
        )
    H_src, info = source
    return _core.workflows_spectral_streaming_symmetry_cross_irrep(
        H_src,
        info,
        int(num_sites),
        float(spin_l),
        observable_transforms,
        opts,
        fixed_sz_n_up,
        int(delta_n_up),
        int(sz_parity),
        bool(flip_sectors),
    )


def _spectral_streaming_symmetry_ftlm_cross_irrep(
    source: SymmetricSource,
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
    sz_parity: int = -1,
    flip_sectors: bool = False,
) -> Any:
    """Finite-T cross-irrep streaming-symmetry spectral: the call goes to
    ``_core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep``
    (``(Operator, info)`` source).

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
            f"source=in-memory  N={num_sites}  "
            f"fixed_sz_n_up={fixed_sz_n_up}  delta_n_up={delta_n_up}  "
            f"Q={list(opts.momentum_transfer)}  T={Ts}  "
            f"num_samples={num_samples}  terms={len(observable_transforms)}"
        )
    H_src, info = source
    agg = _core.workflows_spectral_streaming_symmetry_ftlm_cross_irrep(
        H_src,
        info,
        int(num_sites),
        float(spin_l),
        observable_transforms,
        opts,
        fixed_sz_n_up,
        int(delta_n_up),
        Ts,
        int(num_samples),
        int(random_seed),
        int(sz_parity),
        bool(flip_sectors),
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
                f"'ftlm_dynamical', or 'kpm_dynamical'."
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
        # Audit fix (2026-07-30): the plain in-memory lane has NO working
        # finite-T estimator. GroundStateCF ignores `temperatures` outright
        # (measured: T=1.0 and T=0.5 both returned the T=0 spectrum,
        # machine-identical), and the in-memory FtlmDynamical/KpmDynamical
        # kernels hardcode temperature=0.0 in the orchestrator (measured:
        # the returned spectrum equals the T=infinity correlator for any
        # requested T). Refuse loudly instead of silently answering a
        # different physical question. The finite-T machinery that IS
        # verified correct lives on the symmetry lane (ftlm cross-irrep
        # kernel).
        if any(t > 0.0 for t in opts.temperatures):
            # 2026-09-11: route finite T through the same FTLM estimator the
            # symmetry lane uses (ftlm_cross_irrep_kernel with source = target
            # = this block). Without symmetry this was refused outright.
            if len(observables) != 1:
                raise NotImplementedError(
                    "qed.spectral: finite-T on the plain lane takes one observable per call.")
            _ws = [float(w) for w in omega]
            _ns = int(num_random_vectors) if num_random_vectors is not None else 30
            _kd = int(krylov_dim) if krylov_dim is not None else 100
            d = _core.workflows_spectral_ftlm_plain(
                H, observables[0], [float(t) for t in opts.temperatures], _ws,
                float(opts.broadening), _ns, _kd, 0)
            return FiniteTSpectralResult(
                omega=np.asarray(d["omega"], dtype=float),
                temperatures=list(d["temperatures"]),
                S_real_by_T={float(t): np.asarray(s, dtype=float) for t, s in zip(d["temperatures"], d["S_real"])},
            )

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
            # Explicit request: the GPU auto-promotion dim floor must
            # not second-guess the caller.
            opts.backend.gpu_dim_floor = 0
        elif dev_lc == "cpu":
            opts.backend.allow_gpu = False
            opts.backend.allow_mpi = False
        elif dev_lc in ("mpi", "mpi_gpu"):
            raise NotImplementedError(
                f"qed.spectral(device={device!r}): MPI spectral is "
                "not supported; use device='cpu' or 'gpu'.")
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
    res = _core.workflows_spectral(H, obs_list, opts)
    _kry = getattr(res, "krylov", None)
    if (opts.method == _core.SpectralMethod.GroundStateCF and _kry is not None
            and getattr(_kry, "iters_done", 0) > 0 and not getattr(_kry, "converged", True)):
        import warnings as _warnings
        _warnings.warn(
            f"qed.spectral: the continued fraction is not converged at krylov_dim="
            f"{opts.krylov_dim} (the spectrum changed by "
            f"{100.0 * float(getattr(_kry, 'residual_norm', float('nan'))):.0f}% between "
            f"half and full Krylov depth at eta={opts.broadening:g}); increase krylov_dim.",
            RuntimeWarning, stacklevel=3)
    return res


def spectral(
    H: Union[Operator, FixedSzOperator],
    observables: Optional[Sequence[Operator]] = None,
    *,
    T: Optional[Union[float, Iterable[float]]] = None,
    omega: Optional[Iterable[float]] = None,
    method: Optional[str] = None,
    eta: Optional[float] = None,
    krylov_dim: Optional[int] = None,
    num_random_vectors: Optional[int] = None,
    energy_shift: Optional[float] = None,
    output_dir: str = "",
    observable_type: str = "",
    # KpmDynamical knobs, forwarded to ``_core.SpectralOptions.kpm_*``.
    kpm_moments: Optional[int] = None,
    kpm_kernel: Optional[str] = None,
    kpm_lorentz_lambda: Optional[float] = None,
    device: Optional[str] = None,
    verbose: bool = True,
    # Streaming-symmetry (cross-irrep) knobs ------------------------------
    symmetry: Union[bool, str, None] = None,
    spin_l: float = 0.5,
    sz: Optional[int] = None,
    momentum_transfer: Optional[Sequence[float]] = None,
    momentum_tolerance: float = 1e-6,
    selected_sectors: Optional[Sequence[int]] = None,
    # Stage 8e (SymmetryEngine v2): per-symmetry toggles. Stage 8d:
    # spin_flip= IS consumed (parity halves + flip sectors route DSSF
    # end-to-end); time_reversal= is NOT exploited by the spectral verb
    # -- 'require' raises a loud NotImplementedError instead of running
    # degraded (no lane folds k <-> -k here), 'on' detects and reports.
    spin_flip: Union[str, bool, int, None] = "auto",
    time_reversal: Union[str, bool, int, None] = "auto",
    point_group: Union[str, bool, None] = "auto",
    # Pillar 3 of the "Save and DSSF Upgrades" plan (May 2026) --------
    initial_state: Optional[Any] = None,
):
    """Spectral / structure-factor calculation on an in-memory operator.

    Runs the C++ orchestrator (``_core.workflows_spectral``) on ``H`` and
    ``observables``; with ``symmetry=`` and ``momentum_transfer=``, the
    cross-irrep sector lanes (ground-state continued fraction at T=0,
    FTLM at finite T).

    Parameters
    ----------
    H : Operator or FixedSzOperator
    observables : list of Operator
        The probes O (required).
    T : float, sequence of floats, or None
        Temperature axis (None means T=0).
    omega : sequence of floats or None
        Frequency grid.
    method : str, optional
        ``"ground_state_cf"``, ``"ftlm_dynamical"`` or ``"kpm_dynamical"``;
        auto-picked from ``T`` when omitted.
    eta : float, optional
        Lorentzian broadening.
    krylov_dim : int, optional
        Continued-fraction / FTLM Krylov subspace dimension.
    num_random_vectors : int, optional
        FTLM number of random initial vectors.
    energy_shift : float, optional
        Spectral energy shift (e.g. subtract the ground-state energy).
    output_dir : str, optional
        Where the C++ engine writes HDF5 artifacts.
    observable_type : str, optional
        Label used in the HDF5 group naming (e.g. "Sz").
    kpm_moments, kpm_kernel, kpm_lorentz_lambda : optional
        KpmDynamical knobs.
    device : str, optional
        ``"cpu"`` or ``"gpu"`` (``None``/``"auto"`` lets the backend choose).
    verbose : bool, optional
        Print one-line progress.
    symmetry : str or generators, optional
        ``"auto"``, a GeneratorSet or a permutation list: engages the
        cross-irrep sector lanes (requires ``momentum_transfer``).
    spin_l : float, optional
        Spin magnitude; defaults to 0.5.
    sz : int, optional
        Fixed-Sz projection (``n_up``).
    momentum_transfer : sequence of floats, optional
        Momentum transfer Q of the probe, in fractional reciprocal-lattice
        units; selects the destination irrep ``k_final = k_initial + Q``.
    momentum_tolerance : float, optional
        Tolerance for the Q match.
    selected_sectors : sequence of ints, optional
        Restrict the initial-sector search to a subset of irreps.

    Returns
    -------
    A :class:`_core.SpectralResult` (or a list, one per observable, on
    the symmetry lanes); :class:`FiniteTSpectralResult` for the plain
    finite-T lane.

    Example
    -------
    .. code-block:: python

        H  = qed.input.HamiltonianBuilder(8).heisenberg(...).to_operator()
        Sz = qed.input.HamiltonianBuilder(8).build_sz_operator()
        res = qed.spectral(H, [Sz], omega=np.linspace(-2, 2, 200), eta=0.05)
    """
    # Input validation (2026-09-11).
    if eta is not None and not (float(eta) > 0.0):
        raise ValueError(f"qed.spectral: eta (broadening) must be > 0, got {eta!r}")
    if omega is not None:
        import numpy as _npv
        _w = _npv.asarray(list(omega), dtype=float)
        if _w.size == 0 or not _npv.all(_npv.isfinite(_w)):
            raise ValueError("qed.spectral: omega must be a non-empty finite grid")
    if krylov_dim is not None and int(krylov_dim) < 2:
        raise ValueError(f"qed.spectral: krylov_dim must be >= 2, got {krylov_dim!r}")
    if num_random_vectors is not None and int(num_random_vectors) < 1:
        raise ValueError(f"qed.spectral: num_random_vectors must be >= 1, got {num_random_vectors!r}")
    if not isinstance(H, Operator):
        raise TypeError(
            f"qed.spectral first argument must be an Operator / "
            f"FixedSzOperator; got {type(H).__name__}"
        )
    if observables is None:
        raise TypeError(
            "qed.spectral(H, observables, ...) requires an "
            "``observables`` list."
        )
    sf_i = -1
    if spin_flip not in (None, "auto", -1) or \
            time_reversal not in (None, "auto", -1):
        from .workflow import resolve_discrete_toggle
        sf_i = resolve_discrete_toggle(
            H, spin_flip, "spin_flip", verbose=verbose)
        tr_i = resolve_discrete_toggle(
            H, time_reversal, "time_reversal",
            verbose=verbose)
        if sf_i == 1 or tr_i == 1:
            det = _core.detect_hamiltonian_symmetries(H)
            if sf_i == 1 and not det["spin_flip"]:
                raise RuntimeError(
                    "qed.spectral: spin_flip='require' but "
                    "[H, prod sigma^x] != 0 at the term level.")
            if tr_i == 1 and not det["time_reversal"]:
                raise RuntimeError(
                    "qed.spectral: time_reversal='require' but H has "
                    "complex matrix elements.")
        # Stage 8d: spin_flip is now CONSUMED by the sector lanes when
        # ``sz`` is None (full-space / parity-half flip sectors) --
        # sf_i rides into ``_spectral_in_memory_with_symmetry`` below.
        #
        # Diction cleanup (2026-07-16): time reversal is NOT EXPLOITED
        # by any spectral lane (its k <-> -k fold trades solves the
        # spectral verb does not batch). An accepted-but-inert knob is
        # the worst kind of inconsistency, so an EXPLICIT toggle is now
        # loud instead of silently ignored: 'require' raises (there is
        # nothing to require of this verb), any other explicit value
        # warns unconditionally. The default ('auto'/None) stays a
        # silent no-op.
        if tr_i == 1:
            raise NotImplementedError(
                "qed.spectral: time_reversal='require' -- the spectral "
                "verb does not exploit time reversal (no lane folds "
                "k <-> -k here). H does carry the symmetry; drop the "
                "argument, or use the verbs that exploit it "
                "(solve/full_spectrum/thermal).")
        if time_reversal in ("on", True):
            import warnings as _warnings
            _warnings.warn(
                "qed.spectral: time_reversal='on' is not exploited on "
                "the spectral verb; the toggle has no effect here "
                "(turning it 'off' is equally a no-op).",
                RuntimeWarning, stacklevel=2)
    if (symmetry is not None and observables is not None
            and isinstance(point_group, str)
            and point_group.lower() == "full"
            and omega is not None and T is None):
        # PROPER non-abelian GS-DSSF -- Stage 9d: the FACTORIZED
        # little-group lane (GS localized by the star walk, O|0>
        # scattered into every raw destination sector via the Stage-8d
        # CrossSectorOrbitObservable rep lane, one continued-fraction
        # Lanczos per receiving sector; memory O(#reps)). This retired
        # the monolithic SAB engine's last production route.
        import numpy as np
        from .workflow import resolve_auto_symmetry as _ras
        from .point_group_routing import resolve_projection_lane
        _sym = _ras(H, symmetry, verbose=verbose)
        lane = resolve_projection_lane(
            _sym, point_group=point_group, consumer="spectral",
            eigenvalues_only=True, verbose=verbose)
        if lane.mode == "project":
            ws = list(omega)
            results = []
            _dssf_gpu = (isinstance(device, str)
                         and device.lower() in ("gpu", "cuda"))
            for obs in observables:
                d = dict(_core.little_group_gs_dssf(
                    H, obs, lane.A, lane.residues,
                    float(min(ws)), float(max(ws)), int(len(ws)),
                    float(eta if eta is not None else 0.1),
                    krylov_dim=int(krylov_dim) if krylov_dim else 200,
                    # GS-DSSF GPU lane (2026-07-20): an explicit gpu request
                    # forces the device rep-gather on every receiving
                    # sector's continued-fraction matvec and batches the
                    # GS-subspace scan's eigensolves on the device.
                    use_gpu=_dssf_gpu))
                r = _GsDssfResult(
                    omega=np.asarray(d["omega"], dtype=float),
                    S_real=np.asarray(d["s_omega"], dtype=float),
                    gs_energy=float(d["gs_energy"]),
                    total_weight=float(d["total_weight"]),
                    gpu_engaged=bool(d.get("gpu_engaged", False)))
                results.append(r)
            if verbose:
                print(f"[qed.spectral] non-abelian LITTLE-GROUP GS-DSSF "
                      f"(factorized): |A| = {len(lane.A)}, residues = "
                      f"{len(lane.residues)}.")
            return results[0] if len(results) == 1 else results
    if symmetry is not None:
        routed = _spectral_in_memory_with_symmetry(
            H, observables,
            symmetry=symmetry, sz=sz, T=T, omega=omega, method=method,
            eta=eta, krylov_dim=krylov_dim,
            num_random_vectors=num_random_vectors,
            energy_shift=energy_shift,
            momentum_transfer=momentum_transfer,
            momentum_tolerance=momentum_tolerance,
            selected_sectors=selected_sectors,
            output_dir=output_dir, observable_type=observable_type,
            spin_l=spin_l, verbose=verbose,
            spin_flip=sf_i)
        if routed is not NotImplemented:
            return routed
        if verbose:
            print("[qed.spectral] symmetry= requested but the call "
                  "could not be routed through the sector machinery "
                  "(trivial group / non-transform observable / "
                  "unsupported method); running the plain in-memory "
                  "lane.")
    # Audit 2026-09: ``sz=`` used to be consumed only by the symmetry lanes;
    # the plain in-memory lane silently ignored it and solved the GLOBAL
    # ground state (measured on an XXZ chain in a field: the block GS and the
    # global GS differ, and the returned S(omega) belonged to the latter).
    # Project H and every Sz-conserving probe onto the named block here.
    if sz is not None and isinstance(H, Operator) \
            and not isinstance(H, FixedSzOperator):
        _n_up = int(sz)
        _obs_proj = []
        for _o in (observables or []):
            _u1 = False
            try:
                _u1 = bool(dict(_core.detect_hamiltonian_symmetries(_o)).get("u1", False))
            except Exception:
                _u1 = False
            if not _u1:
                raise ValueError(
                    "qed.spectral: sz= names a fixed-Sz block but an observable does "
                    "not conserve Sz, so it cannot act inside that block; pass "
                    "symmetry='auto' (the cross-sector lane routes k_final = "
                    "k_initial + Q and delta_n_up) or drop sz=.")
            _obs_proj.append(_o.make_fixed_sz(_n_up))
        H = H.make_fixed_sz(_n_up)
        observables = _obs_proj
        if verbose:
            print(f"[qed.spectral] sz={_n_up}: H and probes projected onto the "
                  f"fixed-Sz block.")
    return _spectral_in_memory(
        H,
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
