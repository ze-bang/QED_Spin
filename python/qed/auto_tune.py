"""Heuristic auto-tuning for DSSF / structure-factor knobs.

Pure helper functions used by :func:`qed.dssf.compute` (and mirrored in
``include/ed/auto/dssf_tune.h`` for the C++ ``ed::auto_pilot::dssf``
facade). Every function takes (problem-size, machine-size, aggressiveness)
and returns a single scalar / tuple — no I/O, no side effects, easy to
unit-test.

Aggressiveness levels (matches the C++ ``DSSFTuneLevel`` enum):

* ``"conservative"`` — correctness-first. Wider η, fewer Krylov steps,
  more random vectors. Cheaper, may over-broaden narrow features.
* ``"balanced"`` (default) — recommended. Scales with sector dim.
* ``"aggressive"`` — resolution-first. Tighter η, deeper Krylov, more
  samples. Slower but spectrally sharper.

Heuristic rationale
-------------------

Bandwidth W
    The full eigenvalue range. Auto-estimated from the operator's
    L1-norm of coefficients × num_sites when the user does not supply
    one. This is an upper bound — the true W can be 2–4× smaller.

Frequency window
    Defaults to ``[-1.1·W, 1.1·W]`` so the ω-grid covers the entire
    spectrum with a small margin. Override with ``omega=`` (an explicit
    grid) or ``bandwidth=`` (only the ω endpoints).

η broadening
    Lorentzian width. The ω-grid spacing is ``Δω = 2.2·W / N_ω``; we
    set ``η = c · Δω`` with ``c ∈ {2, 3, 5}`` for
    aggressive / balanced / conservative. This guarantees the
    Lorentzian peaks are wider than the grid spacing (no aliasing) but
    not so wide they smear neighbouring Δ-functions.

Krylov dim m
    For FTLM/LTLM/dyn-resp continued fractions, ``m`` controls how many
    poles fit the spectral function per random vector. Scales as
    ``min(M_max, max(M_min, ⌈D^{1/3}⌉))`` so growth is sublinear in
    sector dim. Caps: 60 / 200 / 400 (cons / bal / agg).

Random samples R
    Variance of FTLM estimators is ``∝ 1/(R·D)``. We pick
    ``R = max(R_min, ⌈C/√D⌉)``: tiny sectors get many samples,
    large sectors get one or two (where the trace is already
    self-averaging). Caps: 8 / 32 / 64 (cons / bal / agg).

KPM moments M
    Resolution of KPM-DOS scales like ``W/M``. We pick
    ``M = clamp(round(W·1000), M_min, M_max)`` with caps
    512 / 2048 / 8192. Independent of sector dim because the
    Chebyshev recursion already auto-averages.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

Level = Literal["conservative", "balanced", "aggressive"]

# ---------------------------------------------------------------------------
# Per-level constants — kept in one place so the C++ mirror in
# include/ed/auto/dssf_tune.h can copy them verbatim.
# ---------------------------------------------------------------------------

_ETA_GRID_FACTOR = {"conservative": 5.0, "balanced": 3.0, "aggressive": 2.0}
_KRYLOV_MIN     = {"conservative": 40,  "balanced": 80,  "aggressive": 120}
_KRYLOV_MAX     = {"conservative": 60,  "balanced": 200, "aggressive": 400}
_RANDOM_MIN     = {"conservative": 4,   "balanced": 4,   "aggressive": 8}
_RANDOM_MAX     = {"conservative": 8,   "balanced": 32,  "aggressive": 64}
_KPM_MOMENTS    = {"conservative": 512, "balanced": 2048, "aggressive": 8192}
_NUM_OMEGA      = {"conservative": 200, "balanced": 400, "aggressive": 800}


def _check_level(level: str) -> Level:
    if level not in _ETA_GRID_FACTOR:
        raise ValueError(
            f"unknown auto-tune level {level!r}; "
            f"expected one of {tuple(_ETA_GRID_FACTOR)}"
        )
    return level  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Bandwidth estimator
# ---------------------------------------------------------------------------

def estimate_bandwidth(operator, *, fallback: float = 4.0) -> float:
    """Cheap upper-bound estimate of the spectral bandwidth ``E_max - E_min``.

    Uses Gershgorin-style summation of the operator's per-term coefficient
    moduli scaled by ``num_sites``. For an Operator with ``len(transform_data_)``
    accessible (the common case for the C++ ``Operator`` Python bindings)
    we sum |c_t| · num_sites; otherwise we return ``fallback * num_sites``.

    Conservative — the true bandwidth is typically 2–4× smaller, so the
    auto-tuned ω-window will always cover the spectrum.
    """
    try:
        n = int(operator.num_sites)
    except (AttributeError, TypeError):
        return float(fallback)

    total_coeff = 0.0
    seen_any = False
    for attr in ("transform_data_", "three_body_data_"):
        try:
            terms = getattr(operator, attr)
        except AttributeError:
            continue
        for t in terms or []:
            try:
                total_coeff += abs(complex(t.coefficient))
                seen_any = True
            except (AttributeError, TypeError, ValueError):
                continue
    if not seen_any:
        return float(fallback) * n
    # 2 × |Σ c| per site is a safe upper bound on the half-bandwidth.
    return 2.0 * total_coeff


# ---------------------------------------------------------------------------
# Knob pickers
# ---------------------------------------------------------------------------

def pick_omega_window(
    bandwidth: float,
    *,
    margin: float = 0.1,
) -> Tuple[float, float]:
    """Return ``(omega_min, omega_max)`` covering the full spectrum.

    Defaults to symmetric window since DSSF spectra are typically two-sided
    and the dispatcher expects ``omega_min < 0 < omega_max``. Override by
    passing your own grid via ``qed.dssf.compute(omega=...)``.
    """
    half = (1.0 + margin) * abs(bandwidth)
    return (-half, half)


def pick_num_omega_points(level: Level = "balanced") -> int:
    """Default number of ω-grid points for the auto-tuner."""
    return _NUM_OMEGA[_check_level(level)]


def pick_eta(
    bandwidth: float,
    num_omega_points: int,
    *,
    level: Level = "balanced",
    margin: float = 0.1,
) -> float:
    """Lorentzian broadening η.

    Set so η is a small multiple of the ω-grid spacing — wide enough to
    eliminate aliasing but tight enough to resolve features.
    """
    level = _check_level(level)
    if num_omega_points <= 1:
        return _ETA_GRID_FACTOR[level] * 0.05 * abs(bandwidth)
    omega_min, omega_max = pick_omega_window(bandwidth, margin=margin)
    delta_omega = (omega_max - omega_min) / float(num_omega_points - 1)
    return _ETA_GRID_FACTOR[level] * delta_omega


def pick_krylov_dim(sector_dim: int, *, level: Level = "balanced") -> int:
    """FTLM / continued-fraction Krylov subspace dimension."""
    level = _check_level(level)
    target = max(1, int(round(sector_dim ** (1.0 / 3.0))))
    return int(max(_KRYLOV_MIN[level], min(_KRYLOV_MAX[level], target)))


def pick_num_random_vectors(sector_dim: int, *, level: Level = "balanced") -> int:
    """Number of random initial vectors R for FTLM / dynamical response."""
    level = _check_level(level)
    if sector_dim <= 1:
        return _RANDOM_MAX[level]
    target = max(1, int(math.ceil(64.0 / math.sqrt(float(sector_dim)))))
    return int(max(_RANDOM_MIN[level], min(_RANDOM_MAX[level], target)))


def pick_kpm_moments(*, level: Level = "balanced") -> int:
    """Number of Chebyshev moments for KPM-DOS."""
    return _KPM_MOMENTS[_check_level(level)]


# ---------------------------------------------------------------------------
# Device picker — mirrors qed.workflow._resolve_device for the DSSF surface.
# Kept separate from the pure-numerical helpers above so callers can audit it.
# ---------------------------------------------------------------------------

def pick_device(
    sector_dim: int,
    *,
    has_cuda_build: bool,
    has_mpi_build: bool,
    user_request: Optional[str] = None,
    gpu_dim_threshold: int = 1 << 17,
    mpi_dim_threshold: int = 1 << 22,
) -> str:
    """Choose ``"cpu" | "gpu" | "mpi" | "mpi_gpu"`` for a DSSF run.

    The DSSF kernels (FTLM, ground-state continued fraction, KPM-DOS) all
    benefit from GPU once the per-vector matvec dominates over launch
    latency (~2^17 ≈ 131k); MPI helps once the sector dimension no longer
    fits a single node (~2^22 ≈ 4M, very rough).
    """
    if user_request is not None:
        u = user_request.lower()
        if u not in ("auto", "cpu", "gpu", "mpi", "mpi_gpu"):
            raise ValueError(
                f"device={user_request!r} is not recognised; "
                f"expected one of auto, cpu, gpu, mpi, mpi_gpu"
            )
        if u != "auto":
            return u
    use_gpu = has_cuda_build and sector_dim >= gpu_dim_threshold
    use_mpi = has_mpi_build  and sector_dim >= mpi_dim_threshold
    if use_gpu and use_mpi:
        return "mpi_gpu"
    if use_gpu:
        return "gpu"
    if use_mpi:
        return "mpi"
    return "cpu"


# ---------------------------------------------------------------------------
# Convenience bundle — used by qed.dssf.compute(...) to fill in EVERY
# missing knob in one shot. Returns a dataclass whose fields map 1:1 to
# the ``--dyn-*`` / ``--static-*`` / ``--ftlm-*`` CLI flags consumed by
# ``./ED dssf``.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TunedDSSFKnobs:
    """Auto-selected DSSF knobs (output of :func:`tune_dssf`)."""
    omega_min: float
    omega_max: float
    num_omega_points: int
    eta: float
    krylov_dim: int
    num_random_vectors: int
    kpm_moments: int
    device: str
    bandwidth: float
    level: str

    def to_cli_args(
        self,
        *,
        method: str,
    ) -> list[str]:
        """Render to ``--dyn-* / --static-* / --ftlm-*`` flags for ``./ED dssf``.

        ``method`` is one of the DSSF method tokens; we emit the right
        prefix for the chosen workflow:

        * ``dynamical_thermal``    -> ``--dyn-*``  + ``--ftlm-krylov``
        * ``ground_state_dssf``    -> ``--dyn-*``  (broadening + omega
                                        consumed by the GS workflow too)
        * ``static_thermal``       -> ``--static-*``
        * ``single_expectation``   -> no extra args (one-shot)
        * ``kpm_thermodynamics``   -> no CLI flags (KPM tuning is via
                                       ``EDParameters`` only — emit nothing)
        """
        args: list[str] = []
        if method in ("dynamical_thermal", "ground_state_dssf"):
            args += [
                f"--dyn-omega-min={self.omega_min:.10g}",
                f"--dyn-omega-max={self.omega_max:.10g}",
                f"--dyn-omega-points={self.num_omega_points}",
                f"--dyn-broadening={self.eta:.10g}",
                f"--dyn-krylov={self.krylov_dim}",
                f"--dyn-samples={self.num_random_vectors}",
            ]
            if method == "dynamical_thermal":
                args.append(f"--ftlm-krylov={self.krylov_dim}")
        elif method == "static_thermal":
            args += [
                f"--static-krylov={self.krylov_dim}",
                f"--static-samples={self.num_random_vectors}",
            ]
        if self.device in ("gpu", "mpi_gpu"):
            args.append("--use-gpu")
        return args


def tune_dssf(
    *,
    operator=None,
    sector_dim: Optional[int] = None,
    bandwidth: Optional[float] = None,
    omega: Optional[Union[Sequence[float], Iterable[float]]] = None,
    eta: Optional[float] = None,
    krylov_dim: Optional[int] = None,
    num_random_vectors: Optional[int] = None,
    kpm_moments: Optional[int] = None,
    device: Optional[str] = None,
    has_cuda_build: bool = False,
    has_mpi_build: bool = False,
    level: Level = "balanced",
) -> TunedDSSFKnobs:
    """Pick every DSSF knob, honouring user overrides.

    Any kwarg the caller supplies is passed through verbatim. Anything
    they leave as ``None`` is auto-selected from
    (``operator`` / ``sector_dim``) and the requested ``level``.
    """
    level = _check_level(level)

    # Bandwidth.
    if bandwidth is None:
        bandwidth = estimate_bandwidth(operator) if operator is not None else 4.0

    # Sector dim — fall back to the operator's full Hilbert dim.
    if sector_dim is None and operator is not None:
        try:
            sector_dim = int(operator.dimension)
        except AttributeError:
            sector_dim = None
    if sector_dim is None:
        sector_dim = 1024  # mild default for pure-helper invocation

    # Omega axis.
    if omega is None:
        omega_min, omega_max = pick_omega_window(bandwidth)
        num_omega_points = pick_num_omega_points(level)
    else:
        omega_arr = list(omega)
        if len(omega_arr) < 2:
            raise ValueError("omega= must contain at least two grid points")
        omega_min = float(min(omega_arr))
        omega_max = float(max(omega_arr))
        num_omega_points = len(omega_arr)

    # Broadening.
    if eta is None:
        eta = pick_eta(bandwidth, num_omega_points, level=level)

    # Krylov.
    if krylov_dim is None:
        krylov_dim = pick_krylov_dim(sector_dim, level=level)

    # Random vectors.
    if num_random_vectors is None:
        num_random_vectors = pick_num_random_vectors(sector_dim, level=level)

    # KPM moments.
    if kpm_moments is None:
        kpm_moments = pick_kpm_moments(level=level)

    # Device.
    chosen_device = pick_device(
        sector_dim,
        has_cuda_build=has_cuda_build,
        has_mpi_build=has_mpi_build,
        user_request=device,
    )

    return TunedDSSFKnobs(
        omega_min=float(omega_min),
        omega_max=float(omega_max),
        num_omega_points=int(num_omega_points),
        eta=float(eta),
        krylov_dim=int(krylov_dim),
        num_random_vectors=int(num_random_vectors),
        kpm_moments=int(kpm_moments),
        device=chosen_device,
        bandwidth=float(bandwidth),
        level=level,
    )


__all__ = [
    "Level",
    "TunedDSSFKnobs",
    "estimate_bandwidth",
    "pick_omega_window",
    "pick_num_omega_points",
    "pick_eta",
    "pick_krylov_dim",
    "pick_num_random_vectors",
    "pick_kpm_moments",
    "pick_device",
    "tune_dssf",
]
