"""Heuristic auto-tuning for DSSF / structure-factor knobs.

Pure helper functions used by :func:`qed.spectral`. The C++ mirror of
these heuristics is the ``DSSFKernelOptions`` plumbing inside
``ed::workflows::spectral`` (the surface-unification collapse retired
the cross-cutting ``include/ed/auto/dssf_tune.h``). Every function
takes (problem-size, machine-size, aggressiveness) and returns a
single scalar / tuple — no I/O, no side effects, easy to unit-test.

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
# ``DSSFKernelOptions`` (and friends, inside the orchestrator) can copy
# them verbatim.
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

    Walks the bound ``Operator``'s typed term iterators
    (``iter_one_body_terms`` / ``iter_two_body_terms`` /
    ``iter_three_body_terms``) and sums the moduli of every coefficient.
    For any object that does not expose those iterators (e.g. a non-QED
    operator-shaped duck type) we fall back to ``fallback * num_sites``.

    Conservative — the true bandwidth is typically 2–4× smaller, so the
    auto-tuned ω-window will always cover the spectrum.

    Note: the previous version probed the private C++ attributes
    ``transform_data_`` / ``three_body_data_`` which pybind never exposes,
    so the function silently fell back to ``4 * num_sites`` for every
    bound ``Operator``. The DSSF auto-tuner then chose η / ω-window /
    Krylov against a bandwidth that ignored every coefficient. The
    typed iterators are the supported surface.
    """
    try:
        n = int(operator.num_sites)
    except (AttributeError, TypeError):
        return float(fallback)

    total_coeff = 0.0
    seen_any = False
    # The bound Operator exposes (op_type, site, coeff) tuples for one-body,
    # (op_type_1, site_1, op_type_2, site_2, coeff) for two-body, and
    # (...six tags..., coeff) for three-body. We only care about |coeff|.
    for iter_name in (
        "iter_one_body_terms",
        "iter_two_body_terms",
        "iter_three_body_terms",
    ):
        it = getattr(operator, iter_name, None)
        if it is None:
            continue
        try:
            terms = it()
        except TypeError:
            terms = it
        for t in terms or []:
            try:
                coeff = t[-1]  # coefficient is always the last tuple element
                total_coeff += abs(complex(coeff))
                seen_any = True
            except (TypeError, ValueError, IndexError):
                continue
    if not seen_any:
        return float(fallback) * n
    # Each |coeff| acts on the full Hilbert space, so 2 × |Σ c| is a
    # safe Gershgorin-style upper bound on the half-bandwidth.
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
    passing your own grid via ``qed.spectral(omega=...)``.
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
# Convenience bundle — used by qed.spectral(...) to fill in EVERY
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
    "TunedDiagKnobs",
    "estimate_bandwidth",
    "pick_omega_window",
    "pick_num_omega_points",
    "pick_eta",
    "pick_krylov_dim",
    "pick_num_random_vectors",
    "pick_kpm_moments",
    "pick_device",
    "pick_solver",
    "pick_max_iterations",
    "pick_block_size",
    "pick_tolerance",
    "pick_ftlm_krylov_dim",
    "pick_ltlm_krylov_dim",
    "pick_tpq_taylor_order",
    "pick_tpq_delta_beta",
    "pick_num_thermal_samples",
    "tune_dssf",
    "tune_diag",
]


# ===========================================================================
#                       ED solver auto-tuning helpers
# ===========================================================================
#
# Mirrors the DSSF helpers above for the diagonalization side. Used
# by :func:`qed.solve` (via :func:`tune_diag`). The C++ mirror of
# these knobs now lives in the per-kernel options structs
# (``FtlmKernelOptions``, ``LtlmKernelOptions``, ``TpqKernelOptions``,
# ...) that ``ed::workflows::solve`` and ``ed::workflows::thermal``
# build up from ``SolveOptions`` / ``ThermalOptions``. The numeric
# constants are kept here so the C++ side can copy them verbatim.
#
# Per-level knob bounds were chosen to match the bake-off vs xdiag in
# docs/benchmarks/bench_vs_xdiag.md. "Balanced" is the default and is
# what `qed.solve` ships today; "conservative" trims memory at the cost
# of a few extra restarts; "aggressive" widens subspace + ncv for
# stiff problems.
# ---------------------------------------------------------------------------


# Solver picker thresholds. Matches qed.workflow._resolve_solver and
# the orchestrator's solver picker in ``ed::workflows::solve``. Surface
# them as module-level constants so the C++ mirror (per-kernel options
# structs) can copy them verbatim.
_SMALL_DIM_THRESHOLD = 2048           # FULL below this
_LANCZOS_NEIG_THRESHOLD = 5           # plain Lanczos at or below
_KRYLOV_SCHUR_NEIG_THRESHOLD = 20     # KRYLOV_SCHUR at or below
                                       # else BLOCK_LANCZOS

# Tolerance defaults — looser for big sectors (the eigensolver typically
# converges to ε_machine · ‖H‖, which dominates 1e-12 once ‖H‖·D > 1e10).
_TOLERANCE = {
    "conservative": 1e-8,
    "balanced":     1e-10,
    "aggressive":   1e-12,
}

# Lanczos / Krylov-Schur outer-iteration bounds (per requested eigenvalue).
# Each level: (max_iter_floor, max_iter_per_eig). Used by pick_max_iterations.
_KRYLOV_BOUNDS = {
    "conservative": (150,  6),
    "balanced":     (200,  8),
    "aggressive":   (400, 16),
}

# FTLM / LTLM Krylov dim. Larger M reduces statistical error in
# ⟨e^{-βH}⟩_R but costs O(M) reorthogonalisations per random vector.
_FTLM_KRYLOV = {"conservative": 80,  "balanced": 100, "aggressive": 160}
_LTLM_KRYLOV = {"conservative": 150, "balanced": 200, "aggressive": 320}

# mTPQ Taylor order + delta_beta. Imaginary-time Taylor expansion of
# e^{-Δβ·H/2}: order p ≈ 50 is accurate to 1e-10 for ‖H‖·Δβ ≲ 5;
# delta_beta should be ≲ 0.5 / ‖H‖ to keep truncation error bounded.
_TPQ_TAYLOR_ORDER = {"conservative": 50,  "balanced": 100, "aggressive": 200}
_TPQ_DELTA_BETA   = {"conservative": 5e-2, "balanced": 1e-2, "aggressive": 2e-3}

# Thermal sample count R. Variance scales as 1/(R·D); for big sectors
# even one sample is adequate.
_THERMAL_SAMPLES_MIN = {"conservative": 1, "balanced": 1,  "aggressive": 4}
_THERMAL_SAMPLES_MAX = {"conservative": 4, "balanced": 16, "aggressive": 32}


def pick_solver(num_eigenvalues: int, sector_dim: int) -> str:
    """Return the recommended ``DiagonalizationMethod`` name string.

    Same rule as :func:`qed.solve`'s default (and the C++ auto-pilot):

    * ``sector_dim ≤ 2048``         → ``"FULL"``      (LAPACK)
    * ``num_eigenvalues ≤ 5``       → ``"LANCZOS"``
    * ``num_eigenvalues ≤ 20``      → ``"KRYLOV_SCHUR"``
    * else                          → ``"BLOCK_LANCZOS"``

    Returned as a string so callers without ``qed._core`` available
    (notebook prototyping, doctest) can still introspect the rule.
    """
    if sector_dim <= _SMALL_DIM_THRESHOLD:
        return "FULL"
    if num_eigenvalues <= _LANCZOS_NEIG_THRESHOLD:
        return "LANCZOS"
    if num_eigenvalues <= _KRYLOV_SCHUR_NEIG_THRESHOLD:
        return "KRYLOV_SCHUR"
    return "BLOCK_LANCZOS"


def pick_max_iterations(num_eigenvalues: int, sector_dim: int,
                        *, level: Level = "balanced") -> int:
    """Krylov outer-iteration cap. Floor + per-eig term, capped at D-1."""
    level = _check_level(level)
    floor, per_eig = _KRYLOV_BOUNDS[level]
    target = max(floor, per_eig * max(1, num_eigenvalues) + 80)
    if sector_dim > 1:
        target = min(target, max(1, sector_dim - 1))
    return int(target)


def pick_block_size(num_eigenvalues: int, *, level: Level = "balanced") -> int:
    """Block size for BLOCK_LANCZOS."""
    _ = _check_level(level)
    return max(1, min(num_eigenvalues, 4))


def pick_tolerance(*, level: Level = "balanced") -> float:
    """Convergence tolerance for eigenvalue solvers."""
    return _TOLERANCE[_check_level(level)]


def pick_ftlm_krylov_dim(*, level: Level = "balanced") -> int:
    """FTLM Lanczos micro-basis dimension M (per random vector)."""
    return _FTLM_KRYLOV[_check_level(level)]


def pick_ltlm_krylov_dim(*, level: Level = "balanced") -> int:
    """LTLM excitation Krylov dim."""
    return _LTLM_KRYLOV[_check_level(level)]


def pick_tpq_taylor_order(bandwidth: float, delta_beta: float,
                          *, level: Level = "balanced") -> int:
    """mTPQ Taylor order p.

    For e^{-Δβ·H/2} the truncation error of an order-p Taylor expansion
    is bounded by ``(‖H‖·Δβ/2)^p / p!``. We pick ``p`` such that this is
    < 1e-12 at the chosen ``level``, with per-level minima.
    """
    level = _check_level(level)
    base = _TPQ_TAYLOR_ORDER[level]
    if bandwidth <= 0 or delta_beta <= 0:
        return base
    arg = 0.5 * bandwidth * delta_beta
    if arg <= 1.0:
        return base
    p = base
    log_arg = math.log(arg)
    for _ in range(2 * base):
        if p * log_arg - sum(math.log(k) for k in range(1, p + 1)) < -27.6:
            return p
        p += 10
    return p


def pick_tpq_delta_beta(bandwidth: float,
                        *, level: Level = "balanced") -> float:
    """mTPQ imaginary-time step Δβ.

    Capped at ``0.5 / ‖H‖`` so the per-level baseline never violates the
    truncation-error bound; otherwise picks the level default.
    """
    level = _check_level(level)
    base = _TPQ_DELTA_BETA[level]
    if bandwidth <= 0:
        return base
    return min(base, 0.5 / bandwidth)


def pick_num_thermal_samples(sector_dim: int,
                             *, level: Level = "balanced") -> int:
    """Thermal sample count R. ∝ 1/√D, clamped to per-level [min, max]."""
    level = _check_level(level)
    lo, hi = _THERMAL_SAMPLES_MIN[level], _THERMAL_SAMPLES_MAX[level]
    if sector_dim <= 1:
        return hi
    target = max(1, int(math.ceil(64.0 / math.sqrt(float(sector_dim)))))
    return int(max(lo, min(hi, target)))


# ---------------------------------------------------------------------------
# Convenience bundle — output of :func:`tune_diag`. Mirrors the
# `TunedDSSFKnobs` shape so callers can introspect / log uniformly.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TunedDiagKnobs:
    """Auto-selected ED-solver knobs (output of :func:`tune_diag`)."""
    solver: str
    device: str
    tolerance: float
    max_iterations: int
    block_size: int
    ftlm_krylov_dim: int
    ltlm_krylov_dim: int
    ltlm_ground_krylov: int
    tpq_taylor_order: int
    tpq_delta_beta: float
    num_thermal_samples: int
    bandwidth: float
    level: str

    def to_extra_params(self) -> dict[str, object]:
        """Render to an ``extra_params=`` dict for :func:`qed.solve`.

        Only includes fields actually consumed by the requested
        ``solver`` family — keeps the dict small so logging is readable.
        """
        d: dict[str, object] = {
            "tolerance":          self.tolerance,
            "max_iterations":     self.max_iterations,
        }
        s = self.solver.upper()
        if "BLOCK" in s:
            d["block_size"] = self.block_size
        if s.startswith("FTLM"):
            d["ftlm_krylov_dim"] = self.ftlm_krylov_dim
        if s.startswith("LTLM"):
            d["ltlm_krylov_dim"] = self.ltlm_krylov_dim
            d["ltlm_ground_krylov"] = self.ltlm_ground_krylov
        if "TPQ" in s:
            d["tpq_taylor_order"] = self.tpq_taylor_order
            d["tpq_delta_beta"]   = self.tpq_delta_beta
        if any(t in s for t in ("FTLM", "LTLM", "TPQ")):
            d["num_samples"] = self.num_thermal_samples
        return d


def tune_diag(
    *,
    operator=None,
    sector_dim: Optional[int] = None,
    num_eigenvalues: int = 1,
    bandwidth: Optional[float] = None,
    solver: Optional[str] = None,
    device: Optional[str] = None,
    has_cuda_build: bool = False,
    has_mpi_build: bool = False,
    level: Level = "balanced",
    # Per-knob overrides — anything provided wins over the heuristic.
    tolerance: Optional[float] = None,
    max_iterations: Optional[int] = None,
    block_size: Optional[int] = None,
    ftlm_krylov_dim: Optional[int] = None,
    ltlm_krylov_dim: Optional[int] = None,
    ltlm_ground_krylov: Optional[int] = None,
    tpq_taylor_order: Optional[int] = None,
    tpq_delta_beta: Optional[float] = None,
    num_thermal_samples: Optional[int] = None,
) -> TunedDiagKnobs:
    """Pick every ED-solver knob, honouring user overrides.

    Sister of :func:`tune_dssf`. Used by :func:`qed.solve` to fill in the
    family-specific fields of :class:`EDParameters` that the legacy
    ``_make_params`` path does not handle (FTLM / LTLM Krylov dim,
    mTPQ Taylor order + delta_beta, thermal sample count).

    Anything left as ``None`` is auto-selected from
    (``operator`` / ``sector_dim`` / ``num_eigenvalues``) and the
    requested ``level``.
    """
    level = _check_level(level)

    # Bandwidth.
    if bandwidth is None:
        bandwidth = (estimate_bandwidth(operator)
                     if operator is not None else 4.0)

    # Sector dim.
    if sector_dim is None and operator is not None:
        try:
            sector_dim = int(operator.dimension)
        except AttributeError:
            sector_dim = None
    if sector_dim is None:
        sector_dim = 1024

    # Solver + device.
    chosen_solver = solver or pick_solver(num_eigenvalues, sector_dim)
    chosen_device = pick_device(
        sector_dim,
        has_cuda_build=has_cuda_build,
        has_mpi_build=has_mpi_build,
        user_request=device,
    )

    return TunedDiagKnobs(
        solver=chosen_solver,
        device=chosen_device,
        tolerance=float(tolerance if tolerance is not None
                        else pick_tolerance(level=level)),
        max_iterations=int(max_iterations if max_iterations is not None
                           else pick_max_iterations(num_eigenvalues,
                                                    sector_dim, level=level)),
        block_size=int(block_size if block_size is not None
                       else pick_block_size(num_eigenvalues, level=level)),
        ftlm_krylov_dim=int(ftlm_krylov_dim if ftlm_krylov_dim is not None
                            else pick_ftlm_krylov_dim(level=level)),
        ltlm_krylov_dim=int(ltlm_krylov_dim if ltlm_krylov_dim is not None
                            else pick_ltlm_krylov_dim(level=level)),
        ltlm_ground_krylov=int(ltlm_ground_krylov
                               if ltlm_ground_krylov is not None
                               else pick_ftlm_krylov_dim(level=level)),
        tpq_delta_beta=float(tpq_delta_beta if tpq_delta_beta is not None
                             else pick_tpq_delta_beta(bandwidth, level=level)),
        tpq_taylor_order=int(tpq_taylor_order if tpq_taylor_order is not None
                             else pick_tpq_taylor_order(
                                 bandwidth,
                                 (tpq_delta_beta if tpq_delta_beta is not None
                                  else pick_tpq_delta_beta(bandwidth, level=level)),
                                 level=level)),
        num_thermal_samples=int(num_thermal_samples
                                if num_thermal_samples is not None
                                else pick_num_thermal_samples(sector_dim,
                                                              level=level)),
        bandwidth=float(bandwidth),
        level=level,
    )
