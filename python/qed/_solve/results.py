"""``qed._solve.results``: result shaping for the solve lanes.

Carved out of ``workflow.py`` (WP11). Holds the little-group project
lane's backend envelope (the abelian lane gets its own from C++) and the
Stage 12e total-spin labelling of a complete spectrum by highest-weight
spectral differencing.
"""

from __future__ import annotations

from typing import Optional

from .. import _core as _core
from .._core import (  # type: ignore[attr-defined]
    EDResults,
    Operator,
)


# ---------------------------------------------------------------------------

class _ProjectLaneBackend:
    """Minimal backend envelope for the little-group project lane.

    The abelian lane copies C++ ``GroundStateResult.backend`` (which
    ``select_backend`` fills). The project lane has no such struct -- the
    little-group engine is its own CPU engine with an optional GPU rep-gather
    -- so it reports the engine's OWN answer here. Same duck-type
    (``.lane`` / ``.mpi_size``) as the abelian one, so callers need not branch.
    """

    __slots__ = ("lane", "mpi_size")

    def __init__(self, lane: str, mpi_size: int = 1):
        self.lane = lane
        self.mpi_size = mpi_size

    def __repr__(self) -> str:      # pragma: no cover - diagnostics only
        return f"Backend(lane={self.lane!r}, mpi_size={self.mpi_size})"


# ---------------------------------------------------------------------------
# Stage 12e (SU(2) rollout): full-spectrum total-spin resolution via
# highest-weight SPECTRAL DIFFERENCING -- no vectors, no dense S^2 blocks.
#
# For an SU(2)-invariant H, every spin-S multiplet contributes exactly one
# level to each Sz block with |Sz| <= S, so per |2Sz| class m (using the
# blocks the sweep already solved):
#
#     tower(two_S) = spectrum(m = two_S)  minus  spectrum(m = two_S + 2)
#
# as an exact multiset difference, and each Sz block's spectrum tiles as
# the disjoint union of the towers with two_S >= m. Matching is by sorted
# two-pointer walk with a tolerance; degenerate levels are interchangeable
# within their cluster, so any consistent assignment is a valid labeling.
# ---------------------------------------------------------------------------

def _su2_multiset_diff(a: list, b: list, tol: float) -> list:
    """Sorted-multiset a minus b (b known to embed in a within tol)."""
    out, j = [], 0
    for x in a:
        if j < len(b) and abs(x - b[j]) <= tol:
            j += 1
        else:
            out.append(x)
    return out


def _su2_label_blocks(
    blocks: "list[tuple[Optional[int], list[float]]]",
    n_sites: int,
    tol: float = 1e-8,
) -> "Optional[list[list[int]]]":
    """Label every (n_up, sorted spectrum) block with per-level two_S.

    Returns one int list per input block (parallel), or None when the
    sweep is incomplete (a needed |Sz| class is missing). -1 marks a
    level the tiling failed to match (should not happen for a genuine
    SU(2)-invariant H; kept as a soft failure mode).
    """
    # Representative spectrum per |2Sz| class.
    avail: dict[int, list[float]] = {}
    for n_up, spec in blocks:
        if n_up is None:
            return None
        m = abs(2 * n_up - n_sites)
        avail.setdefault(m, sorted(spec))
    # Need every class from N%2 up to N (missing top classes are fine
    # only if no block needs them -- iterate what the tiling requires).
    towers: dict[int, list[float]] = {}
    for ts in range(n_sites, n_sites % 2 - 1, -2):
        hi = avail.get(ts + 2, [] if ts + 2 > n_sites else None)
        lo = avail.get(ts)
        if lo is None or hi is None:
            # A class the differencing needs was not solved: label only
            # if no block references these towers -- conservatively bail.
            return None
        towers[ts] = _su2_multiset_diff(lo, hi, tol)
    out: list[list[int]] = []
    for n_up, spec in blocks:
        m = abs(2 * n_up - n_sites)
        pool = sorted(
            (e, ts) for ts, lst in towers.items() if ts >= m for e in lst)
        labels, j = [], 0
        for x in sorted(spec):
            if j < len(pool) and abs(x - pool[j][0]) <= tol:
                labels.append(pool[j][1])
                j += 1
            else:
                labels.append(-1)
        out.append(labels)
    return out


def _attach_su2_full_spectrum_labels(
    out: EDResults,
    blocks: "list[tuple[Optional[int], list[float]]]",
    n_sites: int,
    operator: Operator,
    *,
    enabled: bool,
    require: bool,
) -> None:
    """Attach ``out.spin`` / ``out.two_S`` (parallel to the merged sorted
    ``out.eigenvalues``) via highest-weight spectral differencing. Soft
    no-op when disabled / not SU(2)-invariant / sweep incomplete, unless
    ``require`` (then raise on a non-SU(2) Hamiltonian)."""
    import os as _os
    if _os.environ.get("ED_SYM_SU2", "") == "0":
        if require:
            raise RuntimeError(
                "qed.full_spectrum: total_spin='require' but the SU(2) "
                "axis is vetoed by ED_SYM_SU2=0")
        return
    try:
        su2 = bool(_core.detect_hamiltonian_symmetries(operator)["su2"])
    except Exception:
        su2 = False
    if not su2:
        if require:
            raise RuntimeError(
                "qed.full_spectrum: total_spin='require' but the "
                "term-level [H, S_tot] check failed (H is not "
                "SU(2)-invariant)")
        return
    if not enabled or not blocks:
        return
    labels = _su2_label_blocks(blocks, n_sites)
    if labels is None:
        return
    pairs = sorted(
        (e, ts)
        for (nu, spec), labs in zip(blocks, labels)
        for e, ts in zip(sorted(spec), labs))
    if len(pairs) != len(out.eigenvalues):
        return  # sweep/merge mismatch: leave unlabeled rather than lie
    try:
        out.two_S = [ts for _, ts in pairs]
        out.spin = [(ts / 2.0 if ts >= 0 else None) for _, ts in pairs]
    except AttributeError:
        pass
