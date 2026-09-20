"""``qed._solve.request``: the frozen record the solve lanes read.

WP12. :func:`qed.solve` used to thread three dozen loose locals through
one thousand-line body -- the caller's kwargs, then the normalised Sz
axis, then the resolved operator / method / device / parameter bag.
They live here instead, in one frozen record that every lane reads and
no lane can write back into.

The record is built once, near the top of :func:`qed.solve`, from the
kwargs as given. Two refinements follow, at exactly the points where the
old body computed the same values -- position matters, because the lanes
that short-circuit first must not see (or pay for) work that came later:

* ``_resolve_axes`` -- the quantum-number axes (total spin, Sz parity,
  the fixed-Sz flag and the dimensions), which the full-spectrum lane
  returns before;
* ``_resolve_execution`` -- the operator actually solved, the method /
  device decision, the output directory, the ``EDParameters`` bag and
  the resolved symmetry group, which the Sz sweep returns before.

Both live in :mod:`qed._solve.entry` and are wired into the lane table
as the ``prepare`` step of the first lane that needs them. Each returns
a NEW record (:func:`dataclasses.replace`), never a mutated one.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, NamedTuple, Optional, Sequence, Union

from .._core import (  # type: ignore[attr-defined]
    DiagonalizationMethod,
    FixedSzOperator,
    Operator,
)
from .symmetry_input import SymmetryArg


class _Declined:
    """Sentinel: the lane looked at the request and handed it back.

    The old body said this by falling out of an ``if`` -- the vector
    lane whose projection declined, the little-group lane whose engine
    raised -- and continuing down the function. A lane returns this
    instead, and the table moves to the next row.
    """

    __slots__ = ()

    def __repr__(self) -> str:      # pragma: no cover - diagnostics only
        return "<declined>"


DECLINED = _Declined()


@dataclass(frozen=True)
class SolveRequest:
    """Everything :func:`qed.solve` decided, in one immutable record."""

    # -- as the caller spelled them, after the prologue's sz= normalisation
    H: Union[Operator, FixedSzOperator]
    num_eigenvalues: int
    tolerance: float
    compute_eigenvectors: bool
    solver: Optional[Union[str, DiagonalizationMethod]]
    device: Optional[str]
    symmetry: SymmetryArg
    sector: Optional[Sequence[int]]
    irrep: Optional[dict]
    flip: Optional[int]
    sz: Union[int, str, tuple, None]
    auto_sz: bool
    spin_flip: Union[str, bool, int, None]
    time_reversal: Union[str, bool, int, None]
    point_group: Union[str, bool, None]
    total_spin: Union[int, float, str, None]
    lattice: Optional[Any]
    output_dir: str
    max_iterations: Optional[int]
    block_size: Optional[int]
    num_samples: Optional[int]
    target_beta: Optional[float]
    num_temp_points: Optional[int]
    temp_min: Optional[float]
    temp_max: Optional[float]
    verbose: bool
    full_spectrum: bool
    extra_params: Optional[dict[str, Any]]
    # ``sz=(lo, hi)`` after normalize_sz: the magnetisation window the
    # sweeps walk instead of 0..N.
    sz_window: Optional[tuple[int, int]] = None

    # -- resolved by _resolve_axes (quantum-number axes)
    fixed_sz_input: bool = False
    num_sites: int = 0
    base_dim: int = 0
    #: ``2S`` of the targeted SU(2) tower, or -1 when the axis is off.
    two_s: int = -1
    total_spin_label: int = 0
    #: 0 = even, 1 = odd, ``None`` when ``sz=`` did not name a parity half.
    sz_parity: Optional[int] = None

    # -- resolved by _resolve_execution (what actually runs)
    #: The operator handed to the kernels: ``H`` or its fixed-Sz block.
    op: Optional[Operator] = None
    sector_dim: int = 0
    method: Optional[DiagonalizationMethod] = None
    use_gpu: bool = False
    use_mpi: bool = False
    is_thermal: bool = False
    is_tpq: bool = False
    #: ``output_dir`` after the thermal-method substitution.
    effective_output: str = ""
    params: Optional[Any] = None

    @property
    def auto_method(self) -> bool:
        """True when ``solver=`` was unset: the orchestrator picks."""
        return self.solver is None

    def refine(self, **updates: Any) -> "SolveRequest":
        """A new record with ``updates`` applied; the record stays frozen."""
        return replace(self, **updates)


def _as_is(req: SolveRequest) -> SolveRequest:
    """No normalisation is owed before this lane."""
    return req


class Lane(NamedTuple):
    """One row of the ordered lane table.

    ``prepare`` runs before ``predicate`` is evaluated -- whether or not
    the predicate then fires -- and returns the request the rest of the
    table sees; it carries the normalisation the old body did between
    two ``if`` blocks. ``run`` returns an ``EDResults`` or
    :data:`DECLINED`, in which case the table moves on.
    """

    name: str
    predicate: Callable[[SolveRequest], bool]
    run: Callable[[SolveRequest], Any]
    prepare: Callable[[SolveRequest], SolveRequest] = _as_is
