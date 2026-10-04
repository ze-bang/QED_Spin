"""``qed.dssf``: operators for dynamical structure factors.

Builds the momentum-resolved spin operators, one per (Q, component), with their names and
ordering; feed them to :func:`qed.dynamics`.

.. code-block:: python

    import numpy as np
    import qed

    spec = qed.dssf.OperatorSpec()
    spec.operator_type   = "transverse"
    spec.basis           = "xyz"
    spec.components      = [0, 1]          # Sx, Sy in the xyz basis
    spec.momentum_points = [[0.0, 0.0, 0.0], [3.14159, 0.0, 0.0]]
    spec.polarization    = [0.0, 0.0, 1.0]
    spec.num_sites       = 4
    spec.positions_file  = "/abs/path/to/positions.dat"

    obs = qed.dssf.build_observables(spec)
    S = qed.dynamics(H, obs.operators[0], np.linspace(-2, 2, 200)).S[0]

Operator types (``R_i`` from ``positions_file``, ``phi_i = e^{-i Q.R_i} / sqrt(N)``, the package's
convention: qed.Family.fourier, qed._geometry):

* ``"sum"``: ``sum_i phi_i S^a_i`` per component ``a``; name ``{C}_q_Qx{Qx}_Qy{Qy}_Qz{Qz}``.
* ``"transverse"``: ``sum_i phi_i (e.z_{i mod 4}) S^a_i`` for the two transverse directions
  ``e1`` = the polarization and ``e2`` (:func:`compute_transverse_bases`), with the pyrochlore
  local axes ``z_mu``; names ``..._NSF`` (e1) and ``..._SF`` (e2).
* ``"sublattice"``: the sum over the sites ``s, s + U, s + 2U, ...`` (``U = unit_cell_size``) of
  one sublattice ``s`` (every one unless ``sublattice`` is set); name ``..._sub{s}``.
* ``"experimental"`` / ``"transverse_experimental"``: ``cos(theta) S^z + sin(theta) S^x`` with
  the sum / transverse weights (``components`` and ``basis`` are not used).

``basis="ladder"`` takes components 0, 1, 2 as S+, S-, S^z (names Sp, Sm, Sz); ``"xyz"`` as
S^x, S^y, S^z.
"""

from __future__ import annotations

import math
import operator as _op
from typing import List, Optional, Sequence

from . import _core
from .errors import InvalidRequest

__all__ = [
    "Observables",
    "OperatorSpec",
    "build_observables",
    "compute_transverse_bases",
]

_TYPES = ("sum", "transverse", "sublattice", "experimental", "transverse_experimental")
_INV_SQRT3 = 0.5773502691896258
_PYRO_AXES = (
    (-_INV_SQRT3, -_INV_SQRT3, -_INV_SQRT3),
    (-_INV_SQRT3, _INV_SQRT3, _INV_SQRT3),
    (_INV_SQRT3, -_INV_SQRT3, _INV_SQRT3),
    (_INV_SQRT3, _INV_SQRT3, -_INV_SQRT3),
)
_ZERO_TOL = 1e-10


def _index(x, what: str, minimum: int = 0) -> int:
    if isinstance(x, (bool, float)) or not hasattr(x, "__index__"):
        raise TypeError(f"OperatorSpec.{what} must be an integer, got {type(x).__name__}")
    i = _op.index(x)
    if i < minimum:
        raise TypeError(f"OperatorSpec.{what} must be >= {minimum}, got {i}")
    return i


def _floats(xs, what: str) -> List[float]:
    out = []
    for x in xs:
        if isinstance(x, (str, bytes)) or not isinstance(x, (int, float)) and not hasattr(x, "__float__"):
            raise TypeError(f"OperatorSpec.{what} entries must be numbers, got {type(x).__name__}")
        out.append(float(x))
    return out


class OperatorSpec:
    """What :func:`build_observables` builds: ``operator_type``, ``basis``, ``components``,
    ``momentum_points`` (3-vectors, absolute units), ``polarization``, ``theta``,
    ``unit_cell_size``, ``num_sites``, ``positions_file`` and ``sublattice`` (None: all)."""

    __slots__ = (
        "_operator_type",
        "_basis",
        "_components",
        "_momentum_points",
        "_polarization",
        "_theta",
        "_unit_cell_size",
        "_num_sites",
        "_positions_file",
        "_sublattice",
    )

    def __init__(self):
        self._operator_type = "sum"
        self._basis = "ladder"
        self._components: List[int] = []
        self._momentum_points: List[List[float]] = []
        self._polarization: List[float] = [1.0, 0.0, 0.0]
        self._theta = 0.0
        self._unit_cell_size = 4
        self._num_sites = 0
        self._positions_file = ""
        self._sublattice: Optional[int] = None

    def _str(self, v, what):
        if not isinstance(v, str):
            raise TypeError(f"OperatorSpec.{what} must be a str, got {type(v).__name__}")
        return v

    operator_type = property(
        lambda s: s._operator_type, lambda s, v: setattr(s, "_operator_type", s._str(v, "operator_type"))
    )
    basis = property(lambda s: s._basis, lambda s, v: setattr(s, "_basis", s._str(v, "basis")))
    positions_file = property(
        lambda s: s._positions_file, lambda s, v: setattr(s, "_positions_file", s._str(v, "positions_file"))
    )
    components = property(
        lambda s: list(s._components),
        lambda s, v: setattr(s, "_components", [_index(c, "components", -(2**63)) for c in v]),
    )
    momentum_points = property(
        lambda s: [list(q) for q in s._momentum_points],
        lambda s, v: setattr(s, "_momentum_points", [_floats(q, "momentum_points") for q in v]),
    )
    polarization = property(
        lambda s: list(s._polarization), lambda s, v: setattr(s, "_polarization", _floats(v, "polarization"))
    )
    theta = property(lambda s: s._theta, lambda s, v: setattr(s, "_theta", _floats([v], "theta")[0]))
    unit_cell_size = property(
        lambda s: s._unit_cell_size, lambda s, v: setattr(s, "_unit_cell_size", _index(v, "unit_cell_size"))
    )
    num_sites = property(lambda s: s._num_sites, lambda s, v: setattr(s, "_num_sites", _index(v, "num_sites")))
    sublattice = property(
        lambda s: s._sublattice, lambda s, v: setattr(s, "_sublattice", None if v is None else _index(v, "sublattice"))
    )

    def __repr__(self) -> str:
        return (
            f"<qed.dssf.OperatorSpec operator_type='{self._operator_type}' basis='{self._basis}' "
            f"num_sites={self._num_sites} momenta={len(self._momentum_points)} "
            f"components={len(self._components)}>"
        )


class Observables:
    """The operators :func:`build_observables` made, with their names (same order)."""

    __slots__ = ("_operators", "_names")

    def __init__(self, operators: Sequence, names: Sequence[str]):
        self._operators = list(operators)
        self._names = list(names)

    @property
    def operators(self) -> list:
        return list(self._operators)

    @property
    def names(self) -> List[str]:
        return list(self._names)

    def __len__(self) -> int:
        return len(self._names)


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def _normalize(v):
    n = math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    if n < _ZERO_TOL:
        return [0.0, 0.0, 0.0]
    return [v[0] / n, v[1] / n, v[2] / n]


def compute_transverse_bases(Q: Sequence[float], polarization: Sequence[float]):
    """``(e1, e2)``: ``e1`` is the polarization as given, ``e2`` the unit vector along
    ``Q x polarization`` (along ``y x pol`` or ``x x pol`` when Q is parallel to it)."""
    Q, pol = [float(x) for x in Q], [float(x) for x in polarization]
    if len(Q) != 3:
        raise InvalidRequest("ed::dssf::compute_transverse_bases: Q must be a 3-vector")
    if len(pol) != 3:
        raise InvalidRequest("ed::dssf::compute_transverse_bases: polarization must be a 3-vector")
    c = _cross(Q, pol)
    if math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) < _ZERO_TOL:
        e2 = _normalize(_cross([0.0, 1.0, 0.0] if abs(pol[0]) > 0.5 else [1.0, 0.0, 0.0], pol))
    else:
        e2 = _normalize(c)
    return list(pol), e2


def _read_positions(path: str, n: int) -> List[List[float]]:
    """One line per site, ``x y z`` or ``id x y z`` (the id is the site's index, counting from 0;
    the 4-column form of ``qed.input.lattice.from_cluster_file``); empty lines, lines starting with
    '#' and lines whose first field is not a number are skipped. Exactly ``n`` sites. Any other
    column count, an id that is not the site's index, a coordinate that is not a number and a
    file that cannot be opened raise InvalidRequest."""
    try:
        f = open(path)
    except OSError:
        raise InvalidRequest(f"ed::core::detail::read_positions_file: could not open {path}") from None
    out = []
    with f:
        for k, line in enumerate(f, 1):
            if not line.strip() or line.startswith("#"):
                continue
            fields = line.split()
            try:
                float(fields[0])
            except ValueError:
                continue
            if len(fields) not in (3, 4):
                raise InvalidRequest(
                    f"qed.dssf: {path} line {k}: a site is 'x y z' or 'id x y z', read {len(fields)} columns"
                )
            if len(fields) == 4:
                try:
                    site_id = int(fields[0])
                except ValueError:
                    site_id = -1
                if site_id != len(out):
                    raise InvalidRequest(
                        f"qed.dssf: {path} line {k}: the id must be the site's index, counting from 0: "
                        f"expected {len(out)}, read {fields[0]!r}"
                    )
                fields = fields[1:]
            try:
                out.append([float(x) for x in fields])
            except ValueError:
                raise InvalidRequest(f"qed.dssf: {path} line {k}: a coordinate is not a number") from None
    if len(out) != n:
        raise InvalidRequest(f"qed.dssf: {path} lists {len(out)} sites for num_sites = {n}")
    return out


def _phases(Q, positions, norm: float) -> List[complex]:
    out = []
    for r in positions:
        dot = 0.0
        for d in range(3):
            dot += Q[d] * r[d]
        out.append(complex(norm * math.cos(dot), -norm * math.sin(dot)))   # e^{-i Q.R} (qed._geometry)
    return out


def _transverse_phases(Q, v, positions) -> List[complex]:
    norm = 1.0 / math.sqrt(len(positions))
    out = []
    for i, r in enumerate(positions):
        dot = 0.0
        for d in range(3):
            dot += Q[d] * r[d]
        z = _PYRO_AXES[i % 4]
        vz = 0.0
        for d in range(3):
            vz += v[d] * z[d]
        w = norm * vz
        out.append(complex(w * math.cos(dot), -w * math.sin(dot)))   # e^{-i Q.R}
    return out


def _component_name(c: int, xyz: bool) -> str:
    return (("Sx", "Sy", "Sz") if xyz else ("Sp", "Sm", "Sz"))[c]


def _stem(prefix: str, Q) -> str:
    return f"{prefix}_q_Qx{format(Q[0], 'g')}_Qy{format(Q[1], 'g')}_Qz{format(Q[2], 'g')}"


def _site_terms(op, c: int, site: int, phase: complex, xyz: bool) -> None:
    if not xyz:
        op.add_one_body(c, site, phase)
    elif c == 0:  # Sx = (S+ + S-)/2
        op.add_one_body(0, site, complex(phase.real * 0.5, phase.imag * 0.5))
        op.add_one_body(1, site, complex(phase.real * 0.5, phase.imag * 0.5))
    elif c == 1:  # Sy = -i (S+ - S-)/2
        op.add_one_body(0, site, phase * complex(0.0, -0.5))
        op.add_one_body(1, site, phase * complex(0.0, 0.5))
    else:
        op.add_one_body(2, site, phase)


def _experimental_terms(op, site: int, phase: complex, cos_t: float, sin_t: float) -> None:
    op.add_one_body(2, site, complex(phase.real * cos_t, phase.imag * cos_t))
    h = 0.5 * sin_t
    op.add_one_body(0, site, complex(phase.real * h, phase.imag * h))
    op.add_one_body(1, site, complex(phase.real * h, phase.imag * h))


def build_observables(spec: OperatorSpec) -> Observables:
    """One :class:`qed.Operator` per (Q, component) -- and per transverse direction or
    sublattice -- as ``spec`` describes, with names; see the module docstring."""
    if not isinstance(spec, OperatorSpec):
        raise TypeError("build_observables takes a qed.dssf.OperatorSpec")
    t = spec._operator_type
    experimental = t in ("experimental", "transverse_experimental")
    if t not in _TYPES:
        raise InvalidRequest(f"ed::dssf::build_observables: unknown operator_type '{t}'")
    if not spec._components and not experimental:
        raise InvalidRequest("ed::dssf::build_observables: components is empty")
    if not spec._momentum_points:
        raise InvalidRequest("ed::dssf::build_observables: momentum_points is empty")
    if len(spec._polarization) != 3:
        raise InvalidRequest("ed::dssf::build_observables: polarization must be a 3-vector")
    n = spec._num_sites
    if n == 0:
        raise InvalidRequest("ed::dssf::build_observables: num_sites must be > 0")
    for Q in spec._momentum_points:
        if len(Q) != 3:
            raise InvalidRequest(f"ed::dssf::build_observables: momentum point {Q} is not a 3-vector")
    if not experimental:
        for c in spec._components:
            if c not in (0, 1, 2):
                raise InvalidRequest(f"ed::dssf::build_observables: component {c} is not 0, 1 or 2")
    U = spec._unit_cell_size
    if t == "sublattice":
        if U == 0:
            raise InvalidRequest("ed::dssf::build_observables: unit_cell_size must be >= 1")
        if spec._sublattice is not None and spec._sublattice >= U:
            raise InvalidRequest(
                f"ed::dssf::build_observables: sublattice {spec._sublattice} is not below " f"unit_cell_size {U}"
            )
    xyz = spec._basis == "xyz"
    positions = _read_positions(spec._positions_file, n)
    ops, names = [], []

    def fresh():
        return _core.Operator(n)

    for Q in spec._momentum_points:
        if t in ("transverse", "transverse_experimental"):
            e1, e2 = compute_transverse_bases(Q, spec._polarization)
        if experimental:
            cos_t, sin_t = math.cos(spec._theta), math.sin(spec._theta)
            theta = f"_theta{format(spec._theta, 'g')}"
            if t == "experimental":
                ph = _phases(Q, positions, 1.0 / math.sqrt(n))
                op = fresh()
                for i in range(n):
                    _experimental_terms(op, i, ph[i], cos_t, sin_t)
                ops.append(op)
                names.append(_stem("Experimental", Q) + theta)
            else:
                stem = _stem("TransverseExperimental", Q) + theta
                for e, tag in ((e1, "_NSF"), (e2, "_SF")):
                    ph = _transverse_phases(Q, e, positions)
                    op = fresh()
                    for i in range(n):
                        _experimental_terms(op, i, ph[i], cos_t, sin_t)
                    ops.append(op)
                    names.append(stem + tag)
            continue
        for c in spec._components:
            stem = _stem(_component_name(c, xyz), Q)
            if t == "sum":
                ph = _phases(Q, positions, 1.0 / math.sqrt(n))
                op = fresh()
                for i in range(n):
                    _site_terms(op, c, i, ph[i], xyz)
                ops.append(op)
                names.append(stem)
            elif t == "transverse":
                for e, tag in ((e1, "_NSF"), (e2, "_SF")):
                    ph = _transverse_phases(Q, e, positions)
                    op = fresh()
                    for i in range(n):
                        _site_terms(op, c, i, ph[i], xyz)
                    ops.append(op)
                    names.append(stem + tag)
            else:  # sublattice: the sum over sites s, s + U, ... (phases normalised over all N)
                ph = _phases(Q, positions, 1.0 / math.sqrt(n))
                subs = [spec._sublattice] if spec._sublattice is not None else range(U)
                for s in subs:
                    op = fresh()
                    for i in range(s, n, U):
                        _site_terms(op, c, i, ph[i], xyz)
                    ops.append(op)
                    names.append(f"{stem}_sub{s}")
    return Observables(ops, names)
