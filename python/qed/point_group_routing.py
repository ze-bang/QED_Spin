"""Stage 9c (SymmetryEngine v2): ONE point-group routing decision.

Every verb used to carry its own ``point_group == "full"`` branch with a
monolithic-SAB fallback. This module is the single decision point for the
projection lane instead:

* ``point_group="auto"`` (the default): PROJECT through the factorized
  little-group engine whenever the consumer is eigenvalue-only (no
  eigenvectors, an eigenvalue method) and a (abelian, residue) split
  exists. (``sector=`` no longer declines: a named momentum restricts the
  engine's star walk via ``only_k0`` instead of forcing the abelian lane.) Any decline falls back silently to the
  abelian rep lane with star/TR/flip *folds* -- the pre-9c behaviour.
  The thermal verb is the exception: its sampling methods (mTPQ/FTLM/
  LTLM) are the large-N design point, and the projection lane computes
  EXACT full-spectrum thermodynamics per block -- auto never hijacks a
  sampling run into an exponentially costlier exact one, so thermal
  projects only under the explicit ``point_group="full"`` contract.
* ``point_group="full"``: REQUIRE projection -- raises with the decline
  reason instead of silently degrading. (Until Stage 9c this fell back
  to the monolithic SAB engine; that engine is now a test oracle and is
  no longer production-routed.)
* ``point_group=off/none/False``: the abelian lane with star folds also
  disabled (unchanged semantics).

``split_nonabelian`` generalizes the old ``_little_group_parts``: it also
accepts an EXPLICIT (possibly non-abelian) permutation list, closing the
full group and carving out a maximal-abelian subgroup + coset residues --
the case that previously had no route except the monolithic engine. A
greedy commuting-subgroup choice is always safe: a smaller A only means
larger stars (less momentum reduction), never wrong physics -- the
engine's covering sum rule remains the tripwire.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

__all__ = ["ProjectionLane", "split_nonabelian", "resolve_projection_lane",
           "decode_star_for_sector", "decode_irrep_for_character"]

_GROUP_CLOSURE_CAP = 4096   # A (and the closed full group) must stay enumerable


@dataclass
class ProjectionLane:
    mode: str                          # "project" | "abelian"
    A: Optional[list] = None           # closed abelian elements (project mode)
    residues: Optional[list] = None    # point-group coset representatives
    reason: str = ""                   # why the lane declined (diagnostics)


def _compose(g, e):
    """U-composition convention (matches irreps.cpp): (g.e)[i] = e[g[i]]."""
    return tuple(e[g[i]] for i in range(len(g)))


def _close(gens, cap=_GROUP_CLOSURE_CAP):
    """BFS closure of a permutation set. None when the group exceeds cap."""
    gens = [tuple(g) for g in gens]
    if not gens:
        return None
    n = len(gens[0])
    ident = tuple(range(n))
    elems = {ident}
    frontier = [ident]
    while frontier:
        nxt = []
        for e in frontier:
            for g in gens:
                c = _compose(g, e)
                if c not in elems:
                    if len(elems) >= cap:
                        return None
                    elems.add(c)
                    nxt.append(c)
        frontier = nxt
    return sorted(elems)


def _commute(a, b):
    return _compose(a, b) == _compose(b, a)


def split_nonabelian(symmetry_or_gens):
    """``(abelian_elements, residue_perms)`` for the factorized engine,
    or a ``str`` decline reason.

    * ``GeneratorSet``-like input (``generators`` + ``star_perms``): the
      abelian clique is closed; the retained residue is the coset set --
      the original ``_little_group_parts`` behaviour.
    * explicit permutation list: the FULL group is closed, a maximal
      abelian subgroup is chosen greedily (any commuting closure is
      valid -- smaller A just folds less), and one representative per
      A-coset of the remainder becomes the residue set.
    """
    gens = getattr(symmetry_or_gens, "generators", None)
    if gens is not None:
        star = list(getattr(symmetry_or_gens, "star_perms", None) or [])
        if not gens:
            return "the symmetry has no spatial generators"
        A = _close([list(g) for g in gens])
        if A is None:
            return (f"the abelian group exceeds the {_GROUP_CLOSURE_CAP}-"
                    "element closure cap")
        if not star:
            return ("no point-group residue is retained on the symmetry "
                    "(symmetry='auto' or a find_symmetries GeneratorSet "
                    "carry one; pure-abelian input has nothing to project)")
        # Dedup the retained residues modulo A. find_symmetries carries the
        # FULL non-clique content as star_perms (e.g. all 30 elements of the
        # C2*A coset on a 5x6 torus; 396 on the 6x6 C6v torus). Same-coset
        # residues act as proportional monomials -- the engine validates and
        # discards each duplicate, paying the O(dim) monomial build per
        # element for zero extra reduction (measured 4x total at 30 sites).
        # One representative per A-coset is the complete input.
        Aset = {tuple(a) for a in A}
        residues, covered = [], set(Aset)
        for p in star:
            tp = tuple(p)
            if tp in covered:
                continue
            residues.append(list(tp))
            covered.update(_compose(a, tp) for a in Aset)
        if not residues:
            return ("every retained residue lies in the abelian group -- "
                    "nothing to project beyond the momentum sectors")
        return ([list(e) for e in A], residues)

    # Explicit raw permutation list.
    perms = [tuple(p) for p in (symmetry_or_gens or [])]
    if not perms:
        return "no generators given"
    G = _close(perms)
    if G is None:
        return (f"the closed group exceeds the {_GROUP_CLOSURE_CAP}-element "
                "cap -- pass a GeneratorSet from find_symmetries instead")
    # Greedy maximal commuting subgroup (the closure of pairwise-commuting
    # elements is abelian by construction). Seed with the HIGHEST-order
    # elements first: long cycles (translations) build a large cyclic core,
    # where naive sorted order tends to lock in an early involution (e.g.
    # a reflection) and end up with a small Klein-type subgroup. Any
    # commuting closure is *valid* -- smaller A only folds less -- this
    # ordering just maximizes the reduction.
    def _order(e):
        k, c = 1, e
        ident = tuple(range(len(e)))
        while c != ident:
            c = _compose(e, c)
            k += 1
        return k

    A = [tuple(range(len(perms[0])))]
    Aset = set(A)
    for e in sorted(G, key=lambda e: (-_order(e), e)):
        if e in Aset:
            continue
        if all(_commute(e, a) for a in A):
            closed = _close(A + [e])
            if closed is not None and len(closed) <= _GROUP_CLOSURE_CAP:
                A = [tuple(a) for a in closed]
                Aset = set(A)
    if len(A) <= 1:
        return "no non-trivial abelian subgroup found in the closed group"
    # One representative per A-coset of the remainder.
    residues, covered = [], set(Aset)
    for e in G:
        if e in covered:
            continue
        residues.append(list(e))
        covered.update(_compose(a, e) for a in Aset)
    if not residues:
        return ("the input group is abelian -- nothing to project beyond "
                "the momentum sectors (use the abelian rep lane)")
    return ([list(a) for a in sorted(Aset)], residues)


def decode_star_for_sector(stars, irrep_characters, A, generators, orders,
                           sector, flip=None):
    """Which star(s) does the caller's momentum live in?

    ``sector`` names one quantum number per GENERATOR: generator ``a_i`` (of
    order ``o_i``) acts with phase ``exp(-2*pi*i*q_i/o_i)``, the directory's
    ``phase_factors`` convention. Returns ``(k0, k_raw)`` -- the star
    REPRESENTATIVE to hand to ``only_k0``, and the caller's own raw irrep
    index -- or a ``str`` decline reason.

    Why this exists: the engine's ``k_raw`` is an INTERNAL irrep index from
    ``decompose_irreps`` and is NOT the momentum (on a 12-ring, k_raw=8 is
    q=0 and k_raw=9 is q=pi). The only honest way across that boundary is the
    one the engine's own header prescribes: read the momentum off the
    generator's phase in ``irrep_characters[k_raw][a]``, never trust an index
    convention. And ``only_k0`` filters on star REPRESENTATIVES, so a momentum
    that is a non-representative member has to be mapped to its star -- the
    members are isospectral by construction, so the representative's spectrum
    is the answer for every one of them.
    """
    import cmath

    qns = [int(q) for q in sector]
    gens = [list(g) for g in generators]
    ords = [int(o) for o in orders]
    if len(qns) != len(gens):
        return (f"sector= names {len(qns)} quantum number(s) but the symmetry "
                f"has {len(gens)} generator(s)")

    # Locate each generator inside the caller's own A ordering: that is the
    # index axis irrep_characters is expressed in.
    A_t = [tuple(int(x) for x in a) for a in A]
    want = []
    for g, o, q in zip(gens, ords, qns):
        try:
            a_idx = A_t.index(tuple(g))
        except ValueError:
            return ("a generator is not present in the closed abelian group "
                    "(cannot decode its character)")
        want.append((a_idx, cmath.exp(-2j * cmath.pi * (q % o) / o)))

    k_raw = None
    for kk, row in enumerate(irrep_characters):
        if all(abs(complex(row[a_idx]) - ph) < 1e-8 for a_idx, ph in want):
            k_raw = kk
            break
    if k_raw is None:
        return (f"no abelian irrep carries the requested quantum numbers "
                f"{qns} (orders {ords})")

    # Flip splits each momentum into (k,+) and (k,-), carried as EXTENDED
    # indices k_raw + s*n_irr_raw. A raw k_raw therefore only ever matches the
    # PARITY-0 star, so decoding it naively returns the + half and silently
    # drops the - half -- the caller names a momentum and gets half its
    # spectrum with no indication the rest exists.
    n_irr = len(irrep_characters)
    engaged = any(int(st.get("flip_parity", -1)) >= 0 for st in stars)
    if flip is None:
        # Name nothing -> get ALL of it: both parities when flip is engaged.
        wanted = [k_raw + s * n_irr for s in (0, 1)] if engaged else [k_raw]
    else:
        if not engaged:
            return ("flip= was given but the spin-flip Z2 is not engaged here "
                    "(it needs [H, prod sigma^x] = 0 AND a flip-invariant "
                    "subspace: n_up = N/2, an Sz-parity half with N even, or "
                    "the full space)")
        if int(flip) not in (0, 1):
            return f"flip= must be 0 (+) or 1 (-), got {flip!r}"
        wanted = [k_raw + int(flip) * n_irr]

    k0s = []
    for ext in wanted:
        for st in stars:
            members = [int(m) for m in (st.get("members") or [])]
            if ext in members or int(st["k0"]) == ext:
                k0s.append(int(st["k0"]))
                break
    if not k0s:
        return (f"the decoded irrep k_raw={k_raw} (extended {wanted}) is in no "
                f"star -- the star walk did not cover it")
    return k0s, k_raw


def decode_irrep_for_character(star, character):
    """Little-co-group irrep index for a requested CHARACTER.

    ``star`` is one entry of the plan's ``stars`` list (it carries
    ``little_elems`` / ``little_characters`` / ``little_irrep_dims``).

    ``character`` names the irrep physically, as ``{residue_index: value}`` --
    e.g. ``{0: +1}`` is "the irrep on which residue 0 (say the reflection) acts
    with character +1". Use ``-1`` as the key for the identity. Only the
    elements you name are matched, so a partial spec is fine as long as it
    picks out exactly one irrep.

    Returns ``(irrep_index, dim)`` or a ``str`` decline reason.

    Why by character and not by index: ``LittleGroupLabel::irrep`` is an index
    into decompose_irreps' own per-star ordering. Naming an irrep by that index
    would hand callers an engine-internal convention -- the same trap as
    reading ``k_raw`` as a momentum (on a 12-ring k_raw=8 is q=0) or ``sector=``
    as a raw index. The character IS the physics, so it is what callers name.
    """
    elems = [int(e) for e in (star.get("little_elems") or [])]
    chars = [list(map(complex, row))
             for row in (star.get("little_characters") or [])]
    dims = [int(d) for d in (star.get("little_irrep_dims") or [])]
    if not chars:
        return ("this star was not projected (its little co-group is trivial, "
                "or the decomposition declined) -- it has no irreps to name")

    try:
        want = {int(k): complex(v) for k, v in dict(character).items()}
    except Exception:
        return ("irrep= must be a {residue_index: character} mapping "
                "(-1 keys the identity)")
    if not want:
        return "irrep= named no elements, so it selects nothing"

    cols = {}
    for e_key in want:
        if e_key not in elems:
            return (f"element {e_key} is not in this star's little co-group "
                    f"(its elements are {elems}, -1 = identity)")
        cols[e_key] = elems.index(e_key)

    hits = [s for s, row in enumerate(chars)
            if all(abs(row[cols[e]] - v) < 1e-8 for e, v in want.items())]
    if not hits:
        table = {s: [complex(round(x.real, 6), round(x.imag, 6)) for x in row]
                 for s, row in enumerate(chars)}
        return (f"no irrep of this star has character {want} on elements "
                f"{elems}. The table is {table}")
    if len(hits) > 1:
        return (f"character {want} does not pick out a unique irrep "
                f"(matches {hits}) -- name more elements")
    return hits[0], dims[hits[0]]


def _pg_key(point_group) -> str:
    if point_group in (False, 0, None, "off", "none", ""):
        return "off"
    if point_group is True:
        return "auto"
    if isinstance(point_group, str):
        k = point_group.strip().lower()
        if k in ("auto", "full"):
            return k
        if k in ("off", "none", ""):
            return "off"
    raise ValueError(
        f"point_group must be 'auto' | 'full' | 'off' (or bool/None), "
        f"got {point_group!r}")


def resolve_projection_lane(
    symmetry_or_gens,
    *,
    point_group,
    consumer: str,                 # "solve" | "thermal" | "full_spectrum" | "spectral"
    eigenvalues_only: bool = True,
    prefer_abelian: bool = False,  # soft veto: 'auto' declines, 'full' still projects
    method: Optional[str] = None,  # thermal only: the sampling method name
    verbose: bool = False,
) -> ProjectionLane:
    """Decide project-vs-abelian for one verb call. ``point_group='full'``
    raises on any decline; ``'auto'`` returns the abelian lane with the
    decline reason recorded."""
    pg = _pg_key(point_group)

    def _decline(reason: str) -> ProjectionLane:
        if pg == "full":
            raise RuntimeError(
                f"point_group='full' cannot engage the little-group "
                f"projection lane on this {consumer} call: {reason}. "
                "Use point_group='auto' to degrade to the abelian rep "
                "lane with isospectral folds instead of raising.")
        return ProjectionLane(mode="abelian", reason=reason)

    if pg == "off":
        return ProjectionLane(mode="abelian", reason="point_group off")
    if symmetry_or_gens is None:
        return _decline("no spatial symmetry was resolved")
    if os.environ.get("ED_SYM_LITTLE_GROUP", "1") == "0":
        return _decline("ED_SYM_LITTLE_GROUP=0 disables the lane")
    if not eigenvalues_only:
        return _decline(
            "the call consumes eigenvectors or a sampling method, which "
            "the projection lane does not produce")
    if pg == "auto" and prefer_abelian:
        return ProjectionLane(
            mode="abelian",
            reason="the abelian lane better serves this call "
                   "(e.g. an explicit GPU device request)")
    if pg == "auto" and consumer == "thermal":
        # U1b (lane unification): thermal 'auto' PROJECTS for the sampling
        # methods -- the run stays a sampling run, inside the smaller
        # (n_up, k, +/-, sigma) blocks (little_group_thermal); only
        # point_group='full' opts into EXACT per-block spectra
        # (little_group_thermodynamics), unchanged. KPM_DOS keeps the
        # abelian lane: its deliverable is one full-spectrum DOS, which
        # per-block Chebyshev grids cannot recombine into.
        _m = (method or "").upper()
        if os.environ.get("ED_SYM_LG_THERMAL", "1") == "0":
            return _decline(
                "ED_SYM_LG_THERMAL=0 restores the pre-U1b behaviour "
                "(thermal auto never projects)")
        if _m in ("KPM_DOS", "KPMDOS"):
            return _decline(
                "KPM_DOS produces one full-spectrum DOS; per-block "
                "sub-DOS cannot recombine -- the abelian lane serves it")
        if _m not in ("FTLM", "LTLM", "MTPQ", "OFTLM"):
            return _decline(
                f"thermal method {method!r} has no sample-inside-block "
                f"lane; only FTLM/LTLM/mTPQ/OFTLM project under 'auto'")
    if pg == "auto" and consumer == "spectral":
        return _decline(
            "spectral 'auto' keeps the abelian per-probe cross-irrep "
            "lanes (momentum-resolved panels); point_group='full' opts "
            "into the factorized little-group GS-DSSF")
    split = split_nonabelian(symmetry_or_gens)
    if isinstance(split, str):
        return _decline(split)
    A, residues = split
    return ProjectionLane(mode="project", A=A, residues=residues)
