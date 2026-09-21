"""``qed._thermal.lanes``: the four early-return lanes of :func:`qed.thermal`.

``thermal()`` is a sequence of lanes, each of which either RETURNS a
``ThermalResult`` or declines and lets the next one look. Four of them are
self-contained early returns -- the SU(2) tower lane, the exact block lane,
and the two little-group projection lanes (in-memory and directory) -- and
they are the bodies extracted here. What stays in ``thermal()`` is the
preparation they read: Sz normalisation, symmetry resolution, the device
pick, the method-knob merge and the operator load, each of which mutates
state the LATER lanes depend on. That ordering is the contract; the lane
functions take everything they read as explicit arguments so it cannot drift
into implicit shared state again.

A declining lane returns the shared ``DECLINED`` sentinel (the same object
``qed._solve`` uses), never ``None`` -- a lane may legitimately produce a
falsy result, and a sentinel that is also the "no result" value is how a
declined lane starts looking like a successful one.

The bodies below are the pre-split text, unchanged except for the relative
import depth (this module sits one package deeper). Two inconsistencies were
found in them during the move and deliberately NOT fixed here, so the split
stays reviewable as a pure move: the in-memory projection lane recomputes
``use_gpu`` from ``device`` while the directory lane uses the resolved
``_use_gpu``, and the two lanes word their decline messages differently.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import Any, Optional, Sequence, Union

import numpy as np

from .. import _core as _core
from .._core import Operator
from .._solve.request import DECLINED

__all__ = [
    "DECLINED",
    "su2_tower_lane",
    "exact_lane",
    "inmemory_projection_lane",
    "directory_projection_lane",
]


def su2_tower_lane(
    H, *, total_spin, method, T_min, T_max, num_T, num_samples,
    krylov_dim, ftlm_krylov_dim, random_seed, tpq_delta_beta,
    tpq_taylor_order, is_directory, verbose,
    symmetry, sector, sz, sz_min, sz_max, star_maps, output_dir,
    probe_betas, spin_flip, time_reversal, point_group, device,
):
    """Per-spin-tower thermodynamics, Z = sum_S (2S+1) Z_S. Declines
    unless ``total_spin`` names a tower."""
    from ..thermal import _thermal_su2_towers

    # ------------------------------------------------------------------
    # Stage 12f (SU(2) rollout): per-spin-tower thermodynamics.
    # Z = sum_S (2S+1) Z_S with each tower sampled once in its
    # highest-weight sector (small blocks: exact highest-weight
    # spectral differencing). Opt-in; see the total_spin kwarg doc.
    # ------------------------------------------------------------------
    if not (total_spin is None or total_spin is False
            or (isinstance(total_spin, str)
                and total_spin.lower() == "off")):
        # Audit fix (2026-07-30): the tower lane used to silently drop
        # every kwarg its helper does not thread -- including symmetry
        # composition and Sz windows the caller may believe are active.
        # Refuse the named ones loudly; warn on device='gpu' (the Lowdin
        # targeting is host-only for now -- the documented Stage-12h
        # follow-up -- and the fallback is graceful but should not be
        # silent).
        _unsupported = [name for name, val in (
            ("symmetry", symmetry), ("sector", sector),
            ("sz", sz), ("sz_min", sz_min), ("sz_max", sz_max),
            ("star_maps", star_maps),
            ("output_dir", output_dir or None),
            ("probe_betas", probe_betas),
        ) if val is not None]
        for name, val in (("spin_flip", spin_flip),
                          ("time_reversal", time_reversal)):
            if isinstance(val, str) and val.lower() == "require":
                _unsupported.append(name + "='require'")
        if isinstance(point_group, str) and point_group.lower() == "full":
            _unsupported.append("point_group='full'")
        if _unsupported:
            raise NotImplementedError(
                f"qed.thermal(total_spin=...): {_unsupported} are not "
                f"threaded through the SU(2) tower lane (each spin-S "
                f"tower runs in its highest-weight sector with its own "
                f"Lowdin projection). Drop them, or drop total_spin= to "
                f"use the composed symmetry lanes.")
        if isinstance(device, str) and device.lower() == "gpu":
            warnings.warn(
                "qed.thermal(total_spin=...): the Lowdin tower targeting "
                "is host-only (Stage-12h follow-up); device='gpu' runs "
                "this lane on the CPU.", RuntimeWarning, stacklevel=2)
        return _thermal_su2_towers(
            H, total_spin=total_spin, method=method,
            T_min=T_min, T_max=T_max, num_T=num_T,
            num_samples=num_samples,
            krylov_dim=(krylov_dim or ftlm_krylov_dim or 100),
            krylov_dim_explicit=bool(krylov_dim or ftlm_krylov_dim),
            random_seed=random_seed,
            tpq_delta_beta=tpq_delta_beta,
            tpq_taylor_order=tpq_taylor_order,
            is_directory=is_directory, verbose=verbose)


    return DECLINED


def exact_lane(
    H, *, method, is_directory, num_sites, spin, symmetry, sector,
    T_min, T_max, num_T, device, spin_flip, time_reversal, verbose,
):
    """``method="exact"``: exact canonical thermodynamics from the block
    engine's full per-block spectra. Declines for every other method.

    Note the local rebinding of ``H`` / ``symmetry`` / ``is_directory`` for
    the directory form: it is confined to this lane because the lane always
    returns, which is exactly why it can stay verbatim here."""
    from ..thermal import ThermalResult, _sym_toggle_int
    # method="exact": exact canonical thermodynamics from the block
    # engine's full per-block spectra. This is a METHOD, sitting beside
    # FTLM/LTLM/mTPQ -- point_group stays a pure symmetry-routing knob
    # ('auto' project-when-possible / 'full' require / 'off' abelian).
    # Historically this computation was reachable only through the
    # point_group='full' spelling, which conflated routing with solver
    # strategy; that spelling now warns (see below) and requires
    # projection without changing the method.
    # ------------------------------------------------------------------
    if isinstance(method, str) and method.upper() == "EXACT":
        if is_directory:
            # U4a pattern: the directory's own deck + automorphisms feed
            # the exact block engine. Same guards as the sampling route:
            # the Python loader reads only Trans/InterAll (refuse
            # ThreeBodyG.dat), and the group comes from
            # automorphisms.json (validated permutations).
            directory = str(H)
            if num_sites is None:
                raise ValueError(
                    "qed.thermal: pass num_sites= with the directory "
                    "form.")
            if os.path.exists(os.path.join(directory, "ThreeBodyG.dat")):
                raise NotImplementedError(
                    "qed.thermal(method='exact'): this directory carries "
                    "ThreeBodyG.dat, which the Python-side loader does "
                    "not read -- the exact lane would silently miss "
                    "terms. Use the sampling methods (exact below the "
                    "small-dim cutoff) or the in-memory form.")
            _autos_path = os.path.join(directory, "automorphism_results",
                                       "automorphisms.json")
            if not os.path.exists(_autos_path):
                raise ValueError(
                    "qed.thermal(method='exact'): the directory carries "
                    "no automorphism_results/automorphisms.json to build "
                    "the block engine from; pass the in-memory form with "
                    "symmetry= instead.")
            import json as _json
            with open(_autos_path) as f:
                _cand = _json.load(f)
            _N = int(num_sites)
            if not (isinstance(_cand, list) and _cand
                    and all(isinstance(p, list) and len(p) == _N
                            and sorted(p) == list(range(_N))
                            for p in _cand)):
                raise ValueError(
                    "qed.thermal(method='exact'): automorphisms.json is "
                    "not a list of site permutations.")
            H_ex = Operator(num_sites=_N, spin=float(spin))
            _trans = os.path.join(directory, "Trans.dat")
            _inter = os.path.join(directory, "InterAll.dat")
            if os.path.exists(_trans):
                H_ex.load_trans(_trans)
            if os.path.exists(_inter):
                H_ex.load_inter_all(_inter)
            H, symmetry, is_directory = H_ex, _cand, False
        if symmetry is None:
            raise NotImplementedError(
                "qed.thermal: method='exact' rides the little-group block "
                "engine and needs symmetry=. (Without symmetry, the "
                "sampling methods are already exact below dim 512 via the "
                "small-dim fallback.)")
        if sector is not None:
            raise NotImplementedError(
                "qed.thermal: sector= is not honoured on the exact lane "
                "yet -- refusing to silently ignore it.")
        from ..point_group_routing import split_nonabelian, _close
        _split = split_nonabelian(symmetry)
        if isinstance(_split, str):
            # No non-abelian residue (or unsplittable): exact per plain
            # (n_up, k) block -- close the abelian generators, no
            # residues. Correct, merely less reduced.
            _gens = [list(g) for g in symmetry.generators] \
                if hasattr(symmetry, "generators") else \
                [list(g) for g in symmetry]
            _A = _close(_gens)
            if _A is None:
                raise ValueError(
                    f"qed.thermal(method='exact'): could not close the "
                    f"symmetry group ({_split})")
            _A, _res = [list(g) for g in _A], []
        else:
            _A, _res = _split
        temps = list(np.linspace(T_min, T_max, num_T))
        td = dict(_core.little_group_thermodynamics(
            H, _A, _res, temps, n_up=-1,
            use_gpu=(isinstance(device, str)
                     and device.lower() in ("gpu", "cuda")),
            spin_flip=_sym_toggle_int(spin_flip, "spin_flip"),
            time_reversal=_sym_toggle_int(time_reversal,
                                          "time_reversal")))
        if verbose:
            print(f"[qed.thermal] EXACT little-group lane: |A| = "
                  f"{len(_A)}, residues = {len(_res)}.")
        _E = np.asarray(td["energy"], dtype=float)
        return ThermalResult(
            temperatures=np.asarray(td["temperatures"], dtype=float),
            energy=_E,
            specific_heat=np.asarray(td["specific_heat"], dtype=float),
            entropy=np.asarray(td["entropy"], dtype=float),
            free_energy=np.asarray(td["free_energy"], dtype=float),
            method="exact",
            ground_state_energy=float(_E[0]) if len(_E) else 0.0,
            used_sz_decomposition=False,
            used_symmetry_decomposition=True,
        )

    return DECLINED


def inmemory_projection_lane(
    H, *, symmetry, is_directory, sector, point_group, method,
    T_min, T_max, num_T, num_samples, krylov_dim, random_seed,
    device, spin_flip, time_reversal, verbose,
):
    """U1b: an in-memory operator with a spatial group runs the SAMPLING
    method inside the little-group blocks. Declines when the preconditions
    do not hold or when the lane resolver declines the projection."""
    from ..thermal import _sym_toggle_int, _thermal_result_from_block_lane
    if (symmetry is not None and not is_directory
            and sector is None
            and isinstance(point_group, str)
            and point_group.lower() in ("auto", "full")):
        # U1b (lane unification): thermal 'auto' + a sampling method now
        # PROJECTS -- the run stays a sampling run, executed inside the
        # (n_up, k, +/-, sigma) little-group blocks via
        # _core.little_group_thermal (F-shift Z-recombination). The lane
        # resolver declines KPM_DOS (full-spectrum DOS) and honours
        # ED_SYM_LG_THERMAL=0; any decline falls through to the abelian
        # sector lane below unchanged. sector= keeps the abelian
        # filtering lane (the block engine has only_k0/only_irrep but the
        # QN decode for thermal is future work -- refusing to guess).
        from ..point_group_routing import resolve_projection_lane
        lane = resolve_projection_lane(
            symmetry, point_group=point_group.lower(), consumer="thermal",
            eigenvalues_only=True, method=str(method),
            verbose=verbose)
        if lane.mode == "project":
            out = dict(_core.little_group_thermal(
                H, lane.A, lane.residues, method=str(method),
                t_min=float(T_min), t_max=float(T_max), num_t=int(num_T),
                num_samples=int(num_samples),
                krylov_dim=int(krylov_dim) if krylov_dim else 100,
                random_seed=int(random_seed) if random_seed else 0,
                use_gpu=(isinstance(device, str)
                         and device.lower() in ("gpu", "cuda")),
                spin_flip=_sym_toggle_int(spin_flip, "spin_flip"),
                time_reversal=_sym_toggle_int(time_reversal,
                                              "time_reversal")))
            if verbose:
                print(f"[qed.thermal] little-group SAMPLING lane "
                      f"(U1b): {len(out['block_dim'])} blocks, "
                      f"projected_any={out['projected_any']}, "
                      f"max block dim={max(out['block_dim'])}.")
            return _thermal_result_from_block_lane(out, method)
        elif verbose:
            print(f"[qed.thermal] projection declined ({lane.reason}); "
                  f"abelian sector lane.")

    return DECLINED


def directory_projection_lane(
    H_op, *, is_directory, has_sym, sector, point_group, output_dir,
    probe_betas, method, directory, sym_dir, sz_conserved, lo, hi, N,
    T_min, T_max, num_T, num_samples, krylov_dim, random_seed,
    use_gpu, spin_flip, time_reversal, verbose,
):
    """U4a: the directory form rides the same block lane as the in-memory
    one, guarded by every contract the block lane cannot honour (three-body
    terms the Python loader does not read, per-sector files, TPQ snapshots,
    ``sector=``, a partial Sz window). Declines otherwise."""
    from ..thermal import _sym_toggle_int, _thermal_result_from_block_lane
    _use_gpu = use_gpu
    if (is_directory and has_sym and sector is None
            and isinstance(point_group, str)
            and point_group.lower() in ("auto", "full")
            and not output_dir and not probe_betas
            and str(method).upper() in ("FTLM", "LTLM", "MTPQ", "OFTLM")
            and not os.path.exists(os.path.join(directory, "ThreeBodyG.dat"))
            and (not sz_conserved or (lo == 0 and hi == N))):
        _autos_path = os.path.join(sym_dir, "automorphisms.json")
        _autos = None
        if os.path.exists(_autos_path):
            import json as _json
            try:
                with open(_autos_path) as f:
                    _cand = _json.load(f)
                if (isinstance(_cand, list) and _cand
                        and all(isinstance(p, list) and len(p) == N
                                and sorted(p) == list(range(N))
                                for p in _cand)):
                    _autos = _cand
            except Exception:
                _autos = None
        if _autos is not None:
            from ..point_group_routing import resolve_projection_lane
            lane = resolve_projection_lane(
                _autos, point_group=point_group.lower(), consumer="thermal",
                eigenvalues_only=True, method=str(method), verbose=verbose)
            if lane.mode == "project":
                out = dict(_core.little_group_thermal(
                    H_op, lane.A, lane.residues, method=str(method),
                    t_min=float(T_min), t_max=float(T_max),
                    num_t=int(num_T), num_samples=int(num_samples),
                    krylov_dim=int(krylov_dim) if krylov_dim else 100,
                    random_seed=int(random_seed) if random_seed else 0,
                    use_gpu=_use_gpu,
                    spin_flip=_sym_toggle_int(spin_flip, "spin_flip"),
                    time_reversal=_sym_toggle_int(time_reversal,
                                                  "time_reversal")))
                if verbose:
                    print(f"[qed.thermal] directory -> little-group "
                          f"SAMPLING lane (U4a): "
                          f"{len(out['block_dim'])} blocks, "
                          f"projected_any={out['projected_any']}.")
                return _thermal_result_from_block_lane(out, method)
            elif verbose:
                print(f"[qed.thermal] directory projection declined "
                      f"({lane.reason}); flat-pool sector lane.")

    return DECLINED
