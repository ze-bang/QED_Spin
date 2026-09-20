"""``qed._solve.directories``: the on-disk operator / symmetry fixtures.

Carved out of ``workflow.py`` (WP11). Writes the legacy mVMC-convention
``Trans.dat`` / ``InterAll.dat`` / ``ThreeBodyG.dat`` term files and the
four ``automorphism_results/`` JSON files, i.e. everything the
directory-form C++ loaders need to see an in-memory model on disk.
"""

from __future__ import annotations

import json
import os
from typing import Any

from .._core import Operator  # type: ignore[attr-defined]
from .symmetry_input import _closed_symmetry_info


def _write_operator_directory(operator: Operator, directory: str) -> None:
    """Dump the operator's terms to ``Trans.dat`` / ``InterAll.dat`` /
    ``ThreeBodyG.dat`` in ``directory`` using the legacy mVMC header
    convention the C++ loader expects."""
    one_body = list(operator.iter_one_body_terms())
    two_body = list(operator.iter_two_body_terms())
    three_body = list(operator.iter_three_body_terms())

    _write_dat_file(
        os.path.join(directory, "Trans.dat"),
        rows=[
            (int(op_type), int(site), complex(coeff))
            for op_type, site, coeff in one_body
        ],
        formatter=_format_one_body_row,
    )
    _write_dat_file(
        os.path.join(directory, "InterAll.dat"),
        rows=[
            (int(op1), int(s1), int(op2), int(s2), complex(coeff))
            for op1, s1, op2, s2, coeff in two_body
        ],
        formatter=_format_two_body_row,
    )
    if three_body:
        _write_dat_file(
            os.path.join(directory, "ThreeBodyG.dat"),
            rows=[
                (int(op1), int(s1), int(op2), int(s2), int(op3), int(s3),
                 complex(coeff))
                for op1, s1, op2, s2, op3, s3, coeff in three_body
            ],
            formatter=_format_three_body_row,
        )


def _write_dat_file(path: str, rows: list[Any], formatter) -> None:
    """Write the standard 5-line header followed by formatted rows."""
    with open(path, "w") as f:
        f.write("===================\n")
        f.write(f"num {len(rows):>8d}\n")
        f.write("===================\n")
        f.write("===================\n")
        f.write("===================\n")
        for row in rows:
            f.write(formatter(row))


def _format_one_body_row(row) -> str:
    op_type, site, coeff = row
    return f" {op_type:>8d}  {site:>8d}    {coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"


def _format_two_body_row(row) -> str:
    op1, s1, op2, s2, coeff = row
    return (
        f" {op1:>8d}  {s1:>8d}    {op2:>8d}    {s2:>8d}    "
        f"{coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"
    )


def _format_three_body_row(row) -> str:
    op1, s1, op2, s2, op3, s3, coeff = row
    return (
        f" {op1:>8d}  {s1:>8d}    {op2:>8d}    {s2:>8d}    "
        f"{op3:>8d}    {s3:>8d}    "
        f"{coeff.real:>15.8e}    {coeff.imag:>15.8e}\n"
    )


def _write_symmetry_directory(directory: str, info: dict[str, Any]) -> None:
    """Write the four JSON files the C++ streaming-symmetry kernel needs.

    The C++ ``generate_automorphisms`` helper in
    ``ed/core/system_utils.h`` only re-runs the Python finder when
    ``automorphisms.json`` is missing, so we write all four files. That
    way the user-supplied ``GeneratorSet`` is honored verbatim instead
    of being silently overwritten by the full automorphism search.

    Files written into ``<directory>/automorphism_results/``:

    * ``automorphisms.json``      — flat array of permutations (gates the
      C++ regeneration check)
    * ``max_clique.json``         — flat array of permutations (the
      commuting group chosen by the user)
    * ``minimal_generators.json`` — ``{"generators":
      [{"permutation":..., "order":...}, ...]}``
    * ``sector_metadata.json``    — ``{"sectors": [{"sector_id":...,
      "quantum_numbers":..., "phase_factors": [{"real":..., "imag":...},
      ...]}, ...]}``
    """
    out_dir = os.path.join(directory, "automorphism_results")
    os.makedirs(out_dir, exist_ok=True)

    info = _closed_symmetry_info(info)
    max_clique = info.get("max_clique", [])
    generators = info.get("generators", [])
    generator_orders = info.get("generator_orders", [])
    sectors = info.get("sectors", [])

    max_clique_int = [list(map(int, p)) for p in max_clique]

    # automorphisms.json: full nauty-style list. We give the C++ side
    # exactly the user's group; the downstream max-clique finder treats
    # it as already-commuting so the same permutations come back out.
    # Stale marker from a previous translation_only run would force a
    # regeneration -- delete it pre-emptively.
    marker_file = os.path.join(out_dir, ".translation_only")
    try:
        os.remove(marker_file)
    except FileNotFoundError:
        pass
    with open(os.path.join(out_dir, "automorphisms.json"), "w") as f:
        json.dump(max_clique_int, f, indent=2)

    with open(os.path.join(out_dir, "max_clique.json"), "w") as f:
        json.dump(max_clique_int, f, indent=2)

    gen_records = []
    for gen, order in zip(generators, generator_orders):
        gen_records.append({
            "permutation": list(map(int, gen)),
            "order": int(order),
        })
    with open(os.path.join(out_dir, "minimal_generators.json"), "w") as f:
        json.dump({"generators": gen_records}, f, indent=2)

    # ------------------------------------------------------------------
    # phase_factors: the on-disk schema is one entry PER GENERATOR (length
    # = num_generators), each = exp(2πi q_k / o_k). The C++ kernel then
    # composes the per-element character via
    #     χ_q(g) = ∏_k phase_factors[k]^{powers[k]}
    # where ``powers[k]`` comes from `power_representation[g]`.
    #
    # NOTE: this differs from the convention `group_from_generators`
    # uses in-memory (one full character per group element, length =
    # |max_clique|). We always recompute the per-generator form here so
    # the JSON path stays consistent regardless of how `info` was
    # produced.
    # ------------------------------------------------------------------
    import math as _math

    sector_records = []
    for s in sectors:
        sid = int(s.get("sector_id", 0))
        qn = list(map(int, s.get("quantum_numbers", [])))
        pf = []
        for k, q_k in enumerate(qn):
            o_k = int(generator_orders[k]) if k < len(generator_orders) else 1
            angle = 2.0 * _math.pi * float(q_k) / float(o_k) if o_k else 0.0
            pf.append({
                "real": float(_math.cos(angle)),
                "imag": float(_math.sin(angle)),
            })
        sector_records.append({
            "sector_id": sid,
            "quantum_numbers": qn,
            "phase_factors": pf,
        })
    with open(os.path.join(out_dir, "sector_metadata.json"), "w") as f:
        json.dump({"sectors": sector_records}, f, indent=2)
