"""Graph automorphisms of a spin Hamiltonian (pynauty), behind ``qed.find_symmetries``.

The terms become a coloured graph: vertex colours from the one-body terms, a subdivision
vertex per interacting pair whose colour encodes the bond's couplings, and one per triple of
sites carrying three-body terms. Its automorphisms that leave H invariant are the site
symmetries; ``qed._groups`` splits them for the sector engine.
"""

import itertools
from collections import defaultdict
from typing import Optional

from pynauty import Graph, autgrp

from . import _log


def _round_tuple(t, decimals=8):
    """Round all floats in a tuple to given decimals."""
    return tuple(round(x, decimals) for x in t)


def _merged(terms):
    """``(key, coeff)`` pairs -> ``{key: summed coeff}``, zeros dropped. How H spells a coupling
    (a coefficient split over repeated terms, an explicit zero) must not change its colour."""
    acc = defaultdict(complex)
    for key, c in terms:
        acc[key] += complex(c)
    return {k: c for k, c in acc.items() if abs(c) > 1e-12}


def bond_signatures(edges):
    """``{(lo, hi): signature}`` for every pair of distinct sites with two-body terms: the terms of
    the pair merged by operator pair, written from the end that gives the lesser tuple. Site labels
    order the pair and a symmetry need not preserve that order, so the colour of a bond does not
    depend on its orientation; the exact term check after the search removes the elements that
    reverse an antisymmetric coupling (a DM vector)."""
    by_pair = defaultdict(list)
    for e in edges:
        v1, v2 = e['vertex1'], e['vertex2']
        if v1 == v2:
            continue
        ops = (e['type1'], e['type2']) if v1 < v2 else (e['type2'], e['type1'])
        by_pair[(min(v1, v2), max(v1, v2))].append((ops, complex(*e['weight'])))
    out = {}
    for pair, terms in by_pair.items():
        m = _merged(terms)
        if m:
            fwd = tuple(sorted((a, b, _round_tuple((c.real, c.imag))) for (a, b), c in m.items()))
            bwd = tuple(sorted((b, a, _round_tuple((c.real, c.imag))) for (a, b), c in m.items()))
            out[pair] = min(fwd, bwd)
    return out


def compute_vertex_colors(vertex_weights, bonds, wl_iterations=10):
    """Weisfeiler-Lehman colours of the sites: each starts from its one-body signature
    (``vertex_weights``: site -> tuple of merged ``(op, re, im)`` terms) and is refined by the
    multiset of (bond signature, neighbour colour) over ``bonds`` (:func:`bond_signatures`).
    Returns ``{site: colour}``."""
    vertex_ids = set(vertex_weights)
    for lo, hi in bonds:
        vertex_ids.update((lo, hi))
    vertex_ids = sorted(vertex_ids)
    adj = {v: [] for v in vertex_ids}
    for (lo, hi), sig in bonds.items():
        adj[lo].append((hi, sig))
        adj[hi].append((lo, sig))
    token = {v: ('v', vertex_weights.get(v, ())) for v in vertex_ids}
    colors = {}
    for _ in range(wl_iterations):
        ids = {tok: i for i, tok in enumerate(sorted(set(token.values())))}
        colors = {v: ids[token[v]] for v in vertex_ids}
        nxt = {v: (colors[v], tuple(sorted((sig, colors[u]) for u, sig in adj[v]))) for v in vertex_ids}
        if all(token[v] == nxt[v] for v in vertex_ids):
            break
        token = nxt
    final = {c: i for i, c in enumerate(sorted(set(colors.values())))}
    return {v: final[colors[v]] for v in vertex_ids}


def _triple_signature(triple, terms):
    """A label of the three-body terms on a site triple that does not depend on how the triple
    is ordered: the least, over its six orderings, of the terms written by position."""
    best = None
    for order in itertools.permutations(triple):
        pos = {s: k for k, s in enumerate(order)}
        sig = tuple(sorted(tuple(sorted((pos[s], o) for s, o in zip(sites, ops))) + (w,) for sites, ops, w in terms))
        if best is None or sig < best:
            best = sig
    return best


def construct_colored_graph(vertex_weights, edges, triples=()):
    """Construct a colored undirected graph for pynauty with edge-type subdivision.

    Uses the subdivision trick for edge-colored graphs: for each pair (i,j) of
    interacting sites, an auxiliary vertex is inserted whose color encodes the
    bond type (the full set of coupling terms on that bond). This ensures that
    nauty's automorphisms can only map bonds of the same type to each other.
    ``triples`` holds the three-body terms as ``((s1, s2, s3), (op1, op2, op3), coeff)``:
    each triple of distinct sites gets an auxiliary vertex joined to its three sites and
    coloured by :func:`_triple_signature`. That colour ignores the orientation of the
    triple (a scalar chirality and its reverse look alike), so the group found may still
    hold elements that reverse it; the exact term check that follows removes them.

    Returns:
        tuple: (graph, vertex_colors, idx_to_vid, vid_to_idx)
            - graph: pynauty Graph object
            - vertex_colors: dict mapping original vertex_id -> color
            - idx_to_vid: list mapping nauty index -> original vertex_id
            - vid_to_idx: dict mapping original vertex_id -> nauty index
    """
    # Two-body terms merged per pair, then WL-refined colours of the sites
    bonds = bond_signatures(edges)
    vertex_colors = compute_vertex_colors(vertex_weights, bonds)

    # Build stable vertex index mapping for original vertices (0..n-1)
    all_vertices = sorted(vertex_colors.keys())
    vid_to_idx = {v: i for i, v in enumerate(all_vertices)}
    idx_to_vid = list(all_vertices)
    n_original = len(all_vertices)

    # One auxiliary vertex per interacting pair, coloured by its signature
    bond_pairs = sorted(bonds)
    unique_sigs = sorted(set(bonds.values()))
    sig_to_color = {sig: i for i, sig in enumerate(unique_sigs)}

    # Three-body terms, merged per triple of distinct sites (each term as its site -> operator
    # map), one auxiliary vertex per triple; a term that repeats a site constrains nothing here,
    # and the exact check after the search still sees it.
    by_triple = defaultdict(list)
    for sites, ops, coeff in triples:
        if len(set(sites)) == 3 and all(s in vid_to_idx for s in sites):
            by_triple[tuple(sorted(sites))].append((tuple(sorted(zip(sites, ops))), complex(coeff)))
    triple_terms = {}
    for t, terms in by_triple.items():
        m = _merged(terms)
        if m:
            triple_terms[t] = [
                (tuple(s for s, _ in key), tuple(o for _, o in key), _round_tuple((c.real, c.imag)))
                for key, c in sorted(m.items())
            ]
    triple_list = sorted(triple_terms)
    triple_signatures = {t: _triple_signature(t, triple_terms[t]) for t in triple_list}
    unique_triple_sigs = sorted(set(triple_signatures.values()))
    triple_to_color = {sig: len(unique_sigs) + i for i, sig in enumerate(unique_triple_sigs)}

    n_bonds = len(bond_pairs)
    n_total = n_original + n_bonds + len(triple_list)  # original vertices + auxiliary vertices

    _log.log(
        _log.DEBUG,
        "edge-coloured graph: %d vertices + %d auxiliary bond vertices (%d types) " "+ %d triple vertices (%d types)",
        n_original,
        n_bonds,
        len(unique_sigs),
        len(triple_list),
        len(unique_triple_sigs),
    )

    # Build adjacency for expanded graph
    adjacency_dict = {i: [] for i in range(n_total)}

    for bond_idx, (lo, hi) in enumerate(bond_pairs):
        aux_idx = n_original + bond_idx  # index of auxiliary vertex
        i, j = vid_to_idx[lo], vid_to_idx[hi]
        # Connect both endpoints to the auxiliary vertex
        adjacency_dict[i].append(aux_idx)
        adjacency_dict[aux_idx].append(i)
        adjacency_dict[j].append(aux_idx)
        adjacency_dict[aux_idx].append(j)
    for t_idx, t in enumerate(triple_list):
        aux_idx = n_original + n_bonds + t_idx
        for s in t:
            adjacency_dict[vid_to_idx[s]].append(aux_idx)
            adjacency_dict[aux_idx].append(vid_to_idx[s])

    # Build vertex coloring
    # Original vertices: color from WL
    # Auxiliary vertices: color based on bond signature (offset by max vertex color + 1)
    max_vertex_color = max(vertex_colors.values()) + 1 if vertex_colors else 0

    color_to_vertices = defaultdict(list)
    for v, c in vertex_colors.items():
        color_to_vertices[c].append(vid_to_idx[v])

    for bond_idx, pair in enumerate(bond_pairs):
        aux_idx = n_original + bond_idx
        bond_color = max_vertex_color + sig_to_color[bonds[pair]]
        color_to_vertices[bond_color].append(aux_idx)
    for t_idx, t in enumerate(triple_list):
        color_to_vertices[max_vertex_color + triple_to_color[triple_signatures[t]]].append(n_original + n_bonds + t_idx)

    coloring = [set(sorted(ids)) for _, ids in sorted(color_to_vertices.items(), key=lambda kv: kv[0])]

    # Create pynauty graph
    g = Graph(n_total, directed=False, adjacency_dict=adjacency_dict, vertex_coloring=coloring)
    return g, idx_to_vid, vid_to_idx


def automorphisms(vertex_weights, edges, triples, cap: int) -> tuple[Optional[list[list[int]]], float]:
    """Run nauty: ``(permutations, |Aut|)``, the automorphisms of the coloured interaction graph
    on the original vertices and nauty's group order (a supergroup of H's symmetries: the exact
    term check of :func:`qed.discovery.find_symmetries` follows). Above ``cap`` nothing is
    enumerated and the list is None (|Aut| reaches N! for field-only, empty or all-to-all H)."""
    from ._perm import close_group

    graph, idx_to_vid, vid_to_idx = construct_colored_graph(vertex_weights, edges, triples)
    aut = autgrp(graph)
    # One auxiliary vertex per interacting pair, so the group of the expanded graph is that
    # of the original vertices: nauty's count (grpsize1 * 10^grpsize2) is |Aut| itself.
    size = float(aut[1]) * 10.0 ** int(aut[2])
    if size > cap:
        return None, size
    gens = [tuple(int(x) for x in g) for g in aut[0]]
    expanded = close_group(gens, cap=cap) if gens else [tuple(range(graph.number_of_vertices))]
    # Back to permutations of the original vertices, each once.
    seen: set[tuple[int, ...]] = set()
    autos: list[list[int]] = []
    for perm in expanded:
        proj = [idx_to_vid[perm[vid_to_idx[vid]]] for vid in idx_to_vid]
        key = tuple(proj)
        if key not in seen:
            seen.add(key)
            autos.append(proj)
    return autos, size
