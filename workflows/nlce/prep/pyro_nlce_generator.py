
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, FrozenSet
import itertools
from collections import defaultdict, deque
from pathlib import Path

import networkx as nx
from networkx.algorithms.graph_hashing import weisfeiler_lehman_graph_hash as wl_hash


@dataclass(frozen=True)
class DiamondNode:
    i: int
    j: int
    k: int
    s: int  # 0 for A, 1 for B

    def key(self, L: int) -> Tuple[int, int, int, int]:
        return (self.i % L, self.j % L, self.k % L, self.s)


@dataclass
class LatticeGraphs:
    diamond: nx.Graph
    pyrochlore: nx.Graph
    diamond_edge_to_pyro: Dict[Tuple[DiamondNode, DiamondNode], int]
    pyro_to_diamond_edge: Dict[int, Tuple[DiamondNode, DiamondNode]]
    diamond_node_to_pyro_nodes: Dict[DiamondNode, List[int]]


def build_diamond_supercell(L: int) -> nx.Graph:
    G = nx.Graph()
    nodes_A = [DiamondNode(i, j, k, 0) for i in range(L) for j in range(L) for k in range(L)]
    nodes_B = [DiamondNode(i, j, k, 1) for i in range(L) for j in range(L) for k in range(L)]
    for n in nodes_A + nodes_B:
        G.add_node(n)

    def wrap(a):
        return a % L

    for i in range(L):
        for j in range(L):
            for k in range(L):
                a = DiamondNode(i, j, k, 0)
                neighbors = [
                    DiamondNode(i, j, k, 1),
                    DiamondNode(i-1, j, k, 1),
                    DiamondNode(i, j-1, k, 1),
                    DiamondNode(i, j, k-1, 1),
                ]
                for b in neighbors:
                    b_wrapped = DiamondNode(wrap(b.i), wrap(b.j), wrap(b.k), 1)
                    G.add_edge(a, b_wrapped)
    return G


def diamond_node_position(node: DiamondNode) -> Tuple[float, float, float]:
    if node.s == 0:
        return (float(node.i), float(node.j), float(node.k))
    else:
        return (node.i + 0.5, node.j + 0.5, node.k + 0.5)


def midpoint(p: Tuple[float, float, float], q: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return ((p[0] + q[0]) / 2.0, (p[1] + q[1]) / 2.0, (p[2] + q[2]) / 2.0)


def build_pyrochlore_from_diamond(Gd: nx.Graph) -> LatticeGraphs:
    def edge_key(u: DiamondNode, v: DiamondNode):
        a = (u.i, u.j, u.k, u.s)
        b = (v.i, v.j, v.k, v.s)
        return (u, v) if a < b else (v, u)

    diamond_edge_to_pyro: Dict[Tuple[DiamondNode, DiamondNode], int] = {}
    pyro_to_diamond_edge: Dict[int, Tuple[DiamondNode, DiamondNode]] = {}

    pyro_graph = nx.Graph()
    next_pyro_id = 0

    for u, v in Gd.edges():
        e = edge_key(u, v)
        if e not in diamond_edge_to_pyro:
            pid = next_pyro_id
            next_pyro_id += 1
            diamond_edge_to_pyro[e] = pid
            pyro_to_diamond_edge[pid] = e
            pyro_graph.add_node(pid)

    diamond_node_to_pyro_nodes: Dict[DiamondNode, List[int]] = defaultdict(list)
    for dn in Gd.nodes():
        incident_pyro = []
        for nbr in Gd.neighbors(dn):
            e = edge_key(dn, nbr)
            pid = diamond_edge_to_pyro[e]
            incident_pyro.append(pid)
            diamond_node_to_pyro_nodes[dn].append(pid)
        for a, b in itertools.combinations(incident_pyro, 2):
            pyro_graph.add_edge(a, b)

    return LatticeGraphs(
        diamond=Gd,
        pyrochlore=pyro_graph,
        diamond_edge_to_pyro=diamond_edge_to_pyro,
        pyro_to_diamond_edge=pyro_to_diamond_edge,
        diamond_node_to_pyro_nodes=diamond_node_to_pyro_nodes,
    )


@dataclass
class TopoCluster:
    order: int
    diamond_nodes: Set[DiamondNode]
    diamond_subgraph: nx.Graph
    wl_cert: str
    pyro_vertices: Set[int] = field(default_factory=set)
    pyro_edges: Set[Tuple[int, int]] = field(default_factory=set)
    tetrahedra: List[Tuple[int, int, int, int]] = field(default_factory=list)
    multiplicity_per_site: float = 0.0
    subclusters: Dict[int, List[Tuple[int, int]]] = field(default_factory=dict)


def enumerate_topo_clusters_diamond(Gd: nx.Graph, L: int, max_order: int) -> Dict[int, List[TopoCluster]]:
    unique_by_order: Dict[int, Dict[str, TopoCluster]] = {m: {} for m in range(1, max_order+1)}

    nodes_list = list(Gd.nodes())
    nodes_list.sort(key=lambda d: (d.s, d.i, d.j, d.k))

    nbrs = {dn: set(Gd.neighbors(dn)) for dn in Gd.nodes()}

    def canonicalize(subset: Set[DiamondNode]) -> Tuple[str, nx.Graph]:
        H = nx.Graph()
        mapping = {dn: idx for idx, dn in enumerate(sorted(subset, key=lambda d: (d.s, d.i, d.j, d.k)))}
        for dn in subset:
            H.add_node(mapping[dn])
        for dn in subset:
            for nb in nbrs[dn]:
                if nb in subset:
                    H.add_edge(mapping[dn], mapping[nb])
        cert = wl_hash(H, node_attr=None, iterations=3)
        return cert, H

    seen_global: Dict[int, Set[str]] = {m: set() for m in range(1, max_order+1)}

    for root in nodes_list:
        init = frozenset([root])
        queue = deque([init])
        local_seen: Set[FrozenSet[DiamondNode]] = set([init])
        while queue:
            curr = queue.popleft()
            order = len(curr)
            cert, H = canonicalize(set(curr))
            if cert not in seen_global[order]:
                tc = TopoCluster(order=order, diamond_nodes=set(curr), diamond_subgraph=H, wl_cert=cert)
                unique_by_order[order][cert] = tc
                seen_global[order].add(cert)
            if order == max_order:
                continue
            boundary = set()
            for dn in curr:
                boundary |= nbrs[dn]
            boundary -= set(curr)
            for nb in sorted(boundary, key=lambda d: (d.s, d.i, d.j, d.k)):
                new_set = frozenset(set(curr) | {nb})
                if new_set not in local_seen:
                    local_seen.add(new_set)
                    queue.append(new_set)

    ordered: Dict[int, List[TopoCluster]] = {}
    for m in range(1, max_order+1):
        lst = list(unique_by_order[m].values())
        lst.sort(key=lambda tc: (tc.order, tc.diamond_subgraph.number_of_edges(), tc.wl_cert))
        ordered[m] = lst
    return ordered


def topo_to_pyrochlore(tc: TopoCluster, latt: LatticeGraphs):
    pyro_nodes: Set[int] = set()
    for dn in tc.diamond_nodes:
        pyro_nodes.update(latt.diamond_node_to_pyro_nodes[dn])
    pyro_edges: Set[Tuple[int, int]] = set()
    for dn in tc.diamond_nodes:
        pins = latt.diamond_node_to_pyro_nodes[dn]
        for a, b in itertools.combinations(sorted(pins), 2):
            pyro_edges.add((min(a, b), max(a, b)))
    tetra = []
    for dn in tc.diamond_nodes:
        pins = sorted(latt.diamond_node_to_pyro_nodes[dn])
        tetra.append(tuple(pins))
    tc.pyro_vertices = pyro_nodes
    tc.pyro_edges = pyro_edges
    tc.tetrahedra = tetra


def count_unique_subgraph_images(H: nx.Graph, G: nx.Graph) -> int:
    GM = nx.algorithms.isomorphism.GraphMatcher(G, H)
    images: Set[FrozenSet[int]] = set()
    for mapping in GM.subgraph_isomorphisms_iter():
        image_nodes = frozenset(mapping.keys())
        images.add(image_nodes)
    return len(images)


def compute_multiplicities(clusters_by_order: Dict[int, List[TopoCluster]], latt: LatticeGraphs) -> None:
    Gd = latt.diamond
    num_diamond_edges = Gd.number_of_edges()
    for m, lst in clusters_by_order.items():
        for tc in lst:
            count = count_unique_subgraph_images(tc.diamond_subgraph, Gd)
            tc.multiplicity_per_site = count / float(num_diamond_edges)


def compute_subclusters(clusters_by_order: Dict[int, List[TopoCluster]]) -> Dict[int, List[TopoCluster]]:
    for m, lst in clusters_by_order.items():
        for tc in lst:
            H_big = tc.diamond_subgraph
            submap: Dict[int, List[Tuple[int, int]]] = {}
            for k in range(1, m):
                entries: List[Tuple[int, int]] = []
                for idx_small, small_tc in enumerate(clusters_by_order[k], start=1):
                    cnt = count_unique_subgraph_images(small_tc.diamond_subgraph, H_big)
                    if cnt > 0:
                        entries.append((idx_small, cnt))
                submap[k] = entries
            tc.subclusters = submap
    return clusters_by_order


def write_cluster_files(output_dir: Path, clusters_by_order: Dict[int, List[TopoCluster]], latt: LatticeGraphs, L: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pyro_coords: Dict[int, Tuple[int, float, float, float]] = {}
    for pid, (u, v) in latt.pyro_to_diamond_edge.items():
        pu = diamond_node_position(u)
        pv = diamond_node_position(v)
        mx, my, mz = midpoint(pu, pv)
        pyro_coords[pid] = (pid, mx, my, mz)

    # Use global cluster ID counter
    global_cluster_id = 1
    # Use global cluster ID counter
    global_cluster_id = 1
    for m, lst in clusters_by_order.items():
        for tc in lst:
            node_list = sorted(tc.pyro_vertices)
            index_of = {nid: i for i, nid in enumerate(node_list)}
            N = len(node_list)
            adj = [[0]*N for _ in range(N)]
            for (a, b) in tc.pyro_edges:
                ia = index_of[a]; ib = index_of[b]
                adj[ia][ib] = 1
                adj[ib][ia] = 1
            fname = output_dir / f"cluster_{global_cluster_id}_order_{m}.dat"
            with fname.open("w") as f:
                f.write(f"# Cluster ID: {global_cluster_id}\n")
                f.write(f"# Order (number of tetrahedra): {m}\n")
                f.write(f"# Multiplicity: {tc.multiplicity_per_site}\n")
                f.write(f"# Number of vertices: {N}\n")
                f.write(f"# Number of edges: {sum(sum(row) for row in adj)//2}\n")
                f.write("\n")
                f.write("# Vertices (index, x, y, z):\n")
                for nid in node_list:
                    orig_id, x, y, z = pyro_coords[nid]
                    f.write(f"{orig_id}, {x:.6f}, {y:.6f}, {z:.6f}\n")
                f.write("\n")
                f.write("# Edges (u, v):\n")
                for (a, b) in sorted(tc.pyro_edges):
                    f.write(f"{a}, {b}\n")
                f.write("\n")
                f.write("# Tetrahedra (vertex1, vertex2, vertex3, vertex4):\n")
                for tet in tc.tetrahedra:
                    f.write(f"{tet[0]}, {tet[1]}, {tet[2]}, {tet[3]}\n")
                f.write("\n")
                f.write("# Adjacency Matrix:\n")
                for i in range(N):
                    f.write(" ".join(str(v) for v in adj[i]) + "\n")
                f.write("\n")
                f.write("# Node Mapping (original_id: matrix_index):\n")
                for nid in node_list:
                    f.write(f"{nid}: {index_of[nid]}\n")
            global_cluster_id += 1

    subs_path = output_dir / "subclusters_info.txt"
    with subs_path.open("w") as f:
        f.write("# Subclusters information for each topologically distinct cluster\n")
        f.write("# Format: Cluster_ID, Order, Subclusters[(ID, Multiplicity), ...]\n\n")
        
        # Create mapping from (order, local_idx) to global_cluster_id
        global_id_map = {}
        global_cluster_id = 1
        for m, lst in clusters_by_order.items():
            for local_idx, tc in enumerate(lst, start=1):
                global_id_map[(m, local_idx)] = global_cluster_id
                global_cluster_id += 1
        
        # Now write the subclusters info using global IDs
        global_cluster_id = 1
        for m, lst in clusters_by_order.items():
            for local_idx, tc in enumerate(lst, start=1):
                f.write(f"Cluster {global_cluster_id} (Order {m}):\n")
                if m == 1:
                    f.write("  No subclusters (order 1 cluster)\n\n")
                else:
                    f.write("  Subclusters: ")
                    parts = []
                    for k in range(1, m):
                        for (local_cid, cnt) in tc.subclusters.get(k, []):
                            global_cid = global_id_map[(k, local_cid)]
                            parts.append(f"({global_cid}, {cnt})")
                    f.write(", ".join(parts) + "\n\n")
                global_cluster_id += 1


def generate_nlce_pyrochlore(output_root: str, max_order: int, L: int | None = None) -> Path:
    if L is None:
        L = max(3, max_order + 2)
    Gd = build_diamond_supercell(L)
    latt = build_pyrochlore_from_diamond(Gd)
    clusters_by_order = enumerate_topo_clusters_diamond(Gd, L, max_order)
    for m, lst in clusters_by_order.items():
        for tc in lst:
            topo_to_pyrochlore(tc, latt)
    compute_multiplicities(clusters_by_order, latt)
    compute_subclusters(clusters_by_order)
    outdir = Path(output_root) / f"cluster_info_order_{max_order}"
    write_cluster_files(outdir, clusters_by_order, latt, L)
    return outdir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate NLCE clusters for a pyrochlore lattice.")
    parser.add_argument("-o", "--output_dir", required=True, help="Output root directory.")
    parser.add_argument("-m", "--max_order", type=int, required=True, help="Maximum cluster order.")
    parser.add_argument("-L", "--lattice_size", dest="L", type=int, default=None,
                        help="Diamond supercell size (default=max(3, max_order+2)).")
    args = parser.parse_args()

    outdir = generate_nlce_pyrochlore(args.output_dir, args.max_order, args.L)
    print(str(outdir))