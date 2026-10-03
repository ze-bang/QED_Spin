// =============================================================================
// src/input/lattice.cpp
//
// The lattice generators. Each one describes its lattice as cells (a basis of
// sites repeated along the lattice vectors) and the nearest-neighbour links of
// one cell. `build` lays the cells out, follows the links (wrapping them on a
// periodic lattice) and files every pair of sites under its distance shell, which
// gives the next-nearest and third-nearest neighbours and checks the links.
// =============================================================================

#include <ed/input/lattice.h>

#include <ed/core/errors.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>

namespace ed::input {

namespace {

using Pair = std::pair<std::size_t, std::size_t>;
using Cell = std::array<long, 3>;

constexpr double kSqrt3 = 1.7320508075688772;

Pair unordered(std::size_t a, std::size_t b) { return a < b ? Pair{a, b} : Pair{b, a}; }

// Appends the bond i -> j unless it joins a site to itself or a pair that already has a bond.
void add_bond(std::vector<Bond>& out, std::set<Pair>& seen, std::size_t i, std::size_t j, int type = 0) {
    if (i != j && seen.insert(unordered(i, j)).second) out.push_back(Bond{i, j, type});
}

std::vector<Pair> pairs_of(const std::vector<Bond>& bonds) {
    std::vector<Pair> out;
    out.reserve(bonds.size());
    for (const auto& b : bonds) out.emplace_back(b.i, b.j);
    return out;
}

// A nearest-neighbour link, from basis site u of a cell to basis site v of the cell d away.
struct Link {
    int u, v;
    Cell d;
    int type = 0;
};

// n[0] x n[1] x n[2] cells along the lattice vectors a[0..2], the first direction fastest in
// the site order: site (c, u) is ((c2 * n1 + c1) * n0 + c0) * basis.size() + u. Only the first
// `dims` directions exist; the others hold one cell and never wrap.
struct Cells {
    std::array<std::size_t, 3> n{1, 1, 1};
    std::array<Position, 3> a{};
    int dims = 1;
    std::vector<Position> basis;
    std::vector<Link> links;
};

class Layout {
public:
    Layout(const Cells& C, bool pbc) : C_(C), pbc_(pbc) {}

    std::size_t index(const Cell& c, int u) const {
        const std::size_t cell = (static_cast<std::size_t>(c[2]) * C_.n[1] + static_cast<std::size_t>(c[1])) * C_.n[0]
                                 + static_cast<std::size_t>(c[0]);
        return cell * C_.basis.size() + static_cast<std::size_t>(u);
    }

    // The cell d away from c, wrapped along a periodic direction; false past an open edge.
    bool shift(const Cell& c, const Cell& d, Cell& out) const {
        for (int k = 0; k < 3; ++k) {
            const long n = static_cast<long>(C_.n[k]);
            long t = c[k] + d[k];
            if (pbc_ && k < C_.dims)
                t = ((t % n) + n) % n;
            else if (t < 0 || t >= n)
                return false;
            out[k] = t;
        }
        return true;
    }

    Position position(const Cell& c, int u) const {
        Position p = C_.basis[static_cast<std::size_t>(u)];
        for (int k = 0; k < 3; ++k)
            for (int x = 0; x < 3; ++x) p[x] += static_cast<double>(c[k]) * C_.a[k][x];
        return p;
    }

    std::vector<Cell> cells() const {
        std::vector<Cell> out;
        for (long c2 = 0; c2 < static_cast<long>(C_.n[2]); ++c2)
            for (long c1 = 0; c1 < static_cast<long>(C_.n[1]); ++c1)
                for (long c0 = 0; c0 < static_cast<long>(C_.n[0]); ++c0) out.push_back({c0, c1, c2});
        return out;
    }

private:
    const Cells& C_;
    bool pbc_;
};

// Files every pair of sites under the nearest of its periodic images: its shell is the rank of
// that distance among the distances of the infinite lattice. Every lattice vector up to the third
// shell spans fewer than kReach cells per direction for the lattices here, so following those
// vectors from every site finds every pair within the third shell. The first shell must be
// exactly the bonds the links made.
void fill_shells(Lattice& L, const Cells& C, const Layout& lay, const std::vector<Cell>& cells) {
    constexpr long kReach = 3;
    struct Vec {
        int u, v;
        Cell d;
        double r;
        int rank = 0;
    };
    std::vector<Vec> all;
    std::array<long, 3> reach{};
    for (int k = 0; k < C.dims; ++k) reach[k] = kReach;
    const int nb = static_cast<int>(C.basis.size());
    for (int u = 0; u < nb; ++u)
        for (int v = 0; v < nb; ++v)
            for (long d2 = -reach[2]; d2 <= reach[2]; ++d2)
                for (long d1 = -reach[1]; d1 <= reach[1]; ++d1)
                    for (long d0 = -reach[0]; d0 <= reach[0]; ++d0) {
                        const Cell d{d0, d1, d2};
                        const Position p = lay.position(d, v);
                        const Position& q = C.basis[static_cast<std::size_t>(u)];
                        const double r = std::hypot(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
                        // scale-free: lattice geometry
                        if (r > 1e-9) all.push_back({u, v, d, r});
                    }
    std::vector<double> radii;
    radii.reserve(all.size());
    for (const auto& w : all) radii.push_back(w.r);
    std::sort(radii.begin(), radii.end());
    std::vector<double> shells;              // the three smallest distinct distances
    for (double r : radii) {
        // scale-free: lattice geometry
        if (shells.empty() || r > shells.back() * (1.0 + 1e-9)) shells.push_back(r);
        if (shells.size() == 3) break;
    }
    std::vector<Vec> vecs;                   // the lattice vectors up to the third shell
    for (auto w : all) {
        for (w.rank = 0; w.rank < static_cast<int>(shells.size()); ++w.rank)
            // scale-free: lattice geometry
            if (std::abs(w.r - shells[w.rank]) <= 1e-9 * shells[w.rank]) break;
        if (w.rank < static_cast<int>(shells.size())) vecs.push_back(w);
    }
    std::map<Pair, int> nearest;
    for (const auto& c : cells) {
        for (const auto& w : vecs) {
            Cell t;
            if (!lay.shift(c, w.d, t)) continue;
            const std::size_t i = lay.index(c, w.u), j = lay.index(t, w.v);
            if (i == j) continue;
            const auto [it, fresh] = nearest.emplace(unordered(i, j), w.rank);
            if (!fresh) it->second = std::min(it->second, w.rank);
        }
    }
    std::set<Pair> bonded, first;
    for (const auto& b : L.nn_bonds) bonded.insert(unordered(b.i, b.j));
    for (const auto& [p, rank] : nearest) {
        if (rank == 0)
            first.insert(p);
        else
            (rank == 1 ? L.nnn_bonds : L.nnnn_bonds).push_back(Bond{p.first, p.second});
    }
    if (first != bonded)
        throw std::logic_error(L.label
                               + ": the nearest-neighbour bonds are not the first distance "
                                 "shell of the lattice");
    L.shells_known = true;
}

Lattice build(const Cells& C, bool pbc, std::string label) {
    const Layout lay(C, pbc);
    const auto cells = lay.cells();
    Lattice L;
    L.num_sites = cells.size() * C.basis.size();
    L.pbc = pbc;
    L.label = std::move(label);
    L.lattice_vectors = C.a;
    for (const auto& c : cells)
        for (int u = 0; u < static_cast<int>(C.basis.size()); ++u) {
            L.positions.push_back(lay.position(c, u));
            L.sublattice.push_back(u);
        }
    std::set<Pair> seen;
    for (const auto& c : cells)
        for (const auto& l : C.links) {
            Cell t;
            if (lay.shift(c, l.d, t)) add_bond(L.nn_bonds, seen, lay.index(c, l.u), lay.index(t, l.v), l.type);
        }
    fill_shells(L, C, lay, cells);
    return L;
}

void require_cells(const char* name, bool pbc, bool has_basis, std::initializer_list<std::size_t> n) {
    for (std::size_t len : n) {
        if (len == 0) throw InvalidRequest(std::string(name) + ": every length must be > 0");
        if (has_basis && pbc && len == 1)
            throw InvalidRequest(std::string(name)
                                 + ": a periodic length of 1 joins bonds of "
                                   "different kinds to one pair of sites; use at least 2 cells, or "
                                   "pbc=False");
    }
}

std::string size_label(const char* name, std::initializer_list<std::size_t> n, bool pbc) {
    std::string s = std::string(name) + "[";
    for (auto it = n.begin(); it != n.end(); ++it) s += (it == n.begin() ? "" : "x") + std::to_string(*it);
    return s + (pbc ? " PBC]" : " OBC]");
}

}  // namespace

// ---------------------------------------------------------------------------
// Lattice helpers
// ---------------------------------------------------------------------------

std::vector<std::pair<std::size_t, std::size_t>> Lattice::nn_pairs() const { return pairs_of(nn_bonds); }

std::vector<std::pair<std::size_t, std::size_t>> Lattice::nnn_pairs() const {
    if (!shells_known && nnn_bonds.empty())
        throw InvalidRequest("nnn_pairs: " + label
                             + " was built from an adjacency list and knows no "
                               "next-nearest neighbours; pass the pairs yourself");
    return pairs_of(nnn_bonds);
}

std::vector<std::pair<std::size_t, std::size_t>> Lattice::nnnn_pairs() const {
    if (!shells_known && nnnn_bonds.empty())
        throw InvalidRequest("nnnn_pairs: " + label
                             + " was built from an adjacency list and knows "
                               "no third-nearest neighbours; pass the pairs yourself");
    return pairs_of(nnnn_bonds);
}

std::vector<std::size_t> Lattice::all_sites() const {
    std::vector<std::size_t> out(num_sites);
    for (std::size_t i = 0; i < num_sites; ++i) out[i] = i;
    return out;
}

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------

namespace lattice {

Lattice chain(std::size_t length, bool pbc) {
    require_cells("chain", pbc, false, {length});
    Cells C;
    C.n = {length, 1, 1};
    C.a = {{{1.0, 0.0, 0.0}, {}, {}}};
    C.basis = {{0.0, 0.0, 0.0}};
    C.links = {{0, 0, {1, 0, 0}}};
    return build(C, pbc, "chain[L=" + std::to_string(length) + (pbc ? " PBC]" : " OBC]"));
}

Lattice square(std::size_t Lx, std::size_t Ly, bool pbc) {
    require_cells("square", pbc, false, {Lx, Ly});
    Cells C;
    C.n = {Lx, Ly, 1};
    C.dims = 2;
    C.a = {{{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {}}};
    C.basis = {{0.0, 0.0, 0.0}};
    C.links = {{0, 0, {1, 0, 0}}, {0, 0, {0, 1, 0}}};
    return build(C, pbc, size_label("square", {Lx, Ly}, pbc));
}

Lattice triangular(std::size_t Lx, std::size_t Ly, bool pbc) {
    require_cells("triangular", pbc, false, {Lx, Ly});
    Cells C;
    C.n = {Lx, Ly, 1};
    C.dims = 2;
    C.a = {{{1.0, 0.0, 0.0}, {0.5, kSqrt3 / 2.0, 0.0}, {}}};
    C.basis = {{0.0, 0.0, 0.0}};
    C.links = {{0, 0, {1, 0, 0}}, {0, 0, {0, 1, 0}}, {0, 0, {-1, 1, 0}}};
    return build(C, pbc, size_label("triangular", {Lx, Ly}, pbc));
}

Lattice honeycomb(std::size_t Lx, std::size_t Ly, bool pbc) {
    require_cells("honeycomb", pbc, true, {Lx, Ly});
    // a1 = (3/2, sqrt(3)/2), a2 = (3/2, -sqrt(3)/2); A = (0,0), B = (1,0). Every bond runs from
    // A to B: z inside the cell, x to the cell -a1, y to the cell -a2 (Kitaev colours 2, 0, 1).
    Cells C;
    C.n = {Lx, Ly, 1};
    C.dims = 2;
    C.a = {{{1.5, kSqrt3 / 2.0, 0.0}, {1.5, -kSqrt3 / 2.0, 0.0}, {}}};
    C.basis = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}};
    C.links = {{0, 1, {0, 0, 0}, 2}, {0, 1, {-1, 0, 0}, 0}, {0, 1, {0, -1, 0}, 1}};
    return build(C, pbc, size_label("honeycomb", {Lx, Ly}, pbc));
}

Lattice kagome(std::size_t Lx, std::size_t Ly, bool pbc) {
    require_cells("kagome", pbc, true, {Lx, Ly});
    // Triangular Bravais lattice; A = (0,0), B = (1/2,0), C = (1/4, sqrt(3)/4). Both triangles
    // run counter-clockwise: the up one A -> B -> C inside the cell, the down one
    // B(R) -> C(R+a1-a2) -> A(R+a1) -> B(R).
    Cells C;
    C.n = {Lx, Ly, 1};
    C.dims = 2;
    C.a = {{{1.0, 0.0, 0.0}, {0.5, kSqrt3 / 2.0, 0.0}, {}}};
    C.basis = {{0.0, 0.0, 0.0}, {0.5, 0.0, 0.0}, {0.25, kSqrt3 / 4.0, 0.0}};
    C.links = {{0, 1, {0, 0, 0}},  {1, 2, {0, 0, 0}},  {2, 0, {0, 0, 0}},
               {0, 1, {-1, 0, 0}}, {1, 2, {1, -1, 0}}, {2, 0, {0, 1, 0}}};
    return build(C, pbc, size_label("kagome", {Lx, Ly}, pbc));
}

Lattice pyrochlore(std::size_t Lx, std::size_t Ly, std::size_t Lz, bool pbc) {
    require_cells("pyrochlore", pbc, true, {Lx, Ly, Lz});
    // FCC primitive vectors a0 = (0,1/2,1/2), a1 = (1/2,0,1/2), a2 = (1/2,1/2,0); sublattice u > 0
    // sits at a_{u-1}/2. Cell (i, j, k) spans i a0 + j a1 + k a2, k fastest in the site order, so
    // the directions here run (a2, a1, a0) and a cell offset reads (dk, dj, di). The up
    // tetrahedron is the cell's own four sites; the down one through site 0 of cell R holds
    // sublattice u > 0 of the cell R - a_{u-1}. Bonds run from the lower sublattice to the higher.
    const Position a0{0.0, 0.5, 0.5}, a1{0.5, 0.0, 0.5}, a2{0.5, 0.5, 0.0};
    Cells C;
    C.n = {Lz, Ly, Lx};
    C.dims = 3;
    C.a = {a2, a1, a0};
    C.basis = {{0.0, 0.0, 0.0}, {0.0, 0.25, 0.25}, {0.25, 0.0, 0.25}, {0.25, 0.25, 0.0}};
    C.links = {{0, 1, {0, 0, 0}},  {0, 2, {0, 0, 0}},  {0, 3, {0, 0, 0}},  {1, 2, {0, 0, 0}},
               {1, 3, {0, 0, 0}},  {2, 3, {0, 0, 0}},  {0, 1, {0, 0, -1}}, {0, 2, {0, -1, 0}},
               {0, 3, {-1, 0, 0}}, {1, 2, {0, -1, 1}}, {1, 3, {-1, 0, 1}}, {2, 3, {-1, 1, 0}}};
    Lattice L = build(C, pbc, size_label("pyrochlore", {Lx, Ly, Lz}, pbc));
    L.lattice_vectors = {a0, a1, a2};
    return L;
}

Lattice from_neighbor_lists(const std::vector<Position>& positions,
                            const std::vector<std::pair<std::size_t, std::size_t>>& nn_pairs,
                            const std::vector<int>& sublattice) {
    Lattice L;
    L.num_sites = positions.size();
    L.positions = positions;
    if (sublattice.empty()) {
        L.sublattice.assign(L.num_sites, 0);
    } else {
        if (sublattice.size() != L.num_sites) {
            throw InvalidRequest("from_neighbor_lists: sublattice size must equal positions size");
        }
        L.sublattice = sublattice;
    }
    std::set<Pair> seen;
    L.nn_bonds.reserve(nn_pairs.size());
    for (auto [i, j] : nn_pairs) {
        if (i >= L.num_sites || j >= L.num_sites) {
            throw std::out_of_range("from_neighbor_lists: bond (" + std::to_string(i) + ", " + std::to_string(j)
                                    + ") has an endpoint out of range");
        }
        if (i == j) {
            throw InvalidRequest("from_neighbor_lists: bond (" + std::to_string(i) + ", " + std::to_string(j)
                                 + ") joins a site to itself");
        }
        add_bond(L.nn_bonds, seen, i, j);
    }
    L.pbc = false;
    L.label = "custom[N=" + std::to_string(L.num_sites) + "]";
    return L;
}

namespace {

bool parse_count(const std::string& s, std::size_t& out) {
    if (s.empty() || s.size() > 18 || !std::all_of(s.begin(), s.end(), [](unsigned char c) { return std::isdigit(c); }))
        return false;
    out = std::stoull(s);
    return true;
}

bool parse_real(const std::string& s, double& out) {
    char* end = nullptr;
    out = std::strtod(s.c_str(), &end);
    return end != s.c_str() && *end == '\0' && std::isfinite(out);
}

}  // namespace

Lattice from_cluster_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) { throw std::runtime_error("from_cluster_file: cannot open " + path); }
    enum Section { None, Positions, Edges };
    Section sect = None;
    std::array<bool, 3> opened{};
    std::array<std::size_t, 3> stated{}, stated_at{};   // a block's stated length, and its line
    std::array<bool, 3> has_count{};
    bool block_start = false;
    std::vector<Position> positions;
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    std::vector<std::size_t> edge_line;
    std::string line;
    std::size_t no = 0;
    const auto fail = [&](std::size_t at, const std::string& why) {
        throw InvalidRequest("from_cluster_file: " + path + ":" + std::to_string(at) + ": " + why);
    };
    while (std::getline(in, line)) {
        ++no;
        std::istringstream iss(line);
        std::vector<std::string> tok;
        for (std::string t; iss >> t;) tok.push_back(t);
        if (tok.empty() || tok[0][0] == '#') continue;
        std::string head = tok[0];
        std::transform(head.begin(), head.end(), head.begin(), [](unsigned char c) { return std::tolower(c); });
        if (!head.empty() && head.back() == ':') head.pop_back();
        if (head == "positions" || head == "edges" || head == "bonds") {
            sect = head == "positions" ? Positions : Edges;
            if (opened[sect]) fail(no, "a second '" + tok[0] + "' block");
            opened[sect] = true;
            if (tok.size() > 2) fail(no, "a header is the block's name and at most its length: '" + line + "'");
            if (tok.size() == 2) {
                if (!parse_count(tok[1], stated[sect])) fail(no, "not a length: '" + line + "'");
                has_count[sect] = true;
                stated_at[sect] = no;
            }
            block_start = !has_count[sect];
            continue;
        }
        if (sect == None) fail(no, "data before a 'positions' or 'edges' header: '" + line + "'");
        if (tok.size() == 1 && block_start) {
            if (!parse_count(tok[0], stated[sect])) fail(no, "not a length: '" + line + "'");
            has_count[sect] = true;
            stated_at[sect] = no;
            block_start = false;
            continue;
        }
        block_start = false;
        if (sect == Positions) {
            std::vector<double> v(tok.size());
            if (tok.size() < 2 || tok.size() > 4)
                fail(no, "a position is 'x y', 'x y z' or 'id x y z': '" + line + "'");
            for (std::size_t k = 0; k < tok.size(); ++k)
                if (!parse_real(tok[k], v[k])) fail(no, "not a number: '" + tok[k] + "'");
            if (tok.size() == 4) {
                std::size_t id = 0;
                if (!parse_count(tok[0], id) || id != positions.size())
                    fail(no, "the id must be the site's index, counting from 0: expected "
                                 + std::to_string(positions.size()) + ", read '" + tok[0] + "'");
                positions.push_back({v[1], v[2], v[3]});
            } else {
                positions.push_back({v[0], v[1], tok.size() == 3 ? v[2] : 0.0});
            }
        } else {
            std::size_t i = 0, j = 0;
            if (tok.size() != 2 || !parse_count(tok[0], i) || !parse_count(tok[1], j))
                fail(no, "an edge is 'i j' with site indices: '" + line + "'");
            if (i == j) fail(no, "an edge joins site " + tok[0] + " to itself");
            edges.emplace_back(i, j);
            edge_line.push_back(no);
        }
    }
    if (positions.empty()) throw InvalidRequest("from_cluster_file: " + path + " lists no positions");
    const std::array<std::size_t, 3> got{0, positions.size(), edges.size()};
    for (int s : {Positions, Edges})
        if (has_count[s] && stated[s] != got[s])
            fail(stated_at[s],
                 "the block states " + std::to_string(stated[s]) + " lines and holds " + std::to_string(got[s]));
    for (std::size_t e = 0; e < edges.size(); ++e)
        if (std::max(edges[e].first, edges[e].second) >= positions.size())
            fail(edge_line[e], "edge (" + std::to_string(edges[e].first) + ", " + std::to_string(edges[e].second)
                                   + ") names a site past the " + std::to_string(positions.size()) + " positions");
    return from_neighbor_lists(positions, edges);
}

}  // namespace lattice

}  // namespace ed::input
