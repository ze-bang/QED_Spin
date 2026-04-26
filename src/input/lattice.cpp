// =============================================================================
// src/input/lattice.cpp
//
// Implementation of `ed::input::lattice::*` factory functions that
// replace the per-helper Python files in `python/edlib/helper_*.py`.
// =============================================================================

#include <ed/input/lattice.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ed::input {

namespace {

constexpr double kSqrt3 = 1.7320508075688772;

inline std::pair<std::size_t, std::size_t> ordered_pair(std::size_t a,
                                                        std::size_t b) {
    return a < b ? std::pair<std::size_t, std::size_t>{a, b}
                 : std::pair<std::size_t, std::size_t>{b, a};
}

inline void push_unique_bond(std::vector<Bond>& out,
                             std::set<std::pair<std::size_t, std::size_t>>& seen,
                             std::size_t i, std::size_t j, int type = 0) {
    if (i == j) return;
    auto key = ordered_pair(i, j);
    if (seen.insert(key).second) {
        out.push_back(Bond{key.first, key.second, type});
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// Lattice helpers
// ---------------------------------------------------------------------------

namespace {
std::vector<std::pair<std::size_t, std::size_t>> bonds_to_pairs(
    const std::vector<Bond>& bonds) {
    std::vector<std::pair<std::size_t, std::size_t>> out;
    out.reserve(bonds.size());
    for (const auto& b : bonds) out.emplace_back(b.i, b.j);
    return out;
}
}  // namespace

std::vector<std::pair<std::size_t, std::size_t>> Lattice::nn_pairs() const {
    return bonds_to_pairs(nn_bonds);
}
std::vector<std::pair<std::size_t, std::size_t>> Lattice::nnn_pairs() const {
    return bonds_to_pairs(nnn_bonds);
}
std::vector<std::pair<std::size_t, std::size_t>> Lattice::nnnn_pairs() const {
    return bonds_to_pairs(nnnn_bonds);
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
    if (length == 0) {
        throw std::invalid_argument("chain: length must be > 0");
    }
    Lattice L;
    L.num_sites = length;
    L.positions.reserve(length);
    L.sublattice.assign(length, 0);
    for (std::size_t i = 0; i < length; ++i) {
        L.positions.push_back({static_cast<double>(i), 0.0, 0.0});
    }
    L.lattice_vectors[0] = {1.0, 0.0, 0.0};
    L.pbc = pbc;
    L.label = "chain[L=" + std::to_string(length) + (pbc ? " PBC]" : " OBC]");

    std::set<std::pair<std::size_t, std::size_t>> seen;
    for (std::size_t i = 0; i + 1 < length; ++i) {
        push_unique_bond(L.nn_bonds, seen, i, i + 1);
    }
    if (pbc && length > 2) {
        push_unique_bond(L.nn_bonds, seen, length - 1, 0);
    }
    return L;
}

Lattice square(std::size_t Lx, std::size_t Ly, bool pbc) {
    if (Lx == 0 || Ly == 0) {
        throw std::invalid_argument("square: Lx, Ly must be > 0");
    }
    Lattice L;
    L.num_sites = Lx * Ly;
    L.positions.reserve(L.num_sites);
    L.sublattice.assign(L.num_sites, 0);
    auto idx = [&](std::size_t x, std::size_t y) {
        return y * Lx + x;
    };
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            L.positions.push_back({static_cast<double>(x),
                                   static_cast<double>(y), 0.0});
        }
    }
    L.lattice_vectors[0] = {1.0, 0.0, 0.0};
    L.lattice_vectors[1] = {0.0, 1.0, 0.0};
    L.pbc = pbc;
    L.label = "square[" + std::to_string(Lx) + "x" + std::to_string(Ly) +
              (pbc ? " PBC]" : " OBC]");

    std::set<std::pair<std::size_t, std::size_t>> seen;
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            std::size_t i = idx(x, y);
            // +x neighbour
            if (x + 1 < Lx) push_unique_bond(L.nn_bonds, seen, i, idx(x + 1, y));
            else if (pbc && Lx > 2)
                push_unique_bond(L.nn_bonds, seen, i, idx(0, y));
            // +y neighbour
            if (y + 1 < Ly) push_unique_bond(L.nn_bonds, seen, i, idx(x, y + 1));
            else if (pbc && Ly > 2)
                push_unique_bond(L.nn_bonds, seen, i, idx(x, 0));
        }
    }
    return L;
}

Lattice triangular(std::size_t Lx, std::size_t Ly, bool pbc) {
    if (Lx == 0 || Ly == 0) {
        throw std::invalid_argument("triangular: Lx, Ly must be > 0");
    }
    Lattice L;
    L.num_sites = Lx * Ly;
    L.positions.reserve(L.num_sites);
    L.sublattice.assign(L.num_sites, 0);
    auto idx = [&](std::size_t x, std::size_t y) { return y * Lx + x; };
    const double a1x = 1.0, a1y = 0.0;
    const double a2x = 0.5, a2y = kSqrt3 / 2.0;
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            L.positions.push_back({a1x * x + a2x * y, a1y * x + a2y * y, 0.0});
        }
    }
    L.lattice_vectors[0] = {a1x, a1y, 0.0};
    L.lattice_vectors[1] = {a2x, a2y, 0.0};
    L.pbc = pbc;
    L.label = "triangular[" + std::to_string(Lx) + "x" + std::to_string(Ly) +
              (pbc ? " PBC]" : " OBC]");

    auto wrap = [&](long v, std::size_t L_) -> long {
        if (pbc) return ((v % static_cast<long>(L_)) + static_cast<long>(L_)) %
                        static_cast<long>(L_);
        if (v < 0 || v >= static_cast<long>(L_)) return -1;
        return v;
    };
    std::set<std::pair<std::size_t, std::size_t>> seen;
    static const std::array<std::pair<int, int>, 3> nn_offsets = {{
        {1, 0}, {0, 1}, {-1, 1}}};
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            std::size_t i = idx(x, y);
            for (auto [dx, dy] : nn_offsets) {
                long nx = wrap(static_cast<long>(x) + dx, Lx);
                long ny = wrap(static_cast<long>(y) + dy, Ly);
                if (nx < 0 || ny < 0) continue;
                std::size_t j = idx(static_cast<std::size_t>(nx),
                                    static_cast<std::size_t>(ny));
                push_unique_bond(L.nn_bonds, seen, i, j);
            }
        }
    }
    return L;
}

Lattice honeycomb(std::size_t Lx, std::size_t Ly, bool pbc) {
    if (Lx == 0 || Ly == 0) {
        throw std::invalid_argument("honeycomb: Lx, Ly must be > 0");
    }
    Lattice L;
    L.num_sites = 2 * Lx * Ly;
    L.positions.reserve(L.num_sites);
    L.sublattice.assign(L.num_sites, 0);
    auto idx = [&](std::size_t x, std::size_t y, std::size_t s) {
        return (y * Lx + x) * 2 + s;
    };
    // Lattice vectors a1 = (3/2, sqrt(3)/2), a2 = (3/2, -sqrt(3)/2);
    // basis: A = (0,0), B = (1, 0).
    const double a1x = 1.5, a1y = kSqrt3 / 2.0;
    const double a2x = 1.5, a2y = -kSqrt3 / 2.0;
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            double rx = a1x * x + a2x * y;
            double ry = a1y * x + a2y * y;
            L.positions.push_back({rx, ry, 0.0});
            L.positions.push_back({rx + 1.0, ry, 0.0});
            L.sublattice[idx(x, y, 0)] = 0;
            L.sublattice[idx(x, y, 1)] = 1;
        }
    }
    L.lattice_vectors[0] = {a1x, a1y, 0.0};
    L.lattice_vectors[1] = {a2x, a2y, 0.0};
    L.pbc = pbc;
    L.label = "honeycomb[" + std::to_string(Lx) + "x" + std::to_string(Ly) +
              (pbc ? " PBC]" : " OBC]");

    auto wrap = [&](long v, std::size_t L_) -> long {
        if (pbc) return ((v % static_cast<long>(L_)) + static_cast<long>(L_)) %
                        static_cast<long>(L_);
        if (v < 0 || v >= static_cast<long>(L_)) return -1;
        return v;
    };
    // From every A site: NN to B(x,y) [bond_type=2 / z], B(x-1,y) [bond_type=0 / x],
    // B(x,y-1) [bond_type=1 / y]. Bond colours follow the standard Kitaev
    // convention.
    std::set<std::pair<std::size_t, std::size_t>> seen;
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            std::size_t a = idx(x, y, 0);
            // z bond
            std::size_t bz = idx(x, y, 1);
            push_unique_bond(L.nn_bonds, seen, a, bz, 2);
            // x bond: A(x,y) - B(x-1,y)
            long nx = wrap(static_cast<long>(x) - 1, Lx);
            if (nx >= 0) {
                std::size_t bx = idx(static_cast<std::size_t>(nx), y, 1);
                push_unique_bond(L.nn_bonds, seen, a, bx, 0);
            }
            // y bond: A(x,y) - B(x,y-1)
            long ny = wrap(static_cast<long>(y) - 1, Ly);
            if (ny >= 0) {
                std::size_t by = idx(x, static_cast<std::size_t>(ny), 1);
                push_unique_bond(L.nn_bonds, seen, a, by, 1);
            }
        }
    }
    return L;
}

Lattice kagome(std::size_t Lx, std::size_t Ly, bool pbc) {
    if (Lx == 0 || Ly == 0) {
        throw std::invalid_argument("kagome: Lx, Ly must be > 0");
    }
    Lattice L;
    L.num_sites = 3 * Lx * Ly;
    L.positions.reserve(L.num_sites);
    L.sublattice.assign(L.num_sites, 0);
    auto idx = [&](std::size_t x, std::size_t y, std::size_t s) {
        return (y * Lx + x) * 3 + s;
    };
    // Triangular Bravais lattice; basis: A = (0,0), B = (1/2, 0),
    // C = (1/4, sqrt(3)/4).
    const double a1x = 1.0, a1y = 0.0;
    const double a2x = 0.5, a2y = kSqrt3 / 2.0;
    static const std::array<std::array<double, 2>, 3> basis = {{
        {0.0, 0.0}, {0.5, 0.0}, {0.25, kSqrt3 / 4.0}}};
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            double cx = a1x * x + a2x * y;
            double cy = a1y * x + a2y * y;
            for (std::size_t s = 0; s < 3; ++s) {
                L.positions.push_back(
                    {cx + basis[s][0], cy + basis[s][1], 0.0});
                L.sublattice[idx(x, y, s)] = static_cast<int>(s);
            }
        }
    }
    L.lattice_vectors[0] = {a1x, a1y, 0.0};
    L.lattice_vectors[1] = {a2x, a2y, 0.0};
    L.pbc = pbc;
    L.label = "kagome[" + std::to_string(Lx) + "x" + std::to_string(Ly) +
              (pbc ? " PBC]" : " OBC]");

    auto wrap = [&](long v, std::size_t L_) -> long {
        if (pbc) return ((v % static_cast<long>(L_)) + static_cast<long>(L_)) %
                        static_cast<long>(L_);
        if (v < 0 || v >= static_cast<long>(L_)) return -1;
        return v;
    };

    std::set<std::pair<std::size_t, std::size_t>> seen;
    for (std::size_t y = 0; y < Ly; ++y) {
        for (std::size_t x = 0; x < Lx; ++x) {
            std::size_t a = idx(x, y, 0);
            std::size_t b = idx(x, y, 1);
            std::size_t c = idx(x, y, 2);
            // Up triangle
            push_unique_bond(L.nn_bonds, seen, a, b);
            push_unique_bond(L.nn_bonds, seen, a, c);
            push_unique_bond(L.nn_bonds, seen, b, c);
            // Down triangle: A(x,y) connects across cells
            long nx_a = wrap(static_cast<long>(x) + 1, Lx);
            long ny_a = wrap(static_cast<long>(y), Ly);
            if (nx_a >= 0 && ny_a >= 0) {
                std::size_t a_right = idx(static_cast<std::size_t>(nx_a),
                                          static_cast<std::size_t>(ny_a), 0);
                push_unique_bond(L.nn_bonds, seen, b, a_right);
            }
            long nx_b = wrap(static_cast<long>(x), Lx);
            long ny_b = wrap(static_cast<long>(y) + 1, Ly);
            if (nx_b >= 0 && ny_b >= 0) {
                std::size_t a_up = idx(static_cast<std::size_t>(nx_b),
                                       static_cast<std::size_t>(ny_b), 0);
                push_unique_bond(L.nn_bonds, seen, c, a_up);
            }
            // C - B across cells
            long nx_c = wrap(static_cast<long>(x) - 1, Lx);
            long ny_c = wrap(static_cast<long>(y) + 1, Ly);
            if (nx_c >= 0 && ny_c >= 0) {
                std::size_t b_diag = idx(static_cast<std::size_t>(nx_c),
                                         static_cast<std::size_t>(ny_c), 1);
                push_unique_bond(L.nn_bonds, seen, c, b_diag);
            }
        }
    }
    return L;
}

Lattice pyrochlore(std::size_t Lx, std::size_t Ly, std::size_t Lz, bool pbc) {
    if (Lx == 0 || Ly == 0 || Lz == 0) {
        throw std::invalid_argument("pyrochlore: dims must be > 0");
    }
    Lattice L;
    L.num_sites = 4 * Lx * Ly * Lz;
    L.positions.reserve(L.num_sites);
    L.sublattice.assign(L.num_sites, 0);

    // FCC primitive vectors
    const std::array<Position, 3> a = {{
        {0.0, 0.5, 0.5},
        {0.5, 0.0, 0.5},
        {0.5, 0.5, 0.0},
    }};
    // Sublattice basis (corner of the up tetrahedron + 3 neighbours)
    const std::array<Position, 4> sub = {{
        {0.0, 0.0, 0.0},
        {0.0, 0.25, 0.25},
        {0.25, 0.0, 0.25},
        {0.25, 0.25, 0.0},
    }};
    auto idx = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t u) {
        return ((i * Ly + j) * Lz + k) * 4 + u;
    };
    for (std::size_t i = 0; i < Lx; ++i) {
        for (std::size_t j = 0; j < Ly; ++j) {
            for (std::size_t k = 0; k < Lz; ++k) {
                Position cell = {a[0][0] * i + a[1][0] * j + a[2][0] * k,
                                 a[0][1] * i + a[1][1] * j + a[2][1] * k,
                                 a[0][2] * i + a[1][2] * j + a[2][2] * k};
                for (std::size_t u = 0; u < 4; ++u) {
                    L.positions.push_back({cell[0] + sub[u][0],
                                           cell[1] + sub[u][1],
                                           cell[2] + sub[u][2]});
                    L.sublattice[idx(i, j, k, u)] = static_cast<int>(u);
                }
            }
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        L.lattice_vectors[axis] = a[axis];
    }
    L.pbc = pbc;
    L.label = "pyrochlore[" + std::to_string(Lx) + "x" + std::to_string(Ly) +
              "x" + std::to_string(Lz) + (pbc ? " PBC]" : " OBC]");

    auto wrap = [&](long v, std::size_t L_) -> long {
        if (pbc) return ((v % static_cast<long>(L_)) + static_cast<long>(L_)) %
                        static_cast<long>(L_);
        if (v < 0 || v >= static_cast<long>(L_)) return -1;
        return v;
    };
    std::set<std::pair<std::size_t, std::size_t>> seen;
    // Up tetrahedron: every (i,j,k) cell hosts the 6 NN bonds among its 4 sites.
    for (std::size_t i = 0; i < Lx; ++i) {
        for (std::size_t j = 0; j < Ly; ++j) {
            for (std::size_t k = 0; k < Lz; ++k) {
                std::size_t s0 = idx(i, j, k, 0);
                std::size_t s1 = idx(i, j, k, 1);
                std::size_t s2 = idx(i, j, k, 2);
                std::size_t s3 = idx(i, j, k, 3);
                push_unique_bond(L.nn_bonds, seen, s0, s1);
                push_unique_bond(L.nn_bonds, seen, s0, s2);
                push_unique_bond(L.nn_bonds, seen, s0, s3);
                push_unique_bond(L.nn_bonds, seen, s1, s2);
                push_unique_bond(L.nn_bonds, seen, s1, s3);
                push_unique_bond(L.nn_bonds, seen, s2, s3);
            }
        }
    }
    // Down tetrahedron: each sublattice site connects to its mirrored
    // neighbours one cell over along the corresponding FCC vector. The
    // canonical pyrochlore connectivity (matching helper_pyrochlore.py
    // get_neighbors):
    //   sub 0 connects to sub {1,2,3} in cell (i-1, j, k), (i, j-1, k),
    //                                          (i, j, k-1) respectively.
    //   sub 1 connects to sub 0 in cell (i+1, j, k), and sub {2,3} in
    //                cells (i+1, j-1, k), (i+1, j, k-1).
    //   ... and so on.
    auto add_down_bond = [&](std::size_t i, std::size_t j, std::size_t k,
                             int u_a, int u_b, long di, long dj, long dk) {
        long ni = wrap(static_cast<long>(i) + di, Lx);
        long nj = wrap(static_cast<long>(j) + dj, Ly);
        long nk = wrap(static_cast<long>(k) + dk, Lz);
        if (ni < 0 || nj < 0 || nk < 0) return;
        std::size_t a = idx(i, j, k, u_a);
        std::size_t b = idx(static_cast<std::size_t>(ni),
                            static_cast<std::size_t>(nj),
                            static_cast<std::size_t>(nk), u_b);
        push_unique_bond(L.nn_bonds, seen, a, b);
    };
    for (std::size_t i = 0; i < Lx; ++i) {
        for (std::size_t j = 0; j < Ly; ++j) {
            for (std::size_t k = 0; k < Lz; ++k) {
                add_down_bond(i, j, k, 0, 1, -1, 0, 0);
                add_down_bond(i, j, k, 0, 2, 0, -1, 0);
                add_down_bond(i, j, k, 0, 3, 0, 0, -1);
                add_down_bond(i, j, k, 1, 2, 0, -1, 1);
                add_down_bond(i, j, k, 1, 3, 0, 0, 0);  // already present in up tet
                add_down_bond(i, j, k, 2, 3, 0, 0, 0);  // already present in up tet
            }
        }
    }
    return L;
}

Lattice from_neighbor_lists(
    const std::vector<Position>& positions,
    const std::vector<std::pair<std::size_t, std::size_t>>& nn_pairs,
    const std::vector<int>& sublattice) {
    Lattice L;
    L.num_sites = positions.size();
    L.positions = positions;
    if (sublattice.empty()) {
        L.sublattice.assign(L.num_sites, 0);
    } else {
        if (sublattice.size() != L.num_sites) {
            throw std::invalid_argument(
                "from_neighbor_lists: sublattice size must equal positions size");
        }
        L.sublattice = sublattice;
    }
    std::set<std::pair<std::size_t, std::size_t>> seen;
    L.nn_bonds.reserve(nn_pairs.size());
    for (auto [i, j] : nn_pairs) {
        if (i >= L.num_sites || j >= L.num_sites) {
            throw std::out_of_range(
                "from_neighbor_lists: bond endpoint out of range");
        }
        push_unique_bond(L.nn_bonds, seen, i, j);
    }
    L.pbc = false;
    L.label = "custom[N=" + std::to_string(L.num_sites) + "]";
    return L;
}

Lattice from_cluster_file(const std::string& path) {
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("from_cluster_file: cannot open " + path);
    }
    std::string line;
    enum class Section { None, Positions, Edges } sect = Section::None;
    std::vector<Position> positions;
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    while (std::getline(in, line)) {
        // Strip leading whitespace
        std::size_t p = 0;
        while (p < line.size() && std::isspace(static_cast<unsigned char>(line[p]))) ++p;
        if (p >= line.size() || line[p] == '#') continue;
        std::string trimmed = line.substr(p);
        if (trimmed.rfind("positions", 0) == 0 ||
            trimmed.rfind("Positions", 0) == 0 ||
            trimmed.rfind("POSITIONS", 0) == 0) {
            sect = Section::Positions;
            continue;
        }
        if (trimmed.rfind("edges", 0) == 0 || trimmed.rfind("Edges", 0) == 0 ||
            trimmed.rfind("EDGES", 0) == 0 || trimmed.rfind("bonds", 0) == 0 ||
            trimmed.rfind("Bonds", 0) == 0) {
            sect = Section::Edges;
            continue;
        }
        std::istringstream iss(trimmed);
        if (sect == Section::Positions) {
            double x, y, z = 0.0;
            std::size_t id_unused;
            // accept either "x y z" or "id x y z"
            if (!(iss >> x)) continue;
            double maybe_y;
            if (iss >> maybe_y) {
                double maybe_z;
                if (iss >> maybe_z) {
                    // 3 numbers => x y z
                    positions.push_back({x, maybe_y, maybe_z});
                } else {
                    // 2 numbers => x y, z = 0
                    positions.push_back({x, maybe_y, 0.0});
                }
            } else {
                // Single number was an id; parse the rest
                id_unused = static_cast<std::size_t>(x);
                (void)id_unused;
                double xx, yy, zz = 0.0;
                std::istringstream iss2(trimmed);
                std::size_t id_;
                iss2 >> id_ >> xx >> yy;
                if (iss2 >> zz) {
                    positions.push_back({xx, yy, zz});
                } else {
                    positions.push_back({xx, yy, 0.0});
                }
            }
        } else if (sect == Section::Edges) {
            std::size_t a, b;
            if (iss >> a >> b) {
                edges.emplace_back(a, b);
            }
        }
    }
    return from_neighbor_lists(positions, edges);
}

}  // namespace lattice

}  // namespace ed::input
