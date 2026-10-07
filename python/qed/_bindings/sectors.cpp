// =============================================================================
// python/qed/_bindings/sectors.cpp -- _core.sectors: the symmetry sectors of a
// Hamiltonian and the lowest-level eigensolve over them (include/ed/sectors/sectors.h).
// =============================================================================

#include "bindings.h"

#include <ed/sectors/sectors.h>
#include <ed/sectors/thermal.h>
#include <ed/sectors/dynamics.h>
#include <ed/sectors/expect.h>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <map>
#include <string>

namespace py = pybind11;
namespace sec = ed::sectors;

namespace {

py::array_t<std::complex<double>> to_array(std::vector<std::complex<double>> v) {
    auto* heap = new std::vector<std::complex<double>>(std::move(v));
    py::capsule owner(heap, [](void* p) { delete static_cast<std::vector<std::complex<double>>*>(p); });
    return py::array_t<std::complex<double>>({heap->size()}, {sizeof(std::complex<double>)}, heap->data(), owner);
}

py::array_t<double> to_real_array(std::vector<double> v) {
    auto* heap = new std::vector<double>(std::move(v));
    py::capsule owner(heap, [](void* p) { delete static_cast<std::vector<double>*>(p); });
    return py::array_t<double>({heap->size()}, {sizeof(double)}, heap->data(), owner);
}

// ---- EigsResult <-> flat arrays (EigResult.save / qed.load_eigs) ---------------------------
// Everything a result needs to be used again: its levels with their block labels, its vectors
// in the sector basis they were solved in, each basis once (representatives, norms, characters,
// group action), and the Spec. Derived tables (the permutation lookup) are rebuilt on load.

template <class T> py::array_t<T> arr(const std::vector<T>& v) {
    return py::array_t<T>(static_cast<py::ssize_t>(v.size()), v.data());
}

template <class T> std::vector<T> vec(const py::dict& d, const char* key) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(d[key]);
    if (!a) throw std::invalid_argument(std::string("load_eigs: bad array ") + key);
    return std::vector<T>(a.data(), a.data() + a.size());
}

py::object antiunitary_name(sec::Antiunitary a) {
    if (a == sec::Antiunitary::K) return py::str("K");
    if (a == sec::Antiunitary::Theta) return py::str("theta");
    return py::none();
}

py::dict eigs_to_arrays(const sec::EigsResult& r, const sec::Spec& s) {
    py::dict d;
    const std::size_t nl = r.levels.size();
    std::vector<double> energy(nl);
    std::vector<std::int64_t> mirror(nl), fold(nl), mult(nl), vector(nl), tag(nl * 11);
    for (std::size_t i = 0; i < nl; ++i) {
        const auto& L = r.levels[i];
        const auto& t = L.tag;
        energy[i] = L.energy;
        mirror[i] = L.mirror;
        fold[i] = static_cast<std::int64_t>(L.fold);
        mult[i] = static_cast<std::int64_t>(L.multiplicity);
        vector[i] = L.vector;
        const std::int64_t row[11] = {t.n_up,
                                      t.sz_parity,
                                      t.k0,
                                      t.k_raw,
                                      t.flip_parity,
                                      t.irrep,
                                      t.irrep_dim,
                                      t.star_size,
                                      t.tr_folded ? 1 : 0,
                                      static_cast<std::int64_t>(t.dim),
                                      static_cast<std::int64_t>(t.multiplicity)};
        std::copy(row, row + 11, tag.begin() + static_cast<std::ptrdiff_t>(11 * i));
    }
    d["level_energy"] = arr(energy);
    d["level_mirror"] = arr(mirror);
    d["level_fold"] = arr(fold);
    d["level_multiplicity"] = arr(mult);
    d["level_vector"] = arr(vector);
    d["level_tag"] = arr(tag);
    // Physical labels, ragged per level: momentum characters, co-group (residue, character).
    std::vector<std::int64_t> moff{0}, ioff{0}, ielem;
    std::vector<std::complex<double>> mchi, ichi;
    for (const auto& L : r.levels) {
        mchi.insert(mchi.end(), L.momentum.begin(), L.momentum.end());
        moff.push_back(static_cast<std::int64_t>(mchi.size()));
        for (const auto& [e, c] : L.irrep_characters) {
            ielem.push_back(e);
            ichi.push_back(c);
        }
        ioff.push_back(static_cast<std::int64_t>(ielem.size()));
    }
    d["level_momentum"] = arr(mchi);
    d["level_momentum_offset"] = arr(moff);
    d["level_irrep_elems"] = arr(ielem);
    d["level_irrep_chars"] = arr(ichi);
    d["level_irrep_offset"] = arr(ioff);

    // Vectors, with each distinct basis stored once.
    std::map<const void*, std::int64_t> basis_id;
    std::vector<std::int64_t> vbasis, voffset{0};
    std::vector<std::complex<double>> amps;
    for (const auto& v : r.vectors) {
        auto [it, fresh] = basis_id.emplace(v.basis.get(), static_cast<std::int64_t>(basis_id.size()));
        if (fresh) {
            const auto& b = *v.basis;
            const std::string p = "basis" + std::to_string(it->second) + "_";
            d[py::str(p + "reps")] = arr(b.reps);
            d[py::str(p + "inv_norms")] = arr(b.inv_norms);
            d[py::str(p + "characters")] = arr(b.characters);
            d[py::str(p + "perms")] = arr(b.perms_flat);
            d[py::str(p + "flip_masks")] = arr(b.flip_masks);
            d[py::str(p + "shape")] = arr(std::vector<std::int64_t>{b.group_size, b.n_sites, b.n_up, b.irrep_dim});
            // which members represent the orbits: the sublattice key order's fingerprint, 0 the plain order
            d[py::str(p + "sublattice")] = arr(std::vector<std::uint64_t>{b.slc ? b.slc->fingerprint() : 0});
            if (b.irrep_dim > 1) { // a sector of a d > 1 irrep (rep_sector.h)
                d[py::str(p + "irrep_D")] = arr(b.irrep_D);
                d[py::str(p + "class_rank")] = arr(std::vector<std::int64_t>(b.class_rank.begin(), b.class_rank.end()));
                d[py::str(p + "class_C")] = arr(b.class_C);
                d[py::str(p + "rep_class")] = arr(std::vector<std::int64_t>(b.rep_class.begin(), b.rep_class.end()));
                d[py::str(p + "state_offset")] = arr(b.state_offset);
            }
        }
        vbasis.push_back(it->second);
        amps.insert(amps.end(), v.amplitudes.begin(), v.amplitudes.end());
        voffset.push_back(static_cast<std::int64_t>(amps.size()));
    }
    d["n_bases"] = arr(std::vector<std::int64_t>{static_cast<std::int64_t>(basis_id.size())});
    d["vector_basis"] = arr(vbasis);
    d["vector_offset"] = arr(voffset);
    d["vector_amplitudes"] = arr(amps);

    auto perms = [](const std::vector<sec::Perm>& ps) {
        std::vector<std::int64_t> flat;
        for (const auto& p : ps) flat.insert(flat.end(), p.begin(), p.end());
        return flat;
    };
    d["spec_abelian"] = arr(perms(s.abelian));
    d["spec_residues"] = arr(perms(s.residues));
    d["spec_only_k0"] = arr(std::vector<std::int64_t>(s.only_k0.begin(), s.only_k0.end()));
    d["spec_only_irrep"] = arr(std::vector<std::int64_t>(s.only_irrep.begin(), s.only_irrep.end()));
    d["spec_scalars"] = arr(std::vector<std::int64_t>{s.n_up, s.sz_parity, s.use_sz ? 1 : 0, s.spin_flip,
                                                      s.time_reversal, s.two_S, r.n_sites});
    d["result_scalars"] = arr(std::vector<std::int64_t>{
        static_cast<std::int64_t>(r.total_dim), static_cast<std::int64_t>(r.partial_blocks), r.complete ? 1 : 0,
        r.flip_engaged ? 1 : 0, static_cast<std::int64_t>(r.time_reversal), static_cast<std::int64_t>(r.device_blocks),
        static_cast<std::int64_t>(r.pruned_blocks)});
    return d;
}

// Every index and offset of a saved result is checked before it is used: a damaged or
// hand-edited file raises ValueError instead of reading out of bounds.
py::tuple eigs_from_arrays(const py::dict& d) {
    auto fail = [](const std::string& what) { throw std::invalid_argument("load_eigs: " + what); };
    // offsets into an array of `size` entries: n + 1 non-decreasing values in 0..size
    auto check_offsets = [&](const std::vector<std::int64_t>& off, std::size_t n, std::size_t size, const char* name) {
        if (off.size() != n + 1) fail(std::string(name) + " has the wrong length");
        for (std::size_t i = 0; i <= n; ++i)
            if (off[i] < 0 || static_cast<std::size_t>(off[i]) > size || (i > 0 && off[i] < off[i - 1]))
                fail(std::string(name) + " is not a list of offsets into its array");
    };
    auto check_perms = [&](const auto& flat, std::size_t n_perm, int n, const std::string& name) {
        if (n < 1 || flat.size() != n_perm * static_cast<std::size_t>(n)) fail(name + " has the wrong shape");
        for (std::size_t p = 0; p < n_perm; ++p) {
            std::vector<char> seen(static_cast<std::size_t>(n), 0);
            for (int i = 0; i < n; ++i) {
                const auto x = flat[p * static_cast<std::size_t>(n) + static_cast<std::size_t>(i)];
                if (x < 0 || x >= n || seen[static_cast<std::size_t>(x)])
                    fail(name + " holds an entry that is not a permutation of the sites");
                seen[static_cast<std::size_t>(x)] = 1;
            }
        }
    };
    sec::EigsResult r;
    const auto energy = vec<double>(d, "level_energy");
    const auto mirror = vec<std::int64_t>(d, "level_mirror");
    const auto mult = vec<std::int64_t>(d, "level_multiplicity");
    const auto vector = vec<std::int64_t>(d, "level_vector");
    const auto tag = vec<std::int64_t>(d, "level_tag");
    if (tag.size() != 11 * energy.size()) throw std::invalid_argument("load_eigs: level_tag has the wrong shape");
    if (mirror.size() != energy.size() || mult.size() != energy.size() || vector.size() != energy.size())
        fail("the level arrays differ in length");
    for (std::size_t i = 0; i < energy.size(); ++i) {
        sec::Level L;
        const std::int64_t* t = tag.data() + 11 * i;
        L.energy = energy[i];
        L.mirror = static_cast<int>(mirror[i]);
        L.multiplicity = static_cast<std::uint64_t>(mult[i]);
        L.vector = static_cast<int>(vector[i]);
        L.tag.n_up = static_cast<int>(t[0]);
        L.tag.sz_parity = static_cast<int>(t[1]);
        L.tag.k0 = static_cast<int>(t[2]);
        L.tag.k_raw = static_cast<int>(t[3]);
        L.tag.flip_parity = static_cast<int>(t[4]);
        L.tag.irrep = static_cast<int>(t[5]);
        L.tag.irrep_dim = static_cast<int>(t[6]);
        L.tag.star_size = static_cast<int>(t[7]);
        L.tag.tr_folded = t[8] != 0;
        L.tag.dim = static_cast<std::uint64_t>(t[9]);
        L.tag.multiplicity = static_cast<std::uint64_t>(t[10]);
        L.fold = L.tag.tr_folded ? sec::Antiunitary::K : sec::Antiunitary::None; // files without level_fold
        r.levels.push_back(L);
    }
    // 0 none, 1 K, 2 Theta: the antiunitary pairing of each level.
    auto antiunitary = [&fail](std::int64_t a, const char* what) {
        if (a < 0 || a > 2) fail(std::string(what) + " names no antiunitary map");
        return static_cast<sec::Antiunitary>(a);
    };
    if (d.contains("level_fold")) {
        const auto fold = vec<std::int64_t>(d, "level_fold");
        if (fold.size() != energy.size()) fail("the level arrays differ in length");
        for (std::size_t i = 0; i < fold.size(); ++i) r.levels[i].fold = antiunitary(fold[i], "level_fold");
    }
    if (d.contains("level_momentum")) {
        const auto mchi = vec<std::complex<double>>(d, "level_momentum");
        const auto moff = vec<std::int64_t>(d, "level_momentum_offset");
        const auto ielem = vec<std::int64_t>(d, "level_irrep_elems");
        const auto ichi = vec<std::complex<double>>(d, "level_irrep_chars");
        const auto ioff = vec<std::int64_t>(d, "level_irrep_offset");
        if (moff.size() != r.levels.size() + 1 || ioff.size() != r.levels.size() + 1)
            throw std::invalid_argument("load_eigs: level labels have the wrong shape");
        check_offsets(moff, r.levels.size(), mchi.size(), "level_momentum_offset");
        check_offsets(ioff, r.levels.size(), std::min(ielem.size(), ichi.size()), "level_irrep_offset");
        for (std::size_t i = 0; i < r.levels.size(); ++i) {
            auto& L = r.levels[i];
            L.momentum.assign(mchi.begin() + moff[i], mchi.begin() + moff[i + 1]);
            for (auto j = ioff[i]; j < ioff[i + 1]; ++j)
                L.irrep_characters.emplace_back(static_cast<int>(ielem[static_cast<std::size_t>(j)]),
                                                ichi[static_cast<std::size_t>(j)]);
        }
    }
    std::vector<std::shared_ptr<const ed::symmetry::RepSectorData>> bases;
    const auto nb = vec<std::int64_t>(d, "n_bases").at(0);
    for (std::int64_t b = 0; b < nb; ++b) {
        const std::string p = "basis" + std::to_string(b) + "_";
        auto rd = std::make_shared<ed::symmetry::RepSectorData>();
        rd->reps = vec<std::uint64_t>(d, (p + "reps").c_str());
        rd->inv_norms = vec<double>(d, (p + "inv_norms").c_str());
        rd->characters = vec<std::complex<double>>(d, (p + "characters").c_str());
        rd->perms_flat = vec<int>(d, (p + "perms").c_str());
        rd->flip_masks = vec<std::uint64_t>(d, (p + "flip_masks").c_str());
        const auto shape = vec<std::int64_t>(d, (p + "shape").c_str());
        if (shape.size() < 3 || shape[0] < 1 || shape[1] < 1 || shape[1] > 63 || shape[2] < -1 || shape[2] > shape[1])
            fail("basis " + std::to_string(b) + " has an impossible shape");
        rd->group_size = static_cast<int>(shape.at(0));
        rd->n_sites = static_cast<int>(shape.at(1));
        rd->n_up = static_cast<int>(shape.at(2));
        rd->irrep_dim = shape.size() > 3 ? static_cast<int>(shape[3]) : 1; // files before P6.3: 1
        if (rd->irrep_dim < 1 || rd->irrep_dim > 8)
            fail("basis " + std::to_string(b) + " has an impossible irrep dimension");
        check_perms(rd->perms_flat, static_cast<std::size_t>(rd->group_size), rd->n_sites, p + "perms");
        if (rd->irrep_dim > 1) {
            const std::size_t dd = static_cast<std::size_t>(rd->irrep_dim * rd->irrep_dim);
            rd->irrep_D = vec<std::complex<double>>(d, (p + "irrep_D").c_str());
            rd->class_C = vec<std::complex<double>>(d, (p + "class_C").c_str());
            rd->state_offset = vec<std::uint64_t>(d, (p + "state_offset").c_str());
            for (const auto x : vec<std::int64_t>(d, (p + "class_rank").c_str())) {
                if (x < 0 || x > rd->irrep_dim) fail("basis " + std::to_string(b) + " has an impossible class rank");
                rd->class_rank.push_back(static_cast<std::uint8_t>(x));
            }
            for (const auto x : vec<std::int64_t>(d, (p + "rep_class").c_str())) {
                if (x < 0 || static_cast<std::size_t>(x) >= rd->class_rank.size())
                    fail("basis " + std::to_string(b) + " names a stabiliser class that is not in the file");
                rd->rep_class.push_back(static_cast<std::uint16_t>(x));
            }
            bool offsets =
                rd->state_offset.size() == rd->reps.size() + 1 && !rd->state_offset.empty() && rd->state_offset[0] == 0;
            for (std::size_t r = 0; offsets && r < rd->reps.size(); ++r)
                offsets = rd->state_offset[r + 1] == rd->state_offset[r] + rd->class_rank[rd->rep_class[r]];
            if (rd->irrep_D.size() != static_cast<std::size_t>(rd->group_size) * dd
                || rd->class_C.size() != rd->class_rank.size() * dd || rd->rep_class.size() != rd->reps.size()
                || !offsets)
                fail("basis " + std::to_string(b) + " has irrep arrays of the wrong length");
        }
        if (rd->characters.size() != static_cast<std::size_t>(rd->group_size)
            || rd->inv_norms.size() != (rd->irrep_dim == 1 ? rd->reps.size() : 0)
            || (!rd->flip_masks.empty() && rd->flip_masks.size() != static_cast<std::size_t>(rd->group_size)))
            fail("basis " + std::to_string(b) + " has arrays of the wrong length");
        if (!rd->usable()) fail("basis " + std::to_string(b) + " is not a usable sector");
        rd->build_perm_lut();
        // Which members represent the orbits: the sublattice key order the sector was computed with
        // (<ed/basis/sublattice_code.h>), rebuilt from the group whatever ED_SYM_SUBLATTICE says now;
        // a file without the field predates the codes and holds the plain order.
        const std::string slc_key = p + "sublattice";
        const std::uint64_t fp = d.contains(slc_key.c_str()) ? vec<std::uint64_t>(d, slc_key.c_str()).at(0) : 0;
        if (fp != 0) {
            rd->slc = ed::symmetry::SublatticeCode::of(rd->perms_flat.data(),
                                                       rd->flip_masks.empty() ? nullptr : rd->flip_masks.data(),
                                                       rd->group_size, rd->n_sites, true);
            if (!rd->slc || rd->slc->fingerprint() != fp)
                fail("basis " + std::to_string(b)
                     + " was computed with a sublattice key order this build does "
                       "not reproduce");
        }
        { // the stored representatives are the ones that rule finds (a sample)
            const auto pol = rd->make_policy();
            const std::size_t n = rd->reps.size(), samples = std::min<std::size_t>(n, 64);
            for (std::size_t i = 0; i < samples; ++i) {
                const std::uint64_t r = rd->reps[samples == 1 ? 0 : i * (n - 1) / (samples - 1)];
                if (pol.representative(r) != r)
                    fail("basis " + std::to_string(b) + " holds a state that is not its orbit's representative");
            }
        }
        bases.push_back(std::move(rd));
    }
    const auto vbasis = vec<std::int64_t>(d, "vector_basis");
    const auto voffset = vec<std::int64_t>(d, "vector_offset");
    const auto amps = vec<std::complex<double>>(d, "vector_amplitudes");
    check_offsets(voffset, vbasis.size(), amps.size(), "vector_offset");
    for (std::size_t i = 0; i < vbasis.size(); ++i) {
        if (vbasis[i] < 0 || vbasis[i] >= nb) fail("vector_basis names a basis that is not in the file");
        sec::BlockVector v;
        v.basis = bases.at(static_cast<std::size_t>(vbasis[i]));
        v.amplitudes.assign(amps.begin() + voffset[i], amps.begin() + voffset[i + 1]);
        if (v.amplitudes.size() != v.basis->states())
            throw std::invalid_argument("load_eigs: a vector does not match its basis");
        r.vectors.push_back(std::move(v));
    }
    for (const auto& L : r.levels)
        if (L.vector < -1 || L.vector >= static_cast<int>(r.vectors.size()))
            fail("level_vector names a vector that is not in the file");
    const auto sc = vec<std::int64_t>(d, "result_scalars");
    r.total_dim = static_cast<std::uint64_t>(sc.at(0));
    r.partial_blocks = static_cast<std::size_t>(sc.at(1));
    r.complete = sc.at(2) != 0;
    r.flip_engaged = sc.at(3) != 0;
    r.time_reversal = antiunitary(sc.at(4), "result_scalars");
    r.device_blocks = static_cast<std::size_t>(sc.at(5));
    r.pruned_blocks = static_cast<std::size_t>(sc.at(6));

    sec::Spec s;
    const auto ss = vec<std::int64_t>(d, "spec_scalars");
    const int n = static_cast<int>(ss.at(6));
    if (n < 1 || n > 63) fail("the number of sites is outside 1..63");
    for (const char* key : {"spec_abelian", "spec_residues"}) {
        const auto flat = vec<std::int64_t>(d, key);
        check_perms(flat, flat.size() / static_cast<std::size_t>(n), n, key);
    }
    auto perms = [n](const std::vector<std::int64_t>& flat) {
        std::vector<sec::Perm> out;
        for (std::size_t i = 0; n > 0 && i + static_cast<std::size_t>(n) <= flat.size();
             i += static_cast<std::size_t>(n))
            out.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(i),
                             flat.begin() + static_cast<std::ptrdiff_t>(i) + n);
        return out;
    };
    s.abelian = perms(vec<std::int64_t>(d, "spec_abelian"));
    s.residues = perms(vec<std::int64_t>(d, "spec_residues"));
    for (auto k : vec<std::int64_t>(d, "spec_only_k0")) s.only_k0.push_back(static_cast<int>(k));
    for (auto k : vec<std::int64_t>(d, "spec_only_irrep")) s.only_irrep.push_back(static_cast<int>(k));
    s.n_up = static_cast<int>(ss.at(0));
    s.sz_parity = static_cast<int>(ss.at(1));
    s.use_sz = ss.at(2) != 0;
    s.spin_flip = static_cast<int>(ss.at(3));
    s.time_reversal = static_cast<int>(ss.at(4));
    s.two_S = static_cast<int>(ss.at(5));
    if (s.n_up < -1 || s.n_up > n || s.sz_parity < -1 || s.sz_parity > 1 || s.two_S < -1 || s.two_S > n)
        fail("the saved symmetry labels are out of range");
    r.n_sites = n;
    return py::make_tuple(std::move(r), std::move(s));
}

// Where a verb's solves ran (sec::Placement), as a dict.
py::dict placement_to_py(const sec::Placement& p) {
    py::dict d;
    d["device_krylov"] = p.device_krylov;
    d["device_dense"] = p.device_dense;
    d["host_krylov"] = p.host_krylov;
    d["host_dense"] = p.host_dense;
    return d;
}

// One dict per solved block (EigsResult::block_stats).
py::list block_stats_to_py(const std::vector<sec::BlockStats>& v) {
    py::list out;
    for (const auto& b : v) {
        py::dict d;
        d["k0"] = b.k0;
        d["irrep"] = b.irrep;
        d["flip_parity"] = b.flip_parity;
        d["n_up"] = b.n_up;
        d["dim"] = b.dim;
        d["kind"] = b.kind;
        d["lane"] = b.lane;
        d["context_orbit_s"] = b.context_orbit_s;
        d["star_orbit_s"] = b.star_orbit_s;
        d["star_build_s"] = b.star_build_s;
        d["build_s"] = b.build_s;
        d["nnz"] = b.nnz;
        d["csr_bytes"] = b.csr_bytes;
        d["applies"] = b.applies;
        d["apply_s"] = b.apply_s;
        d["other_s"] = b.other_s;
        d["solve_s"] = b.solve_s;
        out.append(std::move(d));
    }
    return out;
}
} // namespace

void bind_sectors(py::module_& m) {
    auto s = m.def_submodule("sectors", "Symmetry sectors and the lowest-level eigensolve over them.");

    py::class_<sec::Spec>(s, "Spec")
        .def(py::init<>())
        .def_readwrite("abelian", &sec::Spec::abelian)
        .def_readwrite("residues", &sec::Spec::residues)
        .def_readwrite("n_up", &sec::Spec::n_up)
        .def_readwrite("sz_parity", &sec::Spec::sz_parity)
        .def_readwrite("use_sz", &sec::Spec::use_sz)
        .def_readwrite("spin_flip", &sec::Spec::spin_flip)
        .def_readwrite("time_reversal", &sec::Spec::time_reversal)
        .def_readwrite("two_S", &sec::Spec::two_S)
        .def_readwrite("only_k0", &sec::Spec::only_k0)
        .def_readwrite("only_irrep", &sec::Spec::only_irrep)
        .def_readwrite("only_momentum", &sec::Spec::only_momentum)
        .def_readwrite("only_irrep_chars", &sec::Spec::only_irrep_chars);

    py::enum_<sec::Device>(s, "Device")
        .value("Cpu", sec::Device::Cpu)
        .value("Gpu", sec::Device::Gpu)
        .value("Auto", sec::Device::Auto);

    py::enum_<sec::SzContent>(s, "SzContent")
        .value("U1", sec::SzContent::U1)
        .value("Parity", sec::SzContent::Parity)
        .value("None_", sec::SzContent::None);
    s.def("sz_content", &sec::sz_content, py::arg("H"));

    py::class_<sec::Level>(s, "Level")
        .def_readonly("energy", &sec::Level::energy)
        .def_readonly("multiplicity", &sec::Level::multiplicity)
        .def_readonly("mirror", &sec::Level::mirror)
        .def_readonly("vector", &sec::Level::vector)
        .def_readonly("momentum", &sec::Level::momentum)
        .def_readonly("irrep_characters", &sec::Level::irrep_characters)
        .def_property_readonly("n_up", [](const sec::Level& l) { return l.tag.n_up; })
        .def_property_readonly("sz_parity", [](const sec::Level& l) { return l.tag.sz_parity; })
        .def_property_readonly("k0", [](const sec::Level& l) { return l.tag.k0; })
        .def_property_readonly("k_raw", [](const sec::Level& l) { return l.tag.k_raw; })
        .def_property_readonly("flip_parity", [](const sec::Level& l) { return l.tag.flip_parity; })
        .def_property_readonly("irrep", [](const sec::Level& l) { return l.tag.irrep; })
        .def_property_readonly("irrep_dim", [](const sec::Level& l) { return l.tag.irrep_dim; })
        .def_property_readonly("star_size", [](const sec::Level& l) { return l.tag.star_size; })
        .def_property_readonly("tr_folded", [](const sec::Level& l) { return l.tag.tr_folded; })
        .def_property_readonly(
            "fold", [](const sec::Level& l) { return antiunitary_name(l.fold); },
            "The antiunitary map pairing the level with states its block does not hold: "
            "'K' (complex conjugation), 'theta' (time reversal; also the map of a mirror "
            "in Sz sector N - n_up), or None.")
        .def_property_readonly("block_dim", [](const sec::Level& l) { return l.tag.dim; })
        .def("__repr__", [](const sec::Level& l) {
            return "Level(E=" + std::to_string(l.energy) + ", mult=" + std::to_string(l.multiplicity)
                   + ", n_up=" + std::to_string(l.tag.n_up) + ", k_raw=" + std::to_string(l.tag.k_raw)
                   + ", irrep=" + std::to_string(l.tag.irrep) + ")";
        });

    py::class_<sec::EigsResult>(s, "EigsResult")
        .def_readonly("levels", &sec::EigsResult::levels)
        .def_readonly("n_sites", &sec::EigsResult::n_sites)
        .def_readonly("complete", &sec::EigsResult::complete)
        .def_property_readonly(
            "time_reversal", [](const sec::EigsResult& r) { return antiunitary_name(r.time_reversal); },
            "The antiunitary map that folded any level: 'K', 'theta' or None.")
        .def_readonly("device_blocks", &sec::EigsResult::device_blocks)
        .def_property_readonly("placement", [](const sec::EigsResult& r) { return placement_to_py(r.placement); })
        .def_readonly("pruned_blocks", &sec::EigsResult::pruned_blocks)
        .def_readonly("diagnostics", &sec::EigsResult::diagnostics)
        .def_property_readonly(
            "block_stats", [](const sec::EigsResult& r) { return block_stats_to_py(r.block_stats); },
            "Per solved block: dim, lane, phase seconds, nnz, applies (one dict each).")
        .def("energies", &sec::EigsResult::energies, py::arg("k"))
        .def(
            "multiplet",
            [](const sec::EigsResult& r, const sec::Spec& spec, int level, int n_up, std::size_t max_vectors) {
                const auto& L = r.levels.at(static_cast<std::size_t>(level));
                if (L.vector < 0) throw std::invalid_argument("multiplet: the level carries no vector");
                std::vector<std::vector<std::complex<double>>> vs;
                {
                    py::gil_scoped_release nogil;
                    vs = sec::multiplet(spec, r.n_sites, L, r.vectors[static_cast<std::size_t>(L.vector)], n_up,
                                        max_vectors);
                }
                py::list out;
                for (auto& v : vs) out.append(to_array(std::move(v)));
                return out;
            },
            py::arg("spec"), py::arg("level"), py::arg("n_up") = -1, py::arg("max_vectors") = 0,
            "The level's degenerate multiplet expanded into Sz sector n_up (n_up < 0: full space); "
            "at most max_vectors of its vectors when that is > 0.")
        .def(
            "evaluate",
            [](const sec::EigsResult& r, const sec::Spec& spec, const std::vector<const ::Operator*>& singles,
               const std::vector<std::pair<std::vector<const ::Operator*>, std::vector<const ::Operator*>>>& pairs) {
                std::vector<sec::PairRequest> reqs;
                reqs.reserve(pairs.size());
                std::size_t width = singles.size();
                for (const auto& [A, B] : pairs) {
                    reqs.push_back({A, B});
                    width += A.size() * B.size();
                }
                std::vector<std::vector<sec::Complex>> c;
                {
                    py::gil_scoped_release nogil;
                    c = sec::evaluate(r, spec, singles, reqs);
                }
                py::array_t<std::complex<double>> out(
                    {static_cast<py::ssize_t>(c.size()), static_cast<py::ssize_t>(width)});
                auto* dst = out.mutable_data();
                for (const auto& level : c) dst = std::copy(level.begin(), level.end(), dst);
                return out;
            },
            py::arg("spec"), py::arg("singles"), py::arg("pairs"),
            "One multiplet-averaged sweep per level: complex array [level, x] listing <X> for X in singles, "
            "then <A_a^dag B_b> at a * len(B) + b for each (A, B) in pairs.")
        .def(
            "matrix_element",
            [](const sec::EigsResult& r, const ::Operator& O, std::size_t i, std::size_t j) {
                py::gil_scoped_release nogil;
                return sec::matrix_element(r, O, i, j);
            },
            py::arg("O"), py::arg("i"), py::arg("j"), "<v_i|O|v_j> between the vectors of levels i and j.");

    py::class_<sec::SpectrumResult>(s, "SpectrumResult")
        .def_readonly("levels", &sec::SpectrumResult::levels)
        .def_readonly("device_blocks", &sec::SpectrumResult::device_blocks)
        .def_property_readonly("time_reversal",
                               [](const sec::SpectrumResult& r) { return antiunitary_name(r.time_reversal); })
        .def_property_readonly("placement", [](const sec::SpectrumResult& r) { return placement_to_py(r.placement); })
        .def_readonly("diagnostics", &sec::SpectrumResult::diagnostics)
        .def("expanded", [](const sec::SpectrumResult& r) { return to_real_array(r.expanded()); });

    s.def(
        "spectrum",
        [](const ::Operator& H, const sec::Spec& spec, sec::Device device) {
            py::gil_scoped_release nogil;
            return sec::spectrum(H, spec, device);
        },
        py::arg("H"), py::arg("spec"), py::arg("device") = sec::Device::Cpu,
        "The complete spectrum of H, every symmetry block diagonalised densely.");

    py::enum_<sec::ThermalSpec::Method>(s, "ThermalMethod")
        .value("Exact", sec::ThermalSpec::Method::Exact)
        .value("FTLM", sec::ThermalSpec::Method::FTLM)
        .value("mTPQ", sec::ThermalSpec::Method::mTPQ);

    py::class_<sec::ThermalSpec>(s, "ThermalSpec")
        .def(py::init<>())
        .def_readwrite("method", &sec::ThermalSpec::method)
        .def_readwrite("temperatures", &sec::ThermalSpec::temperatures)
        .def_readwrite("samples", &sec::ThermalSpec::samples)
        .def_readwrite("krylov", &sec::ThermalSpec::krylov)
        .def_readwrite("steps", &sec::ThermalSpec::steps)
        .def_readwrite("exact_states", &sec::ThermalSpec::exact_states)
        .def_readwrite("dense_max_dim", &sec::ThermalSpec::dense_max_dim)
        .def_readwrite("seed", &sec::ThermalSpec::seed)
        .def_readwrite("device", &sec::ThermalSpec::device)
        .def_readwrite("observables", &sec::ThermalSpec::observables)
        .def_property(
            "observable_pairs",
            [](const sec::ThermalSpec& t) {
                py::list out;
                for (const auto& p : t.observable_pairs) out.append(py::make_tuple(p.A, p.B));
                return out;
            },
            [](sec::ThermalSpec& t,
               const std::vector<std::pair<std::vector<const ::Operator*>, std::vector<const ::Operator*>>>& pairs) {
                t.observable_pairs.clear();
                for (const auto& [A, B] : pairs) t.observable_pairs.push_back({A, B});
            },
            "Equal-time pair requests [(A, B), ...]: <A_a^dag B_b>(T) after the observables in O, a-major.");

    py::class_<sec::ThermalCurves>(s, "ThermalCurves")
        .def_readonly("T", &sec::ThermalCurves::T)
        .def_readonly("lnZ", &sec::ThermalCurves::lnZ)
        .def_readonly("E", &sec::ThermalCurves::E)
        .def_readonly("C", &sec::ThermalCurves::C)
        .def_readonly("S", &sec::ThermalCurves::S)
        .def_readonly("F", &sec::ThermalCurves::F)
        .def_readonly("M", &sec::ThermalCurves::M)
        .def_readonly("chi", &sec::ThermalCurves::chi)
        .def_readonly("O", &sec::ThermalCurves::O)
        .def_readonly("e0", &sec::ThermalCurves::e0)
        .def_readonly("blocks", &sec::ThermalCurves::blocks)
        .def_readonly("device_blocks", &sec::ThermalCurves::device_blocks)
        .def_property_readonly("placement", [](const sec::ThermalCurves& r) { return placement_to_py(r.placement); })
        .def_readonly("diagnostics", &sec::ThermalCurves::diagnostics);

    s.def(
        "thermal",
        [](const ::Operator& H, const sec::Spec& spec, const sec::ThermalSpec& t) {
            py::gil_scoped_release nogil;
            return sec::thermal(H, spec, t);
        },
        py::arg("H"), py::arg("spec"), py::arg("thermal"),
        "Thermodynamics of H over every symmetry block, combined in log space.");

    py::class_<sec::DynamicsSpec>(s, "DynamicsSpec")
        .def(py::init<>())
        .def_readwrite("omega", &sec::DynamicsSpec::omega)
        .def_readwrite("eta", &sec::DynamicsSpec::eta)
        .def_readwrite("qfi_moments", &sec::DynamicsSpec::qfi_moments)
        .def_readwrite("temperatures", &sec::DynamicsSpec::temperatures)
        .def_readwrite("krylov", &sec::DynamicsSpec::krylov)
        .def_readwrite("samples", &sec::DynamicsSpec::samples)
        .def_readwrite("seed", &sec::DynamicsSpec::seed)
        .def_readwrite("degeneracy_tol", &sec::DynamicsSpec::degeneracy_tol)
        .def_readwrite("dense_max_dim", &sec::DynamicsSpec::dense_max_dim)
        .def_readwrite("prune", &sec::DynamicsSpec::prune)
        .def_readwrite("device", &sec::DynamicsSpec::device)
        .def_readwrite("thermodynamics", &sec::DynamicsSpec::thermodynamics)
        .def_readwrite("observables", &sec::DynamicsSpec::observables)
        .def_property(
            "observable_pairs",
            [](const sec::DynamicsSpec& d) {
                py::list out;
                for (const auto& p : d.observable_pairs) out.append(py::make_tuple(p.A, p.B));
                return out;
            },
            [](sec::DynamicsSpec& d,
               const std::vector<std::pair<std::vector<const ::Operator*>, std::vector<const ::Operator*>>>& pairs) {
                d.observable_pairs.clear();
                for (const auto& [A, B] : pairs) d.observable_pairs.push_back({A, B});
            },
            "T > 0 thermal pass: equal-time pair requests [(A, B), ...], after the observables in O.");

    py::class_<sec::DynamicsCurves>(s, "DynamicsCurves")
        .def_readonly("omega", &sec::DynamicsCurves::omega)
        .def_readonly("S", &sec::DynamicsCurves::S)
        .def_readonly("e0", &sec::DynamicsCurves::e0)
        .def_readonly("ground_manifold", &sec::DynamicsCurves::ground_manifold)
        .def_readonly("lnZ", &sec::DynamicsCurves::lnZ)
        .def_readonly("E", &sec::DynamicsCurves::E)
        .def_readonly("V", &sec::DynamicsCurves::V)
        .def_readonly("O", &sec::DynamicsCurves::O)
        .def_readonly("device_blocks", &sec::DynamicsCurves::device_blocks)
        .def_property_readonly("placement", [](const sec::DynamicsCurves& r) { return placement_to_py(r.placement); })
        .def_readonly("diagnostics", &sec::DynamicsCurves::diagnostics);

    s.def(
        "dynamics",
        [](const ::Operator& H, const sec::Spec& spec,
           const std::vector<std::pair<const ::Operator*, const ::Operator*>>& probes, const sec::DynamicsSpec& d) {
            std::vector<sec::Probe> ps;
            for (const auto& [a, b] : probes) {
                if (!a) throw ed::InvalidRequest("dynamics: a probe without its operator A");
                ps.push_back({a, b});
            }
            py::gil_scoped_release nogil;
            return sec::dynamics(H, spec, ps, d);
        },
        py::arg("H"), py::arg("spec"), py::arg("probes"), py::arg("dynamics"),
        "S_AB(omega) = <A^dag delta(omega - H + E) B> over the momentum sectors of H, per (A, B) probe "
        "(B None: A's autocorrelation); S is [probe][row][omega], complex.");

    s.def("eigs_to_arrays", &eigs_to_arrays, py::arg("result"), py::arg("spec"),
          "A result as named arrays (EigResult.save).");
    s.def("eigs_from_arrays", &eigs_from_arrays, py::arg("arrays"),
          "(result, spec) from eigs_to_arrays output (qed.load_eigs); result.n_sites is restored.");
    s.def(
        "transition_amplitudes",
        [](const sec::EigsResult& ri, const sec::Spec& si, const std::vector<std::size_t>& initial,
           const sec::EigsResult& rf, const sec::Spec& sf, const std::vector<std::size_t>& final,
           const std::vector<const ::Operator*>& ops) {
            sec::TransitionAmplitudes t;
            {
                py::gil_scoped_release nogil;
                t = sec::transition_amplitudes(ri, si, initial, rf, sf, final, ops);
            }
            py::array_t<std::complex<double>> a({static_cast<py::ssize_t>(t.n_ops), static_cast<py::ssize_t>(t.n_final),
                                                 static_cast<py::ssize_t>(t.n_initial)});
            std::copy(t.amplitudes.begin(), t.amplitudes.end(), a.mutable_data());
            py::dict d;
            d["amplitudes"] = a;
            d["initial_offsets"] = t.initial_offsets;
            d["final_offsets"] = t.final_offsets;
            return d;
        },
        py::arg("initial_result"), py::arg("initial_spec"), py::arg("initial"), py::arg("final_result"),
        py::arg("final_spec"), py::arg("final"), py::arg("ops"),
        "<m'|O|n'> over the members of the initial and final levels' multiplets: a dict with amplitudes "
        "[op, final member, initial member] and the members' offsets per level.");
    s.def(
        "eigs",
        [](const ::Operator& H, const sec::Spec& spec, int k, bool vectors, int dense_max_dim, bool allow_partial,
           sec::Device device, bool prune, double window, int per_block, std::uint64_t max_iter) {
            sec::EigsOptions o;
            o.k = k;
            o.max_iter = max_iter;
            o.per_block = per_block;
            o.cut = per_block <= 0; // per_block: every block's lowest rows, no window across blocks
            o.vectors = vectors;
            o.dense_max_dim = dense_max_dim;
            o.allow_partial = allow_partial;
            o.device = device;
            o.prune = prune;
            o.window = window;
            py::gil_scoped_release nogil;
            return sec::eigs(H, spec, o);
        },
        py::arg("H"), py::arg("spec"), py::arg("k") = 1, py::arg("vectors") = false, py::arg("dense_max_dim") = -1,
        py::arg("allow_partial") = false, py::arg("device") = sec::Device::Cpu, py::arg("prune") = true,
        py::arg("window") = 0.0, py::arg("per_block") = 0, py::arg("max_iter") = 0,
        "Lowest k eigenvalues (with multiplicity) over every symmetry block of H; per_block > 0: the lowest "
        "per_block levels of every block instead. max_iter > 0: each block's Krylov iteration budget.");
}
