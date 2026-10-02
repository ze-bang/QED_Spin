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
    return py::array_t<std::complex<double>>({heap->size()}, {sizeof(std::complex<double>)},
                                             heap->data(), owner);
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

template <class T>
py::array_t<T> arr(const std::vector<T>& v) {
    return py::array_t<T>(static_cast<py::ssize_t>(v.size()), v.data());
}

template <class T>
std::vector<T> vec(const py::dict& d, const char* key) {
    auto a = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(d[key]);
    if (!a) throw std::invalid_argument(std::string("load_eigs: bad array ") + key);
    return std::vector<T>(a.data(), a.data() + a.size());
}

py::dict eigs_to_arrays(const sec::EigsResult& r, const sec::Spec& s) {
    py::dict d;
    const std::size_t nl = r.levels.size();
    std::vector<double> energy(nl);
    std::vector<std::int64_t> mirror(nl), mult(nl), vector(nl), tag(nl * 11);
    for (std::size_t i = 0; i < nl; ++i) {
        const auto& L = r.levels[i];
        const auto& t = L.tag;
        energy[i] = L.energy; mirror[i] = L.mirror;
        mult[i] = static_cast<std::int64_t>(L.multiplicity); vector[i] = L.vector;
        const std::int64_t row[11] = {t.n_up, t.sz_parity, t.k0, t.k_raw, t.flip_parity, t.irrep,
                                      t.irrep_dim, t.star_size, t.tr_folded ? 1 : 0,
                                      static_cast<std::int64_t>(t.dim),
                                      static_cast<std::int64_t>(t.multiplicity)};
        std::copy(row, row + 11, tag.begin() + static_cast<std::ptrdiff_t>(11 * i));
    }
    d["level_energy"] = arr(energy); d["level_mirror"] = arr(mirror);
    d["level_multiplicity"] = arr(mult); d["level_vector"] = arr(vector); d["level_tag"] = arr(tag);
    // Physical labels, ragged per level: momentum characters, co-group (residue, character).
    std::vector<std::int64_t> moff{0}, ioff{0}, ielem;
    std::vector<std::complex<double>> mchi, ichi;
    for (const auto& L : r.levels) {
        mchi.insert(mchi.end(), L.momentum.begin(), L.momentum.end());
        moff.push_back(static_cast<std::int64_t>(mchi.size()));
        for (const auto& [e, c] : L.irrep_characters) { ielem.push_back(e); ichi.push_back(c); }
        ioff.push_back(static_cast<std::int64_t>(ielem.size()));
    }
    d["level_momentum"] = arr(mchi); d["level_momentum_offset"] = arr(moff);
    d["level_irrep_elems"] = arr(ielem); d["level_irrep_chars"] = arr(ichi); d["level_irrep_offset"] = arr(ioff);

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
            d[py::str(p + "shape")] = arr(std::vector<std::int64_t>{b.group_size, b.n_sites, b.n_up});
        }
        vbasis.push_back(it->second);
        amps.insert(amps.end(), v.amplitudes.begin(), v.amplitudes.end());
        voffset.push_back(static_cast<std::int64_t>(amps.size()));
    }
    d["n_bases"] = arr(std::vector<std::int64_t>{static_cast<std::int64_t>(basis_id.size())});
    d["vector_basis"] = arr(vbasis); d["vector_offset"] = arr(voffset); d["vector_amplitudes"] = arr(amps);

    auto perms = [](const std::vector<sec::Perm>& ps) {
        std::vector<std::int64_t> flat;
        for (const auto& p : ps) flat.insert(flat.end(), p.begin(), p.end());
        return flat;
    };
    d["spec_abelian"] = arr(perms(s.abelian)); d["spec_residues"] = arr(perms(s.residues));
    d["spec_only_k0"] = arr(std::vector<std::int64_t>(s.only_k0.begin(), s.only_k0.end()));
    d["spec_only_irrep"] = arr(std::vector<std::int64_t>(s.only_irrep.begin(), s.only_irrep.end()));
    d["spec_scalars"] = arr(std::vector<std::int64_t>{s.n_up, s.sz_parity, s.use_sz ? 1 : 0, s.spin_flip,
                                                      s.time_reversal, s.two_S, r.n_sites});
    d["result_scalars"] = arr(std::vector<std::int64_t>{
        static_cast<std::int64_t>(r.total_dim), static_cast<std::int64_t>(r.partial_blocks),
        r.complete ? 1 : 0, r.flip_engaged ? 1 : 0, r.tr_engaged ? 1 : 0,
        static_cast<std::int64_t>(r.device_blocks), static_cast<std::int64_t>(r.pruned_blocks)});
    return d;
}

py::tuple eigs_from_arrays(const py::dict& d) {
    sec::EigsResult r;
    const auto energy = vec<double>(d, "level_energy");
    const auto mirror = vec<std::int64_t>(d, "level_mirror");
    const auto mult = vec<std::int64_t>(d, "level_multiplicity");
    const auto vector = vec<std::int64_t>(d, "level_vector");
    const auto tag = vec<std::int64_t>(d, "level_tag");
    if (tag.size() != 11 * energy.size()) throw std::invalid_argument("load_eigs: level_tag has the wrong shape");
    for (std::size_t i = 0; i < energy.size(); ++i) {
        sec::Level L;
        const std::int64_t* t = tag.data() + 11 * i;
        L.energy = energy[i]; L.mirror = static_cast<int>(mirror[i]);
        L.multiplicity = static_cast<std::uint64_t>(mult[i]); L.vector = static_cast<int>(vector[i]);
        L.tag.n_up = static_cast<int>(t[0]); L.tag.sz_parity = static_cast<int>(t[1]);
        L.tag.k0 = static_cast<int>(t[2]); L.tag.k_raw = static_cast<int>(t[3]);
        L.tag.flip_parity = static_cast<int>(t[4]); L.tag.irrep = static_cast<int>(t[5]);
        L.tag.irrep_dim = static_cast<int>(t[6]); L.tag.star_size = static_cast<int>(t[7]);
        L.tag.tr_folded = t[8] != 0; L.tag.dim = static_cast<std::uint64_t>(t[9]);
        L.tag.multiplicity = static_cast<std::uint64_t>(t[10]);
        r.levels.push_back(L);
    }
    if (d.contains("level_momentum")) {
        const auto mchi = vec<std::complex<double>>(d, "level_momentum");
        const auto moff = vec<std::int64_t>(d, "level_momentum_offset");
        const auto ielem = vec<std::int64_t>(d, "level_irrep_elems");
        const auto ichi = vec<std::complex<double>>(d, "level_irrep_chars");
        const auto ioff = vec<std::int64_t>(d, "level_irrep_offset");
        if (moff.size() != r.levels.size() + 1 || ioff.size() != r.levels.size() + 1)
            throw std::invalid_argument("load_eigs: level labels have the wrong shape");
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
        rd->group_size = static_cast<int>(shape.at(0));
        rd->n_sites = static_cast<int>(shape.at(1));
        rd->n_up = static_cast<int>(shape.at(2));
        rd->build_perm_lut();
        bases.push_back(std::move(rd));
    }
    const auto vbasis = vec<std::int64_t>(d, "vector_basis");
    const auto voffset = vec<std::int64_t>(d, "vector_offset");
    const auto amps = vec<std::complex<double>>(d, "vector_amplitudes");
    for (std::size_t i = 0; i < vbasis.size(); ++i) {
        sec::BlockVector v;
        v.basis = bases.at(static_cast<std::size_t>(vbasis[i]));
        v.amplitudes.assign(amps.begin() + voffset[i], amps.begin() + voffset[i + 1]);
        if (v.amplitudes.size() != v.basis->reps.size())
            throw std::invalid_argument("load_eigs: a vector does not match its basis");
        r.vectors.push_back(std::move(v));
    }
    const auto sc = vec<std::int64_t>(d, "result_scalars");
    r.total_dim = static_cast<std::uint64_t>(sc.at(0)); r.partial_blocks = static_cast<std::size_t>(sc.at(1));
    r.complete = sc.at(2) != 0; r.flip_engaged = sc.at(3) != 0; r.tr_engaged = sc.at(4) != 0;
    r.device_blocks = static_cast<std::size_t>(sc.at(5)); r.pruned_blocks = static_cast<std::size_t>(sc.at(6));

    sec::Spec s;
    const auto ss = vec<std::int64_t>(d, "spec_scalars");
    const int n = static_cast<int>(ss.at(6));
    auto perms = [n](const std::vector<std::int64_t>& flat) {
        std::vector<sec::Perm> out;
        for (std::size_t i = 0; n > 0 && i + static_cast<std::size_t>(n) <= flat.size(); i += static_cast<std::size_t>(n))
            out.emplace_back(flat.begin() + static_cast<std::ptrdiff_t>(i), flat.begin() + static_cast<std::ptrdiff_t>(i) + n);
        return out;
    };
    s.abelian = perms(vec<std::int64_t>(d, "spec_abelian"));
    s.residues = perms(vec<std::int64_t>(d, "spec_residues"));
    for (auto k : vec<std::int64_t>(d, "spec_only_k0")) s.only_k0.push_back(static_cast<int>(k));
    for (auto k : vec<std::int64_t>(d, "spec_only_irrep")) s.only_irrep.push_back(static_cast<int>(k));
    s.n_up = static_cast<int>(ss.at(0)); s.sz_parity = static_cast<int>(ss.at(1)); s.use_sz = ss.at(2) != 0;
    s.spin_flip = static_cast<int>(ss.at(3)); s.time_reversal = static_cast<int>(ss.at(4));
    s.two_S = static_cast<int>(ss.at(5));
    r.n_sites = n;
    return py::make_tuple(std::move(r), std::move(s));
}

// Where a verb's solves ran (sec::Placement), as a dict.
py::dict placement_to_py(const sec::Placement& p) {
    py::dict d;
    d["device_krylov"] = p.device_krylov; d["device_dense"] = p.device_dense;
    d["host_krylov"] = p.host_krylov; d["host_dense"] = p.host_dense;
    return d;
}

// One dict per solved block (EigsResult::block_stats).
py::list block_stats_to_py(const std::vector<sec::BlockStats>& v) {
    py::list out;
    for (const auto& b : v) {
        py::dict d;
        d["k0"] = b.k0; d["irrep"] = b.irrep; d["flip_parity"] = b.flip_parity; d["n_up"] = b.n_up;
        d["dim"] = b.dim; d["kind"] = b.kind; d["lane"] = b.lane;
        d["context_orbit_s"] = b.context_orbit_s; d["star_orbit_s"] = b.star_orbit_s;
        d["star_build_s"] = b.star_build_s; d["build_s"] = b.build_s;
        d["nnz"] = b.nnz; d["csr_bytes"] = b.csr_bytes;
        d["applies"] = b.applies; d["apply_s"] = b.apply_s; d["other_s"] = b.other_s; d["solve_s"] = b.solve_s;
        out.append(std::move(d));
    }
    return out;
}
}  // namespace

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
        .def_readonly("tr_engaged", &sec::EigsResult::tr_engaged)
        .def_readonly("device_blocks", &sec::EigsResult::device_blocks)
        .def_property_readonly("placement", [](const sec::EigsResult& r) { return placement_to_py(r.placement); })
        .def_readonly("pruned_blocks", &sec::EigsResult::pruned_blocks)
        .def_readonly("diagnostics", &sec::EigsResult::diagnostics)
        .def_property_readonly("block_stats", [](const sec::EigsResult& r) { return block_stats_to_py(r.block_stats); },
                               "Per solved block: dim, lane, phase seconds, nnz, applies (one dict each).")
        .def("energies", &sec::EigsResult::energies, py::arg("k"))
        .def("multiplet", [](const sec::EigsResult& r, const sec::Spec& spec, int level, int n_up) {
                 const auto& L = r.levels.at(static_cast<std::size_t>(level));
                 if (L.vector < 0) throw std::invalid_argument("multiplet: the level carries no vector");
                 std::vector<std::vector<std::complex<double>>> vs;
                 {
                     py::gil_scoped_release nogil;
                     vs = sec::multiplet(spec, r.n_sites, L, r.vectors[static_cast<std::size_t>(L.vector)], n_up);
                 }
                 py::list out;
                 for (auto& v : vs) out.append(to_array(std::move(v)));
                 return out;
             }, py::arg("spec"), py::arg("level"), py::arg("n_up") = -1,
             "The level's degenerate multiplet expanded into Sz sector n_up (n_up < 0: full space).")
        .def("expect", [](const sec::EigsResult& r, const sec::Spec& spec,
                          const std::vector<const ::Operator*>& ops) {
                 py::gil_scoped_release nogil;
                 return sec::expect(r, spec, ops);
             }, py::arg("spec"), py::arg("ops"),
             "<O> per level averaged over its symmetry multiplet: [level][op].")
        .def("matrix_element", [](const sec::EigsResult& r, const ::Operator& O,
                                  std::size_t i, std::size_t j) {
                 py::gil_scoped_release nogil;
                 return sec::matrix_element(r, O, i, j);
             }, py::arg("O"), py::arg("i"), py::arg("j"),
             "<v_i|O|v_j> between the vectors of levels i and j.");

    py::class_<sec::SpectrumResult>(s, "SpectrumResult")
        .def_readonly("levels", &sec::SpectrumResult::levels)
        .def_readonly("device_blocks", &sec::SpectrumResult::device_blocks)
        .def_property_readonly("placement", [](const sec::SpectrumResult& r) { return placement_to_py(r.placement); })
        .def_readonly("diagnostics", &sec::SpectrumResult::diagnostics)
        .def("expanded", [](const sec::SpectrumResult& r) { return to_real_array(r.expanded()); });

    s.def("spectrum",
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
        .def_readwrite("exact_states", &sec::ThermalSpec::exact_states)
        .def_readwrite("dense_max_dim", &sec::ThermalSpec::dense_max_dim)
        .def_readwrite("seed", &sec::ThermalSpec::seed)
        .def_readwrite("device", &sec::ThermalSpec::device)
        .def_readwrite("observables", &sec::ThermalSpec::observables);

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

    s.def("thermal",
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
        .def_readwrite("temperatures", &sec::DynamicsSpec::temperatures)
        .def_readwrite("krylov", &sec::DynamicsSpec::krylov)
        .def_readwrite("samples", &sec::DynamicsSpec::samples)
        .def_readwrite("seed", &sec::DynamicsSpec::seed)
        .def_readwrite("degeneracy_tol", &sec::DynamicsSpec::degeneracy_tol)
        .def_readwrite("dense_max_dim", &sec::DynamicsSpec::dense_max_dim)
        .def_readwrite("device", &sec::DynamicsSpec::device);

    py::class_<sec::DynamicsCurves>(s, "DynamicsCurves")
        .def_readonly("omega", &sec::DynamicsCurves::omega)
        .def_readonly("S", &sec::DynamicsCurves::S)
        .def_readonly("e0", &sec::DynamicsCurves::e0)
        .def_readonly("ground_manifold", &sec::DynamicsCurves::ground_manifold)
        .def_readonly("device_blocks", &sec::DynamicsCurves::device_blocks)
        .def_property_readonly("placement", [](const sec::DynamicsCurves& r) { return placement_to_py(r.placement); })
        .def_readonly("diagnostics", &sec::DynamicsCurves::diagnostics);

    s.def("dynamics",
          [](const ::Operator& H, const sec::Spec& spec, const ::Operator& O, const sec::DynamicsSpec& d) {
              py::gil_scoped_release nogil;
              return sec::dynamics(H, spec, O, d);
          },
          py::arg("H"), py::arg("spec"), py::arg("O"), py::arg("dynamics"),
          "S(omega) = <O^dag delta(omega - H + E) O> over the momentum sectors of H.");

    s.def("eigs_to_arrays", &eigs_to_arrays, py::arg("result"), py::arg("spec"),
          "A result as named arrays (EigResult.save).");
    s.def("eigs_from_arrays", &eigs_from_arrays, py::arg("arrays"),
          "(result, spec) from eigs_to_arrays output (qed.load_eigs); result.n_sites is restored.");
    s.def("eigs",
          [](const ::Operator& H, const sec::Spec& spec, int k, bool vectors,
             int dense_max_dim, bool allow_partial, sec::Device device,
             bool prune, double window) {
              sec::EigsOptions o;
              o.k = k; o.vectors = vectors; o.dense_max_dim = dense_max_dim;
              o.allow_partial = allow_partial; o.device = device;
              o.prune = prune; o.window = window;
              py::gil_scoped_release nogil;
              return sec::eigs(H, spec, o);
          },
          py::arg("H"), py::arg("spec"), py::arg("k") = 1,
          py::arg("vectors") = false, py::arg("dense_max_dim") = -1,
          py::arg("allow_partial") = false, py::arg("device") = sec::Device::Cpu,
          py::arg("prune") = true, py::arg("window") = 0.0,
          "Lowest k eigenvalues (with multiplicity) over every symmetry block of H.");
}
