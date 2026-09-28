// =============================================================================
// python/qed/_bindings/sectors_bindings.cpp -- _core.sectors: the symmetry sectors of a
// Hamiltonian and the lowest-level eigensolve over them (include/ed/sectors/sectors.h).
// =============================================================================

#include "sectors_bindings.h"

#include <ed/sectors/sectors.h>

#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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
        .def_readwrite("only_k0", &sec::Spec::only_k0)
        .def_readwrite("only_irrep", &sec::Spec::only_irrep);

    py::enum_<sec::SzContent>(s, "SzContent")
        .value("U1", sec::SzContent::U1)
        .value("Parity", sec::SzContent::Parity)
        .value("None_", sec::SzContent::None);
    s.def("sz_content", &sec::sz_content, py::arg("H"));

    py::class_<sec::Subspace>(s, "Subspace")
        .def_readonly("n_up", &sec::Subspace::n_up)
        .def_readonly("sz_parity", &sec::Subspace::sz_parity)
        .def_readonly("mirror", &sec::Subspace::mirror);
    s.def("subspaces", &sec::subspaces, py::arg("H"), py::arg("n_sites"), py::arg("spec"));

    py::class_<sec::Level>(s, "Level")
        .def_readonly("energy", &sec::Level::energy)
        .def_readonly("multiplicity", &sec::Level::multiplicity)
        .def_readonly("mirror", &sec::Level::mirror)
        .def_readonly("vector", &sec::Level::vector)
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
        .def_readonly("total_dim", &sec::EigsResult::total_dim)
        .def_readonly("partial_blocks", &sec::EigsResult::partial_blocks)
        .def_readonly("complete", &sec::EigsResult::complete)
        .def_readonly("flip_engaged", &sec::EigsResult::flip_engaged)
        .def_readonly("tr_engaged", &sec::EigsResult::tr_engaged)
        .def("energies", &sec::EigsResult::energies, py::arg("k"))
        .def("sector_vector", [](const sec::EigsResult& r, int i) {
                 const auto& v = r.vectors.at(static_cast<std::size_t>(i));
                 return py::make_tuple(py::array_t<std::uint64_t>(v.basis->reps.size(), v.basis->reps.data()),
                                       to_array(v.amplitudes));
             }, py::arg("i"),
             "(representatives, amplitudes) of vector i in the rep basis it was solved in.")
        .def("multiplet", [](const sec::EigsResult& r, const sec::Spec& spec, int n_sites,
                             int level, int n_up) {
                 const auto& L = r.levels.at(static_cast<std::size_t>(level));
                 if (L.vector < 0) throw std::invalid_argument("multiplet: the level carries no vector");
                 std::vector<std::vector<std::complex<double>>> vs;
                 {
                     py::gil_scoped_release nogil;
                     vs = sec::multiplet(spec, n_sites, L, r.vectors[static_cast<std::size_t>(L.vector)], n_up);
                 }
                 py::list out;
                 for (auto& v : vs) out.append(to_array(std::move(v)));
                 return out;
             }, py::arg("spec"), py::arg("n_sites"), py::arg("level"), py::arg("n_up") = -1,
             "The level's degenerate multiplet expanded into Sz sector n_up (n_up < 0: full space).");

    py::class_<sec::SpectrumResult>(s, "SpectrumResult")
        .def_readonly("levels", &sec::SpectrumResult::levels)
        .def_readonly("total_dim", &sec::SpectrumResult::total_dim)
        .def_readonly("flip_engaged", &sec::SpectrumResult::flip_engaged)
        .def_readonly("tr_engaged", &sec::SpectrumResult::tr_engaged)
        .def("expanded", [](const sec::SpectrumResult& r) { return to_real_array(r.expanded()); });

    s.def("spectrum",
          [](const ::Operator& H, int n_sites, const sec::Spec& spec) {
              py::gil_scoped_release nogil;
              return sec::spectrum(H, n_sites, spec);
          },
          py::arg("H"), py::arg("n_sites"), py::arg("spec"),
          "The complete spectrum of H, every symmetry block diagonalised densely.");

    s.def("eigs",
          [](const ::Operator& H, int n_sites, const sec::Spec& spec, int k, bool vectors,
             int dense_max_dim, int block_size, bool allow_partial) {
              sec::EigsOptions o;
              o.k = k; o.vectors = vectors; o.dense_max_dim = dense_max_dim;
              o.block_size = block_size; o.allow_partial = allow_partial;
              py::gil_scoped_release nogil;
              return sec::eigs(H, n_sites, spec, o);
          },
          py::arg("H"), py::arg("n_sites"), py::arg("spec"), py::arg("k") = 1,
          py::arg("vectors") = false, py::arg("dense_max_dim") = 64, py::arg("block_size") = 1,
          py::arg("allow_partial") = false,
          "Lowest k eigenvalues (with multiplicity) over every symmetry block of H.");
}
