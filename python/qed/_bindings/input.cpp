// =============================================================================
// python/qed/_bindings/input.cpp
//
// pybind11 bindings for the standalone `ed::input` C++ library.
//
// Surface mounted under `qed._core.input`:
//
//   * Op enum                     - Sp, Sm, Sz with integer values 0/1/2.
//   * Bond / Plaquette structs    - lightweight POD records.
//   * Lattice                     - geometry container (positions, sublattice,
//                                   nn_bonds, nnn_bonds, nnnn_bonds, label,
//                                   pbc, lattice_vectors).
//   * lattice.chain / square /    - lattice generators
//     triangular / honeycomb /
//     kagome / pyrochlore /
//     from_neighbor_lists /
//     from_cluster_file
//
// The Python-side facade in `python/qed/input.py` re-exports this submodule under
// `qed.input`, next to the Python HamiltonianBuilder (python/qed/_builder.py).
// =============================================================================

#include "bindings.h"

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ed/ops/operator.h>
#include <ed/input/lattice.h>
#include <ed/input/types.h>

#include <complex>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using ed::input::Bond;
using ed::input::Lattice;
using ed::input::Op;
using ed::input::Plaquette;

}  // namespace

void bind_input(py::module_& parent) {
    py::module_ m = parent.def_submodule("input", "Lattices and the Hamiltonian builder (the ed::input library).");

    // ---------------------------------------------------------------------
    // Op enum
    // ---------------------------------------------------------------------
    py::enum_<Op>(m, "Op",
                  "Spin operator code matching TransformData::op_type "
                  "(Sp=0, Sm=1, Sz=2).")
        .value("Sp", Op::Sp)
        .value("Sm", Op::Sm)
        .value("Sz", Op::Sz)
        .export_values();

    // ---------------------------------------------------------------------
    // Bond / Plaquette PODs
    // ---------------------------------------------------------------------
    py::class_<Bond>(m, "Bond",
                     "A bond from site i to site j (kept in that orientation), with an "
                     "optional bond_type tag, e.g. a Kitaev colour.")
        .def(py::init<std::size_t, std::size_t, int>(), py::arg("i"), py::arg("j"), py::arg("bond_type") = 0)
        .def_readwrite("i", &Bond::i)
        .def_readwrite("j", &Bond::j)
        .def_readwrite("bond_type", &Bond::bond_type)
        .def("__repr__", [](const Bond& b) {
            return "Bond(i=" + std::to_string(b.i) + ", j=" + std::to_string(b.j)
                   + ", bond_type=" + std::to_string(b.bond_type) + ")";
        });

    py::class_<Plaquette>(m, "Plaquette")
        .def(py::init<>())
        .def_readwrite("sites", &Plaquette::sites)
        .def_readwrite("plaquette_type", &Plaquette::plaquette_type);

    // ---------------------------------------------------------------------
    // Lattice
    // ---------------------------------------------------------------------
    py::class_<Lattice>(m, "Lattice", R"pbdoc(
        Geometry container produced by the lattice generators.

        Attributes
        ----------
        num_sites : int
        positions : list[tuple[float, float, float]]
        sublattice : list[int]
        nn_bonds : list[Bond]
            Each nearest-neighbour pair once, oriented as generated (the
            chain's wrap bond runs N-1 -> 0, kagome triangles counter-
            clockwise, honeycomb bonds A -> B), so a uniform DM vector over
            ``nn_pairs()`` is translation invariant (not along a periodic length of
            2, where a pair's two bonds are stored as one).
        nnn_bonds, nnnn_bonds : list[Bond]
            The second and third distance shells (minimum image on a
            periodic lattice), i < j. ``nnn_pairs()`` / ``nnnn_pairs()``
            raise for a lattice built from an adjacency list.
        lattice_vectors : tuple of three (float, float, float)
        pbc : bool
        label : str
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("num_sites", &Lattice::num_sites)
        .def_readwrite("positions", &Lattice::positions)
        .def_readwrite("sublattice", &Lattice::sublattice)
        .def_readwrite("nn_bonds", &Lattice::nn_bonds)
        .def_readwrite("nnn_bonds", &Lattice::nnn_bonds)
        .def_readwrite("nnnn_bonds", &Lattice::nnnn_bonds)
        .def_readwrite("lattice_vectors", &Lattice::lattice_vectors)
        .def_readwrite("pbc", &Lattice::pbc)
        .def_readwrite("label", &Lattice::label)
        .def("nn_pairs", &Lattice::nn_pairs)
        .def("nnn_pairs", &Lattice::nnn_pairs)
        .def("nnnn_pairs", &Lattice::nnnn_pairs)
        .def("all_sites", &Lattice::all_sites)
        .def("__repr__", [](const Lattice& L) {
            return "<qed.input.Lattice " + L.label + " num_sites=" + std::to_string(L.num_sites)
                   + " nn_bonds=" + std::to_string(L.nn_bonds.size()) + ">";
        });

    // ---------------------------------------------------------------------
    // Lattice factory functions: mirror the C++ namespace ed::input::lattice
    // under qed.input.lattice.
    // ---------------------------------------------------------------------
    py::module_ ml = m.def_submodule("lattice", "Lattice generators (chain, square, triangular, honeycomb, kagome, "
                                                "pyrochlore, custom-from-edges, cluster.txt).");

    ml.def("chain", &ed::input::lattice::chain, py::arg("length"), py::arg("pbc") = false);
    ml.def("square", &ed::input::lattice::square, py::arg("Lx"), py::arg("Ly"), py::arg("pbc") = false);
    ml.def("triangular", &ed::input::lattice::triangular, py::arg("Lx"), py::arg("Ly"), py::arg("pbc") = false);
    ml.def("honeycomb", &ed::input::lattice::honeycomb, py::arg("Lx"), py::arg("Ly"), py::arg("pbc") = false);
    ml.def("kagome", &ed::input::lattice::kagome, py::arg("Lx"), py::arg("Ly"), py::arg("pbc") = false);
    ml.def("pyrochlore", &ed::input::lattice::pyrochlore, py::arg("Lx"), py::arg("Ly"), py::arg("Lz"),
           py::arg("pbc") = false);
    ml.def("from_neighbor_lists", &ed::input::lattice::from_neighbor_lists, py::arg("positions"), py::arg("nn_pairs"),
           py::arg("sublattice") = std::vector<int>{});
    ml.def("from_cluster_file", &ed::input::lattice::from_cluster_file, py::arg("path"));
}
