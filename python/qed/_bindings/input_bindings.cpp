// =============================================================================
// python/qed/_bindings/input_bindings.cpp
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
//   * lattice.chain / square /    - generators that mirror the legacy
//     triangular / honeycomb /      `python/edlib/helper_*.py` family.
//     kagome / pyrochlore /
//     from_neighbor_lists /
//     from_cluster_file
//   * HamiltonianBuilder          - fluent term accumulator.
//   * FileOptions                 - directory-output configuration.
//   * Free file writers           - low-level `Trans.dat` / `InterAll.dat` /
//                                   `ThreeBodyG.dat` / `positions.dat` /
//                                   `one_body_correlations*.dat` /
//                                   `two_body_correlations**.dat` /
//                                   momentum-projected observable writers.
//
// `HamiltonianBuilder.to_operator()` returns a `qed.Operator`, the
// same Python class produced by `Operator(num_sites)` -- so users can drop
// the result straight into `lanczos`, `full_diagonalization`,
// `finite_temperature_lanczos`, etc.
//
// The Python-side facade in `python/qed/input.py` re-exports this
// submodule under `qed.input`.
// =============================================================================

#include "input_bindings.h"

#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ed/core/construct_ham.h>
#include <ed/input/input.h>

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
using ed::input::FileOptions;
using ed::input::HamiltonianBuilder;
using ed::input::Lattice;
using ed::input::Op;
using ed::input::Plaquette;
using ed::input::Position;

using BondPair = std::pair<std::size_t, std::size_t>;

// Convert a Python iterable of (i, j) tuples into the std::vector<BondPair>
// that the HamiltonianBuilder shortcuts expect.
inline std::vector<BondPair> bond_pairs_from_py(const py::iterable& it) {
    std::vector<BondPair> out;
    for (auto h : it) {
        auto t = h.cast<py::tuple>();
        if (t.size() != 2) {
            throw std::invalid_argument(
                "bond list entries must be (i, j) tuples");
        }
        out.emplace_back(t[0].cast<std::size_t>(), t[1].cast<std::size_t>());
    }
    return out;
}

inline std::vector<std::array<double, 3>> vec3_from_py(const py::iterable& it) {
    std::vector<std::array<double, 3>> out;
    for (auto h : it) {
        auto t = h.cast<py::tuple>();
        if (t.size() != 3) {
            throw std::invalid_argument("vector entries must be length-3 tuples");
        }
        out.push_back({t[0].cast<double>(), t[1].cast<double>(),
                       t[2].cast<double>()});
    }
    return out;
}

inline std::vector<std::array<std::size_t, 4>> plaquettes_from_py(
    const py::iterable& it) {
    std::vector<std::array<std::size_t, 4>> out;
    for (auto h : it) {
        auto t = h.cast<py::tuple>();
        if (t.size() != 4) {
            throw std::invalid_argument(
                "plaquette entries must be length-4 site tuples");
        }
        out.push_back({t[0].cast<std::size_t>(), t[1].cast<std::size_t>(),
                       t[2].cast<std::size_t>(), t[3].cast<std::size_t>()});
    }
    return out;
}

}  // namespace

void bind_input(py::module_& parent) {
    py::module_ m = parent.def_submodule(
        "input",
        "Standalone C++ lattice + Hamiltonian builder library "
        "(replaces python/edlib/helper_*.py).");

    // ---------------------------------------------------------------------
    // Op enum
    // ---------------------------------------------------------------------
    py::enum_<Op>(m, "Op", "Spin operator code matching TransformData::op_type "
                            "(Sp=0, Sm=1, Sz=2).")
        .value("Sp", Op::Sp)
        .value("Sm", Op::Sm)
        .value("Sz", Op::Sz)
        .export_values();

    // ---------------------------------------------------------------------
    // Bond / Plaquette PODs
    // ---------------------------------------------------------------------
    py::class_<Bond>(m, "Bond")
        .def(py::init<std::size_t, std::size_t, int>(),
             py::arg("i"), py::arg("j"), py::arg("bond_type") = 0)
        .def_readwrite("i", &Bond::i)
        .def_readwrite("j", &Bond::j)
        .def_readwrite("bond_type", &Bond::bond_type)
        .def("__repr__", [](const Bond& b) {
            return "Bond(i=" + std::to_string(b.i) +
                   ", j=" + std::to_string(b.j) +
                   ", bond_type=" + std::to_string(b.bond_type) + ")";
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
        nn_bonds, nnn_bonds, nnnn_bonds : list[Bond]
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
            return "<qed.input.Lattice " + L.label +
                   " num_sites=" + std::to_string(L.num_sites) +
                   " nn_bonds=" + std::to_string(L.nn_bonds.size()) + ">";
        });

    // ---------------------------------------------------------------------
    // Lattice factory functions: mirror the C++ namespace ed::input::lattice
    // under qed.input.lattice.
    // ---------------------------------------------------------------------
    py::module_ ml = m.def_submodule(
        "lattice",
        "Lattice generators (chain, square, triangular, honeycomb, kagome, "
        "pyrochlore, custom-from-edges, cluster.txt).");

    ml.def("chain", &ed::input::lattice::chain,
           py::arg("length"), py::arg("pbc") = false);
    ml.def("square", &ed::input::lattice::square,
           py::arg("Lx"), py::arg("Ly"), py::arg("pbc") = false);
    ml.def("triangular", &ed::input::lattice::triangular,
           py::arg("Lx"), py::arg("Ly"), py::arg("pbc") = false);
    ml.def("honeycomb", &ed::input::lattice::honeycomb,
           py::arg("Lx"), py::arg("Ly"), py::arg("pbc") = false);
    ml.def("kagome", &ed::input::lattice::kagome,
           py::arg("Lx"), py::arg("Ly"), py::arg("pbc") = false);
    ml.def("pyrochlore", &ed::input::lattice::pyrochlore,
           py::arg("Lx"), py::arg("Ly"), py::arg("Lz"), py::arg("pbc") = false);
    ml.def("from_neighbor_lists",
           &ed::input::lattice::from_neighbor_lists,
           py::arg("positions"), py::arg("nn_pairs"),
           py::arg("sublattice") = std::vector<int>{});
    ml.def("from_cluster_file", &ed::input::lattice::from_cluster_file,
           py::arg("path"));

    // ---------------------------------------------------------------------
    // FileOptions
    // ---------------------------------------------------------------------
    py::class_<FileOptions>(m, "FileOptions", R"pbdoc(
        Output configuration for `HamiltonianBuilder.write_directory`.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("trans_filename", &FileOptions::trans_filename)
        .def_readwrite("inter_all_filename", &FileOptions::inter_all_filename)
        .def_readwrite("three_body_filename", &FileOptions::three_body_filename)
        .def_readwrite("positions_filename", &FileOptions::positions_filename)
        .def_readwrite("tol", &FileOptions::tol)
        .def_readwrite("one_body_obs", &FileOptions::one_body_obs)
        .def_readwrite("two_body_obs", &FileOptions::two_body_obs)
        .def_readwrite("write_positions", &FileOptions::write_positions)
        .def_readwrite("write_lattice_metadata",
                       &FileOptions::write_lattice_metadata);

    // ---------------------------------------------------------------------
    // HamiltonianBuilder
    // ---------------------------------------------------------------------
    py::class_<HamiltonianBuilder>(m, "HamiltonianBuilder", R"pbdoc(
        Fluent C++-backed Hamiltonian builder.

        Accumulates one-/two-/three-body terms in the canonical (S+, S-, Sz)
        basis used by `qed.Operator`. Materialise the result via
        :meth:`to_operator` (in-memory, no file I/O) or :meth:`write_directory`
        (legacy `InterAll.dat` / `Trans.dat` / `ThreeBodyG.dat` /
        `positions.dat` files consumed by the production `./ED` driver).
    )pbdoc")
        .def(py::init<std::size_t, double>(),
             py::arg("num_sites"), py::arg("spin") = 0.5)

        // Low-level term insertion ---------------------------------------
        .def("add_one_body",
             [](HamiltonianBuilder& self, Op op, std::size_t site,
                std::complex<double> coeff) -> HamiltonianBuilder& {
                 return self.add_one_body(op, site, coeff);
             },
             py::arg("op"), py::arg("site"), py::arg("coeff"),
             py::return_value_policy::reference_internal)
        .def("add_two_body",
             [](HamiltonianBuilder& self, Op op_i, std::size_t site_i,
                Op op_j, std::size_t site_j,
                std::complex<double> coeff) -> HamiltonianBuilder& {
                 return self.add_two_body(op_i, site_i, op_j, site_j, coeff);
             },
             py::arg("op_i"), py::arg("site_i"),
             py::arg("op_j"), py::arg("site_j"),
             py::arg("coeff"),
             py::return_value_policy::reference_internal)
        .def("add_three_body",
             [](HamiltonianBuilder& self, Op op_i, std::size_t site_i,
                Op op_j, std::size_t site_j,
                Op op_k, std::size_t site_k,
                std::complex<double> coeff) -> HamiltonianBuilder& {
                 return self.add_three_body(op_i, site_i, op_j, site_j,
                                            op_k, site_k, coeff);
             },
             py::arg("op_i"), py::arg("site_i"),
             py::arg("op_j"), py::arg("site_j"),
             py::arg("op_k"), py::arg("site_k"),
             py::arg("coeff"),
             py::return_value_policy::reference_internal)

        // High-level shortcuts -------------------------------------------
        .def("heisenberg",
             [](HamiltonianBuilder& self, py::iterable bonds, double J)
                 -> HamiltonianBuilder& {
                 return self.heisenberg(bond_pairs_from_py(bonds), J);
             },
             py::arg("bonds"), py::arg("J") = 1.0,
             py::return_value_policy::reference_internal)
        .def("xxz",
             [](HamiltonianBuilder& self, py::iterable bonds,
                double Jxy, double Jz) -> HamiltonianBuilder& {
                 return self.xxz(bond_pairs_from_py(bonds), Jxy, Jz);
             },
             py::arg("bonds"), py::arg("Jxy"), py::arg("Jz"),
             py::return_value_policy::reference_internal)
        .def("xyz",
             [](HamiltonianBuilder& self, py::iterable bonds,
                double Jxx, double Jyy, double Jzz) -> HamiltonianBuilder& {
                 return self.xyz(bond_pairs_from_py(bonds), Jxx, Jyy, Jzz);
             },
             py::arg("bonds"), py::arg("Jxx"), py::arg("Jyy"), py::arg("Jzz"),
             py::return_value_policy::reference_internal)
        .def("ising",
             [](HamiltonianBuilder& self, py::iterable bonds, double J)
                 -> HamiltonianBuilder& {
                 return self.ising(bond_pairs_from_py(bonds), J);
             },
             py::arg("bonds"), py::arg("J") = 1.0,
             py::return_value_policy::reference_internal)
        .def("transverse_field_ising",
             [](HamiltonianBuilder& self, py::iterable bonds,
                double J, double h) -> HamiltonianBuilder& {
                 return self.transverse_field_ising(
                     bond_pairs_from_py(bonds), J, h);
             },
             py::arg("bonds"), py::arg("J"), py::arg("h"),
             py::return_value_policy::reference_internal)
        .def("kitaev",
             [](HamiltonianBuilder& self, py::iterable bonds,
                std::vector<int> bond_axis, double K) -> HamiltonianBuilder& {
                 return self.kitaev(bond_pairs_from_py(bonds),
                                    std::move(bond_axis), K);
             },
             py::arg("bonds"), py::arg("bond_axis"), py::arg("K") = 1.0,
             py::return_value_policy::reference_internal)
        .def("dm",
             [](HamiltonianBuilder& self, py::iterable bonds,
                py::iterable D_per_bond) -> HamiltonianBuilder& {
                 return self.dm(bond_pairs_from_py(bonds),
                                vec3_from_py(D_per_bond));
             },
             py::arg("bonds"), py::arg("D_per_bond"),
             py::return_value_policy::reference_internal)
        .def("zeeman",
             [](HamiltonianBuilder& self, py::tuple h) -> HamiltonianBuilder& {
                 if (h.size() != 3) {
                     throw std::invalid_argument(
                         "zeeman h must be a length-3 tuple");
                 }
                 return self.zeeman({h[0].cast<double>(),
                                     h[1].cast<double>(),
                                     h[2].cast<double>()});
             },
             py::arg("h"),
             py::return_value_policy::reference_internal)
        .def("zeeman_per_site",
             [](HamiltonianBuilder& self, py::iterable h_per_site)
                 -> HamiltonianBuilder& {
                 return self.zeeman_per_site(vec3_from_py(h_per_site));
             },
             py::arg("h_per_site"),
             py::return_value_policy::reference_internal)
        .def("on_site_field",
             [](HamiltonianBuilder& self, double h_z) -> HamiltonianBuilder& {
                 return self.on_site_field(h_z);
             },
             py::arg("h_z"),
             py::return_value_policy::reference_internal)
        .def("ring_exchange",
             [](HamiltonianBuilder& self, py::iterable plaquettes, double K)
                 -> HamiltonianBuilder& {
                 return self.ring_exchange(plaquettes_from_py(plaquettes), K);
             },
             py::arg("plaquettes"), py::arg("K") = 1.0,
             py::return_value_policy::reference_internal)
        .def("pyrochlore_non_kramers",
             [](HamiltonianBuilder& self, const Lattice& lat,
                double Jxx, double Jyy, double Jzz, bool include_isotropic)
                 -> HamiltonianBuilder& {
                 return self.pyrochlore_non_kramers(lat, Jxx, Jyy, Jzz,
                                                    include_isotropic);
             },
             py::arg("lattice"), py::arg("Jxx"), py::arg("Jyy"),
             py::arg("Jzz"), py::arg("include_isotropic") = true,
             py::return_value_policy::reference_internal)

        // Output paths ---------------------------------------------------
        // `to_operator()` returns a freshly-allocated unique-owned Operator
        // (matching the default pybind11 holder for the `Operator` class
        // bound in qed_bindings.cpp). The C++ API returns a
        // shared_ptr; we copy-construct into a unique_ptr so Python takes
        // sole ownership and there is no shared_ptr/unique_ptr holder
        // conflict.
        .def("to_operator",
             [](const HamiltonianBuilder& self) {
                 auto op = std::make_unique<Operator>(
                     static_cast<uint64_t>(self.num_sites()),
                     static_cast<float>(self.spin()));
                 self.emit_into(*op);
                 return op;
             },
             "Materialise into an in-memory `qed.Operator`. No file I/O.")
        .def("emit_into",
             [](const HamiltonianBuilder& self, Operator& op) {
                 self.emit_into(op);
             },
             py::arg("operator"),
             "Append the accumulated terms onto an existing operator.")
        .def("write_directory",
             [](const HamiltonianBuilder& self, const std::string& output_dir,
                const Lattice* lat, const FileOptions& opts) {
                 self.write_directory(output_dir, lat, opts);
             },
             py::arg("output_dir"),
             py::arg("lattice") = static_cast<const Lattice*>(nullptr),
             py::arg("opts") = FileOptions{},
             "Write the legacy directory format (`InterAll.dat`, `Trans.dat`, "
             "`ThreeBodyG.dat`, `positions.dat`) consumed by `./ED <dir>`.")

        // Inspection -----------------------------------------------------
        .def_property_readonly("num_sites", &HamiltonianBuilder::num_sites)
        .def_property_readonly("spin", &HamiltonianBuilder::spin)
        .def_property_readonly("l1_norm", &HamiltonianBuilder::l1_norm)
        .def("clear", &HamiltonianBuilder::clear,
             py::return_value_policy::reference_internal)
        .def("__len__", [](const HamiltonianBuilder& self) {
            return self.one_body_terms().size() +
                   self.two_body_terms().size() +
                   self.three_body_terms().size();
        });

    // ---------------------------------------------------------------------
    // Free file writers (low-level escape hatches matching file_io.h)
    // ---------------------------------------------------------------------
    py::module_ mio = m.def_submodule(
        "io",
        "Low-level Trans.dat / InterAll.dat / ThreeBodyG.dat / positions.dat "
        "writers (most users want HamiltonianBuilder.write_directory).");

    mio.def("write_one_body_correlation_file",
            &ed::input::write_one_body_correlation_file,
            py::arg("path"), py::arg("op"), py::arg("num_sites"));
    mio.def("write_two_body_correlation_file",
            &ed::input::write_two_body_correlation_file,
            py::arg("path"), py::arg("op_i"), py::arg("op_j"),
            py::arg("num_sites"));
    mio.def("write_positions_file", &ed::input::write_positions_file,
            py::arg("path"), py::arg("positions"));
    mio.def("write_momentum_observable_file",
            [](const std::string& path, Op op,
               const std::array<double, 3>& q,
               const std::vector<Position>& positions) {
                ed::input::write_momentum_observable_file(path, op, q, positions);
            },
            py::arg("path"), py::arg("op"), py::arg("q"),
            py::arg("positions"));
}
