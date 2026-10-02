// =============================================================================
// python/qed/_bindings/core.cpp -- the pybind11 module `qed._core`.
//
//   * Operator: the spin-1/2 Hamiltonian / observable builder (terms,
//     apply, and the term iterators symmetry discovery reads);
//   * input (input.cpp): lattices and the Hamiltonian DSL;
//   * sectors (sectors.cpp): the symmetry-sector verbs behind qed._verbs;
//   * dssf, symmetry: observable assembly and site-permutation helpers;
//   * the environment registry (env_*) and build / device probes;
//   * the log bridge (log_*) behind qed.set_log_level, and the translation of the
//     ed:: error types (ed/core/errors.h) into qed.errors.
//
// Complex vectors cross as numpy complex128 arrays; long solves release the GIL.
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include <ed/core/config.h>
#include <ed/ops/construct_ham.h>
#include <ed/core/errors.h>
#include <ed/core/log.h>
#include <ed/core/select_backend.h>
#include <ed/dssf/operator_spec.h>
#include <ed/ops/invariance.h>
#include <ed/basis/group.h>

#include "bindings.h"

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef WITH_CUDA
#include <cuda_runtime.h>                     // cudaGetDeviceCount
#endif

namespace py = pybind11;

namespace {

using Complex = std::complex<double>;
using ComplexVec = std::vector<Complex>;
using ComplexArray = py::array_t<Complex, py::array::c_style | py::array::forcecast>;

// Helpers to convert NumPy arrays <-> std::vector<Complex>.
ComplexVec from_numpy(const ComplexArray& arr) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(
            "expected a 1-D complex128 array, got ndim=" +
            std::to_string(arr.ndim()));
    }
    const auto n = static_cast<size_t>(arr.shape(0));
    const Complex* p = arr.data();
    return ComplexVec(p, p + n);
}

ComplexArray to_numpy(const ComplexVec& v) {
    ComplexArray out(static_cast<py::ssize_t>(v.size()));
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(Complex));
    return out;
}

// Add a one-site term op[site] with coefficient `coeff` to an Operator's
// transform_data_ in the same Structure-of-Arrays format that the C++
// fixtures use.
void op_add_one_body(Operator& op,
                     int op_type,
                     uint64_t site,
                     Complex coeff) {
    if (op_type < 0 || op_type > 2) {
        throw std::invalid_argument("op_type must be 0=S+, 1=S-, or 2=Sz");
    }
    if (site >= op.getNumBits()) {
        throw std::out_of_range("site index >= num_sites");
    }
    Operator::TransformData t;
    t.op_type = static_cast<uint8_t>(op_type);
    t.site_index = site;
    t.coefficient = coeff;
    t.is_two_body = false;
    op.transform_data_.push_back(t);
    // Mark the SoA + isReal() + matvec backend caches stale so a subsequent
    // apply()/isReal() rebuilds. The size-aware commitPendingTransforms()
    // would also catch this, but invalidating eagerly here also
    // resets the isReal() cache --- without this, a real-coeff operator that
    // had isReal() probed once will keep claiming real even after a complex
    // coefficient is added, and apply() would take the real specialisation
    // with the wrong matvec.
    op.invalidateMatrixCaches();
}

void op_add_two_body(Operator& op,
                     int op_type_1, uint64_t site_1,
                     int op_type_2, uint64_t site_2,
                     Complex coeff) {
    if (op_type_1 < 0 || op_type_1 > 2 || op_type_2 < 0 || op_type_2 > 2) {
        throw std::invalid_argument("op_type must be 0=S+, 1=S-, or 2=Sz");
    }
    if (site_1 >= op.getNumBits() || site_2 >= op.getNumBits()) {
        throw std::out_of_range("site index >= num_sites");
    }
    Operator::TransformData t;
    t.op_type = static_cast<uint8_t>(op_type_1);
    t.site_index = site_1;
    t.op_type_2 = static_cast<uint8_t>(op_type_2);
    t.site_index_2 = site_2;
    t.coefficient = coeff;
    t.is_two_body = true;
    op.transform_data_.push_back(t);
    op.invalidateMatrixCaches();
}

void op_add_three_body(Operator& op,
                       int op_type_1, uint64_t site_1,
                       int op_type_2, uint64_t site_2,
                       int op_type_3, uint64_t site_3,
                       Complex coeff) {
    if (op_type_1 < 0 || op_type_1 > 2 ||
        op_type_2 < 0 || op_type_2 > 2 ||
        op_type_3 < 0 || op_type_3 > 2) {
        throw std::invalid_argument("op_type must be 0=S+, 1=S-, or 2=Sz");
    }
    if (site_1 >= op.getNumBits() ||
        site_2 >= op.getNumBits() ||
        site_3 >= op.getNumBits()) {
        throw std::out_of_range("site index >= num_sites");
    }
    Operator::ThreeBodyTransformData t;
    t.op_type_1 = static_cast<uint8_t>(op_type_1);
    t.site_index_1 = site_1;
    t.op_type_2 = static_cast<uint8_t>(op_type_2);
    t.site_index_2 = site_2;
    t.op_type_3 = static_cast<uint8_t>(op_type_3);
    t.site_index_3 = site_3;
    t.coefficient = coeff;
    op.three_body_data_.push_back(t);
    op.invalidateMatrixCaches();
}

ComplexArray op_apply(const Operator& op, const ComplexArray& vin) {
    auto v = from_numpy(vin);
    const uint64_t n = (1ULL << op.getNumBits());
    if (v.size() != n) {
        throw std::invalid_argument(
            "input vector length " + std::to_string(v.size()) +
            " != Hilbert dim 2^N = " + std::to_string(n));
    }
    ComplexVec out(n, Complex(0.0, 0.0));
    op.apply(v.data(), out.data(), n);
    return to_numpy(out);
}


// =============================================================================
// Term iterators, read by symmetry discovery (qed.discovery).
// =============================================================================

// Yields (op_type, site, coeff) tuples for every one-body term.
py::list op_iter_one_body(const Operator& op) {
    py::list out;
    for (const auto& t : op.transform_data_) {
        if (t.is_two_body) continue;
        out.append(py::make_tuple(static_cast<int>(t.op_type),
                                  static_cast<uint64_t>(t.site_index),
                                  t.coefficient));
    }
    return out;
}

// Yields (op_type_1, site_1, op_type_2, site_2, coeff) tuples for every
// two-body term.
py::list op_iter_two_body(const Operator& op) {
    py::list out;
    for (const auto& t : op.transform_data_) {
        if (!t.is_two_body) continue;
        out.append(py::make_tuple(static_cast<int>(t.op_type),
                                  static_cast<uint64_t>(t.site_index),
                                  static_cast<int>(t.op_type_2),
                                  static_cast<uint64_t>(t.site_index_2),
                                  t.coefficient));
    }
    return out;
}

// Yields (op_type_1, site_1, op_type_2, site_2, op_type_3, site_3, coeff)
// tuples for every three-body term.
py::list op_iter_three_body(const Operator& op) {
    py::list out;
    for (const auto& t : op.three_body_data_) {
        out.append(py::make_tuple(static_cast<int>(t.op_type_1),
                                  static_cast<uint64_t>(t.site_index_1),
                                  static_cast<int>(t.op_type_2),
                                  static_cast<uint64_t>(t.site_index_2),
                                  static_cast<int>(t.op_type_3),
                                  static_cast<uint64_t>(t.site_index_3),
                                  t.coefficient));
    }
    return out;
}


// Raise qed.errors.<name>(what); the builtin it derives from if qed.errors cannot be
// imported (the extension loaded outside the qed package).
void set_qed_error(const char* name, const char* what, PyObject* fallback) {
    try {
        py::object cls = py::module_::import("qed.errors").attr(name);
        PyErr_SetString(cls.ptr(), what);
    } catch (py::error_already_set&) {
        PyErr_SetString(fallback, what);
    }
}

void translate_ed_errors(std::exception_ptr p) {
    try {
        if (p) std::rethrow_exception(p);
    } catch (const ed::EmptySelection& e) {
        set_qed_error("EmptySelection", e.what(), PyExc_ValueError);
    } catch (const ed::InvalidRequest& e) {
        set_qed_error("InvalidRequest", e.what(), PyExc_ValueError);
    } catch (const ed::Unsupported& e) {
        set_qed_error("Unsupported", e.what(), PyExc_NotImplementedError);
    } catch (const ed::DeviceUnavailable& e) {
        set_qed_error("DeviceUnavailable", e.what(), PyExc_RuntimeError);
    } catch (const ed::DeviceUnsupported& e) {
        set_qed_error("DeviceUnsupported", e.what(), PyExc_RuntimeError);
    } catch (const ed::ResourceLimit& e) {
        set_qed_error("ResourceLimit", e.what(), PyExc_MemoryError);
    } catch (const ed::ConvergenceError& e) {
        set_qed_error("ConvergenceError", e.what(), PyExc_RuntimeError);
    }
}

int cuda_device_count() {
#ifdef WITH_CUDA
    int n = 0;
    if (cudaGetDeviceCount(&n) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return n;
#else
    return 0;
#endif
}


} // namespace

PYBIND11_MODULE(_core, m) {
    // ed:: error types -> qed.errors (python/qed/errors.py). Anything else falls through
    // to pybind11's standard translation.
    py::register_exception_translator(&translate_ed_errors);

    m.doc() =
        "qed._core: pybind11 binding for the C++ exact-diagonalization "
        "engine. See qed.__init__ for the user-facing facade.";

    // Operator op-type constants. Keep in sync with TransformData::op_type.
    m.attr("OP_SPLUS")  = py::int_(0);
    m.attr("OP_SMINUS") = py::int_(1);
    m.attr("OP_SZ")     = py::int_(2);

    // Standalone ed_input C++ library bindings (lattice generators +
    // HamiltonianBuilder + low-level file writers). Mounted under
    // `qed._core.input`; re-exported as `qed.input` from
    // the Python facade.
    bind_input(m);


    py::class_<Operator>(m, "Operator", R"pbdoc(
        Spin-1/2 Hamiltonian builder backed by the C++ matrix-free apply().

        Parameters
        ----------
        num_sites : int
            Number of spin-1/2 sites (must satisfy ``num_sites < 64``).
    )pbdoc")
        .def(py::init([](uint64_t num_sites) { return std::make_unique<Operator>(num_sites, 0.5f); }),
             py::arg("num_sites"))
        .def_property_readonly("num_sites", &Operator::getNumBits)
        .def_property_readonly("dimension",
                               [](const Operator& op) -> uint64_t {
                                   return 1ULL << op.getNumBits();
                               },
                               "Full Hilbert-space dimension 2^num_sites.")
        .def("add_one_body", &op_add_one_body,
             py::arg("op_type"), py::arg("site"), py::arg("coeff"),
             "Append a one-body term `coeff * Op[site]`. op_type is one of "
             "OP_SPLUS, OP_SMINUS, OP_SZ.")
        .def("add_two_body", &op_add_two_body,
             py::arg("op_type_1"), py::arg("site_1"),
             py::arg("op_type_2"), py::arg("site_2"),
             py::arg("coeff"),
             "Append a two-body term `coeff * Op1[site_1] Op2[site_2]`.")
        .def("transform_tuples",
             [](const Operator& op) {
                 // The one-/two-body terms in the canonical (op_type,
                 // site, coeff, is_two_body, op_type_2, site_2) layout,
                 // a list of 6-tuples mirroring ``Operator::TransformData``
                 // (three-body terms are not included).
                 py::list out;
                 for (const auto& t : op.transform_data_) {
                     out.append(py::make_tuple(
                         static_cast<int>(t.op_type),
                         t.site_index,
                         t.coefficient,
                         t.is_two_body,
                         static_cast<int>(t.op_type_2),
                         t.site_index_2));
                 }
                 return out;
             },
             R"pbdoc(
             Return the operator's one-/two-body terms as a list of
             6-tuples ``(op_type, site, coeff, is_two_body, op_type_2,
             site_2)``. Used by symmetry discovery (qed.discovery) to
             read the operator's TransformData without exposing the SoA
             internals directly. Three-body terms are not included.
             )pbdoc")
        .def("add_three_body", &op_add_three_body,
             py::arg("op_type_1"), py::arg("site_1"),
             py::arg("op_type_2"), py::arg("site_2"),
             py::arg("op_type_3"), py::arg("site_3"),
             py::arg("coeff"),
             "Append a three-body term `coeff * Op1[s1] Op2[s2] Op3[s3]`.")
        .def("apply", &op_apply,
             py::arg("vec"),
             "Compute H * v on a 1-D complex128 array.")
        // In-process introspection used by symmetry discovery (qed.discovery).
        .def("iter_one_body_terms", &op_iter_one_body,
             "List of ``(op_type, site, coeff)`` tuples for every one-body "
             "term currently in the operator. ``op_type`` is one of "
             "``OP_SPLUS`` / ``OP_SMINUS`` / ``OP_SZ``. Order matches the "
             "internal ``transform_data_`` storage order.")
        .def("iter_two_body_terms", &op_iter_two_body,
             "List of ``(op_type_1, site_1, op_type_2, site_2, coeff)`` "
             "tuples for every two-body term. Same ordering convention as "
             "``iter_one_body_terms``.")
        .def("iter_three_body_terms", &op_iter_three_body,
             "List of ``(op_type_1, site_1, op_type_2, site_2, op_type_3, "
             "site_3, coeff)`` tuples for every three-body term.");

    m.def("have_cuda", [] { return ed::have_cuda(); },
          "True when this build has CUDA support AND a device is present "
          "(the same gate the engine's GPU rep-gather consults).");
    m.def("env_dump", [](const std::string& prefix) { return ed::env::dump(prefix.c_str()); },
          py::arg("prefix") = "",
          "Every registered ED_* / QED_* environment variable whose name starts with "
          "`prefix`: live value | default | meaning. The table is "
          "include/ed/core/config.h. Paste into bug reports.");
    m.def("env_snapshot", [] {
              py::dict d;
              for (const auto& kv : ed::env::snapshot()) d[py::str(kv.first)] = kv.second;
              return d;
          },
          "{name: value} for the registered environment variables that are set -- the "
          "environment-dependent inputs of this run, for result metadata.");
    m.def("env_unknown", [] { return ed::env::unknown(); },
          "ED_* / QED_* names present in the environment that the registry does not "
          "declare. Nothing reads them: almost always a misspelt variable.");
    m.def("env_names", [] {
              std::vector<std::string> out;
              for (const auto& r : ed::env::rows()) out.emplace_back(r.name);
              return out;
          },
          "Names of all registered environment variables.");
    m.def("has_cuda_build", [] {
#ifdef WITH_CUDA
              return true;
#else
              return false;
#endif
          },
          "True when this build was compiled with CUDA (a device may still be absent).");
    m.def("cuda_device_count", &cuda_device_count,
          "Visible CUDA devices: 0 on a CPU build or when cudaGetDeviceCount fails.");

    // Log bridge (ed/core/log.h). Python owns the configuration; see python/qed/_log.py.
    m.def("log_configure", [](int level, int fd) {
              if (level < 0 || level > static_cast<int>(ed::logging::Level::Debug))
                  throw ed::InvalidRequest("log level must be 0 (off) .. 4 (debug)");
              ed::logging::set_stream(fd == 1 ? stdout : fd == 2 ? stderr : nullptr);
              ed::logging::set_level(static_cast<ed::logging::Level>(level));
          },
          py::arg("level"), py::arg("fd") = 0,
          "Set the engine's log level (0 off .. 4 debug) and sink: fd 1 / 2 writes each "
          "record to stdout / stderr at once, 0 queues them for log_drain().");
    m.def("log_level", [] { return static_cast<int>(ed::logging::level()); },
          "The engine's log level, 0 (off) .. 4 (debug).");
    m.def("log_drain", [] {
              std::vector<std::pair<int, std::string>> out;
              for (auto& r : ed::logging::drain())
                  out.emplace_back(static_cast<int>(r.level), std::move(r.message));
              return out;
          },
          "The queued (level, message) records, oldest first; empties the queue.");
    m.def("check_generators_commute",
          [](const Operator& op, const std::vector<std::vector<int>>& generators) {
              const ed::ops::MaskedOperator h = ed::ops::masked(op);
              std::vector<bool> out;
              out.reserve(generators.size());
              for (const auto& g : generators) out.push_back(ed::ops::commutes_with_permutation(h, g));
              return out;
          },
          py::arg("op"), py::arg("generators"),
          "Per permutation: [H, U_g] = 0, compared on H's canonical terms (exact, no matvec)?");

    bind_sectors(m);

    // ed::dssf -- structure-factor observable assembly.
    auto m_dssf = m.def_submodule("dssf",
        "Bindings for the ed::dssf C++ library: assemble the momentum-resolved spin "
        "operators of DSSF/SSSF evaluations, one per (Q, component).");

    py::class_<ed::dssf::OperatorSpec>(m_dssf, "OperatorSpec", R"pbdoc(
        Parameter object for ``build_observables``.

        Mirrors the C++ ``ed::dssf::OperatorSpec`` 1:1; see
        ``include/ed/dssf/operator_spec.h`` for the field-by-field
        documentation. Construct the spec, set the fields you care about,
        then pass it to :func:`build_observables`.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("operator_type",     &ed::dssf::OperatorSpec::operator_type)
        .def_readwrite("basis",             &ed::dssf::OperatorSpec::basis)
        .def_readwrite("components",        &ed::dssf::OperatorSpec::components)
        .def_readwrite("momentum_points",   &ed::dssf::OperatorSpec::momentum_points)
        .def_readwrite("polarization",      &ed::dssf::OperatorSpec::polarization)
        .def_readwrite("theta",             &ed::dssf::OperatorSpec::theta)
        .def_readwrite("unit_cell_size",    &ed::dssf::OperatorSpec::unit_cell_size)
        .def_readwrite("num_sites",         &ed::dssf::OperatorSpec::num_sites)
        .def_readwrite("positions_file",    &ed::dssf::OperatorSpec::positions_file)
        .def_readwrite("sublattice",        &ed::dssf::OperatorSpec::sublattice)
        .def("__repr__", [](const ed::dssf::OperatorSpec& s) {
            return "<qed.dssf.OperatorSpec operator_type='" +
                   s.operator_type + "' basis='" + s.basis +
                   "' num_sites=" + std::to_string(s.num_sites) +
                   " momenta=" + std::to_string(s.momentum_points.size()) +
                   " components=" + std::to_string(s.components.size()) +
                   ">";
        });

    py::class_<ed::dssf::Observables>(m_dssf, "Observables", R"pbdoc(
        Result of :func:`build_observables`: two parallel lists of equal length,

        - ``operators`` (list[Operator]): one operator per (Q, component), to pass
          to :func:`qed.dynamics` or :func:`qed.expect`;
        - ``names`` (list[str]): the byte-stable name of each.
    )pbdoc")
        .def_readonly("operators", &ed::dssf::Observables::operators)
        .def_readonly("names", &ed::dssf::Observables::names)
        .def("__len__", [](const ed::dssf::Observables& o) {
            return o.names.size();
        });

    m_dssf.def("build_observables",
        &ed::dssf::build_observables,
        py::arg("spec"),
        R"pbdoc(
        Build the DSSF/SSSF observables requested by ``spec``.

        The single source of DSSF/SSSF observable names and ordering.

        Returns
        -------
        Observables
            Parallel lists of operators / names. Length is the number of
            observables the builder emitted (operator_type x momentum_points x
            components x sublattices; two per entry for the transverse types).

        Raises
        ------
        ValueError
            On unrecognized ``operator_type`` or shape-mismatched inputs
            (see ``ed::dssf::build_observables`` documentation).
        )pbdoc");

    m_dssf.def("compute_transverse_bases",
        [](const std::vector<double>& Q,
           const std::vector<double>& polarization) {
            const auto [e1, e2] = ed::dssf::compute_transverse_bases(Q, polarization);
            return py::make_tuple(
                std::vector<double>{e1[0], e1[1], e1[2]},
                std::vector<double>{e2[0], e2[1], e2[2]});
        },
        py::arg("Q"), py::arg("polarization"),
        R"pbdoc(
        Compute the (e1, e2) basis used for transverse-component DSSF
        operators at one momentum point.

        - ``e1`` is the polarization vector itself (observables named ``..._NSF``).
        - ``e2 = normalize(Q × polarization)`` (observables named ``..._SF``), with
          a fallback to ``{y, polarization}`` or ``{x, polarization}`` when ``Q``
          is parallel to ``polarization``.

        Returns
        -------
        (e1, e2) : tuple[list[float], list[float]]
            Two unit 3-vectors.
        )pbdoc");

    // ed::sym -- programmatic site-permutation symmetry DSL.
    auto m_sym = m.def_submodule("symmetry",
        "Site permutations: identity, composition, powers, order, translations, "
        "reflections, swaps, and the closure of a generating set.");

    m_sym.def("identity", &ed::sym::identity, py::arg("n_sites"),
        "Identity permutation on `n_sites` sites.");
    m_sym.def("compose", &ed::sym::compose, py::arg("a"), py::arg("b"),
        "Composition (a o b)[i] = a[b[i]]. b is applied first.");
    m_sym.def("power", &ed::sym::power, py::arg("g"), py::arg("k"),
        "g^k for k >= 0; g^0 is the identity.");
    m_sym.def("order", &ed::sym::order, py::arg("g"),
        "Smallest positive integer k with g^k == identity.");
    m_sym.def("translation", &ed::sym::translation,
        py::arg("n_sites"), py::arg("shift") = 1,
        "Cyclic translation by `shift` sites on a 1D ring of `n_sites` sites.");
    m_sym.def("reflection_1d", &ed::sym::reflection_1d, py::arg("n_sites"),
        "Spatial reflection on a 1D chain: site i goes to site n_sites-1-i.");
    m_sym.def("site_swap", &ed::sym::site_swap,
        py::arg("n_sites"), py::arg("a"), py::arg("b"),
        "Permutation that swaps sites a and b; identity elsewhere.");
    m_sym.def("generate_group", &ed::sym::generate_group,
        py::arg("generators"),
        "Expand a list of generators into the full group (BFS). The result "
        "is sorted lexicographically for deterministic ordering.");

}
