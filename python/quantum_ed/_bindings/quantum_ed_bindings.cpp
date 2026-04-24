// =============================================================================
// python/quantum_ed/_bindings/quantum_ed_bindings.cpp
//
// pybind11 binding module `quantum_ed._core`.
//
// What we expose (Phase 1, deliberately a small surface):
//   * Operator                     -- spin-1/2 Hamiltonian builder.
//                                     methods: add_one_body, add_two_body,
//                                              add_three_body, load_trans,
//                                              load_inter_all, apply, num_sites,
//                                              dimension, hilbert_dim
//   * FixedSzOperator              -- same builder restricted to a fixed Sz
//                                     sector.
//   * full_diagonalization()       -- dense LAPACK eigensolve via apply().
//   * lanczos()                    -- iterative Lanczos for the bottom of
//                                     the spectrum.
//   * finite_temperature_lanczos() -- FTLM thermodynamics.
//   * low_temperature_lanczos()    -- LTLM thermodynamics.
//   * hybrid_thermal_method()      -- LTLM+FTLM crossover.
//   * compute_thermodynamics_from_spectrum() -- partition-function helper.
//
// Design notes:
//   * NumPy is the only required runtime dependency on the Python side; we
//     marshal complex vectors as `numpy.ndarray[complex128]`.
//   * All long-running solvers release the GIL via `py::call_guard<py::gil_scoped_release>()`.
//   * Builder methods accept Python complex scalars (or floats) for coupling
//     constants.
//   * Operator op-types are a Python IntEnum: SP=0, SM=1, SZ=2 (matching the
//     C++ TransformData convention).
//
// P2.7 / audit "modern python interface".
// =============================================================================

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include <ed/core/construct_ham.h>
#include <ed/solvers/ftlm.h>
#include <ed/solvers/hybrid_thermal.h>
#include <ed/solvers/lanczos.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/observables.h>

#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

py::array_t<double> to_numpy_d(const std::vector<double>& v) {
    py::array_t<double> out(static_cast<py::ssize_t>(v.size()));
    std::memcpy(out.mutable_data(), v.data(), v.size() * sizeof(double));
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

ComplexArray fop_apply(const FixedSzOperator& op, const ComplexArray& vin) {
    auto v = from_numpy(vin);
    const uint64_t d = op.getFixedSzDim();
    if (v.size() != d) {
        throw std::invalid_argument(
            "input vector length " + std::to_string(v.size()) +
            " != fixed-Sz dim " + std::to_string(d));
    }
    ComplexVec out(d, Complex(0.0, 0.0));
    op.apply(v.data(), out.data(), d);
    return to_numpy(out);
}

// A trivial std::function adapter that calls op.apply() for the solvers below.
template <typename Op>
std::function<void(const Complex*, Complex*, int)>
make_hv(const Op& op) {
    const Op* p = &op;
    return [p](const Complex* in, Complex* out, int n) {
        p->apply(in, out, static_cast<size_t>(n));
    };
}

py::array_t<double>
py_full_diag(const Operator& op,
             uint64_t num_eigs,
             const std::string& output_dir) {
    const uint64_t n = 1ULL << op.getNumBits();
    if (num_eigs == 0 || num_eigs > n) num_eigs = n;
    std::vector<double> eigs;
    {
        py::gil_scoped_release release;
        full_diagonalization(make_hv(op), n, num_eigs, eigs, output_dir,
                             /*compute_eigenvectors=*/false);
    }
    return to_numpy_d(eigs);
}

py::array_t<double>
py_lanczos(const Operator& op,
           uint64_t max_iter,
           uint64_t exct,
           double tolerance,
           const std::string& output_dir) {
    const uint64_t n = 1ULL << op.getNumBits();
    std::vector<double> eigs;
    {
        py::gil_scoped_release release;
        lanczos(make_hv(op), n, max_iter, exct, tolerance, eigs, output_dir,
                /*compute_eigenvectors=*/false);
    }
    return to_numpy_d(eigs);
}

py::dict thermo_to_dict(const ThermodynamicData& t) {
    py::dict d;
    d["temperatures"]    = to_numpy_d(t.temperatures);
    d["energy"]          = to_numpy_d(t.energy);
    d["specific_heat"]   = to_numpy_d(t.specific_heat);
    d["entropy"]         = to_numpy_d(t.entropy);
    d["free_energy"]     = to_numpy_d(t.free_energy);
    return d;
}

py::dict py_compute_thermo_from_spectrum(const py::array_t<double>& eigs,
                                         double T_min,
                                         double T_max,
                                         uint64_t num_points) {
    if (eigs.ndim() != 1) {
        throw std::invalid_argument("eigenvalues must be a 1-D array");
    }
    const auto n = static_cast<size_t>(eigs.shape(0));
    std::vector<double> evec(eigs.data(), eigs.data() + n);
    auto t = calculate_thermodynamics_from_spectrum(evec, T_min, T_max, num_points);
    return thermo_to_dict(t);
}

py::dict py_finite_temperature_lanczos(const Operator& op,
                                       const FTLMParameters& params,
                                       double temp_min,
                                       double temp_max,
                                       uint64_t num_temp_bins,
                                       const std::string& output_dir) {
    const uint64_t n = 1ULL << op.getNumBits();
    FTLMResults res;
    {
        py::gil_scoped_release release;
        res = finite_temperature_lanczos(make_hv(op), n, params, temp_min,
                                         temp_max, num_temp_bins, output_dir);
    }
    py::dict d = thermo_to_dict(res.thermo_data);
    d["ground_state_estimate"] = res.ground_state_estimate;
    return d;
}

py::dict py_low_temperature_lanczos(const Operator& op,
                                    const LTLMParameters& params,
                                    double temp_min,
                                    double temp_max,
                                    uint64_t num_temp_bins,
                                    const std::string& output_dir) {
    const uint64_t n = 1ULL << op.getNumBits();
    LTLMResults res;
    {
        py::gil_scoped_release release;
        res = low_temperature_lanczos(make_hv(op), n, params, temp_min,
                                      temp_max, num_temp_bins,
                                      /*ground_state=*/nullptr, output_dir);
    }
    py::dict d = thermo_to_dict(res.thermo_data);
    d["ground_state_energy"] = res.ground_state_energy;
    return d;
}

py::dict py_hybrid_thermal(const Operator& op,
                           const HybridThermalParameters& params,
                           double temp_min,
                           double temp_max,
                           uint64_t num_temp_bins,
                           const std::string& output_dir) {
    const uint64_t n = 1ULL << op.getNumBits();
    HybridThermalResults res;
    {
        py::gil_scoped_release release;
        res = hybrid_thermal_method(make_hv(op), n, params, temp_min, temp_max,
                                    num_temp_bins, output_dir);
    }
    py::dict d = thermo_to_dict(res.thermo_data);
    d["ground_state_energy"] = res.ground_state_energy;
    d["ltlm_points"]         = res.ltlm_points;
    d["ftlm_points"]         = res.ftlm_points;
    return d;
}

} // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() =
        "quantum_ed._core: pybind11 binding for the C++ exact-diagonalization "
        "engine. See quantum_ed.__init__ for the user-facing facade.";

    // Operator op-type constants. Keep in sync with TransformData::op_type.
    m.attr("OP_SPLUS")  = py::int_(0);
    m.attr("OP_SMINUS") = py::int_(1);
    m.attr("OP_SZ")     = py::int_(2);

    py::class_<Operator>(m, "Operator", R"pbdoc(
        Spin-1/2 Hamiltonian builder backed by the C++ matrix-free apply().

        Parameters
        ----------
        num_sites : int
            Number of spins (must satisfy ``num_sites < 64``).
        spin : float, optional
            Local spin quantum number (default 0.5; only spin-1/2 is fully
            supported by the matrix-free path).
    )pbdoc")
        .def(py::init<uint64_t, float>(),
             py::arg("num_sites"),
             py::arg("spin") = 0.5f)
        .def_property_readonly("num_sites", &Operator::getNumBits)
        .def_property_readonly("spin", &Operator::getSpin)
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
        .def("add_three_body", &op_add_three_body,
             py::arg("op_type_1"), py::arg("site_1"),
             py::arg("op_type_2"), py::arg("site_2"),
             py::arg("op_type_3"), py::arg("site_3"),
             py::arg("coeff"),
             "Append a three-body term `coeff * Op1[s1] Op2[s2] Op3[s3]`.")
        .def("load_trans", &Operator::loadFromFile,
             py::arg("filename"),
             "Load one-body terms from an mVMC-style Trans.dat file.")
        .def("load_inter_all", &Operator::loadFromInterAllFile,
             py::arg("filename"),
             "Load two-body terms from an mVMC-style InterAll.dat file.")
        .def("apply", &op_apply,
             py::arg("vec"),
             "Compute H * v on a 1-D complex128 array.");

    py::class_<FixedSzOperator, Operator>(m, "FixedSzOperator", R"pbdoc(
        Spin-1/2 Hamiltonian restricted to a fixed total Sz sector.

        Parameters
        ----------
        num_sites : int
        n_up : int
            Number of up spins. Must satisfy 0 <= n_up <= num_sites.
        spin : float, optional
            Local spin quantum number (default 0.5).
    )pbdoc")
        .def(py::init([](uint64_t num_sites, int64_t n_up, float spin) {
            return std::make_unique<FixedSzOperator>(num_sites, spin, n_up);
        }),
             py::arg("num_sites"),
             py::arg("n_up"),
             py::arg("spin") = 0.5f)
        .def_property_readonly("dimension", &FixedSzOperator::getFixedSzDim,
                               "Reduced sector dimension C(num_sites, n_up).")
        .def("apply", &fop_apply, py::arg("vec"));

    // Solver wrappers ------------------------------------------------------
    m.def("full_diagonalization", &py_full_diag,
          py::arg("operator"),
          py::arg("num_eigs") = 0,
          py::arg("output_dir") = "",
          "Dense LAPACK diagonalization through the matrix-free apply().");

    m.def("lanczos", &py_lanczos,
          py::arg("operator"),
          py::arg("max_iter") = 100,
          py::arg("exct") = 3,
          py::arg("tolerance") = 1e-12,
          py::arg("output_dir") = "",
          "Plain Lanczos (no reorth) for the bottom `exct` eigenvalues.");

    m.def("compute_thermodynamics_from_spectrum",
          &py_compute_thermo_from_spectrum,
          py::arg("eigenvalues"),
          py::arg("T_min"),
          py::arg("T_max"),
          py::arg("num_points"),
          "Partition-function thermodynamics from a precomputed spectrum.");

    // FTLM ----------------------------------------------------------------
    py::class_<FTLMParameters>(m, "FTLMParameters")
        .def(py::init<>())
        .def_readwrite("krylov_dim",                  &FTLMParameters::krylov_dim)
        .def_readwrite("num_samples",                 &FTLMParameters::num_samples)
        .def_readwrite("max_iterations",              &FTLMParameters::max_iterations)
        .def_readwrite("tolerance",                   &FTLMParameters::tolerance)
        .def_readwrite("full_reorthogonalization",    &FTLMParameters::full_reorthogonalization)
        .def_readwrite("reorth_frequency",            &FTLMParameters::reorth_frequency)
        .def_readwrite("random_seed",                 &FTLMParameters::random_seed)
        .def_readwrite("store_intermediate",          &FTLMParameters::store_intermediate)
        .def_readwrite("compute_error_bars",          &FTLMParameters::compute_error_bars);

    m.def("finite_temperature_lanczos", &py_finite_temperature_lanczos,
          py::arg("operator"),
          py::arg("params"),
          py::arg("temp_min"),
          py::arg("temp_max"),
          py::arg("num_temp_bins"),
          py::arg("output_dir") = "");

    // LTLM ----------------------------------------------------------------
    py::class_<LTLMParameters>(m, "LTLMParameters")
        .def(py::init<>())
        .def_readwrite("krylov_dim",                  &LTLMParameters::krylov_dim)
        .def_readwrite("ground_state_krylov",         &LTLMParameters::ground_state_krylov)
        .def_readwrite("num_samples",                 &LTLMParameters::num_samples)
        .def_readwrite("tolerance",                   &LTLMParameters::tolerance)
        .def_readwrite("full_reorthogonalization",    &LTLMParameters::full_reorthogonalization)
        .def_readwrite("random_seed",                 &LTLMParameters::random_seed);

    m.def("low_temperature_lanczos", &py_low_temperature_lanczos,
          py::arg("operator"),
          py::arg("params"),
          py::arg("temp_min"),
          py::arg("temp_max"),
          py::arg("num_temp_bins"),
          py::arg("output_dir") = "");

    // Hybrid --------------------------------------------------------------
    py::class_<HybridThermalParameters>(m, "HybridThermalParameters")
        .def(py::init<>())
        .def_readwrite("crossover_temperature",       &HybridThermalParameters::crossover_temperature)
        .def_readwrite("ltlm_krylov_dim",             &HybridThermalParameters::ltlm_krylov_dim)
        .def_readwrite("ltlm_ground_krylov",          &HybridThermalParameters::ltlm_ground_krylov)
        .def_readwrite("ltlm_full_reorth",            &HybridThermalParameters::ltlm_full_reorth)
        .def_readwrite("ltlm_seed",                   &HybridThermalParameters::ltlm_seed)
        .def_readwrite("ftlm_krylov_dim",             &HybridThermalParameters::ftlm_krylov_dim)
        .def_readwrite("ftlm_num_samples",            &HybridThermalParameters::ftlm_num_samples)
        .def_readwrite("ftlm_full_reorth",            &HybridThermalParameters::ftlm_full_reorth)
        .def_readwrite("ftlm_seed",                   &HybridThermalParameters::ftlm_seed)
        .def_readwrite("ftlm_error_bars",             &HybridThermalParameters::ftlm_error_bars)
        .def_readwrite("tolerance",                   &HybridThermalParameters::tolerance);

    m.def("hybrid_thermal_method", &py_hybrid_thermal,
          py::arg("operator"),
          py::arg("params"),
          py::arg("temp_min"),
          py::arg("temp_max"),
          py::arg("num_temp_bins"),
          py::arg("output_dir") = "");
}
