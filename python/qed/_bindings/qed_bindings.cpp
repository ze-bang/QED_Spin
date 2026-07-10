// =============================================================================
// python/qed/_bindings/qed_bindings.cpp
//
// pybind11 binding module `qed._core`.
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
#include <ed/dssf/operator_spec.h>
#include <ed/planner/basis_policy_hook.h>   // ScopedBasisRepr / prefer_tableless_fixed_sz (leaf)
#include <ed/solvers/ftlm.h>
#include <ed/solvers/lanczos.h>
#include <ed/solvers/ltlm.h>
#include <ed/solvers/observables.h>
#include <ed/bfg/cluster.h>
#include <ed/bfg/correlations.h>
#include <ed/bfg/ring_observables.h>
#include <ed/bfg/spin_structure_factor.h>
#include <ed/bfg/structure_factor.h>
#include <ed/bfg/topology.h>
#include <ed/bfg/wavefunction_io.h>
#include <ed/symmetry/group.h>
#include <ed/symmetry/irreps.h>
#include <ed/symmetry/symmetry_adapted.h>
#include <ed/solvers/symmetry_adapted_solve.h>
#include <ed/solvers/little_group_solve.h>  // Stage 7 factorized non-abelian
#include <ed/symmetry/spin_flip.h>  // sz_axis_of (Stage 8d diagonal-axis compose)

#include "dispatcher_bindings.h"
#include "input_bindings.h"
#include "workflow_bindings.h"

#include <complex>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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
    // Mark the SoA + isReal() + matvec backend caches stale so a subsequent
    // apply()/isReal() rebuilds. The size-aware commitPendingTransforms()
    // (S0 #2) would also catch this, but invalidating eagerly here also
    // resets the isReal() cache --- without this, a real-coeff operator that
    // had isReal() probed once will keep claiming real even after a complex
    // coefficient is added, routing later lanczos() through the lanczos_real
    // fast path with the wrong matvec.
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

// =============================================================================
// Phase 9: in-process introspection helpers used by the unified workflow API
// (`qed.workflow.find_symmetries` / `qed.workflow.diag`).
//
// Without these the Python facade would have to either (a) round-trip the
// operator through `HamiltonianBuilder.write_directory` and re-parse the
// resulting `Trans.dat` / `InterAll.dat`, or (b) crack open the C++
// `transform_data_` POD layout from Python, which is brittle. Exposing
// small "iterate the terms" / "is Sz conserved?" / "clone into FixedSz"
// helpers gives the workflow layer a clean, type-safe surface.
// =============================================================================

// Returns true iff every (one-, two-, three-body) term commutes with total
// Sz. The rule is the same as the on-disk `hamiltonian_conserves_sz` in
// ed/core/ed_wrapper.h: a term preserves Sz iff its operator slots have a
// net Sz-shift of zero (S+ = +1, S- = -1, Sz = 0).
bool op_conserves_sz(const Operator& op) {
    auto sz_shift = [](int op_type) {
        if (op_type == 0) return  1;  // S+ raises by 1
        if (op_type == 1) return -1;  // S- lowers by 1
        return 0;                     // Sz is diagonal
    };

    for (const auto& t : op.transform_data_) {
        if (std::abs(t.coefficient) < 1e-15) continue;
        int delta = sz_shift(t.op_type);
        if (t.is_two_body) delta += sz_shift(t.op_type_2);
        if (delta != 0) return false;
    }
    for (const auto& t : op.three_body_data_) {
        if (std::abs(t.coefficient) < 1e-15) continue;
        int delta = sz_shift(t.op_type_1) + sz_shift(t.op_type_2) +
                    sz_shift(t.op_type_3);
        if (delta != 0) return false;
    }
    return true;
}

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

// Allocate a fresh FixedSzOperator on the same number of sites and copy
// the source operator's term lists across. The fixed-Sz operator inherits
// `transform_data_` / `three_body_data_` straight from `Operator`, so a
// member-wise copy gets us a fully working sector-restricted operator
// without having to re-add each term.
// C(n, k), overflow-clamped to UINT64_MAX (a basis that large is astronomically
// infeasible -> the clamp correctly drives the tableless / refuse decision).
[[nodiscard]] inline std::uint64_t binom_u64(unsigned n, unsigned k) {
    if (k > n) return 0;
    k = std::min(k, n - k);
    std::uint64_t r = 1;
    for (unsigned i = 0; i < k; ++i) {
        const std::uint64_t num = n - i;
        if (r > (std::numeric_limits<std::uint64_t>::max)() / num)
            return (std::numeric_limits<std::uint64_t>::max)();
        r = r * num / (i + 1);   // exact in this multiplicative order
    }
    return r;
}

std::unique_ptr<FixedSzOperator>
op_make_fixed_sz(const Operator& op, int64_t n_up) {
    if (n_up < 0 || n_up > static_cast<int64_t>(op.getNumBits())) {
        throw std::invalid_argument(
            "n_up = " + std::to_string(n_up) +
            " out of range [0, num_sites=" + std::to_string(op.getNumBits()) + "]");
    }

    // Planner removed: pick the fixed-Sz basis representation from the
    // basis_policy_hook leaf -- env ED_FIXED_SZ_TABLELESS wins, otherwise the
    // materialized C(N,n_up) default. (The cost-model "completion guarantee"
    // pre-flight is gone; set ED_FIXED_SZ_TABLELESS=1 for the tableless
    // combinadic basis when the materialized array would not fit.)
    const ed::planner::ScopedBasisRepr basis_guard(
        ed::planner::prefer_tableless_fixed_sz() ? ed::planner::BasisRepr::Tableless
                                                 : ed::planner::BasisRepr::Default);

    auto fop = std::make_unique<FixedSzOperator>(
        op.getNumBits(), op.getSpin(), n_up);
    fop->transform_data_  = op.transform_data_;
    fop->three_body_data_ = op.three_body_data_;
    fop->invalidateMatrixCaches();
    return fop;
}

// A trivial std::function adapter that calls op.apply() for the solvers below.
//
// IMPORTANT: ``Operator::apply`` is *non-virtual*, so dispatch via this lambda
// is purely static. If you template ``Op = Operator`` but pass a
// ``FixedSzOperator`` reference, the lambda will run the base-class apply on
// the FULL Hilbert space (dim = 2^N), not the Sz-projected sector. The Python
// bindings below therefore provide explicit ``FixedSzOperator`` overloads
// that template-dispatch through ``make_hv<FixedSzOperator>(op)`` and use
// ``getFixedSzDim()`` instead of ``2^N`` for the Krylov-space dim.
template <typename Op>
std::function<void(const Complex*, Complex*, int)>
make_hv(const Op& op) {
    const Op* p = &op;
    return [p](const Complex* in, Complex* out, int n) {
        p->apply(in, out, static_cast<size_t>(n));
    };
}

// Helper: dispatch dim for an arbitrary (Fixed)SzOperator. The free-function
// overload makes the FixedSz vs full-Hilbert distinction explicit at the
// callsite and avoids having to remember the rule in every wrapper below.
inline uint64_t hv_dim(const Operator&        op) { return 1ULL << op.getNumBits(); }
inline uint64_t hv_dim(const FixedSzOperator& op) { return op.getFixedSzDim(); }

// Real-arithmetic mat-vec adapter for the lanczos_real fast path. Wraps
// the operator's consolidated ``apply_real`` entry point (which itself
// picks assembled-CSR vs matrix-free internally). Caller must have
// verified ``op.isReal() == true`` before invoking; ``apply_real``
// throws otherwise.
template <class Op>
inline std::function<void(const double*, double*, int)>
make_hv_real(const Op& op) {
    const Op* p = &op;
    return [p](const double* in, double* out, int n) {
        p->apply_real(in, out, static_cast<std::size_t>(n));
    };
}

// Phase 6.1: shared HDF5-output gating for all solver wrappers.
//
// Historically a Python caller that did *not* pass ``output_dir`` got an
// empty string here, which the C++ solvers interpret as "current working
// directory" (``"."``). That triggered HDF5 file creation /
// ed_results.h5 writes on every call, paying ~1 ms / call of pure I/O
// overhead even when the user only wanted the return value (the
// canonical interactive / benchmarking pattern). The xdiag bake-off
// surfaced this as a 2-3x slowdown on small / mid-N Lanczos.
//
// Rather than bake the remap into every individual wrapper (we did this
// for ``py_lanczos`` initially), centralize it: empty -> "/dev/null",
// which the C++ HDF5 layer special-cases as "skip all I/O" (see
// ``HDF5IO::isDisabledOutputPath`` in ``ed/core/hdf5_io.h``).
// Pass ``"."`` explicitly to restore the legacy behavior.
inline std::string output_dir_or_devnull(const std::string& s) {
    return s.empty() ? std::string("/dev/null") : s;
}

py::array_t<double>
py_full_diag(const Operator& op,
             uint64_t num_eigs,
             const std::string& output_dir) {
    const uint64_t n = hv_dim(op);
    if (num_eigs == 0 || num_eigs > n) num_eigs = n;
    const std::string dir = output_dir_or_devnull(output_dir);
    std::vector<double> eigs;
    {
        py::gil_scoped_release release;
        full_diagonalization(make_hv(op), n, num_eigs, eigs, dir,
                             /*compute_eigenvectors=*/false);
    }
    return to_numpy_d(eigs);
}

py::array_t<double>
py_full_diag_fixed_sz(const FixedSzOperator& op,
                      uint64_t num_eigs,
                      const std::string& output_dir) {
    const uint64_t n = hv_dim(op);
    if (num_eigs == 0 || num_eigs > n) num_eigs = n;
    const std::string dir = output_dir_or_devnull(output_dir);
    std::vector<double> eigs;
    {
        py::gil_scoped_release release;
        full_diagonalization(make_hv(op), n, num_eigs, eigs, dir,
                             /*compute_eigenvectors=*/false);
    }
    return to_numpy_d(eigs);
}

// Phase 6 #7: dispatch eigenvalue-only Lanczos to the real-arithmetic
// fast path when the Hamiltonian is real. The real path uses double
// storage end-to-end (cuts BLAS-1 traffic and FLOPs in half vs the
// complex variant) and is the largest residual win at N >= 18 in the
// FixedSz Heisenberg benchmark vs xdiag.
//
// Opt-out via ED_LANCZOS_REAL_DISPATCH=0 (default on).
namespace {
inline bool real_lanczos_dispatch_enabled() {
    const char* env = std::getenv("ED_LANCZOS_REAL_DISPATCH");
    if (!env || env[0] == '\0') return true;
    if (std::strcmp(env, "0")     == 0) return false;
    if (std::strcmp(env, "false") == 0) return false;
    if (std::strcmp(env, "FALSE") == 0) return false;
    return true;
}
}  // namespace

py::array_t<double>
py_lanczos(const Operator& op,
           uint64_t max_iter,
           uint64_t exct,
           double tolerance,
           const std::string& output_dir) {
    const uint64_t n = hv_dim(op);
    const std::string dir = output_dir_or_devnull(output_dir);
    std::vector<double> eigs;
    {
        py::gil_scoped_release release;
        if (op.isReal() && real_lanczos_dispatch_enabled()) {
            lanczos_real(make_hv_real(op), n, max_iter, exct, tolerance, eigs);
        } else {
            lanczos(make_hv(op), n, max_iter, exct, tolerance, eigs, dir,
                    /*compute_eigenvectors=*/false);
        }
    }
    return to_numpy_d(eigs);
}

py::array_t<double>
py_lanczos_fixed_sz(const FixedSzOperator& op,
                    uint64_t max_iter,
                    uint64_t exct,
                    double tolerance,
                    const std::string& output_dir) {
    const uint64_t n = hv_dim(op);
    const std::string dir = output_dir_or_devnull(output_dir);
    std::vector<double> eigs;
    {
        py::gil_scoped_release release;
        if (op.isReal() && real_lanczos_dispatch_enabled()) {
            lanczos_real(make_hv_real(op), n, max_iter, exct, tolerance, eigs);
        } else {
            lanczos(make_hv(op), n, max_iter, exct, tolerance, eigs, dir,
                    /*compute_eigenvectors=*/false);
        }
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
    const uint64_t n = hv_dim(op);
    const std::string dir = output_dir_or_devnull(output_dir);
    FTLMResults res;
    {
        py::gil_scoped_release release;
        res = finite_temperature_lanczos(make_hv(op), n, params, temp_min,
                                         temp_max, num_temp_bins, dir);
    }
    py::dict d = thermo_to_dict(res.thermo_data);
    d["ground_state_estimate"] = res.ground_state_estimate;
    return d;
}

py::dict py_finite_temperature_lanczos_fixed_sz(const FixedSzOperator& op,
                                                const FTLMParameters& params,
                                                double temp_min,
                                                double temp_max,
                                                uint64_t num_temp_bins,
                                                const std::string& output_dir) {
    const uint64_t n = hv_dim(op);
    const std::string dir = output_dir_or_devnull(output_dir);
    FTLMResults res;
    {
        py::gil_scoped_release release;
        res = finite_temperature_lanczos(make_hv(op), n, params, temp_min,
                                         temp_max, num_temp_bins, dir);
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
    const uint64_t n = hv_dim(op);
    const std::string dir = output_dir_or_devnull(output_dir);
    LTLMResults res;
    {
        py::gil_scoped_release release;
        res = low_temperature_lanczos(make_hv(op), n, params, temp_min,
                                      temp_max, num_temp_bins,
                                      /*ground_state=*/nullptr, dir);
    }
    py::dict d = thermo_to_dict(res.thermo_data);
    d["ground_state_energy"] = res.ground_state_energy;
    return d;
}

py::dict py_low_temperature_lanczos_fixed_sz(const FixedSzOperator& op,
                                             const LTLMParameters& params,
                                             double temp_min,
                                             double temp_max,
                                             uint64_t num_temp_bins,
                                             const std::string& output_dir) {
    const uint64_t n = hv_dim(op);
    const std::string dir = output_dir_or_devnull(output_dir);
    LTLMResults res;
    {
        py::gil_scoped_release release;
        res = low_temperature_lanczos(make_hv(op), n, params, temp_min,
                                      temp_max, num_temp_bins,
                                      /*ground_state=*/nullptr, dir);
    }
    py::dict d = thermo_to_dict(res.thermo_data);
    d["ground_state_energy"] = res.ground_state_energy;
    return d;
}

} // namespace

PYBIND11_MODULE(_core, m) {
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

    // NOTE: bind_dispatcher() runs at the END of the module (after
    // Operator and FixedSzOperator are registered) because it attaches
    // `set_symmetry_info_from_dict` / `get_symmetry_info_as_dict` methods
    // to those classes via m.attr("Operator").

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
        .def("transform_tuples",
             [](const Operator& op) {
                 // SOTA cross-irrep spectral path: the Python wrapper
                 // `qed.spectral(symmetry={"observable": Op, ...})`
                 // calls this to extract the one-/two-body terms in
                 // the canonical (op_type, site, coeff, is_two_body,
                 // op_type_2, site_2) layout that
                 // ``workflows_spectral_streaming_symmetry_cross_irrep_directory``
                 // ingests. We return a list of 6-tuples mirroring
                 // ``Operator::TransformData``; three-body terms are
                 // not yet plumbed through the cross-sector observable
                 // (would need a separate scatter path).
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
             site_2)``. Used by the cross-irrep streaming-symmetry
             spectral path (qed.spectral with
             ``symmetry={"observable": Op, ...}``) to extract the
             probe O_Q's TransformData without exposing the SoA
             internals directly.
             )pbdoc")
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
             "Compute H * v on a 1-D complex128 array.")
        // Phase 9: in-process introspection used by the unified workflow API
        // (``qed.workflow.find_symmetries`` / ``.diag``).
        .def("conserves_sz", &op_conserves_sz,
             "True iff every term commutes with total Sz (U(1) symmetry). "
             "Mirrors the on-disk ``hamiltonian_conserves_sz`` check used by "
             "the C++ CLI but works on the in-memory operator directly.")
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
             "site_3, coeff)`` tuples for every three-body term.")
        .def("make_fixed_sz", &op_make_fixed_sz,
             py::arg("n_up"),
             "Return a new ``FixedSzOperator`` on the same sites with the "
             "same one-/two-/three-body terms, restricted to the Sz sector "
             "with ``n_up`` up spins. Equivalent to ``FixedSzOperator(...)`` "
             "+ replaying every ``add_one_body`` / ``add_two_body`` call, "
             "but routed through a single C++ copy of the term arrays.");

    // ------------------------------------------------------------------------
    // Non-abelian symmetry-adapted spectrum. Reduces by the FULL point group
    // (including d≥2 irreps) and returns the complete spectrum with the correct
    // physical degeneracies. `generators` is a list of site permutations
    // (length num_sites each). Correct for any group; reference-grade (works in
    // the full 2^num_sites space, so moderate sizes).
    // ------------------------------------------------------------------------
    m.def("symmetry_adapted_eigenvalues",
          [](const Operator& op, const std::vector<std::vector<int>>& generators,
             int n_up, bool use_gpu) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto max_clique = ed::sym::generate_group(generators);
              auto gi = ed::symmetry::decompose_irreps(max_clique, n_sites);
              // Full reduced spectrum via the production engine (CpuMatVecBackend
              // over op's terms); on GPU the host materialises the blocks and the
              // device runs the batched cuSOLVER eigensolve. n_up >= 0 -> fixed-Sz.
              ed::symmetry::SymAdaptedSpectrum spec;
#ifdef WITH_CUDA
              if (use_gpu)
                  spec = ed::symmetry::symmetry_adapted_spectrum_gpu(
                      op, gi, max_clique, n_sites, n_up);
              else
#endif
                  spec = ed::solvers::symmetry_adapted_full_spectrum(
                      op, gi, max_clique, n_sites, n_up);
              (void)use_gpu;
              py::dict d;
              d["eigenvalues"]     = spec.eigenvalues;
              d["block_irrep_dim"] = spec.block_irrep_dim;
              d["block_size"]      = spec.block_size;
              d["group_order"]     = gi.order;
              d["num_irreps"]      = static_cast<int>(gi.irreps.size());
              std::vector<int> dims; for (const auto& ir : gi.irreps) dims.push_back(ir.dim);
              d["irrep_dims"]      = dims;
              d["is_abelian"]      = gi.is_abelian();
              return d;
          },
          py::arg("operator"), py::arg("generators"), py::arg("n_up") = -1,
          py::arg("use_gpu") = false,
          "Symmetry-adapted spectrum of a Hamiltonian under the (possibly "
          "non-abelian) point group generated by `generators` (a list of site "
          "permutations). With `n_up >= 0`, additionally restricts to the "
          "fixed-Sz sector of that many up-spins (combined U(1)×point-group "
          "reduction); `n_up = -1` (default) uses the full Hilbert space. Returns "
          "a dict with `eigenvalues` (correct d_Γ degeneracies), per-block "
          "(`block_irrep_dim`, `block_size`), and group info. Reduces by the FULL "
          "group including 2-D+ irreps.");

    // Lowest-k eigenvalues via the symmetry reduction, solved per block by a
    // DENSE eigensolve (small blocks) or LANCZOS on the reduced matvec (large
    // blocks) -- the method is chosen by block size, decoupled from the
    // reduction. Use for ground state / low-lying of large non-abelian sectors.
    m.def("symmetry_adapted_lowest",
          [](const Operator& op, const std::vector<std::vector<int>>& generators,
             int k, int n_up, int dense_max_dim) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto max_clique = ed::sym::generate_group(generators);
              auto gi = ed::symmetry::decompose_irreps(max_clique, n_sites);
              // Drives the production engine (CpuMatVecBackend over op's terms via
              // NonAbelianSymmetryBasisPolicy) per block — no parallel matvec.
              auto spec = ed::solvers::symmetry_adapted_lowest_eigenvalues(
                  op, gi, max_clique, n_sites, k, n_up, dense_max_dim);
              py::dict d;
              d["eigenvalues"]     = spec.eigenvalues;
              d["block_irrep_dim"] = spec.block_irrep_dim;
              d["block_size"]      = spec.block_size;
              d["group_order"]     = gi.order;
              d["is_abelian"]      = gi.is_abelian();
              return d;
          },
          py::arg("operator"), py::arg("generators"), py::arg("k") = 1,
          py::arg("n_up") = -1, py::arg("dense_max_dim") = 512,
          "Lowest-`k` eigenvalues (per irrep block, recombined with d_Γ "
          "multiplicity, sorted) of a Hamiltonian under the (possibly non-abelian) "
          "point group from `generators`. Each block H_Γ is solved by a dense "
          "eigensolve when its reduced dimension n_Γ <= `dense_max_dim`, otherwise "
          "by Lanczos on the reduced matvec -- so the symmetry reduction is solved "
          "by EITHER method, chosen by block size. `n_up >= 0` adds the fixed-Sz "
          "restriction. Ground state is the robust output; for a full low-lying "
          "spectrum prefer `symmetry_adapted_eigenvalues` (dense).");

    // Finite-temperature thermodynamics via the symmetry-reduced spectrum
    // (exact canonical, d_Γ-weighted). n_up >= 0 -> combined fixed-Sz reduction.
    m.def("symmetry_adapted_lowest_eigenvalues",
          [](const Operator& op, const std::vector<std::vector<int>>& generators,
             int k, int n_up, int dense_max_dim, int sz_parity) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto group = ed::sym::generate_group(generators);
              auto gi = ed::symmetry::decompose_irreps(group, n_sites);
              auto spec = ed::solvers::symmetry_adapted_lowest_eigenvalues(
                  op, gi, group, n_sites, k, n_up, dense_max_dim,
                  ed::solvers::BlockMethod::Auto, sz_parity);
              py::dict d;
              d["eigenvalues"]     = spec.eigenvalues;
              d["block_irrep_dim"] = spec.block_irrep_dim;
              d["block_size"]      = spec.block_size;
              d["group_order"]     = gi.order;
              return d;
          },
          py::arg("operator"), py::arg("generators"), py::arg("k"),
          py::arg("n_up") = -1, py::arg("dense_max_dim") = 512,
          py::arg("sz_parity") = -1,
          R"pbdoc(
            Lowest-k eigenvalues under the FULL (possibly non-abelian)
            group reduction: each irrep block runs the production
            multi-target matvec through block-size-adaptive dense /
            Lanczos solves; eigenvalues recombined with their d_Gamma
            multiplicities and sorted. The ITERATIVE non-abelian lane
            (blocks ~ dim/|G| including d >= 2 irreps).
          )pbdoc");

    m.def("symmetry_adapted_gs_dssf",
          [](const Operator& op_h, const Operator& op_o,
             const std::vector<std::vector<int>>& generators,
             double omega_min, double omega_max, int n_omega,
             double broadening) {
              const int n_sites = static_cast<int>(op_h.getNumBits());
              auto group = ed::sym::generate_group(generators);
              auto gi = ed::symmetry::decompose_irreps(group, n_sites);
              // Stage 8d: compose the diagonal axis. When H conserves
              // U(1) Sz (or only its Z2 parity remnant), partition the
              // eigen-decomposition by n_up (or parity half): the union
              // of subspace blocks stays complete while every dense
              // solve shrinks. Same term-level detection the
              // composition layer uses everywhere else.
              std::vector<std::pair<int, int>> subspaces{{-1, -1}};
              {
                  ed::matvec::TermStorage soa;
                  ed::matvec::TermStorage::classify_route(
                      soa, op_h.transform_data_, op_h.three_body_data_,
                      [](const std::complex<double>& c) { return c; });
                  const auto ax = ed::symmetry::sz_axis_of(soa);
                  if (ax == ed::symmetry::SzAxis::U1) {
                      subspaces.clear();
                      for (int k = 0; k <= n_sites; ++k)
                          subspaces.emplace_back(k, -1);
                  } else if (ax == ed::symmetry::SzAxis::Parity) {
                      subspaces = {{-1, 0}, {-1, 1}};
                  }
              }
              auto res = ed::solvers::symmetry_adapted_ground_state_dssf(
                  op_h, op_o, gi, group, n_sites,
                  omega_min, omega_max, n_omega, broadening, subspaces);
              py::dict d;
              d["omega"]        = res.omega;
              d["s_omega"]      = res.spectral;
              d["gs_energy"]    = res.ground_energy;
              d["total_weight"] = res.total_weight;
              return d;
          },
          py::arg("op_h"), py::arg("op_o"), py::arg("generators"),
          py::arg("omega_min"), py::arg("omega_max"), py::arg("n_omega"),
          py::arg("broadening"),
          R"pbdoc(
            Ground-state DSSF S(omega) under the FULL non-abelian group
            reduction (all d_Gamma partners summed for completeness).
            Stage 8d: the diagonal axis composes automatically -- a
            U(1)- / parity-conserving H is decomposed per n_up / parity
            half (smaller dense blocks, identical physics).
          )pbdoc");

    // -----------------------------------------------------------------
    // Stage 7 (SymmetryEngine v2): FACTORIZED non-abelian reduction via
    // little co-groups. G = A ⋊ P: solve one momentum per star, project
    // the star representative's MATRIX-FREE k-sector with the little
    // co-group's (numerically decomposed) irreps -- memory O(#reps(k)),
    // never O(2^N), so this scales past the monolithic SAB cap. Every
    // refinement step degrades gracefully to the plain k0 block.
    // -----------------------------------------------------------------
    auto lg_opts = [](int n_up, int sz_parity, int dense_max_dim,
                      bool use_gpu = false, int spin_flip = -1,
                      int time_reversal = -1) {
        ed::solvers::LittleGroupOptions o;
        o.n_up          = n_up;
        o.sz_parity     = sz_parity;
        o.dense_max_dim = dense_max_dim;
        o.spin_flip     = spin_flip;
        o.time_reversal = time_reversal;
#ifdef WITH_CUDA
        o.use_gpu       = use_gpu;
#else
        (void)use_gpu;
#endif
        return o;
    };
    auto lg_stars_dict = [](const ed::solvers::LittleGroupSpectrum& s) {
        py::list stars;
        for (const auto& st : s.stars) {
            py::dict d;
            d["k0"]           = st.k0;
            d["star_size"]    = st.star_size;
            d["little_order"] = st.little_order;
            d["projected"]    = st.projected;
            d["dim_k0"]       = st.dim_k0;
            d["flip_parity"]  = st.flip_parity;
            d["tr_pairs"]     = st.tr_pairs;
            stars.append(d);
        }
        return stars;
    };

    m.def("little_group_full_spectrum",
          [lg_opts, lg_stars_dict](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int n_up, int sz_parity, bool use_gpu, int spin_flip,
             int time_reversal, int dense_max_dim) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto s = ed::solvers::little_group_full_spectrum(
                  op, abelian_group, residue_perms, n_sites,
                  lg_opts(n_up, sz_parity, dense_max_dim, use_gpu, spin_flip,
                          time_reversal));
              py::dict d;
              d["eigenvalues"]    = s.expanded();
              d["block_values"]   = s.eigenvalues;
              d["multiplicities"] = s.multiplicities;
              d["stars"]          = lg_stars_dict(s);
              d["flip_engaged"]   = s.flip_engaged;
              d["tr_engaged"]     = s.tr_engaged;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("n_up") = -1,
          py::arg("sz_parity") = -1, py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          "Full spectrum via the FACTORIZED little-co-group reduction "
          "(one momentum per star, per-irrep blocks inside the star "
          "representative's matrix-free k-sector).");

    m.def("little_group_lowest_eigenvalues",
          [lg_opts](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int k, int n_up, int sz_parity, int dense_max_dim,
             int spin_flip, int time_reversal) {
              const int n_sites = static_cast<int>(op.getNumBits());
              return ed::solvers::little_group_lowest_eigenvalues(
                  op, abelian_group, residue_perms, n_sites, k,
                  lg_opts(n_up, sz_parity, dense_max_dim, false,
                          spin_flip, time_reversal));
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("k") = 1,
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("dense_max_dim") = 64,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          "Lowest-k eigenvalues via the factorized little-co-group "
          "reduction (dense on small blocks, Lanczos on the projected "
          "matrix-free matvec otherwise); multiplicities expanded.");

    m.def("little_group_thermodynamics",
          [lg_opts](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             const std::vector<double>& temperatures,
             int n_up, int sz_parity, bool use_gpu, int spin_flip,
             int time_reversal, int dense_max_dim) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto td = ed::solvers::little_group_thermodynamics(
                  op, abelian_group, residue_perms, n_sites, temperatures,
                  lg_opts(n_up, sz_parity, dense_max_dim, use_gpu, spin_flip,
                          time_reversal));
              py::dict d;
              d["temperatures"]  = td.temperatures;
              d["energy"]        = td.energy;
              d["specific_heat"] = td.specific_heat;
              d["entropy"]       = td.entropy;
              d["free_energy"]   = td.free_energy;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("temperatures"),
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          "Exact canonical thermodynamics from the factorized "
          "little-co-group full spectrum.");

    m.def("symmetry_adapted_thermodynamics",
          [](const Operator& op, const std::vector<std::vector<int>>& generators,
             const std::vector<double>& temperatures, int n_up, bool use_gpu,
             int sz_parity) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto max_clique = ed::sym::generate_group(generators);
              auto gi = ed::symmetry::decompose_irreps(max_clique, n_sites);
              ThermodynamicData td;
#ifdef WITH_CUDA
              if (use_gpu)
                  td = ed::symmetry::symmetry_adapted_thermodynamics_gpu(
                      op, gi, max_clique, n_sites, temperatures, n_up);
              else
#endif
                  td = ed::solvers::symmetry_adapted_thermodynamics(
                      op, gi, max_clique, n_sites, temperatures, n_up, sz_parity);
              (void)use_gpu;
              py::dict d;
              d["temperatures"] = td.temperatures;
              d["energy"]       = td.energy;
              d["specific_heat"]= td.specific_heat;
              d["entropy"]      = td.entropy;
              d["free_energy"]  = td.free_energy;
              return d;
          },
          py::arg("operator"), py::arg("generators"), py::arg("temperatures"),
          py::arg("n_up") = -1, py::arg("use_gpu") = false,
          py::arg("sz_parity") = -1,
          "Exact canonical thermodynamics (E, C, S, F vs T) of a Hamiltonian "
          "reduced by the (possibly non-abelian) point group, with optional "
          "fixed-Sz restriction (n_up>=0). Diagonalises the small per-irrep "
          "blocks and weights eigenvalues by d_Γ.");

    // Ground-state dynamical structure factor S(ω) = Σ_n |<n|O|0>|² L(ω-(E_n-E0)),
    // symmetry-reduced. `observable` supplies O via its term list (any O, incl.
    // Sz-changing — final states over the full reduced spectrum).
    m.def("symmetry_adapted_dssf",
          [](const Operator& H, const Operator& O,
             const std::vector<std::vector<int>>& generators,
             double omega_min, double omega_max, int n_omega, double broadening) {
              const int n_sites = static_cast<int>(H.getNumBits());
              auto max_clique = ed::sym::generate_group(generators);
              auto gi = ed::symmetry::decompose_irreps(max_clique, n_sites);
              auto r = ed::solvers::symmetry_adapted_ground_state_dssf(
                  H, O, gi, max_clique, n_sites,
                  omega_min, omega_max, n_omega, broadening);
              py::dict d;
              d["omega"]         = r.omega;
              d["spectral"]      = r.spectral;
              d["ground_energy"] = r.ground_energy;
              d["total_weight"]  = r.total_weight;
              return d;
          },
          py::arg("hamiltonian"), py::arg("observable"), py::arg("generators"),
          py::arg("omega_min"), py::arg("omega_max"), py::arg("n_omega"),
          py::arg("broadening"),
          "Symmetry-reduced ground-state dynamical structure factor S(ω) for the "
          "observable `O`. Correct for any O (Sz-conserving or -changing).");

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
          "Ground-state (and lowest ``exct``) eigenvalues. Real Hamiltonians use "
          "a real-storage fast path. ``output_dir==\"\"`` disables HDF5 output "
          "(``/dev/null``); pass ``\".\"`` to write ``ed_results.h5`` as before.");

    // ``FixedSzOperator`` overload: dispatches through ``hv_dim`` /
    // ``make_hv<FixedSzOperator>``, so the Krylov space lives in the
    // ``C(N, n_up)`` Sz-projected sector instead of the full ``2^N``
    // Hilbert space. Pybind picks this overload when the first argument
    // is a ``FixedSzOperator`` instance.
    m.def("lanczos", &py_lanczos_fixed_sz,
          py::arg("operator"),
          py::arg("max_iter") = 100,
          py::arg("exct") = 3,
          py::arg("tolerance") = 1e-12,
          py::arg("output_dir") = "",
          "Same as ``lanczos`` for a ``FixedSzOperator``. "
          "``output_dir==\"\"`` suppresses HDF5 (see main overload).");

    m.def("full_diagonalization", &py_full_diag_fixed_sz,
          py::arg("operator"),
          py::arg("num_eigs") = 0,
          py::arg("output_dir") = "",
          "Dense LAPACK diagonalization of a fixed-Sz operator.");

    // FTLM / LTLM / Hybrid overloads for FixedSz are registered next to
    // their base-class counterparts further down the file.

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
    m.def("finite_temperature_lanczos", &py_finite_temperature_lanczos_fixed_sz,
          py::arg("operator"),
          py::arg("params"),
          py::arg("temp_min"),
          py::arg("temp_max"),
          py::arg("num_temp_bins"),
          py::arg("output_dir") = "",
          "FTLM on a fixed-Sz sector (Krylov dim = C(N, n_up)).");

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
    m.def("low_temperature_lanczos", &py_low_temperature_lanczos_fixed_sz,
          py::arg("operator"),
          py::arg("params"),
          py::arg("temp_min"),
          py::arg("temp_max"),
          py::arg("num_temp_bins"),
          py::arg("output_dir") = "",
          "LTLM on a fixed-Sz sector (Krylov dim = C(N, n_up)).");

    // ed::dssf -- structure-factor observable assembly (P2.8 / DSSF PR-G).
    auto m_dssf = m.def_submodule("dssf",
        "Bindings for the ed::dssf C++ library: assemble DSSF/SSSF "
        "observable pairs from a parameter dict instead of hand-rolling "
        "Sum/Transverse/Sublattice/Experimental Operator constructors.");

    py::class_<ed::dssf::OperatorSpec>(m_dssf, "OperatorSpec", R"pbdoc(
        Parameter object for ``build_observable_pairs``.

        Mirrors the C++ ``ed::dssf::OperatorSpec`` 1:1; see
        ``include/ed/dssf/operator_spec.h`` for the field-by-field
        documentation. Construct the spec, set the fields you care about,
        then pass it to :func:`build_observable_pairs`.
    )pbdoc")
        .def(py::init<>())
        .def_readwrite("operator_type",     &ed::dssf::OperatorSpec::operator_type)
        .def_readwrite("basis",             &ed::dssf::OperatorSpec::basis)
        .def_readwrite("spin_combinations", &ed::dssf::OperatorSpec::spin_combinations)
        .def_readwrite("momentum_points",   &ed::dssf::OperatorSpec::momentum_points)
        .def_readwrite("polarization",      &ed::dssf::OperatorSpec::polarization)
        .def_readwrite("theta",             &ed::dssf::OperatorSpec::theta)
        .def_readwrite("unit_cell_size",    &ed::dssf::OperatorSpec::unit_cell_size)
        .def_readwrite("num_sites",         &ed::dssf::OperatorSpec::num_sites)
        .def_readwrite("spin_length",       &ed::dssf::OperatorSpec::spin_length)
        .def_readwrite("use_fixed_sz",      &ed::dssf::OperatorSpec::use_fixed_sz)
        .def_readwrite("n_up",              &ed::dssf::OperatorSpec::n_up)
        .def_readwrite("positions_file",    &ed::dssf::OperatorSpec::positions_file)
        .def_readwrite("single_obs_only",   &ed::dssf::OperatorSpec::single_obs_only)
        .def_readwrite("sublattice_filter", &ed::dssf::OperatorSpec::sublattice_filter)
        .def("__repr__", [](const ed::dssf::OperatorSpec& s) {
            return "<qed.dssf.OperatorSpec operator_type='" +
                   s.operator_type + "' basis='" + s.basis +
                   "' num_sites=" + std::to_string(s.num_sites) +
                   " momenta=" + std::to_string(s.momentum_points.size()) +
                   " combos=" + std::to_string(s.spin_combinations.size()) +
                   ">";
        });

    py::class_<ed::dssf::ObservablePairs>(m_dssf, "ObservablePairs", R"pbdoc(
        Result of :func:`build_observable_pairs`.

        Three parallel lists of equal length:

        - ``obs_1`` (list[Operator]):  left  factor of each pair ⟨ψ|O₁†...|ψ⟩.
        - ``obs_2`` (list[Operator]):  right factor of each pair ⟨...O₂|ψ⟩
                                       (empty when ``OperatorSpec.single_obs_only``).
        - ``names``  (list[str]):      legacy, byte-stable observable name
                                       used as the HDF5 group key.
    )pbdoc")
        .def_readonly("obs_1", &ed::dssf::ObservablePairs::obs_1)
        .def_readonly("obs_2", &ed::dssf::ObservablePairs::obs_2)
        .def_readonly("names", &ed::dssf::ObservablePairs::names)
        .def("__len__", [](const ed::dssf::ObservablePairs& p) {
            return p.names.size();
        });

    m_dssf.def("build_observable_pairs",
        &ed::dssf::build_observable_pairs,
        py::arg("spec"),
        R"pbdoc(
        Build the DSSF/SSSF observable pairs requested by ``spec``.

        This is the canonical entry point that the C++ ``ED dssf``
        subcommand calls internally; using it from Python guarantees
        byte-identical observable names and ordering.

        Returns
        -------
        ObservablePairs
            Parallel lists of obs_1 / obs_2 / names. Length is the number
            of pairs the builder emitted (depends on operator_type x
            momentum_points x spin_combinations x sublattice geometry).

        Raises
        ------
        ValueError
            On unrecognized ``operator_type`` or shape-mismatched inputs
            (see ``ed::dssf::build_observable_pairs`` documentation).
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

        - ``e1`` is the polarization vector itself (SF projection).
        - ``e2 = normalize(Q × polarization)`` (NSF projection), with a
          fallback to ``{y, polarization}`` or ``{x, polarization}``
          when ``Q`` is parallel to ``polarization``.

        Returns
        -------
        (e1, e2) : tuple[list[float], list[float]]
            Two unit 3-vectors.
        )pbdoc");

    // ed::sym -- programmatic site-permutation symmetry DSL (P2.11).
    auto m_sym = m.def_submodule("symmetry",
        "Bindings for the ed::sym C++ library: programmatic site-permutation "
        "symmetry groups (translation, reflection, dihedral, custom). "
        "Replaces the JSON detour through automorphism_finder.py for the "
        "common 1D / point-group cases. The returned dictionary is the "
        "bridge to the C++ engine: assign it to "
        "`Operator.symmetry_info` (when that binding lands) or persist it "
        "back through the legacy automorphism_results/ JSON files.");

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

    // group_from_generators returns SymmetryGroupInfo. We expose it as a
    // Python dict so collaborators don't need to know the C++ struct
    // internals; the dict can be re-marshalled back to JSON via the
    // automorphism_results/ schema if they want to persist it.
    m_sym.def("group_from_generators",
        [](int n_sites,
           std::vector<ed::sym::Permutation> generators,
           std::vector<std::vector<int>> sector_quantum_numbers) {
            auto info = ed::sym::group_from_generators(
                n_sites, std::move(generators),
                std::move(sector_quantum_numbers));
            py::dict d;
            d["num_generators"]       = info.num_generators;
            d["generator_orders"]     = info.generator_orders;
            d["generators"]           = info.generators;
            d["max_clique"]           = info.max_clique;
            d["power_representation"] = info.power_representation;
            py::list sectors;
            for (const auto& s : info.sectors) {
                py::dict sd;
                sd["sector_id"]       = s.sector_id;
                sd["quantum_numbers"] = s.quantum_numbers;
                py::list pf;
                for (const auto& z : s.phase_factors) {
                    pf.append(std::complex<double>(z.real(), z.imag()));
                }
                sd["phase_factors"] = pf;
                sectors.append(sd);
            }
            d["sectors"] = sectors;
            return d;
        },
        py::arg("n_sites"),
        py::arg("generators"),
        py::arg("sector_quantum_numbers") = std::vector<std::vector<int>>{},
        R"pbdoc(
        Build a fully-elaborated SymmetryGroupInfo from generators and
        return it as a dict with the same keys the JSON-driven path
        produces: ``num_generators``, ``generator_orders``,
        ``generators``, ``max_clique``, ``power_representation``,
        and ``sectors`` (list of {sector_id, quantum_numbers,
        phase_factors}). When ``sector_quantum_numbers`` is omitted,
        the full abelian product is enumerated and any phantom irreps
        produced by generator relations are removed.
        )pbdoc");

    m_sym.def("translation_group_1d",
        [](int n_sites) {
            auto info = ed::sym::translation_group_1d(n_sites);
            py::dict d;
            d["num_generators"]       = info.num_generators;
            d["generator_orders"]     = info.generator_orders;
            d["generators"]           = info.generators;
            d["max_clique"]           = info.max_clique;
            d["power_representation"] = info.power_representation;
            py::list sectors;
            for (const auto& s : info.sectors) {
                py::dict sd;
                sd["sector_id"]       = s.sector_id;
                sd["quantum_numbers"] = s.quantum_numbers;
                py::list pf;
                for (const auto& z : s.phase_factors) {
                    pf.append(std::complex<double>(z.real(), z.imag()));
                }
                sd["phase_factors"] = pf;
                sectors.append(sd);
            }
            d["sectors"] = sectors;
            return d;
        },
        py::arg("n_sites"),
        "Convenience: cyclic translation group Z_N on a 1D ring with all "
        "N momentum sectors enumerated.");

    // ed::bfg -- BFG order-parameter library helpers (P2.1).
    auto m_bfg = m.def_submodule("bfg",
        "Bindings for the ed::bfg C++ library: cluster loader, topology "
        "(triangles / bowties), and the two-body spin correlations and bond "
        "expectations used by the BFG order-parameter pipeline. Lets Python "
        "scripts share the same authoritative kernels the CPU and GPU "
        "drivers (`compute_bfg_order_parameters[_gpu]`) call internally.");

    py::class_<ed::bfg::Cluster>(m_bfg, "Cluster",
        "Geometry + connectivity of a kagome / pyrochlore-superlattice "
        "BFG cluster (read-only handle to the C++ struct).")
        .def_readonly("n_sites",          &ed::bfg::Cluster::n_sites)
        .def_readonly("positions",        &ed::bfg::Cluster::positions)
        .def_readonly("sublattice",       &ed::bfg::Cluster::sublattice)
        .def_readonly("edges_nn",         &ed::bfg::Cluster::edges_nn)
        .def_readonly("nn_list",          &ed::bfg::Cluster::nn_list)
        .def_readonly("a1",               &ed::bfg::Cluster::a1)
        .def_readonly("a2",               &ed::bfg::Cluster::a2)
        .def_readonly("b1",               &ed::bfg::Cluster::b1)
        .def_readonly("b2",               &ed::bfg::Cluster::b2)
        .def_readonly("k_points",         &ed::bfg::Cluster::k_points)
        .def_readonly("n_cells_x",        &ed::bfg::Cluster::n_cells_x)
        .def_readonly("n_cells_y",        &ed::bfg::Cluster::n_cells_y)
        .def_readonly("bond_orientation", &ed::bfg::Cluster::bond_orientation)
        .def_readonly("sites_per_cell",   &ed::bfg::Cluster::sites_per_cell)
        .def("minimum_image_displacement",
             &ed::bfg::Cluster::minimum_image_displacement,
             py::arg("i"), py::arg("j"))
        .def("bond_center_pbc",
             &ed::bfg::Cluster::bond_center_pbc,
             py::arg("i"), py::arg("j"));

    m_bfg.def("load_cluster", &ed::bfg::load_cluster, py::arg("cluster_dir"),
        "Load a Cluster from `cluster_dir` (positions.dat + optional "
        "lattice_parameters / nn_list files). Mirrors the loader the CPU "
        "and GPU drivers call internally.");

    py::class_<ed::bfg::Bowtie>(m_bfg, "Bowtie",
        "Two NN-triangles sharing exactly one vertex (`s0`). Outer "
        "vertices: (s1, s2) and (s3, s4). `center` is the mean Cartesian "
        "position of the five sites; `orientation` is the sublattice index "
        "of `s0`.")
        .def(py::init([](int s0, int s1, int s2, int s3, int s4,
                         std::array<double, 2> center, int orientation) {
            return ed::bfg::Bowtie{s0, s1, s2, s3, s4, center, orientation};
        }),
            py::arg("s0") = -1, py::arg("s1"), py::arg("s2"),
            py::arg("s3"), py::arg("s4"),
            py::arg("center") = std::array<double, 2>{0.0, 0.0},
            py::arg("orientation") = 0,
            "Construct a Bowtie from explicit fields. `s0` and `orientation` "
            "default to placeholders (the ring kernels in "
            "ed_bfg::ring_observables ignore them).")
        .def_readwrite("s0",          &ed::bfg::Bowtie::s0)
        .def_readwrite("s1",          &ed::bfg::Bowtie::s1)
        .def_readwrite("s2",          &ed::bfg::Bowtie::s2)
        .def_readwrite("s3",          &ed::bfg::Bowtie::s3)
        .def_readwrite("s4",          &ed::bfg::Bowtie::s4)
        .def_readwrite("center",      &ed::bfg::Bowtie::center)
        .def_readwrite("orientation", &ed::bfg::Bowtie::orientation);

    m_bfg.def("find_triangles", &ed::bfg::find_triangles, py::arg("cluster"),
        "Enumerate every triple (i, j, k) of pairwise nearest-neighbour "
        "sites in the cluster. Each triangle appears once with i<j<k.");

    m_bfg.def("find_bowties", &ed::bfg::find_bowties, py::arg("cluster"),
        "Enumerate every bowtie (pair of NN-triangles sharing exactly one "
        "vertex). Built on top of find_triangles.");

    // The correlation kernels accept the wavefunction as a NumPy
    // complex128 array; we copy it into std::vector<Complex> for the
    // call. Long enough that we release the GIL.
    // Helper: validate that `psi` is 1-D with exactly 2^n_sites entries.
    // Without this, an under-sized array silently produces UB reads in the
    // BFG correlation kernels (memcpy used to copy `psi.shape[0]` doubles
    // into a `v(n)` vector but then the kernel reads `1<<n_sites` of them).
    auto bfg_check_psi = [](const py::array& psi, int n_sites,
                            const char* where) {
        if (psi.ndim() != 1) {
            throw std::runtime_error(std::string(where) +
                ": psi must be a 1-D complex128 array");
        }
        if (n_sites < 0 || n_sites > 63) {
            throw std::runtime_error(std::string(where) +
                ": n_sites must be in [0, 63]");
        }
        const auto expected = (n_sites == 0)
            ? std::size_t{1}
            : (std::size_t{1} << static_cast<unsigned>(n_sites));
        const auto got = static_cast<std::size_t>(psi.shape(0));
        if (got != expected) {
            throw std::runtime_error(std::string(where) +
                ": psi length " + std::to_string(got) +
                " disagrees with 2^n_sites = " + std::to_string(expected));
        }
    };

    m_bfg.def("compute_smsp_correlations",
        [bfg_check_psi](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           int n_sites) {
            bfg_check_psi(psi, n_sites, "compute_smsp_correlations");
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_smsp_correlations(v, n_sites);
        },
        py::arg("psi"), py::arg("n_sites"),
        "Site-to-site <S^- S^+> correlation matrix.");

    m_bfg.def("compute_szsz_correlations",
        [bfg_check_psi](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           int n_sites) {
            bfg_check_psi(psi, n_sites, "compute_szsz_correlations");
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_szsz_correlations(v, n_sites);
        },
        py::arg("psi"), py::arg("n_sites"),
        "Site-to-site <S^z S^z> correlation matrix.");

    m_bfg.def("compute_xy_bond_expectations",
        [bfg_check_psi](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           const ed::bfg::Cluster& cluster) {
            bfg_check_psi(psi, cluster.n_sites, "compute_xy_bond_expectations");
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_xy_bond_expectations(v, cluster);
        },
        py::arg("psi"), py::arg("cluster"),
        "<S^+_i S^-_j + S^-_i S^+_j> per nearest-neighbour edge.");

    m_bfg.def("compute_spsm_bond_expectations",
        [bfg_check_psi](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           const ed::bfg::Cluster& cluster) {
            bfg_check_psi(psi, cluster.n_sites, "compute_spsm_bond_expectations");
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_spsm_bond_expectations(v, cluster);
        },
        py::arg("psi"), py::arg("cluster"),
        "<S^+_i S^-_j> per nearest-neighbour edge (asymmetric).");

    m_bfg.def("compute_szsz_bond_expectations",
        [bfg_check_psi](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           const ed::bfg::Cluster& cluster) {
            bfg_check_psi(psi, cluster.n_sites, "compute_szsz_bond_expectations");
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_szsz_bond_expectations(v, cluster);
        },
        py::arg("psi"), py::arg("cluster"),
        "<S^z_i S^z_j> per nearest-neighbour edge.");

    m_bfg.def("compute_heisenberg_bond_expectations",
        &ed::bfg::compute_heisenberg_bond_expectations,
        py::arg("szsz_bonds"), py::arg("xy_bonds"),
        "Combine SzSz and XY bond maps into the full Heisenberg "
        "<S_i . S_j> per edge. The XY imaginary part is dropped.");

    // ed::bfg::wavefunction_io: HDF5 wavefunction loaders shared by the
    // CPU and GPU drivers (P2.1 third slice). Returning std::vector<Complex>
    // through pybind11 produces a Python list of complex; callers that want
    // a NumPy array can wrap with `np.asarray(...)`. We marshal the result
    // into a NumPy complex128 array directly to avoid the per-element
    // conversion cost on multi-million-amplitude wavefunctions.
    auto wavefunction_to_numpy = [](std::vector<std::complex<double>>&& src) {
        py::array_t<std::complex<double>> arr(static_cast<py::ssize_t>(src.size()));
        std::memcpy(arr.mutable_data(), src.data(),
                    src.size() * sizeof(std::complex<double>));
        return arr;
    };

    m_bfg.def("load_wavefunction",
        [wavefunction_to_numpy](const std::string& filename,
                                int eigenvector_idx,
                                bool verbose) {
            std::vector<std::complex<double>> psi;
            {
                py::gil_scoped_release release;
                psi = ed::bfg::load_wavefunction(filename, eigenvector_idx,
                                                 verbose);
            }
            return wavefunction_to_numpy(std::move(psi));
        },
        py::arg("filename"),
        py::arg("eigenvector_idx") = 0,
        py::arg("verbose") = true,
        "Load a single complex eigenvector from an ED HDF5 results file. "
        "Probes both the canonical `eigendata/eigenvector_<idx>` and legacy "
        "top-level paths, and supports HDF5 compound complex types with "
        "either (real, imag) or (r, i) field names. Returns a NumPy "
        "complex128 array.");

    py::class_<ed::bfg::TPQState>(m_bfg, "TPQState",
        "A single TPQ snapshot loaded from an HDF5 results file. The "
        "wavefunction is exposed as a NumPy complex128 array; "
        "`temperature` and `beta = 1/T` accompany it.")
        .def_property_readonly("psi",
            [wavefunction_to_numpy](const ed::bfg::TPQState& self) {
                std::vector<std::complex<double>> copy(self.psi);
                return wavefunction_to_numpy(std::move(copy));
            })
        .def_readonly("temperature", &ed::bfg::TPQState::temperature)
        .def_readonly("beta",        &ed::bfg::TPQState::beta);

    m_bfg.def("load_all_tpq_states",
        [](const std::string& filename, int sample_idx, bool verbose) {
            py::gil_scoped_release release;
            return ed::bfg::load_all_tpq_states(filename, sample_idx, verbose);
        },
        py::arg("filename"),
        py::arg("sample_idx") = 0,
        py::arg("verbose") = true,
        "Load every TPQ snapshot from `tpq/samples/sample_<idx>/states/"
        "beta_*` in the file, sorted ascending in temperature.");

    m_bfg.def("load_tpq_state",
        [wavefunction_to_numpy](const std::string& filename,
                                int sample_idx, bool verbose) {
            std::vector<std::complex<double>> psi;
            double temperature{0.0};
            {
                py::gil_scoped_release release;
                auto pair = ed::bfg::load_tpq_state(filename, sample_idx,
                                                     verbose);
                psi = std::move(pair.first);
                temperature = pair.second;
            }
            return py::make_tuple(wavefunction_to_numpy(std::move(psi)),
                                  temperature);
        },
        py::arg("filename"),
        py::arg("sample_idx") = 0,
        py::arg("verbose") = true,
        "Load the lowest-temperature (highest-beta) TPQ snapshot from the "
        "file. Returns (psi, temperature).");

    // ed::bfg::structure_factor: bond-bilinear structure factors and
    // Fourier-applied dimer kernels (P2.1 fourth slice). Same NumPy
    // marshalling as the correlation kernels above; the GIL is released
    // around the C++ call because each kernel is O(N_bonds * Hilbert).
    py::class_<ed::bfg::DimerSFResult>(m_bfg, "DimerSFResult",
        "Tuple of dimer-structure-factor pieces returned by the "
        "`compute_dimer_sf_direct` / `compute_heisenberg_sf_direct` "
        "kernels: (overlap = ||D(q)|psi>||^2, expect_q1 = <D(q)>, "
        "expect_q2 = <D(q)>). The structure factor consumed by callers "
        "is `S_D(q) = overlap - |expect_q1|^2`.")
        .def_readonly("overlap",   &ed::bfg::DimerSFResult::overlap)
        .def_readonly("expect_q1", &ed::bfg::DimerSFResult::expect_q1)
        .def_readonly("expect_q2", &ed::bfg::DimerSFResult::expect_q2);

    m_bfg.def("set_memory_efficient_mode",
        &ed::bfg::set_memory_efficient_mode,
        py::arg("n_states"),
        "Enable atomic-update kernels when the thread-local working set "
        "would exceed 4 GB. Call once at startup; the flag is read by "
        "every subsequent apply_*_fourier kernel.");

    m_bfg.def("memory_efficient_mode_enabled",
        &ed::bfg::memory_efficient_mode_enabled,
        "Return whether memory-efficient (atomic-update) mode is on.");

    auto sf_kernel = [wavefunction_to_numpy](
        py::array_t<std::complex<double>,
                    py::array::c_style | py::array::forcecast> psi,
        const std::vector<std::pair<int, int>>& bonds,
        const std::vector<std::array<double, 2>>& bond_centers,
        const std::array<double, 2>& q,
        ed::bfg::DimerSFResult (*fn)(const std::vector<ed::bfg::Complex>&,
                                     const std::vector<std::pair<int, int>>&,
                                     const std::vector<std::array<double, 2>>&,
                                     const std::array<double, 2>&)) {
            if (psi.ndim() != 1) {
                throw std::runtime_error("psi must be a 1-D complex128 array");
            }
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return fn(v, bonds, bond_centers, q);
    };

    m_bfg.def("compute_dimer_sf_direct",
        [sf_kernel](py::array_t<std::complex<double>,
                                py::array::c_style | py::array::forcecast> psi,
                    const std::vector<std::pair<int, int>>& bonds,
                    const std::vector<std::array<double, 2>>& bond_centers,
                    const std::array<double, 2>& q) {
            return sf_kernel(psi, bonds, bond_centers, q,
                             &ed::bfg::compute_dimer_sf_direct);
        },
        py::arg("psi"), py::arg("bonds"), py::arg("bond_centers"), py::arg("q"),
        "S_D(q) = <D^dag(q) D(q)> for the XY dimer D = S+S- + S-S+.");

    m_bfg.def("compute_heisenberg_sf_direct",
        [sf_kernel](py::array_t<std::complex<double>,
                                py::array::c_style | py::array::forcecast> psi,
                    const std::vector<std::pair<int, int>>& bonds,
                    const std::vector<std::array<double, 2>>& bond_centers,
                    const std::array<double, 2>& q) {
            return sf_kernel(psi, bonds, bond_centers, q,
                             &ed::bfg::compute_heisenberg_sf_direct);
        },
        py::arg("psi"), py::arg("bonds"), py::arg("bond_centers"), py::arg("q"),
        "S_D(q) for the Heisenberg dimer D = S_i . S_j.");

    m_bfg.def("apply_dimer_fourier",
        [wavefunction_to_numpy](
            py::array_t<std::complex<double>,
                        py::array::c_style | py::array::forcecast> psi,
            const std::vector<std::pair<int, int>>& bonds,
            const std::vector<std::array<double, 2>>& bond_centers,
            const std::array<double, 2>& q) {
            if (psi.ndim() != 1) {
                throw std::runtime_error("psi must be a 1-D complex128 array");
            }
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            std::vector<ed::bfg::Complex> out;
            {
                py::gil_scoped_release release;
                out = ed::bfg::apply_dimer_fourier(v, bonds, bond_centers, q);
            }
            return wavefunction_to_numpy(std::move(out));
        },
        py::arg("psi"), py::arg("bonds"), py::arg("bond_centers"), py::arg("q"),
        "Apply D_XY(q) = sum_b exp(i q . r_b) (S+S- + S-S+) to psi. "
        "Returns the resulting NumPy complex128 ket.");

    m_bfg.def("apply_heisenberg_dimer_fourier",
        [wavefunction_to_numpy](
            py::array_t<std::complex<double>,
                        py::array::c_style | py::array::forcecast> psi,
            const std::vector<std::pair<int, int>>& bonds,
            const std::vector<std::array<double, 2>>& bond_centers,
            const std::array<double, 2>& q) {
            if (psi.ndim() != 1) {
                throw std::runtime_error("psi must be a 1-D complex128 array");
            }
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            std::pair<std::vector<ed::bfg::Complex>, ed::bfg::Complex> pair;
            {
                py::gil_scoped_release release;
                pair = ed::bfg::apply_heisenberg_dimer_fourier(
                    v, bonds, bond_centers, q);
            }
            return py::make_tuple(wavefunction_to_numpy(std::move(pair.first)),
                                  pair.second);
        },
        py::arg("psi"), py::arg("bonds"), py::arg("bond_centers"), py::arg("q"),
        "Apply D_H(q) = sum_b exp(i q . r_b) (S_i . S_j) to psi. "
        "Returns (D(q)|psi> as NumPy complex128, <D(q)>).");

    m_bfg.def("compute_dimer_dimer_correlation",
        [](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           int i1, int j1, int i2, int j2) {
            if (psi.ndim() != 1) {
                throw std::runtime_error("psi must be a 1-D complex128 array");
            }
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_dimer_dimer_correlation(v, i1, j1, i2, j2);
        },
        py::arg("psi"), py::arg("i1"), py::arg("j1"),
        py::arg("i2"), py::arg("j2"),
        "<psi| D_{b1} D_{b2} |psi> for the XY dimer.");

    m_bfg.def("compute_heisenberg_dimer_dimer_correlation",
        [](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           int i1, int j1, int i2, int j2) {
            if (psi.ndim() != 1) {
                throw std::runtime_error("psi must be a 1-D complex128 array");
            }
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_heisenberg_dimer_dimer_correlation(
                v, i1, j1, i2, j2);
        },
        py::arg("psi"), py::arg("i1"), py::arg("j1"),
        py::arg("i2"), py::arg("j2"),
        "<psi| (S_i1 . S_j1)(S_i2 . S_j2) |psi>.");

    // -------------------------------------------------------------------------
    // P2.1 (5th slice): ring observables (bowtie ring-flip + triangle
    // ring-exchange + Fourier-applied bowtie operator). The Fourier kernel
    // shares the memory-efficient flag bound above with the dimer kernels.
    // -------------------------------------------------------------------------
    m_bfg.def("apply_bowtie_fourier",
        [wavefunction_to_numpy](
            const std::vector<ed::bfg::Bowtie>& bowties,
            py::array_t<std::complex<double>,
                        py::array::c_style | py::array::forcecast> psi,
            const std::array<double, 2>& q) {
            if (psi.ndim() != 1) {
                throw std::runtime_error("psi must be a 1-D complex128 array");
            }
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            std::vector<ed::bfg::Complex> out;
            {
                py::gil_scoped_release release;
                out = ed::bfg::apply_bowtie_fourier(bowties, v, q);
            }
            return wavefunction_to_numpy(std::move(out));
        },
        py::arg("bowties"), py::arg("psi"), py::arg("q"),
        "Apply P(q) = sum_bt exp(i q . r_bt) (S+_1 S-_2 S+_3 S-_4 + h.c.) "
        "to psi over the supplied bowties. Only `s1..s4` and `center` are "
        "read; `s0` / `orientation` are ignored. Returns the resulting "
        "NumPy complex128 ket.");

    m_bfg.def("compute_bowtie_resonance",
        [](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           int s1, int s2, int s3, int s4) {
            if (psi.ndim() != 1) {
                throw std::runtime_error("psi must be a 1-D complex128 array");
            }
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_bowtie_resonance(v, s1, s2, s3, s4);
        },
        py::arg("psi"), py::arg("s1"), py::arg("s2"),
        py::arg("s3"), py::arg("s4"),
        "Real-space bowtie ring-flip expectation "
        "<psi| S+_1 S-_2 S+_3 S-_4 + h.c. |psi> on the four outer corners.");

    m_bfg.def("compute_triangle_chiral",
        [](py::array_t<std::complex<double>,
                       py::array::c_style | py::array::forcecast> psi,
           int s1, int s2, int s3) {
            if (psi.ndim() != 1) {
                throw std::runtime_error("psi must be a 1-D complex128 array");
            }
            const auto n = static_cast<std::size_t>(psi.shape(0));
            std::vector<ed::bfg::Complex> v(n);
            std::memcpy(v.data(), psi.data(), n * sizeof(ed::bfg::Complex));
            py::gil_scoped_release release;
            return ed::bfg::compute_triangle_chiral(v, s1, s2, s3);
        },
        py::arg("psi"), py::arg("s1"), py::arg("s2"), py::arg("s3"),
        "Symmetric triangle ring-exchange expectation "
        "<psi| S+_1 S-_2 S+_3 + S-_1 S+_2 S-_3 |psi>. Note this is the "
        "(S+S-S+ + h.c.) symmetrisation, not the antisymmetric scalar "
        "chirality S_1 . (S_2 x S_3).");

    // -------------------------------------------------------------------------
    // P2.1 (6th slice): site-resolved spin structure factor over precomputed
    // two-body correlations. The kernel does not see a wavefunction;
    // collaborators feed in the (Hermitian-style) S^{-+} and SzSz tables
    // they already obtained from `compute_smsp_correlations` /
    // `compute_szsz_correlations`.
    // -------------------------------------------------------------------------
    py::class_<ed::bfg::StructureFactorResult>(m_bfg, "StructureFactorResult",
        "Output of `compute_spin_structure_factor`. `s_q[ik]` is the "
        "full Heisenberg structure factor at `cluster.k_points[ik]` in "
        "the SzSz + Re S^{-+} reduction, decomposed into longitudinal "
        "(`s_q_szsz`) and transverse (`s_q_smsp`) channels. "
        "`q_max_idx` / `q_max` / `s_q_max` flag the abscissa of the "
        "max |S(q)| and `m_translation = sqrt(max |S(q)| / N)` is the "
        "BFG translation order parameter.")
        .def_readonly("s_q",          &ed::bfg::StructureFactorResult::s_q)
        .def_readonly("s_q_smsp",     &ed::bfg::StructureFactorResult::s_q_smsp)
        .def_readonly("s_q_szsz",     &ed::bfg::StructureFactorResult::s_q_szsz)
        .def_readonly("q_max_idx",    &ed::bfg::StructureFactorResult::q_max_idx)
        .def_readonly("s_q_max",      &ed::bfg::StructureFactorResult::s_q_max)
        .def_readonly("q_max",        &ed::bfg::StructureFactorResult::q_max)
        .def_readonly("m_translation",
                      &ed::bfg::StructureFactorResult::m_translation);

    m_bfg.def("compute_spin_structure_factor",
        [](const std::vector<std::vector<ed::bfg::Complex>>& smsp_corr,
           const std::vector<std::vector<double>>& szsz_corr,
           const ed::bfg::Cluster& cluster) {
            py::gil_scoped_release release;
            return ed::bfg::compute_spin_structure_factor(
                smsp_corr, szsz_corr, cluster);
        },
        py::arg("smsp_corr"), py::arg("szsz_corr"), py::arg("cluster"),
        "Compute S(q) at every k-point in `cluster.k_points` from "
        "precomputed `<S^-_i S^+_j>` and `<S^z_i S^z_j>` tables. "
        "Both inputs are site-by-site `n_sites x n_sites` matrices. "
        "Returns a `StructureFactorResult`.");

    // -------------------------------------------------------------------------
    // Phase 5 (Apr 2026): high-level dispatcher + symmetry setter +
    // streaming/directory dispatchers + build introspection. Must run AFTER
    // Operator and FixedSzOperator are bound (it attaches symmetry methods
    // to them via m.attr("Operator")). See dispatcher_bindings.{h,cpp}.
    // -------------------------------------------------------------------------
    bind_dispatcher(m);

    // -------------------------------------------------------------------------
    // ED Cleanup Sweep Phase 1 (May 2026): `ed::workflows::solve/thermal/
    // spectral` entry points. Routes through select_backend on every call;
    // intended to replace the legacy `exact_diagonalization_*` family.
    // See workflow_bindings.cpp.
    // -------------------------------------------------------------------------
    bind_workflows(m);
}
