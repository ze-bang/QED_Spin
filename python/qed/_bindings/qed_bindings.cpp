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
#include <ed/symmetry/group.h>
#include <ed/symmetry/irreps.h>
#include <ed/solvers/little_group_solve.h>  // Stage 7 factorized non-abelian
#include <ed/solvers/little_group_blocks.h> // U1b: little_group_thermal
#include <ed/core/select_backend.h>         // have_cuda (sweep GPU cell)
#include <ed/core/hdf5_io.h>                // r2b: canonical eigenvector save
#include <ed/symmetry/spin_flip.h>
#include <ed/symmetry/env_gates.h>  // Stage 10b: gate inventory + dump  // sz_axis_of (Stage 8d diagonal-axis compose)
#include <ed/dssf/cross_sector_orbit_observable.h>  // 9d: rectangular rep apply
#include <ed/observables/cf_spectral_kernel.h>      // 9d: cf_spectral_from_vector
#include <ed/matvec/backends/cpu_backend.h>         // 9d: CF backend

#include "dispatcher_bindings.h"
#include "input_bindings.h"
#include "sector_bindings.h"
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

// LTLM thermodynamics == FTLM trace for any function of H. The old
// low_temperature_lanczos seeded a second Lanczos from |0> and summed the
// GS-local density of states, not the thermal trace (it stayed pinned near
// E0 at every T). This binding now routes through the verified FTLM path so
// the public `qed.low_temperature_lanczos` name keeps working but returns
// correct thermodynamics. See CONSOLIDATION_PLAN.md Family 1.
FTLMParameters ltlm_to_ftlm_params(const LTLMParameters& p) {
    FTLMParameters f;
    f.krylov_dim               = p.krylov_dim;
    f.num_samples              = p.num_samples;
    f.max_iterations           = p.max_iterations;
    f.tolerance                = p.tolerance;
    f.full_reorthogonalization = p.full_reorthogonalization;
    f.reorth_frequency         = p.reorth_frequency;
    f.random_seed              = p.random_seed;
    f.store_intermediate       = p.store_intermediate;
    f.compute_error_bars       = p.compute_error_bars;
    return f;
}

py::dict py_low_temperature_lanczos(const Operator& op,
                                    const LTLMParameters& params,
                                    double temp_min,
                                    double temp_max,
                                    uint64_t num_temp_bins,
                                    const std::string& output_dir) {
    const uint64_t n = hv_dim(op);
    const std::string dir = output_dir_or_devnull(output_dir);
    const FTLMParameters fparams = ltlm_to_ftlm_params(params);
    FTLMResults res;
    {
        py::gil_scoped_release release;
        res = finite_temperature_lanczos(make_hv(op), n, fparams, temp_min,
                                         temp_max, num_temp_bins, dir);
    }
    py::dict d = thermo_to_dict(res.thermo_data);
    d["ground_state_energy"] = res.ground_state_estimate;
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
    const FTLMParameters fparams = ltlm_to_ftlm_params(params);
    FTLMResults res;
    {
        py::gil_scoped_release release;
        res = finite_temperature_lanczos(make_hv(op), n, fparams, temp_min,
                                         temp_max, num_temp_bins, dir);
    }
    py::dict d = thermo_to_dict(res.thermo_data);
    d["ground_state_energy"] = res.ground_state_estimate;
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
                      int time_reversal = -1,
                      const std::vector<int>& only_k0 = {},
                      bool plan_only = false,
                      const std::vector<int>& only_irrep = {}) {
        ed::solvers::LittleGroupOptions o;
        o.n_up          = n_up;
        o.sz_parity     = sz_parity;
        o.dense_max_dim = dense_max_dim;
        o.spin_flip     = spin_flip;
        o.time_reversal = time_reversal;
        o.only_k0       = only_k0;
        o.plan_only     = plan_only;
        o.only_irrep    = only_irrep;
#ifdef WITH_CUDA
        o.use_gpu       = use_gpu;
#else
        (void)use_gpu;
#endif
        return o;
    };
    // Stage 9f: per-BLOCK quantum-number labels, parallel to
    // block_values / multiplicities. k_raw is the star REPRESENTATIVE's
    // abelian irrep (fold partners share it; membership is in "stars").
    auto lg_label_arrays = [](const ed::solvers::LittleGroupSpectrum& s,
                              py::dict& d) {
        std::vector<int> kraw, fpar, irr, irrd;
        kraw.reserve(s.labels.size()); fpar.reserve(s.labels.size());
        irr.reserve(s.labels.size());  irrd.reserve(s.labels.size());
        for (const auto& L : s.labels) {
            kraw.push_back(L.k_raw);
            fpar.push_back(L.flip_parity);
            irr.push_back(L.irrep);
            irrd.push_back(L.irrep_dim);
        }
        d["block_k_raw"]       = kraw;
        d["block_flip_parity"] = fpar;
        d["block_irrep"]       = irr;
        d["block_irrep_dim"]   = irrd;
    };
    auto lg_stars_dict = [](const ed::solvers::LittleGroupSpectrum& s) {
        py::list stars;
        for (const auto& st : s.stars) {
            py::dict d;
            d["k0"]           = st.k0;
            d["star_size"]    = st.star_size;
            d["members"]      = st.members;
            // P_k0's character table -- name an irrep by its CHARACTER, not by
            // decompose_irreps' internal index. Columns are identified by
            // little_elems (residue indices into the caller's residue_perms;
            // -1 = identity). Empty when the star was not projected.
            d["little_elems"]      = st.little_elems;
            d["little_characters"] = st.little_characters;
            d["little_irrep_dims"] = st.little_irrep_dims;
            d["little_order"] = st.little_order;
            d["projected"]    = st.projected;
            d["dim_k0"]       = st.dim_k0;
            d["flip_parity"]  = st.flip_parity;
            d["tr_pairs"]     = st.tr_pairs;
            d["gpu_engaged"]  = st.gpu_engaged;
            d["csr_engaged"]  = st.csr_engaged;
            stars.append(d);
        }
        return stars;
    };

    m.def("have_cuda", [] { return ed::have_cuda(); },
          "True when this build has CUDA support AND a device is present "
          "(the same gate the engine's GPU rep-gather consults).");
    m.def("dump_env_gates", [] { return ed::symmetry::dump_env_gates(); },
          "Stage 10b: every symmetry-stack env gate with its live value, "
          "default, and meaning -- paste into bug reports. The X-list in "
          "env_gates.h is the single inventory.");

    m.def("little_group_full_spectrum",
          [lg_opts, lg_stars_dict, lg_label_arrays](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int n_up, int sz_parity, bool use_gpu, int spin_flip,
             int time_reversal, int dense_max_dim,
             const std::vector<int>& only_k0, bool plan_only,
             const std::vector<int>& only_irrep) {
              const int n_sites = static_cast<int>(op.getNumBits());
              auto s = ed::solvers::little_group_full_spectrum(
                  op, abelian_group, residue_perms, n_sites,
                  lg_opts(n_up, sz_parity, dense_max_dim, use_gpu, spin_flip,
                          time_reversal, only_k0, plan_only, only_irrep));
              py::dict d;
              d["eigenvalues"]    = s.expanded();
              d["block_values"]   = s.eigenvalues;
              d["multiplicities"] = s.multiplicities;
              lg_label_arrays(s, d);
              d["irrep_characters"] = s.irrep_characters;
              d["stars"]          = lg_stars_dict(s);
              d["flip_engaged"]   = s.flip_engaged;
              d["tr_engaged"]     = s.tr_engaged;
              d["gpu_engaged"]    = s.gpu_engaged;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("n_up") = -1,
          py::arg("sz_parity") = -1, py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          py::arg("only_k0") = std::vector<int>{},
          py::arg("plan_only") = false,
          py::arg("only_irrep") = std::vector<int>{},
          "Full spectrum via the FACTORIZED little-co-group reduction "
          "(one momentum per star, per-irrep blocks inside the star "
          "representative's matrix-free k-sector). only_k0=[...] solves ONLY "
          "those star representatives (extended irrep indices, as reported by "
          "stars[].k0); the covering sum rule is skipped for a restricted "
          "call because a subset cannot tile the subspace. plan_only=True builds "
          "every star's sector and returns stars[] + irrep_characters WITHOUT "
          "solving -- read the star membership and decode a momentum from "
          "chi_k before naming only_k0 (k_raw is an internal irrep index, not "
          "the momentum).");

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

    m.def("little_group_lowest_eigenvalues_labeled",
          [lg_opts, lg_stars_dict](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int k, int n_up, int sz_parity, int dense_max_dim,
             bool use_gpu, int spin_flip, int time_reversal,
             const std::vector<int>& only_k0,
             const std::vector<int>& only_irrep) {
              // Stage 9f: the labeled twin of little_group_lowest_eigenvalues.
              // Aligned per-eigenvalue arrays (expanded by multiplicity,
              // sorted ascending, truncated to k): momentum k_raw is the star
              // REPRESENTATIVE's abelian irrep (fold partners share it; the
              // membership is in "stars"), irrep indexes the little co-group
              // decomposition at that star (-1 = plain block), flip_parity is
              // the (k, +/-) slot when A' = A x Z2 engaged.
              const int n_sites = static_cast<int>(op.getNumBits());
              const auto s = ed::solvers::little_group_lowest_spectrum(
                  op, abelian_group, residue_perms, n_sites, k,
                  lg_opts(n_up, sz_parity, dense_max_dim, use_gpu,
                          spin_flip, time_reversal, only_k0, false,
                          only_irrep));
              std::vector<std::size_t> order(s.eigenvalues.size());
              std::iota(order.begin(), order.end(), std::size_t{0});
              std::sort(order.begin(), order.end(),
                        [&](std::size_t a, std::size_t b) {
                            return s.eigenvalues[a] < s.eigenvalues[b];
                        });
              std::vector<double> ev;
              std::vector<int> kraw, fpar, irr, irrd, mult;
              std::vector<bool> conv;
              const std::size_t want = static_cast<std::size_t>(
                  std::max(k, 1));
              for (std::size_t idx : order) {
                  const auto& L = s.labels[idx];
                  for (int r = 0; r < s.multiplicities[idx]
                                  && ev.size() < want; ++r) {
                      ev.push_back(s.eigenvalues[idx]);
                      kraw.push_back(L.k_raw);
                      fpar.push_back(L.flip_parity);
                      irr.push_back(L.irrep);
                      irrd.push_back(L.irrep_dim);
                      mult.push_back(s.multiplicities[idx]);
                      conv.push_back(L.converged);
                  }
                  if (ev.size() >= want) break;
              }
              py::dict d;
              d["eigenvalues"]  = ev;
              d["k_raw"]        = kraw;
              d["flip_parity"]  = fpar;
              d["irrep"]        = irr;
              d["irrep_dim"]    = irrd;
              d["multiplicity"] = mult;
              d["converged"]    = conv;
              d["irrep_characters"] = s.irrep_characters;
              d["stars"]        = lg_stars_dict(s);
              d["flip_engaged"] = s.flip_engaged;
              d["tr_engaged"]   = s.tr_engaged;
              d["gpu_engaged"]  = s.gpu_engaged;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("k") = 1,
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("dense_max_dim") = 64, py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("only_k0") = std::vector<int>{},
          py::arg("only_irrep") = std::vector<int>{},
          "Stage 9f: lowest-k eigenvalues WITH aligned per-eigenvalue "
          "quantum-number labels (k_raw = star representative's momentum "
          "irrep, little-group irrep index + dimension, flip parity, "
          "multiplicity) -- the labels the engine always computed and "
          "previously discarded at this boundary. only_k0=[...] restricts the "
          "walk to those star representatives, so NAMING a momentum costs only "
          "that star's work instead of forcing the call off the projection "
          "lane onto the (larger) abelian block.");

    m.def("little_group_block_grounds",
          [lg_opts, lg_stars_dict](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int n_up, int sz_parity, int dense_max_dim,
             bool use_gpu, int spin_flip, int time_reversal,
             const std::vector<int>& only_k0,
             const std::vector<int>& only_irrep) {
              // Lowest eigenvalue of EVERY (momentum, little-co-group irrep)
              // block -- k=1 PER BLOCK, so the no-reorth Lanczos early-exits
              // the instant the lowest Ritz value converges, BEFORE the
              // three-term recurrence loses orthogonality. (The k>1 path of
              // little_group_lowest_spectrum runs to ~400 iters and returns a
              // ghost at N=36 -- a bit-identical spurious constant across all
              // stars; this lane avoids it entirely by never asking a block
              // for more than its ground.) Returns ALL blocks, no global
              // truncation: the per-(k,irrep) minima ARE the Anderson tower's
              // momentum/irrep-resolved low-energy structure -- collect the
              // lowest-few across blocks for the tower multiplet, each entry
              // already carrying (k_raw, irrep, flip_parity).
              const int n_sites = static_cast<int>(op.getNumBits());
              const auto s = ed::solvers::little_group_lowest_spectrum(
                  op, abelian_group, residue_perms, n_sites, /*k=*/1,
                  lg_opts(n_up, sz_parity, dense_max_dim, use_gpu,
                          spin_flip, time_reversal, only_k0, false,
                          only_irrep));
              std::vector<double> ev;
              std::vector<int> kraw, fpar, irr, irrd, mult;
              std::vector<bool> conv;
              for (std::size_t idx = 0; idx < s.eigenvalues.size(); ++idx) {
                  const auto& L = s.labels[idx];
                  ev.push_back(s.eigenvalues[idx]);
                  kraw.push_back(L.k_raw);
                  fpar.push_back(L.flip_parity);
                  irr.push_back(L.irrep);
                  irrd.push_back(L.irrep_dim);
                  mult.push_back(s.multiplicities[idx]);
                  conv.push_back(L.converged);
              }
              py::dict d;
              d["eigenvalues"]  = ev;
              d["k_raw"]        = kraw;
              d["flip_parity"]  = fpar;
              d["irrep"]        = irr;
              d["irrep_dim"]    = irrd;
              d["multiplicity"] = mult;
              d["converged"]    = conv;
              d["irrep_characters"] = s.irrep_characters;
              d["stars"]        = lg_stars_dict(s);
              d["flip_engaged"] = s.flip_engaged;
              d["tr_engaged"]   = s.tr_engaged;
              d["gpu_engaged"]  = s.gpu_engaged;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("n_up") = -1,
          py::arg("sz_parity") = -1, py::arg("dense_max_dim") = 64,
          py::arg("use_gpu") = false, py::arg("spin_flip") = -1,
          py::arg("time_reversal") = -1,
          py::arg("only_k0") = std::vector<int>{},
          py::arg("only_irrep") = std::vector<int>{},
          "Lowest eigenvalue of EVERY (momentum, irrep) block -- k=1 per "
          "block (reliable early-exit, unlike the k>1 spectrum lane which "
          "ghosts at N=36). Aligned per-block arrays {eigenvalues, k_raw, "
          "irrep, irrep_dim, flip_parity, multiplicity, converged} plus "
          "stars, irrep_characters. The momentum/irrep-resolved tower.");

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

    // U1b (lane unification): SAMPLED thermodynamics inside the projected
    // blocks -- FTLM/LTLM/mTPQ/OFTLM per (n_up, k, +/-, sigma) block via
    // ed::workflows::thermal(block.op(), ...), Z-recombined with the block
    // multiplicity folded in as an F-shift. KPM_DOS raises (full-spectrum
    // DOS deliverable; use the abelian lane). Returns the combined thermo
    // plus parallel per-block tag arrays -- the engagement signal the
    // dimension-reduction matrix asserts block structure against.
    m.def("little_group_thermal",
          [lg_opts](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             const std::string& method, double t_min, double t_max,
             std::size_t num_t, std::size_t num_samples,
             std::size_t krylov_dim, std::uint64_t random_seed,
             int n_up, int sz_parity, bool use_gpu, int spin_flip,
             int time_reversal, int dense_max_dim) {
              using Method = ed::workflows::ThermalOptions::Method;
              ed::workflows::ThermalOptions topts;
              if      (method == "FTLM")  topts.method = Method::FTLM;
              else if (method == "LTLM")  topts.method = Method::LTLM;
              else if (method == "mTPQ")  topts.method = Method::mTPQ;
              else if (method == "OFTLM") topts.method = Method::OFTLM;
              else
                  throw std::invalid_argument(
                      "little_group_thermal: method must be one of "
                      "FTLM/LTLM/mTPQ/OFTLM (KPM_DOS recombines on the "
                      "abelian lane only), got '" + method + "'");
              topts.temp_min      = t_min;
              topts.temp_max      = t_max;
              topts.num_temp_bins = num_t;
              topts.num_samples   = num_samples;
              topts.krylov_dim    = krylov_dim;
              topts.random_seed   = random_seed;
              topts.backend.allow_gpu = use_gpu;
              const int n_sites = static_cast<int>(op.getNumBits());
              ed::solvers::LittleGroupThermalResult r;
              {
                  py::gil_scoped_release release;
                  r = ed::solvers::little_group_thermal(
                      op, abelian_group, residue_perms, n_sites, topts,
                      lg_opts(n_up, sz_parity, dense_max_dim, use_gpu,
                              spin_flip, time_reversal));
              }
              py::dict d;
              d["temperatures"]  = r.thermo.temperatures;
              d["energy"]        = r.thermo.energy;
              d["specific_heat"] = r.thermo.specific_heat;
              d["entropy"]       = r.thermo.entropy;
              d["free_energy"]   = r.thermo.free_energy;
              d["ground_state_energy"] = r.ground_state_energy;
              d["projected_any"] = r.projected_any;
              d["gpu_engaged"]   = r.gpu_engaged;
              std::vector<int> b_nup, b_kraw, b_flip, b_irrep, b_idim,
                               b_star;
              std::vector<std::uint64_t> b_dim, b_mult, b_weight;
              for (std::size_t i = 0; i < r.block_tags.size(); ++i) {
                  const auto& t = r.block_tags[i];
                  b_nup.push_back(t.n_up);
                  b_kraw.push_back(t.k_raw);
                  b_flip.push_back(t.flip_parity);
                  b_irrep.push_back(t.irrep);
                  b_idim.push_back(t.irrep_dim);
                  b_star.push_back(t.star_size);
                  b_dim.push_back(t.dim);
                  b_mult.push_back(t.multiplicity);
                  b_weight.push_back(r.weights[i]);
              }
              d["block_n_up"]        = b_nup;
              d["block_k_raw"]       = b_kraw;
              d["block_flip_parity"] = b_flip;
              d["block_irrep"]       = b_irrep;
              d["block_irrep_dim"]   = b_idim;
              d["block_star_size"]   = b_star;
              d["block_dim"]         = b_dim;
              d["block_multiplicity"] = b_mult;
              d["block_weight"]      = b_weight;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("method") = "FTLM",
          py::arg("t_min") = 0.1, py::arg("t_max") = 10.0,
          py::arg("num_t") = 24, py::arg("num_samples") = 40,
          py::arg("krylov_dim") = 100, py::arg("random_seed") = 0,
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("use_gpu") = false,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          "Sampled (FTLM/LTLM/mTPQ/OFTLM) thermodynamics inside the "
          "factorized little-group blocks, Z-recombined with block "
          "multiplicities.");

    // U2b-r2: certified lowest-k eigenpairs on the projection lane.
    // Vectors are returned as COMPUTATIONAL-basis amplitude arrays
    // (flip-aware expansion; moderate-N contract, n_sites <= 30). A row
    // with multiplicity > 1 carries ONE representative vector (fold
    // partners need U3 transport).
    m.def("little_group_lowest_vectors",
          [lg_opts](const Operator& op,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int k, int n_up, int sz_parity, int spin_flip,
             int time_reversal, int dense_max_dim,
             const std::string& output_dir) {
              const int n_sites = static_cast<int>(op.getNumBits());
              ed::solvers::LittleGroupVectors lv;
              {
                  py::gil_scoped_release release;
                  lv = ed::solvers::little_group_lowest_vectors(
                      op, abelian_group, residue_perms, n_sites, k,
                      lg_opts(n_up, sz_parity, dense_max_dim,
                              /*use_gpu=*/false, spin_flip,
                              time_reversal));
              }
              py::dict d;
              std::vector<double> evals;
              std::vector<int> kraw, flip, irrep, idim;
              std::vector<std::uint64_t> mult;
              py::list vecs;
              std::vector<std::vector<std::complex<double>>> psis;
              for (const auto& row : lv.rows) {
                  evals.push_back(row.eigenvalue);
                  kraw.push_back(row.tag.k_raw);
                  flip.push_back(row.tag.flip_parity);
                  irrep.push_back(row.tag.irrep);
                  idim.push_back(row.tag.irrep_dim);
                  mult.push_back(row.tag.multiplicity);
                  auto psi =
                      ed::solvers::expand_rep_vector_to_computational(
                          lv.sectors[row.sector_slot], row.vec);
                  py::array_t<std::complex<double>> arr(
                      static_cast<py::ssize_t>(psi.size()));
                  std::copy(psi.begin(), psi.end(),
                            arr.mutable_data());
                  vecs.append(std::move(arr));
                  if (!output_dir.empty()) psis.push_back(std::move(psi));
              }
              // r2b: persist through the canonical writer (the same
              // /eigendata layout every solver emits). COMPUTATIONAL
              // basis -- directly consumable, unlike the abelian lane's
              // per-sector sector-basis files. Row i's eigenvector pairs
              // with eigenvalue i.
              if (!output_dir.empty()
                  && !HDF5IO::isDisabledOutputPath(output_dir)) {
                  HDF5IO::saveDiagonalizationResults(
                      output_dir, evals, psis, "LITTLE_GROUP_VECTORS");
                  d["hdf5_path"] = output_dir + "/ed_results.h5";
              } else {
                  d["hdf5_path"] = std::string();
              }
              d["eigenvalues"]   = evals;
              d["k_raw"]         = kraw;
              d["flip_parity"]   = flip;
              d["irrep"]         = irrep;
              d["irrep_dim"]     = idim;
              d["multiplicity"]  = mult;
              d["vectors"]       = vecs;
              d["flip_engaged"]  = lv.flip_engaged;
              d["tr_engaged"]    = lv.tr_engaged;
              return d;
          },
          py::arg("operator"), py::arg("abelian_group"),
          py::arg("residue_perms"), py::arg("k") = 1,
          py::arg("n_up") = -1, py::arg("sz_parity") = -1,
          py::arg("spin_flip") = -1, py::arg("time_reversal") = -1,
          py::arg("dense_max_dim") = 4096,
          py::arg("output_dir") = std::string(),
          "Certified lowest-k eigenpairs of the factorized block "
          "decomposition, vectors expanded to the computational basis "
          "(persisted to <output_dir>/ed_results.h5 when given).");

    m.def("little_group_gs_dssf",
          [](const Operator& op_h, const Operator& op_o,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             double omega_min, double omega_max, int n_omega,
             double broadening, int krylov_dim, int dense_max_dim,
             bool use_gpu, int time_reversal) {
              // Stage 9d: FACTORIZED GS-DSSF. The ground state is
              // localized by the star walk (folds shrink the search) and
              // solved PLAIN in its momentum sector; O|0> is scattered
              // into every RAW destination sector by the Stage-8d
              // CrossSectorOrbitObservable rep lane (matrix elements are
              // never folded -- ||phi|| decides every selection rule);
              // one continued-fraction Lanczos per receiving sector.
              // Memory O(#reps) throughout -- this replaces the
              // monolithic SAB DSSF, which materialised the FULL
              // eigenbasis in the computational basis.
              using Complex = std::complex<double>;
              const int n_sites = static_cast<int>(op_h.getNumBits());
              if (!op_o.three_body_data_.empty())
                  throw std::runtime_error(
                      "little_group_gs_dssf: three-body probes are not "
                      "supported (one/two-body TransformData only).");

              // Diagonal axis (same detection as everywhere else).
              ed::matvec::TermStorage soa;
              ed::matvec::TermStorage::classify_route(
                  soa, op_h.transform_data_, op_h.three_body_data_,
                  [](const std::complex<double>& c) { return c; });
              const auto ax = ed::symmetry::sz_axis_of(soa);
              std::vector<std::pair<int, int>> gs_subspaces{{-1, -1}};
              if (ax == ed::symmetry::SzAxis::U1) {
                  gs_subspaces.clear();
                  for (int k = 0; k <= n_sites; ++k)
                      gs_subspaces.emplace_back(k, -1);
              } else if (ax == ed::symmetry::SzAxis::Parity) {
                  gs_subspaces = {{-1, 0}, {-1, 1}};
              }

              // (1) Locate the GS subspace by star-walk lowest-1 solves.
              auto lg_o = [&](int nu, int par) {
                  ed::solvers::LittleGroupOptions o;
                  o.n_up          = nu;
                  o.sz_parity     = par;
                  o.dense_max_dim = dense_max_dim;
#ifdef WITH_CUDA
                  o.use_gpu       = use_gpu;   // batched eigensolve on the
                                               // GS-subspace scan
#endif
                  // U2b-r1b: the SOURCE may be flip-extended (auto). The
                  // rep-lane scatter is flip-aware (Stage-5b masks ride
                  // the policy) and the cross-sector normalization
                  // handles |G_src| != |G_dst| (1/sqrt(Gs*Gd)); the
                  // destinations below stay RAW. Pinned by the
                  // flip-engaged Lehmann test in test_little_group_dssf.
                  o.spin_flip     = -1;
                  o.time_reversal = time_reversal;
                  return o;
              };
              int gs_nu = -1, gs_par = -1;
              double e_best = 0.0;
              bool have = false;
              for (const auto& [nu, par] : gs_subspaces) {
                  const auto ev = ed::solvers::little_group_lowest_eigenvalues(
                      op_h, abelian_group, residue_perms, n_sites, 1,
                      lg_o(nu, par));
                  if (ev.empty()) continue;
                  if (!have || ev[0] < e_best) {
                      have = true;
                      e_best = ev[0];
                      gs_nu = nu;
                      gs_par = par;
                  }
              }
              if (!have)
                  throw std::runtime_error(
                      "little_group_gs_dssf: no non-empty subspace.");

              // (2) The GS eigenvector in its momentum sector.
              const auto gs = ed::solvers::little_group_ground_state(
                  op_h, abelian_group, residue_perms, n_sites,
                  lg_o(gs_nu, gs_par));

              // (3) Destination sweep: 1/2-body probes reach at most
              // n_up +- 2 (U(1)) / both parity halves / the full space.
              std::vector<std::pair<int, int>> dst_subspaces;
              if (ax == ed::symmetry::SzAxis::U1) {
                  for (int d = -2; d <= 2; ++d) {
                      const int nu = gs_nu + d;
                      if (nu >= 0 && nu <= n_sites)
                          dst_subspaces.emplace_back(nu, -1);
                  }
              } else if (ax == ed::symmetry::SzAxis::Parity) {
                  dst_subspaces = {{-1, 0}, {-1, 1}};
              } else {
                  dst_subspaces = {{-1, -1}};
              }

              std::vector<double> omega_grid(
                  static_cast<std::size_t>(std::max(n_omega, 1)));
              const double dw = (n_omega > 1)
                  ? (omega_max - omega_min) / (n_omega - 1) : 0.0;
              for (int i = 0; i < std::max(n_omega, 1); ++i)
                  omega_grid[static_cast<std::size_t>(i)] =
                      omega_min + dw * i;

              std::vector<double> s_omega(omega_grid.size(), 0.0);
              double total_weight = 0.0;
              bool dssf_gpu = false;   // truthful: any receiving sector's
                                       // CF matvec ran the device gather
              ed::matvec::CpuBackend be;
              using Ref = ed::dssf::CrossSectorOrbitObservable::OperatorRef;
              const auto src_ref = Ref::from_rep(
                  gs.rd, static_cast<std::uint64_t>(n_sites));

              for (const auto& [dnu, dpar] : dst_subspaces) {
                  auto sectors = ed::solvers::little_group_k_sectors(
                      op_h, abelian_group, n_sites, dnu, dpar);
                  for (const auto& rd_dst : sectors) {
                      const std::size_t dim_dst = rd_dst.reps.size();
                      ed::dssf::CrossSectorOrbitObservable obs(
                          src_ref, 0,
                          Ref::from_rep(rd_dst,
                                        static_cast<std::uint64_t>(n_sites)),
                          0, op_o.transform_data_,
                          static_cast<float>(op_h.getSpin()));
                      std::vector<Complex> phi(dim_dst, Complex(0, 0));
                      obs.apply(gs.vec.data(), phi.data(), dim_dst);
                      double n2 = 0.0;
                      for (const Complex& c : phi) n2 += std::norm(c);
                      if (n2 < 1e-24) continue;   // selection rule says no
                      total_weight += n2;

                      // GS-DSSF GPU lane (2026-07-20): an explicit GPU
                      // request forces the device rep-gather on the
                      // receiving sector's CF matvec (dimension floor
                      // dropped, reduced CSR demoted to fallback).
                      auto mv = ed::solvers::make_rep_sector_matvec(
                          op_h, rd_dst, /*force_gpu=*/use_gpu);
                      ed::observables::CfSpectralOptions cfopts;
                      cfopts.krylov_dim   = static_cast<std::size_t>(
                          std::max(krylov_dim, 2));
                      cfopts.broadening   = broadening;
                      cfopts.energy_shift = gs.energy;
                      cfopts.tolerance    = 1e-12;
                      cfopts.global_n     = dim_dst;
                      auto apply_H = [&mv](const Complex* in, Complex* out,
                                           std::size_t nn) {
                          mv->apply(in, out, nn);
                      };
                      const auto cf =
                          ed::observables::cf_spectral_from_vector(
                              be, apply_H, dim_dst, phi.data(),
                              omega_grid, cfopts);
                      for (std::size_t i = 0; i < s_omega.size(); ++i)
                          s_omega[i] += cf.spectral_function[i];
                      dssf_gpu = dssf_gpu ||
                          ed::solvers::rep_sector_matvec_gpu_engaged(*mv);
                  }
              }

              py::dict d;
              d["omega"]        = omega_grid;
              d["s_omega"]      = s_omega;
              d["gs_energy"]    = gs.energy;
              d["gs_k0"]        = gs.k0;
              d["total_weight"] = total_weight;
              d["gpu_engaged"]  = dssf_gpu;
              return d;
          },
          py::arg("op"), py::arg("observable"),
          py::arg("abelian_group"), py::arg("residue_perms"),
          py::arg("omega_min"), py::arg("omega_max"),
          py::arg("n_omega"), py::arg("broadening") = 0.1,
          py::arg("krylov_dim") = 200, py::arg("dense_max_dim") = 512,
          py::arg("use_gpu") = false,
          py::arg("time_reversal") = -1,
          "Stage 9d: factorized ground-state DSSF -- GS localized by the "
          "star walk and solved matrix-free in its momentum sector; O|0> "
          "scattered into every raw destination sector "
          "(CrossSectorOrbitObservable rep lane); one continued-fraction "
          "Lanczos per receiving sector. Memory O(#reps): the scalable "
          "replacement for symmetry_adapted_gs_dssf.");

    // STATIC transverse structure factor S^{+-}(q) for a LIST of probes,
    // amortising the ONE (n_up-pinned) little-group GS solve across every q.
    //
    // The static structure factor is the ZEROTH frequency moment of the DSSF:
    //   S_O = <GS| O^dagger O |GS> = || O|GS> ||^2
    // (the ``total_weight`` little_group_gs_dssf already computes as a
    // by-product before its continued fraction). For O_q = N^{-1/2} sum_j
    // e^{-i q.r_j} S^-_j this is exactly S^{+-}(q). No omega grid, no CF --
    // just the scatter norm per probe, so a whole q-mesh costs ONE GS solve.
    //
    // n_up pins the GS magnetisation (the AFM GS is a singlet at N/2); the
    // unpinned U(1) sweep is ~sqrt(N)x more work for the same answer (mirrors
    // little_group_gs_correlators 0461db3). Memory O(#reps): one source + one
    // destination sector resident at a time.
    m.def("little_group_gs_static_sf",
          [](const Operator& op_h,
             const std::vector<Operator>& observables,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             int n_up, int delta_n_up, int dense_max_dim, bool use_gpu,
             int time_reversal, const std::vector<int>& only_k0) {
              using Complex = std::complex<double>;
              const int n_sites = static_cast<int>(op_h.getNumBits());
              for (const Operator& o : observables)
                  if (!o.three_body_data_.empty())
                      throw std::runtime_error(
                          "little_group_gs_static_sf: probes must be "
                          "one/two-body operators.");

              ed::matvec::TermStorage soa;
              ed::matvec::TermStorage::classify_route(
                  soa, op_h.transform_data_, op_h.three_body_data_,
                  [](const std::complex<double>& c) { return c; });
              const auto ax = ed::symmetry::sz_axis_of(soa);

              std::vector<std::pair<int, int>> gs_subspaces;
              if (n_up >= 0) {
                  gs_subspaces = {{n_up, -1}};
              } else if (ax == ed::symmetry::SzAxis::U1) {
                  for (int k = 0; k <= n_sites; ++k)
                      gs_subspaces.emplace_back(k, -1);
              } else if (ax == ed::symmetry::SzAxis::Parity) {
                  gs_subspaces = {{-1, 0}, {-1, 1}};
              } else {
                  gs_subspaces = {{-1, -1}};
              }
              auto lg_o = [&](int nu, int par) {
                  ed::solvers::LittleGroupOptions o;
                  o.n_up = nu; o.sz_parity = par;
                  o.dense_max_dim = dense_max_dim;
#ifdef WITH_CUDA
                  o.use_gpu = use_gpu;
#endif
                  o.spin_flip = -1; o.time_reversal = time_reversal;
                  // Pin the GS to a named momentum star. Needed to keep both
                  // structure-factor channels on the SAME ground state at a
                  // near-degeneracy: two stars within ~1e-5 (e.g. the 4x3
                  // kagome k0=16/20 crossing near Jpm=-0.13) are ranked
                  // inconsistently by the two-phase scan across builds, so
                  // one channel can land on a slightly-excited star.
                  o.only_k0 = only_k0;
                  return o;
              };
              int gs_nu = -1, gs_par = -1; double e_best = 0.0; bool have = false;
              if (gs_subspaces.size() == 1) {
                  gs_nu = gs_subspaces[0].first; gs_par = gs_subspaces[0].second;
                  have = true;
              } else {
                  for (const auto& [nu, par] : gs_subspaces) {
                      const auto ev = ed::solvers::little_group_lowest_eigenvalues(
                          op_h, abelian_group, residue_perms, n_sites, 1, lg_o(nu, par));
                      if (ev.empty()) continue;
                      if (!have || ev[0] < e_best) {
                          have = true; e_best = ev[0]; gs_nu = nu; gs_par = par;
                      }
                  }
              }
              if (!have)
                  throw std::runtime_error(
                      "little_group_gs_static_sf: no non-empty subspace.");

              const auto gs = ed::solvers::little_group_ground_state(
                  op_h, abelian_group, residue_perms, n_sites, lg_o(gs_nu, gs_par));

              (void)delta_n_up;

              // IN-SECTOR expectation <GS|O_q|GS>. Each observable O_q is an
              // Sz- and momentum-CONSERVING operator (e.g. the transverse
              // O_q = sum_{i!=j} e^{iq(r_i-r_j)} S^+_i S^-_j, Hermitian), so it
              // maps the GS momentum sector to ITSELF -- only ONE sector is
              // ever resident (memory O(#reps), no destination sector). The
              // cross-sector norm route needed BOTH the GS source and a
              // destination sector built at once: two ~40 GB sectors OOM a
              // 128 GB node at N=36 (task 50220971 died building the 716M-orbit
              // dst table). Here the rep-sector matvec (GPU rep-gather) applies
              // O_q on the GS vector and we dot with <GS|.
              const std::size_t dim = gs.vec.size();
              std::vector<double> static_sf;
              static_sf.reserve(observables.size());
              std::vector<Complex> ov(dim);
              for (const Operator& o : observables) {
                  // rd is consumed by the matvec factory; copy so the GS sector
                  // survives for the next probe (cheap vs. the GS solve).
                  ed::symmetry::RepSectorData rd_copy = gs.rd;
                  auto mv = ed::solvers::make_rep_sector_matvec(
                      o, std::move(rd_copy), /*force_gpu=*/use_gpu);
                  mv->apply(gs.vec.data(), ov.data(), dim);
                  Complex c(0.0, 0.0);
                  for (std::size_t i = 0; i < dim; ++i)
                      c += std::conj(gs.vec[i]) * ov[i];
                  static_sf.push_back(c.real());   // <GS|O_q|GS> (real: Hermitian)
              }

              py::dict d;
              d["gs_energy"] = gs.energy;
              d["gs_k0"]     = gs.k0;
              d["gs_n_up"]   = gs_nu;
              d["n_reps"]    = static_cast<std::uint64_t>(gs.rd.reps.size());
              d["static_sf"] = static_sf;
              return d;
          },
          py::arg("op"), py::arg("observables"),
          py::arg("abelian_group"), py::arg("residue_perms"),
          py::arg("n_up") = -1, py::arg("delta_n_up") = 1,
          py::arg("dense_max_dim") = 512,
          py::arg("use_gpu") = false, py::arg("time_reversal") = -1,
          py::arg("only_k0") = std::vector<int>{},
          "IN-SECTOR ground-state expectations <GS|O_q|GS> for a LIST of "
          "Sz- and momentum-conserving operators O_q, amortising ONE "
          "n_up-pinned little-group GS solve. Each O_q is applied on the GS "
          "vector by the rep-sector matvec (GPU rep-gather) and dotted with "
          "<GS| -- only the GS momentum sector is ever resident (memory "
          "O(#reps), no destination sector). For the transverse static "
          "structure factor pass O_q = sum_{i!=j} e^{iq(r_i-r_j)} S^+_i S^-_j "
          "(Hermitian); then S^{+-}(q) = <GS|O_q|GS>/N + n_up/N (the i=j "
          "self-term is exactly n_up on the fixed-Sz sector). ``delta_n_up`` "
          "is ignored (kept for call-site compatibility). Returns "
          "{gs_energy, gs_k0, gs_n_up, n_reps, static_sf[]} with static_sf = "
          "the raw <GS|O_q|GS>.");

    m.def("little_group_gs_correlators",
          [](const Operator& op_h,
             const std::vector<std::vector<int>>& abelian_group,
             const std::vector<std::vector<int>>& residue_perms,
             const std::vector<std::vector<int>>& translations,
             int n_up, int dense_max_dim, bool use_gpu,
             int time_reversal) {
              // STATIC structure factor without the DSSF machinery.
              //
              // S^zz(Q) needs only DIAGONAL, G-INVARIANT correlators:
              //   O_d = sum_i S^z_i S^z_{i+d}
              // is diagonal in the computational basis, translation
              // invariant (the sum over i) and flip invariant
              // (S^z -> -S^z gives (-)(-)). For such an operator the
              // orbit cross-terms vanish and the expectation collapses to
              // a weighted sum over REPRESENTATIVES:
              //     <Psi|O_d|Psi> = sum_r |c_r|^2 O_d(rep_r)
              // with O_d(s) = [N - 2*popcount(s ^ T_d s)] / 4.
              //
              // So ONE ground-state solve yields the energy AND every
              // C(d); the caller Fourier transforms C(d) -> S(Q) at ALL
              // momenta for free. No destination sectors, no
              // continued-fraction chains: memory is ONE sector instead of
              // the ~N_Q resident mirrors the multi-Q spectral lane needs,
              // and the flip Z2 is exploited (half the reps).
              using Complex = std::complex<double>;
              const int n_sites = static_cast<int>(op_h.getNumBits());

              ed::matvec::TermStorage soa;
              ed::matvec::TermStorage::classify_route(
                  soa, op_h.transform_data_, op_h.three_body_data_,
                  [](const std::complex<double>& c) { return c; });
              const auto ax = ed::symmetry::sz_axis_of(soa);
              std::vector<std::pair<int, int>> gs_subspaces{{-1, -1}};
              if (n_up >= 0) {
                  // PINNED: the caller knows the GS magnetisation (a
                  // Heisenberg AFM ground state is a total-spin singlet, so
                  // n_up = N/2). Scanning every n_up costs ~7.6x the
                  // half-filled sector alone at N=36 and cannot find a lower
                  // state -- measured: 96 star solves in 10 h, none of them
                  // in the sector that holds the GS.
                  gs_subspaces = {{n_up, -1}};
              } else if (ax == ed::symmetry::SzAxis::U1) {
                  gs_subspaces.clear();
                  for (int k = 0; k <= n_sites; ++k)
                      gs_subspaces.emplace_back(k, -1);
              } else if (ax == ed::symmetry::SzAxis::Parity) {
                  gs_subspaces = {{-1, 0}, {-1, 1}};
              }
              auto lg_o = [&](int nu, int par) {
                  ed::solvers::LittleGroupOptions o;
                  o.n_up          = nu;
                  o.sz_parity     = par;
                  o.dense_max_dim = dense_max_dim;
#ifdef WITH_CUDA
                  o.use_gpu       = use_gpu;   // batched eigensolve on the
                                               // GS-subspace scan
#endif
                  o.spin_flip     = -1;
                  o.time_reversal = time_reversal;
                  return o;
              };
              int gs_nu = -1, gs_par = -1;
              double e_best = 0.0;
              bool have = false;
              if (gs_subspaces.size() == 1) {
                  // Only ONE candidate subspace (the caller pinned n_up), so
                  // the localization scan can only return that subspace --
                  // running it would solve the very same blocks that
                  // little_group_ground_state re-solves below, DOUBLING the
                  // cost of the whole call. Skip straight to the eigenvector.
                  gs_nu = gs_subspaces[0].first;
                  gs_par = gs_subspaces[0].second;
                  have = true;
              } else {
                  for (const auto& [nu, par] : gs_subspaces) {
                      const auto ev =
                          ed::solvers::little_group_lowest_eigenvalues(
                              op_h, abelian_group, residue_perms, n_sites, 1,
                              lg_o(nu, par));
                      if (ev.empty()) continue;
                      if (!have || ev[0] < e_best) {
                          have = true; e_best = ev[0];
                          gs_nu = nu; gs_par = par;
                      }
                  }
              }
              if (!have)
                  throw std::runtime_error(
                      "little_group_gs_correlators: no non-empty subspace.");

              const auto gs = ed::solvers::little_group_ground_state(
                  op_h, abelian_group, residue_perms, n_sites,
                  lg_o(gs_nu, gs_par));

              const auto& reps = gs.rd.reps;
              const std::size_t nr = reps.size();
              if (gs.vec.size() != nr)
                  throw std::runtime_error(
                      "little_group_gs_correlators: vec/reps length mismatch.");
              std::vector<double> w(nr);
              double nrm = 0.0;
              for (std::size_t r = 0; r < nr; ++r) {
                  w[r] = std::norm(gs.vec[r]);
                  nrm += w[r];
              }
              if (!(nrm > 0.0))
                  throw std::runtime_error(
                      "little_group_gs_correlators: zero-norm GS vector.");
              const double inv_nrm = 1.0 / nrm;

              const std::size_t nd = translations.size();
              std::vector<double> C(nd, 0.0);
              for (std::size_t d = 0; d < nd; ++d) {
                  const auto& perm = translations[d];
                  if (static_cast<int>(perm.size()) != n_sites)
                      throw std::runtime_error(
                          "little_group_gs_correlators: translation length "
                          "!= n_sites.");
                  double acc = 0.0;
#ifdef _OPENMP
#   pragma omp parallel for reduction(+ : acc) schedule(static)
#endif
                  for (long long r = 0; r < static_cast<long long>(nr); ++r) {
                      const std::uint64_t s = reps[static_cast<std::size_t>(r)];
                      std::uint64_t t = 0;
                      for (int i = 0; i < n_sites; ++i)
                          t |= ((s >> perm[i]) & 1ULL) << i;
                      const int pc = __builtin_popcountll(s ^ t);
                      acc += w[static_cast<std::size_t>(r)]
                             * static_cast<double>(n_sites - 2 * pc) * 0.25;
                  }
                  C[d] = acc * inv_nrm / static_cast<double>(n_sites);
              }

              py::dict out;
              out["gs_energy"]   = gs.energy;
              out["gs_k0"]       = gs.k0;
              out["gs_n_up"]     = gs_nu;
              out["correlators"] = C;
              out["n_reps"]      = static_cast<std::uint64_t>(nr);
              return out;
          },
          py::arg("op"), py::arg("abelian_group"), py::arg("residue_perms"),
          py::arg("translations"), py::arg("n_up") = -1,
          py::arg("dense_max_dim") = 512, py::arg("use_gpu") = false,
          py::arg("time_reversal") = -1,
          "ONE ground-state solve -> energy AND the translation-averaged "
          "diagonal correlators C(d) = <sum_i S^z_i S^z_{i+d}>/N. Fourier "
          "transform C(d) for S^zz(Q) at EVERY momentum. Memory is one "
          "sector (no destination mirrors, no continued fractions) and the "
          "flip Z2 is exploited -- the cheap replacement for the multi-Q "
          "spectral lane when only the STATIC structure factor is wanted.");

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
    bind_sectors(m);
}
