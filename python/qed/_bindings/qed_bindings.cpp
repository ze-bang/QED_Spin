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

#include <ed/config/env_registry.h>
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
#include <ed/thermal/ftlm_kernel.h>                 // Family-2 front door

#include "dispatcher_bindings.h"
#include "input_bindings.h"
#include "little_group_bindings.h"
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
    const char* env = ed::env::raw("ED_LANCZOS_REAL_DISPATCH");
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

// Family-2 front-door rebind (audit 2026-07-31): the direct bindings
// used to call the legacy ::finite_temperature_lanczos, bypassing the
// ed::thermal::ftlm_kernel front door entirely -- the seam this repo's
// consolidation essay warns about (the Stage-12f seed_transform feature
// exists only behind the front door). All four bindings now route through
// ftlm_kernel<CpuBackend> with the SAME log-spaced grid and (newly
// knob-complete) parameters. Since WP10 C5 the front door runs the
// Backend-templated body, which draws the legacy driver's per-sample
// vectors (tests/unit/test_ftlm_sample_seed.cpp); the driver itself was
// deleted in WP10 C6.

// The exact log-spaced grid the legacy (temp_min, temp_max, num_temp_bins)
// driver overload built internally (deleted in WP10 C6), same operations
// in the same order.
std::vector<double> legacy_log_temperature_grid(double temp_min,
                                                double temp_max,
                                                uint64_t num_temp_bins) {
    if (!(temp_min > 0.0) || !(temp_max > 0.0)) {
        throw std::invalid_argument(
            "finite_temperature_lanczos: temp_min and temp_max must both "
            "be > 0.");
    }
    std::vector<double> temps(num_temp_bins);
    const double log_tmin = std::log(temp_min);
    const double log_step = (std::log(temp_max) - log_tmin)
        / static_cast<double>(std::max<uint64_t>(1, num_temp_bins - 1));
    for (uint64_t i = 0; i < num_temp_bins; ++i) {
        temps[i] = std::exp(log_tmin + static_cast<double>(i) * log_step);
    }
    return temps;
}

// How the front door receives the log grid. kBetas passes beta = 1/T and
// lets the kernel report T = 1/beta (qed.finite_temperature_lanczos, whose
// goldens were blessed that way); kExactTemperatures passes the grid
// verbatim through FtlmOptions::temperatures, so the kernel evaluates and
// reports exactly the exp grid the direct driver call used
// (qed.low_temperature_lanczos, WP10 C4: 1/(1/T) can differ from T by an
// ulp).
enum class FrontDoorGrid { kBetas, kExactTemperatures };

template <class OpT>
ed::thermal::FtlmResult ftlm_via_front_door(const OpT& op,
                                            const FTLMParameters& params,
                                            double temp_min,
                                            double temp_max,
                                            uint64_t num_temp_bins,
                                            const std::string& output_dir,
                                            FrontDoorGrid grid) {
    const uint64_t n = hv_dim(op);
    std::vector<double> temps =
        legacy_log_temperature_grid(temp_min, temp_max, num_temp_bins);
    ed::thermal::FtlmOptions kopts;
    if (grid == FrontDoorGrid::kExactTemperatures) {
        kopts.temperatures = std::move(temps);
    } else {
        kopts.betas.reserve(temps.size());
        for (double t : temps) kopts.betas.push_back(1.0 / t);
    }
    kopts.num_samples              = params.num_samples;
    kopts.krylov_dim               = params.krylov_dim;
    kopts.random_seed              = params.random_seed;
    kopts.output_dir               = output_dir_or_devnull(output_dir);
    kopts.max_iterations           = params.max_iterations;
    kopts.tolerance                = params.tolerance;
    kopts.full_reorthogonalization = params.full_reorthogonalization;
    kopts.reorth_frequency         = params.reorth_frequency;
    kopts.store_intermediate       = params.store_intermediate;
    kopts.compute_error_bars       = params.compute_error_bars;

    ed::thermal::FtlmResult res;
    {
        py::gil_scoped_release release;
        ed::matvec::CpuBackend be;
        auto hv = make_hv(op);
        res = ed::thermal::ftlm_kernel(
            be,
            [&hv](const Complex* in, Complex* out, std::size_t nn) {
                hv(in, out, static_cast<int>(nn));
            },
            static_cast<std::size_t>(n), n, kopts);
    }
    return res;
}

py::dict ftlm_result_to_dict(const ed::thermal::FtlmResult& res,
                             const char* ground_state_key) {
    py::dict d;
    d["temperatures"]     = to_numpy_d(res.temperatures);
    d["energy"]           = to_numpy_d(res.energy);
    d["specific_heat"]    = to_numpy_d(res.heat_capacity);
    d["entropy"]          = to_numpy_d(res.entropy);
    d["free_energy"]      = to_numpy_d(res.free_energy);
    d[ground_state_key]   = res.ground_state_estimate;
    return d;
}

py::dict py_finite_temperature_lanczos(const Operator& op,
                                       const FTLMParameters& params,
                                       double temp_min,
                                       double temp_max,
                                       uint64_t num_temp_bins,
                                       const std::string& output_dir) {
    return ftlm_result_to_dict(
        ftlm_via_front_door(op, params, temp_min, temp_max, num_temp_bins,
                            output_dir, FrontDoorGrid::kBetas),
        "ground_state_estimate");
}

py::dict py_finite_temperature_lanczos_fixed_sz(const FixedSzOperator& op,
                                                const FTLMParameters& params,
                                                double temp_min,
                                                double temp_max,
                                                uint64_t num_temp_bins,
                                                const std::string& output_dir) {
    return ftlm_result_to_dict(
        ftlm_via_front_door(op, params, temp_min, temp_max, num_temp_bins,
                            output_dir, FrontDoorGrid::kBetas),
        "ground_state_estimate");
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

// WP10 C4: the last direct callers of the legacy driver. Routed through
// the same front door as qed.finite_temperature_lanczos, handing it the
// driver's exp grid verbatim (FrontDoorGrid::kExactTemperatures) so the
// temperatures -- and every curve evaluated on them -- stay bit-identical
// to the direct call. Keys unchanged (ground-state key
// "ground_state_energy").
py::dict py_low_temperature_lanczos(const Operator& op,
                                    const LTLMParameters& params,
                                    double temp_min,
                                    double temp_max,
                                    uint64_t num_temp_bins,
                                    const std::string& output_dir) {
    return ftlm_result_to_dict(
        ftlm_via_front_door(op, ltlm_to_ftlm_params(params), temp_min,
                            temp_max, num_temp_bins, output_dir,
                            FrontDoorGrid::kExactTemperatures),
        "ground_state_energy");
}

py::dict py_low_temperature_lanczos_fixed_sz(const FixedSzOperator& op,
                                             const LTLMParameters& params,
                                             double temp_min,
                                             double temp_max,
                                             uint64_t num_temp_bins,
                                             const std::string& output_dir) {
    return ftlm_result_to_dict(
        ftlm_via_front_door(op, ltlm_to_ftlm_params(params), temp_min,
                            temp_max, num_temp_bins, output_dir,
                            FrontDoorGrid::kExactTemperatures),
        "ground_state_energy");
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

    m.def("have_cuda", [] { return ed::have_cuda(); },
          "True when this build has CUDA support AND a device is present "
          "(the same gate the engine's GPU rep-gather consults).");
    m.def("dump_env_gates", [] { return ed::symmetry::dump_env_gates(); },
          "The ED_SYM_* rows of the environment registry with their live values, "
          "defaults and meanings (kept for callers of the old name; see env_dump).");
    m.def("env_dump", [](const std::string& prefix) { return ed::env::dump(prefix.c_str()); },
          py::arg("prefix") = "",
          "Every registered ED_* / QED_* environment variable whose name starts with "
          "`prefix`: live value | default | meaning. The table is "
          "include/ed/config/env_registry.h. Paste into bug reports.");
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


    // The little-group verbs (little_group_bindings.cpp).
    bind_little_group(m);
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

    // -------------------------------------------------------------------------
    // SymmetryGroupInfo as the C++ side sees it, built from a directory (the
    // current symmetric lanes) or from memory (their replacement). Test hooks:
    // the two must agree bit for bit, labels included.
    // -------------------------------------------------------------------------
    auto group_info_dict = [](const SymmetryGroupInfo& g) {
        py::list secs;
        for (const auto& s : g.sectors) {
            py::dict d;
            d["sector_id"]       = s.sector_id;
            d["quantum_numbers"] = s.quantum_numbers;
            d["phase_factors"]   = s.phase_factors;
            secs.append(d);
        }
        py::dict out;
        out["generators"]           = g.generators;
        out["generator_orders"]     = g.generator_orders;
        out["max_clique"]           = g.max_clique;
        out["power_representation"] = g.power_representation;
        out["sectors"]              = secs;
        return out;
    };
    m.def("_symmetry_info_from_directory",
          [group_info_dict](const std::string& directory) {
              SymmetryGroupInfo g;
              g.loadFromDirectory(directory);
              return group_info_dict(g);
          },
          py::arg("directory"));
    m.def("_symmetry_info_from_memory",
          [group_info_dict](const std::vector<std::vector<int>>& max_clique,
                            const std::vector<std::vector<int>>& generators,
                            const std::vector<int>& generator_orders,
                            const std::vector<std::pair<uint64_t, std::vector<int>>>& sectors) {
              return group_info_dict(SymmetryGroupInfo::from_memory(
                  max_clique, generators, generator_orders, sectors));
          },
          py::arg("max_clique"), py::arg("generators"), py::arg("generator_orders"),
          py::arg("sectors"));
}
