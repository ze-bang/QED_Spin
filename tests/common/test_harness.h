// =============================================================================
// Minimal, header-only test harness for the ED package.
//
// Design goals:
//   * Zero external dependencies (no gtest/catch): tests must build in the
//     same environment as the main ED target.
//   * Deterministic: every random fixture takes a seed.
//   * Small: we test against tiny systems (N <= 6 spin-1/2) whose spectra
//     can be cross-checked against a dense Eigen reference in milliseconds.
//   * Compose-friendly: each test file registers individual checks and
//     returns a single int exit code; CTest consumes that directly.
// =============================================================================
#pragma once

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <filesystem>  // P0.12
#include <functional>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <system_error>  // P0.12
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <ed/core/construct_ham.h>

namespace ed_tests {

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

// -----------------------------------------------------------------------------
// Simple check/report utilities.
// -----------------------------------------------------------------------------
struct TestContext {
    std::string suite_name;
    int checks = 0;
    int failures = 0;
    std::vector<std::string> failure_messages;

    explicit TestContext(std::string name) : suite_name(std::move(name)) {}

    void record_pass(const std::string& name) {
        checks++;
        std::cout << "  [PASS] " << name << std::endl;
    }

    void record_fail(const std::string& name, const std::string& detail) {
        checks++;
        failures++;
        std::string msg = name + "  --  " + detail;
        failure_messages.push_back(msg);
        std::cout << "  [FAIL] " << msg << std::endl;
    }

    int summary_exit_code() const {
        std::cout << std::endl;
        std::cout << "[" << suite_name << "] "
                  << (checks - failures) << "/" << checks << " checks passed"
                  << std::endl;
        if (failures > 0) {
            std::cout << "FAILURES:\n";
            for (const auto& m : failure_messages) {
                std::cout << "  * " << m << "\n";
            }
        }
        return failures == 0 ? 0 : 1;
    }
};

// Close comparison helpers.
inline bool near_eq(double a, double b, double tol) {
    return std::abs(a - b) <= tol;
}

inline bool near_eq(Complex a, Complex b, double tol) {
    return std::abs(a - b) <= tol;
}

inline bool check(TestContext& ctx, bool cond, const std::string& name,
                  const std::string& detail = "") {
    if (cond) {
        ctx.record_pass(name);
        return true;
    }
    ctx.record_fail(name, detail);
    return false;
}

inline bool check_near(TestContext& ctx, double a, double b, double tol,
                       const std::string& name) {
    if (near_eq(a, b, tol)) {
        ctx.record_pass(name);
        return true;
    }
    std::ostringstream os;
    os << "expected " << std::setprecision(10) << b
       << " got " << a << " (|diff|=" << std::abs(a - b)
       << ", tol=" << tol << ")";
    ctx.record_fail(name, os.str());
    return false;
}

// Compare two sorted (or sortable) eigenvalue vectors element by element, on
// the overlap length. `tol` is absolute.
inline bool check_eigs_close(TestContext& ctx, std::vector<double> got,
                             std::vector<double> want, size_t n,
                             double tol, const std::string& name) {
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    if (got.size() < n || want.size() < n) {
        std::ostringstream os;
        os << "not enough eigenvalues: got " << got.size()
           << " want " << want.size() << " need " << n;
        ctx.record_fail(name, os.str());
        return false;
    }
    double max_err = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = std::abs(got[i] - want[i]);
        if (e > max_err) { max_err = e; worst = i; }
    }
    if (max_err <= tol) {
        std::ostringstream os;
        os << name << " (n=" << n << ", max|Δλ|=" << max_err << ")";
        ctx.record_pass(os.str());
        return true;
    }
    std::ostringstream os;
    os << "worst at i=" << worst << ": got " << std::setprecision(12)
       << got[worst] << " want " << want[worst] << " |Δ|=" << max_err
       << " > tol=" << tol;
    ctx.record_fail(name, os.str());
    return false;
}

// -----------------------------------------------------------------------------
// Hamiltonian fixtures.
// -----------------------------------------------------------------------------

// Build an open 1D Heisenberg chain on N spin-1/2 sites with coupling J and
// no magnetic field, using the same TransformData storage the real ED code
// uses. The spin operators stored are dimensionful with S = 1/2, so the
// inserted coefficients follow the standard conventions:
//
//   H = J Σ_{i,j bond} ( 1/2*(S+_i S-_j + S-_i S+_j) + Sz_i Sz_j )
//
// We go through the optimized SoA path by pushing TransformData entries and
// letting apply() separateTransformsByType() itself.
inline std::unique_ptr<Operator> build_heisenberg_chain(uint64_t N, double J,
                                                       bool periodic = false) {
    auto op = std::make_unique<Operator>(N, 0.5f);
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    const uint64_t last = periodic ? N : (N - 1);
    for (uint64_t i = 0; i < last; ++i) {
        uint64_t j = (i + 1) % N;
        // Sz_i Sz_j
        {
            Operator::TransformData t;
            t.op_type = 2;
            t.site_index = i;
            t.op_type_2 = 2;
            t.site_index_2 = j;
            t.coefficient = J_real;
            t.is_two_body = true;
            op->transform_data_.push_back(t);
        }
        // 1/2 S+_i S-_j
        {
            Operator::TransformData t;
            t.op_type = 0;
            t.site_index = i;
            t.op_type_2 = 1;
            t.site_index_2 = j;
            t.coefficient = J_half;
            t.is_two_body = true;
            op->transform_data_.push_back(t);
        }
        // 1/2 S-_i S+_j
        {
            Operator::TransformData t;
            t.op_type = 1;
            t.site_index = i;
            t.op_type_2 = 0;
            t.site_index_2 = j;
            t.coefficient = J_half;
            t.is_two_body = true;
            op->transform_data_.push_back(t);
        }
    }
    return op;
}

// Same as above but in a fixed-Sz sector (n_up = number of up spins).
inline std::unique_ptr<FixedSzOperator>
build_heisenberg_chain_fixed_sz(uint64_t N, double J, int64_t n_up,
                                bool periodic = false) {
    auto op = std::make_unique<FixedSzOperator>(N, 0.5f, n_up);
    const Complex J_real(J, 0.0);
    const Complex J_half(0.5 * J, 0.0);
    const uint64_t last = periodic ? N : (N - 1);
    for (uint64_t i = 0; i < last; ++i) {
        uint64_t j = (i + 1) % N;
        Operator::TransformData t;
        t.op_type = 2; t.site_index = i; t.op_type_2 = 2;
        t.site_index_2 = j; t.coefficient = J_real; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 0; t.site_index = i; t.op_type_2 = 1;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op->transform_data_.push_back(t);

        t.op_type = 1; t.site_index = i; t.op_type_2 = 0;
        t.site_index_2 = j; t.coefficient = J_half; t.is_two_body = true;
        op->transform_data_.push_back(t);
    }
    return op;
}

// -----------------------------------------------------------------------------
// Reference dense matrix utilities.
// -----------------------------------------------------------------------------

// Turn the matrix-vector action `Hv` on a Hilbert space of dimension `dim`
// into an explicit `dim x dim` dense matrix by applying H to each canonical
// basis vector. Only viable for tiny dim (we use it for dim <= 64).
template <class Apply>
inline Eigen::MatrixXcd apply_to_dense(Apply&& Hv, uint64_t dim) {
    Eigen::MatrixXcd H = Eigen::MatrixXcd::Zero(dim, dim);
    std::vector<Complex> in(dim), out(dim);
    for (uint64_t j = 0; j < dim; ++j) {
        std::fill(in.begin(), in.end(), Complex(0, 0));
        in[j] = Complex(1.0, 0.0);
        std::fill(out.begin(), out.end(), Complex(0, 0));
        Hv(in.data(), out.data(), static_cast<int>(dim));
        for (uint64_t i = 0; i < dim; ++i) H(i, j) = out[i];
    }
    return H;
}

inline std::vector<double> dense_eigenvalues(const Eigen::MatrixXcd& H) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> s(H);
    std::vector<double> out;
    out.reserve(H.rows());
    for (int i = 0; i < H.rows(); ++i) out.push_back(s.eigenvalues()[i]);
    std::sort(out.begin(), out.end());
    return out;
}

// Convenience: build dense reference from an ED Operator, along with its
// sorted spectrum.
struct DenseReference {
    Eigen::MatrixXcd H;
    std::vector<double> eigs;
};

inline DenseReference reference_from_operator(const Operator& op, uint64_t dim) {
    DenseReference r;
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        op.apply(in, out, static_cast<size_t>(n));
    };
    r.H = apply_to_dense(Hv, dim);
    r.eigs = dense_eigenvalues(r.H);
    return r;
}

inline DenseReference
reference_from_fixed_sz_operator(const FixedSzOperator& op, uint64_t dim) {
    DenseReference r;
    auto Hv = [&](const Complex* in, Complex* out, int n) {
        op.apply(in, out, static_cast<size_t>(n));
    };
    r.H = apply_to_dense(Hv, dim);
    r.eigs = dense_eigenvalues(r.H);
    return r;
}

// -----------------------------------------------------------------------------
// Miscellaneous helpers.
// -----------------------------------------------------------------------------

inline ComplexVector random_unit_vector(uint64_t dim, uint64_t seed) {
    std::mt19937_64 gen(seed);
    std::normal_distribution<double> nd(0.0, 1.0);
    ComplexVector v(dim);
    double n2 = 0.0;
    for (auto& c : v) { c = Complex(nd(gen), nd(gen)); n2 += std::norm(c); }
    double s = 1.0 / std::sqrt(n2);
    for (auto& c : v) c *= s;
    return v;
}

inline double l2_diff(const ComplexVector& a, const ComplexVector& b) {
    if (a.size() != b.size()) return std::numeric_limits<double>::infinity();
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += std::norm(a[i] - b[i]);
    return std::sqrt(s);
}

// Create (and return) a unique per-test temporary directory. The path lives
// under `build/tests/tmp/<suite>_<suffix>/` so concurrent CTest jobs do not
// collide and so artifacts are easy to wipe.
inline std::string make_scratch_dir(const std::string& suite,
                                    const std::string& suffix = "") {
    const char* base_env = std::getenv("ED_TEST_TMP_DIR");
    std::string base = base_env && *base_env ? base_env : "test_scratch";
    std::string dir = base + "/" + suite;
    if (!suffix.empty()) dir += "_" + suffix;
    // P0.12: was system("mkdir -p '...'") (shell-quoted).
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return dir;
}

} // namespace ed_tests
