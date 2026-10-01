// =============================================================================
// Minimal, header-only test fixtures for the ED package.
//
// Design goals:
//   * No test-framework dependency: catch2_harness.h wraps these helpers
//     for the Catch2 suites.
//   * Deterministic: every random fixture takes a seed.
//   * Small: fixtures are tiny spin-1/2 systems whose spectra can be
//     cross-checked against a dense Eigen reference in milliseconds.
// =============================================================================
#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <ed/core/construct_ham.h>

namespace ed_tests {

using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

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
// We go through the optimized SoA path by pushing TransformData entries
// directly into ``op->transform_data_``; ``Operator::commitPendingTransforms``
// (invoked automatically by every matvec entry point) tracks the vector size
// and rebuilds the SoA ``terms_`` cache on the next apply().
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

// A full-space Operator restricted to the fixed-Sz sector with `n_up` set
// bits: embed, apply the full H, gather. Test-only (tiny N); the library's
// Sz sectors are rep sectors of the little-group engine.
class SzSectorOperator final : public ed::LinearOperator {
public:
    SzSectorOperator(std::shared_ptr<const Operator> full, int64_t n_up)
        : full_(std::move(full)) {
        const uint64_t N = full_->getNumBits();
        for (uint64_t s = 0; s < (1ULL << N); ++s)
            if (__builtin_popcountll(s) == n_up) states_.push_back(s);
        xin_.assign(1ULL << N, Complex(0.0, 0.0));
        xout_.assign(1ULL << N, Complex(0.0, 0.0));
    }
    void apply(const Complex* in, Complex* out, std::size_t n) const override {
        std::fill(xin_.begin(), xin_.end(), Complex(0.0, 0.0));
        for (std::size_t i = 0; i < n; ++i) xin_[states_[i]] = in[i];
        full_->apply(xin_.data(), xout_.data(), xin_.size());
        for (std::size_t i = 0; i < n; ++i) out[i] = xout_[states_[i]];
    }
    [[nodiscard]] std::size_t dim() const override { return states_.size(); }
    [[nodiscard]] const Operator& full() const noexcept { return *full_; }

private:
    std::shared_ptr<const Operator> full_;
    std::vector<uint64_t>           states_;
    mutable std::vector<Complex>    xin_, xout_;
};

// Same chain as above, restricted to the fixed-Sz sector with n_up up spins.
inline std::unique_ptr<SzSectorOperator>
build_heisenberg_chain_fixed_sz(uint64_t N, double J, int64_t n_up,
                                bool periodic = false) {
    return std::make_unique<SzSectorOperator>(
        std::shared_ptr<const Operator>(build_heisenberg_chain(N, J, periodic)), n_up);
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

} // namespace ed_tests
