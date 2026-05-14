#include "common/catch2_harness.h"

#include <ed/solvers/ltlm.h>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <vector>

using ed_tests::build_heisenberg_chain;
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;

namespace {

Eigen::MatrixXcd to_dense(std::function<void(const Complex*, Complex*, int)> op,
                          uint64_t dim)
{
    Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(dim, dim);
    ComplexVector in(dim, Complex(0.0, 0.0));
    ComplexVector out(dim, Complex(0.0, 0.0));
    for (uint64_t j = 0; j < dim; ++j) {
        std::fill(in.begin(), in.end(), Complex(0.0, 0.0));
        std::fill(out.begin(), out.end(), Complex(0.0, 0.0));
        in[j] = Complex(1.0, 0.0);
        op(in.data(), out.data(), static_cast<int>(dim));
        for (uint64_t i = 0; i < dim; ++i) {
            M(i, j) = out[i];
        }
    }
    return M;
}

double exact_connected_hh(const Eigen::VectorXd& E, double beta, double e_min)
{
    const int dim = static_cast<int>(E.size());
    double Z = 0.0;
    double e_avg = 0.0;
    double e2_avg = 0.0;
    for (int n = 0; n < dim; ++n) {
        const double w = std::exp(-beta * (E(n) - e_min));
        Z += w;
        e_avg += w * E(n);
        e2_avg += w * E(n) * E(n);
    }
    e_avg /= Z;
    e2_avg /= Z;
    return e2_avg - e_avg * e_avg;
}

} // namespace

TEST_CASE("LTLM connected Q-H response matches exact H-H covariance",
          "[ltlm][static][connected]")
{
    const uint64_t N = 6;
    const uint64_t dim = 1ULL << N;

    auto H_op = build_heisenberg_chain(N, 1.0, true);
    auto* H_ptr = H_op.get();
    auto Hv = [H_ptr](const Complex* in, Complex* out, int n) {
        H_ptr->apply(in, out, static_cast<size_t>(n));
    };

    const Eigen::MatrixXcd H_dense = to_dense(Hv, dim);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(H_dense);
    const Eigen::VectorXd E = es.eigenvalues();
    const double e_min = E(0);

    LTLMParameters params;
    params.krylov_dim = dim;                // keep all Ritz states
    params.ground_state_krylov = dim;       // outer Lanczos reaches full space
    params.full_reorthogonalization = true;
    params.tolerance = 1e-12;
    params.random_seed = 20260513ULL;

    const double temp_min = 0.25;
    const double temp_max = 2.5;
    const uint64_t n_temps = 7;

    const auto res = compute_connected_qh_response_ltlm(
        Hv, Hv, dim, params, temp_min, temp_max, n_temps);

    REQUIRE(res.temperatures.size() == n_temps);
    REQUIRE(res.expectation.size() == n_temps);
    REQUIRE(res.variance.size() == n_temps);
    REQUIRE(res.susceptibility.size() == n_temps);
    REQUIRE(res.total_samples == 1);

    for (uint64_t t = 0; t < n_temps; ++t) {
        const double T = res.temperatures[t];
        const double beta = 1.0 / T;
        const double exact_conn = exact_connected_hh(E, beta, e_min);
        const double exact_alpha = exact_conn / (T * T);
        const double exact_susc = exact_conn / T;

        INFO("T=" << T
             << " conn=" << res.variance[t] << " exact_conn=" << exact_conn
             << " alpha=" << res.expectation[t] << " exact_alpha=" << exact_alpha);

        const double conn_scale = std::max(1.0, std::abs(exact_conn));
        const double alpha_scale = std::max(1.0, std::abs(exact_alpha));
        const double susc_scale = std::max(1.0, std::abs(exact_susc));

        REQUIRE(std::abs(res.variance[t] - exact_conn) <= 1e-7 * conn_scale);
        REQUIRE(std::abs(res.expectation[t] - exact_alpha) <= 1e-7 * alpha_scale);
        REQUIRE(std::abs(res.susceptibility[t] - exact_susc) <= 1e-7 * susc_scale);
        REQUIRE(res.expectation_error[t] == 0.0);
        REQUIRE(res.variance_error[t] == 0.0);
        REQUIRE(res.susceptibility_error[t] == 0.0);
    }
}
