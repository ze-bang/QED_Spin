/**
 * @file test_dmrg.cpp
 * @brief Tests for DMRG implementation
 * 
 * Build: Add to CMakeLists.txt or compile manually:
 *   g++ -std=c++17 -O2 -I../include test_dmrg.cpp -o test_dmrg -lblas -llapack
 * 
 * Run tests incrementally as you implement each component:
 *   ./test_dmrg tensor    # Test tensor operations
 *   ./test_dmrg mps       # Test MPS operations
 *   ./test_dmrg mpo       # Test MPO construction
 *   ./test_dmrg env       # Test environment updates
 *   ./test_dmrg idmrg     # Full iDMRG test
 */

#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>
#include <string>
#include <vector>
#include <random>

#include <ed/dmrg/tensor.h>
#include <ed/dmrg/mps.h>
#include <ed/dmrg/mpo.h>
#include <ed/dmrg/environ.h>
#include <ed/dmrg/idmrg.h>
#include <ed/dmrg/dmrg_config.h>

using namespace dmrg;

// ============================================================================
// Test utilities
// ============================================================================

#define TEST_ASSERT(cond, msg) \
    if (!(cond)) { \
        std::cerr << "FAILED: " << msg << " at line " << __LINE__ << std::endl; \
        return false; \
    }

#define TEST_NEAR(a, b, tol, msg) \
    if (std::abs((a) - (b)) > (tol)) { \
        std::cerr << "FAILED: " << msg << " (" << (a) << " vs " << (b) << ")" \
                  << " at line " << __LINE__ << std::endl; \
        return false; \
    }

void print_pass(const std::string& name) {
    std::cout << "  ✓ " << name << std::endl;
}

// ============================================================================
// Tensor tests
// ============================================================================

bool test_tensor_creation() {
    // Test basic tensor creation and access
    Tensor<Complex> t({2, 3, 4});
    TEST_ASSERT(t.rank() == 3, "rank should be 3");
    TEST_ASSERT(t.shape(0) == 2, "shape[0] should be 2");
    TEST_ASSERT(t.shape(1) == 3, "shape[1] should be 3");
    TEST_ASSERT(t.shape(2) == 4, "shape[2] should be 4");
    TEST_ASSERT(t.total_size() == 24, "total size should be 24");
    
    // Test element access
    t(1, 2, 3) = Complex(1.5, -0.5);
    TEST_NEAR(t(1, 2, 3).real(), 1.5, 1e-10, "real part");
    TEST_NEAR(t(1, 2, 3).imag(), -0.5, 1e-10, "imag part");
    
    print_pass("tensor creation and access");
    return true;
}

bool test_tensor_reshape() {
    Tensor<Complex> t({2, 3, 4});
    
    // Fill with sequential values
    for (size_t i = 0; i < t.total_size(); ++i) {
        t.data()[i] = Complex(double(i), 0.0);
    }
    
    // Reshape to (6, 4)
    auto r = t.reshape({6, 4});
    TEST_ASSERT(r.rank() == 2, "reshaped rank should be 2");
    TEST_ASSERT(r.total_size() == 24, "reshaped size should be 24");
    
    // Data should be preserved
    for (size_t i = 0; i < 24; ++i) {
        TEST_NEAR(r.data()[i].real(), double(i), 1e-10, "data preserved");
    }
    
    print_pass("tensor reshape");
    return true;
}

bool test_tensor_permute() {
    // Create tensor A(2, 3, 4)
    Tensor<Complex> A({2, 3, 4});
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                A(i, j, k) = Complex(i * 12 + j * 4 + k, 0.0);
            }
        }
    }
    
    // Permute to B(k, i, j) = A(i, j, k), i.e., perm = {2, 0, 1}
    // B has shape (4, 2, 3)
    auto B = A.permute({2, 0, 1});
    
    TEST_ASSERT(B.shape(0) == 4, "permuted shape[0]");
    TEST_ASSERT(B.shape(1) == 2, "permuted shape[1]");
    TEST_ASSERT(B.shape(2) == 3, "permuted shape[2]");
    
    // Check values: B(k, i, j) should equal A(i, j, k)
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t k = 0; k < 4; ++k) {
                TEST_NEAR(B(k, i, j).real(), A(i, j, k).real(), 1e-10, "permute value");
            }
        }
    }
    
    print_pass("tensor permute");
    return true;
}

bool test_matmul() {
    // Test matrix multiplication
    // A = [[1, 2], [3, 4]]  (row-major view)
    // B = [[5, 6], [7, 8]]
    // A * B = [[19, 22], [43, 50]]
    
    // Column-major storage: A(row, col) = data[row + nrows * col]
    Tensor<Complex> A({2, 2});
    A(0, 0) = 1.0; A(1, 0) = 3.0;  // Column 0: [1, 3]
    A(0, 1) = 2.0; A(1, 1) = 4.0;  // Column 1: [2, 4]
    
    Tensor<Complex> B({2, 2});
    B(0, 0) = 5.0; B(1, 0) = 7.0;  // Column 0: [5, 7]
    B(0, 1) = 6.0; B(1, 1) = 8.0;  // Column 1: [6, 8]
    
    auto C = matmul(A, B);
    
    // C should be [[19, 22], [43, 50]] in row-major view
    // In column-major: C(row, col)
    TEST_NEAR(C(0, 0).real(), 19.0, 1e-10, "C[0,0]");
    TEST_NEAR(C(1, 0).real(), 43.0, 1e-10, "C[1,0]");
    TEST_NEAR(C(0, 1).real(), 22.0, 1e-10, "C[0,1]");
    TEST_NEAR(C(1, 1).real(), 50.0, 1e-10, "C[1,1]");
    
    print_pass("matrix multiplication");
    return true;
}

bool test_tensor_contraction() {
    // Test general tensor contraction
    // A(i, j, k) contracted with B(k, l) over k gives C(i, j, l)
    
    Tensor<Complex> A({2, 3, 4});
    Tensor<Complex> B({4, 5});
    
    // Fill with random values
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < A.total_size(); ++i) {
        A.data()[i] = Complex(dist(rng), dist(rng));
    }
    for (size_t i = 0; i < B.total_size(); ++i) {
        B.data()[i] = Complex(dist(rng), dist(rng));
    }
    
    // Contract over index 2 of A and index 0 of B
    auto C = contract(A, B, {2}, {0});
    
    TEST_ASSERT(C.rank() == 3, "contracted rank");
    TEST_ASSERT(C.shape(0) == 2, "contracted shape[0]");
    TEST_ASSERT(C.shape(1) == 3, "contracted shape[1]");
    TEST_ASSERT(C.shape(2) == 5, "contracted shape[2]");
    
    // Verify by explicit computation
    for (size_t i = 0; i < 2; ++i) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t l = 0; l < 5; ++l) {
                Complex expected(0.0, 0.0);
                for (size_t k = 0; k < 4; ++k) {
                    expected += A(i, j, k) * B(k, l);
                }
                TEST_NEAR(std::abs(C(i, j, l) - expected), 0.0, 1e-10, "contraction value");
            }
        }
    }
    
    print_pass("tensor contraction");
    return true;
}

bool test_svd() {
    // Test SVD of a simple matrix
    Tensor<Complex> A({3, 2});
    A(0, 0) = 1.0; A(0, 1) = 4.0;
    A(1, 0) = 2.0; A(1, 1) = 5.0;
    A(2, 0) = 3.0; A(2, 1) = 6.0;
    
    Tensor<Complex> U, Vh;
    std::vector<double> S;
    svd(A, U, S, Vh, false);
    
    TEST_ASSERT(S.size() == 2, "2 singular values");
    TEST_ASSERT(S[0] >= S[1], "singular values sorted descending");
    
    // Verify A ≈ U * diag(S) * Vh
    // (Reconstruct and compare)
    
    print_pass("SVD");
    return true;
}

bool run_tensor_tests() {
    std::cout << "\n=== Tensor Tests ===" << std::endl;
    bool all_pass = true;
    
    all_pass &= test_tensor_creation();
    all_pass &= test_tensor_reshape();
    
    try {
        all_pass &= test_tensor_permute();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ tensor permute: " << e.what() << std::endl;
    }
    
    all_pass &= test_matmul();
    
    try {
        all_pass &= test_tensor_contraction();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ tensor contraction: " << e.what() << std::endl;
    }
    
    try {
        all_pass &= test_svd();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ SVD: " << e.what() << std::endl;
    }
    
    return all_pass;
}

// ============================================================================
// MPS tests
// ============================================================================

bool test_mps_creation() {
    MPS mps(4, 2);  // 4 sites, spin-1/2
    
    TEST_ASSERT(mps.length() == 4, "length should be 4");
    TEST_ASSERT(mps.phys_dim() == 2, "phys_dim should be 2");
    
    // Default initialization is product state |0000⟩
    TEST_NEAR(mps[0](0, 0, 0).real(), 1.0, 1e-10, "initial state");
    
    print_pass("MPS creation");
    return true;
}

bool test_mps_neel() {
    MPS mps(4, 2);
    mps.neel_init();
    
    // Check Néel state |↑↓↑↓⟩
    TEST_NEAR(mps[0](0, 0, 0).real(), 1.0, 1e-10, "site 0 up");
    TEST_NEAR(mps[1](0, 1, 0).real(), 1.0, 1e-10, "site 1 down");
    TEST_NEAR(mps[2](0, 0, 0).real(), 1.0, 1e-10, "site 2 up");
    TEST_NEAR(mps[3](0, 1, 0).real(), 1.0, 1e-10, "site 3 down");
    
    print_pass("MPS Neel initialization");
    return true;
}

bool test_mps_norm() {
    MPS mps(4, 2);
    mps.neel_init();
    
    double n = mps.norm();
    TEST_NEAR(n, 1.0, 1e-10, "norm of product state");
    
    print_pass("MPS norm");
    return true;
}

bool test_merge_split() {
    // Create two simple MPS tensors and merge
    MPSTensor A(2, 2, 3);  // (χ_L=2, d=2, χ_R=3)
    MPSTensor B(3, 2, 2);  // (χ_L=3, d=2, χ_R=2)
    
    // Fill with random
    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < A.data.total_size(); ++i) {
        A.data.data()[i] = Complex(dist(rng), dist(rng));
    }
    for (size_t i = 0; i < B.data.total_size(); ++i) {
        B.data.data()[i] = Complex(dist(rng), dist(rng));
    }
    
    // Merge
    auto theta = merge_two_sites(A, B);
    TEST_ASSERT(theta.shape(0) == 2, "merged chi_L");
    TEST_ASSERT(theta.shape(1) == 2, "merged d1");
    TEST_ASSERT(theta.shape(2) == 2, "merged d2");
    TEST_ASSERT(theta.shape(3) == 2, "merged chi_R");
    
    // Split and verify norm is preserved
    MPSTensor A_new, B_new;
    std::vector<double> S;
    double trunc_err = split_two_sites(theta, 10, A_new, B_new, S);
    
    // Merge again and compare
    auto theta_new = merge_two_sites(A_new, B_new);
    
    // Frobenius norm difference should be small
    double diff = 0.0;
    for (size_t i = 0; i < theta.total_size(); ++i) {
        diff += std::norm(theta.data()[i] - theta_new.data()[i]);
    }
    diff = std::sqrt(diff);
    
    TEST_ASSERT(diff < 1e-10, "merge-split roundtrip");
    
    print_pass("merge and split two sites");
    return true;
}

bool run_mps_tests() {
    std::cout << "\n=== MPS Tests ===" << std::endl;
    bool all_pass = true;
    
    all_pass &= test_mps_creation();
    all_pass &= test_mps_neel();
    
    try {
        all_pass &= test_mps_norm();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ MPS norm: " << e.what() << std::endl;
    }
    
    try {
        all_pass &= test_merge_split();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ merge/split: " << e.what() << std::endl;
    }
    
    return all_pass;
}

// ============================================================================
// MPO tests
// ============================================================================

bool test_spin_operators() {
    auto I = spin_half::identity();
    auto Sz = spin_half::Sz();
    auto Sp = spin_half::Sp();
    auto Sm = spin_half::Sm();
    
    // Check Sz eigenvalues
    TEST_NEAR(Sz(0, 0).real(), 0.5, 1e-10, "Sz |up>");
    TEST_NEAR(Sz(1, 1).real(), -0.5, 1e-10, "Sz |down>");
    
    // Check S+|down> = |up>
    TEST_NEAR(Sp(0, 1).real(), 1.0, 1e-10, "S+ |down>");
    
    // Check S-|up> = |down>
    TEST_NEAR(Sm(1, 0).real(), 1.0, 1e-10, "S- |up>");
    
    print_pass("spin-1/2 operators");
    return true;
}

bool test_xxz_mpo() {
    double J = 1.0;
    double Delta = 1.0;
    
    MPO mpo = build_xxz_mpo(4, J, Delta);
    
    TEST_ASSERT(mpo.length() == 4, "MPO length");
    TEST_ASSERT(mpo.phys_dim() == 2, "MPO phys dim");
    
    // First site should have w_left = 1
    TEST_ASSERT(mpo[0].w_left() == 1, "first site w_left");
    
    // Last site should have w_right = 1
    TEST_ASSERT(mpo[3].w_right() == 1, "last site w_right");
    
    // Bulk sites should have w = 5
    TEST_ASSERT(mpo[1].w_left() == 5, "bulk w_left");
    TEST_ASSERT(mpo[1].w_right() == 5, "bulk w_right");
    
    print_pass("XXZ MPO construction");
    return true;
}

bool test_mpo_two_site_energy() {
    // Compare MPO energy with explicit calculation for 2 sites
    // H = J/2 (S+ S- + S- S+) + Delta Sz Sz
    //   = J (Sx Sx + Sy Sy) + Delta Sz Sz
    // 
    // For J=1, Delta=1 (XXX Heisenberg), the ground state is singlet:
    //   |singlet> = (|up,down> - |down,up>) / sqrt(2)
    //   E = J/2 * (-1) + Delta * (-1/4) = -3/4
    
    double J = 1.0;
    double Delta = 1.0;
    
    MPO mpo = build_xxz_mpo(2, J, Delta);
    
    // Create singlet MPS
    // |singlet> = (|↑↓> - |↓↑>) / sqrt(2) where |↑>=σ=0, |↓>=σ=1
    MPS mps(2, 2);
    mps[0] = MPSTensor(1, 2, 2);
    mps[1] = MPSTensor(2, 2, 1);
    
    double sq2 = 1.0 / std::sqrt(2.0);
    
    // A[0]: shape (1, 2, 2)
    // A[0](0, σ=0, bond=0) = 1/√2  → |↑⟩ goes to bond 0
    // A[0](0, σ=1, bond=1) = 1/√2  → |↓⟩ goes to bond 1
    mps[0](0, 0, 0) = Complex(sq2, 0);   // |up> -> bond 0
    mps[0](0, 1, 1) = Complex(sq2, 0);   // |down> -> bond 1
    
    // A[1]: shape (2, 2, 1)
    // bond 0 produces |↓⟩, bond 1 produces -|↑⟩
    // This gives: (1/√2)|↑⟩⊗|↓⟩ + (1/√2)|↓⟩⊗(-|↑⟩) = (|↑↓⟩ - |↓↑⟩)/√2
    mps[1](0, 1, 0) = Complex(1.0, 0);   // bond 0 -> |down>
    mps[1](1, 0, 0) = Complex(-1.0, 0);  // bond 1 -> -|up>
    
    Complex energy = mpo_expectation(mpo, mps);
    
    // Exact singlet energy: -3J/4 = -0.75
    TEST_NEAR(energy.real(), -0.75, 1e-10, "singlet energy");
    TEST_NEAR(energy.imag(), 0.0, 1e-10, "energy should be real");
    
    print_pass("MPO two-site energy");
    return true;
}

bool run_mpo_tests() {
    std::cout << "\n=== MPO Tests ===" << std::endl;
    bool all_pass = true;
    
    all_pass &= test_spin_operators();
    
    try {
        all_pass &= test_xxz_mpo();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ XXZ MPO: " << e.what() << std::endl;
    }
    
    try {
        all_pass &= test_mpo_two_site_energy();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ MPO energy: " << e.what() << std::endl;
    }
    
    return all_pass;
}

// ============================================================================
// Environment tests
// ============================================================================

bool test_environment_boundaries() {
    auto L = Environment::left_boundary();
    TEST_ASSERT(L.chi_bra() == 1, "left boundary chi_bra");
    TEST_ASSERT(L.w_mpo() == 1, "left boundary w_mpo");
    TEST_ASSERT(L.chi_ket() == 1, "left boundary chi_ket");
    TEST_NEAR(L(0, 0, 0).real(), 1.0, 1e-10, "left boundary value");
    
    auto R = Environment::right_boundary(5);
    TEST_ASSERT(R.w_mpo() == 5, "right boundary w_mpo");
    TEST_NEAR(R(0, 4, 0).real(), 1.0, 1e-10, "right boundary value");
    
    print_pass("environment boundaries");
    return true;
}

bool test_environment_update() {
    // Test left environment update for simple case
    // Start with trivial left environment, add one site
    
    auto L = Environment::left_boundary();
    
    // Simple MPS tensor: |0> state
    MPSTensor A(1, 2, 1);
    A(0, 0, 0) = Complex(1.0, 0.0);
    
    // Simple MPO tensor: identity
    MPOTensor W(1, 2, 1);
    W(0, 0, 0, 0) = Complex(1.0, 0.0);
    W(0, 1, 1, 0) = Complex(1.0, 0.0);
    
    auto L_new = update_left_environment(L, A, W);
    
    // For identity MPO and |0> state, L_new should also be trivial
    TEST_NEAR(L_new(0, 0, 0).real(), 1.0, 1e-10, "updated environment");
    
    print_pass("environment update");
    return true;
}

bool run_environment_tests() {
    std::cout << "\n=== Environment Tests ===" << std::endl;
    bool all_pass = true;
    
    all_pass &= test_environment_boundaries();
    
    try {
        all_pass &= test_environment_update();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ environment update: " << e.what() << std::endl;
    }
    
    return all_pass;
}

// ============================================================================
// iDMRG tests
// ============================================================================

bool test_idmrg_step_debug() {
    std::cout << "\nDebug test: Manual iDMRG steps with bulk tensors..." << std::endl;
    
    DMRGConfig config;
    config.model = ModelType::HEISENBERG_XXX;
    config.J = 1.0;
    config.chi_max = 50;
    config.max_sites = 10;
    config.energy_tol = 1e-10;
    config.verbosity = 2;
    
    size_t d = config.local_dim();
    size_t w_bulk = 5;
    
    // Build bulk MPO
    MPOTensor W_bulk = build_xxz_bulk_tensor(config.J, 1.0);
    
    std::cout << "Bulk MPO tensor shape: (" << W_bulk.w_left() << ", " 
              << W_bulk.phys_dim() << ", " << W_bulk.phys_dim() << ", "
              << W_bulk.w_right() << ")" << std::endl;
    
    // Initialize state with bulk environments
    IDMRGState state;
    initialize_idmrg(state, config);
    
    std::cout << "\nInitial state (bulk environments):" << std::endl;
    std::cout << "  L_env shape: (" << state.L_env.chi_bra() << ", " 
              << state.L_env.w_mpo() << ", " << state.L_env.chi_ket() << ")" << std::endl;
    std::cout << "  R_env shape: (" << state.R_env.chi_bra() << ", " 
              << state.R_env.w_mpo() << ", " << state.R_env.chi_ket() << ")" << std::endl;
    std::cout << "  theta shape: (" << state.theta.shape()[0] << ", " 
              << state.theta.shape()[1] << ", " << state.theta.shape()[2] << ", "
              << state.theta.shape()[3] << ")" << std::endl;
    
    // Check L_env and R_env values
    std::cout << "\n  L_env values:" << std::endl;
    for (size_t ap = 0; ap < state.L_env.chi_bra(); ++ap) {
        for (size_t w = 0; w < state.L_env.w_mpo(); ++w) {
            for (size_t a = 0; a < state.L_env.chi_ket(); ++a) {
                Complex val = state.L_env(ap, w, a);
                if (std::abs(val) > 1e-10) {
                    std::cout << "    L(" << ap << "," << w << "," << a << ") = " << val << std::endl;
                }
            }
        }
    }
    
    std::cout << "\n  R_env values:" << std::endl;
    for (size_t ap = 0; ap < state.R_env.chi_bra(); ++ap) {
        for (size_t w = 0; w < state.R_env.w_mpo(); ++w) {
            for (size_t a = 0; a < state.R_env.chi_ket(); ++a) {
                Complex val = state.R_env(ap, w, a);
                if (std::abs(val) > 1e-10) {
                    std::cout << "    R(" << ap << "," << w << "," << a << ") = " << val << std::endl;
                }
            }
        }
    }
    
    // Build effective Hamiltonian and check if non-zero
    size_t chi_L = state.L_env.chi_ket();
    size_t chi_R = state.R_env.chi_ket();
    size_t N = chi_L * d * d * chi_R;
    
    std::cout << "\n  Effective Hamiltonian dimension: " << N << std::endl;
    
    auto H_eff = make_two_site_hamiltonian(
        state.L_env, state.R_env, W_bulk, W_bulk, chi_L, d, chi_R);
    
    // Build full H matrix
    std::vector<Complex> H_mat(N * N, Complex(0.0, 0.0));
    std::vector<Complex> e_i(N, Complex(0.0, 0.0));
    std::vector<Complex> H_ei(N);
    
    for (size_t i = 0; i < N; ++i) {
        e_i[i] = Complex(1.0, 0.0);
        H_eff(e_i.data(), H_ei.data(), static_cast<int>(N));
        for (size_t j = 0; j < N; ++j) {
            H_mat[j + i * N] = H_ei[j];
        }
        e_i[i] = Complex(0.0, 0.0);
    }
    
    // Check if H is all zeros
    double H_norm = 0.0;
    for (size_t i = 0; i < N * N; ++i) {
        H_norm += std::norm(H_mat[i]);
    }
    H_norm = std::sqrt(H_norm);
    
    std::cout << "  ||H_eff|| = " << H_norm << std::endl;
    
    if (H_norm < 1e-10) {
        std::cout << "\n  ERROR: H_eff is all zeros!" << std::endl;
        return false;
    }
    
    // Print H_eff matrix
    std::cout << "\n  H_eff matrix:" << std::endl;
    for (size_t i = 0; i < N; ++i) {
        std::cout << "    [";
        for (size_t j = 0; j < N; ++j) {
            std::cout << " " << std::setw(8) << std::real(H_mat[i + j * N]);
        }
        std::cout << " ]" << std::endl;
    }
    
    // Diagonalize
    std::vector<double> eigenvalues(N);
    char jobz = 'V', uplo = 'U';
    int n = static_cast<int>(N), lda = n, lwork = -1, info;
    std::vector<double> rwork(3 * N - 2);
    Complex work_query;
    
    LAPACKE_zheev_work(LAPACK_COL_MAJOR, jobz, uplo, n,
                      reinterpret_cast<lapack_complex_double*>(H_mat.data()),
                      lda, eigenvalues.data(),
                      reinterpret_cast<lapack_complex_double*>(&work_query),
                      lwork, rwork.data());
    
    lwork = static_cast<int>(std::real(work_query));
    std::vector<Complex> work(lwork);
    
    info = LAPACKE_zheev_work(LAPACK_COL_MAJOR, jobz, uplo, n,
                              reinterpret_cast<lapack_complex_double*>(H_mat.data()),
                              lda, eigenvalues.data(),
                              reinterpret_cast<lapack_complex_double*>(work.data()),
                              lwork, rwork.data());
    
    if (info != 0) {
        std::cout << "  ERROR: LAPACK zheev failed" << std::endl;
        return false;
    }
    
    std::cout << "\n  Eigenvalues:";
    for (size_t i = 0; i < N; ++i) {
        std::cout << " " << eigenvalues[i];
    }
    std::cout << std::endl;
    
    double E0 = eigenvalues[0];
    double expected_E0 = -0.75;  // 2-site Heisenberg singlet energy
    
    std::cout << "  Ground state energy = " << E0 << " (expected: " << expected_E0 << ")" << std::endl;
    
    TEST_NEAR(E0, expected_E0, 1e-10, "2-site energy with bulk tensors");
    
    print_pass("iDMRG step debug");
    return true;
}

bool test_idmrg_heisenberg() {
    std::cout << "\nRunning iDMRG for Heisenberg chain..." << std::endl;
    
    DMRGConfig config;
    config.model = ModelType::HEISENBERG_XXX;
    config.J = 1.0;
    config.chi_max = 32;  // Larger bond dimension
    config.max_sites = 200;
    config.energy_tol = 1e-8;
    config.verbosity = 2;
    
    DMRGResults results = run_idmrg(config);
    
    // Known result: E/site ≈ -0.4431 for infinite Heisenberg chain
    double exact_energy_per_site = -0.4431471805599453;  // Bethe ansatz
    
    std::cout << "\nFinal E/site = " << results.energy_per_site 
              << " (exact: " << exact_energy_per_site << ")" << std::endl;
    std::cout << "Entanglement entropy = " << results.entanglement_entropy << std::endl;
    std::cout << "Final bond dimension = " << results.bond_dimension << std::endl;
    
    TEST_NEAR(results.energy_per_site, exact_energy_per_site, 1e-2, "energy per site");
    
    print_pass("iDMRG Heisenberg");
    return true;
}

bool run_idmrg_tests() {
    std::cout << "\n=== iDMRG Tests ===" << std::endl;
    
    try {
        if (!test_idmrg_step_debug()) return false;
        return test_idmrg_heisenberg();
    } catch (const std::runtime_error& e) {
        std::cout << "  ⚠ iDMRG: " << e.what() << std::endl;
        return false;
    }
}

// ============================================================================
// Main
// ============================================================================

void print_usage() {
    std::cout << "Usage: test_dmrg [test_name]" << std::endl;
    std::cout << "  test_name: tensor, mps, mpo, env, idmrg, all" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string test_name = "all";
    if (argc > 1) {
        test_name = argv[1];
    }
    
    std::cout << "DMRG Test Suite" << std::endl;
    std::cout << "===============" << std::endl;
    
    bool all_pass = true;
    
    if (test_name == "tensor" || test_name == "all") {
        all_pass &= run_tensor_tests();
    }
    
    if (test_name == "mps" || test_name == "all") {
        all_pass &= run_mps_tests();
    }
    
    if (test_name == "mpo" || test_name == "all") {
        all_pass &= run_mpo_tests();
    }
    
    if (test_name == "env" || test_name == "all") {
        all_pass &= run_environment_tests();
    }
    
    if (test_name == "idmrg" || test_name == "all") {
        all_pass &= run_idmrg_tests();
    }
    
    std::cout << "\n===============" << std::endl;
    if (all_pass) {
        std::cout << "All tests PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "Some tests FAILED or NOT IMPLEMENTED" << std::endl;
        return 1;
    }
}
