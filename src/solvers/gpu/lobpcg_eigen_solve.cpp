/**
 * @file lobpcg_eigen_solve.cpp
 * @brief Eigen-based generalized eigenvalue solver for GPU LOBPCG
 * 
 * This file is compiled by g++ (not nvcc) to avoid Eigen/CUDA 
 * compilation issues that cause heap corruption when Eigen is 
 * compiled through nvcc.
 */

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <complex>
#include <vector>
#include <iostream>
#include <cmath>

// C-linkage interface to avoid name mangling issues between nvcc and g++
extern "C" {

/**
 * Solve the LOBPCG generalized eigenvalue problem via Löwdin orthogonalization.
 * 
 * Given subspace Hamiltonian h_sub and overlap matrix s_sub (both nsub x nsub, 
 * Hermitian), compute the lowest eigenvalues and the corresponding coefficient 
 * vectors in the original basis.
 * 
 * @param h_sub_data  Column-major interleaved real/imag complex data for H_sub [2 * nsub * nsub]
 * @param s_sub_data  Column-major interleaved real/imag complex data for S_sub [2 * nsub * nsub]
 * @param nsub        Dimension of the subspace
 * @param block_size  Number of eigenvalues/eigenvectors to extract
 * @param out_eigenvalues  Output: eigenvalues [block_size]
 * @param out_coeffs  Output: coefficient vectors, column-major [nsub * block_size]
 * @param rank_threshold  Threshold for dropping small overlap eigenvalues
 * @return nsub_eff (effective rank), or -1 on failure
 */
int lobpcg_solve_generalized_eigenproblem(
    const double* h_sub_data,
    const double* s_sub_data,
    int nsub,
    int block_size,
    double* out_eigenvalues,
    double* out_coeffs,
    double rank_threshold
) {
    // Map input data to Eigen matrices (complex interleaved format: re0,im0,re1,im1,...)
    // h_sub_data and s_sub_data are stored as interleaved real/imag pairs, column-major
    Eigen::MatrixXcd h_sub(nsub, nsub);
    Eigen::MatrixXcd s_sub(nsub, nsub);
    
    for (int col = 0; col < nsub; ++col) {
        for (int row = 0; row < nsub; ++row) {
            int idx = 2 * (col * nsub + row);  // column-major, interleaved real/imag
            h_sub(row, col) = std::complex<double>(h_sub_data[idx], h_sub_data[idx + 1]);
            s_sub(row, col) = std::complex<double>(s_sub_data[idx], s_sub_data[idx + 1]);
        }
    }
    
    // Enforce Hermitian symmetry
    h_sub = 0.5 * (h_sub + h_sub.adjoint());
    s_sub = 0.5 * (s_sub + s_sub.adjoint());
    
    // Diagonalize overlap: S = U * diag(σ) * U^H
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> s_solver(s_sub);
    if (s_solver.info() != Eigen::Success) {
        std::cerr << "LOBPCG Eigen solve: overlap diagonalization failed" << std::endl;
        return -1;
    }
    
    Eigen::VectorXd s_eigs = s_solver.eigenvalues();
    Eigen::MatrixXcd s_vecs = s_solver.eigenvectors();
    
    // Count effective rank
    int nsub_eff = 0;
    for (int i = 0; i < nsub; ++i) {
        if (s_eigs(i) > rank_threshold) nsub_eff++;
    }
    
    if (nsub_eff < block_size) {
        std::cerr << "LOBPCG Eigen solve: overlap rank (" << nsub_eff 
                  << ") < block_size (" << block_size << ")" << std::endl;
        return -1;
    }
    
    // Construct S^{-1/2}
    Eigen::MatrixXcd s_inv_sqrt = Eigen::MatrixXcd::Zero(nsub, nsub_eff);
    int idx = 0;
    for (int i = 0; i < nsub; ++i) {
        if (s_eigs(i) > rank_threshold) {
            s_inv_sqrt.col(idx) = s_vecs.col(i) / std::sqrt(s_eigs(i));
            idx++;
        }
    }
    
    // Transform: H' = S^{-1/2}^H * H * S^{-1/2}
    Eigen::MatrixXcd h_prime = s_inv_sqrt.adjoint() * h_sub * s_inv_sqrt;
    
    // Enforce Hermiticity
    h_prime = 0.5 * (h_prime + h_prime.adjoint());
    
    // Solve: H' * z = eigenvalue * z
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> h_solver(h_prime);
    if (h_solver.info() != Eigen::Success) {
        std::cerr << "LOBPCG Eigen solve: H' diagonalization failed" << std::endl;
        return -1;
    }
    
    Eigen::VectorXd ritz_eigenvalues = h_solver.eigenvalues();
    Eigen::MatrixXcd ritz_coeffs = h_solver.eigenvectors();
    
    // Back-transform: coefficients in the original basis
    Eigen::MatrixXcd full_coeffs = s_inv_sqrt * ritz_coeffs;
    
    // Copy outputs
    for (int b = 0; b < block_size; ++b) {
        out_eigenvalues[b] = ritz_eigenvalues(b);
    }
    
    // Output coefficients in column-major, interleaved real/imag format
    for (int b = 0; b < block_size; ++b) {
        for (int i = 0; i < nsub; ++i) {
            int oidx = 2 * (b * nsub + i);  // column-major
            out_coeffs[oidx]     = full_coeffs(i, b).real();
            out_coeffs[oidx + 1] = full_coeffs(i, b).imag();
        }
    }
    
    return nsub_eff;
}

}  // extern "C"
