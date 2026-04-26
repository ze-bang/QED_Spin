// =============================================================================
// include/ed/input/file_io.h
//
// Low-level writers for the legacy `Trans.dat` / `InterAll.dat` /
// `ThreeBodyG.dat` / `positions.dat` formats. These are the same files
// `Operator::loadFromDirectory` consumes, with byte-for-byte the same
// header convention emitted by the legacy Python helpers.
//
// Most users should reach for `HamiltonianBuilder::write_directory()`
// instead; these free functions are exposed so that callers building
// terms via paths other than the builder can still produce a directory
// the production `./ED` binary will accept.
// =============================================================================

#pragma once

#include <string>
#include <vector>

#include <ed/input/types.h>

namespace ed::input {

// Write a `Trans.dat` line per `OneBodyTerm`, skipping terms with
// |coeff| < tol. Header matches the C++ loader.
void write_trans_file(
    const std::string& path,
    const std::vector<OneBodyTerm>& terms,
    double tol = 1e-15);

// Write an `InterAll.dat` line per `TwoBodyTerm`.
void write_inter_all_file(
    const std::string& path,
    const std::vector<TwoBodyTerm>& terms,
    double tol = 1e-15);

// Write a `ThreeBodyG.dat` line per `ThreeBodyTerm`. Layout matches
// `Operator::loadThreeBodyTerm`'s parser:
//   op1 site1 op2 site2 op3 site3  Re  Im
void write_three_body_file(
    const std::string& path,
    const std::vector<ThreeBodyTerm>& terms,
    double tol = 1e-15);

// Write a `positions.dat` file: one line per site, "x y z".
void write_positions_file(
    const std::string& path,
    const std::vector<Position>& positions);

// Standard observable files used by every helper. Contents:
//
//   one_body_correlations<OpName>.dat      - one row per site.
//   two_body_correlations<O1><O2>.dat      - one row per (i, j) pair.
//
// `op_name` is "S+" for Op::Sp, "S-" for Op::Sm, "Sz" for Op::Sz.
void write_one_body_correlation_file(
    const std::string& path,
    Op op,
    std::size_t num_sites);

void write_two_body_correlation_file(
    const std::string& path,
    Op op_i,
    Op op_j,
    std::size_t num_sites);

// Write a momentum-projected one-body observable, e.g.
//   O_q = (1/sqrt(N)) sum_i e^{i q . r_i} S^a_i.
//
// Layout:
//   ===================
//   loc        N
//   ===================
//   ===================
//   ===================
//    op   site   Re   Im
//    ...
void write_momentum_observable_file(
    const std::string& path,
    Op op,
    const std::array<double, 3>& q,
    const std::vector<Position>& positions);

}  // namespace ed::input
