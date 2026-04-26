// =============================================================================
// src/input/file_io.cpp
//
// Writers for the legacy `Trans.dat` / `InterAll.dat` / `ThreeBodyG.dat` /
// `positions.dat` / observable file formats. Header conventions match
// the C++ `Operator::loadFromTransFile` / `loadFromInterAllFile` /
// `loadThreeBodyTerm` / `loadFromDirectory` parsers byte-for-byte.
// =============================================================================

#include <ed/input/file_io.h>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <ios>
#include <sstream>
#include <stdexcept>

namespace ed::input {

namespace {

inline double sanitize(double v, double tol) {
    return std::abs(v) < tol ? 0.0 : v;
}

inline std::string op_label(Op op) {
    switch (op) {
        case Op::Sp: return "S+";
        case Op::Sm: return "S-";
        case Op::Sz: return "Sz";
    }
    return "??";
}

void write_header(std::ofstream& out, const std::string& num_label,
                  std::size_t n) {
    out << "===================\n";
    out << num_label << " " << std::setw(8) << n << "\n";
    out << "===================\n";
    out << "===================\n";
    out << "===================\n";
}

}  // namespace

void write_trans_file(const std::string& path,
                      const std::vector<OneBodyTerm>& terms,
                      double tol) {
    std::vector<std::size_t> kept;
    kept.reserve(terms.size());
    for (std::size_t k = 0; k < terms.size(); ++k) {
        if (std::abs(terms[k].coeff) > tol) kept.push_back(k);
    }
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_trans_file: cannot open " + path);
    write_header(out, "num", kept.size());
    out << std::scientific << std::setprecision(8);
    for (auto k : kept) {
        const auto& t = terms[k];
        out << " " << std::setw(8) << static_cast<int>(op_to_int(t.op)) << "  "
            << std::setw(8) << t.site << "    "
            << std::setw(15) << sanitize(t.coeff.real(), tol) << "    "
            << std::setw(15) << sanitize(t.coeff.imag(), tol) << "\n";
    }
}

void write_inter_all_file(const std::string& path,
                          const std::vector<TwoBodyTerm>& terms,
                          double tol) {
    std::vector<std::size_t> kept;
    kept.reserve(terms.size());
    for (std::size_t k = 0; k < terms.size(); ++k) {
        if (std::abs(terms[k].coeff) > tol) kept.push_back(k);
    }
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_inter_all_file: cannot open " + path);
    write_header(out, "num", kept.size());
    out << std::scientific << std::setprecision(8);
    for (auto k : kept) {
        const auto& t = terms[k];
        out << " " << std::setw(8) << static_cast<int>(op_to_int(t.op_i)) << "  "
            << std::setw(8) << t.site_i << "    "
            << std::setw(8) << static_cast<int>(op_to_int(t.op_j)) << "    "
            << std::setw(8) << t.site_j << "    "
            << std::setw(15) << sanitize(t.coeff.real(), tol) << "    "
            << std::setw(15) << sanitize(t.coeff.imag(), tol) << "\n";
    }
}

void write_three_body_file(const std::string& path,
                           const std::vector<ThreeBodyTerm>& terms,
                           double tol) {
    std::vector<std::size_t> kept;
    kept.reserve(terms.size());
    for (std::size_t k = 0; k < terms.size(); ++k) {
        if (std::abs(terms[k].coeff) > tol) kept.push_back(k);
    }
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_three_body_file: cannot open " + path);
    write_header(out, "num", kept.size());
    out << std::scientific << std::setprecision(8);
    for (auto k : kept) {
        const auto& t = terms[k];
        out << " " << std::setw(8) << static_cast<int>(op_to_int(t.op_i)) << "  "
            << std::setw(8) << t.site_i << "    "
            << std::setw(8) << static_cast<int>(op_to_int(t.op_j)) << "    "
            << std::setw(8) << t.site_j << "    "
            << std::setw(8) << static_cast<int>(op_to_int(t.op_k)) << "    "
            << std::setw(8) << t.site_k << "    "
            << std::setw(15) << sanitize(t.coeff.real(), tol) << "    "
            << std::setw(15) << sanitize(t.coeff.imag(), tol) << "\n";
    }
}

void write_positions_file(const std::string& path,
                          const std::vector<Position>& positions) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_positions_file: cannot open " + path);
    out << std::scientific << std::setprecision(10);
    for (const auto& p : positions) {
        out << p[0] << " " << p[1] << " " << p[2] << "\n";
    }
}

void write_one_body_correlation_file(const std::string& path,
                                     Op op,
                                     std::size_t num_sites) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_one_body_correlation_file: cannot open " + path);
    write_header(out, "loc", num_sites);
    for (std::size_t i = 0; i < num_sites; ++i) {
        out << " " << std::setw(8) << static_cast<int>(op_to_int(op)) << "  "
            << std::setw(8) << i << "\n";
    }
}

void write_two_body_correlation_file(const std::string& path,
                                     Op op_i,
                                     Op op_j,
                                     std::size_t num_sites) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_two_body_correlation_file: cannot open " + path);
    const std::size_t num_pairs = num_sites * num_sites;
    write_header(out, "loc", num_pairs);
    for (std::size_t i = 0; i < num_sites; ++i) {
        for (std::size_t j = 0; j < num_sites; ++j) {
            out << " " << std::setw(8) << static_cast<int>(op_to_int(op_i)) << "  "
                << std::setw(8) << i << "    "
                << std::setw(8) << static_cast<int>(op_to_int(op_j)) << "    "
                << std::setw(8) << j << "\n";
        }
    }
    (void)op_label;  // suppress unused warning when op_label not used inline
}

void write_momentum_observable_file(const std::string& path,
                                    Op op,
                                    const std::array<double, 3>& q,
                                    const std::vector<Position>& positions) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("write_momentum_observable_file: cannot open " + path);
    const std::size_t N = positions.size();
    write_header(out, "loc", N);
    out << std::scientific << std::setprecision(10);
    const double inv_sqrtN = 1.0 / std::sqrt(static_cast<double>(N));
    for (std::size_t i = 0; i < N; ++i) {
        const double phase = q[0] * positions[i][0] + q[1] * positions[i][1] +
                             q[2] * positions[i][2];
        const double re = inv_sqrtN * std::cos(phase);
        const double im = inv_sqrtN * std::sin(phase);
        out << " " << std::setw(8) << static_cast<int>(op_to_int(op)) << "  "
            << std::setw(8) << i << "    "
            << std::setw(15) << re << "    "
            << std::setw(15) << im << "\n";
    }
}

}  // namespace ed::input
