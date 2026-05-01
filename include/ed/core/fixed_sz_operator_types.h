#pragma once
// =============================================================================
// include/ed/core/fixed_sz_operator_types.h
//
// Concrete fixed-Sz operator subclasses:
//   FixedSzSingleSiteOperator, FixedSzDoubleSiteOperator,
//   FixedSzBasePositionOperator, FixedSzSumOperator, FixedSzSumOperatorXYZ,
//   FixedSzSublatticeOperator, FixedSzTransverseOperator,
//   FixedSzTransverseOperatorXYZ, FixedSzExperimentalOperator,
//   FixedSzTransverseExperimentalOperator
//
// Depends on: fixed_sz_operator.h.
// =============================================================================

#include <ed/core/fixed_sz_operator.h>

// ============================================================================
// Fixed Sz Derived Operator Classes
// ============================================================================

/**
 * Single site operator for fixed Sz sector (S+, S-, Sz, Sx, Sy)
 */
class FixedSzSingleSiteOperator : public FixedSzOperator {
public:
    FixedSzSingleSiteOperator(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op, uint64_t site_j) 
        : FixedSzOperator(num_site, spin_l, n_up) {
        
        if (op < 0 || op > 4) {
            throw std::invalid_argument("Invalid operator type");
        }
        
        if (site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Invalid site index");
        }
        
        if (op <= 2) {
            // S+, S-, Sz
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                if (op == 2) {
                    return {basis, Complex(spin_l * pow(-1, (basis >> site_j) & 1), 0.0)};
                } else {
                    if (((basis >> site_j) & 1) != op) {
                        uint64_t flipped = basis ^ (1ULL << site_j);
                        return {flipped, Complex(1.0, 0.0)};
                    }
                }
                return {basis, Complex(0.0, 0.0)};
            });
        } else {
            // Sx or Sy
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1ULL << site_j);
                if (op == 3) {
                    // Sx = (S+ + S-) / 2
                    return {flipped, Complex(0.5, 0.0)};
                } else {
                    // Sy = (S+ - S-) / (2i)
                    bool is_up = ((basis >> site_j) & 1) == 0;
                    return {flipped, Complex(0.0, is_up ? 0.5 : -0.5)};
                }
            });
        }
    }
};

/**
 * Two-site operator for fixed Sz sector
 */
class FixedSzDoubleSiteOperator : public FixedSzOperator {
public:
    FixedSzDoubleSiteOperator(uint64_t num_site, float spin_l, int64_t n_up, 
                              uint64_t op_i, uint64_t site_i, uint64_t op_j, uint64_t site_j)
        : FixedSzOperator(num_site, spin_l, n_up) {
        
        if (op_i < 0 || op_i > 2 || op_j < 0 || op_j > 2) {
            throw std::invalid_argument("Invalid operator types");
        }
        
        if (site_i < 0 || site_i >= num_site || site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Invalid site indices");
        }
        
        addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
            uint64_t bit_i = (basis >> site_i) & 1;
            uint64_t bit_j = (basis >> site_j) & 1;
            
            Complex factor(1.0, 0.0);
            
            if (op_i == 2 && op_j == 2) {
                return {basis, Complex(spin_l * spin_l * pow(-1, bit_i) * pow(-1, bit_j), 0.0)};
            }
            
            uint64_t new_basis = basis;
            bool valid = true;
            
            if (op_i != 2) {
                if (bit_i != op_i) {
                    new_basis ^= (1ULL << site_i);
                } else {
                    valid = false;
                }
            } else {
                factor *= Complex(spin_l * pow(-1, bit_i), 0.0);
            }
            
            if (valid && op_j != 2) {
                uint64_t new_bit_j = (new_basis >> site_j) & 1;
                if (new_bit_j != op_j) {
                    new_basis ^= (1ULL << site_j);
                } else {
                    valid = false;
                }
            } else if (valid) {
                uint64_t new_bit_j = (new_basis >> site_j) & 1;
                factor *= Complex(spin_l * pow(-1, new_bit_j), 0.0);
            }
            
            if (valid) return {new_basis, factor};
            return {basis, Complex(0.0, 0.0)};
        });
    }
};

/**
 * Base class for position-dependent operators in fixed Sz sector
 */
class FixedSzBasePositionOperator : public FixedSzOperator {
protected:
    std::vector<std::vector<double>> readPositionsFromFile(const std::string& filename, uint64_t expected_sites) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open positions file: " + filename);
        }
        
        std::vector<std::vector<double>> positions(expected_sites);
        std::string line;
        
        while (std::getline(file, line) && line[0] == '#');
        
        bool process_current = !line.empty() && line[0] != '#';
        
        do {
            if (line.empty() || line[0] == '#') continue;
            
            std::istringstream iss(line);
            uint64_t site_id, matrix_idx, sublattice;
            double x, y, z;
            
            if (iss >> site_id >> matrix_idx >> sublattice >> x >> y >> z) {
                if (site_id >= 0 && site_id < expected_sites) {
                    positions[site_id] = {x, y, z};
                }
            }
        } while (std::getline(file, line));
        
        return positions;
    }
    
    std::vector<Complex> calculatePhaseFactors(const std::vector<double>& Q_vector,
                                                const std::vector<std::vector<double>>& positions,
                                                double normalization) {
        std::vector<Complex> phases(positions.size());
        for (size_t i = 0; i < positions.size(); ++i) {
            double dot_product = 0.0;
            for (size_t d = 0; d < 3; ++d) {
                dot_product += Q_vector[d] * positions[i][d];
            }
            phases[i] = normalization * std::exp(Complex(0.0, dot_product));
        }
        return phases;
    }
    
public:
    FixedSzBasePositionOperator(uint64_t num_site, float spin_l, int64_t n_up) 
        : FixedSzOperator(num_site, spin_l, n_up) {}
};

/**
 * Sum operator for fixed Sz sector: S^α = Σᵢ S^α_i e^(iQ·Rᵢ) / √N
 */
class FixedSzSumOperator : public FixedSzBasePositionOperator {
public:
    FixedSzSumOperator(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op, 
                       const std::vector<double>& Q_vector, const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            // Add to optimized storage
            TransformData tdata;
            tdata.op_type = static_cast<uint8_t>(op);
            tdata.site_index = site;
            tdata.coefficient = phase;
            tdata.is_two_body = false;
            transform_data_.push_back(tdata);
        }
    }
};

/**
 * Sum operator with Cartesian basis for fixed Sz sector (Sx, Sy, Sz)
 */
class FixedSzSumOperatorXYZ : public FixedSzBasePositionOperator {
public:
    FixedSzSumOperatorXYZ(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op,
                          const std::vector<double>& Q_vector, const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            if (op == 0) {
                // Sx = (S+ + S-) / 2
                TransformData tdata_plus, tdata_minus;
                tdata_plus.op_type = 0; // S+
                tdata_plus.site_index = site;
                tdata_plus.coefficient = phase * Complex(0.5, 0.0);
                tdata_plus.is_two_body = false;
                transform_data_.push_back(tdata_plus);
                
                tdata_minus.op_type = 1; // S-
                tdata_minus.site_index = site;
                tdata_minus.coefficient = phase * Complex(0.5, 0.0);
                tdata_minus.is_two_body = false;
                transform_data_.push_back(tdata_minus);
            } else if (op == 1) {
                // Sy = (S+ - S-) / (2i) = -i(S+ - S-)/2
                TransformData tdata_plus, tdata_minus;
                tdata_plus.op_type = 0; // S+
                tdata_plus.site_index = site;
                tdata_plus.coefficient = phase * Complex(0.0, -0.5);
                tdata_plus.is_two_body = false;
                transform_data_.push_back(tdata_plus);
                
                tdata_minus.op_type = 1; // S-
                tdata_minus.site_index = site;
                tdata_minus.coefficient = phase * Complex(0.0, 0.5);
                tdata_minus.is_two_body = false;
                transform_data_.push_back(tdata_minus);
            } else if (op == 2) {
                // Sz - note: apply() already multiplies by spin_l_, so coefficient is just phase
                TransformData tdata;
                tdata.op_type = 2; // Sz
                tdata.site_index = site;
                tdata.coefficient = phase;  // No spin_l factor here - apply() handles it
                tdata.is_two_body = false;
                transform_data_.push_back(tdata);
            }
        }
    }
};

/**
 * Sublattice operator for fixed Sz sector: sum over specific sublattice sites
 */
class FixedSzSublatticeOperator : public FixedSzBasePositionOperator {
public:
    FixedSzSublatticeOperator(uint64_t sublattice_idx, uint64_t unit_cell_size, uint64_t num_site, 
                              float spin_l, int64_t n_up, uint64_t op, 
                              const std::vector<double>& Q_vector, const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = sublattice_idx; site < num_site; site += unit_cell_size) {
            Complex phase = phases[site];
            
            // Add to optimized storage
            TransformData tdata;
            tdata.op_type = static_cast<uint8_t>(op);
            tdata.site_index = site;
            tdata.coefficient = phase;
            tdata.is_two_body = false;
            transform_data_.push_back(tdata);
        }
    }
};

/**
 * Transverse operator for fixed Sz sector with sublattice-dependent weighting
 */
class FixedSzTransverseOperator : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseOperator(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op,
                              const std::vector<double>& Q_vector, const std::vector<double>& v,
                              const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors with sublattice weighting
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            // Add to optimized storage
            TransformData tdata;
            tdata.op_type = static_cast<uint8_t>(op);
            tdata.site_index = site;
            tdata.coefficient = phase;
            tdata.is_two_body = false;
            transform_data_.push_back(tdata);
        }
    }
};

/**
 * Transverse operator with Cartesian basis for fixed Sz sector
 */
class FixedSzTransverseOperatorXYZ : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseOperatorXYZ(uint64_t num_site, float spin_l, int64_t n_up, uint64_t op,
                                 const std::vector<double>& Q_vector, const std::vector<double>& v,
                                 const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors with sublattice weighting
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            if (op == 0) {
                // Sx = (S+ + S-) / 2
                TransformData tdata_plus, tdata_minus;
                tdata_plus.op_type = 0; // S+
                tdata_plus.site_index = site;
                tdata_plus.coefficient = phase * Complex(0.5, 0.0);
                tdata_plus.is_two_body = false;
                transform_data_.push_back(tdata_plus);
                
                tdata_minus.op_type = 1; // S-
                tdata_minus.site_index = site;
                tdata_minus.coefficient = phase * Complex(0.5, 0.0);
                tdata_minus.is_two_body = false;
                transform_data_.push_back(tdata_minus);
            } else if (op == 1) {
                // Sy = (S+ - S-) / (2i) = -i(S+ - S-)/2
                TransformData tdata_plus, tdata_minus;
                tdata_plus.op_type = 0; // S+
                tdata_plus.site_index = site;
                tdata_plus.coefficient = phase * Complex(0.0, -0.5);
                tdata_plus.is_two_body = false;
                transform_data_.push_back(tdata_plus);
                
                tdata_minus.op_type = 1; // S-
                tdata_minus.site_index = site;
                tdata_minus.coefficient = phase * Complex(0.0, 0.5);
                tdata_minus.is_two_body = false;
                transform_data_.push_back(tdata_minus);
            } else if (op == 2) {
                // Sz - note: apply() already multiplies by spin_l_, so coefficient is just phase
                TransformData tdata;
                tdata.op_type = 2; // Sz
                tdata.site_index = site;
                tdata.coefficient = phase;  // No spin_l factor here - apply() handles it
                tdata.is_two_body = false;
                transform_data_.push_back(tdata);
            }
        }
    }
};

/**
 * Experimental operator for fixed Sz sector: cos(θ)Sz + sin(θ)Sx
 */
class FixedSzExperimentalOperator : public FixedSzBasePositionOperator {
public:
    FixedSzExperimentalOperator(uint64_t num_site, float spin_l, int64_t n_up, double theta,
                                const std::vector<double>& Q_vector, const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            // Sz contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                return {basis, phase * Complex(cos_theta * spin_l * pow(-1, (basis >> site) & 1), 0.0)};
            });
            
            // Sx contribution = (S+ + S-) / 2
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1ULL << site);
                return {flipped, phase * Complex(sin_theta * 0.5, 0.0)};
            });
        }
    }
};

/**
 * Transverse experimental operator for fixed Sz sector with sublattice weighting
 */
class FixedSzTransverseExperimentalOperator : public FixedSzBasePositionOperator {
public:
    FixedSzTransverseExperimentalOperator(uint64_t num_site, float spin_l, int64_t n_up, double theta,
                                          const std::vector<double>& Q_vector, const std::vector<double>& v,
                                          const std::string& positions_file)
        : FixedSzBasePositionOperator(num_site, spin_l, n_up) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        
        // Calculate transverse phase factors
        std::vector<Complex> phases(num_site);
        const std::vector<std::vector<double>> z_mu = {
            {-1/std::sqrt(3), -1/std::sqrt(3), -1/std::sqrt(3)},
            {-1/std::sqrt(3), 1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), -1/std::sqrt(3), 1/std::sqrt(3)},
            {1/std::sqrt(3), 1/std::sqrt(3), -1/std::sqrt(3)}
        };
        
        for (uint64_t i = 0; i < num_site; ++i) {
            double Q_dot_R = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                Q_dot_R += Q_vector[d] * positions[i][d];
            }
            
            uint64_t sublattice = i % 4;
            double v_dot_z = 0.0;
            for (uint64_t d = 0; d < 3; ++d) {
                v_dot_z += v[d] * z_mu[sublattice][d];
            }
            
            phases[i] = (1.0 / std::sqrt(num_site)) * v_dot_z * std::exp(Complex(0.0, Q_dot_R));
        }
        
        double cos_theta = std::cos(theta);
        double sin_theta = std::sin(theta);
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            // Sz contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                return {basis, phase * Complex(cos_theta * spin_l * pow(-1, (basis >> site) & 1), 0.0)};
            });
            
            // Sx contribution
            addTransform([=](uint64_t basis) -> std::pair<int, Complex> {
                uint64_t flipped = basis ^ (1ULL << site);
                return {flipped, phase * Complex(sin_theta * 0.5, 0.0)};
            });
        }
    }
};
