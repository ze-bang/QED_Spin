#pragma once
// =============================================================================
// include/ed/core/operator_types.h
//
// Concrete full-basis operator subclasses:
//   SingleSiteOperator, DoubleSiteOperator, BasePositionOperator,
//   SumOperator, SumOperatorXYZ, SublatticeOperator,
//   TransverseOperator, TransverseOperatorXYZ,
//   ExperimentalOperator, TransverseExperimentalOperator
//
// Depends on: operator.h.
// =============================================================================

#include <ed/core/operator.h>

// ============================================================================
// Derived Operator Classes
// ============================================================================

/**
 * Single site operator (S+, S-, Sz, Sx, Sy)
 */
class SingleSiteOperator : public Operator {
public:
    SingleSiteOperator(uint64_t num_site, float spin_l, uint64_t op, uint64_t site_j) 
        : Operator(num_site, spin_l) {
        
        if (op < 0 || op > 4) {
            throw std::invalid_argument("Invalid operator type");
        }
        
        if (site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Invalid site index");
        }
        
        if (op <= 2) {
            // S+, S-, Sz - add to optimized storage
            TransformData tdata;
            tdata.op_type = static_cast<uint8_t>(op);
            tdata.site_index = site_j;
            tdata.coefficient = Complex(1.0, 0.0);
            tdata.is_two_body = false;
            transform_data_.push_back(tdata);
        } else {
            // Sx or Sy - complex operators handled via multiple transforms
            // This needs special handling - keeping as placeholder for now
            throw std::runtime_error("Sx/Sy operators need special handling in optimized path");
        }
    }
};

/**
 * Two-site operator for interactions
 */
class DoubleSiteOperator : public Operator {
public:
    DoubleSiteOperator() : Operator(0, 0.0) {}
    
    DoubleSiteOperator(uint64_t num_site, float spin_l, uint64_t op_i, uint64_t site_i, uint64_t op_j, uint64_t site_j)
        : Operator(num_site, spin_l) {
        
        if (op_i < 0 || op_i > 2 || op_j < 0 || op_j > 2) {
            throw std::invalid_argument("Invalid operator types");
        }
        
        if (site_i < 0 || site_i >= num_site || site_j < 0 || site_j >= num_site) {
            throw std::invalid_argument("Invalid site indices");
        }
        
        // Add to optimized storage
        TransformData tdata;
        tdata.op_type = static_cast<uint8_t>(op_i);
        tdata.site_index = site_i;
        tdata.op_type_2 = static_cast<uint8_t>(op_j);
        tdata.site_index_2 = site_j;
        tdata.coefficient = Complex(1.0, 0.0);
        tdata.is_two_body = true;
        transform_data_.push_back(tdata);
    }
};

/**
 * Base class for position-dependent operators
 */
class BasePositionOperator : public Operator {
protected:
    std::vector<std::vector<double>> readPositionsFromFile(const std::string& filename, uint64_t expected_sites) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open positions file: " + filename);
        }

        // The canonical positions.dat format (written by write_positions_file /
        // HamiltonianBuilder::write_directory) is one line per site:
        //   x y z
        // where the site index equals the (0-based) line number.
        // Lines beginning with '#' are treated as comments and skipped.
        std::vector<std::vector<double>> positions(expected_sites, std::vector<double>(3, 0.0));
        std::string line;
        uint64_t site_id = 0;

        while (std::getline(file, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream iss(line);
            double x, y, z;
            if (iss >> x >> y >> z) {
                if (site_id < expected_sites) {
                    positions[site_id] = {x, y, z};
                }
                ++site_id;
            }
        }

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
    BasePositionOperator(uint64_t num_site, float spin_l) : Operator(num_site, spin_l) {}
};

/**
 * Sum operator: S^α = Σᵢ S^α_i e^(iQ·Rᵢ) / √N
 */
class SumOperator : public BasePositionOperator {
public:
    SumOperator(uint64_t num_site, float spin_l, uint64_t op, const std::vector<double>& Q_vector,
                const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
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
 * Sum operator with Cartesian basis (Sx, Sy, Sz)
 */
class SumOperatorXYZ : public BasePositionOperator {
public:
    SumOperatorXYZ(uint64_t num_site, float spin_l, uint64_t op, const std::vector<double>& Q_vector,
                   const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
        auto positions = readPositionsFromFile(positions_file, num_site);
        auto phases = calculatePhaseFactors(Q_vector, positions, 1.0 / std::sqrt(num_site));
        
        for (uint64_t site = 0; site < num_site; ++site) {
            Complex phase = phases[site];
            
            if (op == 0) {
                // Sx = (S+ + S-) / 2
                // Need to add both S+ and S- with coefficient 0.5
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
                // S+: coefficient = -i/2 * phase = phase * (0, -0.5)
                // S-: coefficient = +i/2 * phase = phase * (0, +0.5)
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
 * Sublattice operator: sum over specific sublattice sites
 */
class SublatticeOperator : public BasePositionOperator {
public:
    SublatticeOperator(uint64_t sublattice_idx, uint64_t unit_cell_size, uint64_t num_site, float spin_l,
                      uint64_t op, const std::vector<double>& Q_vector,
                      const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
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
 * Transverse operator with sublattice-dependent weighting
 */
class TransverseOperator : public BasePositionOperator {
public:
    TransverseOperator(uint64_t num_site, float spin_l, uint64_t op,
                      const std::vector<double>& Q_vector,
                      const std::vector<double>& v,
                      const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
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
 * Transverse operator with Cartesian basis
 */
class TransverseOperatorXYZ : public BasePositionOperator {
public:
    TransverseOperatorXYZ(uint64_t num_site, float spin_l, uint64_t op,
                         const std::vector<double>& Q_vector,
                         const std::vector<double>& v,
                         const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
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
 * Experimental operator: cos(θ)Sz + sin(θ)Sx
 */
class ExperimentalOperator : public BasePositionOperator {
public:
    ExperimentalOperator(uint64_t num_site, float spin_l, double theta,
                        const std::vector<double>& Q_vector,
                        const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
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
 * Transverse experimental operator with sublattice weighting
 */
class TransverseExperimentalOperator : public BasePositionOperator {
public:
    TransverseExperimentalOperator(uint64_t num_site, float spin_l, double theta,
                                  const std::vector<double>& Q_vector,
                                  const std::vector<double>& v,
                                  const std::string& positions_file)
        : BasePositionOperator(num_site, spin_l) {
        
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

