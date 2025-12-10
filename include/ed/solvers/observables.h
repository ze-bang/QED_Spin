#ifndef OBSERVABLES_H
#define OBSERVABLES_H

#include <iostream>
#include <complex>
#include <vector>
#include <functional>
#include <cmath>
#include <ed/core/blas_lapack_wrapper.h>
#include <ed/core/construct_ham.h>
#include <algorithm>

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//// Observables

// Type definition for complex vector and matrix operations
using Complex = std::complex<double>;
using ComplexVector = std::vector<Complex>;


// Calculate thermodynamic quantities directly from eigenvalues
ThermodynamicData calculate_thermodynamics_from_spectrum(
    const std::vector<double>& eigenvalues,
    double T_min = 0.01,        // Minimum temperature
    double T_max = 10.0,        // Maximum temperature
    uint64_t num_points = 100   // Number of temperature points
);

#endif // OBSERVABLES_H
