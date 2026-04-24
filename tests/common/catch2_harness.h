// =============================================================================
// tests/common/catch2_harness.h
//
// Catch2 v3 wrapper around the existing fixture builders in test_harness.h.
//
// We deliberately keep the construct_*, reference_from_*, random_unit_vector,
// and make_scratch_dir helpers in test_harness.h (they are pure functions
// and have no Catch2 dependency). This header just pulls them in plus the
// Catch2 macros and provides ED-specific MATCHER helpers.
//
// P1.8 / audit Q12.
// =============================================================================
#pragma once

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "common/test_harness.h"

namespace ed_tests {

// Compare two sorted-or-sortable eigenvalue vectors element-wise on the
// overlap length, using `tol` as an absolute tolerance. Mirrors the
// pre-Catch2 `check_eigs_close` API but reports via Catch2's assertion
// machinery.
inline void require_eigs_close(std::vector<double> got,
                               std::vector<double> want,
                               size_t n,
                               double tol,
                               const std::string& label = "spectrum") {
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());

    REQUIRE(got.size() >= n);
    REQUIRE(want.size() >= n);

    double max_err = 0.0;
    size_t worst = 0;
    for (size_t i = 0; i < n; ++i) {
        double e = std::abs(got[i] - want[i]);
        if (e > max_err) { max_err = e; worst = i; }
    }
    INFO(label << ": worst index=" << worst
               << " got=" << got[worst]
               << " want=" << want[worst]
               << " |Δ|=" << max_err
               << " tol=" << tol);
    REQUIRE(max_err <= tol);
}

} // namespace ed_tests
