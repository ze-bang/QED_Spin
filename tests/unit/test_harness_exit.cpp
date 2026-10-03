// =============================================================================
// tests/unit/test_harness_exit.cpp -- the unit tests' exit code (tests/common/catch2_main.cpp).
//
// Not discovered test by test: CMakeLists.txt runs each case alone and checks the process exit code
// -- four failed assertions must exit 1 (Catch2's own main exits 4, its all-skipped code, which ctest
// took for a skip), a case that only skips must exit 77.
// =============================================================================
#include <catch2/catch_test_macros.hpp>

TEST_CASE("four failed assertions", "[harness_four]") {
    for (int i = 0; i < 4; ++i) CHECK(i < 0);
}

TEST_CASE("a skip alone", "[harness_skip]") { SKIP("skipped on purpose"); }
