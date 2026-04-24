// =============================================================================
// test_lanczos_basis_buffer (Catch2 v3, P1.8 / audit Q12)
//
// Unit tests for the in-memory Lanczos basis-vector buffer
// (`ed/io/lanczos_basis_buffer.h`).
//
// The force-disk path (ED_LANCZOS_DISK=1) is cached per-process by design, so
// it is covered from the outside via CTest fixture variants.
// =============================================================================

#include "common/catch2_harness.h"

#include <ed/io/lanczos_basis_buffer.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

using namespace ed_tests;
using lanczos_io::ComplexVector;

namespace {

ComplexVector make_vec(uint64_t N, double base) {
    ComplexVector v(N);
    for (uint64_t i = 0; i < N; ++i) {
        v[i] = Complex(base + 0.1 * static_cast<double>(i),
                       -base + 0.01 * static_cast<double>(i));
    }
    return v;
}

} // namespace

TEST_CASE("lanczos basis buffer in-memory lifecycle",
          "[lanczos_basis_buffer][in_memory]") {
    if (lanczos_io::force_disk_storage()) {
        WARN("ED_LANCZOS_DISK=1 -- in-memory path disabled, skipping");
        return;
    }

    const std::string key  = "/tmp/ed_test_buffer_key_A";
    const std::string key2 = "/tmp/ed_test_buffer_key_B";
    const uint64_t N = 8;

    SECTION("register/has/release lifecycle") {
        REQUIRE_FALSE(lanczos_io::has_basis_buffer(key));
        lanczos_io::register_basis_buffer(key, N, /*reserve=*/16);
        REQUIRE(lanczos_io::has_basis_buffer(key));
        REQUIRE(lanczos_io::basis_buffer_size(key) == 0);
        lanczos_io::release_basis_buffer(key);
        REQUIRE_FALSE(lanczos_io::has_basis_buffer(key));
    }

    SECTION("append + read-back + dimension check") {
        lanczos_io::register_basis_buffer(key, N, /*reserve=*/16);
        auto v0 = make_vec(N, 1.0);
        auto v1 = make_vec(N, 2.0);
        auto v2 = make_vec(N, 3.0);
        REQUIRE(lanczos_io::append_basis_vector(key, v0));
        REQUIRE(lanczos_io::append_basis_vector(key, v1));
        REQUIRE(lanczos_io::append_basis_vector(key, v2));
        REQUIRE(lanczos_io::basis_buffer_size(key) == 3);

        // Dimension mismatch should be rejected.
        ComplexVector short_vec(N - 1, Complex(0, 0));
        REQUIRE_FALSE(lanczos_io::append_basis_vector(key, short_vec));

        ComplexVector read_v1;
        REQUIRE(lanczos_io::get_basis_vector(key, 1, read_v1));
        REQUIRE(l2_diff(read_v1, v1) < 1e-15);

        const Complex* p2 = lanczos_io::get_basis_vector_ptr(key, 2);
        REQUIRE(p2 != nullptr);
        double err = 0.0;
        for (uint64_t i = 0; i < N; ++i) err += std::norm(p2[i] - v2[i]);
        REQUIRE(std::sqrt(err) < 1e-15);
        REQUIRE(lanczos_io::get_basis_vector_ptr(key, 99) == nullptr);

        lanczos_io::release_basis_buffer(key);
    }

    SECTION("set_basis_vector overwrites in place") {
        lanczos_io::register_basis_buffer(key, N, /*reserve=*/4);
        lanczos_io::append_basis_vector(key, make_vec(N, 1.0));
        lanczos_io::append_basis_vector(key, make_vec(N, 2.0));
        auto v1b = make_vec(N, 5.0);
        REQUIRE(lanczos_io::set_basis_vector(key, 1, v1b));
        ComplexVector read_v1b;
        lanczos_io::get_basis_vector(key, 1, read_v1b);
        REQUIRE(l2_diff(read_v1b, v1b) < 1e-15);
        REQUIRE_FALSE(lanczos_io::set_basis_vector(key, 99, v1b));
        lanczos_io::release_basis_buffer(key);
    }

    SECTION("truncate semantics") {
        lanczos_io::register_basis_buffer(key, N, /*reserve=*/4);
        for (int i = 0; i < 3; ++i)
            lanczos_io::append_basis_vector(key, make_vec(N, double(i)));
        lanczos_io::truncate_basis_buffer(key, 2);
        REQUIRE(lanczos_io::basis_buffer_size(key) == 2);
        REQUIRE(lanczos_io::get_basis_vector_ptr(key, 2) == nullptr);
        // truncate with new_size >= size is a no-op.
        lanczos_io::truncate_basis_buffer(key, 10);
        REQUIRE(lanczos_io::basis_buffer_size(key) == 2);
        lanczos_io::release_basis_buffer(key);
    }

    SECTION("multi-key isolation + accounting") {
        lanczos_io::register_basis_buffer(key,  N, /*reserve=*/4);
        lanczos_io::register_basis_buffer(key2, N, /*reserve=*/4);
        lanczos_io::append_basis_vector(key,  make_vec(N, 1.0));
        lanczos_io::append_basis_vector(key,  make_vec(N, 2.0));
        lanczos_io::append_basis_vector(key2, make_vec(N, 7.5));
        REQUIRE(lanczos_io::basis_buffer_size(key)  == 2);
        REQUIRE(lanczos_io::basis_buffer_size(key2) == 1);

        uint64_t before = lanczos_io::total_basis_buffer_bytes();
        lanczos_io::append_basis_vector(key2, make_vec(N, 8.5));
        uint64_t after = lanczos_io::total_basis_buffer_bytes();
        INFO("before=" << before << " after=" << after);
        REQUIRE(after > before);

        lanczos_io::release_basis_buffer(key);
        lanczos_io::release_basis_buffer(key2);
    }

    SECTION("operations on non-existent keys are benign") {
        REQUIRE_FALSE(lanczos_io::append_basis_vector("no/such/key",
                                                     make_vec(N, 0.0)));
        REQUIRE_FALSE(lanczos_io::set_basis_vector("no/such/key", 0,
                                                   make_vec(N, 0.0)));
        REQUIRE(lanczos_io::get_basis_vector_ptr("no/such/key", 0) == nullptr);
        // These must not crash:
        lanczos_io::truncate_basis_buffer("no/such/key", 0);
        lanczos_io::release_basis_buffer("no/such/key");
    }
}
