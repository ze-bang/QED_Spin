// =============================================================================
// test_lanczos_basis_buffer
//
// Unit tests for the in-memory Lanczos basis-vector buffer
// (`ed/io/lanczos_basis_buffer.h`). Covers:
//   * register / has / release,
//   * append with dimension checking,
//   * set_basis_vector (overwrite), truncate_basis_buffer,
//   * get_basis_vector (copy) and get_basis_vector_ptr (zero-copy),
//   * total_basis_buffer_bytes accounting,
//   * behavior with multiple distinct keys.
//
// The force-disk path (ED_LANCZOS_DISK=1) is cached per-process by design, so
// it is covered from the outside via CTest fixture variants.
// =============================================================================

#include "common/test_harness.h"

#include <ed/io/lanczos_basis_buffer.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

using namespace ed_tests;
using lanczos_io::ComplexVector;

static ComplexVector make_vec(uint64_t N, double base) {
    ComplexVector v(N);
    for (uint64_t i = 0; i < N; ++i) {
        v[i] = Complex(base + 0.1 * static_cast<double>(i),
                       -base + 0.01 * static_cast<double>(i));
    }
    return v;
}

int main() {
    TestContext ctx("test_lanczos_basis_buffer");

    // If ED_LANCZOS_DISK is forcing disk mode we can't test the in-memory
    // buffer; skip gracefully (exit code 0).
    if (lanczos_io::force_disk_storage()) {
        std::cout << "[skip] ED_LANCZOS_DISK=1 — in-memory path disabled"
                  << std::endl;
        return 0;
    }

    const std::string key = "/tmp/ed_test_buffer_key_A";
    const std::string key2 = "/tmp/ed_test_buffer_key_B";
    const uint64_t N = 8;

    // register/has/release lifecycle
    check(ctx, !lanczos_io::has_basis_buffer(key), "buffer not yet registered");
    lanczos_io::register_basis_buffer(key, N, /*reserve=*/16);
    check(ctx, lanczos_io::has_basis_buffer(key), "register creates buffer");
    check(ctx, lanczos_io::basis_buffer_size(key) == 0,
          "fresh buffer has size 0");

    // append + read-back
    auto v0 = make_vec(N, 1.0);
    auto v1 = make_vec(N, 2.0);
    auto v2 = make_vec(N, 3.0);
    check(ctx, lanczos_io::append_basis_vector(key, v0), "append v0");
    check(ctx, lanczos_io::append_basis_vector(key, v1), "append v1");
    check(ctx, lanczos_io::append_basis_vector(key, v2), "append v2");
    check(ctx, lanczos_io::basis_buffer_size(key) == 3,
          "buffer size after 3 appends");

    // dimension mismatch should be rejected
    ComplexVector short_vec(N - 1, Complex(0, 0));
    check(ctx, !lanczos_io::append_basis_vector(key, short_vec),
          "dimension mismatch rejected");

    // copy-read
    ComplexVector read_v1;
    check(ctx, lanczos_io::get_basis_vector(key, 1, read_v1), "get v1");
    check(ctx, l2_diff(read_v1, v1) < 1e-15, "get returns exact copy",
          "||read - v1|| = " + std::to_string(l2_diff(read_v1, v1)));

    // zero-copy pointer
    const Complex* p2 = lanczos_io::get_basis_vector_ptr(key, 2);
    check(ctx, p2 != nullptr, "get_ptr returns non-null");
    if (p2) {
        double err = 0.0;
        for (uint64_t i = 0; i < N; ++i) err += std::norm(p2[i] - v2[i]);
        check(ctx, std::sqrt(err) < 1e-15,
              "get_ptr returns exact data");
    }
    check(ctx, lanczos_io::get_basis_vector_ptr(key, 99) == nullptr,
          "get_ptr OOB returns null");

    // overwrite via set_basis_vector
    auto v1b = make_vec(N, 5.0);
    check(ctx, lanczos_io::set_basis_vector(key, 1, v1b), "set v1 in place");
    ComplexVector read_v1b;
    lanczos_io::get_basis_vector(key, 1, read_v1b);
    check(ctx, l2_diff(read_v1b, v1b) < 1e-15,
          "set_basis_vector persists overwrite");
    check(ctx, !lanczos_io::set_basis_vector(key, 99, v1b),
          "set OOB fails cleanly");

    // truncate
    lanczos_io::truncate_basis_buffer(key, 2);
    check(ctx, lanczos_io::basis_buffer_size(key) == 2,
          "truncate reduces size");
    check(ctx, lanczos_io::get_basis_vector_ptr(key, 2) == nullptr,
          "truncated index is gone");
    // truncate with new_size >= size is a no-op
    lanczos_io::truncate_basis_buffer(key, 10);
    check(ctx, lanczos_io::basis_buffer_size(key) == 2,
          "truncate no-op when new_size >= current");

    // multi-key isolation
    lanczos_io::register_basis_buffer(key2, N, /*reserve=*/4);
    lanczos_io::append_basis_vector(key2, make_vec(N, 7.5));
    check(ctx, lanczos_io::basis_buffer_size(key) == 2 &&
                lanczos_io::basis_buffer_size(key2) == 1,
          "distinct keys have independent storage");

    // accounting (non-zero and monotone with size)
    uint64_t before = lanczos_io::total_basis_buffer_bytes();
    lanczos_io::append_basis_vector(key2, make_vec(N, 8.5));
    uint64_t after = lanczos_io::total_basis_buffer_bytes();
    check(ctx, after > before,
          "total_basis_buffer_bytes grows after append",
          "before=" + std::to_string(before) +
              " after=" + std::to_string(after));

    // release
    lanczos_io::release_basis_buffer(key);
    lanczos_io::release_basis_buffer(key2);
    check(ctx, !lanczos_io::has_basis_buffer(key), "release drops buffer");
    check(ctx, !lanczos_io::has_basis_buffer(key2), "release drops buffer");
    check(ctx, lanczos_io::basis_buffer_size(key) == 0,
          "size of released buffer is 0");

    // operations on non-existent keys are benign
    check(ctx, !lanczos_io::append_basis_vector("no/such/key",
                                                make_vec(N, 0.0)),
          "append to non-existent key returns false");
    check(ctx, !lanczos_io::set_basis_vector("no/such/key", 0,
                                             make_vec(N, 0.0)),
          "set on non-existent key returns false");
    check(ctx, lanczos_io::get_basis_vector_ptr("no/such/key", 0) == nullptr,
          "get_ptr on non-existent key returns null");
    lanczos_io::truncate_basis_buffer("no/such/key", 0); // must not crash
    lanczos_io::release_basis_buffer("no/such/key");    // must not crash

    return ctx.summary_exit_code();
}
