# C++ quickstart

This page walks through a 4-site Heisenberg chain end-to-end, using only the
public C++ API. The complete example fits in a single file.

```cpp
#include <ed/core/construct_ham.h>
#include <ed/solvers/diagonalization.h>

#include <complex>
#include <iostream>
#include <vector>

int main() {
    // 4-site spin-1/2 chain with periodic boundaries.
    constexpr uint64_t N = 4;
    constexpr float    S = 0.5f;
    Operator H(N, S);

    // J = 1 Heisenberg coupling: J · S_i · S_{i+1}
    //   = J/2 (S+_i S-_{i+1} + S-_i S+_{i+1}) + J · Sz_i Sz_{i+1}
    constexpr double J = 1.0;
    for (uint64_t i = 0; i < N; ++i) {
        const uint64_t j = (i + 1) % N;
        H.add_two_body(/*op_i=*/0, /*site_i=*/i,
                       /*op_j=*/1, /*site_j=*/j, std::complex<double>(0.5 * J, 0.0));
        H.add_two_body(/*op_i=*/1, /*site_i=*/i,
                       /*op_j=*/0, /*site_j=*/j, std::complex<double>(0.5 * J, 0.0));
        H.add_two_body(/*op_i=*/2, /*site_i=*/i,
                       /*op_j=*/2, /*site_j=*/j, std::complex<double>(J,       0.0));
    }

    // Full diagonalization (LAPACK).
    auto result = ed::full_diagonalization(H);
    std::cout << "Ground-state energy: " << result.eigenvalues[0] << "\n";
    std::cout << "First excited:       " << result.eigenvalues[1] << "\n";
}
```

Build it against the installed package:

```cmake
find_package(ED CONFIG REQUIRED)
add_executable(heisenberg_demo heisenberg_demo.cpp)
target_link_libraries(heisenberg_demo PRIVATE ED::ed_solvers_cpu)
```

## Where to go next

- **Lanczos / block-Lanczos** — see {doc}`/api/cpp` (`include/ed/solvers/`).
- **Finite-temperature methods** — see the FTLM / LTLM API reference.
- **DSSF / SSSF** — see `ed::dssf::OperatorSpec` /
  `ed::dssf::build_observable_pairs` for the canonical observable
  assembly entry point.
- **Symmetry-resolved sectors** — see `FixedSzOperator`,
  translation operators, and the streaming-symmetry helper.
