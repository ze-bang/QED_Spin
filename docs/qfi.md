# Thermal quantum Fisher information

`qed.qfi(H, O, T)` evaluates the QFI of Hermitian generators in the full canonical
ensemble. It uses the symmetry-resolved, energy-resolved finite-temperature
Lanczos method and accumulates its pole weights directly. Each pole has its
own transition energy, `E_final - E_initial`. There is no mean-energy shift,
frequency grid, spectral cutoff, or artificial broadening in this calculation.

```python
import qed

lat = qed.input.lattice.chain(8, True)
b = qed.input.HamiltonianBuilder(8)
b.xxz(lat.nn_pairs(), 0.7, 1.0)
H = b.to_operator()
O = qed.Operator(8)
for i in range(8):
    O = O + qed.Operator.product(8, "z", [i], (-1)**i / 8**0.5)
r = qed.qfi(H, O, [0.1, 0.5, 1.0], samples=40, krylov=160, seed=42)
```

The result contains `F`, two independent forms of the same spectral functional
(`F_positive` and `F_squared`), their `balance_error`, and the equal-time spectral
weight. `E`, `C`, and `lnZ` come from the same source Lanczos runs. Operators or
operator families may be passed together; the last array axis is temperature.
Generator normalisation is explicit: dividing an extensive generator by
`sqrt(N)` returns its QFI per site. A complex Fourier mode must be split into
Hermitian cosine and sine components. Their QFI sum is a specified sum of
generator QFIs, not the QFI of a non-Hermitian operator.

For a Hermitian generator with thermal spectral measure `S(w)`, the three
functionals are

\[
F_Q=4\int_{-\infty}^{\infty} S(\omega)\tanh(\omega/2T)\,d\omega
=4\int_0^{\infty}S(\omega)\tanh(\omega/2T)(1-e^{-\omega/T})\,d\omega
=4\int_{-\infty}^{\infty}S(\omega)\tanh^2(\omega/2T)\,d\omega.
\]

They coincide for the canonical Lehmann spectrum. Their difference measures
one aspect of finite-sampling/Krylov error; agreement alone does not certify
accuracy. Raw estimates, including negative stochastic estimates, are retained.
The implementation uses full reorthogonalisation of both Krylov bases. Increase
`krylov` and the number of independent seeds until the reported observables
converge, compare to exact small-system results, and check that a conserved
generator returns zero. At low temperature, compare against certified low-energy
results, including degenerate manifolds. A small Boltzmann factor at one cutoff
does not by itself bound all omitted states or sectors.

Independent runs estimate numerator and partition-function traces. Combine them
as `sum(Z_r * F_r) / sum(Z_r)` with `Z_r = exp(lnZ_r)` evaluated in log space;
use a ratio jackknife or bootstrap for uncertainty. Averaging normalised curves
can introduce bias where the partition-function fluctuations are large.

`qed.thermal(method="mtpq")` provides an independent thermodynamics comparison
using canonical reconstruction from TPQ moments. The QFI dynamics estimator is
FTLM; it is not the single-energy continued fraction of a broad TPQ state.
Finite-temperature results should name these methods separately.

Method references:

- Jaklič and Prelovšek, [Phys. Rev. B 49, 5065 (1994)](https://doi.org/10.1103/PhysRevB.49.5065).
- Sugiura and Shimizu, [Phys. Rev. Lett. 111, 010401 (2013)](https://arxiv.org/abs/1302.3138).
- Schnack, Richter, and Steinigeweg, [Phys. Rev. Research 2, 013186 (2020)](https://doi.org/10.1103/PhysRevResearch.2.013186).
