# AUDIT-ID: C03-bindings-09
# DEVICE: cpu
# SECONDS: 30
"""Claim: EigResult.vectors() catches every ValueError as 'no component in this Sz sector', so real
refusals become a silent empty list: at N >= 35 the default basis='full' hits expand()'s
'N <= 34' refusal and returns [] (while basis='sz' works), and an impossible n_up returns []."""
import qed


def ring(N):
    b = qed.input.HamiltonianBuilder(N)
    b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
    return b.to_operator()


try:
    r36 = qed.eigs(ring(36), 1, sym=qed.Symmetry(spatial=None, sz=1), vectors=True)
    full36, err36 = None, None
    try:
        full36 = r36.vectors()
    except Exception as ex:
        err36 = f"{type(ex).__name__}: {ex}"
    sz36 = r36.vectors(basis="sz", n_up=1)
    r12 = qed.eigs(ring(12), 1, sym=qed.Symmetry(spatial=None), vectors=True)
    bogus, errb = None, None
    try:
        bogus = r12.vectors(basis="sz", n_up=99)
    except Exception as ex:
        errb = f"{type(ex).__name__}: {ex}"
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE setup raised {type(ex).__name__}: {ex}")
    raise SystemExit(0)
print(f"N=36 E0={r36.energies[0]:.10f}; vectors() -> {None if full36 is None else len(full36)} (err {err36}); "
      f"vectors(basis='sz', n_up=1) -> {len(sz36)}")
print(f"N=12 vectors(basis='sz', n_up=99) -> {None if bogus is None else len(bogus)} (err {errb})")
silent36 = full36 is not None and len(full36) == 0 and len(sz36) > 0
silentb = bogus is not None and len(bogus) == 0
if silent36 or silentb:
    print(f"REPRO: CONFIRMED N=36 full-basis vectors() silently [] ({silent36}), sz basis gives {len(sz36)}; "
          f"n_up=99 silently [] ({silentb})")
else:
    print(f"REPRO: NOT_REPRODUCED full36={err36 or len(full36)} bogus={errb or len(bogus)}")
