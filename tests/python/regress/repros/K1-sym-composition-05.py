# AUDIT-ID: K1-sym-composition-05
# DEVICE: cpu
# SECONDS: 30
"""Claim: total_spin refuses (a) non-SU(2)-invariant observables in expect/thermal although the multiplet
average equals that of the rank-0 part (<Sz_i Sz_j> = <S_i.S_j>/3), (b) exact_states (OFTLM), and
(c) Symmetry(total_spin=S>0, spin_flip='require') fails inside the engine ('subspace is not
flip-invariant') although the same toggle works without total_spin.
Test on the Heisenberg ring N=8 (no spatial symmetry). Also checks the well-defined answer for (a): the
dense S=0 ground state gives <Sz0 Sz1> = <S0.S1>/3, and expect(S0.S1) under total_spin works."""

import signal

import numpy as np
import qed

signal.alarm(100)
N = 8
H = qed.Operator(N)
for i in range(N):
    j = (i + 1) % N
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
SzSz = qed.Operator(N)
SzSz.add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 1, 1.0)
SdS = qed.Operator(N)
SdS.add_two_body(qed.OP_SPLUS, 0, qed.OP_SMINUS, 1, 0.5)
SdS.add_two_body(qed.OP_SMINUS, 0, qed.OP_SPLUS, 1, 0.5)
SdS.add_two_body(qed.OP_SZ, 0, qed.OP_SZ, 1, 1.0)
su2 = qed.Symmetry(spatial=None, total_spin=0)

raised = {}


def attempt(tag, fn):
    try:
        fn()
        raised[tag] = None
        print(f"{tag}: no raise")
    except Exception as ex:
        raised[tag] = f"{type(ex).__name__}: {str(ex)[:120]}"
        print(f"{tag}: raised {raised[tag]}")


attempt("a-expect", lambda: qed.expect(H, [SzSz], 1, sym=su2))
attempt("a-thermal", lambda: qed.thermal(H, [1.0], sym=su2, observables=[SzSz], samples=5, seed=1))
attempt("b-oftlm", lambda: qed.thermal(H, [1.0], sym=su2, exact_states=5, samples=5, seed=1))
attempt("c-require", lambda: qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, total_spin=1, spin_flip="require")))
attempt("c-control", lambda: qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz=3, spin_flip="auto")))

# the well-defined value the refusal withholds
v_sds = complex(np.asarray(qed.expect(H, [SdS], 1, sym=su2).values).ravel()[0]).real
v_zz = complex(np.asarray(qed.expect(H, [SzSz], 1, sym=qed.Symmetry(spatial=None, sz=N // 2)).values).ravel()[0]).real
print(f"<S0.S1>/3 under total_spin=0: {v_sds / 3:.10f}; <Sz0Sz1> at sz=N/2 (S=0 GS): {v_zz:.10f}")
hits = [k for k in ("a-expect", "a-thermal", "b-oftlm", "c-require") if raised.get(k)]
if len(hits) == 4 and raised["c-control"] is None:
    print(
        f"REPRO: CONFIRMED all four refused ({'; '.join(raised[k][:60] for k in hits)}); "
        f"withheld value <Sz0Sz1>={v_zz:.8f} = <S0.S1>/3={v_sds / 3:.8f}"
    )
elif hits:
    print(f"REPRO: CONFIRMED partially: refused {hits}")
else:
    print("REPRO: NOT_REPRODUCED none refused")
