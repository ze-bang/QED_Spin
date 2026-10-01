# AUDIT-ID: C03-bindings-05
# DEVICE: cpu
# SECONDS: 30
"""Claim: a None inside an operator list (qed.expect(H, None), thermal observables=[None]) reaches
the engine as a null Operator* (pybind11 converts None to nullptr) and segfaults the interpreter
instead of raising TypeError."""
import subprocess
import sys

PRE = ("import qed\nN=6\nb=qed.input.HamiltonianBuilder(N)\n"
       "b.heisenberg([(i,(i+1)%N) for i in range(N)],J=1.0)\nH=b.to_operator()\n"
       "sym=qed.Symmetry(spatial=None)\n")
CASES = {
    "expect(H, None)": "qed.expect(H, None, sym=sym)",
    "expect(H, [Sz0, None])": "O=qed.Operator(N,0.5); O.add_one_body(qed.OP_SZ,0,1.0); qed.expect(H, [O, None], sym=sym)",
    "thermal(observables=[None])": "qed.thermal(H,[1.0],method='exact',sym=sym,observables=[None])",
}
res = {}
for name, call in CASES.items():
    code = PRE + "try:\n    " + call + "\n    print('RETURNED')\nexcept Exception as e:\n    print('RAISED', type(e).__name__, e)\n"
    try:
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
        out = (p.stdout.strip().splitlines() or [""])[-1]
        res[name] = (p.returncode, out[:120])
    except subprocess.TimeoutExpired:
        res[name] = ("timeout", "")
    print(name, "->", res[name])
crashed = [n for n, (rc, _) in res.items() if isinstance(rc, int) and rc < 0]
if crashed:
    print("REPRO: CONFIRMED killed by signal: " + "; ".join(f"{n} rc={res[n][0]}" for n in crashed))
elif all(o.startswith("RAISED") for _, o in res.values()):
    print("REPRO: NOT_REPRODUCED every case raised a Python exception")
else:
    print(f"REPRO: INCONCLUSIVE {res}")
