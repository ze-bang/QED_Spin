# AUDIT-ID: C11-thermal-03
# DEVICE: cpu
# SECONDS: 60
"""Claim: qed.thermal under Symmetry(total_spin=S) weights every block by 2S+1 (Z, E describe the
full spin-S multiplets, all Sz = -S..S), yet returns M = S and chi = 0 at every temperature, for
exact, ftlm and mtpq. In that ensemble M = 0 and chi = beta * S(S+1) / (3N).
Reference: independent dense numpy ED of the 8-site Heisenberg ring restricted to the S = 1 subspace."""
import numpy as np
import qed

N, S2 = 8, 1.0
Ts = np.array([0.5, 1.0, 2.0])

# dense reference
sp = np.array([[0, 1], [0, 0]], complex); sm = sp.T.copy(); sz = np.diag([0.5, -0.5]).astype(complex)
I2 = np.eye(2)
def site(op, i):
    m = np.array([[1.0 + 0j]])
    for j in range(N):
        m = np.kron(m, op if j == i else I2)
    return m
SP = [site(sp, i) for i in range(N)]; SM = [site(sm, i) for i in range(N)]; SZ = [site(sz, i) for i in range(N)]
Hd = sum(0.5 * (SP[i] @ SM[(i + 1) % N] + SM[i] @ SP[(i + 1) % N]) + SZ[i] @ SZ[(i + 1) % N] for i in range(N))
Sxyz_p = sum(SP); Sxyz_m = sum(SM); Szt = sum(SZ)
S2op = Sxyz_m @ Sxyz_p + Szt @ Szt + Szt
w, V = np.linalg.eigh(S2op)
P = V[:, np.abs(w - S2 * (S2 + 1)) < 1e-8]
Hp = P.conj().T @ Hd @ P; Szp = P.conj().T @ Szt @ P
e, U = np.linalg.eigh(Hp)
sz_diag = np.real(np.einsum("ia,ij,ja->a", U.conj(), Szp, U))
sz2_diag = np.real(np.einsum("ia,ij,ja->a", U.conj(), Szp @ Szp, U))
ref = {}
for T in Ts:
    p = np.exp(-(e - e.min()) / T); z = p.sum()
    ref.setdefault("E", []).append((p * e).sum() / z)
    m = (p * sz_diag).sum() / z; m2 = (p * sz2_diag).sum() / z
    ref.setdefault("M", []).append(m); ref.setdefault("chi", []).append((m2 - m * m) / (T * N))
ref = {k: np.array(v) for k, v in ref.items()}

b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
sym = qed.Symmetry(spatial=None, total_spin=S2)
lines = []
bad = False
for method in ("exact", "ftlm", "mtpq"):
    try:
        r = qed.thermal(H, list(Ts), method=method, sym=sym, samples=20, seed=7)
    except Exception as ex:
        print(f"{method}: raised {type(ex).__name__}: {str(ex)[:150]}")
        continue
    if True:
        dE = float(np.max(np.abs(r.E - ref["E"])))
        M = None if r.M is None else np.round(r.M, 6).tolist()
        chi = None if r.chi is None else np.round(r.chi, 6).tolist()
        print(f"{method}: E err {dE:.2e}  M {M}  chi {chi}")
        if r.M is not None and (np.max(np.abs(r.M - ref["M"])) > 1e-6 or np.max(np.abs(r.chi - ref["chi"])) > 1e-6):
            bad = True
            lines.append(f"{method}:M={M},chi={chi}")
print("reference M", np.round(ref["M"], 6).tolist(), "chi", np.round(ref["chi"], 6).tolist())
if bad:
    print(f"REPRO: CONFIRMED {'; '.join(lines)} vs ref M={np.round(ref['M'],6).tolist()} chi={np.round(ref['chi'],6).tolist()}")
else:
    print("REPRO: NOT_REPRODUCED M and chi match the S=1 ensemble (or are not returned)")
