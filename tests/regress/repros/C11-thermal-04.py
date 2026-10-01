# AUDIT-ID: C11-thermal-04
# DEVICE: cpu
# SECONDS: 150
"""Claim: OFTLM (qed.thermal(method='ftlm', exact_states=Nv)) takes its Nv 'exact' states from one
2*Nv+30-step Lanczos with no residual check and weights them by exp(-beta*theta_i) (Ritz values).
For unconverged Ritz pairs the exact part should be <v_i|exp(-beta H)|v_i> >= exp(-beta theta_i)
(Jensen), so Z (lnZ) is biased LOW by an amount that does not shrink with samples.
Demonstrator: a 12-site Heisenberg ring plus lambda*(S^z_tot)^2 with lambda=10 (wide spectrum,
so a 46-step Lanczos cannot converge the low levels), Symmetry.none() -> one 4096-state block
(above the 512-state exact fallback). Plain FTLM with the same deep stochastic Lanczos is the
control. Reference: independent dense numpy ED."""
import numpy as np
import scipy.sparse as sps
import qed

N, LAM = 12, 10.0
Ts = np.array([0.1, 0.2, 0.4, 0.8])
bonds = [(i, (i + 1) % N) for i in range(N)]
pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]

# independent dense reference (Kronecker products)
sp_ = sps.csr_matrix(np.array([[0.0, 1.0], [0.0, 0.0]])); sm_ = sp_.T.tocsr()
sz_ = sps.csr_matrix(np.diag([0.5, -0.5]))
def at(o, i):
    return sps.kron(sps.kron(sps.identity(2 ** i), o), sps.identity(2 ** (N - i - 1)), format="csr")
SP = [at(sp_, i) for i in range(N)]; SM = [at(sm_, i) for i in range(N)]; SZ = [at(sz_, i) for i in range(N)]
Hs = sps.csr_matrix((2 ** N, 2 ** N))
for i, j in bonds:
    Hs = Hs + 0.5 * (SP[i] @ SM[j] + SM[i] @ SP[j]) + SZ[i] @ SZ[j]
for i, j in pairs:
    Hs = Hs + 2.0 * LAM * (SZ[i] @ SZ[j])
ev = np.linalg.eigvalsh(Hs.toarray())
ref_lnZ, ref_E = [], []
for T in Ts:
    b = 1.0 / T; w = np.exp(-b * (ev - ev[0])); z = w.sum()
    ref_lnZ.append(np.log(z) - b * ev[0]); ref_E.append((w * ev).sum() / z)
ref_lnZ = np.array(ref_lnZ); ref_E = np.array(ref_E)

H = qed.Operator(N, 0.5)
for i, j in bonds:
    H.add_two_body(qed.OP_SPLUS, i, qed.OP_SMINUS, j, 0.5)
    H.add_two_body(qed.OP_SMINUS, i, qed.OP_SPLUS, j, 0.5)
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 1.0)
for i, j in pairs:
    H.add_two_body(qed.OP_SZ, i, qed.OP_SZ, j, 2.0 * LAM)
sym = qed.Symmetry.none()

dev = {"oftlm": [], "ftlm": []}
devE = {"oftlm": [], "ftlm": []}
try:
    for seed in (11, 22, 33, 44):
        for name, nv in (("oftlm", 8), ("ftlm", 0)):
            r = qed.thermal(H, list(Ts), method="ftlm", exact_states=nv, krylov=300, samples=200,
                            seed=seed, sym=sym)
            dev[name].append(np.asarray(r.lnZ) - ref_lnZ)
            devE[name].append(np.asarray(r.E) - ref_E)
except Exception as ex:
    print(f"REPRO: INCONCLUSIVE thermal raised {type(ex).__name__}: {str(ex)[:200]}")
    raise SystemExit(0)

out = []
hit = False
for name in ("oftlm", "ftlm"):
    d = np.array(dev[name]); m = d.mean(0); se = d.std(0, ddof=1) / np.sqrt(len(d))
    dE = np.array(devE[name]).mean(0)
    print(f"{name}: mean dlnZ {np.round(m, 4).tolist()} sem {np.round(se, 4).tolist()} mean dE {np.round(dE, 4).tolist()}")
mo = np.array(dev["oftlm"]).mean(0); so = np.array(dev["oftlm"]).std(0, ddof=1) / 2
mf = np.array(dev["ftlm"]).mean(0); sf = np.array(dev["ftlm"]).std(0, ddof=1) / 2
for t in range(len(Ts)):
    if mo[t] < -5 * so[t] - 1e-3 and abs(mo[t]) > 3 * abs(mf[t]) + 3 * sf[t]:
        hit = True
        out.append(f"T={Ts[t]}: OFTLM dlnZ={mo[t]:.4f}+-{so[t]:.4f} vs FTLM {mf[t]:.4f}+-{sf[t]:.4f}")
if hit:
    print("REPRO: CONFIRMED systematic low lnZ from OFTLM exact part: " + "; ".join(out))
else:
    print(f"REPRO: NOT_REPRODUCED OFTLM dlnZ {np.round(mo, 4).tolist()} (sem {np.round(so, 4).tolist()}), "
          f"FTLM dlnZ {np.round(mf, 4).tolist()}")
