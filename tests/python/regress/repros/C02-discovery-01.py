# AUDIT-ID: C02-discovery-01
# DEVICE: cpu
# SECONDS: 60
"""Claim: Symmetry(sz='even'/'odd') is silently ignored when H conserves U(1) Sz.
subspaces() (src/solvers/little_group/lg_sectors.cpp:175-184) enumerates every n_up in the U1
branch and never reads Spec.sz_parity, so spectrum / eigs / thermal run over the full space
although the docstring promises "one parity half". Test: 8-site Heisenberg ring, spatial=None,
against an independent dense numpy reference restricted to even/odd popcount."""

import signal
import sys

import numpy as np

import qed

signal.alarm(240)
N = 8

# ---- dense reference (independent of qed) ----
sx = np.array([[0, 0.5], [0.5, 0]], complex)
sy = np.array([[0, -0.5j], [0.5j, 0]], complex)
sz = np.array([[0.5, 0], [0, -0.5]], complex)


def site(op, i):
    out = np.array([[1.0 + 0j]])
    for j in range(N):
        out = np.kron(out, op if j == i else np.eye(2))
    return out


Hd = np.zeros((2**N, 2**N), complex)
for i in range(N):
    j = (i + 1) % N
    for s in (sx, sy, sz):
        Hd += site(s, i) @ site(s, j)
pop = np.array([bin(x).count("1") for x in range(2**N)])
E_full = np.linalg.eigvalsh(Hd)
halves = {}
for par in (0, 1):
    idx = np.where(pop % 2 == par)[0]
    halves[par] = np.linalg.eigvalsh(Hd[np.ix_(idx, idx)])

# ---- qed ----
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()

ref_full = np.sort(np.asarray(qed.spectrum(H, sym=qed.Symmetry.none()).energies))
conv_ok = len(ref_full) == 2**N and np.max(np.abs(ref_full - E_full)) < 1e-8
if not conv_ok:
    print(
        "REPRO: INCONCLUSIVE coupling convention differs from S.S dense reference "
        f"(len {len(ref_full)}, max|dE| {np.max(np.abs(ref_full - E_full)) if len(ref_full) == 2**N else 'n/a'})"
    )
    sys.exit(0)

findings = []
for key, par in (("even", 0), ("odd", 1)):
    sym = qed.Symmetry(spatial=None, sz=key)
    try:
        e = np.sort(np.asarray(qed.spectrum(H, sym=sym).energies))
    except Exception as ex:
        print(f"spectrum sz={key}: raised {type(ex).__name__}: {ex}")
        continue
    want = halves[par]
    match_half = len(e) == len(want) and np.max(np.abs(e - want)) < 1e-8
    match_full = len(e) == len(E_full) and np.max(np.abs(e - E_full)) < 1e-8
    print(
        f"spectrum sz={key}: {len(e)} energies (parity half has {len(want)}); "
        f"equals half={match_half} equals full={match_full}"
    )
    if not match_half:
        findings.append(f"spectrum[{key}] n={len(e)} vs {len(want)}")

# eigs k=1, sz='odd': lowest odd-popcount level
try:
    e1 = float(np.asarray(qed.eigs(H, 1, sym=qed.Symmetry(spatial=None, sz="odd")).energies)[0])
    print(f"eigs sz=odd: E0={e1:.12f}  odd-half E0={halves[1][0]:.12f}  global E0={E_full[0]:.12f}")
    if abs(e1 - halves[1][0]) > 1e-8:
        findings.append(f"eigs[odd] E0={e1:.10f} (odd half {halves[1][0]:.10f})")
except Exception as ex:
    print(f"eigs sz=odd: raised {type(ex).__name__}: {ex}")

# thermal exact, sz='odd' vs sz='auto' (full space) vs dense halves: entropy at high T
T = [0.5, 2.0, 50.0]
try:
    th_odd = qed.thermal(H, T, method="exact", sym=qed.Symmetry(spatial=None, sz="odd"))
    th_all = qed.thermal(H, T, method="exact", sym=qed.Symmetry(spatial=None))
    same = np.allclose(th_odd.entropy, th_all.entropy, atol=1e-10) and np.allclose(th_odd.E, th_all.E, atol=1e-10)
    print(f"thermal S(T) sz=odd {np.round(th_odd.entropy, 6)}  full {np.round(th_all.entropy, 6)}  identical={same}")
    print(
        f"  expectations at T=50: odd half S->~{7*np.log(2):.4f} (7 ln2), full ~{8*np.log(2):.4f} (8 ln2)"
        " (per-site normalisation would divide by N)"
    )
    if same:
        findings.append("thermal[odd] identical to full-space thermal")
except Exception as ex:
    print(f"thermal sz=odd: raised {type(ex).__name__}: {ex}")

if findings:
    print("REPRO: CONFIRMED " + "; ".join(findings))
else:
    print("REPRO: NOT_REPRODUCED sz parity honoured on a U(1) H")
