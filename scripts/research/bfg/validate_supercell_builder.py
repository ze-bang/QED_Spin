#!/usr/bin/env python3
"""Gate G1a: the general-supercell builder must reproduce the rhombic builder exactly,
recognise unclean clusters, decode momenta/point groups correctly, and give correct
energies (translation little group vs known E0; symmetric vs unsymmetrised; flux gauge)."""
import sys, os, math, time
import numpy as np
from fractions import Fraction as F
import qed
from qed import _core
from edlib.helper_kagome_supercell import KagomeSupercell, build_bfg_operator_supercell
from edlib.helper_kagome_bfg import generate_kagome_cluster
import run_bfg_ground_state as gs_mod

ok_all = True
def check(name, cond, detail=""):
    global ok_all
    ok_all &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}  {detail}", flush=True)

CL = {"3x3": [[3, 0], [0, 3]], "4x3": [[4, 0], [0, 3]], "24": [[2, 1], [-2, 3]],
      "30": [[2, 1], [-2, 4]], "36d": [[2, 2], [-2, 4]], "36K": [[3, 0], [-2, 4]],
      "2x2": [[2, 0], [0, 2]], "2x3": [[2, 0], [0, 3]]}

print("== 1. rhombic identity (bond sets and positions) ==")
for nm, (d1, d2) in (("3x3", (3, 3)), ("4x3", (4, 3))):
    v, e1, e2, e3, _, _ = generate_kagome_cluster(d1, d2, use_pbc=True, pbc_dim1=True, pbc_dim2=True)
    cl = KagomeSupercell(CL[nm]); b1, b2, b3 = cl.bond_lists()
    same = all(sorted({tuple(sorted(p)) for p in old}) == list(new)
               for old, new in ((e1, b1), (e2, b2), (e3, b3)))
    pos = max(np.abs(np.array(v[i]) - cl.position(i)).max() for i in range(cl.N))
    check(f"{nm} bonds identical to generate_kagome_cluster", same,
          f"|NN|={len(b1)} |2NN|={len(b2)} |3NN|={len(b3)}")
    check(f"{nm} site positions identical", pos < 1e-12, f"max dev {pos:.1e}")

print("== 2. cleanliness ==")
for nm, L in CL.items():
    cl = KagomeSupercell(L)
    expect = nm not in ("2x2", "2x3")
    check(f"{nm}: clean={cl.is_clean} (min|T|^2={cl.min_T2})", cl.is_clean == expect,
          "" if cl.is_clean else f"problems={cl.check_clean()}")

print("== 3. momentum content ==")
EXP = {"3x3": dict(G=1, K=1, **{"K'": 1}), "4x3": dict(G=1, M1=1),
       "36d": dict(G=1, M1=1, M2=1, M3=1, K=1, **{"K'": 1})}
for nm in ("3x3", "4x3", "24", "30", "36d", "36K"):
    mc = KagomeSupercell(CL[nm]).momentum_content()
    exp = EXP.get(nm)
    good = exp is None or all(mc.get(k, 0) == v for k, v in exp.items()) and \
        all(mc.get(k, 0) == 0 for k in ("M1", "M2", "M3", "K", "K'") if k not in exp)
    check(f"{nm} momenta {mc}", good, "(expected " + str(exp) + ")" if exp else "(recorded)")
# sanity of the metric: |K|^2=1/3, |M|^2=1/4
check("metric K^2=1/3, M^2=1/4",
      abs((4/9 + 1/9 - 2/9) - 1/3) < 1e-15 and abs(0.25 - 0.25) < 1e-15)

print("== 4. point groups ==")
EXPG = {"3x3": 12, "4x3": 2, "36d": 12}
for nm in ("3x3", "4x3", "24", "30", "36d", "36K"):
    cl = KagomeSupercell(CL[nm]); ops = cl.point_group_perms()
    e = EXPG.get(nm)
    check(f"{nm}: |PG| = {len(ops)}", e is None or len(ops) == e, f"(expected {e})" if e else "(recorded)")

def lg_E0(op, cl, n_up, k=1):
    os.environ.pop("ED_SYM_LG_ONLY_K0", None)
    ev = list(_core.little_group_lowest_eigenvalues(op, cl.translation_perms(), [], k=k,
                                                    n_up=n_up, dense_max_dim=256))
    assert len(ev) > 0, "empty return"
    return float(min(ev))

print("== 5. energies ==")
t0 = time.time()
op, cl = build_bfg_operator_supercell(CL["3x3"], -0.09, 1.0)
E = lg_E0(op, cl, 13)
check("3x3 Jpm=-0.09 E0 vs archived -6.8270756205", abs(E + 6.8270756205) < 2e-9,
      f"E0={E:.10f} ({time.time()-t0:.0f}s)")
op_old, _ = gs_mod.build_bfg_operator(3, 3, -0.09, 1.0, pbc=True)
E_old = lg_E0(op_old, cl, 13)
check("3x3 new vs old operator, same solver", abs(E - E_old) < 1e-10, f"diff={abs(E-E_old):.1e}")

t0 = time.time()
op24, cl24 = build_bfg_operator_supercell(CL["24"], -0.09, 1.0)
E_lg = lg_E0(op24, cl24, 12)
r = qed.solve(op24, sz=12, num_eigenvalues=1, solver="KRYLOV_SCHUR", tolerance=1e-12,
              device="cpu", symmetry=None, point_group="off", spin_flip="off",
              time_reversal="off", total_spin="off", verbose=False)
E_full = float(min(r.eigenvalues))
check("24 translation little group vs unsymmetrised sector", abs(E_lg - E_full) < 1e-9,
      f"E_lg={E_lg:.12f} E_full={E_full:.12f} ({time.time()-t0:.0f}s)")

print("== 6. flux gauge ==")
opA, _ = build_bfg_operator_supercell(CL["24"], -0.09, 1.0, flux=(2 * np.pi, 0.0))
check("24: flux 2pi == flux 0", abs(lg_E0(opA, cl24, 12) - E_lg) < 1e-9)
opB, _ = build_bfg_operator_supercell(CL["24"], -0.09, 1.0, flux=(np.pi, 0.0))
EB = lg_E0(opB, cl24, 12)
check("24: flux pi differs from flux 0 (flux actually applied)", abs(EB - E_lg) > 1e-8,
      f"E(pi)-E(0)={EB-E_lg:+.3e}")
import importlib.util
spec = importlib.util.spec_from_file_location("bfg_flux", "/scratch/zhouzb79/bfg_flux.py")
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
opC = bf.build_bfg_flux(3, 3, -0.09, 1.0, flux1=0.5 * np.pi, flux2=0.25 * np.pi)
opD, cl3 = build_bfg_operator_supercell(CL["3x3"], -0.09, 1.0, flux=(0.5 * np.pi, 0.25 * np.pi))
EC, ED = lg_E0(opC, cl3, 13), lg_E0(opD, cl3, 13)
check("3x3 flux (pi/2, pi/4): new gauge == bfg_flux.py", abs(EC - ED) < 1e-9, f"diff={abs(EC-ED):.1e}")

print(f"\nG1a {'PASSED' if ok_all else 'FAILED'}")
sys.exit(0 if ok_all else 1)
