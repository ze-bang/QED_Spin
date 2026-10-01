# AUDIT-ID: P6-engine-setup-01
# DEVICE: cpu
# SECONDS: 120
"""Claim: the group-sector orbit table (try_group_path -> group_orbit_table ->
build_orbit_table_fixed_sz_streaming) bypasses the orbit-table registry, so it is rebuilt by a
full C(N, n_up) scan on every walk: two identical eigs calls in one process rebuild it twice
(while the abelian table is served from the registry), and a default pruned eigs re-walks a
surviving star and scans again. Observed through the ED_SYM_PROFILE=1 stderr profile lines."""
import os, re, subprocess, sys, signal

CHILD = r'''
import sys, qed
N = 24
b = qed.input.HamiltonianBuilder(N)
b.heisenberg([(i, (i + 1) % N) for i in range(N)], J=1.0)
H = b.to_operator()
T = [(i + 1) % N for i in range(N)]
R = [(-i) % N for i in range(N)]
sym = qed.Symmetry(spatial=[T, R], sz=N // 2, spin_flip="off", time_reversal="off")
for tag, prune in (("A", False), ("B", False), ("C", True)):
    sys.stderr.write("@@CALL %s\n" % tag); sys.stderr.flush()
    r = qed.eigs(H, 1, sym=sym, prune=prune)
    sys.stderr.write("@@E %s %.12f pruned=%d\n" % (tag, r.energies[0], r.pruned_blocks)); sys.stderr.flush()
'''

def main():
    env = dict(os.environ, ED_SYM_PROFILE="1", ED_SYM_CACHE="0")
    try:
        p = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=280)
    except subprocess.TimeoutExpired:
        print("REPRO: INCONCLUSIVE child timed out"); return
    if p.returncode != 0:
        print(f"REPRO: INCONCLUSIVE child rc={p.returncode} {p.stderr[-400:]!r}"); return
    calls, cur = {}, None
    for line in p.stderr.splitlines():
        m = re.match(r"@@CALL (\w)", line)
        if m:
            cur = m.group(1); calls[cur] = {"scans": [], "group": []}; continue
        if cur is None:
            continue
        m = re.search(r"fused orbit-table.*:\s*([\d.]+) s\s+\((\d+) items\)", line)
        if m:
            calls[cur]["scans"].append(int(m.group(2)))
        m = re.search(r"star k0=(\d+): group-sector path,", line)
        if m:
            calls[cur]["group"].append(int(m.group(1)))
    if not all(k in calls for k in "ABC"):
        print("REPRO: INCONCLUSIVE missing call markers"); return
    A, B, C = calls["A"], calls["B"], calls["C"]
    info = (f"callA scans={A['scans']} group_stars={A['group']} | callB scans={B['scans']} "
            f"group_stars={B['group']} | callC(prune) scans={C['scans']} group_stars={C['group']}")
    if not A["group"]:
        print("REPRO: INCONCLUSIVE no star took the group-sector path; " + info); return
    abelian = A["scans"][0] if A["scans"] else None
    b_group_scans = [x for x in B["scans"] if x != abelian]
    b_abelian = [x for x in B["scans"] if x == abelian]
    rewalk = max((C["group"].count(k) for k in set(C["group"])), default=0)
    if b_group_scans and not b_abelian:
        print(f"REPRO: CONFIRMED repeat call rebuilt {len(b_group_scans)} group table(s) while the abelian "
              f"table ({abelian} reps) came from the registry; pruned call visited a group star {rewalk}x; " + info)
    elif b_group_scans:
        print("REPRO: CONFIRMED group tables rebuilt on the repeat call (abelian also rebuilt); " + info)
    else:
        print("REPRO: NOT_REPRODUCED repeat call did not rescan the group tables; " + info)

if __name__ == "__main__":
    signal.alarm(295)
    main()
