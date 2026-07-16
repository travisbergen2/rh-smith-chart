"""E-LI-1b Gate G2 — tail stability. Fixes binding D_max (cap 2000).

Both primary statistics computed with zeros to T0 (full zeros6 table)
and to T0/2; binding D_max = largest depth with < 1% relative change
in BOTH statistics, capped at 2000. Infeasible if < 300.
(Registration wording 'T0 and 2*T0' is implemented as half-vs-full of
the available table, identical in substance to E-LI-1's G2.)
"""
import json
import numpy as np
import lib1b

G = lib1b.load_zeros()
Tfull = float(G[-1])
Thalf = Tfull / 2.0
G_half = G[G <= Thalf]
print("full:", len(G), "to", Tfull, "| half:", len(G_half), "to", Thalf)

NM = lib1b.DMAX_CAP
n = np.arange(1, NM + 1)
norm = (n / 2.0) * np.log(n + 1.0)

def s_curve(zeros, T):
    lam = lib1b.lambda_zero_side(zeros, NM) + lib1b.lambda_tail(T, NM)
    return lam / norm

s_full = s_curve(G, Tfull)
s_half = s_curve(G_half, Thalf)
rel_s = np.abs(s_half - s_full) / np.abs(s_full)

A_full = lib1b.a_zeta_full(G, NM)
A_half = lib1b.a_zeta_full(G_half, NM)
grid = np.arange(2, NM + 1, lib1b.M_GRID_STEP)
mnF, mxF, trF = lib1b.mu_curve(A_full, grid)
mnH, mxH, trH = lib1b.mu_curve(A_half, grid)
t_full = mnF / trF
t_half = mnH / trH
rel_t = np.abs(t_half - t_full) / np.abs(t_full)

# binding D_max: largest depth d such that ALL depths <= d pass 1%
ok_s = rel_s < 0.01
dmax_s = int(np.argmax(~ok_s)) if (~ok_s).any() else NM   # first fail index -> depth = n before it
dmax_s = dmax_s if (~ok_s).any() else NM
ok_t = rel_t < 0.01
if (~ok_t).any():
    first_fail = int(np.argmax(~ok_t))
    dmax_t = int(grid[first_fail - 1]) if first_fail > 0 else 0
else:
    dmax_t = NM

D_MAX = min(dmax_s if dmax_s > 0 else 0, dmax_t, NM)
report = {
    "n_zeros_full": int(len(G)), "T_full": Tfull,
    "n_zeros_half": int(len(G_half)), "T_half": Thalf,
    "max_rel_s": float(rel_s.max()),
    "first_s_violation_n": (int(np.argmax(~ok_s)) + 1) if (~ok_s).any() else None,
    "max_rel_t": float(rel_t.max()),
    "first_t_violation_M": int(grid[np.argmax(~ok_t)]) if (~ok_t).any() else None,
    "binding_D_max": int(D_MAX),
    "feasible": bool(D_MAX >= 300),
}
np.savez("data/g2_curves.npz", s_full=s_full, s_half=s_half,
         grid=grid, t_full=t_full, t_half=t_half,
         A_full=A_full, A_half=A_half)
with open("data/g2_report.json", "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
