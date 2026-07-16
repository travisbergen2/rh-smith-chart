"""E-LI-1b endpoint stage. Requires G1 PASS + G2 report (binding D_max).

Per protocol §§2-5 + Deviations Log entries 1-4:
  nulls: mass-matched double-pair control + windowed jitter (gamma<=5000),
         seeds 6000-6019, 20 replicates per gamma*.
  cells: 12 = eps {0.1,0.05,0.02,0.01} x gamma* {14.13, 98.83, 999.79},
         pair -> off-line quadruple in the CLEAN multiset.
  Li:    d_n = lambda_mod - mean(lambda_null); fire d_n < -3 SD_null(n).
  Toepl: t_m = mu_min/(tr/m); fire below (null mean - 3 SD) or mu < 0.
  Theta: peak |Theta| on registered probe grids; fire peak > 1 + 5 SD_null.
"""
import json
import numpy as np
import lib1b

G = lib1b.load_zeros()
Tfull = float(G[-1])
g2 = json.load(open("data/g2_report.json"))
DMAX = min(int(g2["binding_D_max"]), lib1b.DMAX_CAP)
assert g2["feasible"], "G2 infeasible; endpoint blocked"
print("binding D_max =", DMAX)

n_arr = np.arange(1, DMAX + 1)
norm = (n_arr / 2.0) * np.log(n_arr + 1.0)
grid = np.arange(2, DMAX + 1, lib1b.M_GRID_STEP)

# ---------------- static (unjittered tail gamma > 5000) ----------------
head_mask = G <= lib1b.JITTER_WINDOW
G_head, G_tail = G[head_mask], G[~head_mask]
print("head zeros:", len(G_head), "| tail zeros:", len(G_tail))

lam_tail_static = lib1b.lambda_zero_side(G_tail, DMAX) + lib1b.lambda_tail(Tfull, DMAX)
A_tail_static = lib1b.a_zeta_full(G_tail, DMAX)

lam_base = lam_tail_static + lib1b.lambda_zero_side(G_head, DMAX)
A_base = A_tail_static + lib1b.a_zeta_full(G_head, DMAX)

# ---------------- N0 validity ------------------------------------------
mnB, mxB, trB = lib1b.mu_curve(A_base, grid)
t_base = mnB / trB
s_base = lam_base / norm
N0 = {"li_min": float(s_base.min()), "toeplitz_min": float(t_base.min()),
      "pass": bool(s_base.min() > 0 and t_base.min() > 0)}
print("N0:", N0)

# ---------------- null ensembles ---------------------------------------
targets = {}
for gs in lib1b.GAMMA_TARGETS:
    idx = int(np.argmin(np.abs(G - gs)))
    targets[gs] = float(G[idx])
print("targets:", targets)

nulls = {}   # gamma* -> dict with lam (20,DMAX), t (20,len(grid))
import os
for gs, gstar in targets.items():
    lam_reps, t_reps = [], []
    for seed in lib1b.SEEDS_CONF:
        ck = f"data/null_{gstar:.0f}_{seed}.npz"
        if os.path.exists(ck):
            d = np.load(ck)
            lam_reps.append(d["lam"]); t_reps.append(d["t"])
            continue
        head_j = lib1b.jittered_head(G_head, seed)
        i_near = int(np.argmin(np.abs(head_j - gstar)))
        g_dup = float(head_j[i_near])
        lam_rep = (lam_tail_static
                   + lib1b.lambda_zero_side(head_j, DMAX)
                   + lib1b.pair_contrib_li(g_dup, DMAX))
        A_rep = (A_tail_static
                 + lib1b.a_zeta_full(head_j, DMAX)
                 + lib1b.pair_contrib_A(g_dup, DMAX))
        mn, tr = lib1b.mu_min_scan(A_rep, grid)
        t_rep = mn / tr
        np.savez(ck, lam=lam_rep, t=t_rep)
        lam_reps.append(lam_rep); t_reps.append(t_rep)
        print(f"  null g*={gstar:.2f} seed={seed} done", flush=True)
    nulls[gstar] = {
        "lam_mean": np.mean(lam_reps, axis=0), "lam_sd": np.std(lam_reps, axis=0, ddof=1),
        "t_mean": np.mean(t_reps, axis=0), "t_sd": np.std(t_reps, axis=0, ddof=1),
        "lam_reps": np.array(lam_reps), "t_reps": np.array(t_reps)}

# ---------------- cells -------------------------------------------------
cells = []
for gs, gstar in targets.items():
    for eps in lib1b.EPSILONS:
        ckc = f"data/cellres_g{gstar:.0f}_e{eps}.json"
        if os.path.exists(ckc):
            cells.append(json.load(open(ckc)))
            continue
        lam_mod = (lam_base
                   - lib1b.pair_contrib_li(gstar, DMAX)
                   + lib1b.pair_contrib_li(gstar, DMAX, eps=eps))
        A_mod = (A_base
                 - lib1b.pair_contrib_A(gstar, DMAX)
                 + lib1b.pair_contrib_A(gstar, DMAX, eps=eps))
        mn, mx, tr = lib1b.mu_curve(A_mod, grid)
        t_mod = mn / tr

        nb = nulls[gstar]
        d_n = lam_mod - nb["lam_mean"]
        li_fire = d_n < (-3.0 * nb["lam_sd"])
        N_li = int(n_arr[np.argmax(li_fire)]) if li_fire.any() else None

        thr = nb["t_mean"] - 3.0 * nb["t_sd"]
        to_fire = (t_mod < thr) | (mn < 0)
        N_to = int(grid[np.argmax(to_fire)]) if to_fire.any() else None

        cells.append({"gamma_star": gstar, "eps": eps,
                      "N_det_li": N_li, "N_det_toeplitz": N_to,
                      "mu_min_at_end": float(mn[-1]),
                      "d_n_min_over_sd": float((d_n / np.maximum(nb["lam_sd"], 1e-300)).min())})
        np.savez(f"data/cell_g{gstar:.0f}_e{eps}.npz",
                 t_mod=t_mod, mu=mn, mumax=mx, d_n=d_n, lam_mod=lam_mod)
        with open(ckc, "w") as fck:
            json.dump(cells[-1], fck)
        print("cell", cells[-1], flush=True)

np.savez("data/nulls.npz", **{f"{k}_{kk}": vv for k, d in nulls.items()
                              for kk, vv in d.items()})
np.savez("data/base.npz", lam_base=lam_base, A_base=A_base,
         t_base=t_base, s_base=s_base, grid=grid)
with open("data/endpoint_primary.json", "w") as f:
    json.dump({"D_max": DMAX, "N0": N0, "targets": targets, "cells": cells},
              f, indent=1)
print("ENDPOINT PRIMARY DONE")
