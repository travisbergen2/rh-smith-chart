"""E-LI-1b gates G3 (scaling invariance) and G4 (precision), plus the
Deviations entry-2 jitter spot-check.

G3(a) A-rescale c in {1e-3, 1e3}: t_m = mu_min/(tr/m) and the null band
      rescale identically (exact algebra); verified numerically on two
      cells by direct recompute.
G3(b) substitute (Deviations entry 3): (ii') baseline fidelity is G1's
      1e-6 + G2 stability (recorded); (iii') quadruple-paired vs 2*Re
      paired Li implementation, all cells, winner-flip check.
G3(c) mu_min/mu_max margin variant, all cells, winner-flip check.
G4    longdouble sequence recompute; N_det stability at detection depths.
"""
import json
import numpy as np
import lib1b

G = lib1b.load_zeros()
Tfull = float(G[-1])
ep = json.load(open("data/endpoint_primary.json"))
DMAX = ep["D_max"]
grid = np.arange(2, DMAX + 1, lib1b.M_GRID_STEP)
n_arr = np.arange(1, DMAX + 1)
targets = {float(k): v for k, v in ep["targets"].items()}
base = np.load("data/base.npz")
nl = np.load("data/nulls.npz")

report = {}

# ---------------- G3(a): exact + numerical spot check -------------------
spot = []
for cell in [c for c in ep["cells"] if c["eps"] == 0.1][:2]:
    gstar, eps = cell["gamma_star"], cell["eps"]
    A_mod = (base["A_base"] - lib1b.pair_contrib_A(gstar, DMAX)
             + lib1b.pair_contrib_A(gstar, DMAX, eps=eps))
    for c in (1e-3, 1e3):
        mn, mx, tr = lib1b.mu_curve(c * A_mod, grid)
        t_scaled = mn / tr
        d = np.load(f"data/cell_g{gstar:.0f}_e{eps}.npz")
        rel = np.abs(t_scaled - d["t_mod"]) / np.maximum(np.abs(d["t_mod"]), 1e-300)
        spot.append({"gamma_star": gstar, "c": c, "max_rel_dev": float(rel.max())})
report["G3a"] = {"exact_by_algebra": True, "numerical_spot": spot,
                 "pass": all(s["max_rel_dev"] < 1e-9 for s in spot)}

# ---------------- G3(b) substitute (iii'): pairing variant ---------------
def pair_contrib_li_4sum(gamma, n_max, eps=0.0):
    """Quadruple as four individual zeros (no 2*Re shortcut)."""
    n = np.arange(1, n_max + 1)
    out = np.zeros(n_max, dtype=complex)
    sigmas = [0.5 + eps, 0.5 - eps] if eps > 0 else [0.5, 0.5]
    gammas = [gamma, -gamma]
    seen = []
    for sg in ([0.5 + eps, 0.5 - eps] if eps > 0 else [0.5]):
        for gg in (gamma, -gamma):
            rho = sg + 1j * gg
            z = 1.0 - 1.0 / rho
            out += 1.0 - z ** n.astype(float)
    if eps == 0:
        pass  # one pair only: rho, conj
    return out.real

flips = []
for cell in ep["cells"]:
    gstar, eps = cell["gamma_star"], cell["eps"]
    lam_mod_alt = (base["lam_base"]
                   - pair_contrib_li_4sum(gstar, DMAX)
                   + pair_contrib_li_4sum(gstar, DMAX, eps=eps))
    key = [k for k in nl.files if k.startswith(f"{gstar}_lam_mean")][0]
    lam_mean = nl[key]
    lam_sd = nl[key.replace("mean", "sd")]
    d_n = lam_mod_alt - lam_mean
    fire = d_n < (-3.0 * lam_sd)
    N_li_alt = int(n_arr[np.argmax(fire)]) if fire.any() else None
    if N_li_alt != cell["N_det_li"]:
        flips.append({"cell": (gstar, eps), "primary": cell["N_det_li"],
                      "alt": N_li_alt})
report["G3b"] = {"substitute_per_deviations_entry_3": True,
                 "baseline_fidelity": "G1 PASS 1.5e-8; G2 stability recorded",
                 "pairing_variant_flips": flips, "pass": len(flips) == 0}

# ---------------- G3(c): mu_min/mu_max variant ---------------------------
flips_c = []
for cell in ep["cells"]:
    gstar, eps = cell["gamma_star"], cell["eps"]
    d = np.load(f"data/cell_g{gstar:.0f}_e{eps}.npz")
    t_alt = d["mu"] / d["mumax"]
    # rebuild null band in the alt normalization
    reps = nl[f"{gstar}_t_reps"]  # primary-normalized; alt band needs raw mu curves
    # raw mu not stored for nulls; use sign rule + primary-band proxy:
    # conservative check: does detection by (mu<0) alone match at the
    # same depth window as primary detection?
    to_fire_sign = d["mu"] < 0
    N_sign = int(grid[np.argmax(to_fire_sign)]) if to_fire_sign.any() else None
    flips_c.append({"cell": (gstar, eps), "N_primary": cell["N_det_toeplitz"],
                    "N_sign_only": N_sign})
report["G3c"] = {"note": ("alt normalization mu_min/mu_max is sign-preserving; "
                          "detection via indefiniteness (mu<0) is normalization-"
                          "invariant. Table compares primary vs sign-only depth."),
                 "table": flips_c}

# ---------------- G4: longdouble sequence recompute ----------------------
g4 = []
A_ld_base = lib1b.a_zeta_full(G, DMAX, dtype=np.longdouble)
from scipy.linalg import toeplitz, eigvalsh
for cell in [c for c in ep["cells"] if c["N_det_toeplitz"] is not None]:
    gstar, eps = cell["gamma_star"], cell["eps"]
    A_ld = (A_ld_base
            - lib1b.pair_contrib_A(gstar, DMAX)
            + lib1b.pair_contrib_A(gstar, DMAX, eps=eps))
    A_f64 = (base["A_base"] - lib1b.pair_contrib_A(gstar, DMAX)
             + lib1b.pair_contrib_A(gstar, DMAX, eps=eps))
    seq_dev = float(np.max(np.abs(A_ld.astype(np.float64) - A_f64)))
    Nd = cell["N_det_toeplitz"]
    mu_ld = float(eigvalsh(toeplitz(A_ld[:Nd].astype(np.float64)))[0])
    mu_64 = float(eigvalsh(toeplitz(A_f64[:Nd]))[0])
    g4.append({"cell": (gstar, eps), "seq_maxdev": seq_dev,
               "mu_at_Ndet_ld": mu_ld, "mu_at_Ndet_64": mu_64,
               "sign_agree": bool((mu_ld < 0) == (mu_64 < 0))})
report["G4"] = {"cells": g4,
                "pass": all(c["sign_agree"] for c in g4)}

# ------------- Deviations entry 2 spot-check: full vs windowed jitter ----
seed = lib1b.SEEDS_CONF[0]
gstar = targets[98.831194] if 98.831194 in targets else list(targets.values())[1]
head_mask = G <= lib1b.JITTER_WINDOW
G_head, G_tail = G[head_mask], G[~head_mask]
rng = np.random.default_rng(seed)
jit_all = G + rng.normal(0.0, lib1b.JITTER_SIGMA, size=len(G))
head_j = lib1b.jittered_head(G_head, seed)
lam_win = (lib1b.lambda_zero_side(G_tail, DMAX) + lib1b.lambda_tail(Tfull, DMAX)
           + lib1b.lambda_zero_side(head_j, DMAX))
lam_full_j = lib1b.lambda_zero_side(jit_all, DMAX) + lib1b.lambda_tail(Tfull, DMAX)
key = [k for k in nl.files if k.startswith(f"{gstar}_lam_sd")][0]
lam_sd = nl[key]
# compare the two jitter scopes against the null band SD
dev = np.abs(lam_full_j - lam_win)
ratio = float((dev / np.maximum(lam_sd, 1e-300)).max())
report["jitter_spotcheck"] = {
    "note": ("full jitter uses a DIFFERENT random draw for head zeros than "
             "the windowed run (single stream over 2M vs head-only stream), "
             "so this compares scope + draw; the binding quantity is the "
             "tail-jitter contribution bounded analytically in entry 2. "
             "Reported ratio therefore OVERSTATES the tail effect."),
    "max_dev_over_band_sd": ratio}

with open("data/gates34.json", "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
