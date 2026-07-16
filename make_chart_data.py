"""Generate JSON data for the interactive RH Smith chart.

Committed parameters (stated, reproducible):
  zeros: Odlyzko table, first 100,000 (data/zeros1.txt)
  delta = 0.05 (Paper 14 primary)
  depth: n, M = 1..445  (E-LI-1 gate G2's binding D_max at this zero count)
  planted demo: quadruple at beta = 0.1, gamma = 98.831194 (nearest table
  zero to 100), replacing the on-line pair at that height — the one cell
  E-LI-1's post-hoc analysis identified as an honest epsilon-driven
  detection (N_det = 272).
"""
import json
import numpy as np
import core

DMAX = 445
DELTA = 0.05
BETA = 0.1
GAMMA_T = 98.831194

g = core.load_zeros(n=100000)

out = {}

# --- zeros on the chart (display subset: first 60 for point markers) ---
z = core.zeros_to_disk(g[:60])
out["zeros_disk"] = [{"re": float(w.real), "im": float(w.imag),
                      "gamma": float(gm)} for w, gm in zip(z, g[:60])]

# --- planted off-line quadruple on the chart ---
zp = []
for sg_b in (+1, -1):
    for sg_g in (+1, -1):
        w = core.li_map(0.5 + sg_b * BETA + 1j * sg_g * GAMMA_T)
        zp.append({"re": float(w.real), "im": float(w.imag),
                   "beta": sg_b * BETA, "gamma": sg_g * GAMMA_T})
out["planted_disk"] = zp
out["planted_params"] = {"beta": BETA, "gamma": GAMMA_T}

# --- Smith grid: sigma-circles (constant Re s) and gamma-arcs ---
grid = {"sigma_circles": [], "gamma_arcs": []}
for sigma in [0.0, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0, 1.5, 3.0]:
    c = core.sigma_circle(sigma)
    grid["sigma_circles"].append({
        "sigma": sigma,
        "re": [float(v) for v in c.real],
        "im": [float(v) for v in c.imag]})
for gamma in [2.0, 5.0, 14.134725, 30.0, 60.0]:
    for sg in (+1, -1):
        a = core.gamma_arc(sg * gamma, sig_min=0.02, sig_max=80.0, num=300)
        grid["gamma_arcs"].append({
            "gamma": sg * gamma,
            "re": [float(v) for v in a.real],
            "im": [float(v) for v in a.imag]})
out["grid"] = grid

# --- lambda_n, true zeros and planted ---
lam_true = core.li_lambda(g, DMAX)
idx = int(np.argmin(np.abs(g - GAMMA_T)))
g_mod = np.delete(g, idx)
lam_planted = core.li_lambda(g_mod, DMAX, beta_pairs=[(BETA, GAMMA_T)])
n = np.arange(1, DMAX + 1)
norm = (n / 2.0) * np.log(n + 1.0)
out["li"] = {
    "n": n.tolist(),
    "lambda_true": [float(v) for v in lam_true],
    "lambda_planted": [float(v) for v in lam_planted],
    "s_true": [float(v) for v in lam_true / norm],
    "s_planted": [float(v) for v in lam_planted / norm],
}

# --- A(m) Toeplitz mu_min curves ---
A_true = core.a_zeta(g, DMAX, DELTA)
A_planted = core.a_zeta(g_mod, DMAX, DELTA, beta_pairs=[(BETA, GAMMA_T)])
Ms = list(range(2, DMAX + 1, 1))
mu_true, mu_planted = [], []
for M in Ms:
    mu_true.append(core.toeplitz_mu_min(A_true, M))
    mu_planted.append(core.toeplitz_mu_min(A_planted, M))
out["toeplitz"] = {
    "M": Ms,
    "mu_true": [float(v) for v in mu_true],
    "mu_planted": [float(v) for v in mu_planted],
    "delta": DELTA,
}

# detection depth for the planted case (first M with mu<0)
det = next((M for M, v in zip(Ms, mu_planted) if v < 0), None)
out["toeplitz"]["planted_detection_M"] = det

with open("data/chart_data.json", "w") as f:
    json.dump(out, f)

print("zeros:", len(g), "| lambda_1 =", lam_true[0],
      "| mu_min(445) true =", mu_true[-1],
      "| planted detection M =", det)


# --- Theta_omega innerness gauge (Suzuki encoding, third meter) ---
# omega = 0.05; probe line Im z = 0.05; fine grid resolving the pole
# width beta - omega = 0.05 (instrument lesson: this gauge is LOCAL).
OMEGA = 0.05
xg = np.linspace(-101.5, -96.0, 551)
th_true = core.theta_innerness_margin(g, OMEGA, xg, v=0.05)
th_planted = core.theta_innerness_margin(
    g, OMEGA, xg, v=0.05, beta_pairs=[(BETA, GAMMA_T)], drop_gamma=GAMMA_T)
out["theta"] = {
    "omega": OMEGA, "v": 0.05,
    "x": [float(v) for v in xg],
    "abs_true": [float(v) for v in th_true],
    "abs_planted": [float(v) for v in th_planted],
    "peak_planted": float(th_planted.max()),
}

with open("data/chart_data.json", "w") as f:
    json.dump(out, f)
print("theta peak (planted):", th_planted.max(),
      "| theta max (true):", th_true.max())
