"""E-LI-1b Gate G1 — reproduction gate. Blocks everything on failure.

Targets (frozen in protocol §6 + Appendix A of E-LI-1):
  Li:  lambda_1 = 0.0230957089661210 (published); zero-side + tail vs
       analytic route <= 1e-6 relative for n <= 20.
  A:   POLE(300) -> 90.41, PRIME(300) -> 90.59 (displayed digits);
       max|A - A_zeta(120 zeros)| -> 0.23, rms -> 0.017 over 301 lags;
       mu_min table under M = w/delta: {1.34e-3, 1.85e-4, 6.4e-5,
       4.4e-5, 3.1e-5, 2.2e-5} at windows {2.5,5,7.5,10,12.5,15}.
  Theta: 4-test battery (already passing in rh_smith at registration).
"""
import json
import numpy as np
import mpmath as mp
from scipy.linalg import toeplitz, eigvalsh
import lib1b

report = {}

# ---------- Li analytic route: Taylor coefficients of s^2 xi'/xi -------
mp.mp.dps = 30

def xi(s):
    return 0.5 * s * (s - 1) * mp.pi ** (-s / 2) * mp.gamma(s / 2) * mp.zeta(s)

def g_of_z(z):
    s = 1 / (1 - z)
    dlogxi = mp.diff(lambda t: mp.log(xi(t)), s)
    return s * s * dlogxi

NLI = 20
K = 256
r = mp.mpf("0.4")
vals = [g_of_z(r * mp.e ** (2j * mp.pi * k / K)) for k in range(K)]
lam_analytic = []
for n in range(1, NLI + 1):
    c = mp.mpf(0)
    for k in range(K):
        th = 2 * mp.pi * k / K
        c += vals[k] * mp.e ** (-1j * (n - 1) * th)
    c = mp.re(c) / K / r ** (n - 1)
    lam_analytic.append(float(c))

report["lambda_analytic_n1_20"] = lam_analytic
LAM1_PUB = 0.0230957089661210
report["lambda1_analytic_vs_published_relerr"] = abs(lam_analytic[0] - LAM1_PUB) / LAM1_PUB

# ---------- Li zero side + tail ----------------------------------------
G = lib1b.load_zeros()
T0 = float(G[-1])
lam_zs = lib1b.lambda_zero_side(G, NLI)
lam_tail = lib1b.lambda_tail(T0, NLI)
lam_full = lam_zs + lam_tail
rel = np.abs(lam_full - np.array(lam_analytic)) / np.abs(lam_analytic)
report["li_zeroside_vs_analytic_maxrel_n20"] = float(rel.max())
report["G1_Li"] = "PASS" if (rel.max() < 1e-6 and
                             report["lambda1_analytic_vs_published_relerr"] < 1e-9) else "FAIL"

# ---------- A(m): arithmetic side reproduction -------------------------
d = lib1b.DELTA
M300 = 300

# POLE
C_delta = (8.0 / d) * (np.cosh(d / 2) - 1.0)
POLE = 2.0 * C_delta * np.cosh(np.arange(0, M300 + 1) * d / 2)
report["POLE_300"] = float(POLE[300])

# PRIME: sieve Lambda(n) up to e^{15.05}
NMAX = int(np.exp((M300 + 1) * d)) + 2   # e^{15.05}
lim = NMAX
lam_von = np.zeros(lim + 1)
is_comp = np.zeros(lim + 1, dtype=bool)
for p in range(2, int(lim ** 0.5) + 1):
    if not is_comp[p]:
        is_comp[p * p::p] = True
primes = [p for p in range(2, lim + 1) if not is_comp[p]]
for p in primes:
    logp = np.log(p)
    pk = p
    while pk <= lim:
        lam_von[pk] = logp
        pk *= p
ns = np.nonzero(lam_von)[0]
logn = np.log(ns)
coef = lam_von[ns] / np.sqrt(ns)

def tent(x):
    return np.maximum(0.0, 1.0 - np.abs(x) / d)   # peak 1 (E-LI-1 F2)

PRIME = np.array([np.sum(coef * tent(logn - m * d)) for m in range(M300 + 1)])
report["PRIME_300"] = float(PRIME[300])

# ARCH via arch.py (dense trapezoid + QUADPACK cos tails; stability 1e-11)
import arch as arch_mod
ARCH = arch_mod.arch_sequence(M300)
A_arith = POLE - PRIME + ARCH

# zero side with first 120 zeros
A_z120 = lib1b.a_zeta_full(G[:120], M300)
dev = np.abs(A_arith - A_z120)
report["maxdev_A_vs_Azeta_120"] = float(dev.max())
report["rmsdev_A_vs_Azeta_120"] = float(np.sqrt(np.mean(dev ** 2)))

# mu_min table, M = w/delta convention (E-LI-1 amendment H6)
mu_table = {}
for w in [2.5, 5, 7.5, 10, 12.5, 15]:
    M = int(round(w / d))
    mu_table[str(w)] = float(eigvalsh(toeplitz(A_arith[:M]))[0])
report["mu_min_table_Mwd"] = mu_table

targets = {"2.5": 1.34e-3, "5": 1.85e-4, "7.5": 6.4e-5,
           "10": 4.4e-5, "12.5": 3.1e-5, "15": 2.2e-5}
ok_mu = all(abs(mu_table[k] - targets[k]) / targets[k] < 0.05 for k in targets)
ok_pp = (abs(POLE[300] - 90.41) < 0.01 and abs(PRIME[300] - 90.59) < 0.01)
ok_dev = (abs(dev.max() - 0.23) < 0.01 and abs(report["rmsdev_A_vs_Azeta_120"] - 0.017) < 0.001)
report["G1_A"] = "PASS" if (ok_mu and ok_pp and ok_dev) else "FAIL"

report["ARCH_stability"] = {str(k): {kk: float(vv) for kk, vv in d.items()} for k, d in arch_mod.verify().items()}

with open("data/g1_report.json", "w") as f:
    json.dump(report, f, indent=1)
print(json.dumps(report, indent=1))
