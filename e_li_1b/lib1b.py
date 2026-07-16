"""E-LI-1b pipeline library.

Protocol: doc cmrmpxh7i01bx07ad73p7145m (thresholds frozen 2026-07-15).
Deviations Log entries 1-4 apply (zeros6 table; jitter window gamma<=5000;
G3(b) substitute; Theta window |gamma-gamma*|<=500 and Toeplitz depth grid
step 5).

All statistics are computed from the ZERO SIDE (design decision D1).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import toeplitz, eigvalsh
from scipy.integrate import quad

DELTA = 0.05
DMAX_CAP = 2000
JITTER_SIGMA = 0.05
JITTER_WINDOW = 5000.0     # Deviations Log entry 2
THETA_WINDOW = 500.0       # Deviations Log entry 4a
M_GRID_STEP = 5            # Deviations Log entry 4b
SEEDS_CONF = list(range(6000, 6020))
GAMMA_TARGETS = [14.134725, 98.831194, 999.791572]
EPSILONS = [0.1, 0.05, 0.02, 0.01]
OMEGAS = [0.005, 0.02, 0.05]
PROBE_VS = [0.02, 0.05]


def load_zeros(path="data/zeros6.txt"):
    return np.loadtxt(path)


# ---------------------------------------------------------------------
# Li side: lambda_n via Chebyshev cosine recurrence + RvM tail integral
# ---------------------------------------------------------------------

def phi_of_gamma(g):
    """Angle of the charted on-line zero z = 1 - 1/(1/2+i g)."""
    return np.angle(1.0 - 1.0 / (0.5 + 1j * g))


def lambda_zero_side(gammas, n_max, block=200000):
    """lambda_n = sum over pairs 2(1 - cos(n phi)); exact on-line form."""
    phi = phi_of_gamma(gammas)
    c1 = np.cos(phi)
    S = np.zeros(n_max + 1)          # S[k] = sum cos(k phi)
    S[0] = len(gammas)
    for i in range(0, len(gammas), block):
        c1b = c1[i:i + block]
        prev = np.ones_like(c1b)
        cur = c1b.copy()
        S[1] += cur.sum()
        for k in range(2, n_max + 1):
            prev, cur = cur, 2.0 * c1b * cur - prev
            S[k] += cur.sum()
    n = np.arange(0, n_max + 1)
    lam = 2.0 * (S[0] - S)           # lam[k] for k=0..n_max; lam[0]=0
    return lam[1:]                   # n = 1..n_max


def lambda_tail(T, n_max):
    """RvM-density tail integral, hybrid scheme (G1 accuracy repair,
    2026-07-15): exact integrand via Gauss-Legendre in t = log(gamma)
    on (T, 100T] (smooth, no endpoint singularity), plus the analytic
    quadratic remainder for gamma > 100T where 2(1-cos(n phi)) =
    (n/gamma)^2 to relative accuracy < (n phi)^2/12 ~ 1e-10 at the
    depths used. Replaces a u=1/gamma GL rule whose log endpoint
    singularity underestimated the n=1 tail by ~1.5%."""
    ns = np.arange(1, n_max + 1)
    K = 600
    x, w = np.polynomial.legendre.leggauss(K)
    t0, t1 = np.log(T), np.log(100.0 * T)
    t = 0.5 * (t1 - t0) * x + 0.5 * (t1 + t0)
    wt = w * 0.5 * (t1 - t0)
    g = np.exp(t)
    phi = phi_of_gamma(g)
    dens = np.log(g / (2 * np.pi)) / (2 * np.pi)
    base = dens * g * wt                  # f(g) dg = f(e^t) e^t dt
    out = np.empty(n_max)
    for j, n in enumerate(ns):
        out[j] = np.sum(2.0 * (1.0 - np.cos(n * phi)) * base)
    # analytic remainder gamma > 100T: int n^2/g^2 * dens dg
    Tr = 100.0 * T
    out += ns.astype(float) ** 2 * (np.log(Tr / (2 * np.pi)) + 1.0) / (2 * np.pi * Tr)
    return out


def pair_contrib_li(gamma, n_max, eps=0.0):
    """Contribution to lambda_n of one on-line pair (eps=0) or one
    off-line QUADRUPLE (1/2±eps)±i gamma (eps>0)."""
    n = np.arange(1, n_max + 1)
    if eps == 0.0:
        phi = phi_of_gamma(np.array([gamma]))[0]
        return 2.0 * (1.0 - np.cos(n * phi))
    out = np.zeros(n_max)
    for sb in (+1.0, -1.0):
        rho = 0.5 + sb * eps + 1j * gamma
        z = 1.0 - 1.0 / rho
        # pair with conjugate: 2*Re(1 - z^n)
        logz = np.log(z)
        zn = np.exp(n * logz)
        out += 2.0 * (1.0 - zn).real
    return out


# ---------------------------------------------------------------------
# Toeplitz side: A_zeta(m) via weighted Chebyshev recurrence
# ---------------------------------------------------------------------

def B_tent(r, delta=DELTA):
    r = np.asarray(r, dtype=complex)
    out = np.full(r.shape, complex(delta))
    nz = r != 0
    rn = r[nz]
    out[nz] = (2.0 / (delta * rn * rn)) * (1.0 - np.cos(delta * rn))
    return out


def a_zeta_full(gammas, m_max, delta=DELTA, block=200000, dtype=np.float64):
    """A_zeta(m) = sum 2 B(gamma) cos(m delta gamma), m = 0..m_max."""
    wgt = (2.0 * B_tent(gammas, delta).real).astype(dtype)
    c1 = np.cos(delta * gammas).astype(dtype)
    A = np.zeros(m_max + 1, dtype=dtype)
    A[0] = wgt.sum()
    for i in range(0, len(gammas), block):
        c1b, wb = c1[i:i + block], wgt[i:i + block]
        prev = np.ones_like(c1b)
        cur = c1b.copy()
        A[1] += (wb * cur).sum()
        for k in range(2, m_max + 1):
            prev, cur = cur, 2.0 * c1b * cur - prev
            A[k] += (wb * cur).sum()
    return A


def pair_contrib_A(gamma, m_max, eps=0.0, delta=DELTA):
    """A(m) contribution of one on-line pair or off-line quadruple
    (Paper 14 Prop. 1 converse form)."""
    m = np.arange(0, m_max + 1)
    if eps == 0.0:
        return 2.0 * B_tent(np.array([gamma]), delta)[0].real * np.cos(m * delta * gamma)
    Bc = B_tent(np.array([gamma + 1j * eps]), delta)[0]
    return 4.0 * Bc.real * np.cos(m * delta * gamma) * np.cosh(m * delta * eps)


def mu_curve(A, m_grid):
    """(mu_min, mu_max, trace/m) at each Toeplitz size in m_grid."""
    mins, maxs, trm = [], [], []
    for M in m_grid:
        T = toeplitz(A[:M])
        ev = eigvalsh(T)
        mins.append(ev[0]); maxs.append(ev[-1]); trm.append(np.trace(T) / M)
    return np.array(mins), np.array(maxs), np.array(trm)


# ---------------------------------------------------------------------
# Theta_omega arm (windowed; Deviations entry 4a)
# ---------------------------------------------------------------------

def theta_peak(gammas_window, omega, gamma_star, v, x_half=3.0, dx=0.005,
               extra_online=None, quad_eps=None, drop_nearest=None):
    """Peak |Theta_omega| on probe line Im z = v, x in [-g*-3, -g*+3].

    gammas_window: on-line zeros with |gamma - gamma*| <= THETA_WINDOW
    (already jittered as needed). extra_online: list of extra on-line
    gammas (double-pair control adds one). quad_eps: if not None, adds
    the off-line quadruple factors at (eps, gamma_star). drop_nearest:
    height whose nearest window zero is removed first.
    """
    g = np.asarray(gammas_window, dtype=float)
    if drop_nearest is not None and len(g):
        g = np.delete(g, int(np.argmin(np.abs(g - drop_nearest))))
    x = np.arange(-gamma_star - x_half, -gamma_star + x_half + dx / 2, dx)
    z = x + 1j * v
    out = np.ones(z.shape, dtype=complex)
    B = 50000   # chunk zeros to bound memory
    for i in range(0, len(g), B):
        gb = g[i:i + B]
        for sg in (+1.0, -1.0):
            w = sg * gb[None, :] + z[:, None]
            out *= np.prod((omega + 1j * w) / (-omega + 1j * w), axis=1)
    if extra_online:
        for ge in extra_online:
            for sg in (+1.0, -1.0):
                w = sg * ge + z
                out *= (omega + 1j * w) / (-omega + 1j * w)
    if quad_eps is not None:
        eps = quad_eps
        for sg in (+1.0, -1.0):
            w = sg * gamma_star + z
            for sb in (+1.0, -1.0):
                out *= (omega + sb * eps + 1j * w) / (-omega + sb * eps + 1j * w)
    return float(np.abs(out).max())


# ---------------------------------------------------------------------
# Null ensemble machinery (Deviations entry 2: windowed jitter)
# ---------------------------------------------------------------------

def jittered_head(gammas_head, seed):
    rng = np.random.default_rng(seed)
    return gammas_head + rng.normal(0.0, JITTER_SIGMA, size=len(gammas_head))


def detection_depth(stat, band_mean, band_sd, grid, also_below_zero=True):
    """First grid depth where stat < band_mean - 3 sd (or < 0)."""
    thresh = band_mean - 3.0 * band_sd
    for i, d in enumerate(grid):
        fire = stat[i] < thresh[i] or (also_below_zero and stat[i] < 0)
        if fire:
            return int(d)
    return None


def mu_min_scan(A, m_grid):
    """(mu_min, trace/m) at each size; smallest eigenvalue only (evr)."""
    from scipy.linalg import eigh
    mins, trm = [], []
    for M in m_grid:
        T = toeplitz(A[:M])
        ev = eigh(T, eigvals_only=True, subset_by_index=[0, 0], driver="evr")
        mins.append(ev[0]); trm.append(np.trace(T) / M)
    return np.array(mins), np.array(trm)
