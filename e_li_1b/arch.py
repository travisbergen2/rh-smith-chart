"""ARCH(m) for Paper 14's A(m): high-accuracy quadrature.

ARCH(m) = (1/pi) int_0^inf B(r) [Re psi(1/4 + ir/2) - log pi] cos(m d r) dr

Scheme (G1 accuracy repair, 2026-07-15, replacing an R=6000 hard-cutoff
trapezoid whose unintegrated tail (~0.017 at m=0) distorted mu_min at the
1e-5 scale):
  [0, R]  : trapezoid, dr = 0.002, R = 2000, psi via central difference
            of scipy loggamma (h = 1e-6, error ~ 1e-13).
  (R, inf): asymptotic kernel Re psi ~ log(r/2) + O(1/r^2); the residual
            O(1/r^2) piece contributes < 1e-8 beyond R = 2000. Tail
            reduces to three cos-weighted integrals
              I(a) = int_R^inf log(r/2pi) cos(a r) / r^2 dr
            evaluated by QUADPACK's oscillatory rule (scipy quad,
            weight='cos') with the analytic value at a = 0.
Verification: dr-halving and R-doubling deltas reported by verify().
"""
import numpy as np
from scipy.special import loggamma
from scipy.integrate import quad

DELTA = 0.05
R_CUT = 2000.0
DR = 0.002
H = 1e-6


def _kern(rr, delta):
    psi_re = ((loggamma(0.25 + 1j * rr / 2 + H)
               - loggamma(0.25 + 1j * rr / 2 - H)) / (2 * H)).real
    B = (2.0 / (delta * rr * rr)) * (1.0 - np.cos(delta * rr))
    return B * (psi_re - np.log(np.pi))


def _I_tail(a, R=R_CUT):
    """int_R^inf log(r/2pi) cos(ar)/r^2 dr."""
    if a == 0.0:
        return (np.log(R / (2 * np.pi)) + 1.0) / R
    f = lambda r: np.log(r / (2 * np.pi)) / (r * r)
    val, _ = quad(f, R, np.inf, weight='cos', wvar=a,
                  limit=400, epsabs=1e-13, epsrel=1e-12)
    return val


def arch_sequence(m_max, delta=DELTA, R=R_CUT, dr=DR):
    # composite trapezoid on [0, R] with endpoint nodes and 1/2-weights
    rr = np.linspace(0.0, R, int(round(R / dr)) + 1)
    kern = np.empty_like(rr)
    kern[1:] = _kern(rr[1:], delta)
    # r = 0 limit: B(0) = delta, psi(1/4) via same central difference
    psi0 = ((loggamma(0.25 + H) - loggamma(0.25 - H)) / (2 * H)).real
    kern[0] = delta * (psi0 - np.log(np.pi))
    wts = np.full_like(rr, dr)
    wts[0] = wts[-1] = dr / 2
    kw = kern * wts
    m = np.arange(0, m_max + 1)
    main = np.empty(m_max + 1)
    B = 20
    for i in range(0, m_max + 1, B):
        mb = m[i:i + B]
        main[i:i + B] = np.cos(np.outer(mb * delta, rr)) @ kw
    main /= np.pi

    # analytic tail
    Ivals = {}
    def I(a):
        a = abs(a)
        if a not in Ivals:
            Ivals[a] = _I_tail(a, R)
        return Ivals[a]
    tail = np.array([
        (2.0 / (np.pi * delta)) *
        (I(mm * delta) - 0.5 * I((mm + 1) * delta) - 0.5 * I((mm - 1) * delta))
        for mm in m])
    return main + tail


def verify(m_probe=(0, 1, 150, 300), delta=DELTA):
    """dr-halving and R-doubling stability check at probe lags."""
    base = arch_sequence(max(m_probe), delta)
    half = arch_sequence(max(m_probe), delta, dr=DR / 2)
    bigR = arch_sequence(max(m_probe), delta, R=2 * R_CUT)
    out = {}
    for mm in m_probe:
        out[mm] = {"dr_half": abs(half[mm] - base[mm]),
                   "R_double": abs(bigR[mm] - base[mm])}
    return out
