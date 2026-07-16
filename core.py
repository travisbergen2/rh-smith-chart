"""
RH Smith Chart — core library.

A "Smith chart for RH": the Möbius map underlying Li's criterion,
    z = 1 - 1/s   (equivalently s = 1/(1-z)),
sends the critical line Re(s) = 1/2 to the unit circle |z| = 1, with
|z| = |s-1|/|s|: the half-plane Re(s) > 1/2 maps to the INTERIOR of
the disk, Re(s) < 1/2 to the exterior (computed and pinned by a unit
test — do not trust intuition here). This is the exact analogue of the classical
Smith chart's Cayley map Gamma = (Z - 1)/(Z + 1), which sends the
right half-plane (positive-real impedances) to the unit disk.

Epistemic labels (Claim Ledger discipline):
  [T]  z = 1 - 1/s maps Re(s)=1/2 bijectively onto |z|=1 minus {1}.
       Elementary Möbius geometry; verified numerically in tests.
  [T]  Li's criterion: RH <=> lambda_n >= 0 for all n >= 1
       (Li 1997; Bombieri–Lagarias 1999). lambda_n computed here from
       the zero side, TRUNCATED to the first N zeros — truncation is
       an instrument approximation, labeled as such everywhere.
  [T]  A(m) definitions transcribed from IMM Paper 14 (Zenodo
       10.5281/zenodo.20684022) with the delta-consistent tent
       transform B(r) = (2/(delta r^2))(1 - cos(delta r)) adopted per
       the E-LI-1 F3 adjudication (factor-delta erratum).
  [A]  All network-theory language ("impedance", "lossless",
       "reflection coefficient") is ANALOGY, mapped through the
       dictionary in the paper draft. Nothing here proves or tests RH.

Sources: Odlyzko zero tables (data/zeros1.txt, first 100,000 zeros).
"""

from __future__ import annotations

import numpy as np

DELTA_DEFAULT = 0.05  # Paper 14 primary delta


# ----------------------------------------------------------------------
# Zero data
# ----------------------------------------------------------------------

def load_zeros(path: str = "data/zeros1.txt", n: int | None = None) -> np.ndarray:
    """Positive ordinates gamma of nontrivial zeros (Odlyzko table)."""
    g = np.loadtxt(path)
    if n is not None:
        g = g[:n]
    return g


# ----------------------------------------------------------------------
# Conformal maps  [T]
# ----------------------------------------------------------------------

def li_map(s: np.ndarray | complex) -> np.ndarray | complex:
    """z = 1 - 1/s. Sends s = 1/2 + i*gamma to the unit circle."""
    return 1.0 - 1.0 / s


def li_map_inverse(z: np.ndarray | complex) -> np.ndarray | complex:
    """s = 1/(1-z)."""
    return 1.0 / (1.0 - z)


def smith_map(Z: np.ndarray | complex) -> np.ndarray | complex:
    """Classical Smith chart: Gamma = (Z-1)/(Z+1), RHP -> unit disk."""
    return (Z - 1.0) / (Z + 1.0)


def zeros_to_disk(gammas: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Map zeros rho = sigma + i*gamma through the Li map."""
    rho = sigma + 1j * gammas
    return li_map(rho)


# ----------------------------------------------------------------------
# Smith-style grid in the z-plane  [T: Möbius images of lines/circles]
# ----------------------------------------------------------------------

def sigma_circle(sigma: float, num: int = 720) -> np.ndarray:
    """Image under li_map of the vertical line Re(s) = sigma.

    A Möbius image of a line is a circle (or line); returned as a
    sampled closed curve in the z-plane. This is the RH analogue of the
    Smith chart's constant-resistance circle family.
    """
    t = np.linspace(-4000.0, 4000.0, num)
    # tan reparametrization concentrates samples near the real axis
    u = np.tan(np.linspace(-np.pi / 2 + 1e-6, np.pi / 2 - 1e-6, num))
    s = sigma + 1j * u
    return li_map(s)


def gamma_arc(gamma: float, sig_min: float = 1e-3, sig_max: float = 60.0,
              num: int = 400) -> np.ndarray:
    """Image under li_map of the horizontal line Im(s) = gamma
    (constant-height arc — the analogue of constant-reactance arcs)."""
    sig = np.linspace(sig_min, sig_max, num)
    s = sig + 1j * gamma
    return li_map(s)


# ----------------------------------------------------------------------
# Li coefficients, zero side, truncated  [T definition; truncated = instrument]
# ----------------------------------------------------------------------

def li_lambda(gammas: np.ndarray, n_max: int, sigma: float = 0.5,
              beta_pairs: list[tuple[float, float]] | None = None) -> np.ndarray:
    """lambda_n = sum_rho [1 - (1 - 1/rho)^n], unnormalized
    (Bombieri–Lagarias convention), truncated to the supplied zeros.

    Zeros are used in symmetric quadruples: for each gamma we include
    rho and 1-rho-bar pairing, i.e. contribution 2*Re[1 - (1-1/rho)^n]
    with rho = sigma + i*gamma; the functional-equation partner at
    1 - sigma is included when sigma != 1/2 via beta_pairs.

    beta_pairs: optional list of (beta, gamma) for planted off-line
    quadruples (1/2 +- beta +- i*gamma), replacing nothing — callers
    manage the multiset.

    Truncation note [labeled]: the true lambda_n sums over ALL zeros;
    with the first 100k zeros the truncation error for n <= ~500 is
    small (E-LI-1 gate G2 measured <0.5% at these depths for the
    normalized statistic) but this function makes NO exactness claim.
    """
    n = np.arange(1, n_max + 1)
    rho = 0.5 + 1j * gammas if sigma == 0.5 else sigma + 1j * gammas
    w = 1.0 - 1.0 / rho                       # = z(rho): the Li-map image!
    # lambda_n = sum over zeros of (1 - w^n), paired: 2*Re for gamma>0
    # accumulate in blocks to bound memory
    lam = np.zeros(n_max, dtype=float)
    B = 20000
    for i in range(0, len(w), B):
        wi = w[i:i + B]
        # wi**n for all n: (len(wi), n_max) — use cumulative products
        P = np.empty((len(wi), n_max), dtype=complex)
        P[:, 0] = wi
        for k in range(1, n_max):
            P[:, k] = P[:, k - 1] * wi
        lam += (2.0 * (1.0 - P.real).sum(axis=0)) if sigma == 0.5 else \
               (2.0 * (1.0 - P).real.sum(axis=0))
    if beta_pairs:
        for beta, gamma in beta_pairs:
            for sg in (+1.0, -1.0):
                rho_b = 0.5 + sg * beta + 1j * gamma
                wb = 1.0 - 1.0 / rho_b
                pw = wb ** n
                lam += 2.0 * (1.0 - pw).real   # pairs gamma>0 with gamma<0
    return lam


# ----------------------------------------------------------------------
# Paper 14 A(m), zero side  [T: transcription; erratum convention]
# ----------------------------------------------------------------------

def B_tent(r: np.ndarray | complex, delta: float = DELTA_DEFAULT):
    """delta-consistent tent transform:
    B(r) = (2/(delta r^2)) (1 - cos(delta r)),  B(0) = delta.
    Entire; accepts complex argument (off-line zeros)."""
    r = np.asarray(r, dtype=complex)
    out = np.full(r.shape, delta, dtype=complex)
    nz = r != 0
    rn = r[nz]
    out[nz] = (2.0 / (delta * rn * rn)) * (1.0 - np.cos(delta * rn))
    if out.imag.max() if out.size else 0:  # pragma: no cover - guard only
        pass
    return out


def a_zeta(gammas: np.ndarray, m_max: int, delta: float = DELTA_DEFAULT,
           beta_pairs: list[tuple[float, float]] | None = None) -> np.ndarray:
    """Zero-side Toeplitz sequence A_zeta(m) = sum_{gamma>0} 2 B(gamma) cos(m delta gamma)
    for m = 0..m_max (Paper 14 §4). Off-line quadruples contribute
    4 Re[B(gamma + i beta)] cos(m delta gamma) cosh(m delta beta)."""
    m = np.arange(0, m_max + 1)
    Bg = B_tent(gammas, delta).real          # real for real gamma
    # A(m) = sum 2 B(g) cos(m delta g): do as matrix product in blocks
    A = np.zeros(m_max + 1)
    Bsz = 20000
    for i in range(0, len(gammas), Bsz):
        g = gammas[i:i + Bsz]
        A += (2.0 * B_tent(g, delta).real[None, :] *
              np.cos(np.outer(m, g) * delta)).sum(axis=1)
    if beta_pairs:
        for beta, gamma in beta_pairs:
            Bc = B_tent(np.array([gamma + 1j * beta]), delta)[0]
            A += 4.0 * Bc.real * np.cos(m * delta * gamma) * np.cosh(m * delta * beta)
    return A


def toeplitz_mu_min(A: np.ndarray, M: int) -> float:
    """Minimum eigenvalue of T_M = [A(|i-j|)]_{i,j<M}."""
    from scipy.linalg import toeplitz, eigvalsh
    T = toeplitz(A[:M])
    return float(eigvalsh(T)[0])


# ----------------------------------------------------------------------
# Map verification helpers (used by tests and the paper)  [T]
# ----------------------------------------------------------------------

def critical_line_to_circle_residual(gammas: np.ndarray) -> float:
    """max | |z|-1 | over zeros mapped from the critical line. Should be ~0."""
    z = zeros_to_disk(gammas)
    return float(np.abs(np.abs(z) - 1.0).max())


def offline_displacement(beta: float, gamma: float) -> float:
    """|z| - 1 for a zero at 1/2 + beta + i*gamma: radial signature of a
    line violation on the chart."""
    z = li_map(0.5 + beta + 1j * gamma)
    return float(abs(z) - 1.0)


# ----------------------------------------------------------------------
# Suzuki Theta_omega inner-function gauge  [T: Suzuki arXiv:1204.1827
# for the object; truncated Blaschke product = instrument approximation]
# ----------------------------------------------------------------------

def theta_omega(z: np.ndarray, gammas: np.ndarray, omega: float,
                beta_pairs: list[tuple[float, float]] | None = None,
                drop_gamma: float | None = None) -> np.ndarray:
    """Truncated zero-side model of Suzuki's
        Theta_omega(z) = xi(1/2 - omega - iz) / xi(1/2 + omega - iz).

    Via the Hadamard pairing, each on-line zero pair +-gamma contributes
    Blaschke-type factors
        (omega + i(gamma + z)) / (-omega + i(gamma + z)),
    taken over both signs of gamma. Unconditionally |Theta_omega| = 1 on
    the real axis; under RH it is (meromorphic) inner: |Theta_omega|<=1
    on the upper half-plane [T: Suzuki 2012 states the innerness<->RH
    connection]. An off-line zero at 1/2 + beta + i*gamma with beta >
    omega contributes a POLE in the upper half-plane at
    z = -gamma + i(beta - omega), destroying innerness -- the gauge.

    Truncation note [labeled]: product over the first len(gammas) zero
    pairs only; an instrument approximation of the true ratio of xi's,
    adequate for LOCAL innerness probes near planted violations (distant
    factors are ~1 + O(omega/gamma)), no exactness claim.

    drop_gamma: remove the on-line pair nearest this height (for
    replace-by-quadruple planted configurations).
    beta_pairs: list of (beta, gamma) off-line quadruples; contributes
    the four zeros (1/2 +- beta +- i gamma), i.e. factors from the two
    beta-shifted "gamma values" gamma -+ i*beta at both signs of gamma.
    """
    z = np.asarray(z, dtype=complex)
    g = gammas
    if drop_gamma is not None:
        idx = int(np.argmin(np.abs(g - drop_gamma)))
        g = np.delete(g, idx)
    out = np.ones(z.shape, dtype=complex)
    B = 20000
    for i in range(0, len(g), B):
        gi = g[i:i + B]
        for sg in (+1.0, -1.0):
            w = sg * gi[None, ...] + z[..., None]     # gamma + z
            out *= np.prod((omega + 1j * w) / (-omega + 1j * w), axis=-1)
    if beta_pairs:
        for beta, gamma in beta_pairs:
            # zeros rho = 1/2 +- beta + i(+-gamma): in the factor form the
            # pair at +gamma with sigma-offset +-beta gives w = gamma + z
            # and numerator/denominator omega +- beta shifts:
            for sg in (+1.0, -1.0):                    # sign of gamma
                w = sg * gamma + z
                for sb in (+1.0, -1.0):                # sign of beta
                    out *= (omega + sb * beta + 1j * w) / (-omega + sb * beta + 1j * w)
    return out


def theta_innerness_margin(gammas: np.ndarray, omega: float,
                           x_grid: np.ndarray, v: float,
                           beta_pairs=None, drop_gamma=None) -> np.ndarray:
    """|Theta_omega| along the horizontal probe line Im z = v > 0.
    Innerness => values <= 1 (up to truncation error)."""
    z = x_grid + 1j * v
    return np.abs(theta_omega(z, gammas, omega,
                              beta_pairs=beta_pairs, drop_gamma=drop_gamma))
