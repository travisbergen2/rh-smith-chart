"""Unit tests for the RH Smith chart core library.

Run: python3 -m pytest test_core.py -q   (or python3 test_core.py)
"""
import numpy as np
import core


def test_li_map_critical_line_to_unit_circle():
    # [T] rho = 1/2 + i*gamma  =>  |1 - 1/rho| = 1 exactly.
    g = np.array([14.134725142, 21.022039639, 1000.0, 1e6])
    z = core.zeros_to_disk(g)
    assert np.abs(np.abs(z) - 1.0).max() < 1e-12


def test_li_map_orientation():
    # |z| = |s-1|/|s|: sigma > 1/2 means s closer to 1 than to 0, so
    # zeros to the RIGHT of the line land INSIDE the circle and zeros
    # to the LEFT land OUTSIDE. (Computed; corrected from the naive
    # first guess — the test exists precisely to pin orientation.)
    assert core.offline_displacement(+0.1, 14.0) < 0
    assert core.offline_displacement(-0.1, 14.0) > 0


def test_li_map_inverse_roundtrip():
    s = np.array([0.5 + 14.1j, 2.0 + 3.0j, 0.1 - 5.0j])
    assert np.abs(core.li_map_inverse(core.li_map(s)) - s).max() < 1e-12


def test_smith_map_rhp_to_disk():
    # Classical Smith chart sanity: Re(Z)>0 => |Gamma|<1; Re(Z)=0 => |Gamma|=1.
    Z = np.array([1.0 + 0j, 0.5 + 2j, 3.0 - 1j])
    assert (np.abs(core.smith_map(Z)) < 1).all()
    Zj = np.array([1j * t for t in (-5, -1, 0.5, 7)])
    assert np.abs(np.abs(core.smith_map(Zj)) - 1).max() < 1e-12


def test_lambda1_matches_published():
    # G1-style reproduction: lambda_1 = 0.0230957... (Keiper-Li).
    # Truncated zero-side sum with 100k zeros: lambda_1 known to converge
    # slowly; accept the E-LI-1-measured truncated accuracy envelope.
    g = core.load_zeros(n=100000)
    lam = core.li_lambda(g, 5)
    # zero-side truncated lambda_1 with 100k zeros lands within ~2e-3
    # of 0.0230957 (tail integral omitted by design in this instrument).
    assert abs(lam[0] - 0.0230957089) < 5e-3


def test_lambda_positivity_over_depth():
    # On the unmodified (RH-true so far) zero set, truncated lambda_n
    # should be positive over the instrument's display range.
    g = core.load_zeros(n=100000)
    lam = core.li_lambda(g, 200)
    assert (lam > 0).all()


def test_B_tent_at_zero_and_positivity():
    d = 0.05
    B = core.B_tent(np.array([0.0, 1.0, 14.13, 500.0]), d).real
    assert abs(B[0] - d) < 1e-15
    assert (B >= 0).all()


def test_a_zeta_toeplitz_positive_on_true_zeros():
    # Paper 14 §4 behaviour: mu_min(T_M) small but positive on the
    # genuine zero set at delta = 0.05.
    g = core.load_zeros(n=100000)
    A = core.a_zeta(g, 300)
    mu = core.toeplitz_mu_min(A, 301)
    assert mu > 0
    assert mu < 1e-3   # small positive, matching Paper 14's scale


def test_planted_offline_zero_shows_on_chart_and_in_A():
    # A planted quadruple at beta=0.1, gamma≈98.83 must (a) leave the
    # unit circle on the chart, (b) eventually drive mu_min negative
    # (E-LI-1 found detection at M≈272).
    g = core.load_zeros(n=100000)
    gamma_t = 98.831194
    # (a) chart displacement
    assert abs(core.offline_displacement(0.1, gamma_t)) > 1e-5
    # (b) remove the on-line pair nearest gamma_t, add off-line quadruple
    idx = int(np.argmin(np.abs(g - gamma_t)))
    g_mod = np.delete(g, idx)
    A = core.a_zeta(g_mod, 300, beta_pairs=[(0.1, gamma_t)])
    mu_300 = core.toeplitz_mu_min(A, 301)
    assert mu_300 < 0   # indefiniteness fires by M=301 (E-LI-1: ~272)


# ---- Theta_omega gauge tests (appended with the Suzuki extension) ----

def test_theta_unimodular_on_real_axis():
    # Unconditionally |Theta_omega(x)| = 1 on the real axis (functional
    # equation + conjugation symmetry); each Blaschke factor is exactly
    # unimodular for real w, so truncation cannot break this.
    g = core.load_zeros(n=20000)
    x = np.linspace(-120.0, 120.0, 41)
    vals = core.theta_innerness_margin(g, omega=0.05, x_grid=x, v=0.0)
    assert np.abs(vals - 1.0).max() < 1e-9


def test_theta_inner_on_true_zeros():
    # Innerness on the probe line Im z = 0.05 for the true zero set:
    # |Theta_omega| <= 1 (small truncation slack allowed).
    g = core.load_zeros(n=100000)
    x = np.linspace(-110.0, -90.0, 81)   # window around gamma ~ 98.8
    vals = core.theta_innerness_margin(g, omega=0.05, x_grid=x, v=0.05)
    assert vals.max() <= 1.0 + 1e-6


def test_theta_planted_pole_breaks_innerness():
    # Planted quadruple beta=0.1 > omega=0.05 at gamma~98.83 puts a pole
    # at z = -gamma + i(beta-omega): |Theta_omega| must exceed 1 near it.
    # NOTE (instrument lesson, kept deliberately): a first version of
    # this test probed with grid spacing 0.25 and MISSED the pole —
    # the innerness violation has width ~ (beta - omega) = 0.05. The
    # Theta gauge is LOCAL; probe grids must resolve that scale.
    g = core.load_zeros(n=100000)
    gamma_t = 98.831194
    x = np.linspace(-99.2, -98.4, 161)   # spacing 0.005 << beta-omega
    vals = core.theta_innerness_margin(
        g, omega=0.05, x_grid=x, v=0.05,
        beta_pairs=[(0.1, gamma_t)], drop_gamma=gamma_t)
    assert vals.max() > 10.0   # near-pole blowup, decisive (measured ~41.8)


def test_theta_beta_below_omega_stays_inner():
    # Control: beta=0.02 < omega=0.05 keeps all poles in the lower
    # half-plane; the probe line must stay <= 1. (The gauge detects
    # beta > omega only — a labeled sensitivity floor, not a bug.)
    g = core.load_zeros(n=100000)
    gamma_t = 98.831194
    x = np.linspace(-110.0, -90.0, 81)
    vals = core.theta_innerness_margin(
        g, omega=0.05, x_grid=x, v=0.05,
        beta_pairs=[(0.02, gamma_t)], drop_gamma=gamma_t)
    assert vals.max() <= 1.0 + 1e-6


if __name__ == "__main__":
    import sys, traceback
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for f in fns:
        try:
            f()
            print(f"PASS {f.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {f.__name__}")
            traceback.print_exc()
    sys.exit(1 if failed else 0)
