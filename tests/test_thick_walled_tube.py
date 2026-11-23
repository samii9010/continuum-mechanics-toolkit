import numpy as np
import sympy as sp

from cmtoolkit.elasticity import lame_from_young_poisson
from cmtoolkit.thick_walled_tube import solve_thick_walled_tube, stationary_radial_temperature

r = sp.Symbol("r", positive=True)
R_I, R_A, P_I, E, NU, ALPHA = sp.Rational(1, 10), sp.Rational(3, 20), 50 * 10**6, 210 * 10**9, sp.Rational(3, 10), 12e-6
DELTA_T = stationary_radial_temperature(r, R_I, R_A, 350, 300) - 300          # outer surface at T_0


def test_lame_solution_and_closed_ends(zero):
    r_i, r_a, p_i, p_a, E_, nu = sp.symbols("r_i r_a p_i p_a E nu", positive=True)
    s = solve_thick_walled_tube(r_i, r_a, p_i, p_a, E_, nu)
    A, B = (p_i * r_i**2 - p_a * r_a**2) / (r_a**2 - r_i**2), (p_i - p_a) * r_i**2 * r_a**2 / (r_a**2 - r_i**2)
    zero(s["sigma_rr"] - (A - B / r**2))
    zero(s["sigma_pp"] - (A + B / r**2))
    closed = solve_thick_walled_tube(r_i, r_a, p_i, 0, E_, nu, plane="closed_ends")
    zero(closed["sigma_zz"] - p_i * r_i**2 / (r_a**2 - r_i**2))


def test_thermal_part_matches_classical_closed_form():
    thermal = solve_thick_walled_tube(R_I, R_A, 0, 0, E, NU, sp.Rational(12, 10**6), DELTA_T)
    ri, ra, rr = float(R_I), float(R_A), np.linspace(float(R_I), float(R_A), 9)
    c, k = ALPHA * E * 50 / (2 * 0.7 * np.log(ra / ri)), ri**2 / (ra**2 - ri**2) * np.log(ra / ri)
    expected = {"sigma_rr": c * (-np.log(ra / rr) - k * (1 - ra**2 / rr**2)),
                "sigma_pp": c * (1 - np.log(ra / rr) - k * (1 + ra**2 / rr**2))}
    for key, values in expected.items():
        np.testing.assert_allclose(sp.lambdify(r, thermal[key])(rr), values, atol=1e-3)       # Pa


def test_combined_load_vs_finite_differences():
    """BCs, equilibrium, and an independent FD solution of u'' + u'/r − u/r² = κ_T ΔT' with traction BCs."""
    tube = solve_thick_walled_tube(R_I, R_A, P_I, 0, E, NU, sp.Rational(12, 10**6), DELTA_T)
    assert np.isclose(float(tube["sigma_rr"].subs(r, R_I)), -P_I) and abs(float(tube["sigma_rr"].subs(r, R_A))) < 1e-3
    n = 801
    rr = np.linspace(0.1, 0.15, n)
    assert np.max(np.abs(sp.lambdify(r, tube["equilibrium_residual"])(rr))) < 1e-9 * P_I / 0.1
    lam, mu = lame_from_young_poisson(210e9, 0.3)
    K, kappa, h = lam + 2 * mu / 3, ALPHA * 1.3 / 0.7, rr[1] - rr[0]
    dT, ddT = sp.lambdify(r, DELTA_T), sp.lambdify(r, DELTA_T.diff(r))
    A, b = np.zeros((n, n)), kappa * ddT(rr)
    for k in range(1, n - 1):
        A[k, k - 1:k + 2] = [1 / h**2 - 1 / (2 * h * rr[k]), -2 / h**2 - 1 / rr[k]**2, 1 / h**2 + 1 / (2 * h * rr[k])]
    for k, stencil, cols, p in ((0, [-3, 4, -1], slice(0, 3), P_I), (n - 1, [1, -4, 3], slice(n - 3, n), 0.0)):
        A[k, cols] = (lam + 2 * mu) * np.array(stencil) / (2 * h)          # σ_rr = (λ+2μ)u' + λu/r − 3Kα_T ΔT
        A[k, k] += lam / rr[k]
        b[k] = -p + 3 * K * ALPHA * dT(rr[k])
    u_exact = sp.lambdify(r, tube["u_r"])(rr)
    assert np.max(np.abs(np.linalg.solve(A, b) - u_exact)) < 1e-5 * np.max(np.abs(u_exact))
