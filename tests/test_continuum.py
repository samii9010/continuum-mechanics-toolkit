import numpy as np
import sympy as sp

import cmtoolkit.elasticity as el
from cmtoolkit.kinematics import green_strain, hill_strain, polar_decomposition
from cmtoolkit.tensor_algebra import I_sym, compose4, det, linear_map, sym, tr

RNG = np.random.default_rng(2)


def test_polar_decomposition_and_strains(zero):
    F = np.eye(3) + 0.3 * RNG.normal(size=(3, 3))
    R, U, V = polar_decomposition(F)
    for residual in (R @ U - F, V @ R - F, R.T @ R - np.eye(3), U - U.T, hill_strain(F, 1) - green_strain(F)):
        zero(residual)
    assert det(F) > 0 and np.isclose(det(R), 1.0) and np.all(np.linalg.eigvalsh(U) > 0)
    assert np.isclose(tr(hill_strain(F, 0)), np.log(det(F)))       # tr(ln U) = ln J


def test_polar_decomposition_of_simple_shear_is_exact(zero):
    F = sp.Matrix([[1, sp.Rational(3, 2), 0], [0, 1, 0], [0, 0, 1]])
    R, U, V = polar_decomposition(F)
    assert R == sp.Matrix([[4, 3, 0], [-3, 4, 0], [0, 0, 5]]) / 5
    zero(R @ U - F)


def test_hooke_thermoelasticity_and_compliance(zero):
    lam, mu = el.lame_from_young_poisson(210e3, 0.3)
    eps = sym(RNG.normal(size=(3, 3)))
    zero(linear_map(el.stiffness_isotropic(lam, mu), eps) - el.hooke_stress(eps, lam, mu), tol=1e-8)
    zero(compose4(el.compliance_isotropic(lam, mu), el.stiffness_isotropic(lam, mu)) - I_sym(False))
    zero(el.thermoelastic_stress(1.2e-5 * 50 * np.eye(3), 50, lam, mu, 1.2e-5), tol=1e-8)   # free expansion
    assert np.isclose(el.von_mises_stress(np.diag([7.0, 0, 0])), 7.0) and el.von_mises_stress(3 * np.eye(3)) < 1e-12
