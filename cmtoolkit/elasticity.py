"""Isotropic linear (thermo-)elasticity: σ = λ tr(ε) I + 2μ ε − 3K α_T ΔT I."""

from __future__ import annotations

import numpy as np
import sympy as sp

from .tensor_algebra import dev, identity2, is_symbolic, isotropic_tensor4, norm, tr


def lame_from_young_poisson(E, nu) -> tuple:
    """(λ, μ) = (Eν/((1+ν)(1−2ν)), E/(2(1+ν)))."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def bulk_modulus(lam, mu):
    """K = λ + 2μ/3."""
    return lam + 2 * mu / 3


def stiffness_isotropic(lam, mu):
    """C = λ I ⊗ I + 2μ I_sym = 3K I_sph + 2μ I_dev."""
    return isotropic_tensor4(3 * bulk_modulus(lam, mu), 2 * mu)


def compliance_isotropic(lam, mu):
    """S = C⁻¹ on Sym = 1/(3K) I_sph + 1/(2μ) I_dev."""
    return isotropic_tensor4(1 / (3 * bulk_modulus(lam, mu)), 1 / (2 * mu))


def hooke_stress(eps, lam, mu):
    """σ = λ tr(ε) I + 2μ ε."""
    return lam * tr(eps) * identity2(is_symbolic(eps, lam, mu)) + 2 * mu * eps


def thermoelastic_stress(eps, delta_T, lam, mu, alpha_T):
    """σ = λ tr(ε) I + 2μ ε − 3K α_T ΔT I = C[ε − α_T ΔT I]."""
    thermal = 3 * bulk_modulus(lam, mu) * alpha_T * delta_T
    return hooke_stress(eps, lam, mu) - thermal * identity2(is_symbolic(eps, delta_T, lam, mu, alpha_T))


def von_mises_stress(sigma):
    """σ_eq = √(3/2) ‖σ'‖."""
    return sp.sqrt(sp.Rational(3, 2)) * norm(dev(sigma)) if is_symbolic(sigma) else np.sqrt(1.5) * norm(dev(sigma))
