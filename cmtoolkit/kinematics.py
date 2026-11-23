"""Finite-deformation kinematics: right Cauchy–Green tensor, polar decomposition F = RU = VR, Hill strains."""

from __future__ import annotations

import numpy as np
import sympy as sp

from .tensor_algebra import identity2, is_symbolic, simplify_tensor, tensor_function


def right_cauchy_green(F):
    """C = Fᵀ F = U²."""
    return F.T @ F


def polar_decomposition(F) -> tuple:
    """F = R U = V R with U = √C (spectral), R = F U⁻¹ ∈ SO(3) for det F > 0, V = R U Rᵀ. Returns (R, U, V)."""
    symbolic = is_symbolic(F)
    root = sp.sqrt if symbolic else np.sqrt
    C = right_cauchy_green(F)
    U = tensor_function(C, root)
    R = F @ tensor_function(C, lambda c: 1 / root(c))
    V = R @ U @ R.T
    return tuple(simplify_tensor(T) for T in (R, U, V)) if symbolic else (R, U, V)


def green_strain(F):
    """Green–Lagrange strain E = (C − I)/2."""
    return (right_cauchy_green(F) - identity2(is_symbolic(F))) / 2


def hill_strain(F, m):
    """Hill strain E^(m) = (U^{2m} − I)/(2m), m = 0: Hencky ln U; evaluated spectrally from C.

    m = 1: Green, m = 1/2: Biot U − I, m = 0: Hencky (tr ln U = ln J).
    """
    symbolic = is_symbolic(F, m)
    log = sp.log if symbolic else np.log
    f = (lambda c: log(c) / 2) if m == 0 else (lambda c: (c**m - 1) / (2 * m))
    E = tensor_function(right_cauchy_green(F), f)
    return simplify_tensor(E) if symbolic else E
