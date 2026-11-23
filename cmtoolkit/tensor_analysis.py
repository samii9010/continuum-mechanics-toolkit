"""Differential operators (symbolic) and generalized gradients of invariants.

grad is the right gradient, it appends the derivative index: (grad u)_ij = ∂u_i/∂x_j, du = (grad u) dx.
div contracts the derivative index with the last index: (div A)_i = ∂A_ij/∂x_j.
Cylindrical operators take and return *physical* components w.r.t. {e_r, e_φ, e_z}; they are
evaluated with the chain rule grad_x(·) = grad_Θ(·) (∂Θ/∂x), so no Christoffel symbols are needed.
"""

from __future__ import annotations

import string
from typing import Sequence

import sympy as sp

from .tensor_algebra import (
    I_dev, I_sym, _from_array, _to_array, cofactor, dev, einsum, identity2, is_symbolic, norm, order,
    permutation_tensor, simplify_tensor, tensor_product, tr,
)

# --------------------------------------------------------------------------- cartesian operators


def grad(field, coords: Sequence[sp.Symbol]):
    """Right gradient of a field of any order: (grad T)_{…k} = ∂T_{…}/∂x_k."""
    k = order(field)
    if k == 0:
        return sp.Matrix([sp.diff(field, c) for c in coords])
    derivative = sp.derive_by_array(_to_array(field), list(coords))      # [m, …] = ∂T_…/∂x_m
    return _from_array(sp.permutedims(derivative, list(range(1, k + 1)) + [0]))


def div(field, coords: Sequence[sp.Symbol]):
    """Divergence of a field of order ≥ 1: div u = ∂u_i/∂x_i, (div A)_i = ∂A_ij/∂x_j."""
    free = string.ascii_lowercase[:order(field) - 1]
    return einsum(f"{free}zz->{free}", grad(field, coords))


def rot(u, coords: Sequence[sp.Symbol]) -> sp.Matrix:
    """Curl (rot u)_i = ε_ijk ∂u_k/∂x_j, i.e. rot u = −ε[grad u]."""
    return -1 * einsum("ijk,jk->i", permutation_tensor(), grad(u, coords))


def laplace(field, coords: Sequence[sp.Symbol]):
    """Laplacian Δ(·) = div grad(·)."""
    return div(grad(field, coords), coords)


# --------------------------------------------------------------------------- cylindrical coordinates


def cylindrical_coordinates() -> tuple[sp.Symbol, sp.Symbol, sp.Symbol]:
    """Coordinate symbols (r, φ, z) with r > 0; x = r cos φ e_1 + r sin φ e_2 + z e_3."""
    return sp.Symbol("r", positive=True), sp.Symbol("varphi", real=True), sp.Symbol("z", real=True)


def _cartesian_grad(field_phys, coords):
    """Cartesian gradient of a field given in physical cylindrical components, plus the frame R."""
    r, phi, z = coords
    R = sp.Matrix([[sp.cos(phi), -sp.sin(phi), 0], [sp.sin(phi), sp.cos(phi), 0], [0, 0, 1]])  # [e_r e_φ e_z]
    dtheta_dx = simplify_tensor(sp.Matrix([r * sp.cos(phi), r * sp.sin(phi), z]).jacobian(coords).inv())
    k = order(field_phys)
    cart = field_phys if k == 0 else (R @ field_phys if k == 1 else R @ field_phys @ R.T)
    letters = string.ascii_lowercase[:k]
    return einsum(f"{letters}m,mj->{letters}j", grad(cart, coords), dtheta_dx), R


def _to_physical(T, R):
    k = order(T)
    return T if k == 0 else simplify_tensor(R.T @ T if k == 1 else R.T @ T @ R)


def grad_cylindrical(field, coords: Sequence[sp.Symbol]):
    """grad of a scalar or vector field, in physical cylindrical components (in and out)."""
    G, R = _cartesian_grad(field, coords)
    return _to_physical(G, R)


def div_cylindrical(field, coords: Sequence[sp.Symbol]):
    """div of a vector or 2nd-order tensor field, in physical cylindrical components (in and out).

    For σ = diag(σ_rr(r), σ_φφ(r), σ_zz): (div σ)_r = σ_rr' + (σ_rr − σ_φφ)/r.
    """
    G, R = _cartesian_grad(field, coords)
    free = string.ascii_lowercase[:order(field) - 1]
    return sp.simplify(_to_physical(einsum(f"{free}zz->{free}", G), R))


def laplace_cylindrical(phi: sp.Expr, coords: Sequence[sp.Symbol]) -> sp.Expr:
    """Δφ = div grad φ = φ'' + φ'/r + φ_,φφ / r² + φ_,zz."""
    return sp.simplify(div_cylindrical(grad_cylindrical(phi, coords), coords))


# --------------------------------------------------------------------------- generalized gradients


def d_first_invariant(A):
    """∂I_A/∂A = I."""
    return identity2(is_symbolic(A))


def d_second_invariant(A):
    """∂II_A/∂A = I_A I − Aᵀ."""
    return tr(A) * identity2(is_symbolic(A)) - A.T


def d_third_invariant(A):
    """∂III_A/∂A = III_A A⁻ᵀ = cof A."""
    return cofactor(A)


def d_normalized_tensor(sigma, deviatoric: bool = False):
    """Derivative of N = σ/‖σ‖ w.r.t. a symmetric σ: (I_sym − N ⊗ N)/‖σ‖.

    With ``deviatoric=True``, the von Mises flow direction N' = σ'/‖σ'‖: ∂N'/∂σ = (I_dev − N' ⊗ N')/‖σ'‖.
    """
    base = dev(sigma) if deviatoric else sigma
    N = base / norm(base)
    projector = (I_dev if deviatoric else I_sym)(is_symbolic(sigma))
    return (projector - tensor_product(N, N)) / norm(base)
