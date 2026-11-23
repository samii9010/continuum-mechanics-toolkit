"""Tensor algebra in a 3D orthonormal basis, for sympy (symbolic) and numpy (numeric) tensors.

Conventions: A = A_ij e_i ⊗ e_j (summation convention), A · B = A_ij B_ij, C[A] = C_ijkl A_kl,
sph(A) = (tr A / 3) I, dev(A) = A − sph(A). Code indices are 0-based, docstring indices 1-based.
Every function returns the backend of its input: sympy Matrix (order 1, 2) / Array (order ≥ 3),
or a numpy ndarray with ndim = tensor order.
"""

from __future__ import annotations

import itertools
import string
from typing import Callable

import numpy as np
import sympy as sp

# --------------------------------------------------------------------------- backend helpers


def is_symbolic(*objects) -> bool:
    """True if any argument (or element of a list argument) is a sympy object."""
    return any(isinstance(o, (sp.Basic, sp.MatrixBase, sp.NDimArray))
               or (isinstance(o, (list, tuple)) and is_symbolic(*o)) for o in objects)


def order(x) -> int:
    """Tensor order: 0 scalar, 1 vector (incl. a single-column Matrix), 2, 3, 4."""
    if isinstance(x, sp.MatrixBase):
        return 1 if 1 in x.shape else 2
    if isinstance(x, sp.NDimArray):
        return x.rank()
    return int(np.ndim(x))


def _to_array(x):
    """Matrix / ndarray / list -> sympy Array (column vectors become rank 1)."""
    if isinstance(x, sp.MatrixBase):
        return sp.Array(list(x)) if 1 in x.shape else sp.Array(x.tolist())
    if isinstance(x, (np.ndarray, list, tuple)):
        return sp.Array(np.asarray(x).tolist())
    return x


def _from_array(arr):
    """sympy Array -> column Matrix (rank 1) / Matrix (rank 2); higher ranks stay Arrays."""
    if not isinstance(arr, sp.NDimArray) or arr.rank() > 2:
        return arr
    return sp.Matrix(list(arr)) if arr.rank() == 1 else sp.Matrix(arr.tolist())


def _entries(x) -> list:
    """Flat list of the components of a scalar, Matrix or Array."""
    return sp.flatten(_to_array(x).tolist()) if isinstance(x, (sp.MatrixBase, sp.NDimArray)) else [x]


def einsum(subscripts: str, *operands):
    """Einstein summation in explicit form ``"in1,in2->out"`` for sympy or numpy operands.

    Examples: ``einsum("ii->", A)`` = tr A,  ``einsum("ijkl,kl->ij", C, A)`` = C[A].
    """
    if not is_symbolic(*operands):
        result = np.einsum(subscripts, *(np.asarray(o, dtype=float) for o in operands))
        return float(result) if np.ndim(result) == 0 else result
    inputs, output = subscripts.replace(" ", "").split("->")
    specs, arrays = inputs.split(","), [_to_array(o) for o in operands]
    dims = {c: n for spec, a in zip(specs, arrays) for c, n in zip(spec, getattr(a, "shape", ()))}
    dummy = [c for c in dims if c not in output]

    def component(free_values: tuple) -> sp.Expr:
        terms = []
        for dummy_values in itertools.product(*(range(dims[c]) for c in dummy)):
            idx = {**dict(zip(output, free_values)), **dict(zip(dummy, dummy_values))}
            factors = [a[tuple(idx[c] for c in spec)] if spec else a for spec, a in zip(specs, arrays)]
            if all(f != 0 for f in factors):
                terms.append(sp.Mul(*factors))
        return sp.Add(*terms)

    shape = tuple(dims[c] for c in output)
    if not shape:
        return component(())
    values = [component(v) for v in itertools.product(*(range(n) for n in shape))]
    return _from_array(sp.ImmutableDenseNDimArray(values, shape))


def to_numpy(x, subs: dict | None = None):
    """Evaluate a sympy scalar / Matrix / Array (after optional ``subs``) to float(s)."""
    def evaluate(e):
        return float(sp.N(e.subs(subs) if subs else e))
    if isinstance(x, (sp.MatrixBase, sp.NDimArray)):
        return np.array([evaluate(e) for e in _entries(x)]).reshape(_to_array(x).shape)
    return evaluate(x) if isinstance(x, sp.Basic) else np.asarray(x, dtype=float)


def simplify_tensor(x, simplifier: Callable = sp.simplify):
    """Apply ``simplifier`` to every component of a sympy tensor (or to a scalar)."""
    return x.applyfunc(simplifier) if isinstance(x, (sp.MatrixBase, sp.NDimArray)) else simplifier(x)


def symbolic_tensor(name: str, rank: int = 2, symmetric: bool = False):
    """Tensor with generic real components ``name_ij…`` (1-based); ``symmetric`` for rank 2."""
    def symbol(idx: tuple) -> sp.Symbol:
        idx = sorted(idx) if symmetric else idx
        return sp.Symbol(f"{name}_{''.join(str(i + 1) for i in idx)}", real=True)
    values = [symbol(i) for i in itertools.product(range(3), repeat=rank)]
    return _from_array(sp.ImmutableDenseNDimArray(values, (3,) * rank))


# --------------------------------------------------------------------------- vectors, products


def identity2(symbolic: bool = True):
    """Second-order identity I = δ_ij e_i ⊗ e_j."""
    return sp.eye(3) if symbolic else np.eye(3)


def permutation_tensor(symbolic: bool = True):
    """Permutation tensor ε_ijk = (e_i × e_j) · e_k."""
    values = [(i - j) * (j - k) * (k - i) // 2 for i, j, k in itertools.product(range(3), repeat=3)]
    return sp.ImmutableDenseNDimArray(values, (3, 3, 3)) if symbolic else np.reshape(values, (3, 3, 3)) * 1.0


def tensor_product(A, B):
    """Dyadic product of any order: (a ⊗ b)_ij = a_i b_j, (A ⊗ B)_ijkl = A_ij B_kl."""
    if is_symbolic(A, B):
        return _from_array(sp.tensorproduct(_to_array(A), _to_array(B)))
    return np.multiply.outer(np.asarray(A, dtype=float), np.asarray(B, dtype=float))


def norm(A):
    """Euclidean / Frobenius norm ‖A‖ = √(A · A) = √(A_ij A_ij) for any order."""
    idx = string.ascii_lowercase[:order(A)]
    square = einsum(f"{idx},{idx}->", A, A)
    return sp.sqrt(square) if is_symbolic(A) else float(np.sqrt(square))


# --------------------------------------------------------------------------- second-order tensors


def tr(A):
    """Trace tr A = A_ii."""
    return einsum("ii->", A)


def sym(A):
    """Symmetric part sym(A) = (A + Aᵀ)/2."""
    return (A + A.T) / 2


def skw(A):
    """Skew part skw(A) = (A − Aᵀ)/2."""
    return (A - A.T) / 2


def sph(A):
    """Spherical part sph(A) = (tr A / 3) I."""
    return tr(A) / 3 * identity2(is_symbolic(A))


def dev(A):
    """Deviator A' = dev(A) = A − sph(A), tr A' = 0."""
    return A - sph(A)


def det(A):
    """Determinant det A."""
    return sp.Matrix(A).det() if is_symbolic(A) else float(np.linalg.det(A))


def principal_invariants(A) -> tuple:
    """(I_A, II_A, III_A) = (tr A, ½[(tr A)² − tr A²], det A)."""
    I_A = tr(A)
    return I_A, (I_A**2 - tr(A @ A)) / 2, det(A)


def cayley_hamilton_residual(A):
    """A³ − I_A A² + II_A A − III_A I, which vanishes for every A (Cayley–Hamilton)."""
    I_A, II_A, III_A = principal_invariants(A)
    return A @ A @ A - I_A * A @ A + II_A * A - III_A * identity2(is_symbolic(A))


def cofactor(A):
    """cof A = det(A) A⁻ᵀ = (A² − I_A A + II_A I)ᵀ (valid also for singular A)."""
    I_A, II_A, _ = principal_invariants(A)
    return (A @ A - I_A * A + II_A * identity2(is_symbolic(A))).T


def eigenprojectors(A, tol: float = 1e-9) -> list[tuple]:
    """Spectral form A = Σ λ_i P_i of a symmetric A, one projector per distinct eigenvalue.

    P_i P_j = δ_ij P_i,  Σ P_i = I,  tr P_i = multiplicity of λ_i.  Returns [(λ_i, P_i), …].
    """
    if is_symbolic(A):
        pairs = []
        for lam, _, vectors in sp.Matrix(A).eigenvects(simplify=True):
            P = sum((v * v.T for v in sp.GramSchmidt(vectors, True)), sp.zeros(3, 3))
            pairs.append((sp.simplify(lam), simplify_tensor(P)))
        return pairs
    w, V = np.linalg.eigh(np.asarray(A, dtype=float))
    pairs, start = [], 0
    for k in range(1, 4):
        if k == 3 or abs(w[k] - w[start]) > tol * max(1.0, np.abs(w).max()):
            pairs.append((float(w[start:k].mean()), V[:, start:k] @ V[:, start:k].T))
            start = k
    return pairs


def tensor_function(A, f: Callable):
    """Isotropic tensor function f(A) = Σ f(λ_i) P_i of a symmetric A, e.g. U = √C."""
    return sum((f(lam) * P for lam, P in eigenprojectors(A)), 0 * A)


# --------------------------------------------------------------------------- rotations


def skew_from_axial(w):
    """Skew tensor W with W v = w × v, W_ij = −ε_ijk w_k."""
    return -1 * einsum("ijk,k->ij", permutation_tensor(is_symbolic(w)), w)


def axial_vector(W):
    """Axial vector of skw(W): w_i = −½ ε_ijk W_jk."""
    return -1 * einsum("ijk,jk->i", permutation_tensor(is_symbolic(W)), W) / 2


def rotation_tensor(n, phi):
    """Euler–Rodrigues: Q(n, φ) = cos φ I + (1 − cos φ) n ⊗ n + sin φ [n]×  (n is normalized)."""
    symbolic = is_symbolic(n, phi)
    n = sp.Matrix(n) if symbolic else np.asarray(n, dtype=float)
    n = n / norm(n)
    cos, sin = (sp.cos, sp.sin) if symbolic else (np.cos, np.sin)
    Q = cos(phi) * identity2(symbolic) + (1 - cos(phi)) * tensor_product(n, n) + sin(phi) * skew_from_axial(n)
    return simplify_tensor(Q) if symbolic else Q


def rotation_axis_angle(Q) -> tuple[np.ndarray, float]:
    """Axis n and angle φ ∈ [0, π] of a numeric rotation: cos φ = (tr Q − 1)/2, skw Q = sin φ [n]×."""
    Q = np.asarray(Q, dtype=float)
    phi = float(np.arccos(np.clip((np.trace(Q) - 1) / 2, -1.0, 1.0)))
    if phi < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0
    if np.pi - phi < 1e-6:                      # skw Q ≈ 0: use (Q + I)/2 = n ⊗ n
        nn = (Q + np.eye(3)) / 2
        k = int(np.argmax(np.diag(nn)))
        return nn[:, k] / np.sqrt(nn[k, k]), phi
    return axial_vector(Q) / np.sin(phi), phi


# --------------------------------------------------------------------------- fourth-order tensors


def linear_map(T, X):
    """Contract the trailing indices of T with all indices of X: C[A]_ij = C_ijkl A_kl."""
    letters, k = string.ascii_lowercase[:order(T)], order(T) - order(X)
    return einsum(f"{letters},{letters[k:]}->{letters[:k]}", T, X)


def compose4(C, D):
    """(C ∘ D)_ijkl = C_ijmn D_mnkl, so that (C ∘ D)[A] = C[D[A]]."""
    return einsum("ijmn,mnkl->ijkl", C, D)


def _tensor4(entry: Callable, symbolic: bool):
    values = [entry(*idx) for idx in itertools.product(range(3), repeat=4)]
    return sp.ImmutableDenseNDimArray(values, (3,) * 4) if symbolic else np.array(values, dtype=float).reshape((3,) * 4)


def _d(i: int, j: int) -> int:
    return int(i == j)


def identity4(symbolic: bool = True):
    """Identity on Lin: 𝕀_ijkl = δ_ik δ_jl, 𝕀[A] = A."""
    return _tensor4(lambda i, j, k, l: _d(i, k) * _d(j, l), symbolic)


def I_sym(symbolic: bool = True):
    """Symmetrizer: (I_sym)_ijkl = ½(δ_ik δ_jl + δ_il δ_jk), I_sym[A] = sym(A)."""
    return _tensor4(lambda i, j, k, l: sp.Rational(_d(i, k) * _d(j, l) + _d(i, l) * _d(j, k), 2), symbolic)


def I_skw(symbolic: bool = True):
    """Skew projector: (I_skw)_ijkl = ½(δ_ik δ_jl − δ_il δ_jk), I_skw[A] = skw(A); dimension 3."""
    return _tensor4(lambda i, j, k, l: sp.Rational(_d(i, k) * _d(j, l) - _d(i, l) * _d(j, k), 2), symbolic)


def I_sph(symbolic: bool = True):
    """Spherical projector I_sph = ⅓ I ⊗ I, I_sph[A] = sph(A); dimension 1."""
    return _tensor4(lambda i, j, k, l: sp.Rational(_d(i, j) * _d(k, l), 3), symbolic)


def I_dev(symbolic: bool = True):
    """Deviatoric projector I_dev = I_sym − I_sph, I_dev[A] = dev(sym A); dimension 5.

    𝕀 = I_sph + I_dev + I_skw is the spectral decomposition of the identity on Lin.
    """
    return I_sym(symbolic) - I_sph(symbolic)


def isotropic_tensor4(lambda_sph, lambda_dev, lambda_skw=0):
    """Isotropic 4th-order tensor in spectral form λ_sph I_sph + λ_dev I_dev + λ_skw I_skw."""
    s = is_symbolic(lambda_sph, lambda_dev, lambda_skw)
    return lambda_sph * I_sph(s) + lambda_dev * I_dev(s) + lambda_skw * I_skw(s)


# --------------------------------------------------------------------------- Mandel notation
# Sym <-> R⁶ and minor-symmetric 4th order <-> R⁶ˣ⁶, with A · B = a · b and C[A] <-> C a.

_MANDEL = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def _mandel_weights(symbolic: bool) -> tuple:
    root2 = sp.sqrt(2) if symbolic else np.sqrt(2.0)
    return 1, 1, 1, root2, root2, root2


def to_mandel_vector(A):
    """Symmetric A -> (A11, A22, A33, √2 A23, √2 A13, √2 A12)."""
    s, w = is_symbolic(A), _mandel_weights(is_symbolic(A))
    values = [w[a] * A[i, j] for a, (i, j) in enumerate(_MANDEL)]
    return sp.Matrix(values) if s else np.array(values, dtype=float)


def from_mandel_vector(v):
    """Inverse of ``to_mandel_vector``."""
    s, w = is_symbolic(v), _mandel_weights(is_symbolic(v))
    A = sp.zeros(3, 3) if s else np.zeros((3, 3))
    for a, (i, j) in enumerate(_MANDEL):
        A[i, j] = A[j, i] = v[a] / w[a]
    return A


def to_mandel_matrix(C):
    """Minor-symmetric C -> 6×6 matrix M_ab = w_a w_b C_ijkl; inversion on Sym = matrix inverse."""
    s, w = is_symbolic(C), _mandel_weights(is_symbolic(C))
    values = [[w[a] * w[b] * C[i, j, k, l] for b, (k, l) in enumerate(_MANDEL)] for a, (i, j) in enumerate(_MANDEL)]
    return sp.Matrix(values) if s else np.array(values, dtype=float)


def from_mandel_matrix(M):
    """Inverse of ``to_mandel_matrix`` (the result has both minor symmetries)."""
    s, w = is_symbolic(M), _mandel_weights(is_symbolic(M))
    C = sp.MutableDenseNDimArray.zeros(3, 3, 3, 3) if s else np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(_MANDEL):
        for b, (k, l) in enumerate(_MANDEL):
            for p, q in {(i, j), (j, i)}:
                for m, n in {(k, l), (l, k)}:
                    C[p, q, m, n] = M[a, b] / (w[a] * w[b])
    return sp.ImmutableDenseNDimArray(C) if s else C
