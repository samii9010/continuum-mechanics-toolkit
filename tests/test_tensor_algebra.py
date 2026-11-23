import numpy as np
import pytest
import sympy as sp

import cmtoolkit.tensor_algebra as ta
from cmtoolkit.elasticity import compliance_isotropic, stiffness_isotropic

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("A", [ta.symbolic_tensor("A"), RNG.normal(size=(3, 3))])
def test_decomposition_and_cayley_hamilton(A, zero):
    zero(ta.sym(A) + ta.skw(A) - A)
    zero(ta.sph(A) + ta.dev(A) - A)
    zero(ta.tr(ta.dev(A)))
    zero(ta.einsum("ij,ij->", ta.sym(A), ta.skw(A)) + ta.einsum("ij,ij->", ta.sph(A), ta.dev(A)))
    zero(ta.cayley_hamilton_residual(A))


@pytest.mark.parametrize("A", [sp.Matrix([[2, 1, 0], [1, 2, 0], [0, 0, 3]]),          # eigenvalues 1, 3, 3
                               (lambda Q: Q @ np.diag([2.0, 2.0, 5.0]) @ Q.T)(ta.rotation_tensor([1, 2, 3], 0.4))])
def test_eigenprojectors_with_repeated_eigenvalue(A, zero):
    pairs = ta.eigenprojectors(A)
    assert sorted(round(float(ta.tr(P))) for _, P in pairs) == [1, 2]
    zero(sum((P for _, P in pairs), 0 * A) - ta.identity2(ta.is_symbolic(A)))
    zero(sum((lam * P for lam, P in pairs), 0 * A) - A)
    for i, (_, P) in enumerate(pairs):
        for j, (_, Q) in enumerate(pairs):
            zero(P @ Q - (P if i == j else 0 * P))
    root = ta.tensor_function(A, sp.sqrt if ta.is_symbolic(A) else np.sqrt)
    zero(root @ root - A)


def test_rotation_tensor_and_axis_angle(zero):
    n = np.array([1.0, 2.0, 3.0])
    for phi in (2.1, np.pi):
        Q = ta.rotation_tensor(n, phi)
        zero(Q.T @ Q - np.eye(3))
        zero(Q @ n - n)
        assert np.isclose(ta.det(Q), 1.0)
        axis, angle = ta.rotation_axis_angle(Q)
        assert np.isclose(angle, phi)
        np.testing.assert_allclose(axis * np.sign(axis @ n), n / ta.norm(n))


def test_isotropic_projectors(zero):
    projectors = [ta.I_sph(False), ta.I_dev(False), ta.I_skw(False)]
    zero(sum(projectors) - ta.identity4(False))
    for i, P in enumerate(projectors):
        for j, Q in enumerate(projectors):
            zero(ta.compose4(P, Q) - (P if i == j else 0 * P))
    assert [round(ta.einsum("ijij->", P)) for P in projectors] == [1, 5, 3]


def test_mandel_round_trip(zero):
    A, B = ta.sym(RNG.normal(size=(3, 3))), ta.sym(RNG.normal(size=(3, 3)))
    zero(ta.from_mandel_vector(ta.to_mandel_vector(A)) - A)
    assert np.isclose(ta.to_mandel_vector(A) @ ta.to_mandel_vector(B), ta.einsum("ij,ij->", A, B))
    C, S = stiffness_isotropic(1.3, 0.7), compliance_isotropic(1.3, 0.7)
    zero(ta.from_mandel_matrix(ta.to_mandel_matrix(C)) - C)
    zero(ta.to_mandel_matrix(C) @ ta.to_mandel_vector(A) - ta.to_mandel_vector(ta.linear_map(C, A)))
    zero(np.linalg.inv(ta.to_mandel_matrix(C)) - ta.to_mandel_matrix(S))
    symbolic = ta.einsum("ijkl,kl->ij", sp.Array(C.tolist()), sp.Matrix(A))      # sympy and numpy backends agree
    np.testing.assert_allclose(ta.to_numpy(symbolic), ta.linear_map(C, A))
