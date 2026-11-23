import numpy as np
import sympy as sp

import cmtoolkit.tensor_analysis as tan
from cmtoolkit.checks import numerical_gradient
from cmtoolkit.tensor_algebra import dev, norm, principal_invariants, sym

RNG = np.random.default_rng(1)


def test_vector_identities_for_generic_fields(zero):
    x = sp.symbols("x_1:4", real=True)
    u = sp.Matrix([sp.Function(f"u_{i}")(*x) for i in (1, 2, 3)])
    zero(tan.div(tan.rot(u, x), x))
    zero(tan.rot(tan.grad(sp.Function("phi")(*x), x), x))
    zero(tan.rot(tan.rot(u, x), x) - tan.grad(tan.div(u, x), x) + tan.laplace(u, x))


def test_cylindrical_operators(zero):
    cyl = r, phi, z = tan.cylindrical_coordinates()
    f = sp.Function("f")(r, phi, z)
    zero(tan.laplace_cylindrical(f, cyl) - (f.diff(r, 2) + f.diff(r) / r + f.diff(phi, 2) / r**2 + f.diff(z, 2)))
    zero(tan.laplace_cylindrical(sp.log(r), cyl))
    a, b, c = (sp.Function(name)(r) for name in "abc")
    zero(tan.div_cylindrical(sp.diag(a, b, c), cyl) - sp.Matrix([a.diff(r) + (a - b) / r, 0, 0]))
    zero(tan.grad_cylindrical(sp.Matrix([a, 0, 0]), cyl) - sp.diag(a.diff(r), a / r, 0))


def test_generalized_gradients_vs_finite_differences():
    A = RNG.normal(size=(3, 3))
    for k, derivative in enumerate((tan.d_first_invariant, tan.d_second_invariant, tan.d_third_invariant)):
        numeric = numerical_gradient(lambda X: principal_invariants(X)[k], A)
        np.testing.assert_allclose(numeric, derivative(A), atol=1e-7)
    sigma = sym(RNG.normal(size=(3, 3)))
    for deviatoric in (False, True):
        part = dev if deviatoric else (lambda X: X)
        numeric = numerical_gradient(lambda X: part(sym(X)) / norm(part(sym(X))), sigma)
        np.testing.assert_allclose(numeric, tan.d_normalized_tensor(sigma, deviatoric), atol=1e-7)
