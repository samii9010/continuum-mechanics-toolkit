import numpy as np
import pytest
import sympy as sp

from cmtoolkit.tensor_algebra import _entries, is_symbolic


@pytest.fixture
def zero():
    def check(x, tol: float = 1e-9):      # sympy: every component simplifies to 0; numpy: |x| < tol
        if is_symbolic(x):
            assert all(sp.simplify(e) == 0 for e in _entries(x)), x
        else:
            np.testing.assert_allclose(np.asarray(x, dtype=float), 0, atol=tol)
    return check
