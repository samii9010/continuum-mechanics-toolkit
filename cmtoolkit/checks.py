"""Verification helpers shared by the notebooks and the tests."""

from __future__ import annotations

from typing import Callable

import numpy as np
import sympy as sp

from .tensor_algebra import _entries, to_numpy


def assert_close(actual, expected, tol: float = 1e-9, label: str = "check") -> None:
    """Assert max |actual − expected| < tol (sympy or numpy input) and print the deviation."""
    err = float(np.max(np.abs(np.asarray(to_numpy(actual)) - np.asarray(to_numpy(expected)))))
    assert err < tol, f"{label}: max |Δ| = {err:.3e} exceeds {tol:.1e}"
    print(f"[OK] {label}: max |Δ| = {err:.1e}")


def assert_symbolic_zero(x, label: str = "check", simplifier: Callable = sp.simplify) -> None:
    """Assert that every component of ``x`` simplifies to zero and print the result."""
    remaining = [e for e in _entries(x) if simplifier(e) != 0]
    assert not remaining, f"{label}: {len(remaining)} component(s) do not vanish, e.g. {remaining[0]}"
    print(f"[OK] {label}")


def show(label: str, value) -> None:
    """Render ``label = value`` as LaTeX in Jupyter."""
    from IPython.display import Math, display
    display(Math(f"{label} = {sp.latex(value)}"))


def numerical_gradient(f: Callable, x, h: float = 1e-6) -> np.ndarray:
    """Central-difference gradient ∂f/∂x of shape f(x).shape + x.shape (x a vector or a tensor)."""
    x = np.asarray(x, dtype=float)
    result = np.zeros(np.shape(f(x)) + x.shape)
    for idx in np.ndindex(x.shape):
        step = np.zeros_like(x)
        step[idx] = h
        result[(Ellipsis,) + idx] = (np.asarray(f(x + step)) - np.asarray(f(x - step))) / (2 * h)
    return result
