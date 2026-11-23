"""Thick-walled tube r_i ≤ r ≤ r_a under pressure p_i, p_a and a radial temperature change ΔT(r).

Axisymmetric small-strain field u = u_r(r) e_r in physical cylindrical components:
ε_rr = u_r', ε_φφ = u_r / r, and the axial assumption
* ``"strain"``:      ε_zz = 0 (long tube, fixed ends),
* ``"closed_ends"``: ε_zz = const, fixed by the axial force balance ∫ σ_zz 2πr dr = π(p_i r_i² − p_a r_a²),
* ``"stress"``:      σ_zz = 0 (thin disc).
"""

from __future__ import annotations

import sympy as sp

from .elasticity import bulk_modulus, lame_from_young_poisson, thermoelastic_stress

PLANES = ("strain", "closed_ends", "stress")


def stationary_radial_temperature(r: sp.Symbol, r_i, r_a, T_i, T_a) -> sp.Expr:
    """Solution of ΔT = 0 with T(r_i) = T_i, T(r_a) = T_a:  T = T_i + (T_a − T_i) ln(r/r_i) / ln(r_a/r_i)."""
    return T_i + (T_a - T_i) * sp.log(r / r_i) / sp.log(r_a / r_i)


def tube_stress_components(u_r, r: sp.Symbol, delta_T, E, nu, alpha_T, plane: str = "strain", eps_zz=0) -> dict:
    """Strains and stresses (physical components) of u = u_r(r) e_r; keys eps_rr, eps_pp, eps_zz, sigma_rr, …"""
    if plane not in PLANES:
        raise ValueError(f"plane must be one of {PLANES}")
    lam, mu = lame_from_young_poisson(E, nu)
    eps_rr, eps_pp = sp.diff(u_r, r), u_r / r
    if plane == "stress":            # σ_zz = 0 solved for ε_zz
        eps_zz = (3 * bulk_modulus(lam, mu) * alpha_T * delta_T - lam * (eps_rr + eps_pp)) / (lam + 2 * mu)
    elif plane == "strain":
        eps_zz = 0
    sigma = thermoelastic_stress(sp.diag(eps_rr, eps_pp, eps_zz), delta_T, lam, mu, alpha_T)
    return {"eps_rr": eps_rr, "eps_pp": eps_pp, "eps_zz": sp.sympify(eps_zz),
            "sigma_rr": sigma[0, 0], "sigma_pp": sigma[1, 1], "sigma_zz": sigma[2, 2]}


def radial_equilibrium(c: dict, r: sp.Symbol) -> sp.Expr:
    """(div σ)_r = σ_rr' + (σ_rr − σ_φφ)/r, the only non-trivial equilibrium equation."""
    return sp.diff(c["sigma_rr"], r) + (c["sigma_rr"] - c["sigma_pp"]) / r


def thick_walled_tube_ode(r: sp.Symbol, delta_T, E, nu, alpha_T, plane: str = "strain") -> tuple:
    """Insert the stress law into radial equilibrium: u'' + u'/r − u/r² = [(r u)'/r]' = κ_T ΔT'.

    κ_T is derived, not hard-coded: α_T(1+ν)/(1−ν) for plane strain / closed ends, α_T(1+ν) for plane stress.
    Returns (ode, u_r, κ_T).
    """
    u, theta = sp.Function("u_r")(r), sp.Function("theta")(r)
    eps_zz = sp.Symbol("varepsilon_zz0", real=True) if plane == "closed_ends" else 0

    def equilibrium(u_r, dT):
        return radial_equilibrium(tube_stress_components(u_r, r, dT, E, nu, alpha_T, plane, eps_zz), r)

    stiffness = sp.expand(equilibrium(u, 0)).coeff(sp.Derivative(u, (r, 2)))       # factor of u''
    kappa_T = sp.simplify(-equilibrium(sp.S.Zero, theta) / (stiffness * theta.diff(r)))
    ode = sp.Eq(u.diff(r, 2) + u.diff(r) / r - u / r**2, kappa_T * sp.diff(delta_T, r))
    return ode, u, kappa_T


def solve_thick_walled_tube(r_i, r_a, p_i, p_a, E, nu, alpha_T=0, delta_T=0, plane: str = "strain",
                            simplify: bool | None = None) -> dict:
    """Closed-form solution: u_r = C_1 r + C_2/r + (κ_T/r) ∫_{r_i}^r ρ ΔT(ρ) dρ, with σ_rr(r_i) = −p_i, σ_rr(r_a) = −p_a.

    ``delta_T`` is an expression in ``sp.Symbol("r", positive=True)``. ``simplify`` defaults to True only
    for symbolic parameters (simplifying numeric logarithms is slow). Returns a dict with r, u_r, constants,
    kappa_T, the strain/stress components and the equilibrium residual.
    """
    r, rho = sp.Symbol("r", positive=True), sp.Dummy("rho", positive=True)
    delta_T = sp.sympify(delta_T)
    C_1, C_2 = sp.symbols("C_1 C_2")
    eps_zz = sp.Symbol("varepsilon_zz0", real=True) if plane == "closed_ends" else 0
    _, _, kappa_T = thick_walled_tube_ode(r, delta_T, E, nu, alpha_T, plane)
    thermal = kappa_T / r * sp.integrate(rho * delta_T.subs(r, rho), (rho, r_i, r)) if delta_T != 0 else 0
    comps = tube_stress_components(C_1 * r + C_2 / r + thermal, r, delta_T, E, nu, alpha_T, plane, eps_zz)

    equations, unknowns = [comps["sigma_rr"].subs(r, r_i) + p_i, comps["sigma_rr"].subs(r, r_a) + p_a], [C_1, C_2]
    if plane == "closed_ends":
        axial_force = sp.integrate(comps["sigma_zz"] * 2 * sp.pi * r, (r, r_i, r_a))
        equations.append(axial_force - sp.pi * (p_i * r_i**2 - p_a * r_a**2))
        unknowns.append(eps_zz)
    matrix, rhs = sp.linear_eq_to_matrix(equations, unknowns)       # BCs are linear in the constants
    constants = dict(zip(unknowns, matrix.LUsolve(rhs)))

    if simplify is None:
        simplify = any(sp.sympify(p).free_symbols - {r} for p in (r_i, r_a, p_i, p_a, E, nu, alpha_T, delta_T))
    tidy = (lambda e: sp.expand_log(sp.simplify(e), force=True)) if simplify else (lambda e: e)
    result = {"r": r, "kappa_T": kappa_T, "constants": {k: tidy(v) for k, v in constants.items()},
              "u_r": tidy((C_1 * r + C_2 / r + thermal).subs(constants))}
    result.update({key: tidy(sp.sympify(value).subs(constants)) for key, value in comps.items()})
    result["equilibrium_residual"] = tidy(radial_equilibrium(result, r))
    return result
