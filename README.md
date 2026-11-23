# continuum-mechanics-toolkit

`cmtoolkit` is a small Python package of building blocks for computational continuum mechanics: tensor algebra,
differential operators, finite-strain kinematics and isotropic linear thermoelasticity. Its flagship example is a
thick-walled tube under pressure and a radial temperature field, derived step by step and checked independently.

Every function takes **sympy** (symbolic) or **numpy** (numeric) input and returns the same kind, so one code path
gives an exact derivation and its numerical cross-check.

## Install

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .
pytest                                                  # 16 tests, about 15 s
```

Python ≥ 3.11; tested with the pinned versions in `requirements.txt`.

## Usage

```python
import sympy as sp
from cmtoolkit.thick_walled_tube import solve_thick_walled_tube
r_i, r_a, p_i, E, nu = sp.symbols("r_i r_a p_i E nu", positive=True)
solve_thick_walled_tube(r_i, r_a, p_i, 0, E, nu)["sigma_pp"]   # p_i r_i² (r² + r_a²) / (r² (r_a² − r_i²))
```

## Contents

| Module | What it provides |
|---|---|
| `tensor_algebra` | `sym`/`skw`/`sph`/`dev`, invariants, Cayley–Hamilton, eigenprojectors, √C, rotations, isotropic 4th-order projectors, Mandel notation |
| `tensor_analysis` | cartesian `grad`/`div`/`rot`/Δ, cylindrical operators in physical components, gradients of invariants and of the von Mises flow direction |
| `kinematics` | right Cauchy–Green tensor, polar decomposition, Green and Hill strains |
| `elasticity` | Lamé constants, isotropic stiffness/compliance, Hooke's law, thermoelasticity, von Mises stress |
| `thick_walled_tube` | radial temperature field, derivation of the displacement ODE, closed-form solution (plane strain, closed ends, plane stress) |
| `checks` | assertion helpers and central finite differences |

| Notebook | Content | Run time |
|---|---|---|
| `01_tensor_basics` | decompositions, Cayley–Hamilton, eigenprojectors, polar decomposition, rotations, isotropic projectors, Mandel notation, generalized gradients | ≈ 10 s |
| `02_thick_walled_tube` | kinematics → law → equilibrium → ODE → integration → boundary conditions → Lamé and thermal closed forms → finite-difference check → plots | ≈ 40 s |

## Scope and limitations

* 3D, orthonormal basis; curvilinear operators are implemented for cylindrical coordinates only.
* Material behaviour: isotropic linear (thermo-)elasticity with small strains. No plasticity, dynamics or FEM.
* The tube solution assumes axisymmetry, a stationary temperature field and no body forces.
* Symbolic simplification is exact but slow for large expressions; this is a learning and portfolio project, not a validated engineering tool.

## Credit

Built while studying the course "Rechnergestützte Kontinuumsmechanik" at KIT. The project is not affiliated with or
endorsed by KIT or the course instructors, and it contains no lecture slides, exercise sheets or solutions.

## License

MIT, see [LICENSE](LICENSE).
