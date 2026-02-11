# CFD Solver (Club Internal)

This repository contains an internal Computational Fluid Dynamics (CFD) solver developed by the club for studying numerical methods applied to incompressible fluid flow. 

---

## 1. Concept: The Physics
This solver is designed to simulate **Incompressible Newtonian Flows**. The logic is governed by the conservation of mass and momentum.

### Governing Equations

**Continuity Equation (Mass Conservation):** Ensures the velocity field remains solenoidal (divergence-free).
$$\nabla \cdot \mathbf{u} = 0$$

**Momentum Equations (Navier–Stokes):** Describes the balance of convective, pressure, and viscous forces.
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}$$

**Variables:**
* $\mathbf{u}$: Velocity vector field
* $p$: Pressure field
* $\rho$: Density
* $\nu$: Kinematic viscosity

---

## 2. Execution: Numerical Implementation
To solve these partial differential equations (PDEs) on a computer, we use a specific "Numerical Recipe":

* **Spatial Discretization:** Finite Difference Method (FDM).
* **Grid Topology:** Structured Cartesian grid.
* **The Trigger:** To handle the pressure-velocity coupling in incompressible flows, we use the **Projection Method (Chorin’s Method)**.
* **The Path:** 1. Calculate an intermediate velocity $\mathbf{u}^*$ (ignoring pressure).
    2. Solve the **Pressure Poisson Equation** to find the pressure field that makes the velocity divergence-free.
    3. Project the intermediate velocity onto a divergence-free space to get $\mathbf{u}^{n+1}$.



---

## 3. Project Structure
The code is modularized to separate the physics from the math helpers:

```text
src/
    mesh.py         # Logic: Grid generation and spacing
    fields.py       # Logic: Array definitions for U, V, and P
    boundary.py     # Logic: BC implementation (No-slip, Inflow, etc.)
    solver.py       # Logic: Time-stepping and N-S integration
    poisson.py      # Logic: The iterative Pressure Poisson solver
    utils.py        # Logic: Plotting and helper functions

examples/           # Pre-configured benchmark cases
tests/              # Verification and unit tests
docs/               # Detailed derivations and notes
```
---

## 4. Setup and Usage

### Concept: Reproducibility
To ensure the solver runs the same way on everyone's machine, we use a **Virtual Environment**. This prevents "Dependency Hell" where one person's version of NumPy conflicts with another's.

### Execution: Installation Steps

**1. Clone the Repository**
Access the code locally by cloning the private repo:
```bash
git clone https://github.com/[baidik012]/[CFD-Solver].git
cd [CFD-Solver]