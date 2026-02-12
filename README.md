@@ -4,17 +4,28 @@ This repository contains an internal Computational Fluid Dynamics (CFD) solver d

---

## Table of Contents
- [Concept: The Physics](#1-concept-the-physics)
- [Execution: Numerical Implementation](#2-execution-numerical-implementation)
- [Project Structure](#3-project-structure)
- [Setup and Usage](#4-setup-and-usage)
- [Contribution Guidelines](#5-club-contribution-guidelines)

---

## 1. Concept: The Physics
This solver is designed to simulate **Incompressible Newtonian Flows**. The logic is governed by the conservation of mass and momentum.

### Governing Equations

**Continuity Equation (Mass Conservation):**
Ensures the velocity field remains solenoidal (divergence-free).

$$\nabla \cdot \mathbf{u} = 0$$

**Momentum Equations (Navier–Stokes):**
Describes the balance of convective, pressure, and viscous forces.

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}$$

**Variables:**
@@ -29,24 +40,25 @@ $$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} =
To solve these Partial Differential Equations (PDEs), we use the following numerical recipe:

* **Spatial Discretization:** Finite Difference Method (FDM) on a **Structured Cartesian Grid**.
* **The Decoupling Strategy:** For incompressible flows, we use the **Projection Method (Chorin's Method)** to decouple the velocity and pressure fields.
* **The Algorithm Steps:**
    1. **Predictor:** Solve the momentum equation without the pressure gradient to find an intermediate velocity $\mathbf{u}^*$.
    2. **Poisson:** Solve the Pressure Poisson Equation $\nabla^2 p = \frac{\rho}{\Delta t} (\nabla \cdot \mathbf{u}^*)$ to find the pressure field $p^{n+1}$.
    3. **Corrector:** Update the intermediate velocity $\mathbf{u}^*$ using the new pressure gradient to find the divergence-free velocity $\mathbf{u}^{n+1}$.



---

## 3. Project Structure
The following structure is the planned architecture for the solver. These are placeholders for the development phase:

```
src/
    grid.py         # Grid generation and discretization
    solver.py       # Main CFD solver
    physics.py      # Physics computations and updates
    utils.py        # Utility functions

examples/           # Benchmark cases (e.g., Lid-Driven Cavity)
tests/              # Verification scripts and unit tests
docs/               # Detailed derivations and discretization notes
```
@@ -62,7 +74,7 @@ We use **Virtual Environments** to ensure every club member uses the same librar

**1. Clone the Repository:**
```bash
git clone https://github.com/baidik012/CFD-Solver.git
cd CFD-Solver
```

@@ -85,21 +97,39 @@ pip install -r requirements.txt

---

### Quick Start
Get up and running in a few minutes:
```bash
# After installation, run the example
python examples/lid_driven_cavity.py
```

**Expected Output:** The solver will generate velocity and pressure field data for a lid-driven cavity problem. Visualizations will be saved to the `output/` directory.

---

### Running Simulations
**Important:** Run all commands from the **root directory** so Python correctly finds the `src/` folder and resolves internal imports.

```bash
# Run a simulation
python examples/lid_driven_cavity.py

# Run tests
python -m pytest tests/
```

---

## 5. Club Contribution Guidelines
1. **Work on Branches:** Never push directly to `main`. Use `git checkout -b feature-your-name`.
2. **No Data in Repo:** Do not upload `.png`, `.mp4`, or large `.log` files. Use a `.gitignore` file.
3. **Code Standards:** Ensure your code is documented and follows PEP 8 style guidelines.
4. **Review:** Once your feature is ready, open a **Pull Request** for the club leads to review.

---

## License
This project is for **educational use only**. It is not validated for industrial or safety-critical applications.
For licensing details, see the [LICENSE](LICENSE) file.
