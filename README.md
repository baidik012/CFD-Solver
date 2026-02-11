# CFD Solver (Club Internal)

This repository contains an internal Computational Fluid Dynamics (CFD) solver developed by the club for studying numerical methods applied to incompressible fluid flow. 

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
* $\mathbf{u}$: Velocity vector field
* $p$: Pressure field
* $\rho$: Density
* $\nu$: Kinematic viscosity

---

## 2. Execution: Numerical Implementation
To solve these Partial Differential Equations (PDEs), we use the following numerical recipe:

* **Spatial Discretization:** Finite Difference Method (FDM) on a **Structured Cartesian Grid**.
* **The Trigger:** For incompressible flows, we use the **Projection Method (Chorin’s Method)** to decouple the velocity and pressure fields.
* **The Path (Logic Steps):**
    1. **Predictor:** Solve the momentum equation without the pressure gradient to find an intermediate velocity $\mathbf{u}^*$.
    2. **Poisson:** Solve the Pressure Poisson Equation $\nabla^2 p = \frac{\rho}{\Delta t} (\nabla \cdot \mathbf{u}^*)$ to find the pressure field $p^{n+1}$.
    3. **Corrector:** Update the intermediate velocity $\mathbf{u}^*$ using the new pressure gradient to find the divergence-free velocity $\mathbf{u}^{n+1}$.



---

## 3. Project Structure
The following structure is the planned architecture for the solver. These are placeholders for the development phase:

```text
src/
    (TBD).py        # Future Logic: Grid generation, solver, and physics modules

examples/           # Planned benchmark cases (e.g., Lid-Driven Cavity)
tests/              # Verification scripts and unit tests
docs/               # Detailed derivations and discretization notes
```

---

## 4. Setup and Usage

### Concept: Reproducibility
We use **Virtual Environments** to ensure every club member uses the same library versions, preventing "dependency hell" across different operating systems.

### Execution: Installation

**1. Clone the Repository:**
```bash
git clone [https://github.com/baidik012/CFD-Solver.git](https://github.com/baidik012/CFD-Solver.git)
cd CFD-Solver
```

**2. Create and Activate Virtual Environment:**
```bash
# Create the environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

**3. Install Dependencies:**
```bash
pip install -r requirements.txt
```

---

### Execution: Running a Simulation
**Trigger:** Run all commands from the **root directory** so Python finds the `src/` folder and resolves internal imports.

```bash
# Example Command (To be updated as scripts are developed)
python examples/lid_driven_cavity.py
```



---

## 5. Club Contribution Guidelines
1. **Work on Branches:** Never push directly to `main`. Use `git checkout -b feature-your-name`.
2. **No Data in Repo:** Do not upload `.png`, `.mp4`, or large `.log` files. Use a `.gitignore` file.
3. **Review:** Once your feature is ready, open a **Pull Request** for the club leads to review.

> **Note:** This project is for **educational use only**. It is not validated for industrial or safety-critical applications.