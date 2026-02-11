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
* u: Velocity vector field
* p: Pressure field
* rho: Density
* nu: Kinematic viscosity

---

## 2. Execution: Numerical Implementation
To solve these Partial Differential Equations (PDEs), we use the following numerical recipe:

* **Spatial Discretization:** Finite Difference Method (FDM) on a Structured Cartesian Grid.
* **The Trigger:** For incompressible flows, pressure is a constraint to satisfy continuity. We use the **Projection Method (Chorin’s Method)**.
* **The Path (Logic Steps):** 1. Calculate an intermediate velocity u* (Predictor step).
    2. Solve the **Pressure Poisson Equation** to find the pressure field.
    3. Project u* onto a divergence-free space to get the final velocity u_n+1 (Corrector step).



---

## 3. Project Structure
The code is modularized to separate physics logic from helper functions:

src/
    mesh.py         # Logic: Grid generation and spacing
    fields.py       # Logic: Array definitions for U, V, and P
    boundary.py     # Logic: BC implementation (No-slip, Inflow, etc.)
    solver.py       # Logic: Time-stepping and N-S integration
    poisson.py      # Logic: The iterative Pressure Poisson solver
    utils.py        # Logic: Plotting and helper functions

examples/           # Pre-configured benchmark cases
tests/              # Verification and unit tests
docs/               # Detailed derivations and discretization notes

---

## 4. Setup and Usage

### Concept: Reproducibility
We use **Virtual Environments** to ensure every club member uses the same library versions, preventing "it works on my machine" errors.

### Execution: Installation
**1. Clone the Repository:**
git clone [https://github.com/baidik012/CFD-Solver.git](https://github.com/baidik012/CFD-Solver.git)
cd CFD-Solver

**2. Create and Activate Virtual Environment:**
# Create venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

**3. Install Dependencies:**
pip install -r requirements.txt

### Execution: Running a Simulation
**Trigger:** Run all commands from the root directory so Python finds the src/ folder.

**Lid-Driven Cavity Benchmark:**
python examples/lid_driven_cavity.py



**Couette Flow Validation:**
python examples/couette_flow.py

---

## 5. Club Contribution Guidelines
1. **Work on Branches:** Never push directly to main. Use 'git checkout -b feature-name'.
2. **No Data in Repo:** Do not upload .png, .mp4, or large .log files.
3. **Review:** Once your feature is ready, open a Pull Request for the club leads to review.

> **Note:** This project is for **educational use only**. It is not validated for industrial or safety-critical applications.