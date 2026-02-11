# CFD Solver (Club Internal)

This repository contains an internal Computational Fluid Dynamics (CFD) solver
developed by the club for studying numerical methods applied to incompressible
fluid flow. The project is educational in nature and intended for collaborative
development and experimentation.

---

## Scope

- Educational and exploratory use only
- Focused on understanding numerical implementation of the incompressible Navier–Stokes equations
- Not validated for industrial, commercial, or safety-critical applications

---

## Governing Equations

The solver is based on the two-dimensional incompressible Navier–Stokes equations:

Continuity equation:

∇ · u = 0

Momentum equations:

∂u/∂t + (u · ∇)u = - (1/ρ) ∇p + ν ∇²u

Where:
- u = velocity vector field
- p = pressure field
- ρ = density
- ν = kinematic viscosity

---

## Numerical Method

- Spatial discretization: Finite Difference Method (FDM)
- Grid type: Structured Cartesian grid
- Pressure–velocity coupling: Projection method
- Time integration: Explicit scheme (current implementation)
- Pressure solution: Poisson equation solver

Future improvements may include stability enhancements and improved discretization schemes.

---

## Current Capabilities

- 2D incompressible flow
- Uniform structured mesh
- Basic boundary condition framework
- Benchmark validation cases under development

---

## Project Structure

src/  
    mesh.py              Grid generation and spacing  
    fields.py            Velocity and pressure field definitions  
    boundary.py          Boundary condition handling  
    solver.py            Time-stepping and Navier–Stokes integration  
    poisson.py           Pressure Poisson equation solver  
    utils.py             Helper and utility functions  

examples/  
    lid_driven_cavity.py Standard benchmark case  
    couette_flow.py      Analytical validation case  

tests/  
    test_mesh.py         Mesh verification  
    test_poisson.py      Pressure solver consistency tests  

docs/  
    derivations.md       Discretization details and derivations  

---

## Requirements

- Python 3.10+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
