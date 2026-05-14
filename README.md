# CFD Solver

An internal tool for simulating how incompressible fluids move. Built by the club to learn numerical methods and get hands-on with computational physics.

---

## Table of Contents
- [The Physics](#the-physics)
- [How We Solve It](#how-we-solve-it)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Contributing](#contributing)

---

## The Physics

We're solving for fluid flow that doesn't compress — water, air at low speeds, that kind of thing.

### Conservation Laws

**Mass doesn't disappear:**
$$\nabla \cdot \mathbf{u} = 0$$

The velocity field has zero divergence. No fluid is appearing or vanishing.

**Forces add up (Navier-Stokes):**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}$$

Fluid accelerates from pressure differences and spreads out via viscosity.

| Symbol | Meaning |
|--------|---------|
| $\mathbf{u}$ | Velocity at each point |
| $p$ | Pressure |
| $\rho$ | Density |
| $\nu$ | Kinematic viscosity |

---

## How We Solve It

We break the problem into steps a computer can handle.

1. **Grid it up** — Divide space into small boxes and solve at each corner.
2. **Guess the flow** — Solve the momentum equation ignoring pressure. We get an intermediate velocity that probably violates mass conservation.
3. **Fix the pressure** — Solve the pressure equation to find what pressure field would make the flow divergence-free.
4. **Correct the velocity** — Apply the pressure gradient to get the real velocity.
5. **Repeat** — Advance in time, repeating steps 2–4 until we're done.

This is the **Projection Method** (also called Chorin's Method). It sidesteps solving velocity and pressure simultaneously, which is numerically painful.

| Term | What it means |
|------|---------------|
| FDM | Finite Difference Method — simplest way to approximate derivatives on a grid |
| Projection | Split the problem: guess first, fix later |
| Divergence-free | Fluid coming in must go out — no disappearing or multiplying |

---

## Project Structure

```
CFD-Solver/
├── src/                  # Core solver code
├── tests/                # Unit tests
├── data/                 # Input files for different cases
├── docs/                 # Notes and derivations
├── examples/             # Ready-to-run simulations
├── output/                # Plots and results go here
├── requirements.txt      # Python dependencies
└── .gitignore
```

---

## Getting Started

**1. Clone and enter the repo:**
```bash
git clone https://github.com/baidik012/CFD-Solver.git
cd CFD-Solver
```

**2. Set up a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

### Run a simulation

```bash
python examples/lid_driven_cavity.py
```

Results go to the `output/` directory — velocity fields, pressure maps, whatever the example dumps out.

### Run tests

```bash
pytest tests/
```

All commands run from the root directory so Python finds the `src/` folder correctly.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on branches, code style, and the PR process.

---

## License

Educational use only — this isn't validated for industrial or safety-critical applications.
See [LICENSE](LICENSE).
