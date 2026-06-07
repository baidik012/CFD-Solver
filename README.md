# CFD Solver

An internal tool for simulating incompressible fluid flow. Built by the club to learn numerical methods and get hands-on with computational physics.

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

No fluid appears or vanishes. The velocity field is divergence-free.

**Forces add up (Navier-Stokes):**
$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho} \nabla p + \nu \nabla^2 \mathbf{u}$$

Fluid accelerates from pressure differences and spreads out via viscosity.

| Symbol | Meaning |
|--------|---------|
| $\mathbf{u}$ | Velocity (u for x-direction, v for y-direction) |
| $p$ | Pressure |
| $\rho$ | Density |
| $\nu$ | Kinematic viscosity |

---

## How We Solve It

**Chorin Projection Method** — split the problem into predictor/corrector steps:

1. **Predictor** — Guess the velocity ignoring pressure:
   $$u^* = u^n - \Delta t(u \cdot \nabla)u + \Delta t \cdot \nu \nabla^2 u$$

2. **Poisson** — Find the pressure that makes the flow divergence-free:
   $$\nabla^2 p = \frac{\nabla \cdot u^*}{\Delta t}$$

3. **Corrector** — Apply pressure gradient to get the real velocity:
   $$u^{n+1} = u^* - \Delta t \nabla p$$

**Staggered Grid (Arakawa C-grid)** — velocities at cell faces, pressure at centers. This eliminates the pressure oscillations that plague simpler grids.

**Numerical schemes:**
- Advection: upwind (1st order) or central (2nd order) — pluggable
- Diffusion: Crank-Nicolson semi-implicit (unconditionally stable) or explicit Euler
- Pressure: Conjugate gradient with sparse matrix

| Term | What it means |
|------|---------------|
| C-grid | Staggered arrangement: u/v at faces, p at centers |
| Projection | Guess first, fix the pressure later |
| Divergence-free | Fluid coming in must go out |
| Crank-Nicolson | Semi-implicit diffusion, stable for any dt |

---

## Project Structure

```
CFD-Solver/
├── src/cfd_solver/             # The solver package
│   ├── solver/
│   │   ├── mesh.py             # Staggered grid generation
│   │   ├── bc.py               # Boundary conditions
│   │   ├── advection.py        # Upwind & central difference schemes
│   │   ├── diffusion.py        # Explicit Euler & Crank-Nicolson
│   │   ├── pressure.py         # Poisson solver (CG + sparse matrix)
│   │   ├── diagnostics.py      # CFL, divergence, blowup detection
│   │   ├── projection.py       # Chorin step orchestration
│   │   ├── solver.py           # Public API (Solver class)
│   │   └── viz.py              # Visualization (quiver + contour)
│   └── cli/
│       └── __init__.py         # CLI entry point
├── examples/                    # Ready-to-run simulations
├── output/                      # Results and plots
├── tests/                       # Unit tests (25 tests)
├── run_interactive.py           # Interactive parameter setup
├── setup.bat / setup.sh         # One-click environment setup
├── run.bat / run.sh             # One-click solver launcher
├── pyproject.toml               # Package configuration
└── requirements.txt             # Dependencies
```

---

## Getting Started

**1. Clone the repo:**
```bash
git clone https://github.com/baidik012/CFD-Solver.git
cd CFD-Solver
```

**2. Set up environment:**

- **Windows** — double-click `setup.bat`
- **Mac/Linux** — run `./setup.sh`

Both create a virtual environment and install everything automatically.

### Run a simulation

- **Windows** — double-click `run.bat`
- **Mac/Linux** — run `./run.sh`

You'll be asked for simulation parameters (grid size, viscosity, time steps, etc.) with sensible defaults. Just press Enter to accept the defaults. The solver runs and opens the result image automatically.

No code editing or YAML files needed.

### Run tests

First, activate the virtual environment:
```bash
# Mac/Linux
source venv/bin/activate

# Windows (Command Prompt)
venv\Scripts\activate

# Windows (Git Bash)
source venv/Scripts/activate
```

Then run:
```bash
pytest
```

All commands run from the root directory.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on branches, code style, and the PR process.

---

## License

Educational use only — not validated for industrial or safety-critical applications.
See [LICENSE](LICENSE).
