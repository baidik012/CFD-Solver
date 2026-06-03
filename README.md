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
- Advection: QUICK (3rd order)
- Diffusion: 2nd order central difference
- Time: Adams-Bashforth (2nd order)
- Pressure: Conjugate gradient with sparse matrix

| Term | What it means |
|------|---------------|
| C-grid | Staggered arrangement: u/v at faces, p at centers |
| Projection | Guess first, fix the pressure later |
| Divergence-free | Fluid coming in must go out |
| QUICK | Better accuracy than upwind, doesn't overshoot |

---

## Project Structure

```
CFD-Solver/
├── src/cfd_solver/         # The solver package
│   ├── solver/             # Core modules
│   │   ├── grid.py         # Staggered C-grid
│   │   ├── staggered_solver.py  # Production solver
│   │   ├── solver.py       # Simple solver (for learning)
│   │   ├── boundaries.py   # Boundary conditions
│   │   └── viz.py          # Plotting
│   └── cli/                # Command-line interface
├── examples/                # Ready-to-run simulations
├── output/                  # Results and plots
├── tests/                   # Unit tests
├── run_interactive.py       # Interactive parameter setup + solver launch
├── setup.bat                # Windows setup
├── setup.sh                 # Mac/Linux setup
├── run.bat                  # Windows one-click run
├── run.sh                   # Mac/Linux one-click run
├── pyproject.toml           # Package configuration
└── requirements.txt         # Dependencies
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

```bash
pytest tests/
```

All commands run from the root directory.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on branches, code style, and the PR process.

---

## License

Educational use only — not validated for industrial or safety-critical applications.
See [LICENSE](LICENSE).