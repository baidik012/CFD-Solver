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
- Diffusion: Crank-Nicolson semi-implicit (unconditionally stable) with pre-factorized direct LU solver
- Pressure: Direct sparse LU solver (pre-factorized at init) — fast constant-coefficient solve per step

| Term | What it means |
|------|---------------|
| C-grid | Staggered arrangement: u/v at faces, p at centers |
| Projection | Guess first, fix the pressure later |
| Divergence-free | Fluid coming in must go out |
| Direct Solver | Solve AX=B once by factoring A; very fast for constant matrices |

---

## Project Structure

```
CFD-Solver/
├── src/cfd_solver/             # The solver package
│   ├── solver/
│   │   ├── mesh.py             # Staggered grid generation
│   │   ├── bc.py               # Boundary conditions
│   │   ├── advection.py        # Upwind & central difference schemes
│   │   ├── diffusion.py        # Crank-Nicolson with direct LU solver
│   │   ├── pressure.py         # Poisson solver (direct LU decomposition)
│   │   ├── diagnostics.py      # CFL, divergence, blowup detection
│   │   ├── validate.py         # YAML config schema validation
│   │   ├── solver.py           # Public API (Chorin step & Solver class)
│   │   └── viz.py              # Visualization (quiver + contour)
│   └── cli/
│       └── __init__.py         # CLI entry point
├── examples/                    # Ready-to-run simulations
├── output/                      # Results and plots
├── tests/                       # Unit tests (46 tests)
├── run_interactive.py           # Interactive parameter setup
├── run_ghia_validation.py       # Ghia et al. (1982) benchmark validation
├── setup.bat / setup.sh         # One-click environment setup
├── run.bat / run.sh             # One-click solver launcher
├── pyproject.toml               # Package configuration
└── requirements.txt             # Dependencies
```

---

## System Requirements

- **Python** 3.10 or newer
- **OS:** Windows, macOS, or Linux
- **Hardware:** Any modern CPU (x86_64 or ARM). No GPU required.

**Original development machine:** Intel Core i7-13620H (10 cores, up to 5.0 GHz), 16 GB RAM, Linux.

**Expected performance (20 seconds simulated time, Re=100):**

| Grid | Time |
|------|------|
| 32×32 | 0.1 s |
| 64×64 | 0.2 s |
| 128×128 | 1.0 s |
| 256×256 | 10.5 s |
| 512×512 | 28.3 s |

The solver auto-scales `dt` to keep the simulation stable at fine grids. Smaller viscosity (higher Re) produces smaller `dt` values, increasing wall time. The default `simulation_time` adapts to your flow parameters automatically.

---

## Getting Started

**1. Clone the repo (or download a release):**
```bash
git clone https://github.com/baidik012/CFD-Solver.git
cd CFD-Solver
```
Or download a versioned zip from [Releases](https://github.com/baidik012/CFD-Solver/releases).

**2. Run the solver:**

- **Windows** — double-click `run.bat`
- **Mac/Linux** — run `./run.sh`

That's it. On first run, the script automatically creates a virtual environment and installs all dependencies. You'll be asked for simulation parameters (grid size, viscosity, simulation time, etc.) with sensible defaults. Just press Enter to accept the defaults. The solver runs and opens the result image automatically.

No code editing, YAML files, or separate setup step needed.

### Run tests

Install dev dependencies and run pytest:
```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

All commands run from the root directory.

### Updating

If you cloned the repo (not a release download), pull the latest changes:

- **Windows** — double-click `update.bat`
- **Mac/Linux:**
  ```bash
  ./update.sh
  ```

This fetches the latest code from GitHub and reinstalls dependencies if the virtual environment exists. If you downloaded a release zip, download the latest from [Releases](https://github.com/baidik012/CFD-Solver/releases).

The solver also checks for updates on startup and prints a notice if your copy is behind.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for details on branches, code style, and the PR process.

---

## License

Educational use only — not validated for industrial or safety-critical applications.
See [LICENSE](LICENSE).
