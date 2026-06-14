# CFD Solver

An internal tool for simulating incompressible fluid flow. Built by the club to learn numerical methods and get hands-on with computational physics.

---

## Table of Contents
- [The Physics](#the-physics)
- [How We Solve It](#how-we-solve-it)
- [Validation](#validation)
- [Examples](#examples)
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
- Diffusion: Crank-Nicolson semi-implicit (unconditionally stable) with FFT spectral solver (DST-I) for large grids
- Pressure: Poisson solver with FFT spectral method (DCT-II) for large grids — O(N log N) per step

| Term | What it means |
|------|---------------|
| C-grid | Staggered arrangement: u/v at faces, p at centers |
| Projection | Guess first, fix the pressure later |
| Divergence-free | Fluid coming in must go out |
| FFT Solver | Spectral diagonalization in frequency domain — O(N log N), no factorization |

---

## Validation

Tested against the benchmark data from [Ghia, Ghia & Shin (1982)](https://doi.org/10.1016/0021-9991(82)90055-1) for lid-driven cavity flow:

| Re | Grid | Advection | u-L2 | v-L2 |
|----|------|-----------|------|------|
| 100 | 128x128 | central | 0.007 | 0.005 |
| 400 | 128x128 | central | 0.011 | 0.035 |
| 1000 | 256x256 | upwind | 0.050 | 0.054 |

Reproduce with:
```bash
python run_ghia_validation.py 100
python run_ghia_validation.py 400
python run_ghia_validation.py 1000
```

---

## Examples

Bundled simulations with configs and scripts in `examples/`:

| Example | Key Feature | BCs |
|---------|-------------|-----|
| **Cavity** | Lid-driven recirculation | Smooth lid, no-slip walls |
| **Couette** | Parallel plate flow | No-slip top/bottom, periodic x |
| **Taylor-Green** | Decaying vortex (analytical) | Free-slip y, periodic x |
| **Channel** | Pressure-driven Poiseuille | Inlet/outlet or body force |

See [EXAMPLES.md](EXAMPLES.md) for output images, error analysis, and grid convergence results.

---

## Project Structure

```
CFD-Solver/
├── src/cfd_solver/             # The solver package
│   ├── solver/
│   │   ├── mesh.py             # Staggered grid generation
│   │   ├── bc.py               # Boundary conditions (wall types + periodic)
│   │   ├── advection.py        # Upwind & central difference schemes
│   │   ├── diffusion.py        # Crank-Nicolson (splu) & FFT (DST-I) solvers
│   │   ├── pressure.py         # Poisson solver: splu + FFT + periodic spectral
│   │   ├── diagnostics.py      # CFL, divergence, blowup detection
│   │   ├── validate.py         # YAML config schema validation
│   │   ├── solver.py           # Public API (Chorin step & Solver class)
│   │   └── viz.py              # Visualization (quiver + contour)
│   ├── utils.py                # Shared error handling utilities
│   └── cli/
│       └── __init__.py         # CLI entry point
├── examples/                    # Ready-to-run simulations
│   ├── cavity/                  # Lid-driven cavity
│   ├── channel_flow/            # Poiseuille flow (inlet or body force)
│   ├── couette/                 # Couette flow (periodic x)
│   └── taylor_green/            # Taylor-Green vortex (periodic x)
├── images/                      # Validation & example output plots
├── tests/                       # Unit tests (132 tests)
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

**Expected performance (200 time steps):**

| Grid | splu (s) | FFT (s) | Speedup |
|------|---------|---------|---------|
| 32×32 | 0.07 | — | — |
| 64×64 | 0.22 | — | — |
| 128×128 | 1.1 | 1.1 | 1× |
| 256×256 | 5.0 | 0.8 | 6× |
| 512×512 | 31.4 | 8.0 | 4× |
| 1024×1024 | ~170 | 58 | 3× |

For grids > 128, the solver automatically switches to FFT-based spectral solvers (DCT-II for pressure, DST-I for diffusion), giving O(N log N) performance instead of O(N^1.5).

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
