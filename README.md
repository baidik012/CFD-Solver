# CFD Solver

A 2D incompressible Navier–Stokes solver using Chorin projection on a staggered grid.
Built by the Computational Club to learn numerical methods by writing them.

---

## The Physics

We simulate fluids that don't compress — water, air at low Mach numbers.

**Mass conservation (incompressibility):**

```math
\nabla \cdot \mathbf{u} = 0
```

Fluid neither appears nor vanishes. The velocity field is divergence-free.

**Momentum (Navier–Stokes):**

```math
\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu\nabla^2\mathbf{u}
```

Fluid accelerates from pressure gradients and diffuses momentum through viscosity.

| Symbol | Meaning |
|--------|---------|
| $`\mathbf{u} = (u, v)`$ | Velocity (x, y components) |
| $`p`$ | Pressure |
| $`\rho`$ | Density |
| $`\nu`$ | Kinematic viscosity |

---

## How We Solve It

**Chorin projection (predictor–corrector):**

1. **Predictor** — advance velocity ignoring pressure:

```math
u^* = u^n - \Delta t\,(\mathbf{u}\cdot\nabla)\mathbf{u} + \Delta t\,\nu\nabla^2\mathbf{u}
```

2. **Poisson** — find pressure that enforces $\nabla\cdot\mathbf{u}^{n+1}=0$:

```math
\nabla^2 p = \frac{\nabla\cdot u^*}{\Delta t}
```

3. **Corrector** — project onto divergence-free space:

```math
\mathbf{u}^{n+1} = u^* - \Delta t\,\nabla p
```

**Grid:** Staggered (Arakawa C-grid). Velocities live on cell faces, pressure at centers. This eliminates the checkerboard pressure mode that appears on collocated grids.

**Discretisation choices (pluggable):**
- Advection: upwind (1st order, stable) or central (2nd order, less diffusive)
- Diffusion: Crank–Nicolson semi-implicit — unconditionally stable. Small grids use sparse LU (`splu`); large grids use FFT spectral solver (DST‑I).
- Pressure: Poisson with `splu` + FFT spectral (DCT‑II) + periodic spectral. Auto-switches at 128 cells for O(N log N) scaling.

---

## What It Gets Right (Validation)

| Case | Reference | Grid / Re | Key Metrics |
|------|-----------|-----------|-------------|
| **Taylor–Green vortex** | Exact decaying solution | 64×64, $`\nu=0.01`$, t=2s | L2(u,v) ≈ 3.79×10⁻² |
| **Couette flow** | Analytical parallel-plate | 32×32, periodic x | 1st-order convergence |
| **Channel / Poiseuille** | Parabolic profile | 128×32, t=10s | L2 ≈ 4.34×10⁻⁴ |
| **Lid-driven cavity** | Ghia et al. (1982) | 128×128, Re=100 | u-L2 ≈ 0.0066, v-L2 ≈ 0.0046 |

All cases run from `examples/` with validation scripts:
```bash
python -m examples.taylor_green.validate
python -m examples.couette.validate
python -m examples.channel_flow.validate
python run_ghia_validation.py 100
```
Output images and convergence plots land in `images/`.

---

## Quick Start

```bash
git clone https://github.com/baidik012/CFD-Solver.git
cd CFD-Solver

# Windows
run.bat

# Mac / Linux
./run.sh
```

First run creates a virtual environment, installs deps, and prompts for parameters (grid, viscosity, time). Press Enter for defaults. Result image opens automatically.

**Run tests:**
```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest
```
184 tests pass.

**Update:**
```bash
# Windows
update.bat

# Mac / Linux
./run.sh
```
The solver also checks for updates on startup.

---

## Project Layout (Brief)

```
CFD-Solver/
├── src/cfd_solver/solver/   # Core numerics (mesh, bc, advection, diffusion, pressure, diagnostics, solver, viz)
├── examples/                # Cavity, Couette, Taylor-Green, Channel — each with config.yaml + run.py
├── tests/                   # 184 unit tests
├── images/                  # Generated validation plots
├── run.bat / run.sh         # One-click launchers
├── update.bat / update.sh   # One-click update
├── pyproject.toml           # Package config
└── requirements.txt
```

Full extension guide: [DEVELOPMENT.md](DEVELOPMENT.md)

Example gallery: [EXAMPLES.md](EXAMPLES.md)

---

## License

Educational use only — not validated for industrial or safety-critical applications.
See [LICENSE](LICENSE).