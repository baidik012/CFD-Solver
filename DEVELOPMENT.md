# Development Guide

Notes on how the solver works and how to extend it.

## Project Structure

```
CFD-Solver/
├── src/cfd_solver/
│   ├── solver/
│   │   ├── mesh.py             # Staggered grid generation
│   │   ├── bc.py               # Boundary conditions & smooth profiles
│   │   ├── advection.py        # Pluggable advection schemes
│   │   ├── diffusion.py        # Explicit Euler & Crank-Nicolson (direct LU)
│   │   ├── pressure.py         # Poisson solver (direct LU decomposition)
│   │   ├── diagnostics.py      # CFL, divergence, blowup detection
│   │   ├── validate.py         # YAML config schema validation
│   │   ├── solver.py           # Public API (Chorin step & Solver class)
│   │   └── viz.py              # Visualization (quiver + contour)
│   └── cli/
│       └── __init__.py         # CLI entry point
├── examples/                    # Example scripts and configs
├── output/                      # Solver output images
├── tests/                       # Unit tests (26 tests)
├── run_interactive.py           # Interactive parameter prompts
├── run_ghia_validation.py       # Ghia et al. (1982) benchmark validation
├── setup.bat / setup.sh         # One-click environment setup
├── run.bat / run.sh             # One-click solver launcher
├── pyproject.toml               # Package config
└── requirements.txt             # Dependencies
```

## Architecture

The solver is split into focused modules with clear responsibilities:

| Module | Responsibility |
|--------|---------------|
| `mesh.py` | Grid generation, coordinate arrays, spacing |
| `bc.py` | Boundary conditions, smooth lid profiles |
| `advection.py` | Upwind (1st order) & central (2nd order) schemes |
| `diffusion.py` | Explicit Euler & Crank-Nicolson (matrices pre-factorized with `splu`) |
| `pressure.py` | Poisson matrix assembly + direct LU solve (pre-factorized at init) |
| `diagnostics.py` | CFL, divergence norms, blowup detection |
| `validate.py` | YAML config schema validation |
| `solver.py` | Public API: Chorin step, `Solver` class, checkpoint/resume |
| `viz.py` | Unified visualization (quiver + contour) |

## Core Concepts

### Staggered Grid (C-Grid)

```
       v[j+1]
    ──►──────
 u[i] │    P[i,j]
    ──►──────
       v[j]
```

- `u` (x-velocity) stored at cell faces: `(Nx+1, Ny)`
- `v` (y-velocity) stored at cell faces: `(Nx, Ny+1)`
- `p` (pressure) stored at cell centers: `(Nx, Ny)`

This arrangement avoids the "checkerboard" pressure instability that plagues colocated grids.

### The Algorithm (Chorin Splitting)

1. **Predictor** — Solve momentum without pressure:
   ```
   u* = u^n - dt*(u·∇)u + dt*ν∇²u
   ```

2. **Poisson** — Find pressure that enforces divergence-free:
   ```
   ∇²p = ∇·u* / dt
   ```

3. **Corrector** — Apply pressure gradient:
   ```
   u^{n+1} = u* - dt*∇p
   ```

### Boundary Conditions

```python
from cfd_solver.solver import Solver

# Constant lid
s = Solver(grid_size=(64, 64), nu=0.01, dt=0.001, lid_speed=1.0)

# Smooth sinusoidal lid (removes corner singularity)
s = Solver(grid_size=(64, 64), nu=0.01, dt=0.001, lid_speed=1.0, smooth_lid=True)
```

The smooth lid applies `u(x) = U * sin(πx/L)` on the top wall — zero at corners, maximum in the center. This removes the velocity discontinuity that causes instabilities at fine grids.

### Pluggable Advection Schemes

```python
from cfd_solver.solver import Solver

# First-order upwind (diffusive but stable)
s = Solver(grid_size=(64, 64), nu=0.01, dt=0.001, advection_scheme="upwind")

# Second-order central (less diffusive, may oscillate)
s = Solver(grid_size=(64, 64), nu=0.01, dt=0.001, advection_scheme="central")
```

## Adding a New Advection Scheme

1. Add a function in `src/cfd_solver/solver/advection.py`:
   ```python
   def my_scheme(u, v, dx, dy):
       """My custom advection scheme."""
       adv_u = np.zeros_like(u)
       adv_v = np.zeros_like(v)
       # ... compute advection at interior faces ...
       return adv_u, adv_v
   ```

2. Register it in `src/cfd_solver/solver/solver.py`:
   ```python
   if advection_scheme == "my_scheme":
       self._advection_fn = advection.my_scheme
   ```

3. Add a test in `tests/test_core.py`.

## Verifying Your Results

**Mass conservation** — The divergence should be near zero:
```python
print(f"Max divergence: {s.max_divergence():.2e}")
print(f"L2 divergence: {s.divergence_norm():.2e}")
```

**CFL stability** — Must be < 1:
```python
print(f"CFL: {s.cfl():.3f}")
if s.cfl() > 1:
    print("WARNING: CFL > 1, reduce dt")
```

**Reference data** — Lid-driven cavity has established benchmarks (Ghia et al., 1982). Use `run_ghia_validation.py` for a direct comparison:
```bash
python run_ghia_validation.py
```
The script runs Re=100 (128×128 grid, 10 000 steps) and prints u- and v-centerline profiles for comparison against published Ghia tables.

## Stability Limits

**Diffusion:** Crank-Nicolson semi-implicit scheme is unconditionally stable for any dt. No diffusion stability constraint.

**Advection:** The 1st-order upwind advection is conditionally stable. At finer grids, the numerical dissipation per cell decreases, which can cause instabilities at high resolution. Practical limits:

| Grid | Max stable dt | Notes |
|------|--------------|-------|
| 32×32 | Any (tested to 0.1) | Fully stable |
| 64×64 | ~0.001 for 100+ steps | Advection limits long runs |
| 128×128 | ~0.001 for ~30 steps | Advection instability earlier |

Use `smooth_lid=True` (default) to push these limits further.

## Physics Conventions

- All units are SI: meters, seconds, kg
- Positive u is to the right, positive v is up
- Pressure is relative — only pressure *differences* matter
- Grid indexing: `u[i, j]` where i = x index, j = y index

## Debugging

If results look wrong:

1. **Check CFL first.** Unstable simulations show diverging velocities.
   ```python
   if s.cfl() > 1:
       print("Reduce dt!")
   ```

2. **Check divergence.** Non-zero divergence means pressure solver failed.
   ```python
   print(f"Divergence: {s.max_divergence():.2e}")  # should be ~1e-6
   ```

3. **Reduce resolution.** A 32x32 grid runs fast and reveals algorithm bugs.

4. **Plot intermediate results.** Call `s.step()` once, inspect `s.u`, `s.v`, `s.p`.

5. **Check the pressure solver.** The solver uses a pre-factorized direct LU solve; if pressure values look wrong, check the Poisson matrix assembly in `pressure.py`.

## Performance Tips

- 32x32 is good for quick tests (< 1 second)
- 128x128 for standard runs (~1-2 seconds)
- 256x256 for detailed runs (~5-10 seconds)

Both the pressure Poisson matrix and the Crank-Nicolson diffusion matrices are pre-factorized via `splu` at `Solver.__init__`. Each time step only does a fast triangular solve — there is no iterative inner loop.

## Checkpointing

The solver can save and resume state:

```python
# Save state after N steps
s.checkpoint("output/run.npz")

# Resume from checkpoint
s = Solver.from_checkpoint("output/run.npz")
s.solve(1000)  # continue running
```

The `.npz` file stores `u`, `v`, `p`, and all solver parameters (`Nx`, `Ny`, `Lx`, `Ly`, `dt`, `nu`, `lid_speed`, `smooth_lid`, `advection_scheme`, `diffusion_scheme`). Missing optional keys fall back to defaults, so old checkpoints remain loadable.

## Running Tests

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

All 26 tests should pass. Tests cover mesh construction, boundary conditions, advection schemes, diffusion, pressure solving, diagnostics, visualization, checkpoint/resume, and the full Solver API.
