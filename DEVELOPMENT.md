# Development Guide

How the solver works and how to extend it.

## Project Structure

```
CFD-Solver/
├── src/cfd_solver/
│   ├── solver/           # Core numerical modules
│   │   ├── mesh.py       # Staggered grid (C-grid)
│   │   ├── bc.py         # Boundary conditions (WallType hierarchy)
│   │   ├── advection.py  # Upwind & central schemes
│   │   ├── diffusion.py  # Explicit, Crank-Nicolson (splu), FFT-CN (DST-I)
│   │   ├── pressure.py   # Poisson: splu + FFT (DCT-II) + periodic spectral
│   │   ├── diagnostics.py# CFL, divergence, blowup detection
│   │   ├── validate.py   # YAML config schema
│   │   ├── solver.py     # Public API: Chorin step, Solver class
│   │   └── viz.py        # Visualization
│   ├── cli/              # CLI entry point
│   ├── config_loader.py  # Load + validate YAML
│   ├── validation.py     # Error norms, convergence helpers
│   ├── utils.py          # Friendly error handling
│   └── version_check.py  # Cached git update check
├── examples/             # Ready-to-run cases
├── tests/                # 173 tests
└── pyproject.toml
```

## Core Concepts

### Staggered Grid (C-Grid)
```
      v[j+1]
   ──►──────
u[i] │    P[i,j]
   ──►──────
      v[j]
```
- `u` (x-velocity) at x-faces: `(Nx+1, Ny+2)`
- `v` (y-velocity) at y-faces: `(Nx+2, Ny+1)`
- `p` (pressure) at cell centers: `(Nx+2, Ny+2)`

Eliminates checkerboard pressure instability.

### Algorithm (Chorin Splitting)
1. **Predictor:** `u* = u^n - dt*(u·∇)u + dt*ν∇²u`
2. **Poisson:** `∇²p = ∇·u* / dt`
3. **Corrector:** `u^{n+1} = u* - dt*∇p`

### Boundary Conditions
```python
from cfd_solver.solver import Solver, BoundaryConditions, NoSlipWall, InletWall, PeriodicWall

# Constant lid
s = Solver(grid_size=(64,64), nu=0.01, dt=0.001, lid_speed=1.0)

# Smooth sinusoidal lid (removes corner singularity)
s = Solver(grid_size=(64,64), nu=0.01, dt=0.001, lid_speed=1.0, smooth_lid=True)

# Per-wall objects (new API)
bc = BoundaryConditions(
    top=NoSlipWall(u=1.0),
    bottom=NoSlipWall(u=0.0),
    left=PeriodicWall(),
    right=PeriodicWall(),
)
s = Solver(grid_size=(64,64), nu=0.01, dt=0.001, boundary_config=bc, force=True)
```

Smooth lid applies `u(x) = U * sin(πx/L)` — zero at corners, max at center.

### Advection Schemes
```python
s = Solver(..., advection_scheme="upwind")   # 1st order, diffusive, stable
s = Solver(..., advection_scheme="central")  # 2nd order, less diffusive, may oscillate
```

### Adding a New Advection Scheme
1. Add function in `advection.py` returning `(adv_u, adv_v)`
2. Register in `solver.py` constructor: `self._advection_fn = advection.my_scheme`
3. Add test in `tests/test_core.py`

## Verifying Results

**Mass conservation:**
```python
print(f"Max divergence: {s.max_divergence():.2e}")   # ~1e-6
print(f"L2 divergence: {s.divergence_norm():.2e}")
```

**CFL stability (must be < 1):**
```python
print(f"CFL: {s.cfl():.3f}")
```

**Ghia benchmark (Re=100):**
```bash
python run_ghia_validation.py 100
```

## Stability Limits

| Grid | Max stable dt | Notes |
|------|--------------|-------|
| 32×32 | Any (tested to 0.1) | Fully stable |
| 64×64 | ~0.001 for 100+ steps | Advection limits long runs |
| 128×128 | ~0.001 for ~30 steps | Advection instability earlier |

Use `smooth_lid=True` (default) to push limits further.

## Performance

For grids > 128×128, auto-switches from sparse LU (`splu`) to **FFT spectral solvers** (DCT-II pressure, DST-I diffusion). Threshold: 128 cells.

| Grid | splu | FFT | Speedup |
|------|------|-----|---------|
| 512×512 | 0.157 s/step | 0.04 s/step | ~4× |
| 1024×1024 | OOM | 0.29 s/step | — |

## Checkpointing

```python
s.checkpoint("output/run.npz")
s = Solver.from_checkpoint("output/run.npz")
s.solve(1000)
```

Stores `u, v, p` + all solver params. Old checkpoints remain loadable (missing keys → defaults).

## Running Tests

```bash
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

All 173 tests should pass. Coverage: mesh, BCs, advection, diffusion, pressure, diagnostics, viz, checkpoint/resume, Solver API, body forces, ICs, time tracking, convergence, periodic BCs, example validation.