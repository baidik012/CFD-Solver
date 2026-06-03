# Development Guide

Notes on how the solver works and how to extend it.

## Project Structure

```
CFD-Solver/
├── src/cfd_solver/
│   ├── solver/
│   │   ├── grid.py             # Staggered C-grid (u/v at faces, p at centers)
│   │   ├── solver.py           # Original colocated solver (for learning)
│   │   ├── staggered_solver.py # Production solver (accurate, efficient)
│   │   ├── boundaries.py       # Boundary conditions (used by Solver only)
│   │   └── viz.py              # Plotting utilities
│   └── cli/
│       └── __init__.py         # CLI entry point
├── examples/                   # Example scripts and configs
├── output/                     # Solver output images
├── tests/                      # Unit tests
├── run_interactive.py          # Interactive parameter prompts + solver launch
├── setup.bat / setup.sh        # One-click environment setup
├── run.bat / run.sh            # One-click solver launcher
├── pyproject.toml              # Package config
└── requirements.txt            # Dependencies
```

## Two Solvers

**Original (`Solver`)** — Good for learning, simpler code:
- Collocated grid (all variables at cell centers)
- Forward Euler time integration
- Upwind advection (1st order)
- Jacobi pressure solver

**Staggered (`StaggeredSolver`)** — For research and accuracy:
- Arakawa C-grid (prevents odd-even decoupling)
- Adams-Bashforth time integration (2nd order)
- QUICK advection (3rd order)
- Conjugate gradient + sparse matrix for pressure

Use the staggered solver for actual projects.

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

### Time Integration

Adams-Bashforth (AB2) for the advection term:
```
u^{n+1} = u^n + dt*(1.5*f^n - 0.5*f^{n-1})
```
This is 2nd order accurate and has better stability than forward Euler.

### Boundary Conditions

The staggered solver accepts arbitrary BCs via dictionaries:
```python
solver = StaggeredSolver(
    ..., u_bc={"top": 1.0, "bottom": 0.0, "left": 0.0, "right": 0.0},
            v_bc={"top": 0.0, "bottom": 0.0, "left": 0.0, "right": 0.0})
```

`_set_bc(u, v)` applies the BCs to any velocity array. `_apply_bc()` is a convenience wrapper for `self.u` / `self.v`.

Currently implemented:
- **No-slip walls**: velocity = 0 (default for all sides)
- **Moving lid**: specify `u_bc={"top": U}` for lid-driven cavity
- **Arbitrary values**: set any side to any constant

Not yet implemented:
- **Periodic**: left flows into right
- **Inlet/outlet**: specify velocity or pressure at open boundaries

## Adding a New Test Case

1. Create a file in `examples/` — copy `staggered_cavity.py` as a template
2. Instantiate the solver:
   ```python
   from cfd_solver.solver.staggered_solver import StaggeredSolver
   from cfd_solver.solver.viz import save_velocity_contour

   s = StaggeredSolver(Lx=1.0, Ly=1.0, Nx=128, Ny=128,
                        nu=0.01, dt=0.001,
                        u_bc={"top": 1.0})
   s.solve(steps=1000, verbose=True)
   save_velocity_contour(s, "output/my_result.png")
   ```
3. Check divergence: `s.max_divergence()` (should be < 1e-6)

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

**Reference data** — Lid-driven cavity has established benchmarks (Ghia et al., 1982). Compare centerline velocities:
```
u_center at x=0.5 should match published results
v_center at y=0.5 should match published results
```

## Stability Limits

Explicit diffusion has a stability limit:
```
dt <= dx² / (4 * nu)
```

For a 32x32 grid (`dx = 1/32 ≈ 0.031`) with `nu = 0.01`, the limit is `dt ≈ 0.0024`. Exceeding this causes the solver to blow up.

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

5. **Check the pressure solver.** If CG doesn't converge in < 100 iterations, the matrix is wrong.

## Performance Tips

- 32x32 is good for quick tests (< 1 second)
- 128x128 for standard runs (~1-2 seconds)
- 256x256 for detailed runs (~5-10 seconds)
- 512x512 for publication quality (~30-60 seconds)

For large grids, the conjugate gradient solver converges in ~20-50 iterations instead of 50 full Jacobi passes — that's the main speedup over the original solver.

## Running Tests

```bash
pytest tests/
```

All 9 tests should pass. Tests cover grid construction, boundary conditions, time-stepping, divergence, CFL, and pressure symmetry.
