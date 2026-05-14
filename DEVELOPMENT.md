# Development Guide

Notes on how the solver works and how to extend it.

## How the Solver Is Organized

```
src/cfd_solver/
├── solver/
│   ├── grid.py             # Staggered C-grid (u/v at faces, p at centers)
│   ├── solver.py           # Original colocated solver (for learning)
│   ├── staggered_solver.py # Production solver (accurate, efficient)
│   ├── boundaries.py       # Boundary conditions
│   └── viz.py              # Plotting utilities
└── cli/
    └── __init__.py         # CLI entry point
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

Currently supports the lid-driven cavity (moving top wall). Other BCs planned:
- **No-slip walls**: velocity = 0
- **Periodic**: left flows into right (not implemented)
- **Inlet/outlet**: specify velocity or pressure (not implemented)

To add a new boundary type, edit `_apply_bc()` in `staggered_solver.py`.

## Adding a New Test Case

1. Create a file in `examples/` — copy `staggered_cavity.py` as a template
2. Instantiate the solver:
   ```python
   from cfd_solver.solver import StaggeredSolver

   solver = StaggeredSolver(Lx=1.0, Ly=1.0, Nx=128, Ny=128,
                            nu=0.01, dt=0.001,
                            u_bc={"top": 1.0, "bottom": 0.0, ...})
   ```
3. Run it: `solver.solve(steps=1000)`
4. Check divergence: `solver.max_divergence()` (should be < 1e-6)
5. Plot results: `from cfd_solver.solver.viz import save_velocity_contour`

## Verifying Your Results

**Mass conservation** — The divergence should be near zero:
```python
print(f"Max divergence: {solver.max_divergence():.2e}")
print(f"L2 divergence: {solver.divergence_norm():.2e}")
```

**CFL stability** — Must be < 1:
```python
print(f"CFL: {solver.cfl():.3f}")
if solver.cfl() > 1:
    print("WARNING: CFL > 1, reduce dt")
```

**Reference data** — Lid-driven cavity has established benchmarks (Ghia et al., 1982). Compare centerline velocities:
```
u_center at x=0.5 should match published results
v_center at y=0.5 should match published results
```

## Physics Conventions

- All units are SI: meters, seconds, kg
- Positive u is to the right, positive v is up
- Pressure is relative — only pressure *differences* matter
- Grid indexing: `u[i, j]` where i = x index, j = y index

## Debugging

If results look wrong:

1. **Check CFL first.** Unstable simulations show diverging velocities.
   ```python
   if solver.cfl() > 1:
       print("Reduce dt!")
   ```

2. **Check divergence.** Non-zero divergence means pressure solver failed.
   ```python
   print(f"Divergence: {solver.max_divergence():.2e}")  # should be ~1e-6
   ```

3. **Reduce resolution.** A 32×32 grid runs fast and reveals algorithm bugs.

4. **Plot intermediate results.** Call `step()` once, inspect `solver.u`, `solver.v`, `solver.p`.

5. **Check the pressure solver.** If CG doesn't converge in < 100 iterations, the matrix is wrong.

## Performance Tips

- 128×128 is good for quick tests (< 1 second)
- 256×256 for detailed runs (~2-5 seconds)
- 512×512 for publication quality (~15-30 seconds)
- Your 16 GB RAM handles 1024×1024 easily (~100 MB)

For large grids, the conjugate gradient solver converges in ~20-50 iterations instead of 50 full Jacobi passes — that's the main speedup over the original solver.