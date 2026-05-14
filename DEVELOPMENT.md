# Development Guide

Notes on how the solver works and how to extend it.

## How the Solver Is Organized

```
src/
├── solver.py      # Main loop: advance time, call each step
├── step_1.py      # Predictor: solve momentum without pressure
├── step_2.py      # Poisson: solve for pressure
├── step_3.py      # Corrector: update velocity with pressure gradient
├── grid.py        # Grid geometry and spacing
├── boundary.py   # Boundary conditions
└── utils.py      # Helpers: derivative approximations, norms
```

## Core Concepts

### The Grid

The domain is divided into a regular mesh. Every variable (velocity, pressure) lives at grid points.

- `Nx`, `Ny`: number of points in x and y
- `dx`, `dy`: spacing between points
- Boundaries are handled separately — interior points use the scheme

### Time Stepping

The main loop in `solver.py` advances from `t=0` to `t=final_time` in steps of `dt`.

```
for t in range(0, final_time, dt):
    u_star = predictor(u, dt)
    p = poisson(u_star, dt)
    u = corrector(u_star, p)
```

### Boundary Conditions

Where the fluid enters/exits or hits walls. Currently implemented:
- **No-slip walls**: velocity = 0 at the boundary
- **Periodic**: left edge flows into right edge (not implemented yet)
- **Inlet/outlet**: specify velocity or pressure (not implemented yet)

To add a new boundary type, edit `boundary.py`.

## Adding a New Test Case

1. Create a file in `examples/` — copy an existing one as a template
2. Define the geometry: `grid = Grid(Lx, Ly, Nx, Ny)`
3. Set initial conditions: `u0`, `p0`
4. Set boundary conditions: `bc = BoundaryConditions(...)`
5. Run the solver: `solver.solve(grid, u0, p0, bc)`
6. Plot or save the results

## Verifying Your Results

A common sanity check: the divergence of the final velocity should be close to zero.

```python
divergence = (np.gradient(u.x, dx) + np.gradient(u.y, dy))
print(f"Max divergence: {np.max(np.abs(divergence))}")
```

If it's not near machine epsilon (~1e-15), something's wrong in the pressure step.

## Physics Conventions

- All units are SI: meters, seconds, kg
- Positive u is to the right, positive v is up
- Pressure is relative — only pressure *differences* matter

## Debugging

If results look wrong:

1. **Check the CFL condition.** `dt` must be small enough:
   ```
   dt < min(dx, dy) / max(|u|)
   ```
   Violating this makes the scheme unstable.

2. **Plot intermediate steps.** Call the predictor, poisson, and corrector separately and inspect each field.

3. **Reduce resolution.** Start with a coarse grid (e.g., 32x32) to see problems faster.

4. **Print residuals.** The Poisson solver should converge — if it doesn't, the pressure field will be wrong.