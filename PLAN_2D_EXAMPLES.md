# Plan: 2D CFD Test Cases Implementation

## Suggested Examples (Ordered by Difficulty)

### 1. Easy / Immediate Wins
1. **Poiseuille / Channel Flow** — Fully developed flow between parallel plates. Tests inlet/outlet BCs. Analytical solution available.
2. **Couette Flow** — Shear-driven channel, moving wall. Simple extension of lid-driven cavity.
3. **Taylor-Green Vortex** — Decaying vortices with periodic BCs. Exact analytical solution. Tests temporal accuracy.

### 2. Medium Difficulty
4. **Backward-Facing Step** — Flow separation & reattachment. Classic benchmark (Armaly et al.).
5. **Flow Over a Circular Cylinder** — Vortex shedding (Karman street). Tests outflow BCs and immersed boundary.
6. **Natural Convection in a Cavity** — Boussinesq approximation. Adds buoyancy + scalar transport.

### 3. More Advanced
7. **Double Shear Layer / Kelvin-Helmholtz** — Periodic domain, thin shear layer roll-up.
8. **Oscillating Lid** — Time-varying BCs.
9. **Pulsating Inlet** — Unsteady forcing.

---

## Current State Summary

2D incompressible Navier-Stokes solver, Chorin's projection, Arakawa C-grid.
**Hardcoded for lid-driven cavity** with solid walls on all four sides.

| Gap | Blocks |
|-----|--------|
| No inlet/outlet BCs | Channel flow, Poiseuille, backward-facing step, cylinder |
| No periodic BCs | Taylor-Green, double shear layer |
| No body forces | Natural convection (buoyancy), pressure-driven channel |
| No initial condition API | Taylor-Green, double shear layer |
| No scalar transport | Natural convection |
| No obstacle/immersed boundary | Cylinder, backward-facing step |
| No time-varying BCs | Oscillating lid, pulsating inlet |
| No steady-state convergence check | All steady-state validation cases |

---

## Phase 0: Flexible Boundary Condition System (Foundation)

Everything else depends on this. The current `BoundaryConditions` class only supports solid walls.

### 0.1 Redesign `bc.py`

New API:
```python
from cfd_solver.solver.bc import BoundaryConditions, WallType

bc = BoundaryConditions(
    left=WallType.inlet(profile="parabolic", U_max=1.0),
    right=WallType.outlet(method="zero_gradient"),
    top=WallType.wall(u=0.0, v=0.0),
    bottom=WallType.wall(u=0.0, v=0.0),
)
```

Wall types:
- `wall(u=0.0, v=0.0)` — no-slip wall (existing behavior)
- `free_slip(u=0.0, v=0.0)` — symmetry/free-slip
- `inlet(profile="parabolic"|"uniform", U_max=1.0)` — specified velocity profile
- `outlet(method="zero_gradient"|"convective")` — outflow BC
- `periodic()` — wraps to opposite wall

Changes to `bc.apply()`:
- Normal velocity at inlet: set to inlet profile value
- Normal velocity at outlet: one-sided extrapolation
- Tangential velocity at inlet: from profile
- Ghost cells for outlet: copy interior (zero-gradient)
- Periodic: copy from opposite wall

### 0.2 Pressure Solver Changes

- At outlet: pin pressure to reference value (p=0) instead of zero-mean
- Critical for channel/pipe flows with pressure-driven flow

### 0.3 Crank-Nicolson Matrix Changes

- CN matrices assume uniform Dirichlet walls — need per-wall customization
- Option: use explicit diffusion for inlet/outlet cases (simpler, stable for moderate Re)
- Option: rebuild CN matrices (expensive, ~O(N) per step)

### 0.4 YAML Schema Update

```yaml
boundary:
  left:   { type: inlet, profile: parabolic, U_max: 1.0 }
  right:  { type: outlet, method: zero_gradient }
  top:    { type: wall, u: 0.0, v: 0.0 }
  bottom: { type: wall, u: 0.0, v: 0.0 }
```

### 0.5 Files to Modify
- `src/cfd_solver/solver/bc.py` — complete rewrite
- `src/cfd_solver/solver/pressure.py` — outlet pressure pinning
- `src/cfd_solver/solver/diffusion.py` — per-wall CN matrices
- `src/cfd_solver/solver/solver.py` — step function for new BCs
- `src/cfd_solver/solver/validate.py` — new schema
- `src/cfd_solver/cli/__init__.py` — parse new BC config
- `tests/test_core.py` — new BC tests

**Estimated effort:** ~400-500 lines

---

## Phase 1: Body Forces & Initial Conditions

### 1.1 Body Force Term
```python
# In Solver.__init__:
self._body_force_fn = None  # callable(u, v, t) -> (fu, fv)

# In Solver.step():
if self._body_force_fn is not None:
    fu, fv = self._body_force_fn(self.u, self.v, self.time)
    u_star[1:-1, 1:-1] += dt * fu
    v_star[1:-1, 1:-1] += dt * fv
```

### 1.2 Initial Condition API
```python
self._initial_condition_fn = None  # callable(mesh) -> (u, v, p)
```

### 1.3 Time Tracking
```python
self.time = 0.0
# In step(): self.time += self.dt
```

### 1.4 Steady-State Convergence Check
```python
def solve(self, ..., convergence_tol=None, convergence_window=100):
    # Track residual norm over window
    # Stop if max|u^{n+1} - u^n| < tol for convergence_window steps
```

**Estimated effort:** ~200 lines

---

## Phase 2: Easy Examples

### 2.1 Poiseuille / Channel Flow

- Domain: Lx=5.0, Ly=1.0
- BCs: inlet (parabolic), outlet (zero-gradient), top/bottom walls
- Analytical: `u(y) = (1/2*nu) * (-dp/dx) * y * (H - y)`
- Files: `examples/channel_flow.yaml`, `examples/channel_flow.py`, `examples/validate_channel.py`

### 2.2 Couette Flow

- Domain: Lx=1.0, Ly=1.0
- BCs: all walls, top u=U, bottom u=0
- Analytical (transient): Fourier series solution
- Steady: `u(y) = U*y/H` (linear)
- Files: `examples/couette_flow.yaml`, `examples/couette_flow.py`, `examples/validate_couette.py`

### 2.3 Taylor-Green Vortex

- Domain: [0, 2pi] x [0, 2pi]
- BCs: periodic in x AND y
- IC: `u = -U*cos(x)*sin(y)`, `v = U*sin(x)*cos(y)`
- Analytical: `u(x,y,t) = -U*cos(x)*sin(y)*exp(-2*nu*t)`
- Files: `examples/taylor_green.yaml`, `examples/taylor_green.py`, `examples/validate_taylor_green.py`

**Estimated effort:** ~300 lines

---

## Phase 3: Immersed Boundary / Obstacle Support

### 3.1 Mask-Based Approach

```python
# In Mesh:
self.mask = np.zeros((Nx, Ny), dtype=bool)  # True = solid cell

# Utility:
def mark_cells(mesh, geometry_type, **params):
    if geometry_type == "step":
        # backward-facing step geometry
    elif geometry_type == "cylinder":
        # mark cells inside cylinder
```

### 3.2 Solver Modifications

- Advection: skip flux across solid-fluid interfaces
- Diffusion: solid cells = Dirichlet (u=0) at interfaces
- Pressure: solid cells excluded from Poisson solve
- Simplest: set u=v=0 in solid cells every step

**Estimated effort:** ~300 lines

---

## Phase 4: Medium Examples

### 4.1 Backward-Facing Step

- Domain: Lx=4.0, Ly=2.0 with step geometry
- BCs: inlet (parabolic), outlet (zero-gradient), walls
- Benchmark: Armaly et al. (1983) reattachment length vs Re
- Files: `examples/backward_step.yaml`, `examples/backward_step.py`, `examples/validate_backward_step.py`

### 4.2 Flow Over a Circular Cylinder

- Domain: Lx=8.0, Ly=4.0, cylinder at (2.0, 2.0), D=0.5
- BCs: inlet (uniform), outlet (convective), top/bottom (free-slip)
- Re = U*D/nu = 40-200
- Benchmark: Cantwell & Coles (1983), Park et al. (1998)
- Additional: lift/drag computation
- Files: `examples/cylinder_flow.yaml`, `examples/cylinder_flow.py`, `examples/validate_cylinder.py`

**Estimated effort:** ~500 lines

---

## Phase 5: Scalar Transport

### 5.1 Temperature Field

```python
# New module: scalar.py
class ScalarTransport:
    def __init__(self, mesh, dt, kappa, bc_T): ...
    def step(self, u, v, T): ...
```

### 5.2 Boussinesq Coupling

```python
def boussinesq_force(T, T_ref, beta, g):
    fv = beta * (T - T_ref) * g
    return 0, fv
```

### 5.3 Temperature BCs
- Dirichlet: T = T_hot on left, T = T_cold on right
- Neumann: dT/dn = 0 on adiabatic walls

**Estimated effort:** ~400 lines

---

## Phase 6: Natural Convection in a Cavity

- Domain: unit square
- BCs: all walls no-slip, left T=T_hot, right T=Cold, top/bottom adiabatic
- Parameters: Ra = 10^3 to 10^6
- Benchmark: de Vahl Davis (1983) Nusselt number, centerline velocities
- Files: `examples/natural_convection.yaml`, `examples/natural_convection.py`, `examples/validate_natural_convection.py`

**Estimated effort:** ~300 lines

---

## Phase 7: Advanced Examples

### 7.1 Double Shear Layer
- Periodic in both directions
- IC: `u = U*tanh(k*(y - Ly/4))` for y < Ly/2, else `-U*tanh(k*(y - 3Ly/4))`
- Thin shear layer rolls up into vortices

### 7.2 Oscillating Lid
- `u_lid(t) = U * sin(omega*t)`
- Requires time-varying BCs

### 7.3 Pulsating Inlet
- `u_inlet(t) = U_mean * (1 + A*sin(omega*t))`

**Estimated effort:** ~200 lines each

---

## Implementation Order & Dependencies

```
Phase 0 (Flexible BCs)
  +-- Phase 1 (Body Forces + ICs + Time)
  |     +-- Phase 2 (Easy Examples)
  |     +-- Phase 7 (Oscillating Lid, Pulsating Inlet)
  +-- Phase 3 (Immersed Boundary)
  |     +-- Phase 4 (Backward Step, Cylinder)
  +-- Phase 5 (Scalar Transport)
        +-- Phase 6 (Natural Convection)

Phase 7.1 (Double Shear Layer) needs Phase 1 only
```

**Recommended execution order:**
1. Phase 0 — Foundation (~400-500 lines)
2. Phase 1 — Body forces + ICs (~200 lines)
3. Phase 2 — Easy examples (~300 lines)
4. Phase 3 — Immersed boundary (~300 lines)
5. Phase 4 — Backward step + cylinder (~500 lines)
6. Phase 5 — Scalar transport (~400 lines)
7. Phase 6 — Natural convection (~300 lines)
8. Phase 7 — Advanced examples (~400 lines)

**Total estimated effort:** ~2,800-3,000 lines + ~1,000 lines of tests

---

## Risk Areas & Design Decisions

1. **Crank-Nicolson with non-uniform BCs:** CN matrices assume uniform Dirichlet walls. With inlet/outlet, 1D operators change per wall.
   - Option A: Rebuild CN matrices each step (expensive)
   - Option B: Use explicit diffusion for inlet/outlet cases (simpler)
   - Option C: Per-wall CN matrix customization (moderate complexity)

2. **Pressure pinning:** Channel flow needs fixed reference pressure at outlet instead of zero-mean.

3. **Immersed boundary accuracy:** Simple masking = first-order near obstacles. Ghost-cell immersed boundary = higher order but more complex.

4. **Backward-facing step geometry:** Masked cells (stair-step on coarse grids) vs non-uniform grid (not yet supported) vs L-shaped domain (complex).

5. **Convective outflow BC:** Requires storing previous time step + convective velocity estimate. Adds memory/complexity.

---

## Key Files

| File | Changes |
|------|---------|
| `src/cfd_solver/solver/bc.py` | Complete rewrite — per-wall type system |
| `src/cfd_solver/solver/solver.py` | Body forces, IC API, time tracking, convergence check |
| `src/cfd_solver/solver/pressure.py` | Outlet pressure pinning |
| `src/cfd_solver/solver/diffusion.py` | Per-wall CN matrices |
| `src/cfd_solver/solver/mesh.py` | Obstacle mask support |
| `src/cfd_solver/solver/validate.py` | New YAML schema |
| `src/cfd_solver/cli/__init__.py` | Parse new BC config |
| `src/cfd_solver/solver/scalar.py` | NEW — temperature transport |
| `tests/test_core.py` | New tests for all phases |
| `examples/*.yaml` | Config files for each example |
| `examples/*.py` | Runner scripts |
| `examples/validate_*.py` | Validation against benchmarks |
