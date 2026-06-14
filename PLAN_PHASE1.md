# Plan: Phase 1 — Body Forces, Initial Conditions, Time Tracking, Convergence

**Date:** 2026-06-14
**Branch:** feature/2d-examples
**Depends on:** Phase 0 (complete, 87/87 tests passing)

---

## Overview

Phase 1 adds four capabilities that are prerequisites for the easy examples
(Couette transient, Taylor-Green, body-force-driven Poiseuille) and later phases:

| Feature | Why needed |
|---------|-----------|
| Body force | Taylor-Green decay, periodic Poiseuille, natural convection Boussinesq |
| Initial conditions | Couette transient starts from rest; Taylor-Green starts from cosine IC |
| Time tracking | Transient validation against analytical solutions (Couette, Taylor-Green) |
| Steady-state check | Stop early when channel/lid-driven cavity has converged |

---

## Detailed Design

### 1. Body Forces

**What:** A user-supplied callable `(u, v, t) → (fu, fv)` that adds external
forces to the momentum equations each time step.

**Where in `solver.py`:**

```
__init__:
    self._body_force_fn = None

step() — insert after diffusion, before pressure:
    # Current flow:
    #   adv_u, adv_v = advection(...)
    #   u_star, v_star = diffusion(...)
    #   p = pressure.solve(u_star, v_star, dt)
    #   u = u_star - dt * grad(p)
    #
    # New flow:
    #   adv_u, adv_v = advection(...)
    #   u_star, v_star = diffusion(...)
    #   u_star += dt * fu          ← NEW
    #   v_star += dt * fv          ← NEW
    #   p = pressure.solve(u_star, v_star, dt)
    #   u = u_star - dt * grad(p)
```

The force is applied to `u_star` (the intermediate velocity), **before** the
pressure projection. This is physically correct: the pressure step enforces
divergence-free on the corrected field, and the body force contributes to the
divergence that must be cancelled by pressure.

**`__init__` signature change:**
```python
def __init__(self, ..., body_force=None, ...):
    ...
    self._body_force_fn = body_force  # callable(u, v, t) -> (fu, fv) or None
```

**`step()` change** (3 lines inserted):
```python
# After diffusion / explicit step, before pressure:
if self._body_force_fn is not None:
    fu, fv = self._body_force_fn(self.u, self.v, self.time)
    u_star += dt * fu[1:-1, 1:-1]
    v_star += dt * fv[1:-1, 1:-1]
```

**YAML format:**
```yaml
body_force:
  u: "0.0"          # expression in x, y, u, v, t
  v: "0.0"
```
For Phase 1 we support only constant expressions. A later phase can add
full `(x, y, t)` expressions.

**CLI parsing** (`_parse_boundary_config` sibling):
```python
def _parse_body_force(raw):
    if raw is None:
        return None
    # Evaluate constant string → float
    fu_val = float(raw.get('u', '0.0'))
    fv_val = float(raw.get('v', '0.0'))
    # Return a lambda that ignores spatial/temporal dependence
    def bf(u, v, t):
        return (np.full_like(u, fu_val), np.full_like(v, fv_val))
    return bf
```

---

### 2. Initial Conditions

**What:** A user-supplied callable that sets `(u, v, p)` before the first step.

**Design options considered:**

| Option | Pros | Cons |
|--------|------|------|
| `callable(Nx, Ny, dx, dy) → (u, v, p)` | Simple, no Mesh dependency | User must know grid params |
| `callable(mesh) → (u, v, p)` | Clean API, mesh has everything |耦合 to Mesh class |
| `callable(x, y) → (u, v, p)` where x, y are 2D arrays | Most flexible, spatially varying | Overhead for uniform ICs |
| String presets: `"zero"`, `"taylor_green"` | Simple for common cases | Not extensible |

**Chosen approach:** Accept `callable(x, y) → (u, v, p)` where `x` and `y` are
the 2D staggered-grid coordinate arrays. This covers Taylor-Green (spatially
varying cosine/sine), Couette (uniform zeros), and Poiseuille (parabolic).
String presets can be added later without breaking anything.

```python
# In __init__:
self._initial_condition_fn = None  # callable(x, y) -> (u, v, p) or None

# In solve(), before the main loop:
if self._initial_condition_fn is not None:
    self.u[:], self.v[:], self.p[:] = self._initial_condition_fn(
        self.mesh.x_u, self.mesh.y_u,
        self.mesh.x_v, self.mesh.y_v,
        self.mesh.x_p, self.mesh.y_p,
    )
    self.bc.apply(self.u, self.v, Nx, Ny)
```

Wait — the mesh arrays have different shapes for u (Nx+1, Ny), v (Nx, Ny+1),
and p (Nx, Ny). We need a cleaner API.

**Revised API:** `callable(xc, yc) → (u, v, p)` where `xc, yc` are cell-center
coordinates (shape Nx, Ny). The user sets u-face and v-face values manually
using numpy broadcasting. This is simplest and covers all cases:

```python
def my_ic(xc, yc):
    u = np.zeros_like(xc)   # u-face values, shape (Nx+1, Ny) — set separately
    v = np.zeros_like(xc)   # v-face values
    p = np.zeros_like(xc)   # cell-center
    return u, v, p
```

Hmm, this doesn't work because xc/yc have shape (Nx, Ny) but u has shape
(Nx+1, Ny).

**Final API — simplest approach:**
```python
def __init__(self, ..., initial_condition=None, ...):
    self._initial_condition_fn = initial_condition
    # callable(Nx, Ny, dx, dy, Lx, Ly) -> (u, v, p)
```

This matches how the solver itself constructs arrays. The user has full control:

```python
def taylor_green_ic(Nx, Ny, dx, dy, Lx, Ly):
    x = (np.arange(Nx) + 0.5) * dx  # cell centers
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.zeros((Nx+1, Ny))   # u-faces
    v = np.zeros((Nx, Ny+1))   # v-faces
    p = np.zeros((Nx, Ny))     # cell centers
    # Set u on interior faces (approximate at cell centers)
    u[1:-1, :] = -np.cos(X) * np.sin(Y)
    return u, v, p
```

Actually this is getting complex. Let me reconsider.

**Cleanest approach:** The IC function receives the Mesh object and returns
u, v, p arrays of the correct shapes. The Mesh object has all the coordinate
data the user needs.

```python
self._initial_condition_fn = None  # callable(mesh) -> (u, v, p)
```

```python
def taylor_green_ic(mesh):
    X, Y = np.meshgrid(mesh.x_u, mesh.y_u, indexing='ij')  # wrong shapes...
```

Still tricky because u/v/p live on different grids.

**Simplest possible approach (chosen):** The IC function takes `(Nx, Ny)` and
returns `(u, v, p)` of exactly the right shapes. The user is responsible for
the staggered grid layout:

```python
def my_ic(Nx, Ny):
    u = np.zeros((Nx+1, Ny))   # u-faces: Nx+1 faces in x, Ny cells in y
    v = np.zeros((Nx, Ny+1))   # v-faces: Nx cells in x, Ny+1 faces in y
    p = np.zeros((Nx, Ny))     # cell centers
    return u, v, p
```

This is the most explicit and least error-prone. The user must match the
shapes, but the shapes are well-defined by `(Nx, Ny)`. For Taylor-Green:

```python
def tg_ic(Nx, Ny):
    x = (np.arange(Nx) + 0.5) * dx
    y = (np.arange(Ny) + 0.5) * dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    u = np.zeros((Nx+1, Ny))
    v = np.zeros((Nx, Ny+1))
    p = np.zeros((Nx, Ny))
    u[1:-1, :] = -np.cos(X) * np.sin(Y)
    # For u on faces, interpolate
    u[0, :] = u[1, :]   # periodic-ish
    u[-1, :] = u[-2, :]
    return u, v, p
```

Actually the user needs `dx, dy, Lx, Ly` too. Let me just pass the mesh.

**FINAL FINAL decision:**

```python
self._initial_condition_fn = None  # callable(mesh) -> (u, v, p)
```

The `Mesh` object already has `Nx, Ny, dx, dy, Lx, Ly, x_u, y_u, x_v, y_v,
x_p, y_p` — everything the user needs. This is clean, extensible, and
Pythonic. The user gets one object that provides all spatial information.

```python
def taylor_green_ic(mesh):
    X, Y = np.meshgrid(mesh.x_p, mesh.y_p, indexing='ij')
    u = np.zeros(mesh.shape_u)
    v = np.zeros(mesh.shape_v)
    p = np.zeros(mesh.shape_p)
    # u on interior faces ≈ -cos(x)sin(y) at cell centers
    u[1:-1, :] = -np.cos(X) * np.sin(Y)
    v[:, 1:-1] = np.sin(X) * np.cos(Y)
    return u, v, p
```

**Where in `solve()`:**
```python
def solve(self, ...):
    ...
    if self._initial_condition_fn is not None:
        self.u[:], self.v[:], self.p[:] = self._initial_condition_fn(self.mesh)
        self.bc.apply(self.u, self.v, self.Nx, self.Ny)
    ...
```

---

### 3. Time Tracking

**What:** Expose `self.time` so transient analytical solutions can be evaluated.

**Implementation:**
```python
# In __init__:
self.time = 0.0

# In step():
self.time += self.dt
```

That's it. 2 lines.

**YAML:** None needed. Time is a solver-level attribute.

---

### 4. Steady-State Convergence Check

**What:** Optional early stopping when the velocity field has converged.

**Design:**
```python
def solve(self, ..., convergence_tol=None, convergence_window=100):
    # Track max |u^{n+1} - u^n| over a sliding window
    # If the norm stays below convergence_tol for convergence_window steps,
    # the solution is considered converged and we stop early.
```

**Implementation in the solve loop:**
```python
 converged_count = 0
 u_old = self.u.copy()

 for i in range(steps):
     self.step()

     if convergence_tol is not None:
         delta = np.max(np.abs(self.u - u_old))
         if delta < convergence_tol:
             converged_count += 1
             if converged_count >= convergence_window:
                 if verbose:
                     print(f"\nConverged at step {i+1} (max|du|={delta:.2e} < {convergence_tol:.2e} for {convergence_window} steps)")
                 return True
         else:
             converged_count = 0
         u_old[:] = self.u

 return True
```

**Important:** Copy `u_old` from `self.u` **after** the step, not before,
to avoid the reference issue. `u_old[:] = self.u` does an in-place copy.

**YAML:**
```yaml
convergence:
  tol: 1e-6
  window: 100
```

**CLI:** Parse from YAML config, pass to `solve()`.

---

## API Summary

```python
# New __init__ parameters (all optional, backward compatible):
Solver(
    ...,
    body_force=None,          # callable(u, v, t) -> (fu, fv)
    initial_condition=None,   # callable(mesh) -> (u, v, p)
)

# New solve parameters:
solver.solve(
    ...,
    convergence_tol=None,     # float, e.g. 1e-6
    convergence_window=100,   # int, number of steps below tol
)
```

**New public attribute:**
```python
solver.time  # float, current physical time (seconds)
```

---

## Files to Modify

| File | Changes |
|------|---------|
| `src/cfd_solver/solver/solver.py` | `__init__`: accept `body_force`, `initial_condition`. `step()`: apply body force, increment `self.time`. `solve()`: convergence check, apply IC. |
| `src/cfd_solver/solver/validate.py` | Add `body_force`, `initial_condition`, `convergence` to schema. |
| `src/cfd_solver/cli/__init__.py` | Parse `body_force`, `initial_condition`, `convergence` from YAML. |
| `tests/test_core.py` | 12-15 new tests for all Phase 1 features. |

**Estimated effort:** ~150-200 lines of production code, ~200 lines of tests.

---

## Test Plan

### Body Force Tests
1. **bf_constant_zero:** Constant zero body force → same results as no force.
2. **bf_constant_uniform:** Constant force `(1.0, 0.0)` accelerates fluid uniformly.
3. **bf_pressure_balance:** Constant x-force in periodic channel → velocity ramps linearly until force balances viscous drag. Check `du/dt → 0`.
4. **bf_time_varying:** Sinusoidal force `(sin(t), 0)` → oscillating velocity. Check peak velocity matches.
5. **bf_callable:** Lambda body force → same as constant for same values.

### Initial Condition Tests
6. **ic_zero:** `initial_condition=lambda mesh: (zeros, zeros, zeros)` → same as default.
7. **ic_taylor_green:** IC function sets cosine/sine velocity → check initial u field matches analytical.
8. **ic_mesh_access:** Verify IC function receives correct mesh object with all attributes.

### Time Tracking Tests
9. **time_starts_zero:** `solver.time == 0.0` before `solve()`.
10. **time_increments:** After `solve(simulation_time=1.0)` with `dt=0.01`, `solver.time ≈ 1.0`.
11. **time_step_method:** After 100 calls to `solver.step()`, `solver.time == 100 * dt`.

### Convergence Tests
12. **convergence_stops_early:** Run lid-driven cavity with `convergence_tol=1e-4` → completes in fewer steps than fixed step count.
13. **convergence_no_tol:** Without `convergence_tol`, runs all requested steps.
14. **convergence_returns_true:** `solve()` returns `True` on convergence.
15. **convergence_window:** Small window → converges faster (more sensitive); large window → converges slower.

---

## Risk Assessment

| Risk | Mitigation |
|------|-----------|
| IC function receives wrong mesh shapes | Tests verify mesh attributes; shapes are `(Nx+1,Ny)`, `(Nx,Ny+1)`, `(Nx,Ny)` |
| Body force applied at wrong point in step() | Inserted between diffusion and pressure; tests verify balance |
| Convergence check copies u reference | Use `u_old[:] = self.u` (in-place copy) |
| Backward compatibility | All new params default to `None`; existing code unchanged |
| Performance of convergence check | `np.max(np.abs(...))` is O(N); negligible compared to step() cost |

---

## Implementation Order

1. **Time tracking** (2 lines, trivial) — do first
2. **Body force** (apply in step(), 5-10 lines) — independent of IC
3. **Initial condition** (apply in solve(), 5-10 lines) — independent of force
4. **Convergence check** (in solve loop, 15-20 lines) — depends on time tracking
5. **Tests** (15 tests, ~200 lines)
6. **CLI/YAML parsing** (20-30 lines)
7. **Validate schema** (10-15 lines)

Total estimated: ~150-200 lines production + ~200 lines tests.

---

## Example Usage After Phase 1

### Taylor-Green Vortex (periodic, decaying)
```python
from cfd_solver.solver import Solver

def tg_ic(mesh):
    X, Y = np.meshgrid(mesh.x_p, mesh.y_p, indexing='ij')
    u = np.zeros(mesh.shape_u)
    v = np.zeros(mesh.shape_v)
    p = np.zeros(mesh.shape_p)
    u[1:-1, :] = -np.cos(X) * np.sin(Y)
    v[:, 1:-1] = np.sin(X) * np.cos(Y)
    return u, v, p

s = Solver(grid_size=(64, 64), nu=0.01, dt=0.001,
           Lx=2*np.pi, Ly=2*np.pi,
           initial_condition=tg_ic)
s.solve(simulation_time=1.0)
# At t=1.0: u_analytical = -cos(x)sin(y) * exp(-2*0.01*1.0)
```

### Couette Flow (transient)
```python
from cfd_solver.solver import Solver, BoundaryConditions, NoSlipWall

bc = BoundaryConditions(
    top=NoSlipWall(u=1.0),
    bottom=NoSlipWall(u=0.0),
)
s = Solver(grid_size=(32, 64), nu=0.01, dt=0.001,
           Lx=1.0, Ly=1.0, boundary_config=bc)
s.solve(simulation_time=2.0)
# Compare to Fourier series solution at t=2.0
```

### Channel Flow with Body Force (periodic x)
```python
# With periodic BCs in x + constant body force, fully developed
# channel flow is driven by the force instead of inlet/outlet pressure.
# u_analytical = (f/(2*nu)) * y * (H - y)
def channel_force(u, v, t):
    return (np.full_like(u, 0.1), np.zeros_like(v))

s = Solver(grid_size=(32, 64), nu=0.01, dt=0.001,
           Lx=5.0, Ly=1.0, body_force=channel_force,
           boundary_config=bc)
```

---

## Open Questions

None — design is finalized.
