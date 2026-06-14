# Phase 0: Flexible Boundary Condition System — Detailed Design

## Goal

Redesign the BC system so the solver can handle inlet/outlet, periodic, and free-slip walls
in addition to the existing no-slip walls. This is the foundation for ALL 2D examples.

## Constraints

1. **Backward compatible** — existing lid-driven cavity configs and tests must still work unchanged
2. **BC type is fixed at construction** — matrices/eigenvalues are built once; only BC values can change at runtime
3. **Minimize disruption** — touch as few files as possible; prefer additive changes
4. **Branch: `feature/2d-examples`** — all changes on this branch only

---

## 1. New `BoundaryConditions` Class Design

### 1.1 Wall Type Classes

```python
# bc.py — new wall type hierarchy

class WallType:
    """Base class for wall boundary condition types."""
    pass

class NoSlipWall(WallType):
    """No-slip wall with specified tangential velocity."""
    def __init__(self, u=0.0, v=0.0):
        self.u = u  # tangential u-velocity (for top/bottom walls)
        self.v = v  # tangential v-velocity (for left/right walls)

class FreeSlipWall(WallType):
    """Free-slip / symmetry wall: zero normal gradient for tangential velocity."""
    def __init__(self, u=0.0, v=0.0):
        self.u = u
        self.v = v

class InletWall(WallType):
    """Inlet: specified velocity profile at the boundary."""
    def __init__(self, profile="uniform", U_max=1.0):
        self.profile = profile  # "uniform" or "parabolic"
        self.U_max = U_max

class OutletWall(WallType):
    """Outlet: zero-gradient or convective outflow."""
    def __init__(self, method="zero_gradient"):
        self.method = method  # "zero_gradient" or "convective"

class PeriodicWall(WallType):
    """Periodic: wraps to the opposite wall."""
    pass
```

### 1.2 `BoundaryConditions` Constructor

```python
class BoundaryConditions:
    def __init__(self, top=None, bottom=None, left=None, right=None, smooth_lid=False):
        # Backward-compatible: if scalar is passed, treat as NoSlipWall
        self.top = self._normalize_wall(top, NoSlipWall, default_u=1.0)
        self.bottom = self._normalize_wall(bottom, NoSlipWall, default_u=0.0)
        self.left = self._normalize_wall(left, NoSlipWall, default_v=0.0)
        self.right = self._normalize_wall(right, NoSlipWall, default_v=0.0)
        self.smooth_lid = smooth_lid
        # ... cache attributes ...

    def _normalize_wall(self, wall, default_type, **defaults):
        """Convert scalar/None to WallType for backward compatibility."""
        if wall is None:
            return default_type(**defaults)
        if isinstance(wall, (int, float)):
            # Legacy: scalar value means NoSlipWall with that tangential speed
            return default_type(u=wall)
        if isinstance(wall, WallType):
            return wall
        raise TypeError(f"Expected WallType or scalar, got {type(wall)}")
```

### 1.3 Backward Compatibility Properties

```python
    # Legacy attribute access — reads/writes through to the WallType objects
    @property
    def top_u(self):
        if isinstance(self.top, NoSlipWall):
            return self.top.u
        return 0.0

    @top_u.setter
    def top_u(self, value):
        if isinstance(self.top, NoSlipWall):
            self.top.u = value

    # Same for bottom, left, right...
```

This ensures existing code like `self.bc.top` (used in diffusion.py RHS) still works,
but now `self.bc.top` returns the WallType object, not a float.

**PROBLEM:** Currently `bc.top` is a float (e.g., 1.0). If we change it to a WallType object,
all existing code like `rhs_u[:, -1] += 2.0 * ry * self.bc.top` breaks.

**SOLUTION:** Keep `bc.top` as a float for backward compatibility. Add new attributes for the
wall type objects. The old API continues to work unchanged.

### 1.4 Revised Approach: Keep Old API, Add New API

```python
class BoundaryConditions:
    def __init__(self, top=1.0, bottom=0.0, left=0.0, right=0.0, smooth_lid=False):
        # OLD API — preserved exactly as-is
        self.top = top          # float: tangential u at top wall
        self.bottom = bottom    # float: tangential u at bottom wall
        self.left = left        # float: tangential v at left wall
        self.right = right      # float: tangential v at right wall
        self.smooth_lid = smooth_lid

        # NEW API — per-wall type configuration
        self.walls = {
            'top':    NoSlipWall(u=top),
            'bottom': NoSlipWall(u=bottom),
            'left':   NoSlipWall(v=left),
            'right':  NoSlipWall(v=right),
        }

        # Legacy cache
        self._lid_profile = None
        self._lid_profile_key = None
```

Now `self.bc.top` is still a float (backward compatible), and `self.bc.walls['top']`
gives the WallType object. The `apply()` method uses `self.walls` internally but the
old attributes remain accessible.

---

## 2. Changes to `bc.apply()`

### 2.1 Current apply() Flow

```
1. Set normal velocities to 0 (u at left/right, v at top/bottom)
2. Set tangential u at top/bottom via ghost cells
3. Set tangential v at left/right via ghost cells
```

### 2.2 New apply() Flow

```python
def apply(self, u, v, Nx, Ny):
    for wall_name, wall in self.walls.items():
        if isinstance(wall, NoSlipWall):
            self._apply_no_slip(u, v, Nx, Ny, wall_name, wall)
        elif isinstance(wall, FreeSlipWall):
            self._apply_free_slip(u, v, Nx, Ny, wall_name, wall)
        elif isinstance(wall, InletWall):
            self._apply_inlet(u, v, Nx, Ny, wall_name, wall)
        elif isinstance(wall, OutletWall):
            self._apply_outlet(u, v, Nx, Ny, wall_name, wall)
        elif isinstance(wall, PeriodicWall):
            self._apply_periodic(u, v, Nx, Ny, wall_name)
```

### 2.3 Per-Wall Application Logic

**NoSlipWall (existing behavior, unchanged):**
- top/bottom: `u[:, ghost] = 2*U_wall - u[:, interior]`; `v[interior, face] = 0`
- left/right: `v[ghost, :] = 2*V_wall - v[interior, :]`; `u[face, interior] = 0`

**FreeSlipWall:**
- top/bottom: `u[:, ghost] = u[:, interior]` (zero gradient); `v[interior, face] = 0`
- left/right: `v[ghost, :] = v[interior, :]` (zero gradient); `u[face, interior] = 0`

**InletWall:**
- top/bottom inlet: Set `u[:, face]` to inlet profile; `v[interior, face] = 0`
- left/right inlet: Set `v[face, :]` to inlet profile; `u[face, interior] = 0`
- Profile functions:
  - uniform: `u(y) = U_max`
  - parabolic: `u(y) = 4*U_max*y*(H-y)/H²` (channel flow)

**OutletWall (zero-gradient):**
- top/bottom: `u[:, ghost] = u[:, interior]` (same as free_slip); `v[interior, face] = 0`
- left/right: `v[ghost, :] = v[interior, :]`; `u[face, interior] = 0`
- Note: For the NORMAL velocity at outlet, we use extrapolation from interior

**PeriodicWall:**
- top/bottom pair: `u[:, 0] = u[:, -2]` and `u[:, -1] = u[:, 1]` (ghost = opposite interior)
- left/right pair: `v[0, :] = v[-2, :]` and `v[-1, :] = v[1, :]`
- Normal velocity is NOT set to 0 (fluid can flow through)

### 2.4 Preserving Old apply() for Backward Compatibility

The old `apply()` method is preserved as a fallback. When `self.walls` is not set
(i.e., old-style construction), the old logic runs unchanged.

---

## 3. Changes to `diffusion.py`

### 3.1 The Problem

CrankNicolson and FFTCrankNicolson build matrices/eigenvalues in `__init__()` that
encode the BC TYPE. The matrices assume:
- u x-direction: Dirichlet (u=0 at walls)
- u y-direction: ghost-cell (tangential u at top/bottom)
- v x-direction: ghost-cell (tangential v at left/right)
- v y-direction: Dirichlet (v=0 at walls)

If a wall changes type (e.g., top wall goes from NoSlipWall to InletWall),
the matrix structure must change.

### 3.2 Strategy: Explicit Diffusion Fallback for Non-Standard BCs

Since BC type is fixed at construction, and the matrix structure depends on BC type:

**Option chosen: Use explicit diffusion for cases with non-standard BCs.**

Rationale:
- Explicit diffusion is already implemented and works
- It's stable for the moderate Re cases we're targeting (Re < 1000)
- Avoids the complexity of rebuilding CN matrices
- The CN matrices are only needed for high-Re cases where dt must be large

In `Solver.__init__()`:
```python
has_nonstandard_bcs = any(
    not isinstance(wall, NoSlipWall)
    for wall in bc.walls.values()
)

if diffusion_scheme == "crank_nicolson" and has_nonstandard_bcs:
    print("  [info] Non-standard BCs detected; using explicit diffusion "
          "(Crank-Nicolson requires uniform wall types).", file=sys.stderr)
    self._diffusion = None  # forces explicit path in step()
elif diffusion_scheme == "crank_nicolson":
    self._diffusion = create_diffusion_solver(self.mesh, nu, dt, self.bc)
```

### 3.3 Future Enhancement: Rebuildable CN Matrices

For a future phase, we could add the ability to rebuild CN matrices when BC types change.
This would require:
- Storing the BC type configuration at matrix build time
- A method to check if BC types have changed
- A rebuild + refactorize method

This is NOT part of Phase 0.

---

## 4. Changes to `pressure.py`

### 4.1 The Problem

Pressure Poisson solver uses pure Neumann BCs on all walls. For channel flow with
outlet, we need:
- Neumann on walls (top, bottom)
- Neumann on inlet (left)
- Fixed pressure (Dirichlet) on outlet (right)

### 4.2 Strategy: Pass BC Info to Pressure Solver

```python
class PressureSolver:
    def __init__(self, mesh, bc=None):
        self.bc = bc
        self.A = self._build_matrix()
        self._solve = splu(self.A).solve

    def _build_matrix(self):
        Nx, Ny = self.Nx, self.Ny
        inv_dx2 = 1.0 / (self.dx**2)
        inv_dy2 = 1.0 / (self.dy**2)

        # x-direction Laplacian
        diag_x = np.full(Nx, 2.0 * inv_dx2)
        off_x = np.full(Nx - 1, -inv_dx2)

        # Apply BC-specific modifications
        if self.bc is not None:
            left_wall = self.bc.walls.get('left')
            right_wall = self.bc.walls.get('right')

            if isinstance(left_wall, OutletWall):
                # Dirichlet at left outlet: full stencil, pin pressure
                pass  # standard diagonal
            else:
                # Neumann at left: halved stencil
                diag_x[0] = inv_dx2

            if isinstance(right_wall, OutletWall):
                # Dirichlet at right outlet: full stencil, pin pressure
                pass
            else:
                # Neumann at right: halved stencil
                diag_x[-1] = inv_dx2
        else:
            # Default: Neumann everywhere (backward compatible)
            diag_x[0] = inv_dx2
            diag_x[-1] = inv_dx2

        # ... rest of matrix assembly ...
```

### 4.3 Pressure Pinning for Outlet

When an outlet has Dirichlet pressure (p=0):
- Pin the outlet cell(s) to p=0 instead of pinning cell (0,0)
- Remove zero-mean normalization (the outlet defines the reference)

```python
def solve(self, u_star, v_star, dt):
    # ... compute divergence and RHS ...

    # Pin pressure at outlet instead of cell (0,0)
    if self.bc is not None:
        for wall_name, wall in self.bc.walls.items():
            if isinstance(wall, OutletWall):
                # Pin pressure at the outlet face
                if wall_name == 'right':
                    rhs[Nx-1, :] = 0.0  # last column
                    A_pin = ...  # set Dirichlet row
                elif wall_name == 'left':
                    rhs[0, :] = 0.0
    else:
        # Default: pin at cell (0,0)
        rhs[0] = 0.0
```

### 4.4 FFT Pressure Solver

For the FFT solver, changing Neumann to Dirichlet at one boundary changes the
eigenvalue structure. The simplest approach: fall back to sparse direct solver
(SpressureSolver) when non-Neumann pressure BCs are needed.

```python
def create_pressure_solver(mesh, bc=None):
    # Check if any wall has non-Neumann pressure BC
    needs_direct = False
    if bc is not None:
        for wall in bc.walls.values():
            if isinstance(wall, OutletWall):
                needs_direct = True
                break

    if needs_direct or mesh.Nx <= FFT_THRESHOLD and mesh.Ny <= FFT_THRESHOLD:
        return PressureSolver(mesh, bc=bc)
    return FFTPressureSolver(mesh)
```

---

## 5. Changes to `solver.py`

### 5.1 Constructor Changes

```python
class Solver:
    def __init__(self, grid_size, nu, dt, lid_speed=1.0, smooth_lid=True,
                 advection_scheme="upwind", diffusion_scheme="crank_nicolson",
                 Lx=1.0, Ly=1.0, force=False,
                 # NEW parameters:
                 boundary_config=None):
```

When `boundary_config` is provided (dict with per-wall specs), it overrides
the legacy `lid_speed` parameter.

### 5.2 Step Function Changes

```python
def step(self):
    Nx, Ny = self.Nx, self.Ny
    dx, dy = self.mesh.dx, self.mesh.dy
    dt = self.dt

    # Ensure BCs are up to date
    self.bc.apply(self.u, self.v, Nx, Ny)

    # 1. Prediction Step
    adv_u, adv_v = self._advection_fn(self.u, self.v, dx, dy)

    if self._diffusion is not None:
        u_star, v_star = self._diffusion.solve(self.u, self.v, adv_u, adv_v)
    else:
        from .diffusion import explicit
        u_star, v_star = explicit(
            self.u, self.v, adv_u, adv_v, dx, dy, dt,
            self.nu, self.bc, Nx, Ny,
        )

    # 2. Pressure Step
    self.p[:] = self._pressure.solve(u_star, v_star, dt)

    # 3. Correction Step
    grad_p_x = (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / dx
    grad_p_y = (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / dy
    self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x
    self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y

    # 4. Re-enforce BCs (handles inlet override of normal velocity)
    self.bc.apply(self.u, self.v, Nx, Ny)
```

The step function itself doesn't change much. The key is that `bc.apply()`
now handles the different wall types internally.

### 5.3 Checkpoint Changes

```python
def checkpoint(self, path):
    np.savez_compressed(
        path,
        u=self.u, v=self.v, p=self.p,
        Nx=self.Nx, Ny=self.Ny,
        Lx=self.Lx, Ly=self.Ly,
        dt=self.dt, nu=self.nu,
        lid_speed=self.bc.top,        # legacy
        smooth_lid=self.bc.smooth_lid,
        advection_scheme=self.advection_scheme,
        diffusion_scheme=self.diffusion_scheme,
        # NEW: save wall types for full reconstruction
        wall_types={name: type(w).__name__ for name, w in self.bc.walls.items()},
        wall_params={name: vars(w) for name, w in self.bc.walls.items()},
    )
```

---

## 6. Changes to `validate.py`

### 6.1 New BC Schema

```python
"boundary": {
    "type": dict,
    "fields": {
        "smooth_lid": {"type": bool},
        # Legacy fields (still supported)
        "top": {
            "type": dict,
            "fields": {
                "u": {"type": (int, float)},
                "v": {"type": (int, float)},
            },
        },
        "other": {
            "type": dict,
            "fields": {
                "u": {"type": (int, float)},
                "v": {"type": (int, float)},
            },
        },
        # NEW: per-wall type specification
        "left": {
            "type": dict,
            "fields": {
                "type": {"type": str, "values": ["wall", "inlet", "outlet", "periodic", "free_slip"]},
                "u": {"type": (int, float)},
                "v": {"type": (int, float)},
                "profile": {"type": str, "values": ["uniform", "parabolic"]},
                "U_max": {"type": (int, float)},
                "method": {"type": str, "values": ["zero_gradient", "convective"]},
            },
        },
        "right": { /* same as left */ },
        "top_wall": { /* same, for clarity */ },
        "bottom": { /* same */ },
    },
},
```

---

## 7. Changes to `cli/__init__.py`

### 7.1 Parse New BC Config

```python
def _parse_boundary_config(bc_cfg):
    """Parse boundary config into a dict suitable for BoundaryConditions."""
    result = {}

    # Check for new per-wall specification
    for wall_name in ('left', 'right', 'top', 'bottom'):
        wall_cfg = bc_cfg.get(wall_name)
        if wall_cfg is None:
            continue
        if not isinstance(wall_cfg, dict):
            continue

        wall_type = wall_cfg.get('type', 'wall')
        if wall_type == 'wall':
            result[wall_name] = NoSlipWall(
                u=wall_cfg.get('u', 0.0),
                v=wall_cfg.get('v', 0.0),
            )
        elif wall_type == 'inlet':
            result[wall_name] = InletWall(
                profile=wall_cfg.get('profile', 'uniform'),
                U_max=wall_cfg.get('U_max', 1.0),
            )
        elif wall_type == 'outlet':
            result[wall_name] = OutletWall(
                method=wall_cfg.get('method', 'zero_gradient'),
            )
        elif wall_type == 'free_slip':
            result[wall_name] = FreeSlipWall(
                u=wall_cfg.get('u', 0.0),
                v=wall_cfg.get('v', 0.0),
            )
        elif wall_type == 'periodic':
            result[wall_name] = PeriodicWall()

    # Fall back to legacy parsing if no new-style walls found
    if not result:
        top = bc_cfg.get('top', {})
        top_u = top.get('u', 1.0)
        result['top'] = NoSlipWall(u=top_u)

    return result
```

### 7.2 Construct Solver with New BCs

```python
wall_specs = _parse_boundary_config(bc_cfg)
bc = BoundaryConditions(
    top=wall_specs.get('top', NoSlipWall(u=1.0)),
    bottom=wall_specs.get('bottom', NoSlipWall(u=0.0)),
    left=wall_specs.get('left', NoSlipWall(v=0.0)),
    right=wall_specs.get('right', NoSlipWall(v=0.0)),
    smooth_lid=bc_cfg.get('smooth_lid', True),
)

solver = Solver(
    grid_size=(Nx, Ny), nu=nu, dt=dt,
    Lx=Lx, Ly=Ly,
    advection_scheme=advection_scheme,
    diffusion_scheme=diffusion_scheme,
    boundary_config=bc,  # pass the constructed BC object
)
```

---

## 8. Changes to `mesh.py`

### 8.1 No Changes in Phase 0

Mesh shapes remain the same. Periodic BCs (which require extra ghost cells)
are deferred to a later phase.

---

## 9. New Example Configs

### 9.1 Channel Flow (Poiseuille) — Basic Test Case

```yaml
# examples/channel_flow.yaml
geometry:
  Lx: 5.0
  Ly: 1.0
  Nx: 128
  Ny: 32

nu: 0.01
dt: 0.001
simulation_time: 5.0

boundary:
  left:
    type: inlet
    profile: parabolic
    U_max: 1.0
  right:
    type: outlet
    method: zero_gradient
  top:
    type: wall
    u: 0.0
    v: 0.0
  bottom:
    type: wall
    u: 0.0
    v: 0.0
```

---

## 10. Test Plan for Phase 0

### 10.1 Unit Tests for New Wall Types

```python
def test_walls_types():
    """WallType objects are created correctly."""
    w = NoSlipWall(u=1.0, v=0.5)
    assert w.u == 1.0
    assert w.v == 0.5

    w = InletWall(profile="parabolic", U_max=2.0)
    assert w.profile == "parabolic"
    assert w.U_max == 2.0

    w = OutletWall(method="zero_gradient")
    assert w.method == "zero_gradient"
```

### 10.2 Backward Compatibility Tests

```python
def test_bc_backward_compat():
    """Old-style BoundaryConditions constructor still works."""
    bc = BoundaryConditions(top=1.0, smooth_lid=True)
    assert bc.top == 1.0
    assert bc.smooth_lid is True
    assert isinstance(bc.walls['top'], NoSlipWall)
    assert bc.walls['top'].u == 1.0

def test_bc_apply_unchanged():
    """Old-style BC apply produces same results as before."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(top=1.0)
    bc.apply(u, v, m.Nx, m.Ny)
    assert np.allclose(u[:, -1], 2.0)
    assert np.allclose(u[:, 0], 0.0)
```

### 10.3 New BC Type Tests

```python
def test_inlet_bc_parabolic():
    """Inlet wall sets parabolic profile."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=1.0),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
        right=NoSlipWall(v=0.0),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # At left wall, u should be parabolic in y
    # (implementation depends on exact profile specification)

def test_outlet_bc_zero_gradient():
    """Outlet wall sets zero-gradient for ghost cells."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    # Set some interior values
    u[3, 1:-1] = 1.0
    v[1:-1, 3] = 0.5
    bc = BoundaryConditions(
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
        left=NoSlipWall(v=0.0),
    )
    bc.apply(u, v, m.Nx, m.Ny)
    # Outlet ghost should equal interior
    assert np.allclose(v[-1, :], v[-2, :])

def test_free_slip_bc():
    """Free-slip wall sets zero gradient for tangential velocity."""
    m = Mesh(1.0, 1.0, 8, 6)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc = BoundaryConditions(
        top=FreeSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
        left=NoSlipWall(v=0.0),
        right=NoSlipWall(v=0.0),
    )
    # Set some interior u near top
    u[4, -2] = 0.5
    bc.apply(u, v, m.Nx, m.Ny)
    # Free-slip: ghost = interior
    assert np.allclose(u[4, -1], u[4, -2])
```

### 10.4 Integration Tests

```python
def test_channel_flow_runs():
    """Full solver with inlet/outlet BCs completes without blowup."""
    bc = BoundaryConditions(
        left=InletWall(profile="parabolic", U_max=1.0),
        right=OutletWall(method="zero_gradient"),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    s = Solver(grid_size=(32, 16), nu=0.01, dt=0.001,
               Lx=5.0, Ly=1.0, boundary_config=bc)
    s.solve(simulation_time=0.1, verbose=False)
    assert np.isfinite(s.u).all()
    assert np.isfinite(s.v).all()
    assert s.max_divergence() < 1e-4

def test_explicit_diffusion_with_inlet_outlet():
    """Explicit diffusion works with non-standard BCs."""
    bc = BoundaryConditions(
        left=InletWall(profile="uniform", U_max=1.0),
        right=OutletWall(),
        top=NoSlipWall(u=0.0),
        bottom=NoSlipWall(u=0.0),
    )
    m = Mesh(5.0, 1.0, 32, 16)
    u = np.zeros(m.shape_u)
    v = np.zeros(m.shape_v)
    bc.apply(u, v, m.Nx, m.Ny)
    from cfd_solver.solver.diffusion import explicit
    adv_u = np.zeros_like(u)
    adv_v = np.zeros_like(v)
    u_s, v_s = explicit(u, v, adv_u, adv_v, m.dx, m.dy, 0.001, 0.01, bc, m.Nx, m.Ny)
    assert np.isfinite(u_s).all()
```

### 10.5 Existing Tests Must Pass

All 57 existing tests in `test_core.py` must pass without modification.

---

## 11. File Change Summary

| File | Change Type | Lines Changed | Description |
|------|-------------|---------------|-------------|
| `bc.py` | Major rewrite | ~200 new | Wall type classes, new apply() logic |
| `solver.py` | Minor edit | ~30 new | boundary_config param, explicit diffusion fallback |
| `pressure.py` | Moderate edit | ~60 new | Accept bc param, outlet pressure pinning |
| `validate.py` | Moderate edit | ~40 new | New BC schema fields |
| `cli/__init__.py` | Moderate edit | ~50 new | Parse new BC config format |
| `diffusion.py` | No change | 0 | Uses bc.apply() which handles new types |
| `advection.py` | No change | 0 | Relies on ghost cells set by bc.apply() |
| `mesh.py` | No change | 0 | Shapes unchanged in Phase 0 |
| `tests/test_core.py` | Additions | ~150 new | New tests for new BC types |

**Total: ~530 lines of changes**

---

## 12. Implementation Order

1. **Create wall type classes** in `bc.py` (no existing code changes)
2. **Add `walls` dict** to `BoundaryConditions.__init__()` (backward compatible)
3. **Rewrite `apply()`** to dispatch on wall types, with fallback to old logic
4. **Add bc param** to `pressure.py` `PressureSolver` and `create_pressure_solver()`
5. **Add explicit diffusion fallback** in `solver.py` for non-standard BCs
6. **Update `validate.py`** with new BC schema
7. **Update `cli/__init__.py`** to parse new BC config
8. **Write all new tests**
9. **Run existing test suite** — must pass unchanged
10. **Create example configs** and validation scripts

---

## 13. Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| Breaking existing tests | Run test suite after each step; keep old apply() as fallback |
| Pressure solver singular system with outlet | Pin outlet cells to p=0; remove zero-mean normalization |
| Inlet profile interpolation on staggered grid | Carefully index profile array to match face positions |
| Explicit diffusion stability at high Re | Document limitation; CN matrices still work for all-NoSlip cases |
| CLI backward compat with old YAML configs | New schema accepts both old and new formats |
