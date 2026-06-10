# Code Review & Fixes for CFD-Solver

## Executive Summary
The solver implements Chorin's projection method correctly overall, but suffers from **numerical instability at moderate CFL numbers** due to:
1. **Overly aggressive CFL checks** that allow unsafe time steps
2. **Missing viscosity-based stability constraint** for explicit advection
3. **Suboptimal handling of the smooth lid profile** in boundary conditions
4. **Potential divide-by-zero in CFL calculation** when velocities are very small

This document details the issues, their impact, and fixes.

---

## Issues Found

### 1. **CFL Stability Check is Insufficient** 🔴 CRITICAL

**Location:** `src/cfd_solver/solver/solver.py`, lines 140–145

**Issue:**
```python
dt_max = min(dx, dy) / max(lid_speed, 1e-10) * 0.1
if dt > dt_max and not force:
    raise ValueError(...)
```

This assumes **lid_speed** is the characteristic velocity, but during the simulation velocities can grow arbitrarily large. The check only validates at initialization; it doesn't account for:
- Vortex formation (flow can reach speeds >> lid_speed)
- Recirculating flow that generates higher local velocities
- The fact that CFL = (|u|*dt/dx) + (|v|*dt/dy) depends on **actual flow velocity**, not just boundary conditions

**Impact:** 
Your error shows `CFL=1953...` (astronomically large), which happens because velocities grew during the simulation. The initial check didn't prevent this.

**Fix:**
Replace with a conservative multi-constraint approach that accounts for viscosity:

```python
# Conservative CFL limit: max speed likely won't exceed ~2-3x lid speed
# Also apply diffusive stability constraint
dx_min = min(dx, dy)
# Advection stability: CFL < 1 for upwind, even stricter for safety
dt_advection = 0.5 * dx_min / max(lid_speed, 1e-10)
# Diffusion stability (explicit): dt < dx²/(4*nu)
if nu > 1e-10:
    dt_diffusion = 0.25 * dx_min**2 / nu
else:
    dt_diffusion = float('inf')
# Take minimum
dt_max = min(dt_advection, dt_diffusion) * 0.5  # 50% safety margin
```

**For your case:** 32×32 grid, dx=1/32≈0.031, nu=0.01, lid=1.0:
- Old limit: ~0.003
- New limit: min(0.5×0.031/1, 0.25×0.031²/0.01) × 0.5 ≈ **0.0005** (10× stricter!)

---

### 2. **CFL Diagnostic Can Produce Garbage Values** 🟡 MODERATE

**Location:** `src/cfd_solver/solver/diagnostics.py`, lines 68–82

**Issue:**
```python
def cfl(u, v, dx, dy, dt):
    return np.max(np.abs(u[:, 1:-1])) * dt / dx + np.max(np.abs(v[1:-1, :])) * dt / dy
```

When a NaN/Inf appears in `u` or `v`, `np.max()` returns NaN, and then the arithmetic produces invalid output. In your error log, the CFL value is extremely large—this is likely NaN converted to a huge number during printing.

**Fix:**
```python
def cfl(u, v, dx, dy, dt):
    """Calculate the maximum Courant-Friedrichs-Lewy (CFL) number.
    
    Returns inf if the field contains NaN/Inf (indicating blowup).
    """
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return np.inf  # Signal blowup early
    u_max = np.max(np.abs(u[:, 1:-1]))
    v_max = np.max(np.abs(v[1:-1, :]))
    return u_max * dt / dx + v_max * dt / dy
```

---

### 3. **Smooth Lid Profile Index Error** 🟡 MODERATE

**Location:** `src/cfd_solver/solver/diffusion.py`, lines 236–239

**Issue:**
```python
if self.bc.smooth_lid:
    rhs_u[:, -1] += 2.0 * ry * self.bc._get_lid_profile(Nx)[1:-1]
```

The lid profile is indexed `[1:-1]`, which assumes it has Nx+1 elements and discards the corners. But then it's added to `rhs_u[:, -1]` which has Nx-1 rows. **The shapes don't match: (Nx+1) - 2 = Nx-1, but semantically the operation is slicing away the corner values.**

**More importantly:** If the lid is indeed supposed to vary in x, it should be correctly applied at each x-location. The [1:-1] slice removes the corners (which are supposed to be zero anyway), but then the interior profile has Nx-1 elements being added to Nx-1 slots—this is correct by accident, but **fragile and undocumented**.

**Fix:**
```python
if self.bc.smooth_lid:
    # Profile has Nx+1 elements; take only interior u-faces (1..Nx-1)
    lid_interior = self.bc._get_lid_profile(Nx)[1:-1]
    assert len(lid_interior) == (Nx - 1), f"Lid profile mismatch: {len(lid_interior)} vs {Nx-1}"
    rhs_u[:, -1] += 2.0 * ry * lid_interior
```

Or better yet, extract the boundary condition contribution to a helper method in `BoundaryConditions`:

```python
# In bc.py
def lid_bc_contribution(self, Nx: int) -> np.ndarray:
    """Return the interior RHS contribution for the Crank-Nicolson solver."""
    if self.smooth_lid:
        return self._get_lid_profile(Nx)[1:-1]
    return np.full(Nx - 1, self.top)

# In diffusion.py
rhs_u[:, -1] += 2.0 * ry * self.bc.lid_bc_contribution(Nx)
```

---

### 4. **Pressure Gradient Indexing Can Be Subtle** 🟡 MODERATE

**Location:** `src/cfd_solver/solver/solver.py`, lines 240–244

**Issue:**
```python
grad_p_x = (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / dx
grad_p_y = (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / dy

self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x
self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y
```

On a staggered grid:
- `p` is at cell **centers**: shape (Nx+2, Ny+2)
- `u` is at x-**faces**: shape (Nx+1, Ny+2), indexed as u[i, j] where i ∈ [0, Nx], j ∈ [0, Ny+1]
- `v` is at y-**faces**: shape (Nx+2, Ny+1), indexed as v[i, j] where i ∈ [0, Nx+1], j ∈ [0, Ny]

The indexing `self.p[2:-1, 1:-1]` yields shape (Nx, Ny), which is correct for the interior u-faces. But **the slicing is obscure and error-prone**. It's not immediately clear why `[2:-1, 1:-1]` gives the right cells.

**Why it works:** 
- `p[2:-1, ...]` = p[i] for i ∈ [2, Nx] (skipping ghost cells 0 and Nx+1)
- `p[1:-2, ...]` = p[i] for i ∈ [1, Nx-1] (left neighbor)
- So grad_p_x lives at p locations i ∈ [2, Nx], which corresponds to u-face locations i ∈ [1, Nx-1]

But this is **hard to verify and maintain**. It should be more explicit.

**Fix:**
```python
# Pressure gradient at u-faces (i=1..Nx-1, j=1..Ny)
# ∂p/∂x ≈ (p[i+1,j] - p[i,j]) / dx
# where i (for u) corresponds to i+1 (for p with ghosts)
grad_p_x = (self.p[3:-1, 2:-2] - self.p[2:-2, 2:-2]) / dx  # interior cells
grad_p_y = (self.p[2:-2, 3:-1] - self.p[2:-2, 2:-2]) / dy  # interior cells

# Apply to interior velocity faces
self.u[1:-1, 1:-1] = u_star[1:-1, 1:-1] - dt * grad_p_x
self.v[1:-1, 1:-1] = v_star[1:-1, 1:-1] - dt * grad_p_y
```

**Actually, the original code is correct, but here's a clearer version:**
```python
# Pressure gradient evaluated at staggered face locations
# u-faces at (i, j) correspond to pressure cells (i, j) and (i+1, j)
u_interior = slice(1, self.Nx)      # u indices
v_interior = slice(1, self.Ny)      # v indices
grad_p_x = (self.p[2:-1, 1:-1] - self.p[1:-2, 1:-1]) / self.dx
grad_p_y = (self.p[1:-1, 2:-1] - self.p[1:-1, 1:-2]) / self.dy

self.u[u_interior, 1:-1] = u_star[u_interior, 1:-1] - self.dt * grad_p_x
self.v[1:-1, v_interior] = v_star[1:-1, v_interior] - self.dt * grad_p_y
```

---

### 5. **Missing Intermediate Velocity Boundary Enforcement** 🟡 MODERATE

**Location:** `src/cfd_solver/solver/solver.py`, line 236

**Issue:**
```python
# 2. Pressure Step: Solve ∇²p = (∇·u*) / dt
self.p[:] = self._pressure.solve(u_star, v_star, dt)
```

The intermediate velocity `u_star, v_star` is passed to the pressure solver without explicit boundary condition enforcement. While `diffusion.py` does call `bc.apply()` after solving, there's **no explicit guarantee** that `u_star` respects the physical boundaries when passed to the pressure solver.

In a staggered grid, the pressure is computed at cell centers, so boundary velocities don't directly affect the Poisson RHS calculation. But **for code clarity**, you should document this or enforce BCs explicitly before the pressure solve:

```python
# 2. Pressure Step: Ensure u_star respects boundaries before solving Poisson
# (pressure depends on divergence of u_star at interior cells only)
# Note: u_star already has BCs applied from the diffusion step, so this is
# mainly for safety / clarity.

self.p[:] = self._pressure.solve(u_star, v_star, dt)
```

---

### 6. **No Maximum Time Step Adaptation During Solve** 🟡 MODERATE

**Location:** `src/cfd_solver/solver/solver.py`, line 249–277

**Issue:**
```python
def solve(self, steps, verbose=True):
    ...
    for i in range(steps):
        self.step()
        if is_blowup(self.u, self.v):
            # Print warning and exit, but too late!
            return
```

Once blowup is detected, the simulation has already diverged irreversibly. **The time step is fixed and can't be adjusted mid-run**. A more robust approach would:
1. Accept an **optional adaptive time stepping strategy**
2. Monitor CFL and reduce dt if it grows too large
3. Provide a semi-implicit fallback (e.g., auto-switch from explicit to Crank-Nicolson)

For now, this is acceptable for an educational solver, but users should be warned to set a conservative `dt` from the start.

---

## Recommended Fixes (Priority Order)

### Priority 1: Fix CFL Stability Check (CRITICAL)
**File:** `src/cfd_solver/solver/solver.py`

Replace lines 138–145:
```python
# OLD:
dx, dy = self.mesh.dx, self.mesh.dy
dt_max = min(dx, dy) / max(lid_speed, 1e-10) * 0.1
if dt > dt_max and not force:
    raise ValueError(...)

# NEW:
dx, dy = self.mesh.dx, self.mesh.dy
dx_min = min(dx, dy)

# Conservative CFL constraint: assume flow reaches ~3x lid speed during simulation
# CFL stability requires CFL < 1 for upwind; use 0.5 for margin
cfl_advection = 0.5 * dx_min / max(3.0 * lid_speed, 1e-10)

# Diffusion stability (relevant for explicit diffusion): dt < dx²/(4*nu)
if nu > 1e-10:
    cfl_diffusion = 0.25 * dx_min**2 / nu
else:
    cfl_diffusion = float('inf')

dt_max = min(cfl_advection, cfl_diffusion)

if dt > dt_max and not force:
    raise ValueError(
        f"dt={dt} exceeds the numerical stability limit ({dt_max:.4g}). "
        f"This limit accounts for advection (CFL < 0.5) and diffusion stability. "
        f"Suggested dt <= {dt_max:.4g}. Use force=True to override."
    )
```

### Priority 2: Fix CFL Diagnostic (HIGH)
**File:** `src/cfd_solver/solver/diagnostics.py`

Replace the `cfl()` function:
```python
def cfl(u, v, dx, dy, dt):
    """Calculate the maximum Courant-Friedrichs-Lewy (CFL) number.

    The CFL number is a measure of how much information travels across a
    grid cell in a single time step:
        CFL = max(|u|*dt/dx + |v|*dt/dy)
    For numerical stability in explicit schemes, CFL should be < 1.0.
    
    Returns inf if the field contains NaN/Inf (blowup detected).

    Returns
    -------
    float
        The maximum CFL number in the domain, or inf if blowup detected.
    """
    # Check for blowup first
    if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v))):
        return np.inf
    
    u_max = np.max(np.abs(u[:, 1:-1]))
    v_max = np.max(np.abs(v[1:-1, :]))
    return u_max * dt / dx + v_max * dt / dy
```

### Priority 3: Document and Fix Smooth Lid in CrankNicolson (MEDIUM)
**File:** `src/cfd_solver/solver/diffusion.py`

Add a helper method to `BoundaryConditions` in `bc.py`:
```python
def get_interior_lid_profile(self, Nx: int) -> np.ndarray:
    """Return the interior lid BC values for u-faces (excluding corners).
    
    Returns an array of length Nx-1 with the BC values at interior u-faces.
    """
    if self.smooth_lid:
        full_profile = self._get_lid_profile(Nx)
        return full_profile[1:-1]  # Exclude corner faces
    return np.full(Nx - 1, self.top)
```

Then update `diffusion.py` line 236–239:
```python
# y-direction (via ghost cells)
interior_lid = self.bc.get_interior_lid_profile(Nx)
rhs_u[:, -1] += 2.0 * ry * interior_lid
```

---

## Test Cases to Verify Fixes

After implementing fixes, test with your original parameters but **reduced time step**:

```yaml
# test_config.yaml
geometry:
  Lx: 1.0
  Ly: 1.0
  Nx: 32
  Ny: 32

nu: 0.01
dt: 0.0001          # ← START HERE (original was 0.001)
steps: 200

advection_scheme: upwind
diffusion_scheme: crank_nicolson

boundary:
  smooth_lid: true
  top:
    u: 1.0
    v: 0.0
```

**Expected behavior:**
- Step 200: CFL ≈ 0.5–0.8, divergence ≈ 1e-5 to 1e-6
- Velocity magnitude stays in range [0, ~1.5]
- No NaN/Inf at any step

Then gradually increase `dt` to find the actual stability boundary:
- Try `dt: 0.0002`, `0.0003`, `0.0005`, etc.
- Plot CFL number vs. step
- Verify smooth, stable convergence

---

## Summary Table

| Issue | Severity | Location | Fix |
|-------|----------|----------|-----|
| CFL check too lenient | 🔴 CRITICAL | solver.py:140 | Use 0.5×dx_min / (3×lid_speed) + diffusion constraint |
| CFL diagnostic returns NaN | 🟡 HIGH | diagnostics.py:82 | Check for NaN/Inf before computing |
| Smooth lid indexing unclear | 🟡 MEDIUM | diffusion.py:237 | Extract to helper method in bc.py |
| Pressure gradient indexing | 🟡 MEDIUM | solver.py:240 | Add slice variables for clarity (optional) |
| No BCs before Poisson (doc) | 🟠 LOW | solver.py:236 | Add comment; code is correct |
| No adaptive dt | 🟠 LOW | solver.py:249 | Document limitation; acceptable for v0.1 |

---

## Next Steps for Robustness

1. **Add energy conservation check**: Compare kinetic energy input (work by lid) vs. dissipation (nu * integral of strain rate²). Should be balanced in steady state.
2. **Implement CFL monitoring**: Track CFL history and warn if it approaches 1.0.
3. **Add restart/checkpoint on near-blowup**: Detect CFL > 0.8 and suggest checkpoint/restart with smaller dt.
4. **Validation against Ghia et al.** Run on 64×64 and 128×128 grids; compare velocity profiles at Re=400.

