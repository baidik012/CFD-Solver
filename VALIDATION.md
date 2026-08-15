# Validation

Detailed validation results for each example, comparing numerical solutions
against analytical solutions or benchmark data.

For quick output images, see [EXAMPLES.md](EXAMPLES.md).

---

## Taylor-Green Vortex

### Analytical Solution

Exact decaying vortex solution for incompressible Navier-Stokes:

```
u(x, y, t) = -U₀ sin(kx·x) cos(ky·y) exp(-d·t)
v(x, y, t) =  U₀ cos(kx·x) sin(ky·y) exp(-d·t)
p(x, y, t) = -(ρ/4) U₀² [cos(2kx·x) + cos(2ky·y)] exp(-2d·t)
```

where `kx = 2π/Lx`, `ky = 2π/Ly`, `d = ν(kx² + ky²)`.

**Parameters:** U₀ = 1.0, ν = 0.01, Lx = Ly = 2π, dt = 0.001

### Error Metrics (64×64, t=2s)

| Component | L2 error | L∞ error |
|-----------|----------|----------|
| u | 3.79e-02 | 7.59e-02 |
| v | 3.79e-02 | 7.59e-02 |

**Kinetic energy:** Numerical KE = 1.95e-01, Analytical KE = 2.31e-01.

### Grid Convergence

| Grid | L2 error | Rate |
|------|----------|------|
| 16×16 | 1.17e-01 | — |
| 32×32 | 6.92e-02 | 0.76 |
| 64×64 | 3.79e-02 | 0.87 |
| 128×128 | 1.99e-02 | 0.93 |

Convergence rate ~0.9 (first-order from explicit diffusion with periodic BCs).

![Taylor-Green Convergence](images/taylor_green_convergence.png)

### Reproduction

```bash
python -m examples.taylor_green.validate
python -m examples.taylor_green.convergence
```

---

## Couette Flow

### Analytical Solution

Flow between two infinite parallel plates. With periodic x BCs and no imposed
pressure gradient, the exact pressure gradient is zero and the velocity is a
one-dimensional transient diffusion problem.

**Transient from rest (Fourier series):**
```
u(y, t) = U·y/H + Σ_{n=1}^{50} [2U/(nπ)]·(-1)^n·sin(nπy/H)·exp(-n²π²νt/H²)
```

**Steady state:** `u(y) = U·y/H` (linear profile)

**Parameters:** U = 1.0, H = 1.0, ν = 0.01, dt = 0.001

### Important interpretation of the bundled result

The default configuration uses `simulation_time = 10 s`, but the viscous
diffusion time is

```
H² / ν = 1² / 0.01 = 100 s.
```

Therefore the bundled `couette_result.png` is **not a steady-state image**.
At `t = 10 s` the flow is still developing from rest, so the velocity profile
is expected to be curved rather than linear. The pressure should be spatially
constant up to numerical error because no streamwise pressure gradient is
applied.

The old v0.3.1 image showed a strong pressure gradient and was not a physically
consistent representation of this periodic-x Couette case. The regenerated
v0.3.2 image correctly shows essentially constant pressure.

### Error Metrics

The Couette run and convergence study compare against the **transient** Fourier
solution at the solver's actual final time. They should not be compared against
the steady linear profile until the flow has been evolved for several viscous
diffusion times.

### Grid Convergence

The convergence script scales `dt` with `dy²` to maintain explicit diffusion
stability. Consequently the measured rate is a **combined spatial/temporal
refinement rate**, not a pure spatial order. The current rate near 2 is
consistent with second-order spatial discretization combined with a first-order
time integrator whose timestep is itself O(h²).

![Couette Convergence](images/couette_convergence.png)

### Reproduction

```bash
python -m examples.couette.run
python -m examples.couette.convergence
```
