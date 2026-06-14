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

Flow between two infinite parallel plates. With periodic x BCs, the profile
converges to the linear Couette solution.

**Transient from rest (Fourier series):**
```
u(y, t) = U·y/H + Σ_{n=1}^{50} [2U/(nπ)]·(-1)^n·sin(nπy/H)·exp(-n²π²νt/H²)
```

**Steady state:** `u(y) = U·y/H` (linear profile)

**Parameters:** U = 1.0, H = 1.0, ν = 0.01, dt = 0.001

### Error Metrics (32×64, t=5s)

| Metric | Value |
|--------|-------|
| L2 error vs transient | 2.90e-01 |
| L∞ error vs transient | 4.49e-01 |

Note: The validate.py runs with no-slip walls on all 4 sides (closed box), so the flow develops a circulation pattern rather than a pure linear Couette profile. The L2 error reflects this.

### Grid Convergence

| Grid | L2 error | Rate |
|------|----------|------|
| 16×32 | 9.48e-04 | — |
| 32×64 | 4.64e-04 | 1.03 |
| 64×128 | 2.30e-04 | 1.01 |
| 128×256 | 1.15e-04 | 1.01 |

Convergence rate ~1.0 (first-order from explicit diffusion). Note: the convergence study uses periodic BCs and scaled dt for stability.

![Couette Convergence](images/couette_convergence.png)

### Reproduction

```bash
python -m examples.couette.validate
python -m examples.couette.convergence
```

---

## Channel Flow (Poiseuille)

### Analytical Solution

Pressure-driven fully developed flow between two parallel plates.

**Steady-state profile:**
```
u(y) = (4·U_max/H²) · y · (H - y)
```

where H is the channel height and U_max is the centerline velocity.

**Parameters:** U_max = 1.0, H = 1.0, Lx = 10.0, ν = 0.01, dt = 0.0005

### Error Metrics (128×32, t=10s)

| Metric | Value |
|--------|-------|
| L2 error vs parabolic | 4.34e-04 |
| L∞ error vs parabolic | 8.85e-04 |
| Profile collapse (x=2 vs x=8) | 1.59e-04 |
| Centerline u | 0.9985 (exact: 0.9990) |

### Grid Convergence

| Grid | L2 error | Rate |
|------|----------|------|
| 32×16 | 1.71e-03 | — |
| 64×32 | 4.35e-04 | 1.98 |
| 128×32 | 4.34e-04 | 0.00 |
| 256×64 | 1.09e-04 | 2.00 |

Convergence rate ~2.0 (second-order from Crank-Nicolson diffusion).

![Channel Convergence](images/channel_convergence.png)

### Reproduction

```bash
python -m examples.channel_flow.validate
python -m examples.channel_flow.convergence
```

---

## Lid-Driven Cavity

### Benchmark Data

Validation against the classic benchmark data from:
> Ghia, U., Ghia, K.N., & Shin, C.T. (1982).
> High-Re solutions for incompressible flow using the Navier-Stokes
> equations and a multigrid method. *Journal of Computational Physics*, 48(3), 387-411.

Reference data at 17 stations for u-profile (along x=0.5) and v-profile (along y=0.5).

### Error Metrics (128×128, Re=100)

| Profile | L2 error | L∞ error |
|---------|----------|----------|
| u (x=0.5) | 6.57e-03 | 2.57e-02 |
| v (y=0.5) | 4.62e-03 | 7.76e-03 |

### Grid Convergence (Re=100)

| Grid | L2 error (u) | Rate |
|------|--------------|------|
| 32×32 | 2.92e-02 | — |
| 64×64 | 1.57e-02 | 0.89 |
| 128×128 | 6.57e-03 | 1.26 |

Convergence rate ~1.0, typical for cavity flow with corner singularities.

![Cavity Convergence](images/cavity_convergence.png)

### Ghia Comparison Plots

**Re = 100:**
![Ghia Re=100](images/ghia_re100.png)

**Re = 400:**
![Ghia Re=400](images/ghia_re400.png)

**Re = 1000:**
![Ghia Re=1000](images/ghia_re1000.png)

### Reproduction

```bash
python -m examples.cavity.validate
python -m examples.cavity.validate 400
python -m examples.cavity.validate 1000
python -m examples.cavity.convergence
python run_ghia_validation.py 100
```
