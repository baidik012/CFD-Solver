# Quickstart Guide

For club members getting started with the solver.

---

## What is this?

A tool that simulates fluid flowing through a space. You describe the box, set how fast the fluid moves at the walls, and the solver figures out what happens inside.

**Common use case:** The "lid-driven cavity" — a square box with a moving lid. The fluid gets dragged along by the lid and swirls around. It's the "Hello World" of fluid simulation.

---

## Run Your First Simulation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the example
python examples/staggered_cavity.py
```

You'll see output like:
```
Lid-driven cavity: 128x128 grid, 1000 steps

Step    0: |∇·u|∞ = 5.12e-03, CFL = 0.012
Step  100: |∇·u|∞ = 4.21e-04, CFL = 0.089
...
Step  900: |∇·u|∞ = 1.33e-06, CFL = 0.152
```

After it finishes, check `output/staggered_result.png` for the visualization.

---

## What Just Happened

The solver ran 1000 time steps. Each step advances the simulation by `dt = 0.001` seconds, so you're simulating 1 second of fluid flow.

**Divergence (`|∇·u|∞`)** — Should get smaller as the simulation runs. It's a measure of how "wrong" the velocity field is. By step 900 it's near zero, meaning mass conservation is satisfied.

**CFL** — A stability check. As long as it's less than 1, the simulation is stable.

---

## Parameters Explained

When you look at `examples/staggered_cavity.py`, you'll see:

```python
Nx, Ny = 128, 128  # Grid resolution
nu = 0.01          # Viscosity
dt = 0.001         # Time step
steps = 1000       # Number of steps
```

**What to change:**

| Parameter | Effect | Too small | Too large |
|-----------|--------|-----------|-----------|
| `Nx, Ny` | Detail in results | Blurry | Slow |
| `nu` | "Thickness" of fluid | Unstable | Over-damped |
| `dt` | Speed of simulation | Slow | Unstable |
| `steps` | How long to simulate | Not enough time | Wasted time |

**Typical starting values for lid-driven cavity:**
```
Nx, Ny: 64-256 (128 is good for learning)
nu: 0.01 (water-like)
dt: 0.001
steps: 500-2000
```

---

## Reading the Output

After running, check `output/staggered_result.png`. It shows:

**Left plot (Pressure):** High and low pressure zones in the fluid. Red = high, blue = low. The pressure field drives the flow correction.

**Right plot (Velocity):** Color shows speed, arrows show direction. You'll see:
- Fast fluid near the moving lid (top)
- Slow fluid in corners (stagnation zones)
- A main vortex in the center

---

## Is My Answer Correct?

The lid-driven cavity is a classic problem with published reference data (Ghia et al., 1982). If you're getting wrong answers:

1. **Did divergence drop to near zero?** If not, something's broken.
2. **Is CFL < 1?** If not, reduce `dt`.
3. **Are velocities reasonable?** Lid speed is 1.0, so max velocity should be around 0.5-0.8 (the fluid doesn't reach lid speed everywhere).

---

## What's Next?

1. **Change the lid speed** — Edit `u_bc={"top": 1.0}` to `"top": 2.0`. Watch the flow speed up.

2. **Change viscosity** — Edit `nu=0.001` for "thinner" fluid. You'll need to reduce `dt` to stay stable.

3. **Increase resolution** — Change `Nx, Ny = 256, 256`. More detail, but slower.

4. **Run for longer** — Increase `steps`. The flow starts chaotic but settles into a steady state after enough time.

---

## Common Errors

**`CFL > 1, simulation diverged`**
```
dt too large. Reduce it to 0.0005 or smaller.
```

**`Max divergence stays large`**
```
Something's wrong with the pressure solver or boundary conditions.
Check that the grid dimensions are correct.
```

**`ImportError: No module named 'cfd_solver'`**
```
Make sure you're running from the repo root directory.
```

---

Questions? Open an issue or ask in the club channel.