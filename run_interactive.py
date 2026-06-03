"""Interactive solver launcher — asks for parameters, runs simulation."""

import os
import sys
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import yaml


def ask(prompt, default):
    """Prompt user for input, return default if empty."""
    val = input(f"  {prompt} [{default}]: ").strip()
    return type(default)(val) if val else default


def main():
    print()
    print("========================================")
    print("  CFD Solver — Interactive Setup")
    print("========================================")
    print()
    print("Press Enter to accept defaults shown in [brackets].")
    print()

    Nx = ask("Grid cells in x", 32)
    Ny = ask("Grid cells in y", 32)
    nu = ask("Viscosity (nu)", 0.01)
    dt = ask("Time step (dt)", 0.001)
    steps = ask("Number of steps", 200)
    top_u = ask("Lid speed", 1.0)
    smooth_lid = input("  Use smooth lid profile? (y/n) [y]: ").strip().lower() != "n"

    print()
    print(f"  Grid: {Nx}x{Ny}  |  nu={nu}  |  dt={dt}  |  steps={steps}  |  lid={top_u}  |  smooth={smooth_lid}")
    print()

    cfg = {
        "geometry": {"Lx": 1.0, "Ly": 1.0, "Nx": Nx, "Ny": Ny},
        "nu": nu,
        "dt": dt,
        "steps": steps,
        "boundary": {"top": {"u": top_u, "v": 0.0}, "other": {"u": 0.0, "v": 0.0}},
    }

    config_path = os.path.join(os.path.dirname(__file__), "_run_config.yaml")
    with open(config_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)

    print("  Running solver...")
    print()

    from cfd_solver.solver.staggered_solver import StaggeredSolver
    from cfd_solver.solver.viz import save_velocity_contour

    s = StaggeredSolver(
        1.0, 1.0, Nx, Ny, nu, dt,
        u_bc={"top": top_u, "bottom": 0.0, "left": 0.0, "right": 0.0},
        smooth_lid=smooth_lid,
    )
    s.solve(steps, verbose=True)

    os.makedirs("output", exist_ok=True)
    out_path = "output/result.png"
    save_velocity_contour(s, out_path)

    print()
    print(f"  Result saved to {out_path}")

    # Try to open the image
    if sys.platform == "darwin":
        subprocess.run(["open", out_path])
    elif sys.platform.startswith("linux"):
        subprocess.run(["xdg-open", out_path])
    elif sys.platform == "win32":
        os.startfile(out_path)

    # Clean up temp config
    os.remove(config_path)


if __name__ == "__main__":
    main()
