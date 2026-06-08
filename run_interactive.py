"""Interactive solver launcher — asks for parameters, runs simulation."""

import os
import sys
import subprocess

import yaml


_FRIENDLY_ERRORS = {
    ModuleNotFoundError: (
        "Missing required package. Run:\n"
        "  pip install -r requirements.txt\n"
        "or:\n"
        "  pip install -e ."
    ),
    MemoryError: "Not enough memory. Try a smaller grid size.",
}


def _handle_error(exc):
    """Print a user-friendly message and exit."""
    print("\n" + "=" * 50)
    print("  ERROR")
    print("=" * 50)

    msg = _FRIENDLY_ERRORS.get(type(exc))
    if msg:
        print(f"  {msg}")
    else:
        print(f"  {type(exc).__name__}: {exc}")

    print("=" * 50 + "\n")
    raise SystemExit(1)


def ask(prompt, default):
    """Prompt user for input, return default if empty."""
    val = input(f"  {prompt} [{default}]: ").strip()
    return type(default)(val) if val else default


def main():
    # Check for updates before showing the banner so the notice appears first
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from cfd_solver.version_check import check_for_updates
        check_for_updates(repo_dir=os.path.dirname(__file__) or ".")
    except Exception:
        pass  # never let a version-check failure block the solver

    try:
        from cfd_solver import __version__
        version_str = __version__
    except Exception:
        version_str = "unknown"

    print()
    print("========================================")
    print("  CFD Solver — Interactive Setup")
    print("========================================")
    print(f"  Version: {version_str}")
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

    print("  Running solver...")
    print()

    try:
        from cfd_solver.solver import Solver
        from cfd_solver.solver.viz import save_contour
    except ModuleNotFoundError as exc:
        _handle_error(exc)

    try:
        s = Solver(
            grid_size=(Nx, Ny), nu=nu, dt=dt,
            lid_speed=top_u, smooth_lid=smooth_lid,
        )
        s.solve(steps, verbose=True)
    except Exception as exc:
        _handle_error(exc)

    os.makedirs("output", exist_ok=True)
    out_path = "output/result.png"
    save_contour(s.mesh, s.u, s.v, s.p, out_path)

    print()
    print(f"  Result saved to {out_path}")

    # Try to open the image
    opened = False
    if sys.platform == "darwin":
        subprocess.run(["open", out_path])
        opened = True
    elif sys.platform.startswith("linux"):
        for cmd in [["xdg-open", out_path],
                     ["explorer.exe", os.path.abspath(out_path)]]:
            try:
                subprocess.run(cmd, check=True)
                opened = True
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
    elif sys.platform == "win32":
        os.startfile(out_path)
        opened = True

    if not opened:
        print(f"  Open output/result.png to view the result.")


if __name__ == "__main__":
    main()
