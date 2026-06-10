"""
Interactive solver launcher — asks for parameters, runs simulation.

This script provides a user-friendly way to configure and run the CFD solver
interactively through the terminal. It prompts for physical and numerical
parameters, executes the simulation, and displays the result.
"""

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
    """
    Print a user-friendly message and exit.

    Parameters
    ----------
    exc : Exception
        The exception that was raised.
    """
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
    """
    Prompt user for input, return default if empty.

    Parameters
    ----------
    prompt : str
        The question to display to the user.
    default : any
        The default value if the user provides no input.

    Returns
    -------
    any
        The user input cast to the type of the default value, or the default value.
    """
    val = input(f"  {prompt} [{default}]: ").strip()
    return type(default)(val) if val else default


def main():
    """
    Main entry point for the interactive CFD solver.

    Handles the interactive loop: checking for updates, displaying the banner,
    gathering parameters, running the solver, and saving/opening the results.
    """
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

    # Gather simulation parameters
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
        # Initialize and run the solver
        s = Solver(
            grid_size=(Nx, Ny), nu=nu, dt=dt,
            lid_speed=top_u, smooth_lid=smooth_lid,
        )
        s.solve(steps, verbose=True)
    except Exception as exc:
        _handle_error(exc)

    # Prepare output directory
    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the visualization
    out_path = os.path.join(output_dir, "result.png")
    save_contour(s.mesh, s.u, s.v, s.p, out_path)

    print()
    print(f"  Result saved to {out_path}")

    # Attempt to automatically open the generated image
    opened = False
    if sys.platform == "darwin":
        subprocess.run(["open", out_path], check=False)
        opened = True
    elif sys.platform.startswith("linux"):
        for cmd in [["xdg-open", out_path],
                    ["explorer.exe", os.path.abspath(out_path)]]:
            try:
                subprocess.run(cmd, check=False)
                opened = True
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
    elif sys.platform == "win32":
        try:
            os.startfile(out_path)
            opened = True
        except OSError:
            pass

    if not opened:
        print(f"  Open output/result.png to view the result.")


if __name__ == "__main__":
    main()
