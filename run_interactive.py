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

from cfd_solver.utils import handle_error


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


def run_example(example):
    """Run a bundled example from the examples/ directory.

    Parameters
    ----------
    example : str
        Example name: 'cavity', 'couette', 'taylor_green', 'channel'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    examples = {
        "cavity": "examples/cavity/run.py",
        "couette": "examples/couette/run.py",
        "taylor_green": "examples/taylor_green/run.py",
        "channel": "examples/channel_flow/run.py",
    }

    script_path = os.path.join(script_dir, examples[example])
    print(f"  Running {example} example...")
    print()
    subprocess.run([sys.executable, script_path], check=False)


def run_custom():
    """Run the interactive custom setup (cavity with user-specified parameters)."""
    print("Press Enter to accept defaults shown in [brackets].")
    print()

    Nx = ask("Grid cells in x", 32)
    Ny = ask("Grid cells in y", 32)
    nu = ask("Viscosity (nu)", 0.01)
    dt = ask("Time step (dt)", 0.001)
    top_u = ask("Lid speed", 1.0)
    smooth_lid = input("  Use smooth lid profile? (y/n) [y]: ").strip().lower() != "n"

    Re = abs(top_u) * 1.0 / max(nu, 1e-10)
    t_conv = 1.0 / max(abs(top_u), 1e-10)
    default_time = t_conv * min(max(10.0, 0.1 * Re), 200.0)
    simulation_time = ask("Simulation time (seconds)", round(default_time, 1))

    print()
    print(f"  Grid: {Nx}x{Ny}  |  nu={nu}  |  dt={dt}  |  time={simulation_time}s  |  lid={top_u}  |  smooth={smooth_lid}")
    print()

    print("  Running solver...")
    print()

    try:
        from cfd_solver.solver import Solver
        from cfd_solver.solver.viz import save_contour
    except ModuleNotFoundError as exc:
        handle_error(exc)

    try:
        s = Solver(
            grid_size=(Nx, Ny), nu=nu, dt=dt,
            lid_speed=top_u, smooth_lid=smooth_lid,
        )
        ok = s.solve(simulation_time=simulation_time, verbose=True)
    except Exception as exc:
        handle_error(exc)

    if not ok:
        print()
        print("  Simulation aborted due to blowup; not saving NaN output.")
        print("  Reduce dt (or lid speed), then try again.")
        raise SystemExit(1)

    project_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(project_root, "output")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "result.png")
    save_contour(s.mesh, s.u, s.v, s.p, out_path)

    print()
    print(f"  Result saved to {out_path}")

    _open_image(out_path)


def _open_image(out_path):
    """Attempt to open an image file with the system viewer."""
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


def main():
    """Main entry point for the interactive CFD solver."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
        from cfd_solver.version_check import check_for_updates
        check_for_updates(repo_dir=os.path.dirname(__file__) or ".")
    except Exception:
        pass

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
    print("  Select an example to run:")
    print()
    print("    1) Lid-Driven Cavity  — classic benchmark")
    print("    2) Couette Flow       — parallel plates, periodic x")
    print("    3) Taylor-Green Vortex — decaying vortex, analytical")
    print("    4) Channel Flow       — Poiseuille (inlet or body force)")
    print("    5) Custom Setup       — configure parameters manually")
    print("    0) Quit")
    print()

    choice = input("  Choice [1]: ").strip() or "1"

    examples = {
        "1": "cavity",
        "2": "couette",
        "3": "taylor_green",
        "4": "channel",
    }

    if choice == "0":
        return
    elif choice in examples:
        run_example(examples[choice])
    elif choice == "5":
        run_custom()
    else:
        print(f"  Unknown choice: {choice}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
