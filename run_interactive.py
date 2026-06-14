"""
Interactive solver launcher — asks for parameters, runs simulation.

This script provides a user-friendly way to configure and run the CFD solver
interactively through the terminal. It prompts for physical and numerical
parameters, executes the simulation, and displays the result.
"""

import os
import sys
import subprocess
import tempfile

import yaml

from cfd_solver.utils import handle_error


EXAMPLE_SCRIPTS = {
    "cavity": "examples/cavity/run.py",
    "couette": "examples/couette/run.py",
    "taylor_green": "examples/taylor_green/run.py",
    "channel": "examples/channel_flow/run.py",
}

EXAMPLE_DEFAULTS = {
    "cavity": {
        "Nx": 64, "Ny": 64, "nu": 0.01, "dt": 0.001,
        "simulation_time": 20.0, "Lx": 1.0, "Ly": 1.0,
        "lid_speed": 1.0, "smooth_lid": True,
    },
    "couette": {
        "Nx": 32, "Ny": 64, "nu": 0.01, "dt": 0.001,
        "simulation_time": 10.0, "Lx": 1.0, "Ly": 1.0,
    },
    "taylor_green": {
        "Nx": 64, "Ny": 64, "nu": 0.01, "dt": 0.001,
        "simulation_time": 2.0, "Lx": 6.283185307, "Ly": 6.283185307,
    },
    "channel": {
        "Nx": 128, "Ny": 32, "nu": 0.01, "dt": 0.0005,
        "simulation_time": 10.0, "Lx": 10.0, "Ly": 1.0,
    },
}


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


def _make_config(example, params):
    """Build a config dict for the given example from user parameters.

    Parameters
    ----------
    example : str
        Example name: 'cavity', 'couette', 'taylor_green', 'channel'.
    params : dict
        User-specified parameters (Nx, Ny, nu, dt, simulation_time, etc.).

    Returns
    -------
    dict
        A config dict suitable for writing to YAML.
    """
    geo = {
        "Nx": params["Nx"],
        "Ny": params["Ny"],
        "Lx": params["Lx"],
        "Ly": params["Ly"],
    }

    cfg = {
        "geometry": geo,
        "nu": params["nu"],
        "dt": params["dt"],
        "simulation_time": params["simulation_time"],
    }

    if example == "cavity":
        cfg["boundary"] = {
            "top": {"u": params["lid_speed"], "v": 0.0},
            "other": {"u": 0.0, "v": 0.0},
        }
        cfg["smooth_lid"] = params.get("smooth_lid", True)
    elif example == "couette":
        cfg["boundary"] = {
            "top": {"type": "wall", "u": 1.0, "v": 0.0},
            "bottom": {"type": "wall", "u": 0.0, "v": 0.0},
            "left": {"type": "periodic"},
            "right": {"type": "periodic"},
        }
    elif example == "taylor_green":
        cfg["boundary"] = {
            "top": {"type": "free_slip", "u": 0.0},
            "bottom": {"type": "free_slip", "u": 0.0},
            "left": {"type": "periodic"},
            "right": {"type": "periodic"},
        }
    elif example == "channel":
        variant = params.get("variant", "inlet")
        if variant == "inlet":
            cfg["boundary"] = {
                "left": {"type": "inlet", "profile": "parabolic", "U_max": 1.0},
                "right": {"type": "outlet", "method": "zero_gradient"},
                "top": {"type": "wall", "u": 0.0, "v": 0.0},
                "bottom": {"type": "wall", "u": 0.0, "v": 0.0},
            }
        else:
            cfg["body_force"] = {"u": 8.0, "v": 0.0}
            cfg["boundary"] = {
                "top": {"type": "wall", "u": 0.0, "v": 0.0},
                "bottom": {"type": "wall", "u": 0.0, "v": 0.0},
                "left": {"type": "wall", "u": 0.0, "v": 0.0},
                "right": {"type": "wall", "u": 0.0, "v": 0.0},
            }

    return cfg


def _prompt_params(example):
    """Prompt user for parameters for a given example.

    Parameters
    ----------
    example : str
        Example name.

    Returns
    -------
    dict
        User-specified parameters.
    """
    defaults = EXAMPLE_DEFAULTS[example]
    params = {}

    print(f"\n  Configure {example.replace('_', ' ').title()}")
    print("  Press Enter to accept defaults in [brackets].\n")

    params["Nx"] = ask("Grid cells in x", defaults["Nx"])
    params["Ny"] = ask("Grid cells in y", defaults["Ny"])
    params["nu"] = ask("Viscosity (nu)", defaults["nu"])
    params["dt"] = ask("Time step (dt)", defaults["dt"])
    params["simulation_time"] = ask("Simulation time (s)", defaults["simulation_time"])
    params["Lx"] = ask("Domain length Lx", defaults["Lx"])
    params["Ly"] = ask("Domain height Ly", defaults["Ly"])

    if example == "cavity":
        params["lid_speed"] = ask("Lid speed", defaults["lid_speed"])
        smooth = input("  Use smooth lid profile? (y/n) [y]: ").strip().lower()
        params["smooth_lid"] = smooth != "n"
    elif example == "channel":
        variant = input("  Variant (inlet / body-force) [inlet]: ").strip() or "inlet"
        params["variant"] = variant

    return params


def run_example(example):
    """Run a bundled example from the examples/ directory.

    Parameters
    ----------
    example : str
        Example name: 'cavity', 'couette', 'taylor_green', 'channel'.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, EXAMPLE_SCRIPTS[example])
    print(f"  Running {example.replace('_', ' ').title()} (defaults)...")
    print()
    subprocess.run([sys.executable, script_path], check=False)


def run_custom_example(example):
    """Prompt for parameters, write temp config, and run the example.

    Parameters
    ----------
    example : str
        Example name: 'cavity', 'couette', 'taylor_green', 'channel'.
    """
    params = _prompt_params(example)
    cfg = _make_config(example, params)

    # Write temp config
    suffix = f"_{example}_custom.yaml"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="cfd_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        script_path = os.path.join(script_dir, EXAMPLE_SCRIPTS[example])

        print()
        label = example.replace("_", " ").title()
        print(f"  Running {label}...")
        print()
        subprocess.run([sys.executable, script_path, "--config", tmp_path], check=False)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


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
    print("    1) Cavity (defaults)        — lid-driven benchmark")
    print("    2) Couette (defaults)       — parallel plates, periodic x")
    print("    3) Taylor-Green (defaults)  — decaying vortex, analytical")
    print("    4) Channel (defaults)       — Poiseuille (inlet or body force)")
    print()
    print("    5) Custom Cavity            — set grid, viscosity, time, etc.")
    print("    6) Custom Couette           — set grid, viscosity, time, etc.")
    print("    7) Custom Taylor-Green      — set grid, viscosity, time, etc.")
    print("    8) Custom Channel           — set grid, viscosity, time, etc.")
    print()
    print("    0) Quit")
    print()

    choice = input("  Choice [1]: ").strip() or "1"

    simple = {"1": "cavity", "2": "couette", "3": "taylor_green", "4": "channel"}
    custom = {"5": "cavity", "6": "couette", "7": "taylor_green", "8": "channel"}

    if choice == "0":
        return
    elif choice in simple:
        run_example(simple[choice])
    elif choice in custom:
        run_custom_example(custom[choice])
    else:
        print(f"  Unknown choice: {choice}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
