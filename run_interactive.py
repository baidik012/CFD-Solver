"""
Interactive solver launcher — asks for parameters, runs simulation.

Provides a REPL-style loop so you can run multiple studies in one session,
validates inputs with physics-aware recommendations (CFL + diffusion
stability), and auto-opens the result image after each run.
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


def _default_output_path(example, project_root, variant=None):
    """Return the canonical output path for a given example run."""
    if example == "channel" and variant:
        return os.path.join(project_root, "output", "channel_flow", f"result_{variant}.png")
    return os.path.join(project_root, "output", example, "result.png")


def ask(prompt, default):
    """
    Prompt user for input, return default if empty.

    Re-prompts (without exiting) on invalid input that can't be cast
    to the default's type.
    """
    while True:
        val = input(f"  {prompt} [{default}]: ").strip()
        if not val:
            return default
        try:
            return type(default)(val)
        except (ValueError, TypeError):
            print(f"    ! Invalid input. Expected a {type(default).__name__}. Try again.")


def _recommend_dt(Nx, Ny, nu, Lx, Ly, example, lid_speed=1.0):
    """
    Recommend a safe dt from CFL (advection) + diffusion stability.

    For explicit diffusion (Couette periodic): dt <= 0.25 * dx^2 / nu.
    For Crank-Nicolson (cavity, channel, Taylor-Green): diffusion-stable,
    only CFL on advection matters: dt <= dx / |u|.
    
    Note: The solver assumes peak speed reaches 3x lid_speed due to
    recirculation, so the effective CFL limit is dt <= 0.5 * dx / (3 * lid_speed).
    """
    dx = Lx / Nx
    dy = Ly / Ny
    if example == "couette":
        # Explicit diffusion — keep dt below the stability limit.
        dt_diff = 0.25 * min(dx, dy) ** 2 / max(nu, 1e-12)
        return round(0.5 * dt_diff, 6)
    # Crank-Nicolson: CFL on advection. Solver assumes 3x peak speed.
    dt_cfl = 0.5 * min(dx, dy) / max(3.0 * abs(lid_speed), 1e-12)
    return round(dt_cfl, 6)


def _validate_params(params, example):
    """
    Check physical/numerical sanity. If issues found, print recommendations
    and ask whether to apply them. Returns (params, ok_to_run).

    Rules:
      - dt must be positive and finite
      - nu must be positive
      - dt stability (per scheme)
      - Reasonable Re / time range
    """
    warnings = []
    Nx, Ny = params["Nx"], params["Ny"]
    nu = params["nu"]
    dt = params["dt"]
    Lx, Ly = params["Lx"], params["Ly"]
    sim_time = params["simulation_time"]
    lid_speed = params.get("lid_speed", 1.0)

    if Nx < 8 or Ny < 8:
        warnings.append(f"Grid {Nx}x{Ny} is very coarse. Use >=16 in each direction.")
    if nu <= 0:
        warnings.append(f"Viscosity must be > 0 (got {nu}).")
    if dt <= 0:
        warnings.append(f"Time step must be > 0 (got {dt}).")
    if sim_time <= 0:
        warnings.append(f"Simulation time must be > 0 (got {sim_time}).")

    dx = Lx / Nx
    dy = Ly / Ny
    if example == "couette":
        # Explicit diffusion stability.
        dt_max = 0.25 * min(dx, dy) ** 2 / max(nu, 1e-12)
        if dt > dt_max:
            warnings.append(
                f"dt={dt} exceeds explicit-diffusion stability limit {dt_max:.6f} "
                f"(0.25*min(dx,dy)^2/nu). Recommended dt={_recommend_dt(Nx, Ny, nu, Lx, Ly, example):.6f}."
            )
    else:
        # Crank-Nicolson: CFL on advection. Solver assumes 3x peak speed.
        dt_cfl = 0.5 * min(dx, dy) / max(3.0 * abs(lid_speed), 1e-12)
        if dt > dt_cfl:
            warnings.append(
                f"dt={dt} exceeds CFL limit {dt_cfl:.6f} (0.5*min(dx,dy)/(3*|u|)). "
                f"Recommended dt={_recommend_dt(Nx, Ny, nu, Lx, Ly, example, lid_speed):.6f}."
            )

    Re = abs(lid_speed) * 1.0 / max(nu, 1e-12)
    if Re > 1e5:
        warnings.append(
            f"Reynolds number Re={Re:.0f} is very high. Solver may be unstable; "
            f"consider reducing lid_speed or raising nu."
        )
    if sim_time > 1000:
        warnings.append(
            f"Simulation time {sim_time}s is very long. "
            f"For cavity, 20-50s usually reaches steady state at Re<1000."
        )

    if not warnings:
        return params, True

    print()
    print("  ⚠ Parameter issues found:")
    for w in warnings:
        print(f"    - {w}")

    # Offer recommendations if dt was the issue.
    rec_dt = _recommend_dt(Nx, Ny, nu, Lx, Ly, example, lid_speed)
    if dt > 0 and abs(rec_dt - dt) / max(dt, 1e-12) > 0.1:
        choice = input(f"  Use recommended dt={rec_dt}? [Y/n]: ").strip().lower()
        if choice != "n":
            params["dt"] = rec_dt
            print(f"    -> dt set to {rec_dt}")

    ans = input("  Continue with these parameters anyway? [Y/n]: ").strip().lower()
    return params, ans != "n"


def _make_config(example, params):
    """Build a config dict for the given example from user parameters."""
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
            "smooth_lid": params.get("smooth_lid", True),
            "top": {"type": "wall", "u": params["lid_speed"], "v": 0.0},
            "bottom": {"type": "wall", "u": 0.0, "v": 0.0},
            "left": {"type": "wall", "u": 0.0, "v": 0.0},
            "right": {"type": "wall", "u": 0.0, "v": 0.0},
        }
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
    """Prompt user for parameters for a given example."""
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


def _resolve_output_path(example, args, project_root, variant=None):
    """Determine output path from CLI args or default location."""
    if "--output" in args:
        i = args.index("--output")
        return args[i + 1] if i + 1 < len(args) else _default_output_path(example, project_root, variant)
    if "-o" in args:
        i = args.index("-o")
        return args[i + 1] if i + 1 < len(args) else _default_output_path(example, project_root, variant)
    return _default_output_path(example, project_root, variant)


def _spawn_example(example, project_root, extra_args=None):
    """
    Run an example script as a subprocess and return its output path.

    The output path is computed from the canonical default, so the caller
    can open the result image after the subprocess returns.
    """
    extra_args = list(extra_args or [])
    script_path = os.path.join(project_root, EXAMPLE_SCRIPTS[example])

    # Determine variant for channel flows.
    variant = None
    if example == "channel":
        for i, a in enumerate(extra_args):
            if a == "--variant" and i + 1 < len(extra_args):
                variant = extra_args[i + 1]
                break
        if variant is None:
            variant = "inlet"

    out_path = _resolve_output_path(example, extra_args, project_root, variant)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    # Always pass --output so we know exactly where the result lands.
    cmd = [sys.executable, script_path] + extra_args
    if "--output" not in extra_args and "-o" not in extra_args:
        cmd += ["--output", out_path]

    print(f"  Running {example.replace('_', ' ').title()}...")
    print()
    subprocess.run(cmd, check=False)
    return out_path


def run_example(example, project_root):
    """Run a bundled example with its default parameters."""
    return _spawn_example(example, project_root, [])


def run_custom_example(example, project_root):
    """Prompt for parameters, validate, write temp config, run example."""
    while True:
        params = _prompt_params(example)
        params, ok = _validate_params(params, example)
        if ok:
            break
        retry = input("  Re-enter parameters? [Y/n]: ").strip().lower()
        if retry == "n":
            print("  Aborted.")
            return None

    cfg = _make_config(example, params)

    suffix = f"_{example}_custom.yaml"
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="cfd_")
    try:
        with os.fdopen(fd, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False)

        script_dir = os.path.dirname(os.path.abspath(__file__))
        # We bypass _spawn_example here because we need to pass the temp config;
        # but we still compute the output path the same way.
        extra_args = ["--config", tmp_path]
        if example == "channel":
            extra_args += ["--variant", params.get("variant", "inlet")]
        return _spawn_example(example, script_dir, extra_args)
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _open_image(out_path):
    """Open an image file with the system viewer (best-effort)."""
    if not out_path or not os.path.exists(out_path):
        print(f"  (Image not found at {out_path})")
        return
    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", out_path])
        elif sys.platform.startswith("linux"):
            for cmd in ("xdg-open", "explorer.exe"):
                try:
                    subprocess.Popen([cmd, os.path.abspath(out_path)])
                    return
                except FileNotFoundError:
                    continue
        elif sys.platform == "win32":
            os.startfile(out_path)
            return
        print(f"  Open {out_path} to view the result.")
    except Exception:
        print(f"  Open {out_path} to view the result.")


def _print_menu():
    print()
    print("========================================")
    print("  CFD Solver — Interactive Setup")
    print("========================================")
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


def _run_choice(choice, project_root):
    """Dispatch one menu choice. Returns True if user wants to continue, False to quit."""
    simple = {"1": "cavity", "2": "couette", "3": "taylor_green", "4": "channel"}
    custom = {"5": "cavity", "6": "couette", "7": "taylor_green", "8": "channel"}

    if choice == "0":
        return False
    elif choice in simple:
        out_path = run_example(simple[choice], project_root)
        if out_path:
            print(f"\n  Result saved to {out_path}")
            _open_image(out_path)
    elif choice in custom:
        out_path = run_custom_example(custom[choice], project_root)
        if out_path:
            print(f"\n  Result saved to {out_path}")
            _open_image(out_path)
    else:
        print(f"  Unknown choice: {choice!r}. Try again.")
    return True


def main():
    """Main entry point — loops until user quits."""
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

    project_root = os.path.dirname(os.path.abspath(__file__))

    print()
    print("========================================")
    print(f"  CFD Solver — Interactive Setup  (v{version_str})")
    print("========================================")
    print("  Run studies back-to-back. Press Ctrl+C to exit at any prompt.")

    try:
        while True:
            _print_menu()
            choice = input("  Choice [1]: ").strip() or "1"
            if not _run_choice(choice, project_root):
                break
            # Pause before redrawing menu so users can see the result.
            input("\n  Press Enter to return to the menu...")
    except KeyboardInterrupt:
        print("\n  Interrupted. Bye.")


if __name__ == "__main__":
    main()