"""
Lightweight version check against the remote GitHub repository.

Compares the local git HEAD commit to the latest commit on origin/main.
Prints a one-line notice if the local copy is behind. No network call is
made unless git fetch succeeds silently in the background.

Designed to be called once at CLI startup. All failures are swallowed so
a missing git binary or network error never breaks the solver.
"""

import subprocess


# Human-readable update command shown in the notice
_UPDATE_CMD_UNIX = "./update.sh"
_UPDATE_CMD_WIN  = "update.bat"


def _run(cmd, **kwargs):
    """
    Run a subprocess and return its stdout.

    Parameters
    ----------
    cmd : list of str
        The command to execute as a list of arguments.
    **kwargs : dict
        Additional keyword arguments passed to subprocess.run.

    Returns
    -------
    str or None
        The trimmed stdout of the command, or None if the command failed.
    """
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=5,
            **kwargs,
        )
        if result.returncode == 0:
            return result.stdout.decode().strip()
    except Exception:
        pass
    return None


def check_for_updates(repo_dir=None):
    """
    Print a notice if the local repository is behind origin/main.

    Parameters
    ----------
    repo_dir : str, optional
        Path to the git repository root. Defaults to the current working directory.

    Returns
    -------
    bool
        True if an update is available (local is behind remote),
        False otherwise (including on any error or if up-to-date).
    """
    cwd = repo_dir  # None → subprocess inherits the caller's cwd

    # Silently fetch origin so we have fresh remote refs.
    # Use --quiet and a short timeout to avoid visible delays.
    _run(["git", "fetch", "origin", "main", "--quiet"], cwd=cwd)

    # Get local and remote commit hashes
    local  = _run(["git", "rev-parse", "HEAD"],            cwd=cwd)
    remote = _run(["git", "rev-parse", "origin/main"],     cwd=cwd)

    if not local or not remote:
        return False  # Not a git repo, git not installed, or remote not reachable

    if local == remote:
        import sys
        print("  You're up to date.", file=sys.stderr)
        return False

    # Count how many commits behind the local HEAD is
    behind_str = _run(
        ["git", "rev-list", "--count", f"HEAD..origin/main"],
        cwd=cwd,
    )
    behind = int(behind_str) if behind_str and behind_str.isdigit() else 0

    import sys
    import os
    cmd = _UPDATE_CMD_WIN if sys.platform == "win32" else _UPDATE_CMD_UNIX
    plural = "commit" if behind == 1 else "commits"

    print(
        f"\n⚠  Update available: your copy is {behind} {plural} behind.\n"
        f"   Run  {cmd}  to get the latest changes.\n",
        file=sys.stderr,
    )
    return True
