"""
Lightweight version check against the remote repository.

Compares the local git HEAD commit of THIS package's repository to the
latest commit on origin/main. Prints a one-line notice if the local copy
is behind.

Designed to be called once at CLI startup. All failures are swallowed so
a missing git binary or network error never breaks the solver. Set the
environment variable CFD_SOLVER_NO_UPDATE_CHECK=1 to disable entirely.

Caching (audit finding P2-14):
    The check is cached on disk for 24 hours so that repeated CLI
    invocations do not each pay the network latency of
    ``git fetch origin``.  The cache file lives at
    ``~/.cache/cfd-solver/last_check`` (XDG-compatible) and stores the
    last check timestamp.  Pass ``force=True`` to bypass the cache.
"""

import os
import subprocess
import sys
import time


# How long (in seconds) between update checks.  24 hours.
_CACHE_TTL_SECONDS = 24 * 60 * 60


def _cache_path():
    """Return the path to the update-check cache file."""
    # Respect XDG_CACHE_HOME if set; otherwise default to ~/.cache.
    cache_dir = os.environ.get(
        'XDG_CACHE_HOME',
        os.path.join(os.path.expanduser('~'), '.cache'),
    )
    return os.path.join(cache_dir, 'cfd-solver', 'last_check')


def _cache_is_fresh():
    """Return True if the update check has run within the cache TTL."""
    try:
        mtime = os.path.getmtime(_cache_path())
        return (time.time() - mtime) < _CACHE_TTL_SECONDS
    except OSError:
        return False


def _touch_cache():
    """Create or update the cache file's mtime."""
    try:
        path = _cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Touch: create if missing, update mtime if exists.
        with open(path, 'a'):
            os.utime(path, None)
    except OSError:
        # Cache failures must never break the CLI.
        pass


def _run(cmd, cwd=None, timeout=3):
    """
    Run a subprocess and return its stdout.

    Parameters
    ----------
    cmd : list of str
        The command to execute as a list of arguments.
    cwd : str, optional
        Working directory for the command.
    timeout : float, optional
        Maximum seconds to wait (default 3).

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
            timeout=timeout,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.decode().strip()
    except Exception:
        pass
    return None


def _package_repo_dir():
    """Return the repository root containing this package (repo/src/cfd_solver)."""
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(os.path.dirname(pkg_dir))


def check_for_updates(repo_dir=None, force=False):
    """
    Print a notice if the local repository is behind origin/main.

    Git operations are anchored to this package's own repository (or the
    explicit repo_dir), never the caller's working directory, so running
    the CLI from inside an unrelated git repository cannot produce a
    bogus notice.

    The check is cached on disk for 24 hours (see module docstring) so
    that repeated CLI invocations do not each pay the network latency.
    Pass ``force=True`` to bypass the cache.

    Parameters
    ----------
    repo_dir : str, optional
        Path to the git repository root. Defaults to the repository
        containing this package.
    force : bool, optional
        If True, bypass the on-disk cache and always run the check.

    Returns
    -------
    bool
        True if an update is available (local is behind remote),
        False otherwise (including on any error or if up-to-date).
    """
    if os.environ.get("CFD_SOLVER_NO_UPDATE_CHECK"):
        return False

    # Cache check (audit finding P2-14).  Skip the network call if we
    # have checked recently.
    if not force and _cache_is_fresh():
        return False

    cwd = repo_dir or _package_repo_dir()

    # Only proceed if the anchored directory actually is a git work tree.
    inside = _run(["git", "rev-parse", "--is-inside-work-tree"], cwd=cwd)
    if inside != "true":
        return False

    # Silently fetch origin so we have fresh remote refs.
    _run(["git", "fetch", "origin", "main", "--quiet"], cwd=cwd, timeout=5)

    # Mark the cache as fresh regardless of whether the fetch succeeded,
    # so a transient network failure does not cause a check storm.
    _touch_cache()

    # Get local and remote commit hashes
    local = _run(["git", "rev-parse", "HEAD"], cwd=cwd)
    remote = _run(["git", "rev-parse", "origin/main"], cwd=cwd)

    if not local or not remote or local == remote:
        # Not resolvable, or up to date: stay silent.
        return False

    # Count how many commits behind the local HEAD is
    behind_str = _run(["git", "rev-list", "--count", "HEAD..origin/main"], cwd=cwd)
    behind = int(behind_str) if behind_str and behind_str.isdigit() else 0
    if behind == 0:
        # Local is ahead of (or diverged from) origin/main; not "behind".
        return False

    cmd = "update.bat" if sys.platform == "win32" else "./update.sh"
    plural = "commit" if behind == 1 else "commits"
    print(
        f"\n!  Update available: your copy is {behind} {plural} behind.\n"
        f"   Run  {cmd}  from the repository root to get the latest changes.\n"
        f"   (Set CFD_SOLVER_NO_UPDATE_CHECK=1 to disable this check.)\n",
        file=sys.stderr,
    )
    return True
