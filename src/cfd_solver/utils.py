"""Shared utilities for user-facing error handling."""

import sys
import yaml


_FRIENDLY_ERRORS = {
    ModuleNotFoundError: (
        "Missing required package. Run:\n"
        "  pip install -r requirements.txt\n"
        "or:\n"
        "  pip install -e ."
    ),
    FileNotFoundError: "File not found. Check that the path is correct.",
    PermissionError: "Permission denied. Check file/folder permissions.",
    yaml.YAMLError: "Config file is invalid YAML. Check the syntax.",
    KeyError: "Config file is missing a required field. Check the YAML keys.",
    ValueError: "Invalid parameter value in config file.",
    MemoryError: "Not enough memory. Try a smaller grid size.",
}


def handle_error(exc):
    """Print a user-friendly message and exit.

    Parameters
    ----------
    exc : Exception
        The exception that was raised.
    """
    print("\n" + "=" * 50, file=sys.stderr)
    print("  ERROR", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    print(f"  {type(exc).__name__}: {exc}", file=sys.stderr)
    hint = _FRIENDLY_ERRORS.get(type(exc))
    if hint:
        print(f"\n  Hint: {hint}", file=sys.stderr)

    print("=" * 50 + "\n", file=sys.stderr)
    raise SystemExit(1)
