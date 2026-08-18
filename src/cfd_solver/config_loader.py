"""Shared helper for loading and validating YAML configs.

This module exists to break the duplication between the CLI's ``run``
command and the example ``run.py`` scripts. Both previously loaded YAML
configs and constructed a Solver with slightly different boilerplate;
the examples skipped :func:`~cfd_solver.solver.validate.validate_config`
entirely, so typos in ``examples/*/config.yaml`` failed at runtime with
cryptic errors rather than at load time with clear messages.

Usage::

    from cfd_solver.config_loader import load_config
    cfg = load_config("examples/cavity/config.yaml")
    geo = cfg["geometry"]
    solver = Solver(grid_size=(geo["Nx"], geo["Ny"]), ...)

The returned dict is guaranteed to have passed schema validation.
"""

import yaml

from .solver.validate import validate_config


def load_config(path):
    """Load a YAML config file and validate it against the schema.

    Parameters
    ----------
    path : str
        Path to the YAML config file.

    Returns
    -------
    dict
        The parsed and validated config.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    SystemExit(1)
        If the config fails schema validation. Error messages are
        printed to stderr before exit, mirroring the CLI's behaviour.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    errors = validate_config(cfg)
    if errors:
        import sys
        print("Config validation errors:", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        raise SystemExit(1)

    return cfg