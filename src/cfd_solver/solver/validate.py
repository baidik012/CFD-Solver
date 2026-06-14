"""YAML config validation for the CFD solver.

Provides validate_config() to check a parsed YAML dict before passing
it to the Solver. Returns a list of human-readable error strings;
an empty list means the config is valid.
"""

_SCHEMA = {
    "geometry": {
        "required": True,
        "type": dict,
        "fields": {
            "Lx": {"type": (int, float), "min": 0, "exclusive_min": True, "required": True},
            "Ly": {"type": (int, float), "min": 0, "exclusive_min": True, "required": True},
            "Nx": {"type": int, "min": 2, "required": True},
            "Ny": {"type": int, "min": 2, "required": True},
        },
    },
    "nu": {"type": (int, float), "min": 0, "required": True},
    "dt": {"type": (int, float), "min": 0, "exclusive_min": True, "required": True},
    "steps": {"type": int, "min": 1},
    "simulation_time": {"type": (int, float), "min": 0, "exclusive_min": True},
    "boundary": {
        "type": dict,
        "fields": {
            "smooth_lid": {"type": bool},
            # Per-wall specification (all four walls use the same schema)
            "top": {
                "type": dict,
                "fields": {
                    "type": {"type": str, "values": [
                        "wall", "inlet", "outlet", "periodic", "free_slip",
                    ]},
                    "u": {"type": (int, float)},
                    "v": {"type": (int, float)},
                    "profile": {"type": str, "values": ["uniform", "parabolic"]},
                    "U_max": {"type": (int, float)},
                    "method": {"type": str, "values": ["zero_gradient", "convective"]},
                },
            },
            "other": {
                "type": dict,
                "fields": {
                    "u": {"type": (int, float)},
                    "v": {"type": (int, float)},
                },
            },
            # New per-wall specification
            "left": {
                "type": dict,
                "fields": {
                    "type": {"type": str, "values": [
                        "wall", "inlet", "outlet", "periodic", "free_slip",
                    ]},
                    "u": {"type": (int, float)},
                    "v": {"type": (int, float)},
                    "profile": {"type": str, "values": ["uniform", "parabolic"]},
                    "U_max": {"type": (int, float)},
                    "method": {"type": str, "values": ["zero_gradient", "convective"]},
                },
            },
            "right": {
                "type": dict,
                "fields": {
                    "type": {"type": str, "values": [
                        "wall", "inlet", "outlet", "periodic", "free_slip",
                    ]},
                    "u": {"type": (int, float)},
                    "v": {"type": (int, float)},
                    "profile": {"type": str, "values": ["uniform", "parabolic"]},
                    "U_max": {"type": (int, float)},
                    "method": {"type": str, "values": ["zero_gradient", "convective"]},
                },
            },
            "bottom": {
                "type": dict,
                "fields": {
                    "type": {"type": str, "values": [
                        "wall", "inlet", "outlet", "periodic", "free_slip",
                    ]},
                    "u": {"type": (int, float)},
                    "v": {"type": (int, float)},
                    "profile": {"type": str, "values": ["uniform", "parabolic"]},
                    "U_max": {"type": (int, float)},
                    "method": {"type": str, "values": ["zero_gradient", "convective"]},
                },
            },
        },
    },
    "advection_scheme": {"type": str, "values": ["upwind", "central"]},
    "diffusion_scheme": {"type": str, "values": ["crank_nicolson", "explicit"]},
    "body_force": {
        "type": dict,
        "fields": {
            "u": {"type": (int, float, str)},
            "v": {"type": (int, float, str)},
        },
    },
    "convergence": {
        "type": dict,
        "fields": {
            "tol": {"type": (int, float), "min": 0, "exclusive_min": True},
            "window": {"type": int, "min": 1},
        },
    },
}


def validate_config(cfg):
    """Validate a parsed YAML config dict.

    Parameters
    ----------
    cfg : dict
        The loaded YAML config.

    Returns
    -------
    list of str
        Empty if valid, otherwise a list of error messages.
    """
    if not isinstance(cfg, dict):
        return ["Config must be a YAML mapping (key: value pairs)."]

    errors = []

    # Flag unknown top-level keys (e.g. the typo 'step' instead of 'steps'),
    # which previously passed validation silently.
    known = set(_SCHEMA.keys())
    for key in cfg:
        if key not in known:
            errors.append(f"Unknown field: {key}")

    _check_dict(cfg, _SCHEMA, "", errors)

    # Either steps or simulation_time must be provided
    if "steps" not in cfg and "simulation_time" not in cfg:
        errors.append("Missing required field: either 'steps' or 'simulation_time' must be set")

    return errors


def _check_dict(data, schema, prefix, errors):
    """Recursively validate a dict against a schema."""
    for key, spec in schema.items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"

        if spec.get("required") and key not in data:
            errors.append(f"Missing required field: {path}")
            continue

        if key not in data:
            continue

        val = data[key]
        _check_value(val, spec, path, errors)


def _check_fields(data, schema, prefix, errors):
    """Check extra/unknown keys and validate known ones."""
    if "fields" not in schema:
        return

    known = set(schema["fields"].keys())
    for key in data:
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if key not in known:
            errors.append(f"Unknown field: {path}")

    for key, spec in schema["fields"].items():
        path = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if spec.get("required") and key not in data:
            errors.append(f"Missing required field: {path}")
        elif key in data:
            _check_value(data[key], spec, path, errors)


def _check_value(val, spec, path, errors):
    """Validate a single value against a spec."""
    expected = spec.get("type")
    if expected is not None:
        if not isinstance(expected, tuple):
            expected = (expected,)
        if not isinstance(val, expected):
            type_names = " or ".join(t.__name__ for t in expected)
            errors.append(f"{path}: expected {type_names}, got {type(val).__name__}")
            return

    if "values" in spec and val not in spec["values"]:
        allowed = ", ".join(repr(v) for v in spec["values"])
        errors.append(f"{path}: must be one of {allowed}, got {val!r}")

    if "min" in spec:
        try:
            if spec.get("exclusive_min"):
                if val <= spec["min"]:
                    errors.append(f"{path}: must be > {spec['min']}, got {val}")
            else:
                if val < spec["min"]:
                    errors.append(f"{path}: must be >= {spec['min']}, got {val}")
        except TypeError:
            pass

    if isinstance(val, dict) and "fields" in spec:
        _check_fields(val, spec, path, errors)
