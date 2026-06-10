"""
Main execution entry point for the CFD Solver package.

This module allows the package to be run directly using `python -m cfd_solver.cli`.
It imports the `main` function from the CLI subpackage and executes it.
"""

from cfd_solver.cli import main

if __name__ == "__main__":
    main()
