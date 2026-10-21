"""Module entry point so ``python -m pytracer_cli`` runs the CLI.

Robust in environments where the ``pytracer`` console script is not on PATH (for
example a clean checkout that installs the project but not onto PATH), which is
how the cross-branch runner invokes a collection.
"""

from __future__ import annotations

from .main import app

if __name__ == "__main__":
    app()
