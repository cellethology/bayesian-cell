"""Compatibility entry point; prefer ``python -m cellsim``.

The command-line implementation lives in :mod:`cellsim.cli`.
"""

from cellsim.cli import build_parser, main

if __name__ == "__main__":
    raise SystemExit(main())
