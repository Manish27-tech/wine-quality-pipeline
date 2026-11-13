"""Convenience entry point so the pipeline works without installing the package.

Usage:
    python run.py download
    python run.py eda
    python run.py train
    python run.py all
"""

from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wine.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
