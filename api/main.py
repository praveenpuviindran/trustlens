"""Thin wrapper for Render entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from trustlens.api.main import app
except ModuleNotFoundError:
    # Support running from repo root without an installed package.
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    sys.path.insert(0, str(src_path))
    from trustlens.api.main import app  # type: ignore

__all__ = ["app"]
