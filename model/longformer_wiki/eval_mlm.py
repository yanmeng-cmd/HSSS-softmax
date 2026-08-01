#!/usr/bin/env python3
"""Thin entry point for the Longformer fixed-mask MLM experiment."""

from __future__ import annotations

try:
    from .mlm_runner import main
except ImportError:  # pragma: no cover - supports direct script execution
    from mlm_runner import main


if __name__ == "__main__":
    main()
