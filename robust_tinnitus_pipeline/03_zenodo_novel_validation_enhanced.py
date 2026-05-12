"""
Compatibility entry point for the enhanced Zenodo validation command.

The hardened implementation now lives in 04_riemannian_hardening.py and keeps
the tri-ensemble validation layer. This file preserves older launcher commands
while routing them to the modern Riemannian pipeline.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_riemannian_main():
    module_path = Path(__file__).with_name("04_riemannian_hardening.py")
    spec = importlib.util.spec_from_file_location("riemannian_hardening", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.main


if __name__ == "__main__":
    main = _load_riemannian_main()
    main()
