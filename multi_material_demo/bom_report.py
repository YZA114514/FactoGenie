"""Generate a BOM reconciliation report."""

import argparse
from pathlib import Path

def main() -> None:  # pragma: no cover
    """兼容旧入口，直接调用 multi_material_demo.bom.main"""
    from .bom import main as bom_main

    bom_main()


if __name__ == "__main__":  # pragma: no cover
    main()
