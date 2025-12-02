"""Generate a BOM reconciliation report."""

import argparse
from pathlib import Path

def main() -> None:  # pragma: no cover
    """兼容旧入口，直接调用 multi_material_demo.bom.main"""
    try:
        from .bom import main as bom_main
    except ImportError:
        import sys
        from pathlib import Path

        sys.path.append(str(Path(__file__).resolve().parent))
        from bom import main as bom_main  # type: ignore

    bom_main()


if __name__ == "__main__":  # pragma: no cover
    main()
