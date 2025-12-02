"""Show the factory layout using the JSON geometry."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .layout_validation import validate_layout_data
    from .visual_utils import draw_layout
except ImportError:  # pragma: no cover
    from layout_validation import validate_layout_data  # type: ignore
    from visual_utils import draw_layout  # type: ignore

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot layout from JSON")
    parser.add_argument(
        "--layout",
        type=Path,
        default=Path(__file__).resolve().parent / "layouts" / "layout_episode_00002.json",
        help="Path to layout JSON file",
    )
    return parser.parse_args()


def draw_layout(ax, layout_path: Path) -> None:
    import json

    data = json.load(layout_path.open("r", encoding="utf-8-sig"))
    errors = validate_layout_data(data, allow_touching=True)
    if errors:
        raise ValueError("布局几何合法性检查失败:\n" + "\n".join(f"- {msg}" for msg in errors))
    draw_layout(ax, data)


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_layout(ax, args.layout)
    ax.set_title("Factory Layout")
    plt.show()


if __name__ == "__main__":
    main()
