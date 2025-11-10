"""Show the factory layout using the JSON geometry."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

try:
    from .geometry_utils import oriented_rectangle, rotated_center
except ImportError:  # pragma: no cover
    from geometry_utils import oriented_rectangle, rotated_center  # type: ignore

COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
)


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
    data = json.load(layout_path.open("r", encoding="utf-8-sig"))
    factory = data.get("factory", {})
    fus = data.get("fus", [])

    for i, fu in enumerate(fus):
        length = float(fu.get("length", 0.0))
        width = float(fu.get("width", 0.0))
        x = float(fu.get("x", 0.0))
        y = float(fu.get("y", 0.0))
        angle = float(fu.get("angle", fu.get("angle_deg", 0.0)))
        label = str(fu.get("id", f"FU-{i+1}"))

        poly = oriented_rectangle(x, y, length, width, angle)
        ax.add_patch(
            MplPolygon(
                poly,
                closed=True,
                fill=True,
                facecolor=COLORS[i % len(COLORS)],
                alpha=0.35,
                edgecolor="k",
            )
        )
        center = rotated_center(x, y, length, width, angle)
        ax.text(center[0], center[1], label, fontsize=8, ha="center", va="center", color="black")

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, float(factory.get("length", 100)))
    ax.set_ylim(0.0, float(factory.get("width", 60)))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)


def main() -> None:
    args = parse_args()
    fig, ax = plt.subplots(figsize=(9, 6))
    draw_layout(ax, args.layout)
    ax.set_title("Factory Layout")
    plt.show()


if __name__ == "__main__":
    main()
