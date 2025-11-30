from __future__ import annotations

"""Shared layout drawing helpers for visualisations (FUs + obstacles)."""

from typing import Dict, Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

try:  # pragma: no cover
    from .geometry_utils import oriented_rectangle_with_notch, polygon_centroid
except ImportError:  # pragma: no cover
    from geometry_utils import oriented_rectangle_with_notch, polygon_centroid  # type: ignore

Point = Tuple[float, float]

DEFAULT_COLORS = {
    "highlight": "#8fbcd4",
    "used": "#8fbcd4",
    "unused": "#cccccc",
    "obstacle": "#9a9a9a",
}


def draw_layout(
    ax: plt.Axes,
    layout_data: Dict,
    highlight_nodes: Optional[Iterable[str]] = None,
    used_nodes: Optional[Iterable[str]] = None,
    colors: Optional[Dict[str, str]] = None,
) -> Dict[str, Point]:
    """
    Draw FUs和障碍物；返回每个 FU 的中心坐标。

    highlight_nodes: 高亮节点（优先级最高）
    used_nodes: 参与路线的节点（次优先）
    """

    palette = {**DEFAULT_COLORS, **(colors or {})}
    highlight = set(highlight_nodes or [])
    used = set(used_nodes or [])

    centers: Dict[str, Point] = {}
    for idx, fu in enumerate(layout_data.get("fus", [])):
        label = str(fu.get("id"))
        x = float(fu.get("x", 0.0))
        y = float(fu.get("y", 0.0))
        length = float(fu.get("length", 0.0))
        width = float(fu.get("width", 0.0))
        angle = float(fu.get("angle", fu.get("angle_deg", 0.0)))
        notch_l = float(fu.get("notch_length", 0.0))
        notch_w = float(fu.get("notch_width", 0.0))
        if label in highlight:
            face = palette["highlight"]
        elif label in used:
            face = palette["used"]
        else:
            face = palette["unused"]
        ax.add_patch(
            MplPolygon(
                oriented_rectangle_with_notch(x, y, length, width, angle, notch_l, notch_w),
                closed=True,
                facecolor=face,
                edgecolor="k",
                alpha=0.5,
            )
        )
        cx, cy = polygon_centroid(oriented_rectangle_with_notch(x, y, length, width, angle, notch_l, notch_w))
        centers[label] = (cx, cy)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8)

    for obs in layout_data.get("obstacles", []):
        x = float(obs.get("x", 0.0))
        y = float(obs.get("y", 0.0))
        length = float(obs.get("length", 0.0))
        width = float(obs.get("width", 0.0))
        angle = float(obs.get("angle", obs.get("angle_deg", 0.0)))
        notch_l = float(obs.get("notch_length", 0.0))
        notch_w = float(obs.get("notch_width", 0.0))
        poly = oriented_rectangle_with_notch(x, y, length, width, angle, notch_l, notch_w)
        ax.add_patch(
            MplPolygon(
                poly,
                closed=True,
                facecolor=palette["obstacle"],
                edgecolor="k",
                alpha=0.6,
                hatch="///",
            )
        )
        label = str(obs.get("id", "")).strip()
        if label:
            cx, cy = polygon_centroid(poly)
            ax.text(cx, cy, label, ha="center", va="center", fontsize=7, color="black")

    factory = layout_data.get("factory", {})
    ax.set_xlim(0.0, float(factory.get("length", 50.0)))
    ax.set_ylim(0.0, float(factory.get("width", 30.0)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)
    return centers


__all__ = ["draw_layout", "DEFAULT_COLORS"]
