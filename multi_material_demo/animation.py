from __future__ import annotations

"""Animate inventory levels on the factory layout."""

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon as MplPolygon

try:
    from .model import build_model, load_config
    from .planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path
except ImportError:  # pragma: no cover
    import sys

    base = Path(__file__).resolve().parent
    sys.path.append(str(base))
    from model import build_model, load_config  # type: ignore
    from planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path  # type: ignore

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
    parser = argparse.ArgumentParser(description="Animate inventory levels")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "simple_four_station.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=Path(__file__).resolve().parent / "layouts" / "simple_four_station.json",
        help="Path to layout JSON",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=200.0,
        help="Simulation horizon",
    )
    return parser.parse_args()


def extract_node_labels(config_data: Dict) -> List[str]:
    nodes: Set[str] = set()
    nodes.update(config_data.get("initial_inventory", {}).keys())
    for route in config_data.get("routes", []):
        nodes.add(str(route.get("from")))
        nodes.add(str(route.get("to")))
    for assembly in config_data.get("assemblies", []):
        nodes.add(str(assembly.get("station")))
    summary = config_data.get("summary", {})
    if summary.get("finished_node"):
        nodes.add(str(summary["finished_node"]))
    return [node for node in nodes if node]


def material_sets(config_data: Dict) -> Tuple[Set[str], Set[str]]:
    produced = {str(item.get("output")) for item in config_data.get("assemblies", []) if item.get("output")}
    initial = set()
    for materials in config_data.get("initial_inventory", {}).values():
        initial.update(str(mat) for mat in materials.keys())
    raw = {mat for mat in initial if mat and mat not in produced}
    return raw, produced


def draw_layout(ax, layout_path: Path, routes: Iterable[Dict] = ()) -> Dict[str, Tuple[float, float]]:
    data = json.load(layout_path.open("r", encoding="utf-8-sig"))
    factory = data.get("factory", {})
    fus = data.get("fus", [])

    centers: Dict[str, Tuple[float, float]] = {}
    for i, fu in enumerate(fus):
        length = float(fu.get("length", 0.0))
        width = float(fu.get("width", 0.0))
        x = float(fu.get("x", 0.0))
        y = float(fu.get("y", 0.0))
        label = str(fu.get("id", f"FU-{i+1}"))

        poly = [(x, y), (x + length, y), (x + length, y + width), (x, y + width)]
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
        center = (x + length / 2, y + width / 2)
        centers[label] = center
        ax.text(center[0], center[1], label, fontsize=8, ha="center", va="center")

    for route in routes:
        path_pts = route.get("path_points")
        if not path_pts:
            continue
        xs = [float(pt[0]) for pt in path_pts]
        ys = [float(pt[1]) for pt in path_pts]
        ax.plot(xs, ys, linestyle="--", color="gray", alpha=0.4)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0.0, float(factory.get("length", 100)))
    ax.set_ylim(0.0, float(factory.get("width", 60)))
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)

    return centers


def build_time_frames(events: List[Dict[str, object]], nodes: Iterable[str]):
    nodes = list(nodes)
    levels: Dict[str, Dict[str, float]] = {node: {} for node in nodes}
    frames: List[Tuple[float, Dict[str, Dict[str, float]]]] = []
    last_time = 0.0
    frames.append((last_time, copy.deepcopy(levels)))
    for event in events:
        ev_type = event.get("event")
        if ev_type not in {"inventory_put", "inventory_get"}:
            continue
        node = event.get("node")
        if node not in nodes:
            continue
        material = str(event.get("material"))
        level = float(event.get("level", 0.0))
        levels[node][material] = level
        time = float(event.get("time", 0.0))
        last_time = time
        frames.append((time, copy.deepcopy(levels)))
    if not frames:
        frames.append((0.0, copy.deepcopy(levels)))
    return frames


def _format_qty(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def format_inventory(materials: Dict[str, float], raw_materials: Set[str], product_materials: Set[str]) -> str:
    if not materials:
        return "(empty)"

    sections: List[str] = []

    def append_section(title: str, items):
        formatted = [f"{mat}: {_format_qty(qty)}" for mat, qty in items if abs(qty) >= 1e-9]
        if formatted:
            sections.append(f"{title}:\n" + "\n".join(formatted))

    raw_items = [
        (mat, qty)
        for mat, qty in sorted(materials.items())
        if mat in raw_materials
    ]
    append_section("Raw", raw_items)

    product_items = [
        (mat, qty)
        for mat, qty in sorted(materials.items())
        if mat in product_materials
    ]
    append_section("Products", product_items)

    other_items = [
        (mat, qty)
        for mat, qty in sorted(materials.items())
        if mat not in raw_materials and mat not in product_materials
    ]
    append_section("Other", other_items)

    return "\n\n".join(sections) if sections else "(empty)"

def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    layout_path = resolve_layout_path(args.config, config_data, args.layout)
    if not layout_path or not layout_path.exists():
        raise FileNotFoundError("Layout file not found; please provide --layout or set config['layout'].")
    layout_positions = load_layout(layout_path)
    layout_data = load_layout_data(layout_path)
    compute_route_plans(config_data, layout_positions, layout_data)
    raw_materials, product_materials = material_sets(config_data)
    node_labels = extract_node_labels(config_data)

    sim = build_model(config=config_data)
    sim.run(until=args.duration)

    fig, ax = plt.subplots(figsize=(9, 6))
    centers = draw_layout(ax, layout_path, config_data.get("routes", []))

    frames = build_time_frames(sim.events, node_labels)

    text_boxes = {}
    for node in node_labels:
        if node not in centers:
            continue
        x, y = centers[node]
        text = ax.text(
            x,
            y - 4,
            "",
            fontsize=8,
            ha="center",
            va="top",
            color="darkblue",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"),
        )
        text_boxes[node] = text

    time_text = ax.text(
        0.02,
        0.95,
        "",
        transform=ax.transAxes,
        fontsize=10,
        ha="left",
        va="top",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="none"),
    )

    def update(frame_index: int):
        t, snapshot = frames[frame_index]
        for node, text in text_boxes.items():
            text.set_text(format_inventory(snapshot.get(node, {}), raw_materials, product_materials))
        time_text.set_text(f"time = {t:.1f}")
        return list(text_boxes.values()) + [time_text]

    anim = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=200,
        blit=False,
        repeat=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
