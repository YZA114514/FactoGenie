from __future__ import annotations

"""Visualise transporter tasks (deliveries and repositioning) in time and space."""

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

try:  # pragma: no cover - package import convenience
    from .model import build_model, load_config
    from .planning import load_layout, load_layout_data, compute_route_plans
    from .geometry_utils import oriented_rectangle, rotated_center
except ImportError:  # pragma: no cover
    import sys

    base = Path(__file__).resolve().parent
    sys.path.append(str(base))
    from model import build_model, load_config  # type: ignore
    from planning import load_layout, load_layout_data, compute_route_plans  # type: ignore
    from geometry_utils import oriented_rectangle, rotated_center  # type: ignore


Task = Dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise transporter activity timeline and routes")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "chair_factory.json",
        help="Path to configuration JSON file",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Path to layout JSON file (defaults to config['layout'])",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=120.0,
        help="Simulation horizon",
    )
    parser.add_argument(
        "--transporter",
        type=str,
        default=None,
        help="Filter to a specific transporter id",
    )
    parser.add_argument(
        "--vehicle",
        type=str,
        default=None,
        help="Filter to a specific vehicle id (e.g. cart_A_1)",
    )
    parser.add_argument(
        "--no-layout",
        action="store_true",
        help="Skip spatial layout plot, only draw timeline",
    )
    return parser.parse_args()


def resolve_layout_path(config_path: Path, layout_value: Optional[str], override: Optional[Path]) -> Optional[Path]:
    if override is not None:
        return override
    if not layout_value:
        return None
    raw_path = Path(layout_value)
    candidates: List[Path] = []
    if raw_path.is_absolute():
        candidates.append(raw_path)
    else:
        candidates.append(config_path.parent / raw_path)
        module_root = Path(__file__).resolve().parent
        candidates.append(module_root / raw_path)
        candidates.append(module_root.parent / raw_path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_tasks(
    events: Iterable[Dict[str, object]],
    path_lookup: Dict[str, Dict[Tuple[str, str], Dict[str, object]]],
    transporter_filter: Optional[str] = None,
    vehicle_filter: Optional[str] = None,
) -> List[Task]:
    active: Dict[Tuple[str, str], Dict[str, object]] = {}
    tasks: List[Task] = []

    def ensure_path(transporter: Optional[str], from_node: Optional[str], to_node: Optional[str], path):
        if path:
            return path
        if not transporter or not from_node or not to_node:
            return None
        return path_lookup.get(transporter, {}).get((from_node, to_node), {}).get("path_points")

    for event in events:
        etype = event.get("event")
        transporter_id = str(event.get("transporter")) if event.get("transporter") else None
        vehicle_id = str(event.get("vehicle")) if event.get("vehicle") else None
        if transporter_filter and transporter_id != transporter_filter:
            continue
        if vehicle_filter and vehicle_id != vehicle_filter:
            continue

        if etype == "transport_depart":
            key = (transporter_id or "", vehicle_id or "")
            active[key] = {
                "start": float(event.get("time", 0.0)),
                "transporter": transporter_id,
                "vehicle": vehicle_id,
                "from": event.get("from_node"),
                "to": event.get("to_node"),
                "material": event.get("material"),
                "quantity": event.get("quantity"),
                "path": event.get("path"),
            }
        elif etype == "transport_arrive":
            key = (transporter_id or "", vehicle_id or "")
            data = active.pop(key, None)
            if data is None:
                continue
            start = float(data["start"])
            end = float(event.get("time", start))
            path = ensure_path(transporter_id, data.get("from"), data.get("to"), data.get("path"))
            tasks.append(
                {
                    "kind": "delivery",
                    "transporter": transporter_id,
                    "vehicle": vehicle_id,
                    "from": data.get("from"),
                    "to": data.get("to"),
                    "material": data.get("material"),
                    "quantity": data.get("quantity"),
                    "start": start,
                    "end": end,
                    "path": path,
                }
            )
        elif etype == "transport_reposition":
            start = float(event.get("time", 0.0))
            travel_time = float(event.get("travel_time", 0.0))
            end = start + travel_time
            path = ensure_path(
                transporter_id,
                event.get("from_node"),
                event.get("to_node"),
                event.get("path"),
            )
            tasks.append(
                {
                    "kind": "reposition",
                    "transporter": transporter_id,
                    "vehicle": vehicle_id,
                    "from": event.get("from_node"),
                    "to": event.get("to_node"),
                    "material": None,
                    "quantity": None,
                    "start": start,
                    "end": end,
                    "path": path,
                }
            )
    return tasks


def draw_layout(ax, layout_data: Dict, highlighted_nodes: Optional[Iterable[str]] = None) -> None:
    highlighted = set(highlighted_nodes or [])
    for idx, fu in enumerate(layout_data.get("fus", [])):
        label = str(fu.get("id"))
        x = float(fu.get("x", 0.0))
        y = float(fu.get("y", 0.0))
        length = float(fu.get("length", 0.0))
        width = float(fu.get("width", 0.0))
        angle = float(fu.get("angle", fu.get("angle_deg", 0.0)))
        poly = oriented_rectangle(x, y, length, width, angle)
        ax.add_patch(
            MplPolygon(
                poly,
                closed=True,
                facecolor="#8fbcd4" if label in highlighted else "#cccccc",
                edgecolor="black",
                alpha=0.5,
            )
        )
        cx, cy = rotated_center(x, y, length, width, angle)
        ax.text(cx, cy, label, ha="center", va="center", fontsize=8)
    factory = layout_data.get("factory", {})
    ax.set_xlim(0.0, float(factory.get("length", 50.0)))
    ax.set_ylim(0.0, float(factory.get("width", 30.0)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)


def plot_paths(ax, tasks: Iterable[Task], colors: Dict[str, str]) -> None:
    for task in tasks:
        path = task.get("path")
        if not path:
            continue
        pts = [(float(px), float(py)) for px, py in path]
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        key = f"{task.get('transporter')}:{task.get('vehicle')}"
        color = colors.get(key, "#333333")
        linestyle = "-" if task.get("kind") == "delivery" else "--"
        ax.plot(xs, ys, color=color, linewidth=2.0, linestyle=linestyle, alpha=0.9)
        ax.scatter(xs[0], ys[0], color=color, s=30, marker="o")
        ax.scatter(xs[-1], ys[-1], color=color, s=30, marker="s")


def plot_timeline(ax, tasks: Iterable[Task], colors: Dict[str, str]) -> None:
    # Arrange lanes per transporter/vehicle
    lanes: Dict[str, int] = {}
    lane_height = 0.8
    lane_gap = 0.4
    for task in tasks:
        key = f"{task.get('transporter')}:{task.get('vehicle')}"
        if key not in lanes:
            lanes[key] = len(lanes)

    for task in tasks:
        start = float(task["start"])
        end = float(task["end"])
        duration = max(0.0, end - start)
        key = f"{task.get('transporter')}:{task.get('vehicle')}"
        lane_idx = lanes[key]
        y = lane_idx * (lane_height + lane_gap)
        color = colors.get(key, "#333333")
        face_alpha = 0.9 if task.get("kind") == "delivery" else 0.5
        hatch = None if task.get("kind") == "delivery" else "//"
        label = f"{task.get('material')} ({task.get('from')}→{task.get('to')})" if task.get("kind") == "delivery" else f"Reposition {task.get('from')}→{task.get('to')}"
        ax.broken_barh(
            [(start, duration)],
            (y, lane_height),
            facecolors=color,
            edgecolors="black",
            alpha=face_alpha,
            hatch=hatch,
        )
        ax.text(start + duration / 2 if duration > 0 else start, y + lane_height / 2, label, ha="center", va="center", fontsize=7, color="white")

    ax.set_yticks(
        [
            idx * (lane_height + lane_gap) + lane_height / 2
            for idx in range(len(lanes))
        ]
    )
    ax.set_yticklabels(list(lanes.keys()))
    ax.set_xlabel("Time")
    ax.set_title("Transporter timeline")
    ax.grid(True, axis="x", linestyle="--", alpha=0.3)


def assign_colors(tasks: Iterable[Task]) -> Dict[str, str]:
    palette = [
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
    ]
    colors: Dict[str, str] = {}
    for task in tasks:
        key = f"{task.get('transporter')}:{task.get('vehicle')}"
        if key not in colors:
            colors[key] = palette[len(colors) % len(palette)]
    return colors


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    layout_path = resolve_layout_path(args.config, config.get("layout"), args.layout)
    layout_data = load_layout_data(layout_path) if layout_path else None
    positions = load_layout(layout_path) if layout_path else {}
    if layout_data:
        compute_route_plans(config, positions, layout_data)
    else:
        compute_route_plans(config, positions)

    sim = build_model(config=config)
    sim.run(until=args.duration)

    path_lookup = config.get("_transporter_paths", {})
    tasks = extract_tasks(
        sim.events,
        path_lookup,
        transporter_filter=args.transporter,
        vehicle_filter=args.vehicle,
    )
    if not tasks:
        raise RuntimeError("No transporter tasks found for given filters")

    colors = assign_colors(tasks)

    if args.no_layout or layout_data is None:
        fig, ax = plt.subplots(figsize=(10, 4))
        plot_timeline(ax, tasks, colors)
    else:
        fig, (ax_layout, ax_timeline) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [3, 2]})
        involved_nodes = set()
        for task in tasks:
            if task.get("from"):
                involved_nodes.add(str(task["from"]))
            if task.get("to"):
                involved_nodes.add(str(task["to"]))
        draw_layout(ax_layout, layout_data, highlighted_nodes=involved_nodes)
        plot_paths(ax_layout, tasks, colors)
        ax_layout.set_title("Transporter routes")
        plot_timeline(ax_timeline, tasks, colors)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":  # pragma: no cover
    main()
