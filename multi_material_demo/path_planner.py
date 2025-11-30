from __future__ import annotations

"""Compute grid-based shortest paths for transporter routes.

Loads a configuration/layout pair, validates geometry, derives rounded integer
centroids for each FU, then computes obstacle-aware shortest paths (4-neighbour
grid) between node pairs handled by each transporter. Results can be visualised
on top of the layout and optionally written back into the config.
"""

import argparse
import json
import math
from collections import deque
from itertools import combinations, cycle
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt

try:  # pragma: no cover - package/local import convenience
    from .model import load_config
    from .geometry_utils import (
        oriented_rectangle_with_notch,
        point_in_polygon,
        polygon_bounds,
        polygon_centroid,
        min_distance_to_edges,
    )
    from .visual_utils import draw_layout
    from .layout_validation import validate_layout_data as _validate_layout
except ImportError:  # pragma: no cover
    import sys

    base = Path(__file__).resolve().parent
    sys.path.append(str(base))
    from model import load_config  # type: ignore
    from geometry_utils import (  # type: ignore
        oriented_rectangle_with_notch,
        point_in_polygon,
        polygon_bounds,
        polygon_centroid,
        min_distance_to_edges,
    )
    from visual_utils import draw_layout  # type: ignore
    from layout_validation import validate_layout_data as _validate_layout  # type: ignore


GridPoint = Tuple[int, int]

OBSTACLE_MARGIN = 1.0  # grid units to erode FU obstacles for path planning


def round_half_up(value: float) -> int:
    """Round half away from zero (since Python's round is bankers')."""

    return int(math.floor(value + 0.5))


def load_layout_data(layout_path: Path) -> Dict:
    return json.loads(layout_path.read_text(encoding="utf-8-sig"))


def fu_centers(layout_data: Dict) -> Dict[str, GridPoint]:
    centers: Dict[str, GridPoint] = {}
    for fu in layout_data.get("fus", []):
        label = str(fu.get("id"))
        if not label:
            continue
        x = float(fu.get("x", 0.0))
        y = float(fu.get("y", 0.0))
        length = float(fu.get("length", 0.0))
        width = float(fu.get("width", 0.0))
        angle = float(fu.get("angle", fu.get("angle_deg", 0.0)))
        notch_l = float(fu.get("notch_length", 0.0))
        notch_w = float(fu.get("notch_width", 0.0))
        poly = oriented_rectangle_with_notch(x, y, length, width, angle, notch_l, notch_w)
        cx, cy = polygon_centroid(poly)
        centers[label] = (round_half_up(cx), round_half_up(cy))
    return centers


def obstacle_points(layout_data: Dict, allowed_nodes: Set[str], clearance: float = 0.0) -> Set[GridPoint]:
    """Aggregate obstacle cells from other FUs (excluding endpoints) and explicit obstacles."""

    obstacles: Set[GridPoint] = set()

    def accumulate_rect(x: float, y: float, length: float, width: float, angle: float, notch_l: float = 0.0, notch_w: float = 0.0) -> None:
        poly = oriented_rectangle_with_notch(x, y, length, width, angle, notch_l, notch_w)
        (min_x, max_x), (min_y, max_y) = polygon_bounds(poly)
        x_min = math.floor(min_x)
        x_max = math.ceil(max_x)
        y_min = math.floor(min_y)
        y_max = math.ceil(max_y)
        for ix in range(x_min, x_max + 1):
            for iy in range(y_min, y_max + 1):
                pt = (ix, iy)
                if not point_in_polygon(pt, poly, include_boundary=True):
                    continue
                if clearance > 0.0 and min_distance_to_edges(pt, poly) < clearance:
                    continue
                obstacles.add(pt)

    for fu in layout_data.get("fus", []):
        label = str(fu.get("id"))
        if label in allowed_nodes:
            continue
        x = float(fu.get("x", 0.0))
        y = float(fu.get("y", 0.0))
        length = float(fu.get("length", 0.0))
        width = float(fu.get("width", 0.0))
        angle = float(fu.get("angle", fu.get("angle_deg", 0.0)))
        notch_l = float(fu.get("notch_length", 0.0))
        notch_w = float(fu.get("notch_width", 0.0))
        accumulate_rect(x, y, length, width, angle, notch_l, notch_w)

    for obs in layout_data.get("obstacles", []):
        x = float(obs.get("x", 0.0))
        y = float(obs.get("y", 0.0))
        length = float(obs.get("length", 0.0))
        width = float(obs.get("width", 0.0))
        angle = float(obs.get("angle", obs.get("angle_deg", 0.0)))
        if length <= 0 or width <= 0:
            continue
        notch_l = float(obs.get("notch_length", 0.0))
        notch_w = float(obs.get("notch_width", 0.0))
        accumulate_rect(x, y, length, width, angle, notch_l, notch_w)

    return obstacles


def shortest_path(
    start: GridPoint,
    goal: GridPoint,
    obstacles: Set[GridPoint],
    bounds: Tuple[Tuple[int, int], Tuple[int, int]],
) -> List[GridPoint]:
    """Breadth-first search over the integer grid."""

    if start == goal:
        return [start]

    x_bounds, y_bounds = bounds
    min_x, max_x = x_bounds
    min_y, max_y = y_bounds
    if start in obstacles or goal in obstacles:
        raise ValueError("Start or goal lies within an obstacle")

    queue: deque[GridPoint] = deque([start])
    parent: Dict[GridPoint, Optional[GridPoint]] = {start: None}
    neighbours = ((1, 0), (-1, 0), (0, 1), (0, -1))

    while queue:
        current = queue.popleft()
        if current == goal:
            break
        cx, cy = current
        for dx, dy in neighbours:
            nx, ny = cx + dx, cy + dy
            next_pt = (nx, ny)
            if next_pt in parent:
                continue
            if not (min_x <= nx <= max_x and min_y <= ny <= max_y):
                continue
            if next_pt in obstacles:
                continue
            parent[next_pt] = current
            queue.append(next_pt)

    if goal not in parent:
        raise ValueError(f"No feasible path between {start} and {goal}")

    path: List[GridPoint] = []
    current: Optional[GridPoint] = goal
    while current is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def path_length(path: Iterable[GridPoint]) -> float:
    pts = list(path)
    if len(pts) < 2:
        return 0.0
    return float(len(pts) - 1)


def transporter_speed(config: Dict, transporter_id: str) -> float:
    default_speed = float(config.get("default_speed", 1.0)) or 1.0
    for entry in config.get("transporters", []):
        if entry.get("id") == transporter_id:
            return float(entry.get("speed", default_speed)) or default_speed
    return default_speed


def compute_routes(
    config: Dict,
    layout_data: Dict,
    transporter_filter: Optional[str] = None,
    include_all_pairs: bool = True,
) -> List[Dict]:
    factory = layout_data.get("factory", {})
    max_x = int(math.ceil(float(factory.get("length", 0.0))))
    max_y = int(math.ceil(float(factory.get("width", 0.0))))
    centers = fu_centers(layout_data)

    routes = config.get("routes", [])
    grouped: Dict[str, List[Tuple[int, Dict]]] = {}
    for idx, route in enumerate(routes):
        transporter_id = route.get("transporter_id")
        if transporter_id is None:
            continue
        if transporter_filter and transporter_id != transporter_filter:
            continue
        grouped.setdefault(transporter_id, []).append((idx, route))

    results: List[Dict] = []
    obstacle_cache: Dict[Tuple[frozenset[str], float], Set[GridPoint]] = {}

    for transporter_id, entries in grouped.items():
        bounds = ((0, max_x), (0, max_y))
        speed = transporter_speed(config, transporter_id)
        pair_to_index: Dict[Tuple[str, str], int] = {}
        nodes: Set[str] = set()
        for idx, route in entries:
            from_node = str(route.get("from"))
            to_node = str(route.get("to"))
            pair_to_index[(from_node, to_node)] = idx
            nodes.add(from_node)
            nodes.add(to_node)

        pairs_to_process: Set[Tuple[str, str]] = set()
        if include_all_pairs:
            for a, b in combinations(sorted(nodes), 2):
                pairs_to_process.add((a, b))
                pairs_to_process.add((b, a))
        pairs_to_process.update(pair_to_index.keys())

        for from_node, to_node in sorted(pairs_to_process):
            start = centers.get(from_node)
            goal = centers.get(to_node)
            if start is None or goal is None:
                raise KeyError(f"Missing layout entry for nodes {from_node} or {to_node}")
            allow_pair = frozenset({from_node, to_node})
            cache_key = (allow_pair, OBSTACLE_MARGIN)
            obstacles = obstacle_cache.get(cache_key)
            if obstacles is None:
                obstacles = obstacle_points(layout_data, allow_pair, clearance=OBSTACLE_MARGIN)
                obstacle_cache[cache_key] = obstacles
            path = shortest_path(start, goal, obstacles, bounds)
            distance = path_length(path)
            travel_time = distance / speed if speed > 0 else distance
            results.append(
                {
                    "route_index": pair_to_index.get((from_node, to_node)),
                    "transporter_id": transporter_id,
                    "from": from_node,
                    "to": to_node,
                    "path": path,
                    "distance": distance,
                    "travel_time": travel_time,
                    "existing": (from_node, to_node) in pair_to_index,
                }
            )
    return results


def apply_results(config: Dict, results: Iterable[Dict]) -> None:
    routes = config.get("routes", [])
    for entry in results:
        idx = entry.get("route_index")
        if idx is None or idx >= len(routes):
            continue
        path_points = [[float(x), float(y)] for x, y in entry["path"]]
        routes[idx]["path_points"] = path_points
        routes[idx]["travel_time"] = entry["travel_time"]


def visualise(
    layout_data: Dict,
    results: Iterable[Dict],
    transporter_filter: Optional[str] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    filtered_results = [
        entry
        for entry in results
        if not transporter_filter or entry["transporter_id"] == transporter_filter
    ]
    if not filtered_results:
        raise RuntimeError("No routes available for visualisation with the current filter.")
    transporter_order: List[str] = []
    for entry in filtered_results:
        transporter_id = entry["transporter_id"]
        if transporter_id not in transporter_order:
            transporter_order.append(transporter_id)

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
    color_cycle = cycle(palette)
    transporter_colors: Dict[str, str] = {}
    for transporter_id in transporter_order:
        transporter_colors[transporter_id] = next(color_cycle)

    used_nodes: Set[str] = set()
    for entry in filtered_results:
        used_nodes.add(entry["from"])
        used_nodes.add(entry["to"])

    draw_layout(ax, layout_data, used_nodes=used_nodes, highlight_nodes=used_nodes)

    for entry in filtered_results:
        path = entry["path"]
        xs = [pt[0] for pt in path]
        ys = [pt[1] for pt in path]
        route_color = transporter_colors.get(entry["transporter_id"], "#333333")
        existing = entry.get("existing", False)
        label_suffix = "(config)" if existing else "(derived)"
        label = f"{entry['transporter_id']}: {entry['from']}→{entry['to']} {label_suffix}"
        linestyle = "-" if existing else "--"
        alpha = 1.0 if existing else 0.7
        ax.plot(xs, ys, color=route_color, linewidth=2.0, linestyle=linestyle, alpha=alpha, label=label)
        ax.scatter(xs[0], ys[0], color=route_color, s=40, marker="o")
        ax.scatter(xs[-1], ys[-1], color=route_color, s=40, marker="s")

    factory = layout_data.get("factory", {})
    ax.set_xlim(0.0, float(factory.get("length", 50.0)))
    ax.set_ylim(0.0, float(factory.get("width", 30.0)))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Transporter shortest paths")
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute shortest paths for transporter routes")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "simple_four_station.json",
        help="Path to configuration JSON",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Path to layout JSON (defaults to value referenced by config)",
    )
    parser.add_argument(
        "--transporter",
        type=str,
        default=None,
        help="Limit computation/visualisation to a single transporter id",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Persist computed path_points and travel_time back into the config file",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Display a Matplotlib plot of the computed paths",
    )
    parser.add_argument(
        "--existing-only",
        action="store_true",
        help="Restrict computation to routes explicitly listed in the config",
    )
    return parser.parse_args()


def resolve_layout_path(config_path: Path, config: Dict, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    layout_value = config.get("layout")
    if layout_value:
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
    raise FileNotFoundError(
        "Layout path must be provided either via --layout or config['layout'] "
        "and should resolve relative to the config or package directory."
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    layout_path = resolve_layout_path(args.config, config, args.layout)
    layout_data = load_layout_data(layout_path)
    errors = _validate_layout(layout_data, allow_touching=True)
    if errors:
        raise ValueError("布局几何合法性检查失败:\n" + "\n".join(f"- {msg}" for msg in errors))

    results = compute_routes(
        config,
        layout_data,
        transporter_filter=args.transporter,
        include_all_pairs=not args.existing_only,
    )
    if not results:
        raise RuntimeError("No routes matched criteria; nothing to compute")

    def sort_key(entry: Dict) -> Tuple:
        idx = entry.get("route_index")
        idx_key = idx if idx is not None else float("inf")
        return (
            entry["transporter_id"],
            idx_key,
            entry["from"],
            entry["to"],
        )

    results.sort(key=sort_key)
    print("Computed routes:")
    for entry in results:
        node_pair = f"{entry['from']} -> {entry['to']}"
        status = "config" if entry.get("existing") else "derived"
        print(
            f"  [{entry['transporter_id']}] {node_pair:<20} "
            f"distance={entry['distance']:.1f} travel_time={entry['travel_time']:.2f} ({status})"
        )

    if args.write:
        apply_results(config, results)
        args.config.write_text(json.dumps(config, indent=4), encoding="utf-8")
        print(f"Updated {args.config} with path_points and travel_time.")

    if args.visualize:
        visualise(layout_data, results, args.transporter)


if __name__ == "__main__":  # pragma: no cover
    main()
