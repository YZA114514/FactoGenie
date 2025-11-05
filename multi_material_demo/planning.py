from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from .path_planner import compute_routes
    from .layout_validation import validate_layout_data as _validate_layout
except ImportError:  # pragma: no cover
    import sys

    base = Path(__file__).resolve().parent
    sys.path.append(str(base))
    from path_planner import compute_routes  # type: ignore
    from layout_validation import validate_layout_data as _validate_layout  # type: ignore


def load_layout_data(layout_path: Path) -> Dict:
    import json

    return json.load(layout_path.open('r', encoding='utf-8-sig'))


def resolve_layout_path(
    config_path: Path,
    config_data: Dict,
    override: Optional[Path] = None,
) -> Optional[Path]:
    if override is not None:
        return override
    layout_value = config_data.get("layout")
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

def load_layout(layout_path: Path) -> Dict[str, Tuple[float, float]]:
    data = load_layout_data(layout_path)
    positions: Dict[str, Tuple[float, float]] = {}
    for item in data.get('fus', []):
        x = float(item.get('x', 0.0))
        y = float(item.get('y', 0.0))
        length = float(item.get('length', 0.0))
        width = float(item.get('width', 0.0))
        label = str(item.get('id'))
        if label:
            positions[label] = (x + length / 2.0, y + width / 2.0)
    return positions


def compute_route_plans(
    config_data: Dict,
    layout_positions: Dict[str, Tuple[float, float]],
    layout_data: Optional[Dict] = None,
) -> None:
    """Populate routes with travel_time and path_points using layout information."""

    routes = config_data.get('routes', [])
    if not routes:
        return

    # Prefer the detailed path planner when full layout data is available.
    if layout_data:
        errors = _validate_layout(layout_data)
        if errors:
            raise ValueError("布局几何合法性检查失败:\n" + "\n".join(f"- {msg}" for msg in errors))
        results = compute_routes(
            config=config_data,
            layout_data=layout_data,
            transporter_filter=None,
            include_all_pairs=True,
        )
        transporter_paths: Dict[str, Dict[Tuple[str, str], Dict[str, object]]] = {}
        for entry in results:
            idx = entry.get("route_index")
            if idx is None or idx >= len(routes):
                # store derived pairs for repositioning even if not part of config
                pair_key = (entry['from'], entry['to'])
            else:
                pair_key = (entry['from'], entry['to'])
                routes[idx]['path_points'] = [[float(x), float(y)] for x, y in entry['path']]
                routes[idx]['travel_time'] = float(entry['travel_time'])
            tp_entry = transporter_paths.setdefault(entry['transporter_id'], {})
            tp_entry[pair_key] = {
                'travel_time': float(entry['travel_time']),
                'path_points': [[float(x), float(y)] for x, y in entry['path']],
            }
        config_data['_transporter_paths'] = transporter_paths
        return

    speeds = {
        str(item.get('id')): float(item.get('speed', 1.0))
        for item in config_data.get('transporters', [])
        if item.get('id')
    }
    default_speed = float(config_data.get('default_speed', 1.0))

    for route in config_data.get('routes', []):
        from_node = str(route.get('from'))
        to_node = str(route.get('to'))
        start = layout_positions.get(from_node)
        end = layout_positions.get(to_node)
        if start is None or end is None:
            continue
        distance = math.dist(start, end)
        transporter_id = route.get('transporter_id')
        speed = speeds.get(transporter_id, default_speed)
        if speed <= 0:
            speed = default_speed or 1.0
        if not route.get('travel_time'):
            route['travel_time'] = distance / speed if speed > 0 else distance
        route['path_points'] = [list(start), list(end)]
