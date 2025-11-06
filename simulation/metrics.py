from __future__ import annotations

"""评估生产与布局效率的综合指标工具。"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover
    from .model import build_model, load_config
    from .planning import (
        load_layout,
        load_layout_data,
        compute_route_plans,
        resolve_layout_path,
    )
    from .run_simulation import determine_summary
except ImportError:  # pragma: no cover
    import sys

    base = Path(__file__).resolve().parent
    sys.path.append(str(base))
    from model import build_model, load_config  # type: ignore
    from planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path  # type: ignore


def path_length(points: Iterable[Iterable[float]]) -> float:
    pts = list(points or [])
    if len(pts) < 2:
        return 0.0
    total = 0.0
    for i in range(1, len(pts)):
        x0, y0 = pts[i - 1]
        x1, y1 = pts[i]
        dx = float(x1) - float(x0)
        dy = float(y1) - float(y0)
        total += math.hypot(dx, dy)
    return total


def compute_static_metrics(config: Dict, layout_data: Optional[Dict]) -> Dict[str, object]:
    routes = config.get("routes", [])
    distances: List[Dict[str, object]] = []
    total_distance = 0.0
    logistic_total = 0.0

    for route in routes:
        path = route.get("path_points")
        if not path:
            continue
        distance = path_length(path)
        total_distance += distance
        batch = float(route.get("batch_size", 0.0))
        logistic = distance * batch
        logistic_total += logistic
        distances.append(
            {
                "from": route.get("from"),
                "to": route.get("to"),
                "material": route.get("material"),
                "transporter": route.get("transporter_id"),
                "distance": distance,
                "batch_size": batch,
                "logistics_intensity": logistic,
            }
        )

    avg_distance = total_distance / len(distances) if distances else 0.0
    max_distance = max((item["distance"] for item in distances), default=0.0)
    min_distance = min((item["distance"] for item in distances), default=0.0)

    space_metrics = {}
    if layout_data:
        factory = layout_data.get("factory", {})
        factory_area = float(factory.get("length", 0.0)) * float(factory.get("width", 0.0))
        fu_area = 0.0
        for fu in layout_data.get("fus", []):
            fu_area += float(fu.get("length", 0.0)) * float(fu.get("width", 0.0))
        utilization = fu_area / factory_area if factory_area > 0 else 0.0
        space_metrics = {
            "factory_area": factory_area,
            "fu_area": fu_area,
            "space_utilization": utilization,
            "free_area": max(factory_area - fu_area, 0.0),
        }

    return {
        "transport_distance": {
            "per_route": distances,
            "total_distance": total_distance,
            "average_distance": avg_distance,
            "max_distance": max_distance,
            "min_distance": min_distance,
        },
        "logistics_intensity": {
            "total_intensity": logistic_total,
            "per_route": distances,
        },
        "space_utilization": space_metrics,
    }


def compute_dynamic_metrics(
    config: Dict,
    sim,
    duration: float,
) -> Dict[str, object]:
    events = sim.events
    duration = float(duration)
    summary_node, summary_material = determine_summary(config, sim)
    snapshot = sim.store_snapshot(summary_node)
    finished = snapshot.get(summary_material, 0.0) if summary_material else 0.0
    throughput_rate = finished / duration if duration > 0 else 0.0

    # Station utilization
    station_active: Dict[str, float] = defaultdict(float)
    station_last_start: Dict[str, float] = {}

    # Transporter metrics
    transporter_config = {
        entry.get("id"): int(entry.get("count", 1))
        for entry in config.get("transporters", [])
        if entry.get("id")
    }

    path_lookup: Dict[str, Dict[Tuple[str, str], Dict[str, object]]] = config.get("_transporter_paths", {})

    def lookup_path(transporter: Optional[str], from_node: Optional[str], to_node: Optional[str]):
        if not transporter or not from_node or not to_node:
            return None
        entry = path_lookup.get(transporter, {}).get((from_node, to_node))
        if entry:
            return entry.get("path_points")
        return None

    active_moves: Dict[Tuple[str, str], Dict[str, object]] = {}
    transporter_stats: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for event in events:
        etype = event.get("event")
        t = float(event.get("time", 0.0))

        if etype == "assembly_start":
            station = str(event.get("station"))
            station_last_start[station] = t
        elif etype == "assembly_finish":
            station = str(event.get("station"))
            start = station_last_start.pop(station, t)
            station_active[station] += max(0.0, t - start)
        elif etype == "transport_depart":
            transporter = str(event.get("transporter")) if event.get("transporter") else None
            vehicle = str(event.get("vehicle")) if event.get("vehicle") else None
            key = (transporter or "", vehicle or "")
            active_moves[key] = {
                "start": t,
                "from": event.get("from_node"),
                "to": event.get("to_node"),
                "quantity": float(event.get("quantity", 0.0)),
                "transporter": transporter,
            }
        elif etype == "transport_arrive":
            transporter = str(event.get("transporter")) if event.get("transporter") else None
            vehicle = str(event.get("vehicle")) if event.get("vehicle") else None
            key = (transporter or "", vehicle or "")
            data = active_moves.pop(key, None)
            if data is None:
                continue
            start = float(data.get("start", t))
            travel_time = max(0.0, t - start)
            quantity = float(data.get("quantity", 0.0))
            from_node = data.get("from")
            to_node = data.get("to")
            path = lookup_path(transporter, from_node, to_node) or event.get("path")
            distance = path_length(path) if path else travel_time  # fallback
            stats = transporter_stats[transporter or "unknown"]
            stats["loaded_time"] += travel_time
            stats["loaded_distance"] += distance
            stats["quantity"] += quantity
            stats["trips"] += 1
            stats["ton_km"] += distance * quantity
        elif etype == "transport_reposition":
            transporter = str(event.get("transporter")) if event.get("transporter") else None
            travel_time = float(event.get("travel_time", 0.0))
            path = event.get("path")
            if not path and transporter:
                path = lookup_path(transporter, event.get("from_node"), event.get("to_node"))
            distance = path_length(path) if path else travel_time
            stats = transporter_stats[transporter or "unknown"]
            stats["reposition_time"] += travel_time
            stats["reposition_distance"] += distance

    station_utilization = []
    for station, busy_time in station_active.items():
        utilization = busy_time / duration if duration > 0 else 0.0
        station_utilization.append(
            {"station": station, "busy_time": busy_time, "utilization": utilization}
        )
    station_utilization.sort(key=lambda x: x["station"])

    transporter_metrics = []
    for transporter_id, stats in transporter_stats.items():
        vehicle_count = transporter_config.get(transporter_id, 1)
        total_time = (stats["loaded_time"] + stats["reposition_time"])
        capability = duration * vehicle_count if duration > 0 else 0.0
        utilization = total_time / capability if capability > 0 else 0.0
        trips = stats.get("trips", 0.0)
        avg_load = stats["quantity"] / trips if trips else 0.0
        transporter_metrics.append(
            {
                "transporter": transporter_id,
                "vehicles": vehicle_count,
                "utilization": utilization,
                "loaded_time": stats["loaded_time"],
                "reposition_time": stats["reposition_time"],
                "loaded_distance": stats["loaded_distance"],
                "reposition_distance": stats["reposition_distance"],
                "distance_total": stats["loaded_distance"] + stats["reposition_distance"],
                "quantity_moved": stats["quantity"],
                "trips": trips,
                "average_load": avg_load,
                "ton_km": stats["ton_km"],
            }
        )
    transporter_metrics.sort(key=lambda x: x["transporter"])

    return {
        "throughput": {
            "finished_goods": finished,
            "throughput_rate": throughput_rate,
            "summary_node": summary_node,
            "summary_material": summary_material,
        },
        "station_utilization": station_utilization,
        "transporter_metrics": transporter_metrics,
    }


def summarize_metrics(static_metrics: Dict[str, object], dynamic_metrics: Dict[str, object]) -> Dict[str, object]:
    td = static_metrics.get("transport_distance", {})
    su = static_metrics.get("space_utilization", {})

    station_summary = {
        item["station"]: {
            "utilization": item["utilization"],
            "busy_time": item["busy_time"],
        }
        for item in dynamic_metrics.get("station_utilization", [])
    }

    transporter_summary = {
        item["transporter"]: {
            "utilization": item["utilization"],
            "distance_total": item["distance_total"],
            "average_load": item["average_load"],
            "trips": item["trips"],
            "quantity_moved": item["quantity_moved"],
        }
        for item in dynamic_metrics.get("transporter_metrics", [])
    }

    throughput = dynamic_metrics.get("throughput", {})

    return {
        "static": {
            "total_route_distance": td.get("total_distance", 0.0),
            "average_route_distance": td.get("average_distance", 0.0),
            "max_route_distance": td.get("max_distance", 0.0),
            "min_route_distance": td.get("min_distance", 0.0),
            "total_logistics_intensity": static_metrics.get("logistics_intensity", {}).get("total_intensity", 0.0),
            "space_utilization": su.get("space_utilization"),
            "factory_area": su.get("factory_area"),
            "occupied_area": su.get("fu_area"),
            "free_area": su.get("free_area"),
        },
        "dynamic": {
            "finished_goods": throughput.get("finished_goods", 0.0),
            "throughput_rate": throughput.get("throughput_rate", 0.0),
            "summary_node": throughput.get("summary_node"),
            "summary_material": throughput.get("summary_material"),
            "station_utilization": station_summary,
            "transporter_utilization": transporter_summary,
        },
    }


def print_summary(summary: Dict[str, object]) -> None:
    static = summary.get("static", {})
    dynamic = summary.get("dynamic", {})
    print("== 静态指标 ==")
    print(f"总路线距离: {static.get('total_route_distance', 0.0):.2f}")
    print(f"平均路线距离: {static.get('average_route_distance', 0.0):.2f}")
    print(f"最大路线距离: {static.get('max_route_distance', 0.0):.2f}")
    print(f"最小路线距离: {static.get('min_route_distance', 0.0):.2f}")
    print(f"总物流强度: {static.get('total_logistics_intensity', 0.0):.2f}")
    space_util = static.get("space_utilization")
    if space_util is not None:
        print(
            f"空间利用率: {space_util:.2%} "
            f"(工厂面积={static.get('factory_area', 0.0):.2f}, "
            f"工位面积={static.get('occupied_area', 0.0):.2f})"
        )

    print("\n== 动态指标 ==")
    print(
        f"产出(节点 {dynamic.get('summary_node')}): {dynamic.get('finished_goods', 0.0)} 件，"
        f"吞吐率 {dynamic.get('throughput_rate', 0.0):.4f}/时间单位"
    )
    stations = dynamic.get("station_utilization", {})
    if stations:
        print("工位利用率：")
        for station, data in stations.items():
            print(f"  - {station}: 利用率 {data.get('utilization', 0.0):.2%}, 忙碌 {data.get('busy_time', 0.0):.2f}")
    transporters = dynamic.get("transporter_utilization", {})
    if transporters:
        print("运输工具：")
        for transporter, data in transporters.items():
            print(
                f"  - {transporter}: 利用率 {data.get('utilization', 0.0):.2%}, "
                f"总距 {data.get('distance_total', 0.0):.2f}, "
                f"趟次 {data.get('trips', 0.0):.0f}, "
                f"平均载荷 {data.get('average_load', 0.0):.2f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生产与布局效率指标分析")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "simple_four_station.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="布局 JSON 路径（可覆盖配置中的 layout 字段）",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=200.0,
        help="仿真时长",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出结果",
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="输出详细分项数据（JSON 模式下包含 routes / 原始指标列表）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    layout_path = resolve_layout_path(args.config, config_data, args.layout)
    layout_data = load_layout_data(layout_path) if layout_path and layout_path.exists() else None
    positions = load_layout(layout_path) if layout_path and layout_path.exists() else {}
    if layout_data:
        compute_route_plans(config_data, positions, layout_data)
    else:
        compute_route_plans(config_data, positions)

    static_metrics = compute_static_metrics(config_data, layout_data)
    sim = build_model(config=config_data)
    sim.run(until=args.duration)
    dynamic_metrics = compute_dynamic_metrics(config_data, sim, args.duration)

    summary = summarize_metrics(static_metrics, dynamic_metrics)
    if args.json:
        result = {"summary": summary}
        if args.detail:
            result["details"] = {
                "static_metrics": static_metrics,
                "dynamic_metrics": dynamic_metrics,
            }
        print(json.dumps(result, indent=4, ensure_ascii=False))
    else:
        print_summary(summary)
        if args.detail:
            print("\n(完整明细请使用 --json --detail 查看)")


if __name__ == "__main__":  # pragma: no cover
    main()

__all__ = [
    "compute_static_metrics",
    "compute_dynamic_metrics",
    "summarize_metrics",
    "print_summary",
    "main",
]
