import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment.factory_environment import LayoutEnvironment, METRIC_FIELDS
from simulation.interface import compute_metrics


def load_layout_placements(layout_path: Path) -> dict:
    data = json.loads(layout_path.read_text(encoding="utf-8"))
    placements = {}
    for fu in data.get("fus", []):
        fu_id = str(fu.get("id"))
        if not fu_id:
            continue
        placements[fu_id] = fu
    if not placements:
        raise ValueError(f"No functional units found in {layout_path}")
    return placements


def build_flow_clarity_helper(env: LayoutEnvironment, placements: dict) -> None:
    env.placed_units = []
    for idx, unit in enumerate(env.functional_units):
        unit_id = str(unit.get("id")) if unit.get("id") is not None else unit.get("name", str(idx))
        if unit_id not in placements:
            raise ValueError(f"Layout file missing placement for unit '{unit_id}'")
        placement = placements[unit_id]
        x = int(round(placement.get("x", 0)))
        y = int(round(placement.get("y", 0)))
        rotation = int(round(placement.get("angle", placement.get("angle_deg", 0)))) % 360
        env.placed_units.append((idx, x, y, rotation))


def compute_rewards(summary: dict, env: LayoutEnvironment) -> dict:
    static = summary.get("static", {})
    dynamic = summary.get("dynamic", {})

    avg_distance = float(static.get("average_route_distance", 0))
    total_distance = float(static.get("total_route_distance", 0))
    max_distance = float(static.get("max_route_distance", 0))
    min_distance = float(static.get("min_route_distance", 0))
    total_logistics = float(static.get("total_logistics_intensity", 0))
    space_util = float(static.get("space_utilization", 0))
    factory_area = float(static.get("factory_area", 0))
    occupied_area = float(static.get("occupied_area", 0))
    free_area = float(static.get("free_area", 0))

    finished_goods = float(dynamic.get("finished_goods", 0))
    throughput_rate = float(dynamic.get("throughput_rate", 0))

    station_util_dict = dynamic.get("station_utilization", {}) or {}
    transporter_util_dict = dynamic.get("transporter_utilization", {}) or {}
    summary_node_detail = dynamic.get("summary_node", {}) or {}
    summary_material_detail = dynamic.get("summary_material", {}) or {}

    if station_util_dict:
        station_utils = [v.get("utilization", 0) for v in station_util_dict.values() if isinstance(v, dict)]
        avg_station_util = float(sum(station_utils) / len(station_utils)) if station_utils else 0.0
    else:
        avg_station_util = 0.0

    distance_best, distance_worst = 7.0, 24.0
    distance_range = max(distance_worst - distance_best, 1e-6)
    distance_reward = float(np.clip(-(avg_distance - distance_best) / distance_range, -1.0, 0.0))

    logistics_best, logistics_worst = 240.0, 900.0
    logistics_range = max(logistics_worst - logistics_best, 1e-6)
    logistics_reward = float(
        np.clip(-(total_logistics - logistics_best) / logistics_range, -1.0, 0.0)
    )

    flow_clarity_reward = float(np.clip(env._calculate_flow_clarity_reward(), -1.0, 0.0))

    if finished_goods < 200:
        throughput_reward = -1.0
    elif finished_goods < 400:
        throughput_reward = float((finished_goods - 400) / 200.0)
    else:
        throughput_reward = float(min((finished_goods - 400) / 800.0, 0.0))

    if avg_station_util < 0.001:
        utilization_reward = -1.0
    elif avg_station_util < 0.05:
        utilization_reward = float((avg_station_util - 0.05) / 0.05)
    else:
        utilization_reward = float(min(-0.05 / avg_station_util + 1.0, 0.0))
    utilization_reward = float(np.clip(utilization_reward, -1.0, 0.0))

    weights = env.objective_weights
    weight_components = {
        "distance": weights.get("transportation_intensity", 0.25),
        "logistics": weights.get("material_flow_clarity", 0.35),
        "flow": weights.get("flow_clarity", weights.get("space_utilization", 0.15)),
        "throughput": weights.get("throughput_time", 0.20),
        "utilization": weights.get("utilization", 0.05),
    }
    total_weight = sum(weight_components.values()) or 1.0
    final_reward = float(
        (
            weight_components["distance"] * distance_reward
            + weight_components["logistics"] * logistics_reward
            + weight_components["flow"] * flow_clarity_reward
            + weight_components["throughput"] * throughput_reward
            + weight_components["utilization"] * utilization_reward
        )
        / total_weight
    )
    final_reward = float(np.clip(final_reward, -1.0, 0.0))

    return {
        "avg_distance": avg_distance,
        "total_distance": total_distance,
        "max_distance": max_distance,
        "min_distance": min_distance,
        "total_logistics": total_logistics,
        "space_util": space_util,
        "factory_area": factory_area,
        "occupied_area": occupied_area,
        "free_area": free_area,
        "finished_goods": finished_goods,
        "throughput_rate": throughput_rate,
        "avg_station_util": avg_station_util,
        "distance_reward": distance_reward,
        "logistics_reward": logistics_reward,
        "flow_clarity_reward": flow_clarity_reward,
        "throughput_reward": throughput_reward,
        "utilization_reward": utilization_reward,
        "final_reward": final_reward,
        "station_util_detail": station_util_dict,
        "transporter_util_detail": transporter_util_dict,
        "summary_node_detail": summary_node_detail,
        "summary_material_detail": summary_material_detail,
        "static_json": static,
        "dynamic_json": dynamic,
    }


def save_csv(row: dict, output_path: Path) -> None:
    fieldnames = ["episode", "timestamp", *METRIC_FIELDS]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import csv

    file_exists = output_path.exists()
    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a fixed layout via simulation")
    parser.add_argument("--layout", required=True, type=Path, help="Path to layout JSON to evaluate")
    parser.add_argument(
        "--config",
        default=Path("simulation/configs/chair_factory.json"),
        type=Path,
        help="Simulation config file",
    )
    parser.add_argument("--duration", type=float, default=28800, help="Simulation duration")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis/SLP_layout_metrics.csv"),
        help="CSV to append evaluation result",
    )
    args = parser.parse_args()

    layout_path = args.layout.resolve()
    config_path = args.config.resolve()

    placements = load_layout_placements(layout_path)

    env = LayoutEnvironment.from_config(str(config_path), use_simulation=True, simulation_duration=args.duration)
    env.reset()
    build_flow_clarity_helper(env, placements)

    metrics_result = compute_metrics(
        config_path=config_path,
        duration=args.duration,
        layout_path=layout_path,
        detail=True,
    )
    rewards = compute_rewards(metrics_result["summary"], env)

    row = {field: None for field in METRIC_FIELDS}
    row.update(
        {
            "average_route_distance": rewards["avg_distance"],
            "total_route_distance": rewards["total_distance"],
            "max_route_distance": rewards["max_distance"],
            "min_route_distance": rewards["min_distance"],
            "total_logistics_intensity": rewards["total_logistics"],
            "space_utilization": rewards["space_util"],
            "factory_area": rewards["factory_area"],
            "occupied_area": rewards["occupied_area"],
            "free_area": rewards["free_area"],
            "finished_goods": rewards["finished_goods"],
            "throughput_rate": rewards["throughput_rate"],
            "avg_station_utilization": rewards["avg_station_util"],
            "distance_reward": rewards["distance_reward"],
            "logistics_reward": rewards["logistics_reward"],
            "flow_clarity_reward": rewards["flow_clarity_reward"],
            "throughput_reward": rewards["throughput_reward"],
            "utilization_reward": rewards["utilization_reward"],
            "final_reward": rewards["final_reward"],
            "placed_units": env.num_units,
            "total_units": env.num_units,
            "use_simulation": True,
            "early_termination": False,
            "error": "",
            "station_utilization_detail": json.dumps(rewards["station_util_detail"], ensure_ascii=False),
            "transporter_utilization_detail": json.dumps(rewards["transporter_util_detail"], ensure_ascii=False),
            "summary_node_detail": json.dumps(rewards["summary_node_detail"], ensure_ascii=False),
            "summary_material_detail": json.dumps(rewards["summary_material_detail"], ensure_ascii=False),
            "static_summary_json": json.dumps(rewards["static_json"], ensure_ascii=False),
            "dynamic_summary_json": json.dumps(rewards["dynamic_json"], ensure_ascii=False),
        }
    )

    row["episode"] = 1
    row["timestamp"] = datetime.now().isoformat()

    save_csv(row, args.output)

    print("==== SLP Layout Evaluation ====")
    for key in [
        "avg_distance",
        "total_logistics",
        "finished_goods",
        "throughput_rate",
        "avg_station_util",
        "distance_reward",
        "logistics_reward",
        "flow_clarity_reward",
        "throughput_reward",
        "utilization_reward",
        "final_reward",
    ]:
        print(f"{key}: {rewards[key]:.6f}")
    print(f"Metrics saved to {args.output}")


if __name__ == "__main__":
    main()
