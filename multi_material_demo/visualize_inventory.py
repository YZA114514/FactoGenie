"""Plot inventory levels over time for an assembly scenario."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

try:
    from .model import build_model, load_config
    from .planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path
except ImportError:  # pragma: no cover
    import sys

    base = Path(__file__).resolve().parent
    sys.path.append(str(base))
    from model import build_model, load_config  # type: ignore
    from planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path  # type: ignore


def collect(events, tracked):
    series = {key: ([], []) for key in tracked}
    for record in events:
        if record.get("event") in {"inventory_put", "inventory_get"}:
            node = record.get("node")
            material = record.get("material")
            key = (node, material)
            if key in series:
                times, levels = series[key]
                times.append(record["time"])
                levels.append(record["level"])
    return series


def derive_monitors(config_data):
    if "monitor" in config_data:
        return [
            (item.get("node"), item.get("material"))
            for item in config_data["monitor"]
            if item.get("node") and item.get("material")
        ]

    monitors = []
    for assembly in config_data.get("assemblies", []):
        node = assembly.get("station")
        material = assembly.get("output")
        if node and material:
            monitors.append((node, material))
    summary = config_data.get("summary", {})
    node = summary.get("finished_node")
    material = summary.get("finished_material")
    if node and material:
        monitors.append((node, material))
    return monitors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot inventory levels over time")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "chair_factory.json",
        help="Path to JSON configuration file",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="Path to layout JSON file",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=25000.0,
        help="Simulation horizon",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    layout_path = resolve_layout_path(args.config, config_data, args.layout)
    if layout_path and layout_path.exists():
        positions = load_layout(layout_path)
        layout_data = load_layout_data(layout_path)
        compute_route_plans(config_data, positions, layout_data)
    tracked = derive_monitors(config_data)
    if not tracked:
        raise ValueError("No monitor definitions found in configuration")

    sim = build_model(config=config_data)
    sim.run(until=args.duration)
    data = collect(sim.events, tracked)

    fig, ax = plt.subplots(figsize=(10, 6))
    for (node, material), (times, levels) in data.items():
        if times:
            label = f"{node}:{material}"
            ax.step(times, levels, where="post", label=label)

    ax.set_xlabel("Time")
    ax.set_ylabel("Inventory level")
    ax.set_title("Inventory levels over time")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    plt.show()


if __name__ == "__main__":
    main()
