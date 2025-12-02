"""Run an assembly simulation using a JSON configuration."""

import argparse
from pathlib import Path

try:
    from .model import build_model, load_config
    from .planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path
except ImportError:  # pragma: no cover - direct execution fallback
    import sys

    base = Path(__file__).resolve().parent
    sys.path.append(str(base))
    from model import build_model, load_config  # type: ignore
    from planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path  # type: ignore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-material assembly simulation")
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
        default=1000.0,
        help="Simulation horizon",
    )
    return parser.parse_args()


def determine_summary(config_data, sim):
    summary = config_data.get("summary", {})
    node = summary.get("finished_node")
    material = summary.get("finished_material")
    if not node:
        node = "ship_dock" if "ship_dock" in sim._stores else "ship"
    if not material:
        # fall back to the first assembly output or any material present at the node
        outputs = [assembly.get("output") for assembly in config_data.get("assemblies", [])]
        material = outputs[-1] if outputs else None
    return node, material


def main() -> None:
    args = parse_args()
    config_data = load_config(args.config)
    layout_path = resolve_layout_path(args.config, config_data, args.layout)
    positions = None
    if layout_path and layout_path.exists():
        positions = load_layout(layout_path)
        layout_data = load_layout_data(layout_path)
        compute_route_plans(config_data, positions, layout_data)
    sim = build_model(config=config_data)
    sim.run(until=args.duration)

    node, material = determine_summary(config_data, sim)
    snapshot = sim.store_snapshot(node)
    finished = snapshot.get(material, 0.0)
    print(f"Simulation finished. Finished goods at {node}: {finished}")


if __name__ == "__main__":
    main()
