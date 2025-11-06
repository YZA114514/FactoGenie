from __future__ import annotations

"""BOM 分析与报表工具。"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:  # pragma: no cover - 包内调用
    from .model import build_model, load_config
    from .planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path
except ImportError:  # pragma: no cover - 直接运行
    import sys

    base = Path(__file__).resolve().parent
    sys.path.append(str(base))
    from model import build_model, load_config  # type: ignore
    from planning import load_layout, load_layout_data, compute_route_plans, resolve_layout_path  # type: ignore


class BOMReport:
    """汇总物料消耗、产出与运输量。"""

    def __init__(self) -> None:
        self.consumption: Dict[str, float] = defaultdict(float)
        self.production: Dict[str, float] = defaultdict(float)
        self.routes: Dict[Tuple[str, str, str], float] = defaultdict(float)

    def add_consumption(self, material: str, quantity: float) -> None:
        self.consumption[material] += float(quantity)

    def add_production(self, material: str, quantity: float) -> None:
        self.production[material] += float(quantity)

    def add_route(self, from_node: str, to_node: str, material: str, quantity: float) -> None:
        self.routes[(from_node, to_node, material)] += float(quantity)

    def balance(self) -> Dict[str, float]:
        all_materials = set(self.consumption) | set(self.production)
        return {
            mat: self.production.get(mat, 0.0) - self.consumption.get(mat, 0.0)
            for mat in sorted(all_materials)
        }

    def __str__(self) -> str:
        lines = ["BOM Report"]
        lines.append("Production:")
        for mat, qty in sorted(self.production.items()):
            lines.append(f"  {mat}: {qty:.2f}")
        lines.append("Consumption:")
        for mat, qty in sorted(self.consumption.items()):
            lines.append(f"  {mat}: {qty:.2f}")
        lines.append("Balance (production - consumption):")
        for mat, qty in self.balance().items():
            lines.append(f"  {mat}: {qty:.2f}")
        lines.append("Transport totals:")
        for (src, dst, mat), qty in sorted(self.routes.items()):
            lines.append(f"  {src}->{dst} {mat}: {qty:.2f}")
        return "\n".join(lines)


EVENT_ASSEMBLY_START = "assembly_start"
EVENT_ASSEMBLY_FINISH = "assembly_finish"
EVENT_TRANSPORT_DEPART = "transport_depart"


def analyse_events(events: Iterable[Dict[str, object]]) -> BOMReport:
    report = BOMReport()
    for record in events:
        event_type = record.get("event")
        if event_type == EVENT_ASSEMBLY_START:
            inputs = record.get("inputs", {})
            if isinstance(inputs, dict):
                for mat, qty in inputs.items():
                    report.add_consumption(str(mat), float(qty))
        elif event_type == EVENT_ASSEMBLY_FINISH:
            output = record.get("output")
            if output is not None:
                qty = float(record.get("quantity", 1.0))
                report.add_production(str(output), qty)
        elif event_type == EVENT_TRANSPORT_DEPART:
            material = record.get("material")
            if material is not None:
                qty = float(record.get("quantity", 0.0))
                from_node = str(record.get("from_node"))
                to_node = str(record.get("to_node"))
                report.add_route(from_node, to_node, str(material), qty)
    return report


def print_summary(report: BOMReport) -> None:
    print(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BOM reconciliation")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "configs" / "chair_factory.json",
        help="配置文件路径",
    )
    parser.add_argument(
        "--layout",
        type=Path,
        default=None,
        help="布局 JSON 路径，可覆盖配置内 layout 字段",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=1200.0,
        help="仿真时间长度",
    )
    return parser.parse_args()


def generate_report(config_path: Path, layout_path: Optional[Path], duration: float) -> BOMReport:
    config_data = load_config(config_path)
    actual_layout = resolve_layout_path(config_path, config_data, layout_path)
    if actual_layout and actual_layout.exists():
        positions = load_layout(actual_layout)
        layout_data = load_layout_data(actual_layout)
        compute_route_plans(config_data, positions, layout_data)
    sim = build_model(config=config_data)
    sim.run(until=duration)
    return analyse_events(sim.events)


def main() -> None:
    args = parse_args()
    report = generate_report(args.config, args.layout, args.duration)
    print_summary(report)


__all__ = [
    "BOMReport",
    "analyse_events",
    "print_summary",
    "generate_report",
]


if __name__ == "__main__":  # pragma: no cover
    main()
