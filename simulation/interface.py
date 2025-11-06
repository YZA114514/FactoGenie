from __future__ import annotations

"""对外提供的高层接口，便于脚本或外部系统集成。"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .metrics import (
    compute_dynamic_metrics,
    compute_static_metrics,
    summarize_metrics,
)
from .model import build_model, load_config
from .planning import (
    compute_route_plans,
    load_layout,
    load_layout_data,
    resolve_layout_path,
)
from .run_simulation import determine_summary


@dataclass
class SimulationResult:
    summary_node: str
    summary_material: Optional[str]
    finished_goods: float
    duration: float
    config: Dict
    layout_path: Optional[Path]
    events: Optional[List[Dict[str, object]]] = None


def _prepare_configuration(
    config_path: Path,
    layout_path: Optional[Path],
) -> Tuple[Dict, Optional[Path], Optional[Dict]]:
    config_data = load_config(config_path)
    resolved_layout = resolve_layout_path(config_path, config_data, layout_path)
    layout_data = None
    if resolved_layout and resolved_layout.exists():
        positions = load_layout(resolved_layout)
        layout_data = load_layout_data(resolved_layout)
        compute_route_plans(config_data, positions, layout_data)
    else:
        compute_route_plans(config_data, {})
    return config_data, resolved_layout, layout_data


def run_simulation(
    config_path: Path | str,
    duration: float,
    layout_path: Optional[Path | str] = None,
    include_events: bool = False,
) -> SimulationResult:
    """运行仿真，返回核心指标，可选带上事件日志。"""

    config_path = Path(config_path)
    layout = Path(layout_path) if layout_path is not None else None
    config_data, resolved_layout, _ = _prepare_configuration(config_path, layout)

    sim = build_model(config=config_data)
    sim.run(until=duration)

    summary_node, summary_material = determine_summary(config_data, sim)
    snapshot = sim.store_snapshot(summary_node)
    finished = snapshot.get(summary_material, 0.0) if summary_material else 0.0
    events = list(sim.events) if include_events else None

    return SimulationResult(
        summary_node=summary_node,
        summary_material=summary_material,
        finished_goods=float(finished),
        duration=float(duration),
        config=config_data,
        layout_path=resolved_layout,
        events=events,
    )


def compute_metrics(
    config_path: Path | str,
    duration: float,
    layout_path: Optional[Path | str] = None,
    detail: bool = False,
) -> Dict[str, object]:
    """基于配置和仿真结果计算指标，输出结构化结果。"""

    config_path = Path(config_path)
    layout = Path(layout_path) if layout_path is not None else None
    config_data, resolved_layout, layout_data = _prepare_configuration(config_path, layout)

    static_metrics = compute_static_metrics(config_data, layout_data)
    sim = build_model(config=config_data)
    sim.run(until=duration)
    dynamic_metrics = compute_dynamic_metrics(config_data, sim, duration)
    summary = summarize_metrics(static_metrics, dynamic_metrics)

    if detail:
        return {
            "summary": summary,
            "details": {
                "static_metrics": static_metrics,
                "dynamic_metrics": dynamic_metrics,
            },
        }
    return {"summary": summary}


__all__ = [
    "SimulationResult",
    "run_simulation",
    "compute_metrics",
]
