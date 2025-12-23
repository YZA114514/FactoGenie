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


def get_inventory_chart_data(
    config_path: Path | str,
    duration: float,
    layout_path: Optional[Path | str] = None,
) -> Dict[str, object]:
    """
    运行仿真并获取物料量变化图表数据
    
    Args:
        config_path: 工厂配置文件路径
        duration: 仿真时长
        layout_path: 布局文件路径（可选）
        
    Returns:
        包含时间序列数据的字典，格式：
        {
            "series": [
                {
                    "name": "station_1:leg_assy",
                    "data": [[time1, level1], [time2, level2], ...]
                },
                ...
            ],
            "monitors": [["station_1", "leg_assy"], ...],
            "duration": 2000.0
        }
    """
    config_path = Path(config_path)
    layout = Path(layout_path) if layout_path is not None else None
    config_data, resolved_layout, _ = _prepare_configuration(config_path, layout)
    
    # 获取监控列表
    monitors = _derive_monitors(config_data)
    if not monitors:
        return {"series": [], "monitors": [], "duration": duration}
    
    # 运行仿真
    sim = build_model(config=config_data)
    sim.run(until=duration)
    
    # 收集物料量变化数据
    series_data = _collect_inventory_series(sim.events, monitors)
    
    return {
        "series": series_data,
        "monitors": [[node, material] for node, material in monitors],
        "duration": duration,
    }


def _derive_monitors(config_data: Dict) -> List[Tuple[str, str]]:
    """从配置中提取需要监控的节点和物料"""
    # 如果有明确的 monitor 配置且不为空，则使用它
    if config_data.get("monitor"):  # 检查非空
        monitors = []
        for item in config_data["monitor"]:
            node = item.get("node")
            material = item.get("material")
            if node and material:
                if isinstance(material, list):
                    for mat in material:
                        monitors.append((node, mat))
                else:
                    monitors.append((node, material))
        if monitors:  # 如果解析出了监控点，返回
            return monitors
    
    # 否则从 assemblies 和 summary 自动推导
    monitors = []
    for assembly in config_data.get("assemblies", []):
        node = assembly.get("station")
        material = assembly.get("output")
        if node and material:
            monitors.append((node, material))
    
    summary = config_data.get("summary", {})
    node = summary.get("finished_node")
    material = summary.get("finished_material")
    if isinstance(material, list):
        material = material[0] if material else None
    if node and material:
        monitors.append((node, material))
    
    return monitors


def _collect_inventory_series(events: List[Dict], monitors: List[Tuple[str, str]]) -> List[Dict]:
    """收集物料量变化的时间序列数据"""
    series = {key: [] for key in monitors}
    
    for record in events:
        if record.get("event") in {"inventory_put", "inventory_get"}:
            node = record.get("node")
            material = record.get("material")
            key = (node, material)
            if key in series:
                time = record.get("time", 0)
                level = record.get("level", 0)
                series[key].append([time, level])
    
    # 转换为前端需要的格式
    result = []
    for (node, material), data in series.items():
        if data:  # 只返回有数据的序列
            result.append({
                "name": f"{node}:{material}",
                "data": data,
            })
    
    return result


__all__ = [
    "SimulationResult",
    "run_simulation",
    "compute_metrics",
    "get_inventory_chart_data",
]
