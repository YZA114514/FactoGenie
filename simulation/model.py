from __future__ import annotations

"""Generic loader for configuration-driven assembly simulations."""

import json
from pathlib import Path
from typing import Dict, Optional

try:
    from .engine import AssemblyConfig, AssemblySim, RouteConfig, TransporterConfig
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parent))
    from engine import AssemblyConfig, AssemblySim, RouteConfig, TransporterConfig  # type: ignore


def load_config(config_path: Path) -> Dict:
    """Load a JSON configuration, tolerating optional UTF-8 BOM."""

    with config_path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def build_model(
    config_path: Optional[Path] = None,
    config: Optional[Dict] = None,
    run_duration: float = 200.0,
) -> AssemblySim:
    """Construct an assembly simulation from a JSON configuration."""

    if config is None:
        if config_path is None:
            config_path = Path(__file__).resolve().parent / "configs" / "chair_factory.json"
        config = load_config(Path(config_path))

    initial_inventory: Dict[str, Dict[str, float]] = config.get("initial_inventory", {})
    routes_cfg = config.get("routes", [])
    assemblies_cfg = config.get("assemblies", [])
    transporters_cfg = config.get("transporters", [])

    routes = []
    for item in routes_cfg:
        # 处理 material 可能是列表或字符串的情况
        material_raw = item["material"]
        if isinstance(material_raw, list):
            # 如果是列表，为每种物料创建单独的路由
            materials = material_raw
        else:
            materials = [material_raw]
        
        for mat in materials:
            routes.append(RouteConfig(
                from_node=item["from"],
                to_node=item["to"],
                material=mat,
                batch_size=float(item.get("batch_size", 0.0)),
                travel_time=float(item.get("travel_time", 0.0)),
                transporter_id=item.get("transporter_id"),
                path_points=[tuple(pt) for pt in item.get("path_points", [])] if item.get("path_points") else None,
            ))

    assemblies = [
        AssemblyConfig(
            station=item["station"],
            inputs={str(mat): float(qty) for mat, qty in item["inputs"].items()},
            output=item["output"],
            output_quantity=float(item.get("output_quantity", 1.0)),
            process_time=float(item.get("process_time", 1.0)),
        )
        for item in assemblies_cfg
    ]

    transporters = [
        TransporterConfig(
            transporter_id=item.get("id"),
            count=int(item.get("count", 1)),
        )
        for item in transporters_cfg
        if item.get("id")
    ]

    return AssemblySim(
        sources=[],
        routes=routes,
        assemblies=assemblies,
        transporters=transporters,
        initial_inventory=initial_inventory,
        transporter_paths=config.get("_transporter_paths"),
    )
