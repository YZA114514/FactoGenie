"""
结果相关数据模型
"""
from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime


class Placement(BaseModel):
    x: float
    y: float
    angle: int


class LayoutMetrics(BaseModel):
    distance_score: float
    logistics_score: float
    throughput: float
    utilization: float
    flow_clarity: float
    finished_goods: Optional[int] = None


class LayoutResult(BaseModel):
    episode: int
    timestamp: datetime
    reward: float
    placements: Dict[str, Placement]
    metrics: LayoutMetrics


class ActionHeatmap(BaseModel):
    step: int
    unit_id: str
    unit_label: str
    grid_width: int
    grid_height: int
    angle_options: List[int] = [0, 90, 180, 270]
    q_values: List[List[List[float]]]  # [angle][y][x]
    selected_action: Dict  # {x, y, angle, q_value}


class MetricPoint(BaseModel):
    episode: int
    value: float


class MetricsTimeSeries(BaseModel):
    metric: str
    values: List[MetricPoint]

