"""
工厂配置相关数据模型
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional


class Route(BaseModel):
    from_: str = Field(..., alias="from")
    to: str
    material: str
    batch_size: int
    travel_time: float = 0
    transporter_id: str

    class Config:
        populate_by_name = True


class Assembly(BaseModel):
    station: str
    inputs: Dict[str, int]
    output: str
    process_time: float


class MonitorItem(BaseModel):
    node: str
    material: str


class Transporter(BaseModel):
    id: str
    count: int
    speed: float


class Summary(BaseModel):
    finished_node: str
    finished_material: str


class FactoryConfig(BaseModel):
    initial_inventory: Dict[str, Dict[str, int]]
    routes: List[Route]
    assemblies: List[Assembly]
    summary: Summary
    monitor: List[MonitorItem]
    transporters: List[Transporter]
    layout: str


class FunctionalUnit(BaseModel):
    id: str
    label: str
    width: float
    height: float
    movable: bool = True
    x: Optional[float] = None
    y: Optional[float] = None
    angle: Optional[int] = 0


class Obstacle(BaseModel):
    id: str
    label: str
    width: float
    height: float
    movable: bool = False
    x: Optional[float] = None
    y: Optional[float] = None
    angle: Optional[int] = 0


class Canvas(BaseModel):
    width: float
    height: float


class LayoutConfig(BaseModel):
    canvas: Canvas
    fus: List[FunctionalUnit]
    obstacles: List[Obstacle] = []

