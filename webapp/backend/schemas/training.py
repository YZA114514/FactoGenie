"""
训练相关数据模型
"""
from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Literal
from datetime import datetime


class ObjectiveWeights(BaseModel):
    distance: float = 0.25
    logistics: float = 0.25
    flow: float = 0.15
    throughput: float = 0.20
    utilization: float = 0.15

    @field_validator('distance', 'logistics', 'flow', 'throughput', 'utilization')
    @classmethod
    def check_range(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Weight must be between 0 and 1')
        return v


class TrainingParams(BaseModel):
    total_steps: int = 50000
    learning_rate: float = 0.00002
    batch_size: int = 32
    replay_size: int = 50000
    replay_start_size: int = 5000
    epsilon_start: float = 1.0
    epsilon_final: float = 0.05
    epsilon_decay_frames: int = 150000
    sync_target_every: int = 2000
    double_dqn: bool = False
    dueling: bool = False
    noisy_net: bool = False
    prioritized: bool = False
    simulation_duration: int = 2000
    use_simulation: bool = True
    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
    placement_order: Literal[
        "default", "size_desc", "size_asc", "flow_desc", 
        "random", "process_flow", "logistics_intensity"
    ] = "default"
    checkpoint_interval: int = 100  # 每多少episode保存一次权重，0表示不保存
    calibrate_episodes: int = 0      # 校准回合数，0表示不校准，建议100
    throughput_target: Optional[float] = None  # 用户指定的吞吐量目标


class FixedPosition(BaseModel):
    unit_id: str
    x: float
    y: float
    angle: int = 0


class AdjacencyConstraint(BaseModel):
    unit_a: str
    unit_b: str
    direction: Optional[Literal["horizontal", "vertical", "any"]] = "any"


class WallAttachConstraint(BaseModel):
    unit_id: str
    wall: Literal["top", "bottom", "left", "right"]


class Constraints(BaseModel):
    fixed_positions: List[FixedPosition] = []
    adjacency: List[AdjacencyConstraint] = []
    wall_attach: List[WallAttachConstraint] = []


class EpisodeMetrics(BaseModel):
    episode: int
    reward: float
    loss: Optional[float] = None
    epsilon: Optional[float] = None
    metrics: Optional[Dict[str, float]] = None


class TrainingProgress(BaseModel):
    task_id: str
    status: Literal["pending", "running", "paused", "completed", "failed", "stopped"]
    current_step: int
    total_steps: int
    current_episode: int
    elapsed_time: float
    estimated_remaining: float
    latest_metrics: Optional[EpisodeMetrics] = None


class Checkpoint(BaseModel):
    episode: int
    timestamp: datetime
    model_path: str
    reward: float
    is_best: bool = False


class TrainingRecord(BaseModel):
    id: str
    name: str
    created_at: datetime
    status: Literal["running", "completed", "failed", "stopped"]
    factory_config: dict
    layout_config: dict
    constraints: Optional[Constraints] = None
    training_params: TrainingParams
    checkpoints: List[Checkpoint] = []
    final_result: Optional[dict] = None

