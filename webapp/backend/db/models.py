"""
数据库模型定义
"""
from sqlalchemy import Column, String, Integer, Float, Boolean, Text, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from .database import Base


def generate_uuid():
    return str(uuid.uuid4())


class Project(Base):
    """项目/训练任务"""
    __tablename__ = "projects"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String(20), default="pending")  # pending, running, paused, completed, failed, stopped
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 配置（JSON存储）
    factory_config = Column(JSON, nullable=True)
    layout_config = Column(JSON, nullable=True)
    constraints = Column(JSON, nullable=True)
    training_params = Column(JSON, nullable=True)
    
    # 进度信息
    current_step = Column(Integer, default=0)
    total_steps = Column(Integer, default=50000)
    current_episode = Column(Integer, default=0)
    
    # 结果
    best_reward = Column(Float, nullable=True)
    final_reward = Column(Float, nullable=True)
    
    # Celery 任务ID
    celery_task_id = Column(String(255), nullable=True)
    
    # 关系
    checkpoints = relationship("Checkpoint", back_populates="project", cascade="all, delete-orphan")
    metrics_history = relationship("MetricsRecord", back_populates="project", cascade="all, delete-orphan")


class Checkpoint(Base):
    """检查点"""
    __tablename__ = "checkpoints"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    
    episode = Column(Integer, nullable=False)
    reward = Column(Float, nullable=False)
    is_best = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 文件路径
    model_path = Column(String(500), nullable=True)
    layout_path = Column(String(500), nullable=True)
    metrics_path = Column(String(500), nullable=True)
    
    # 指标快照
    metrics_snapshot = Column(JSON, nullable=True)
    
    # 关系
    project = relationship("Project", back_populates="checkpoints")


class MetricsRecord(Base):
    """指标记录（用于绘制曲线）"""
    __tablename__ = "metrics_records"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String(36), ForeignKey("projects.id"), nullable=False)
    
    episode = Column(Integer, nullable=False)
    step = Column(Integer, nullable=False)
    reward = Column(Float, nullable=True)
    loss = Column(Float, nullable=True)
    epsilon = Column(Float, nullable=True)
    
    # 仿真指标
    distance_score = Column(Float, nullable=True)
    logistics_score = Column(Float, nullable=True)
    throughput = Column(Float, nullable=True)
    utilization = Column(Float, nullable=True)
    flow_clarity = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # 关系
    project = relationship("Project", back_populates="metrics_history")


class CalibrationCache(Base):
    """校准缓存"""
    __tablename__ = "calibration_cache"
    
    id = Column(String(36), primary_key=True, default=generate_uuid)
    config_hash = Column(String(32), unique=True, nullable=False)
    
    bounds = Column(JSON, nullable=False)
    n_episodes = Column(Integer, nullable=False)
    simulation_duration = Column(Float, nullable=False)
    throughput_target = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

