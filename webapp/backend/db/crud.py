"""
CRUD 操作
"""
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime

from .models import Project, Checkpoint, MetricsRecord, CalibrationCache


# ==================== Project ====================

def create_project(
    db: Session,
    name: str,
    factory_config: Dict = None,
    layout_config: Dict = None,
    constraints: Dict = None,
    training_params: Dict = None,
    description: str = None,
) -> Project:
    """创建项目"""
    project = Project(
        name=name,
        description=description,
        factory_config=factory_config,
        layout_config=layout_config,
        constraints=constraints,
        training_params=training_params,
        total_steps=training_params.get('total_steps', 50000) if training_params else 50000,
    )
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


def get_project(db: Session, project_id: str) -> Optional[Project]:
    """获取项目"""
    return db.query(Project).filter(Project.id == project_id).first()


def get_projects(
    db: Session,
    skip: int = 0,
    limit: int = 20,
    status: str = None,
) -> List[Project]:
    """获取项目列表"""
    query = db.query(Project)
    if status:
        query = query.filter(Project.status == status)
    return query.order_by(Project.created_at.desc()).offset(skip).limit(limit).all()


def get_projects_by_status(db: Session, status: str) -> List[Project]:
    """获取指定状态的所有项目"""
    return db.query(Project).filter(Project.status == status).all()


def count_projects(db: Session, status: str = None) -> int:
    """统计项目数量"""
    query = db.query(Project)
    if status:
        query = query.filter(Project.status == status)
    return query.count()


def update_project(
    db: Session,
    project_id: str,
    **kwargs
) -> Optional[Project]:
    """更新项目"""
    project = get_project(db, project_id)
    if not project:
        return None
    
    for key, value in kwargs.items():
        if hasattr(project, key):
            setattr(project, key, value)
    
    project.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(project)
    return project


def delete_project(db: Session, project_id: str) -> bool:
    """删除项目"""
    project = get_project(db, project_id)
    if not project:
        return False
    
    db.delete(project)
    db.commit()
    return True


# ==================== Checkpoint ====================

def create_checkpoint(
    db: Session,
    project_id: str,
    episode: int,
    reward: float,
    is_best: bool = False,
    model_path: str = None,
    layout_path: str = None,
    metrics_path: str = None,
    metrics_snapshot: Dict = None,
) -> Checkpoint:
    """创建检查点"""
    checkpoint = Checkpoint(
        project_id=project_id,
        episode=episode,
        reward=reward,
        is_best=is_best,
        model_path=model_path,
        layout_path=layout_path,
        metrics_path=metrics_path,
        metrics_snapshot=metrics_snapshot,
    )
    db.add(checkpoint)
    db.commit()
    db.refresh(checkpoint)
    return checkpoint


def get_checkpoints(
    db: Session,
    project_id: str,
    only_best: bool = False,
) -> List[Checkpoint]:
    """获取检查点列表"""
    query = db.query(Checkpoint).filter(Checkpoint.project_id == project_id)
    if only_best:
        query = query.filter(Checkpoint.is_best == True)
    return query.order_by(Checkpoint.episode.desc()).all()


def get_best_checkpoint(db: Session, project_id: str) -> Optional[Checkpoint]:
    """获取最佳检查点"""
    return db.query(Checkpoint).filter(
        Checkpoint.project_id == project_id,
        Checkpoint.is_best == True
    ).first()


# ==================== MetricsRecord ====================

def add_metrics_record(
    db: Session,
    project_id: str,
    episode: int,
    step: int,
    reward: float = None,
    loss: float = None,
    epsilon: float = None,
    **metrics
) -> MetricsRecord:
    """添加指标记录"""
    record = MetricsRecord(
        project_id=project_id,
        episode=episode,
        step=step,
        reward=reward,
        loss=loss,
        epsilon=epsilon,
        distance_score=metrics.get('distance_score'),
        logistics_score=metrics.get('logistics_score'),
        throughput=metrics.get('throughput'),
        utilization=metrics.get('utilization'),
        flow_clarity=metrics.get('flow_clarity'),
    )
    db.add(record)
    db.commit()
    return record


def get_metrics_history(
    db: Session,
    project_id: str,
    metric_name: str = 'reward',
    start_episode: int = 0,
    end_episode: int = None,
) -> List[Dict]:
    """获取指标历史"""
    query = db.query(MetricsRecord).filter(
        MetricsRecord.project_id == project_id,
        MetricsRecord.episode >= start_episode,
    )
    if end_episode:
        query = query.filter(MetricsRecord.episode <= end_episode)
    
    records = query.order_by(MetricsRecord.episode).all()
    
    result = []
    for r in records:
        value = getattr(r, metric_name, None)
        if value is not None:
            result.append({'episode': r.episode, 'value': value})
    
    return result


# ==================== CalibrationCache ====================

def get_calibration(db: Session, config_hash: str) -> Optional[CalibrationCache]:
    """获取校准缓存"""
    return db.query(CalibrationCache).filter(
        CalibrationCache.config_hash == config_hash
    ).first()


def save_calibration(
    db: Session,
    config_hash: str,
    bounds: Dict,
    n_episodes: int,
    simulation_duration: float,
    throughput_target: float = None,
) -> CalibrationCache:
    """保存校准结果"""
    # 先删除旧的
    existing = get_calibration(db, config_hash)
    if existing:
        db.delete(existing)
    
    cache = CalibrationCache(
        config_hash=config_hash,
        bounds=bounds,
        n_episodes=n_episodes,
        simulation_duration=simulation_duration,
        throughput_target=throughput_target,
    )
    db.add(cache)
    db.commit()
    db.refresh(cache)
    return cache

