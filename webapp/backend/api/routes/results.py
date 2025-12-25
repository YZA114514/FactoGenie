"""
结果查询 API
"""
from fastapi import APIRouter, Depends
from typing import Optional
from pathlib import Path
import csv

from api.deps import get_result_service, get_project_service
from services.result_service import ResultService
from services.project_service import ProjectService

router = APIRouter()


@router.get("/{project_id}/layouts")
async def get_layouts(
    project_id: str,
    page: int = 1,
    size: int = 20,
    service: ResultService = Depends(get_result_service),
):
    """获取布局历史"""
    result = service.get_layouts(project_id, page, size)
    return {
        "code": 0,
        "data": result
    }


@router.get("/{project_id}/best")
async def get_best_layout(
    project_id: str,
    service: ResultService = Depends(get_result_service),
):
    """获取最佳布局"""
    print(f"[INFO] get_best_layout 请求: project_id={project_id}")
    result = service.get_best_layout(project_id)
    
    if not result:
        print(f"[WARN] 项目 {project_id} 没有找到最佳布局")
        return {"code": 1002, "message": "No best layout found", "data": None}
    
    print(f"[INFO] 项目 {project_id} 最佳布局: episode={result.get('episode')}, reward={result.get('reward')}")
    return {
        "code": 0,
        "data": result
    }


@router.get("/{project_id}/metrics")
async def get_metrics(
    project_id: str,
    metric: str = "reward",
    start: int = 0,
    end: Optional[int] = None,
    service: ResultService = Depends(get_result_service),
):
    """获取指标曲线数据"""
    result = service.get_metrics_curve(project_id, metric, start, end)
    return {
        "code": 0,
        "data": result
    }


@router.get("/{project_id}/checkpoints/{episode}")
async def get_checkpoint_detail(
    project_id: str,
    episode: int,
    service: ResultService = Depends(get_result_service),
):
    """获取特定检查点详情"""
    result = service.get_checkpoint_detail(project_id, episode)
    
    if not result:
        return {"code": 1002, "message": "Checkpoint not found", "data": None}
    
    return {
        "code": 0,
        "data": result
    }


@router.get("/{project_id}/losses")
async def get_losses(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
):
    """获取训练loss曲线数据（从CSV文件读取）"""
    project_dir = project_service.get_project_dir(project_id)
    losses_csv = project_dir / "metrics" / "losses.csv"
    
    if not losses_csv.exists():
        return {"code": 0, "data": {"values": [], "message": "Loss file not found"}}
    
    values = []
    try:
        with open(losses_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    step = int(row.get('step', 0))
                    loss = float(row.get('loss', 0))
                    values.append({'step': step, 'loss': loss})
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        return {"code": 5000, "message": str(e), "data": None}
    
    return {
        "code": 0,
        "data": {
            "values": values,
            "count": len(values),
        }
    }


@router.get("/{project_id}/rewards-csv")
async def get_rewards_csv(
    project_id: str,
    project_service: ProjectService = Depends(get_project_service),
):
    """获取训练reward曲线数据（从CSV文件读取，更详细）"""
    project_dir = project_service.get_project_dir(project_id)
    rewards_csv = project_dir / "metrics" / "rewards.csv"
    
    if not rewards_csv.exists():
        return {"code": 0, "data": {"values": [], "message": "Rewards file not found"}}
    
    values = []
    try:
        with open(rewards_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    episode = int(row.get('episode', 0))
                    step = int(row.get('step', 0))
                    reward = float(row.get('reward', 0))
                    # 兼容新旧字段名 (mean_reward_200 vs mean_reward_100)
                    mean_reward = float(row.get('mean_reward_200', row.get('mean_reward_100', 0)))
                    epsilon = float(row.get('epsilon', 0))
                    values.append({
                        'episode': episode,
                        'step': step,
                        'reward': reward,
                        'mean_reward': mean_reward,
                        'epsilon': epsilon,
                    })
                except (ValueError, TypeError):
                    continue
    except Exception as e:
        return {"code": 5000, "message": str(e), "data": None}
    
    return {
        "code": 0,
        "data": {
            "values": values,
            "count": len(values),
        }
    }
