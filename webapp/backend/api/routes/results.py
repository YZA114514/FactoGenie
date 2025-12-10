"""
结果查询 API
"""
from fastapi import APIRouter, Depends
from typing import Optional

from ..deps import get_result_service
from ...services.result_service import ResultService

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
    result = service.get_best_layout(project_id)
    
    if not result:
        return {"code": 1002, "message": "No best layout found", "data": None}
    
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
