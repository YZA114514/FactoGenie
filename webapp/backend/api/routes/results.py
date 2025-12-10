"""
结果查询 API
"""
from fastapi import APIRouter
from typing import Optional

router = APIRouter()


@router.get("/{task_id}/layouts", response_model=dict)
async def get_layouts(task_id: str, page: int = 1, size: int = 20):
    """获取布局历史"""
    # TODO: 从数据库/文件查询
    return {
        "code": 0,
        "data": {
            "total": 0,
            "layouts": []
        }
    }


@router.get("/{task_id}/best", response_model=dict)
async def get_best_layout(task_id: str):
    """获取最佳布局"""
    # TODO: 从数据库/文件查询
    return {
        "code": 0,
        "data": None
    }


@router.get("/{task_id}/metrics", response_model=dict)
async def get_metrics(
    task_id: str,
    metric: str = "reward",
    start: int = 0,
    end: Optional[int] = None
):
    """获取指标曲线数据"""
    # TODO: 从CSV文件读取
    return {
        "code": 0,
        "data": {
            "metric": metric,
            "values": []
        }
    }


@router.get("/{task_id}/heatmap/{episode}/{step}", response_model=dict)
async def get_action_heatmap(task_id: str, episode: int, step: int):
    """获取动作热力图"""
    # TODO: 从文件读取
    return {
        "code": 0,
        "data": None
    }

