"""
训练任务 API
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional, List
import uuid
import asyncio

router = APIRouter()


# ========== 请求/响应模型 ==========

class StartTrainingRequest(BaseModel):
    name: str
    factory_config: dict
    layout_config: dict
    constraints: Optional[dict] = None
    training_params: dict


class TrainingProgress(BaseModel):
    task_id: str
    status: str
    current_step: int
    total_steps: int
    current_episode: int
    elapsed_time: float
    estimated_remaining: float
    latest_metrics: Optional[dict] = None


# ========== 任务存储（临时，后续替换为数据库） ==========
training_tasks = {}


# ========== 路由 ==========

@router.post("/start", response_model=dict)
async def start_training(request: StartTrainingRequest):
    """启动训练任务"""
    task_id = str(uuid.uuid4())
    
    # TODO: 启动Celery异步任务
    training_tasks[task_id] = {
        "name": request.name,
        "status": "pending",
        "config": request.dict()
    }
    
    return {
        "code": 0,
        "data": {"task_id": task_id}
    }


@router.post("/{task_id}/stop", response_model=dict)
async def stop_training(task_id: str):
    """停止训练"""
    if task_id not in training_tasks:
        return {"code": 2002, "message": "Task not found", "data": None}
    
    # TODO: 发送停止信号给Celery任务
    training_tasks[task_id]["status"] = "stopped"
    
    return {
        "code": 0,
        "data": {"status": "stopped"}
    }


@router.post("/{task_id}/pause", response_model=dict)
async def pause_training(task_id: str):
    """暂停训练"""
    if task_id not in training_tasks:
        return {"code": 2002, "message": "Task not found", "data": None}
    
    training_tasks[task_id]["status"] = "paused"
    return {
        "code": 0,
        "data": {"status": "paused"}
    }


@router.post("/{task_id}/resume", response_model=dict)
async def resume_training(task_id: str):
    """恢复训练"""
    if task_id not in training_tasks:
        return {"code": 2002, "message": "Task not found", "data": None}
    
    training_tasks[task_id]["status"] = "running"
    return {
        "code": 0,
        "data": {"status": "running"}
    }


@router.get("/{task_id}/status", response_model=dict)
async def get_training_status(task_id: str):
    """获取训练状态"""
    if task_id not in training_tasks:
        return {"code": 2002, "message": "Task not found", "data": None}
    
    # TODO: 从Redis/数据库获取实际进度
    progress = TrainingProgress(
        task_id=task_id,
        status=training_tasks[task_id]["status"],
        current_step=0,
        total_steps=50000,
        current_episode=0,
        elapsed_time=0,
        estimated_remaining=0
    )
    
    return {
        "code": 0,
        "data": progress.dict()
    }


@router.get("/records", response_model=dict)
async def get_training_records(page: int = 1, size: int = 20):
    """获取训练记录列表"""
    # TODO: 从数据库查询
    return {
        "code": 0,
        "data": {
            "total": 0,
            "records": []
        }
    }


@router.get("/records/{record_id}", response_model=dict)
async def get_training_record(record_id: str):
    """获取单个训练记录详情"""
    # TODO: 从数据库查询
    return {
        "code": 0,
        "data": None
    }


@router.delete("/records/{record_id}", response_model=dict)
async def delete_training_record(record_id: str):
    """删除训练记录"""
    # TODO: 从数据库删除
    return {
        "code": 0,
        "message": "deleted"
    }


# ========== WebSocket ==========

@router.websocket("/ws/{task_id}")
async def training_websocket(websocket: WebSocket, task_id: str):
    """训练进度WebSocket"""
    await websocket.accept()
    
    try:
        while True:
            # TODO: 从Redis订阅实际进度
            # 临时模拟数据
            await websocket.send_json({
                "type": "progress",
                "data": {
                    "current_step": 0,
                    "total_steps": 50000,
                    "current_episode": 0,
                    "epsilon": 1.0,
                    "loss": 0
                }
            })
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass

