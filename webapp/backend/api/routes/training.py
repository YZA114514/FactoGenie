"""
训练任务 API
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from typing import Optional, Dict
import asyncio

from api.deps import get_project_service, get_training_service
from services.project_service import ProjectService
from services.training_service import TrainingService

router = APIRouter()


# ========== 请求模型 ==========

class CreateProjectRequest(BaseModel):
    name: str
    factory_config: dict
    layout_config: dict
    constraints: Optional[dict] = None
    training_params: Optional[dict] = None
    description: Optional[str] = None


# ========== 项目管理 ==========

@router.post("/projects")
async def create_project(
    request: CreateProjectRequest,
    service: ProjectService = Depends(get_project_service),
):
    """创建新项目"""
    project = service.create_project(
        name=request.name,
        factory_config=request.factory_config,
        layout_config=request.layout_config,
        constraints=request.constraints,
        training_params=request.training_params,
        description=request.description,
    )
    
    return {
        "code": 0,
        "data": {
            "project_id": project.id,
            "name": project.name,
            "status": project.status,
        }
    }


@router.get("/projects")
async def list_projects(
    page: int = 1,
    size: int = 20,
    status: Optional[str] = None,
    service: ProjectService = Depends(get_project_service),
):
    """获取项目列表"""
    result = service.list_projects(page=page, size=size, status=status)
    
    projects_data = []
    for p in result['projects']:
        projects_data.append({
            "id": p.id,
            "name": p.name,
            "status": p.status,
            "current_episode": p.current_episode,
            "total_steps": p.total_steps,
            "best_reward": p.best_reward,
            "created_at": p.created_at.isoformat(),
        })
    
    return {
        "code": 0,
        "data": {
            "total": result['total'],
            "page": result['page'],
            "size": result['size'],
            "projects": projects_data,
        }
    }


@router.get("/projects/{project_id}")
async def get_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """获取项目详情"""
    project = service.get_project(project_id)
    if not project:
        return {"code": 1002, "message": "Project not found", "data": None}
    
    return {
        "code": 0,
        "data": {
            "id": project.id,
            "name": project.name,
            "description": project.description,
            "status": project.status,
            "factory_config": project.factory_config,
            "layout_config": project.layout_config,
            "constraints": project.constraints,
            "training_params": project.training_params,
            "current_step": project.current_step,
            "total_steps": project.total_steps,
            "current_episode": project.current_episode,
            "best_reward": project.best_reward,
            "created_at": project.created_at.isoformat(),
            "updated_at": project.updated_at.isoformat(),
        }
    }


@router.delete("/projects/{project_id}")
async def delete_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
):
    """删除项目"""
    success = service.delete_project(project_id)
    if not success:
        return {"code": 1002, "message": "Project not found", "data": None}
    
    return {"code": 0, "message": "deleted", "data": None}


# ========== 训练控制 ==========

@router.post("/projects/{project_id}/start")
async def start_training(
    project_id: str,
    service: TrainingService = Depends(get_training_service),
):
    """启动训练"""
    # 默认使用本地线程，避免 Celery 未配置时的延迟
    result = service.start_training(project_id, use_celery=False)
    
    if not result['success']:
        return {"code": 2001, "message": result['error'], "data": None}
    
    return {
        "code": 0,
        "data": {
            "project_id": result['project_id'],
            "status": result['status'],
        }
    }


@router.post("/projects/{project_id}/stop")
async def stop_training(
    project_id: str,
    service: TrainingService = Depends(get_training_service),
):
    """停止训练"""
    result = service.stop_training(project_id)
    
    if not result['success']:
        return {"code": 2002, "message": result['error'], "data": None}
    
    return {
        "code": 0,
        "data": {"status": result['status']}
    }


@router.get("/projects/{project_id}/status")
async def get_training_status(
    project_id: str,
    service: TrainingService = Depends(get_training_service),
):
    """获取训练状态"""
    progress = service.get_progress(project_id)
    
    if not progress:
        return {"code": 1002, "message": "Project not found", "data": None}
    
    return {
        "code": 0,
        "data": progress
    }


@router.get("/projects/{project_id}/checkpoints")
async def get_checkpoints(
    project_id: str,
    only_best: bool = False,
    service: TrainingService = Depends(get_training_service),
):
    """获取检查点列表"""
    checkpoints = service.get_checkpoints(project_id, only_best)
    
    data = []
    for cp in checkpoints:
        data.append({
            "id": cp.id,
            "episode": cp.episode,
            "reward": cp.reward,
            "is_best": cp.is_best,
            "created_at": cp.created_at.isoformat(),
        })
    
    return {
        "code": 0,
        "data": data
    }


# ========== WebSocket ==========

# 存储活跃的WebSocket连接
active_connections: Dict[str, list] = {}


@router.websocket("/ws/{project_id}")
async def training_websocket(websocket: WebSocket, project_id: str):
    """训练进度WebSocket"""
    await websocket.accept()
    
    # 添加到连接池
    if project_id not in active_connections:
        active_connections[project_id] = []
    active_connections[project_id].append(websocket)
    
    try:
        while True:
            # 等待客户端消息或保持连接
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
                # 处理客户端命令（如 ping）
                if data == "ping":
                    await websocket.send_json({"type": "pong"})
            except asyncio.TimeoutError:
                # 发送心跳
                await websocket.send_json({"type": "heartbeat"})
    except WebSocketDisconnect:
        # 从连接池移除
        if project_id in active_connections:
            active_connections[project_id].remove(websocket)


async def broadcast_progress(project_id: str, data: dict):
    """广播训练进度到所有连接的客户端"""
    if project_id in active_connections:
        for ws in active_connections[project_id]:
            try:
                await ws.send_json(data)
            except:
                pass
