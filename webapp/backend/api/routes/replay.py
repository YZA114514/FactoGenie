"""
回放 API
"""
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from typing import Optional
from pathlib import Path
import json
import asyncio

from api.deps import get_result_service, get_project_service
from services.result_service import ResultService
from services.project_service import ProjectService
from services.replay_service import get_replay_service, clear_replay_service

router = APIRouter()


class StartReplayRequest(BaseModel):
    episode: int


@router.post("/{project_id}/start")
async def start_replay(
    project_id: str,
    request: StartReplayRequest,
    project_service: ProjectService = Depends(get_project_service),
    result_service: ResultService = Depends(get_result_service),
):
    """
    启动回放会话
    
    加载检查点模型，准备回放
    """
    # 获取项目信息
    project = project_service.get_project(project_id)
    if not project:
        return {"code": 1002, "message": "Project not found", "data": None}
    
    # 获取检查点详情
    checkpoint = result_service.get_checkpoint_detail(project_id, request.episode)
    if not checkpoint:
        return {"code": 1002, "message": "Checkpoint not found", "data": None}
    
    # 获取配置文件路径
    project_dir = project_service.get_project_dir(project_id)
    factory_config_path = str(project_dir / "factory_config.json")
    layout_config_path = str(project_dir / "layout_config.json")
    
    # 初始化回放服务
    replay_service = get_replay_service(project_id)
    
    model_path = checkpoint.get('model_path')
    if not model_path:
        return {"code": 1002, "message": "Model path not found", "data": None}
    
    success = replay_service.load_checkpoint(
        model_path=model_path,
        layout_config_path=layout_config_path,
        factory_config_path=factory_config_path,
        training_params=project.training_params,
    )
    
    if not success:
        return {"code": 5000, "message": "Failed to load checkpoint", "data": None}
    
    return {
        "code": 0,
        "data": {
            "project_id": project_id,
            "episode": request.episode,
            "total_steps": replay_service.get_total_steps(),
            "layout": checkpoint.get('layout'),
            "metrics": checkpoint.get('metrics'),
        }
    }


@router.get("/{project_id}/step/{step}")
async def get_step_data(
    project_id: str,
    step: int,
):
    """
    获取指定步骤的数据（包括热力图）
    """
    replay_service = get_replay_service(project_id)
    
    try:
        data = replay_service.get_step_data(step)
        return {
            "code": 0,
            "data": data
        }
    except Exception as e:
        return {
            "code": 5000,
            "message": str(e),
            "data": None
        }


@router.post("/{project_id}/forward")
async def step_forward(project_id: str):
    """执行一步回放"""
    replay_service = get_replay_service(project_id)
    
    try:
        result = replay_service.step_forward()
        return {
            "code": 0,
            "data": result
        }
    except Exception as e:
        return {
            "code": 5000,
            "message": str(e),
            "data": None
        }


@router.get("/{project_id}/heatmap")
async def get_current_heatmap(project_id: str):
    """获取当前步骤的热力图"""
    replay_service = get_replay_service(project_id)
    
    try:
        heatmap = replay_service.get_heatmap()
        layout = replay_service.get_layout_state()
        
        return {
            "code": 0,
            "data": {
                "heatmap": heatmap,
                "layout": layout,
            }
        }
    except Exception as e:
        return {
            "code": 5000,
            "message": str(e),
            "data": None
        }


@router.delete("/{project_id}/session")
async def close_replay_session(project_id: str):
    """关闭回放会话"""
    clear_replay_service(project_id)
    return {"code": 0, "message": "Session closed"}


@router.websocket("/ws/{project_id}/{episode}")
async def replay_websocket(
    websocket: WebSocket,
    project_id: str,
    episode: int,
):
    """
    回放 WebSocket
    
    客户端发送命令：
    - {"action": "play"}: 开始/继续播放
    - {"action": "pause"}: 暂停
    - {"action": "step"}: 单步前进
    - {"action": "seek", "step": N}: 跳转到第N步
    
    服务端推送：
    - {"type": "step", "data": {...}}: 当前步骤数据
    - {"type": "complete"}: 回放完成
    """
    await websocket.accept()
    
    try:
        # 初始化回放状态
        current_step = 0
        is_playing = False
        total_steps = 10  # TODO: 从检查点获取实际步数
        
        while True:
            try:
                # 等待客户端命令（带超时）
                data = await asyncio.wait_for(
                    websocket.receive_json(),
                    timeout=1.0 if is_playing else 30.0
                )
                
                action = data.get('action')
                
                if action == 'play':
                    is_playing = True
                elif action == 'pause':
                    is_playing = False
                elif action == 'step':
                    if current_step < total_steps:
                        current_step += 1
                        await websocket.send_json({
                            "type": "step",
                            "data": {
                                "step": current_step,
                                "total": total_steps,
                                # TODO: 实际的步骤数据
                            }
                        })
                elif action == 'seek':
                    target = data.get('step', 0)
                    current_step = max(0, min(target, total_steps))
                    await websocket.send_json({
                        "type": "step",
                        "data": {
                            "step": current_step,
                            "total": total_steps,
                        }
                    })
                    
            except asyncio.TimeoutError:
                # 自动播放模式
                if is_playing and current_step < total_steps:
                    current_step += 1
                    await websocket.send_json({
                        "type": "step",
                        "data": {
                            "step": current_step,
                            "total": total_steps,
                        }
                    })
                    
                    if current_step >= total_steps:
                        await websocket.send_json({"type": "complete"})
                        is_playing = False
                else:
                    # 发送心跳
                    await websocket.send_json({"type": "heartbeat"})
                    
    except WebSocketDisconnect:
        pass

