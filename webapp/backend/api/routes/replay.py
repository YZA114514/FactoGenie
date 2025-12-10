"""
回放 API
"""
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from typing import Optional
from pathlib import Path
import json
import asyncio

from api.deps import get_result_service
from services.result_service import ResultService

router = APIRouter()


@router.post("/{project_id}/start")
async def start_replay(
    project_id: str,
    episode: int,
    service: ResultService = Depends(get_result_service),
):
    """
    启动回放会话
    
    返回回放所需的数据（检查点布局 + 放置顺序）
    """
    # 获取检查点详情
    checkpoint = service.get_checkpoint_detail(project_id, episode)
    
    if not checkpoint:
        return {"code": 1002, "message": "Checkpoint not found", "data": None}
    
    # 返回回放数据
    return {
        "code": 0,
        "data": {
            "project_id": project_id,
            "episode": episode,
            "layout": checkpoint.get('layout'),
            "metrics": checkpoint.get('metrics'),
            "model_path": checkpoint.get('model_path'),
        }
    }


@router.get("/{project_id}/heatmap/{episode}/{step}")
async def get_heatmap(
    project_id: str,
    episode: int,
    step: int,
):
    """
    获取特定步骤的动作热力图
    
    注意：这需要加载模型并重新计算，可能较慢
    """
    import sys
    project_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        # 获取项目数据目录
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "projects" / project_id
        checkpoint_dir = data_dir / "checkpoints"
        
        # 找到对应的模型文件
        model_path = checkpoint_dir / f"model_ep{episode}.pth"
        if not model_path.exists():
            model_path = checkpoint_dir / "model_best.pth"
        
        if not model_path.exists():
            return {"code": 1002, "message": "Model not found", "data": None}
        
        # TODO: 实现热力图计算
        # 这需要：
        # 1. 加载模型
        # 2. 重建环境状态到第 step 步
        # 3. 对所有可能的动作计算 Q 值
        # 4. 将 Q 值映射为热力图
        
        # 暂时返回模拟数据
        return {
            "code": 0,
            "data": {
                "step": step,
                "unit_id": f"unit_{step}",
                "grid_width": 20,
                "grid_height": 20,
                "angle_options": [0, 90, 180, 270],
                "q_values": None,  # TODO: 实际Q值
                "selected_action": {
                    "x": 5,
                    "y": 5,
                    "angle": 0,
                    "q_value": 0.5,
                },
                "note": "Heatmap calculation not yet implemented"
            }
        }
        
    except Exception as e:
        return {
            "code": 5000,
            "message": str(e),
            "data": None
        }


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

