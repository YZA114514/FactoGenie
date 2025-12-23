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
        # 确保返回的数据包含热力图
        if 'heatmap' not in data or data.get('heatmap') is None:
            data['heatmap'] = replay_service.get_heatmap()
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
        # 添加布局状态
        if 'error' not in result and 'done' not in result:
            result['layout'] = replay_service.get_layout_state()
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


@router.post("/{project_id}/backward")
async def step_backward(project_id: str):
    """回退一步回放"""
    replay_service = get_replay_service(project_id)
    
    try:
        result = replay_service.step_backward()
        # 添加布局状态
        if 'error' not in result:
            result['layout'] = replay_service.get_layout_state()
            # 获取完整的步骤数据
            step_data = replay_service.get_step_data()
            result['step'] = step_data.get('step')
            result['total_steps'] = step_data.get('total_steps')
            result['current_unit'] = step_data.get('current_unit')
            result['placed_units'] = step_data.get('placed_units')
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


@router.get("/{project_id}/inventory-chart")
async def get_inventory_chart(
    project_id: str,
    episode: Optional[int] = None,
    project_service: ProjectService = Depends(get_project_service),
):
    """
    获取物料量变化图表数据
    
    运行仿真并返回各监控点的物料量随时间变化数据
    
    Args:
        project_id: 项目ID
        episode: 可选的episode编号，用于获取对应检查点的布局
    """
    import sys
    from pathlib import Path
    import json
    
    # 添加simulation模块到路径
    project_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from simulation.interface import get_inventory_chart_data
        
        # 获取项目配置文件路径
        project_dir = project_service.get_project_dir(project_id)
        factory_config_path = project_dir / "factory_config.json"
        
        if not factory_config_path.exists():
            return {"code": 1002, "message": "Factory config not found", "data": None}
        
        # 优先使用检查点的布局文件，否则使用best布局，最后使用初始布局
        layout_path = None
        layouts_dir = project_dir / "layouts"
        
        print(f"[INFO] inventory-chart 请求: project_id={project_id}, episode={episode}")
        
        if episode is not None and layouts_dir.exists():
            # 尝试使用指定episode的布局
            episode_layout = layouts_dir / f"layout_ep{episode}.json"
            print(f"[INFO] 尝试查找布局: {episode_layout}, 存在: {episode_layout.exists()}")
            if episode_layout.exists():
                layout_path = episode_layout
        
        if layout_path is None and layouts_dir.exists():
            # 尝试使用best布局
            best_layout = layouts_dir / "layout_best.json"
            if best_layout.exists():
                layout_path = best_layout
            else:
                # 使用最新的布局文件（按episode数字排序）
                layout_files = list(layouts_dir.glob("layout_ep*.json"))
                if layout_files:
                    # 提取episode数字并按数字排序
                    def get_episode_num(p):
                        try:
                            return int(p.stem.replace("layout_ep", ""))
                        except:
                            return 0
                    layout_files.sort(key=get_episode_num, reverse=True)
                    layout_path = layout_files[0]
        
        if layout_path is None:
            # 回退到初始布局配置
            layout_config_path = project_dir / "layout_config.json"
            if layout_config_path.exists():
                layout_path = layout_config_path
        
        if layout_path is None:
            return {"code": 1002, "message": "No layout file found", "data": None}
        
        print(f"[INFO] 使用布局文件: {layout_path}")
        
        # 从训练参数获取仿真时长
        training_params_path = project_dir / "training_params.json"
        duration = 2000.0  # 默认时长
        if training_params_path.exists():
            with open(training_params_path, 'r', encoding='utf-8') as f:
                params = json.load(f)
                duration = params.get('simulation_duration', 2000.0)
        
        # 获取物料量变化数据
        chart_data = get_inventory_chart_data(
            config_path=str(factory_config_path),
            duration=duration,
            layout_path=str(layout_path),
        )
        
        return {"code": 0, "data": chart_data}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"code": 5000, "message": str(e), "data": None}


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

