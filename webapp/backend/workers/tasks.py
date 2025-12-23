"""
Celery 异步任务
"""
import sys
from pathlib import Path
import json
import time
import traceback
import threading
from typing import Dict

# 添加项目路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent.parent))

from workers.celery_app import celery_app
from db.database import SessionLocal
from db import crud


def get_db():
    """获取数据库会话"""
    db = SessionLocal()
    try:
        return db
    finally:
        pass  # 调用者负责关闭


@celery_app.task(bind=True)
def run_training_task(self, project_id: str):
    """
    异步训练任务
    
    Args:
        project_id: 项目ID
    """
    db = get_db()
    
    try:
        # 获取项目配置
        project = crud.get_project(db, project_id)
        if not project:
            return {'success': False, 'error': 'Project not found'}
        
        # 更新状态
        crud.update_project(db, project_id, status='running', celery_task_id=self.request.id)
        
        # 获取配置
        factory_config = project.factory_config
        layout_config = project.layout_config
        constraints = project.constraints or {}
        training_params = project.training_params or {}
        
        # 准备配置文件
        project_dir = Path(__file__).parent.parent.parent.parent / "data" / "projects" / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = project_dir / "checkpoints"
        checkpoints_dir.mkdir(exist_ok=True)
        
        factory_config_path = project_dir / "factory_config.json"
        layout_config_path = project_dir / "layout_config.json"
        
        with open(factory_config_path, 'w') as f:
            json.dump(factory_config, f)
        
        # 合并约束到布局配置
        layout_with_constraints = layout_config.copy()
        if constraints:
            layout_with_constraints['constraints'] = constraints
        
        with open(layout_config_path, 'w') as f:
            json.dump(layout_with_constraints, f)
        
        # 使用优化服务进行训练
        from services.optimization_service import OptimizationService
        
        opt_service = OptimizationService(project_dir)
        
        # 创建环境
        opt_service.create_environment(
            factory_config_path=str(factory_config_path),
            layout_config_path=str(layout_config_path),
            training_params=training_params,
        )
        
        # 进度回调
        def on_progress(data):
            # 更新 Celery 状态
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': data['step'],
                    'episode': data['episode'],
                    'reward': data.get('reward'),
                    'mean_reward': data.get('mean_reward'),
                }
            )
            
            # 更新数据库
            crud.update_project(
                db, project_id,
                current_step=data['step'],
                current_episode=data['episode'],
            )
        
        # 检查点回调
        def on_checkpoint(episode, reward, model_path):
            # 保存到数据库
            crud.create_checkpoint(
                db, project_id,
                episode=episode,
                reward=reward,
                model_path=model_path,
                layout_path=str(layout_config_path),
            )
            
            # 更新最佳奖励
            if project.best_reward is None or reward > project.best_reward:
                crud.update_project(db, project_id, best_reward=reward)
        
        # 运行训练
        result = opt_service.run_training(
            training_params=training_params,
            progress_callback=on_progress,
            checkpoint_callback=on_checkpoint,
        )
        
        if result['success']:
            crud.update_project(
                db, project_id,
                status='completed',
                final_reward=result.get('final_reward', 0),
            )
        else:
            crud.update_project(db, project_id, status='failed')
        
        return result
        
    except Exception as e:
        traceback.print_exc()
        crud.update_project(db, project_id, status='failed')
        return {
            'success': False,
            'error': str(e),
        }
    finally:
        db.close()


@celery_app.task
def run_calibration_task(
    factory_config_path: str,
    layout_config_path: str,
    n_episodes: int = 100,
    simulation_duration: float = 2000,
    throughput_target: float = None,
):
    """
    异步校准任务
    """
    try:
        from calibration.calibrator import calibrate_from_config
        
        bounds = calibrate_from_config(
            factory_config_path=factory_config_path,
            layout_config_path=layout_config_path,
            n_episodes=n_episodes,
            simulation_duration=simulation_duration,
            throughput_target=throughput_target,
        )
        
        return {
            'success': True,
            'bounds': bounds,
        }
    except Exception as e:
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e),
        }


@celery_app.task
def run_replay_task(project_id: str, episode: int):
    """
    回放任务：加载检查点并计算热力图
    """
    db = get_db()
    
    try:
        # 获取检查点
        checkpoints = crud.get_checkpoints(db, project_id)
        checkpoint = None
        for cp in checkpoints:
            if cp.episode == episode:
                checkpoint = cp
                break
        
        if not checkpoint or not checkpoint.model_path:
            return {'success': False, 'error': 'Checkpoint not found'}
        
        # 获取项目信息
        project = crud.get_project(db, project_id)
        if not project:
            return {'success': False, 'error': 'Project not found'}
        
        # 加载回放服务
        from services.replay_service import get_replay_service
        
        replay_service = get_replay_service(project_id)
        
        project_dir = Path(__file__).parent.parent.parent.parent / "data" / "projects" / project_id
        
        success = replay_service.load_checkpoint(
            model_path=checkpoint.model_path,
            layout_config_path=str(project_dir / "layout_config.json"),
            factory_config_path=str(project_dir / "factory_config.json"),
            training_params=project.training_params or {},
        )
        
        if not success:
            return {'success': False, 'error': 'Failed to load checkpoint'}
        
        # 获取所有步骤的热力图
        heatmaps = []
        total_steps = replay_service.get_total_steps()
        
        for step in range(total_steps):
            step_data = replay_service.get_step_data(step)
            heatmaps.append({
                'step': step,
                'unit': step_data.get('current_unit'),
                'heatmap': step_data.get('heatmap'),
            })
            
            # 执行一步
            result = replay_service.step_forward()
            if result.get('done'):
                break
        
        return {
            'success': True,
            'project_id': project_id,
            'episode': episode,
            'total_steps': total_steps,
            'heatmaps': heatmaps,
        }
        
    except Exception as e:
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    finally:
        db.close()


@celery_app.task(bind=True)
def run_training_with_core(self, project_id: str):
    """
    使用核心算法的 train_with_callbacks 运行训练
    """
    db = get_db()
    
    try:
        import threading
        from argparse import Namespace
        
        project = crud.get_project(db, project_id)
        if not project:
            return {'success': False, 'error': 'Project not found'}
        
        # 更新状态
        crud.update_project(db, project_id, status='running', celery_task_id=self.request.id)
        
        # 准备配置
        training_params = project.training_params or {}
        weights = training_params.get('weights', {})
        
        project_dir = Path(__file__).parent.parent.parent.parent / "data" / "projects" / project_id
        project_dir.mkdir(parents=True, exist_ok=True)
        
        factory_config_path = project_dir / "factory_config.json"
        layout_config_path = project_dir / "layout_config.json"
        
        # 保存配置文件
        with open(factory_config_path, 'w') as f:
            json.dump(project.factory_config, f)
        
        layout_with_constraints = project.layout_config.copy()
        if project.constraints:
            layout_with_constraints['constraints'] = project.constraints
        
        with open(layout_config_path, 'w') as f:
            json.dump(layout_with_constraints, f)
        
        # 构建训练参数
        params = Namespace(
            device='cpu',
            gpu_id=0,
            lr=training_params.get('learning_rate', 2e-5),
            batch_size=training_params.get('batch_size', 32),
            replay_size=training_params.get('replay_size', 50000),
            replay_start_size=training_params.get('replay_start_size', 5000),
            sync_target_frames=training_params.get('sync_target_every', 2000),
            epsilon_start=training_params.get('epsilon_start', 1.0),
            epsilon_final=training_params.get('epsilon_final', 0.05),
            epsilon_decay_last_frame=training_params.get('epsilon_decay_frames', 150000),
            total_steps=training_params.get('total_steps', 50000),
            use_prior=training_params.get('prioritized', False),
            use_double=training_params.get('double_dqn', False),
            use_dueling=training_params.get('dueling', False),
            use_noisy=training_params.get('noisy_net', False),
            use_simulation=training_params.get('use_simulation', True),
            simulation_duration=training_params.get('simulation_duration', 2000),
            weight_distance=weights.get('distance', 0.20),
            weight_logistics=weights.get('logistics', 0.30),
            weight_flow=weights.get('flow', 0.20),
            weight_throughput=weights.get('throughput', 0.25),
            weight_utilization=weights.get('utilization', 0.05),
            placement_order=training_params.get('placement_order', 'default'),
            calibrate_episodes=training_params.get('calibrate_episodes', 0),
            throughput_target=training_params.get('throughput_target', None),
            checkpoint_interval=training_params.get('checkpoint_interval', 100),
            factory_config=str(factory_config_path),
        )
        
        # 停止事件
        stop_event = threading.Event()
        _stop_events[project_id] = stop_event
        
        # 进度回调
        def on_progress(**kwargs):
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': kwargs.get('step'),
                    'total': params.total_steps,
                    'episode': kwargs.get('episode'),
                    'reward': kwargs.get('current_reward'),
                    'mean_reward': kwargs.get('mean_reward'),
                    'best_reward': kwargs.get('best_reward'),
                    'progress_pct': kwargs.get('progress_pct'),
                }
            )
            
            crud.update_project(
                db, project_id,
                current_step=kwargs.get('step'),
                current_episode=kwargs.get('episode'),
            )
        
        # 结果回调
        def on_result(layout_data):
            # 保存最佳布局
            best_layout_path = project_dir / "layout_best.json"
            with open(best_layout_path, 'w') as f:
                json.dump(layout_data, f, indent=2)
        
        # 调用核心训练函数
        from agent.trainer import train_with_callbacks
        
        result = train_with_callbacks(
            params,
            progress_callback=on_progress,
            result_callback=on_result,
            stop_event=stop_event,
            db_session=db,
            project_id=project_id,
        )
        
        # 更新最终状态
        crud.update_project(
            db, project_id,
            status='completed',
            best_reward=result.get('best_reward'),
            current_step=result.get('total_steps'),
            current_episode=result.get('total_episodes'),
        )
        
        # 清理
        if project_id in _stop_events:
            del _stop_events[project_id]
        
        return {
            'success': True,
            **result,
        }
        
    except Exception as e:
        traceback.print_exc()
        crud.update_project(db, project_id, status='failed')
        return {'success': False, 'error': str(e)}
    finally:
        db.close()


# 停止事件存储
_stop_events: Dict[str, "threading.Event"] = {}


@celery_app.task
def stop_training_task(project_id: str):
    """停止训练任务"""
    if project_id in _stop_events:
        _stop_events[project_id].set()
        return {'success': True, 'message': 'Stop signal sent'}
    return {'success': False, 'error': 'Task not found'}

