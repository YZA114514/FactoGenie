"""
Celery 异步任务
"""
import sys
from pathlib import Path
import json
import time
import traceback

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
        
        # 导入训练模块
        from environment.gym_wrapper import FactoryEnv
        from agent.trainer import train
        from argparse import Namespace
        
        # 构建训练参数
        params = Namespace(
            lr=training_params.get('learning_rate', 2e-5),
            batch_size=training_params.get('batch_size', 32),
            replay_size=training_params.get('replay_size', 50000),
            replay_start_size=training_params.get('replay_start_size', 5000),
            sync_target_frames=training_params.get('sync_target_every', 2000),
            epsilon_decay_last_frame=training_params.get('epsilon_decay_frames', 150000),
            total_steps=training_params.get('total_steps', 50000),
            epsilon_start=training_params.get('epsilon_start', 1.0),
            epsilon_final=training_params.get('epsilon_final', 0.05),
            device='cpu',
            gpu_id=0,
            use_prior=training_params.get('prioritized', False),
            use_double=training_params.get('double_dqn', False),
            use_dueling=training_params.get('dueling', False),
            use_noisy=training_params.get('noisy_net', False),
            sigma_init=0.5,
            use_simulation=training_params.get('use_simulation', True),
            simulation_duration=training_params.get('simulation_duration', 2000),
            reward_decompose='none',
            reward_gamma=0.9,
            weight_distance=training_params.get('weights', {}).get('distance', 0.20),
            weight_logistics=training_params.get('weights', {}).get('logistics', 0.30),
            weight_flow=training_params.get('weights', {}).get('flow', 0.20),
            weight_throughput=training_params.get('weights', {}).get('throughput', 0.25),
            weight_utilization=training_params.get('weights', {}).get('utilization', 0.05),
            placement_order=training_params.get('placement_order', 'default'),
            calibrate_episodes=training_params.get('calibrate_episodes', 0),
            throughput_target=training_params.get('throughput_target', None),
            checkpoint_interval=training_params.get('checkpoint_interval', 1000),
        )
        
        # TODO: 实际调用训练（需要修改trainer支持进度回调）
        # 这里先模拟训练过程
        total_steps = params.total_steps
        for step in range(0, total_steps, 1000):
            # 检查是否被取消
            if self.is_aborted():
                crud.update_project(db, project_id, status='stopped')
                return {'success': False, 'error': 'Task aborted'}
            
            # 更新进度
            progress = step / total_steps * 100
            self.update_state(
                state='PROGRESS',
                meta={
                    'current': step,
                    'total': total_steps,
                    'progress': progress,
                }
            )
            
            # 更新数据库
            crud.update_project(
                db, project_id,
                current_step=step,
                current_episode=step // 10,
            )
            
            time.sleep(0.1)  # 模拟训练
        
        # 训练完成
        crud.update_project(
            db, project_id,
            status='completed',
            current_step=total_steps,
            final_reward=0.0,  # TODO: 实际奖励
        )
        
        return {
            'success': True,
            'project_id': project_id,
            'status': 'completed',
        }
        
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
        
        # TODO: 加载模型并计算热力图
        # 这需要修改 Agent 类支持 Q 值输出
        
        return {
            'success': True,
            'project_id': project_id,
            'episode': episode,
            'heatmaps': [],  # TODO: 实际热力图数据
        }
        
    except Exception as e:
        traceback.print_exc()
        return {'success': False, 'error': str(e)}
    finally:
        db.close()

