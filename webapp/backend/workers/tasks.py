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

