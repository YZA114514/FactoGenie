"""
训练任务服务
"""
from sqlalchemy.orm import Session
from typing import Dict, Optional
from pathlib import Path
import json
import threading

from db import crud
from db.models import Project, Checkpoint
from db.database import SessionLocal
from services.optimization_service import OptimizationService


class TrainingService:
    """训练任务服务"""
    
    # 存储运行中的任务
    _running_tasks: Dict[str, dict] = {}
    _stop_events: Dict[str, threading.Event] = {}
    
    def __init__(self, db: Session):
        self.db = db
        self.data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    def start_training(self, project_id: str, use_celery: bool = True) -> Dict:
        """
        启动训练任务
        
        Args:
            project_id: 项目ID
            use_celery: 是否使用Celery异步任务（False时同步阻塞）
        """
        project = crud.get_project(self.db, project_id)
        if not project:
            return {'success': False, 'error': 'Project not found'}
        
        if project.status == 'running':
            return {'success': False, 'error': 'Training already running'}
        
        # 创建停止事件
        stop_event = threading.Event()
        self._stop_events[project_id] = stop_event

        # 更新状态
        crud.update_project(self.db, project_id, status='running')

        if use_celery:
            # 启动Celery任务
            try:
                from workers.tasks import run_training_with_core
                task = run_training_with_core.delay(project_id)
                crud.update_project(self.db, project_id, celery_task_id=task.id)
                
                return {
                    'success': True,
                    'project_id': project_id,
                    'task_id': task.id,
                    'status': 'running',
                }
            except Exception as e:
                crud.update_project(self.db, project_id, status='failed')
                return {'success': False, 'error': f'Failed to start Celery task: {e}'}
        else:
            # 启动后台线程（本地同步训练）
            t = threading.Thread(target=self._run_training_sync, args=(project_id, stop_event), daemon=True)
            t.start()
            self._running_tasks[project_id] = {"thread": t}
            
            return {
                'success': True,
                'project_id': project_id,
                'status': 'running',
                'message': 'Running in local thread',
            }
    
    def stop_training(self, project_id: str) -> Dict:
        """停止训练任务"""
        project = crud.get_project(self.db, project_id)
        if not project:
            return {'success': False, 'error': 'Project not found'}
        
        if project.status != 'running':
            return {'success': False, 'error': 'Training not running'}
        
        # 设置停止标志
        if project_id in self._stop_events:
            self._stop_events[project_id].set()
        
        # 尝试发送停止信号到Celery任务（如果可用）
        try:
            from workers.tasks import stop_training_task
            stop_training_task.delay(project_id)
        except Exception as e:
            print(f"Warning: Failed to send stop signal: {e}")
        
        # 取消Celery任务（如果有）
        if hasattr(project, 'celery_task_id') and project.celery_task_id:
            try:
                from workers.celery_app import celery_app
                celery_app.control.revoke(project.celery_task_id, terminate=True)
            except Exception as e:
                print(f"Warning: Failed to revoke Celery task: {e}")
        
        crud.update_project(self.db, project_id, status='stopped')
        
        return {
            'success': True,
            'project_id': project_id,
            'status': 'stopped',
        }
    
    def get_progress(self, project_id: str) -> Optional[Dict]:
        """获取训练进度"""
        project = crud.get_project(self.db, project_id)
        if not project:
            return None
        
        return {
            'project_id': project_id,
            'status': project.status,
            'current_step': project.current_step,
            'total_steps': project.total_steps,
            'current_episode': project.current_episode,
            'best_reward': project.best_reward,
        }
    
    def update_progress(
        self,
        project_id: str,
        current_step: int = None,
        current_episode: int = None,
        best_reward: float = None,
    ):
        """更新训练进度（由训练任务调用）"""
        updates = {}
        if current_step is not None:
            updates['current_step'] = current_step
        if current_episode is not None:
            updates['current_episode'] = current_episode
        if best_reward is not None:
            updates['best_reward'] = best_reward
        
        if updates:
            crud.update_project(self.db, project_id, **updates)
    
    def save_checkpoint(
        self,
        project_id: str,
        episode: int,
        reward: float,
        is_best: bool = False,
        model_path: str = None,
        layout_path: str = None,
        metrics_snapshot: Dict = None,
    ) -> Checkpoint:
        """保存检查点"""
        return crud.create_checkpoint(
            self.db,
            project_id=project_id,
            episode=episode,
            reward=reward,
            is_best=is_best,
            model_path=model_path,
            layout_path=layout_path,
            metrics_snapshot=metrics_snapshot,
        )
    
    def get_checkpoints(self, project_id: str, only_best: bool = False):
        """获取检查点列表"""
        return crud.get_checkpoints(self.db, project_id, only_best)
    
    def should_stop(self, project_id: str) -> bool:
        """检查是否应该停止训练"""
        if project_id in self._stop_events:
            return self._stop_events[project_id].is_set()
        return False

    def _run_training_sync(self, project_id: str, stop_event: threading.Event):
        """在本进程中运行训练（简化版，不依赖 Celery/Redis）"""
        db_local: Session = SessionLocal()
        project_dir = self.data_dir / "projects" / project_id
        try:
            project = crud.get_project(db_local, project_id)
            if not project:
                return
            # 以默认参数为基础，合并前端/项目提供的训练参数（缺失项使用默认值）
            try:
                from backend import config as backend_config  # type: ignore
                training_params = backend_config.DEFAULT_TRAINING_PARAMS.copy()
                # 合并用户提供的参数（覆盖默认值）
                user_params = project.training_params or {}
                if isinstance(user_params, dict):
                    training_params.update(user_params)
                # 如果配置了 SHORT_TRAINING_MAX_STEPS (>0)，则用于开发/调试时限制步数；0 表示不限制
                max_steps = getattr(backend_config, "SHORT_TRAINING_MAX_STEPS", 0)
                if isinstance(max_steps, int) and max_steps > 0:
                    training_params["total_steps"] = min(training_params.get("total_steps", 0), max_steps)
            except Exception:
                # 若无法读取配置，则退回到项目里直接给定的参数（浅拷贝）
                training_params = (project.training_params or {}).copy()

            # 准备项目目录与配置文件
            project_dir.mkdir(parents=True, exist_ok=True)
            checkpoints_dir = project_dir / "checkpoints"
            checkpoints_dir.mkdir(exist_ok=True)

            factory_config_path = project_dir / "factory_config.json"
            layout_config_path = project_dir / "layout_config.json"

            with open(factory_config_path, 'w') as f:
                json.dump(project.factory_config, f, indent=2)

            layout_with_constraints = project.layout_config.copy()
            if project.constraints:
                layout_with_constraints['constraints'] = project.constraints
            with open(layout_config_path, 'w') as f:
                json.dump(layout_with_constraints, f, indent=2)

            # 创建优化服务
            opt_service = OptimizationService(project_dir)
            opt_service.stop_event = stop_event  # 传入停止信号
            opt_service.create_environment(
                factory_config_path=str(factory_config_path),
                layout_config_path=str(layout_config_path),
                training_params=training_params,
            )

            # 回调：进度
            best_so_far = project.best_reward if project.best_reward is not None else float("-inf")

            def on_progress(data):
                crud.update_project(
                    db_local,
                    project_id,
                    current_step=data.get('step'),
                    current_episode=data.get('episode'),
                    best_reward=data.get('mean_reward'),
                )
                try:
                    crud.add_metrics_record(
                        db_local,
                        project_id=project_id,
                        episode=data.get('episode') or 0,
                        step=data.get('step') or 0,
                        reward=data.get('reward'),
                        epsilon=data.get('epsilon'),
                        # 可扩展更多指标
                    )
                except Exception:
                    pass

            # 回调：检查点
            def on_checkpoint(ep, reward, model_path):
                nonlocal best_so_far
                is_best = reward > (best_so_far if best_so_far is not None else float("-inf"))
                crud.create_checkpoint(
                    db_local,
                    project_id,
                    episode=ep,
                    reward=reward,
                    model_path=model_path,
                    layout_path=str(layout_config_path),
                    is_best=is_best,
                )
                if is_best:
                    best_so_far = reward
                    crud.update_project(db_local, project_id, best_reward=reward)

            # 运行训练
            result = opt_service.run_training(
                training_params=training_params,
                progress_callback=on_progress,
                checkpoint_callback=on_checkpoint,
            )

            if result.get('success'):
                crud.update_project(
                    db_local, project_id,
                    status='completed',
                    final_reward=result.get('final_reward', 0),
                )
            else:
                # 记录失败原因
                try:
                    (project_dir / "error.log").write_text(str(result), encoding="utf-8")
                except Exception:
                    pass
                crud.update_project(db_local, project_id, status='failed')
        except Exception:
            import traceback
            err_text = traceback.format_exc()
            # 写入错误日志方便排查
            try:
                project_dir.mkdir(parents=True, exist_ok=True)
                (project_dir / "error.log").write_text(err_text, encoding="utf-8")
            except Exception:
                pass
            crud.update_project(db_local, project_id, status='failed')
        finally:
            db_local.close()

