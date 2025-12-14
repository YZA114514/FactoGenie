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
    
    def start_training(self, project_id: str) -> Dict:
        """
        启动训练任务
        
        注意：实际的训练应该在后台进程或Celery任务中运行
        这里只是更新状态并返回
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

        # 启动后台线程（本地同步训练；如需分布式可替换为 Celery）
        t = threading.Thread(target=self._run_training_sync, args=(project_id, stop_event), daemon=True)
        t.start()
        self._running_tasks[project_id] = {"thread": t}
        
        return {
            'success': True,
            'project_id': project_id,
            'status': 'running',
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
        
        # TODO: 取消Celery任务
        # if project.celery_task_id:
        #     celery_app.control.revoke(project.celery_task_id, terminate=True)

        crud.update_project(self.db, project_id, status='stopped')
        
        return {
            'success': True,
            'project_id': project_id,
            'status': 'stopped',
        }
    
    def pause_training(self, project_id: str) -> Dict:
        """暂停训练"""
        project = crud.get_project(self.db, project_id)
        if not project:
            return {'success': False, 'error': 'Project not found'}
        
        crud.update_project(self.db, project_id, status='paused')
        
        return {
            'success': True,
            'project_id': project_id,
            'status': 'paused',
        }
    
    def resume_training(self, project_id: str) -> Dict:
        """恢复训练"""
        project = crud.get_project(self.db, project_id)
        if not project:
            return {'success': False, 'error': 'Project not found'}
        
        crud.update_project(self.db, project_id, status='running')
        
        return {
            'success': True,
            'project_id': project_id,
            'status': 'running',
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
            # 使用项目参数或默认参数的浅拷贝，避免修改全局
            training_params = (project.training_params or {}).copy()
            if not training_params:
                try:
                    from backend import config as backend_config  # type: ignore
                    training_params = backend_config.DEFAULT_TRAINING_PARAMS.copy()
                except Exception:
                    training_params = {}
                # 压缩步数，避免调试时训练过长
                training_params["total_steps"] = min(training_params.get("total_steps", 50000), 2000)

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

