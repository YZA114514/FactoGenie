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
        
        # 更新状态
        crud.update_project(self.db, project_id, status='running')
        
        # 创建停止事件
        self._stop_events[project_id] = threading.Event()
        
        # TODO: 启动Celery任务
        # task = run_training_task.delay(project_id)
        # crud.update_project(self.db, project_id, celery_task_id=task.id)
        
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

