"""
结果查询服务
"""
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from pathlib import Path
import json

from db import crud
from db.models import Checkpoint


class ResultService:
    """结果查询服务"""
    
    def __init__(self, db: Session):
        self.db = db
        self.data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    def get_layouts(
        self,
        project_id: str,
        page: int = 1,
        size: int = 20,
    ) -> Dict:
        """获取布局历史"""
        checkpoints = crud.get_checkpoints(self.db, project_id)
        
        total = len(checkpoints)
        start = (page - 1) * size
        end = start + size
        
        layouts = []
        for cp in checkpoints[start:end]:
            layout_data = None
            if cp.layout_path:
                layout_path_obj = Path(cp.layout_path)
                if layout_path_obj.exists():
                    try:
                        with open(layout_path_obj, 'r', encoding='utf-8') as f:
                            layout_data = json.load(f)
                    except Exception as e:
                        print(f"Warning: Failed to load layout from {cp.layout_path}: {e}")
                else:
                    print(f"Warning: Layout file not found: {cp.layout_path}")
            
            layouts.append({
                'episode': cp.episode,
                'reward': cp.reward,
                'is_best': cp.is_best,
                'created_at': cp.created_at.isoformat(),
                'layout': layout_data,
                'metrics': cp.metrics_snapshot,
                'layout_path': cp.layout_path,  # 添加路径信息用于调试
            })
        
        return {
            'total': total,
            'page': page,
            'size': size,
            'layouts': layouts,
        }
    
    def get_best_layout(self, project_id: str) -> Optional[Dict]:
        """获取最佳布局"""
        checkpoint = crud.get_best_checkpoint(self.db, project_id)
        if not checkpoint:
            # 如果没有标记best，fallback到最高reward
            cps = crud.get_checkpoints(self.db, project_id)
            if not cps:
                print(f"[DEBUG] 项目 {project_id} 没有checkpoints")
                return None
            checkpoint = max(cps, key=lambda c: c.reward or float("-inf"))
            print(f"[DEBUG] 项目 {project_id} 使用reward最高的checkpoint: ep{checkpoint.episode}")
        else:
            print(f"[DEBUG] 项目 {project_id} 找到best checkpoint: ep{checkpoint.episode}")
        
        layout_data = None
        layout_path = checkpoint.layout_path
        print(f"[DEBUG] checkpoint.layout_path = {layout_path}")
        
        if layout_path and Path(layout_path).exists():
            with open(layout_path, 'r', encoding='utf-8') as f:
                layout_data = json.load(f)
            print(f"[DEBUG] 成功读取布局文件: {layout_path}")
        else:
            print(f"[WARN] 布局文件不存在或路径为空: {layout_path}")
        
        return {
            'episode': checkpoint.episode,
            'reward': checkpoint.reward,
            'created_at': checkpoint.created_at.isoformat(),
            'layout': layout_data,
            'metrics': checkpoint.metrics_snapshot,
            'is_best': checkpoint.is_best,
        }
    
    def get_metrics_curve(
        self,
        project_id: str,
        metric: str = 'reward',
        start: int = 0,
        end: int = None,
    ) -> Dict:
        """获取指标曲线数据"""
        values = crud.get_metrics_history(
            self.db,
            project_id,
            metric_name=metric,
            start_episode=start,
            end_episode=end,
        )
        
        return {
            'metric': metric,
            'values': values,
        }
    
    def get_checkpoint_detail(
        self,
        project_id: str,
        episode: int,
    ) -> Optional[Dict]:
        """获取特定检查点的详细信息"""
        checkpoints = crud.get_checkpoints(self.db, project_id)
        
        for cp in checkpoints:
            if cp.episode == episode:
                layout_data = None
                if cp.layout_path and Path(cp.layout_path).exists():
                    with open(cp.layout_path, 'r') as f:
                        layout_data = json.load(f)
                
                return {
                    'episode': cp.episode,
                    'reward': cp.reward,
                    'is_best': cp.is_best,
                    'created_at': cp.created_at.isoformat(),
                    'model_path': cp.model_path,
                    'layout_path': cp.layout_path,
                    'layout': layout_data,
                    'metrics': cp.metrics_snapshot,
                }
        
        return None

