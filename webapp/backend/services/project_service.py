"""
项目管理服务
"""
from sqlalchemy.orm import Session
from typing import Dict, List, Optional
from pathlib import Path
import shutil
import json

from db import crud
from db.models import Project


class ProjectService:
    """项目管理服务"""
    
    def __init__(self, db: Session):
        self.db = db
        self.data_dir = Path(__file__).parent.parent.parent.parent / "data"
    
    def create_project(
        self,
        name: str,
        factory_config: Dict,
        layout_config: Dict,
        constraints: Dict = None,
        training_params: Dict = None,
        description: str = None,
    ) -> Project:
        """创建新项目"""
        project = crud.create_project(
            self.db,
            name=name,
            factory_config=factory_config,
            layout_config=layout_config,
            constraints=constraints,
            training_params=training_params,
            description=description,
        )
        
        # 创建项目数据目录
        project_dir = self.data_dir / "projects" / project.id
        project_dir.mkdir(parents=True, exist_ok=True)
        (project_dir / "checkpoints").mkdir(exist_ok=True)
        (project_dir / "layouts").mkdir(exist_ok=True)
        
        # 保存配置文件
        with open(project_dir / "factory_config.json", 'w') as f:
            json.dump(factory_config, f, indent=2)
        
        with open(project_dir / "layout_config.json", 'w') as f:
            json.dump(layout_config, f, indent=2)
        
        if constraints:
            with open(project_dir / "constraints.json", 'w') as f:
                json.dump(constraints, f, indent=2)
        
        if training_params:
            with open(project_dir / "training_params.json", 'w') as f:
                json.dump(training_params, f, indent=2)
        
        return project
    
    def get_project(self, project_id: str) -> Optional[Project]:
        """获取项目详情"""
        return crud.get_project(self.db, project_id)
    
    def list_projects(
        self,
        page: int = 1,
        size: int = 20,
        status: str = None,
    ) -> Dict:
        """获取项目列表"""
        skip = (page - 1) * size
        projects = crud.get_projects(self.db, skip=skip, limit=size, status=status)
        total = crud.count_projects(self.db, status=status)
        
        return {
            'total': total,
            'page': page,
            'size': size,
            'projects': projects,
        }
    
    def update_project(self, project_id: str, **kwargs) -> Optional[Project]:
        """更新项目"""
        return crud.update_project(self.db, project_id, **kwargs)
    
    def delete_project(self, project_id: str) -> bool:
        """删除项目"""
        # 删除项目目录
        project_dir = self.data_dir / "projects" / project_id
        if project_dir.exists():
            shutil.rmtree(project_dir)
        
        return crud.delete_project(self.db, project_id)
    
    def get_project_dir(self, project_id: str) -> Path:
        """获取项目目录路径"""
        return self.data_dir / "projects" / project_id

