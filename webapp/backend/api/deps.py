"""
依赖注入
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import Depends
from sqlalchemy.orm import Session

from db.database import get_db
from services.project_service import ProjectService
from services.training_service import TrainingService
from services.result_service import ResultService


def get_project_service(db: Session = Depends(get_db)) -> ProjectService:
    return ProjectService(db)


def get_training_service(db: Session = Depends(get_db)) -> TrainingService:
    return TrainingService(db)


def get_result_service(db: Session = Depends(get_db)) -> ResultService:
    return ResultService(db)

