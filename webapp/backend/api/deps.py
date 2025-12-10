"""
依赖注入
"""
from fastapi import Depends
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..services.project_service import ProjectService
from ..services.training_service import TrainingService
from ..services.result_service import ResultService


def get_project_service(db: Session = Depends(get_db)) -> ProjectService:
    return ProjectService(db)


def get_training_service(db: Session = Depends(get_db)) -> TrainingService:
    return TrainingService(db)


def get_result_service(db: Session = Depends(get_db)) -> ResultService:
    return ResultService(db)

