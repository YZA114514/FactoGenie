"""
配置管理 API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import json

router = APIRouter()


# ========== 请求/响应模型 ==========

class FactoryConfigResponse(BaseModel):
    config: dict
    validation: dict


class LayoutConfigResponse(BaseModel):
    config: dict
    validation: dict


class SaveConfigResponse(BaseModel):
    config_id: str


# ========== 路由 ==========

@router.post("/factory/upload", response_model=dict)
async def upload_factory_config(file: UploadFile = File(...)):
    """上传工厂配置文件"""
    try:
        content = await file.read()
        config = json.loads(content)
        # TODO: 验证配置
        validation = {"valid": True, "errors": []}
        return {
            "code": 0,
            "data": {
                "config": config,
                "validation": validation
            }
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")


@router.post("/factory", response_model=dict)
async def save_factory_config(config: dict):
    """保存工厂配置"""
    # TODO: 保存到数据库/文件
    config_id = "factory_001"  # 临时ID
    return {
        "code": 0,
        "data": {"config_id": config_id}
    }


@router.post("/layout/upload", response_model=dict)
async def upload_layout_config(file: UploadFile = File(...)):
    """上传布局配置文件"""
    try:
        content = await file.read()
        config = json.loads(content)
        # TODO: 验证配置
        validation = {"valid": True, "errors": []}
        return {
            "code": 0,
            "data": {
                "config": config,
                "validation": validation
            }
        }
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")


@router.post("/layout", response_model=dict)
async def save_layout_config(config: dict):
    """保存布局配置"""
    # TODO: 保存到数据库/文件
    config_id = "layout_001"  # 临时ID
    return {
        "code": 0,
        "data": {"config_id": config_id}
    }


@router.post("/constraints", response_model=dict)
async def save_constraints(constraints: dict):
    """保存约束配置"""
    # TODO: 保存到数据库/文件
    constraints_id = "constraints_001"  # 临时ID
    return {
        "code": 0,
        "data": {"constraints_id": constraints_id}
    }

