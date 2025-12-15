"""
配置管理 API
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json

router = APIRouter()


# ========== 验证函数 ==========

def validate_factory_config(config: dict) -> dict:
    """验证工厂配置"""
    errors = []
    
    # 检查必需字段
    required_fields = ['routes', 'assemblies', 'summary', 'transporters']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # 检查 routes
    if 'routes' in config:
        for i, route in enumerate(config['routes']):
            if 'from' not in route or 'to' not in route:
                errors.append(f"Route {i}: missing 'from' or 'to'")
    
    # 检查 assemblies
    if 'assemblies' in config:
        for i, asm in enumerate(config['assemblies']):
            if 'station' not in asm or 'inputs' not in asm or 'output' not in asm:
                errors.append(f"Assembly {i}: missing required fields")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_layout_config(config: dict) -> dict:
    """验证布局配置"""
    errors = []
    
    # 检查 canvas
    if 'canvas' not in config:
        errors.append("Missing 'canvas' field")
    elif 'width' not in config['canvas'] or 'height' not in config['canvas']:
        errors.append("Canvas missing 'width' or 'height'")
    
    # 检查 fus
    if 'fus' not in config:
        errors.append("Missing 'fus' field")
    else:
        for i, fu in enumerate(config['fus']):
            if 'id' not in fu:
                errors.append(f"Functional unit {i}: missing 'id'")
            if 'width' not in fu or 'height' not in fu:
                errors.append(f"Functional unit {i}: missing dimensions")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


def validate_constraints(constraints: dict) -> dict:
    """验证约束配置"""
    errors = []
    
    # 检查固定位置约束
    if 'fixed_positions' in constraints:
        for i, fp in enumerate(constraints['fixed_positions']):
            if 'unit_id' not in fp or 'x' not in fp or 'y' not in fp:
                errors.append(f"Fixed position {i}: missing required fields")
    
    # 检查相邻约束
    if 'adjacency' in constraints:
        for i, adj in enumerate(constraints['adjacency']):
            if 'unit_a' not in adj or 'unit_b' not in adj:
                errors.append(f"Adjacency {i}: missing unit_a or unit_b")
    
    # 检查贴墙约束
    if 'wall_attach' in constraints:
        valid_walls = {'top', 'bottom', 'left', 'right'}
        for i, wa in enumerate(constraints['wall_attach']):
            if 'unit_id' not in wa or 'wall' not in wa:
                errors.append(f"Wall attach {i}: missing required fields")
            elif wa.get('wall') not in valid_walls:
                errors.append(f"Wall attach {i}: invalid wall value")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }


# ========== 路由 ==========

@router.post("/factory")
async def save_factory_config(config: dict):
    """保存工厂配置（当前占位实现：校验后直接回传）"""
    validation = validate_factory_config(config)
    return {
        "code": 0 if validation["valid"] else 1003,
        "data": {"config_id": "factory_placeholder", "validation": validation},
        "message": None if validation["valid"] else "validation failed",
    }

@router.post("/factory/upload")
async def upload_factory_config(file: UploadFile = File(...)):
    """上传工厂配置文件"""
    try:
        content = await file.read()
        config = json.loads(content)
        validation = validate_factory_config(config)
        return {
            "code": 0,
            "data": {
                "config": config,
                "validation": validation
            }
        }
    except json.JSONDecodeError:
        return {
            "code": 1003,
            "message": "Invalid JSON file",
            "data": None
        }


@router.post("/factory/validate")
async def validate_factory(config: dict):
    """验证工厂配置（不保存）"""
    validation = validate_factory_config(config)
    return {
        "code": 0,
        "data": {"validation": validation}
    }

@router.post("/layout")
async def save_layout_config(config: dict):
    """保存布局配置（当前占位实现：校验后直接回传）"""
    validation = validate_layout_config(config)
    return {
        "code": 0 if validation["valid"] else 1003,
        "data": {"config_id": "layout_placeholder", "validation": validation},
        "message": None if validation["valid"] else "validation failed",
    }


@router.post("/layout/upload")
async def upload_layout_config(file: UploadFile = File(...)):
    """上传布局配置文件"""
    try:
        content = await file.read()
        config = json.loads(content)
        validation = validate_layout_config(config)
        return {
            "code": 0,
            "data": {
                "config": config,
                "validation": validation
            }
        }
    except json.JSONDecodeError:
        return {
            "code": 1003,
            "message": "Invalid JSON file",
            "data": None
        }


@router.post("/layout/validate")
async def validate_layout(config: dict):
    """验证布局配置（不保存）"""
    validation = validate_layout_config(config)
    return {
        "code": 0,
        "data": {"validation": validation}
    }


@router.post("/constraints/validate")
async def validate_constraints_api(constraints: dict):
    """验证约束配置"""
    validation = validate_constraints(constraints)
    return {
        "code": 0,
        "data": {"validation": validation}
    }

