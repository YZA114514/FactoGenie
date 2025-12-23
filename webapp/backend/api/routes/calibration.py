"""
校准 API
"""
from fastapi import APIRouter, Depends, BackgroundTasks
from pydantic import BaseModel
from typing import Optional, Dict
from pathlib import Path
import hashlib
import json

router = APIRouter()


class CalibrationRequest(BaseModel):
    factory_config: dict
    layout_config: dict
    n_episodes: int = 100  # 默认100个episode，确保校准准确性
    simulation_duration: float = 2000
    throughput_target: Optional[float] = None
    force_recalibrate: bool = False  # 强制重新校准，忽略缓存


def compute_config_hash(factory_config: dict, layout_config: dict) -> str:
    """计算配置哈希"""
    content = json.dumps(factory_config, sort_keys=True) + json.dumps(layout_config, sort_keys=True)
    return hashlib.md5(content.encode()).hexdigest()[:12]


@router.post("/run")
async def run_calibration(
    request: CalibrationRequest,
    background_tasks: BackgroundTasks,
):
    """
    运行校准（同步执行，可能需要几分钟）
    
    注意：校准过程可能需要较长时间（50个episode × 仿真时间），
    建议前端设置较长的超时时间（至少5分钟）
    """
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        # 保存临时配置文件
        temp_dir = Path(__file__).parent.parent.parent.parent / "data" / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        config_hash = compute_config_hash(request.factory_config, request.layout_config)
        
        factory_path = temp_dir / f"factory_{config_hash}.json"
        layout_path = temp_dir / f"layout_{config_hash}.json"
        
        with open(factory_path, 'w') as f:
            json.dump(request.factory_config, f)
        
        with open(layout_path, 'w') as f:
            json.dump(request.layout_config, f)
        
        # 运行校准
        from calibration.bounds import BoundsManager
        
        print(f"[校准API] 开始校准，episodes={request.n_episodes}, duration={request.simulation_duration}")
        
        manager = BoundsManager()
        bounds = manager.load_or_calibrate(
            factory_config_path=str(factory_path),
            layout_config_path=str(layout_path),
            n_episodes=request.n_episodes,
            simulation_duration=request.simulation_duration,
            throughput_target=request.throughput_target,
            force_recalibrate=request.force_recalibrate,  # 支持强制重新校准
        )
        
        print(f"[校准API] 校准完成")
        
        return {
            "code": 0,
            "data": {
                "config_hash": config_hash,
                "bounds": {
                    "distance": bounds.get('distance', {}),
                    "logistics": bounds.get('logistics', {}),
                    "throughput": bounds.get('throughput', {}),
                    "utilization": bounds.get('utilization', {}),
                }
            }
        }
        
    except Exception as e:
        import traceback
        print(f"[校准API] 校准失败: {e}")
        traceback.print_exc()
        return {
            "code": 5000,
            "message": str(e),
            "data": None
        }


@router.get("/cache")
async def get_calibration_cache(factory_hash: str):
    """查询校准缓存"""
    from pathlib import Path
    
    cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "calibrations"
    cache_path = cache_dir / f"bounds_{factory_hash}.json"
    
    if cache_path.exists():
        with open(cache_path, 'r') as f:
            bounds = json.load(f)
        
        return {
            "code": 0,
            "data": {
                "exists": True,
                "bounds": bounds,
                "created_at": bounds.get('_meta', {}).get('saved_at'),
            }
        }
    
    return {
        "code": 0,
        "data": {
            "exists": False,
            "bounds": None,
        }
    }


@router.delete("/cache/{config_hash}")
async def delete_calibration_cache(config_hash: str):
    """删除校准缓存"""
    from pathlib import Path
    
    cache_dir = Path(__file__).parent.parent.parent.parent / "data" / "calibrations"
    cache_path = cache_dir / f"bounds_{config_hash}.json"
    
    if cache_path.exists():
        cache_path.unlink()
        return {"code": 0, "message": "deleted"}
    
    return {"code": 1002, "message": "Cache not found", "data": None}








