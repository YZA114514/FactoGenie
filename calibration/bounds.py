"""
边界管理器：保存、加载、查询校准结果
"""
import hashlib
import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


class BoundsManager:
    """
    管理校准结果的存储和检索
    
    使用方法:
        manager = BoundsManager()
        
        # 检查是否需要校准
        if manager.needs_calibration(factory_config, layout_config):
            bounds = calibrator.calibrate()
            manager.save(bounds, factory_config, layout_config)
        
        # 加载已有校准
        bounds = manager.load(factory_config, layout_config)
    """
    
    def __init__(self, cache_dir: str = "data/calibrations"):
        """
        Args:
            cache_dir: 校准结果缓存目录
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _compute_hash(self, factory_config_path: str, layout_config_path: str) -> str:
        """计算配置文件的hash"""
        hash_content = ""
        
        with open(factory_config_path, 'r', encoding='utf-8') as f:
            hash_content += f.read()
        
        with open(layout_config_path, 'r', encoding='utf-8') as f:
            hash_content += f.read()
        
        return hashlib.md5(hash_content.encode()).hexdigest()[:12]
    
    def _get_cache_path(self, config_hash: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"bounds_{config_hash}.json"
    
    def needs_calibration(
        self,
        factory_config_path: str,
        layout_config_path: str,
    ) -> bool:
        """
        检查是否需要校准
        
        Returns:
            True 如果需要校准（无缓存或配置已变化）
        """
        config_hash = self._compute_hash(factory_config_path, layout_config_path)
        cache_path = self._get_cache_path(config_hash)
        return not cache_path.exists()
    
    def save(
        self,
        bounds: Dict,
        factory_config_path: str,
        layout_config_path: str,
    ) -> str:
        """
        保存校准结果
        
        Returns:
            保存的文件路径
        """
        config_hash = self._compute_hash(factory_config_path, layout_config_path)
        cache_path = self._get_cache_path(config_hash)
        
        # 添加保存时间
        bounds['_meta']['saved_at'] = datetime.now().isoformat()
        bounds['_meta']['factory_config'] = factory_config_path
        bounds['_meta']['layout_config'] = layout_config_path
        
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(bounds, f, indent=2, ensure_ascii=False)
        
        return str(cache_path)
    
    def load(
        self,
        factory_config_path: str,
        layout_config_path: str,
    ) -> Optional[Dict]:
        """
        加载校准结果
        
        Returns:
            bounds 字典，如果不存在则返回 None
        """
        config_hash = self._compute_hash(factory_config_path, layout_config_path)
        cache_path = self._get_cache_path(config_hash)
        
        if not cache_path.exists():
            return None
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_or_calibrate(
        self,
        factory_config_path: str,
        layout_config_path: str,
        n_episodes: int = 100,
        simulation_duration: float = 2000,
        throughput_target: Optional[float] = None,
        force_recalibrate: bool = False,
    ) -> Dict:
        """
        加载校准结果，如果不存在则自动校准
        
        Args:
            factory_config_path: 工厂配置路径
            layout_config_path: 布局配置路径
            n_episodes: 校准回合数
            simulation_duration: 仿真时长
            throughput_target: 吞吐量目标
            force_recalibrate: 强制重新校准
            
        Returns:
            bounds 字典
        """
        if not force_recalibrate:
            bounds = self.load(factory_config_path, layout_config_path)
            if bounds is not None:
                print(f"已加载校准缓存 (hash: {bounds['_meta']['config_hash']})")
                return bounds
        
        print("开始校准...")
        from .calibrator import Calibrator
        
        calibrator = Calibrator(
            factory_config_path=factory_config_path,
            layout_config_path=layout_config_path,
            simulation_duration=simulation_duration,
            throughput_target=throughput_target,
        )
        
        bounds = calibrator.calibrate(n_episodes=n_episodes)
        
        save_path = self.save(bounds, factory_config_path, layout_config_path)
        print(f"校准结果已保存到: {save_path}")
        
        return bounds
    
    def get_bounds_for_reward(
        self,
        factory_config_path: str,
        layout_config_path: str,
        **kwargs,
    ) -> Dict[str, tuple]:
        """
        获取用于奖励计算的边界值
        
        Returns:
            {
                'distance': (best, worst),
                'logistics': (best, worst),
                'throughput': (best, worst),
                'utilization': (best, worst),
            }
        """
        bounds = self.load_or_calibrate(
            factory_config_path=factory_config_path,
            layout_config_path=layout_config_path,
            **kwargs,
        )
        
        result = {}
        for metric in ['distance', 'logistics', 'throughput', 'utilization']:
            if metric in bounds:
                result[metric] = (bounds[metric]['best'], bounds[metric]['worst'])
        
        return result
    
    def list_cached(self) -> list:
        """列出所有缓存的校准结果"""
        results = []
        for path in self.cache_dir.glob("bounds_*.json"):
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            meta = data.get('_meta', {})
            results.append({
                'path': str(path),
                'hash': meta.get('config_hash'),
                'saved_at': meta.get('saved_at'),
                'factory_config': meta.get('factory_config'),
                'layout_config': meta.get('layout_config'),
            })
        return results
    
    def clear_cache(self):
        """清空所有校准缓存"""
        for path in self.cache_dir.glob("bounds_*.json"):
            path.unlink()
        print("已清空校准缓存")


# 全局单例
_default_manager: Optional[BoundsManager] = None


def get_bounds_manager() -> BoundsManager:
    """获取默认的边界管理器"""
    global _default_manager
    if _default_manager is None:
        _default_manager = BoundsManager()
    return _default_manager

