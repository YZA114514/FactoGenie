"""
指标校准器：通过随机摆放估计指标的合理范围
"""
import hashlib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.factory_environment import LayoutEnvironment
from simulation.interface import compute_metrics


class Calibrator:
    """
    通过随机摆放校准指标边界
    
    使用方法:
        calibrator = Calibrator(factory_config_path, layout_config_path)
        bounds = calibrator.calibrate(n_episodes=100)
    """
    
    def __init__(
        self,
        factory_config_path: str,
        layout_config_path: str,
        simulation_duration: float = 2000,
        throughput_target: Optional[float] = None,
    ):
        """
        Args:
            factory_config_path: 工厂配置文件路径
            layout_config_path: 布局配置文件路径
            simulation_duration: 仿真时长
            throughput_target: 用户指定的吞吐量目标（可选）
        """
        self.factory_config_path = factory_config_path
        self.layout_config_path = layout_config_path
        self.simulation_duration = simulation_duration
        self.throughput_target = throughput_target
        
        # 计算配置hash用于缓存
        self.config_hash = self._compute_config_hash()
    
    def _compute_config_hash(self) -> str:
        """计算配置文件的hash值"""
        hash_content = ""
        
        # 读取工厂配置
        with open(self.factory_config_path, 'r', encoding='utf-8') as f:
            hash_content += f.read()
        
        # 读取布局配置
        with open(self.layout_config_path, 'r', encoding='utf-8') as f:
            hash_content += f.read()
        
        # 添加仿真时长
        hash_content += str(self.simulation_duration)
        
        return hashlib.md5(hash_content.encode()).hexdigest()[:12]
    
    def calibrate(
        self,
        n_episodes: int = 100,
        best_percentile: float = 15,
        worst_percentile: float = 90,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        运行随机摆放校准指标边界
        
        Args:
            n_episodes: 随机摆放的回合数
            best_percentile: 最优值的分位线（如15表示P15）
            worst_percentile: 最差值的分位线（如90表示P90）
            verbose: 是否显示进度
            
        Returns:
            bounds: 各指标的边界值
            {
                'distance': {'best': float, 'worst': float},
                'logistics': {'best': float, 'worst': float},
                'throughput': {'best': float, 'worst': float},
                'utilization': {'best': float, 'worst': float},
            }
        """
        # 创建环境（随机摆放顺序）
        env = LayoutEnvironment.from_config(
            layout_path=self.layout_config_path,
            config_path=self.factory_config_path,
            use_simulation=False,  # 校准时先不用仿真计算奖励
            placement_order='random',
        )
        
        # 收集指标
        metrics_list = {
            'distance': [],
            'logistics': [],
            'throughput': [],
            'utilization': [],
        }
        
        iterator = range(n_episodes)
        if verbose:
            iterator = tqdm(iterator, desc="校准中")
        
        for _ in iterator:
            # 随机摆放一个完整布局
            env.reset()
            done = False
            
            while not done:
                # 随机选择动作
                valid_actions = env._get_valid_actions()
                if len(valid_actions) == 0:
                    break
                action = np.random.choice(valid_actions)
                _, _, done, _, _ = env.step(action)
            
            # 运行仿真获取指标
            layout_json = env._generate_layout_json()
            try:
                sim_results = compute_metrics(
                    layout_json,
                    self.factory_config_path,
                    duration=self.simulation_duration,
                    detail=False,
                )
                
                # 提取指标
                metrics_list['distance'].append(sim_results.get('average_route_distance', 0))
                metrics_list['logistics'].append(sim_results.get('total_logistics_intensity', 0))
                metrics_list['throughput'].append(sim_results.get('finished_goods', 0))
                
                # 计算平均利用率
                station_util = sim_results.get('station_utilization', {})
                if station_util:
                    avg_util = sum(station_util.values()) / len(station_util)
                else:
                    avg_util = 0
                metrics_list['utilization'].append(avg_util)
                
            except Exception as e:
                if verbose:
                    print(f"仿真失败: {e}")
                continue
        
        # 计算分位数边界
        bounds = {}
        
        for metric_name, values in metrics_list.items():
            if len(values) == 0:
                continue
            
            values = np.array(values)
            
            if metric_name in ['distance', 'logistics']:
                # 越小越好：best=低分位，worst=高分位
                best = np.percentile(values, best_percentile)
                worst = np.percentile(values, worst_percentile)
            else:
                # 越大越好：best=高分位，worst=低分位
                best = np.percentile(values, 100 - best_percentile)
                worst = np.percentile(values, 100 - worst_percentile)
            
            bounds[metric_name] = {
                'best': float(best),
                'worst': float(worst),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
        
        # 如果用户指定了吞吐量目标，覆盖best值
        if self.throughput_target is not None and 'throughput' in bounds:
            bounds['throughput']['best'] = self.throughput_target
        
        # 添加元信息
        bounds['_meta'] = {
            'config_hash': self.config_hash,
            'n_episodes': n_episodes,
            'best_percentile': best_percentile,
            'worst_percentile': worst_percentile,
            'simulation_duration': self.simulation_duration,
            'throughput_target': self.throughput_target,
        }
        
        if verbose:
            self._print_bounds(bounds)
        
        return bounds
    
    def _print_bounds(self, bounds: Dict):
        """打印校准结果"""
        print("\n" + "=" * 50)
        print("校准结果")
        print("=" * 50)
        
        for metric, values in bounds.items():
            if metric == '_meta':
                continue
            print(f"\n{metric}:")
            print(f"  best:  {values['best']:.4f}")
            print(f"  worst: {values['worst']:.4f}")
            print(f"  range: [{values['min']:.4f}, {values['max']:.4f}]")
            print(f"  mean:  {values['mean']:.4f} ± {values['std']:.4f}")


def calibrate_from_config(
    factory_config_path: str,
    layout_config_path: str,
    n_episodes: int = 100,
    simulation_duration: float = 2000,
    throughput_target: Optional[float] = None,
    save_path: Optional[str] = None,
) -> Dict:
    """
    便捷函数：校准并可选保存结果
    
    Args:
        factory_config_path: 工厂配置路径
        layout_config_path: 布局配置路径
        n_episodes: 随机回合数
        simulation_duration: 仿真时长
        throughput_target: 吞吐量目标
        save_path: 保存路径（可选）
        
    Returns:
        bounds: 校准结果
    """
    calibrator = Calibrator(
        factory_config_path=factory_config_path,
        layout_config_path=layout_config_path,
        simulation_duration=simulation_duration,
        throughput_target=throughput_target,
    )
    
    bounds = calibrator.calibrate(n_episodes=n_episodes)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(bounds, f, indent=2, ensure_ascii=False)
        print(f"\n校准结果已保存到: {save_path}")
    
    return bounds


if __name__ == "__main__":
    # 测试校准
    import argparse
    
    parser = argparse.ArgumentParser(description="指标校准工具")
    parser.add_argument("--factory", type=str, default="simulation/configs/chair_factory.json")
    parser.add_argument("--layout", type=str, default="simulation/layouts/chair_layout.json")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--duration", type=float, default=2000)
    parser.add_argument("--throughput_target", type=float, default=None)
    parser.add_argument("--save", type=str, default=None)
    
    args = parser.parse_args()
    
    bounds = calibrate_from_config(
        factory_config_path=args.factory,
        layout_config_path=args.layout,
        n_episodes=args.episodes,
        simulation_duration=args.duration,
        throughput_target=args.throughput_target,
        save_path=args.save,
    )

