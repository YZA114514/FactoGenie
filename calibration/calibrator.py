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


# 默认指标边界（基于椅子工厂配置的实验数据）
# 
# 计算方法：
# - SLP专家布局指标作为15%分位线（即SLP在指标区间的15%位置）
# - 574次随机摆放的最差值作为90%分位线
# - 通过线性外推计算0%（最优）和100%（最差）边界
#
# 外推公式（对于越小越好的指标）：
#   best = (6 * SLP_value - random_worst) / 5
#   worst = (random_worst - 0.10 * best) / 0.90
#
# 数据来源：
# - SLP: analysis/SLP_layout_metrics.csv (第9行/第2行)
# - Random: analysis/random_run/summary.csv (574次随机摆放)
#
DEFAULT_METRIC_BOUNDS = {
    'distance': {
        # SLP=10.625 at 15%, random_worst=24.875 at 90%
        # best = (6*10.625 - 24.875)/5 = 7.775
        # worst = (24.875 - 0.1*7.775)/0.9 = 26.78
        'best': 9,
        'worst': 29,
    },
    'logistics': {
        # 使用SLP第2行(7单元配置)与随机摆放(7单元)匹配
        # SLP=232.0 at 15%, random_worst=829.0 at 90%
        # best = (6*232 - 829)/5 = 112.6
        # worst = (829 - 0.1*112.6)/0.9 = 908.6
        'best': 3800,
        'worst': 13000,
    },
    'throughput': {
        # 吞吐量有物理上限(400)，SLP=400已达最优
        # best固定为400（仿真最大产量）
        # random_worst=121 at 90%: worst = 400 - (400-121)/0.9 * 0.1 = 369
        # 实际计算: (400-121)/(400-worst)=0.9 → worst=90
        'best': 400.0,
        'worst': 120.0,
    },
    'utilization': {
        # 使用SLP第2行(与随机摆放配置匹配)
        # SLP=0.0608 at 15%, random_worst=0.0247 at 90%
        # best = (6*0.0608 - 0.0247)/5 = 0.068
        # worst = 0.068 - (0.068-0.0247)/0.9 = 0.020
        'best': 0.8,
        'worst': 0.3,
    },
}


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
        best_percentile: float = 30,
        worst_percentile: float = 90,
        verbose: bool = True,
    ) -> Dict[str, Dict[str, float]]:
        """
        运行随机摆放校准指标边界
        
        Args:
            n_episodes: 随机摆放的回合数
            best_percentile: 最优值的分位线（如30表示P30，即最优30%作为best值）
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
                valid_actions = env.get_valid_actions()  # 使用公共方法，不是私有方法
                if len(valid_actions) == 0:
                    break
                # 从有效动作列表中选择一个（valid_actions 是字典列表）
                action_idx = np.random.randint(len(valid_actions))
                action = valid_actions[action_idx]
                _, _, done, _ = env.step(action)  # step() 返回 4 个值：state, reward, done, info
            
            # 检查是否所有单元都已放置（只有完整布局才进行仿真）
            num_placed = len(env.placed_units)
            num_total = env.num_units
            if num_placed < num_total:
                # 提前终止，跳过此 episode（未完成布局会导致仿真错误）
                continue
            
            # 运行仿真获取指标
            # 使用 LayoutExporter 生成布局 JSON 字典
            from environment.layout_exporter import LayoutExporter
            import tempfile
            import json
            
            # 获取布局模板（从环境内部获取）
            layout_template = env.layout_template if hasattr(env, 'layout_template') else None
            if not layout_template:
                if verbose:
                    print(f"警告: 无法获取布局模板，跳过此episode")
                continue
            
            exporter = LayoutExporter(layout_template, None)  # 不需要输出路径，只生成字典
            layout_json = exporter.export_layout_dict(
                env.placed_units,
                env.functional_units
            )
            
            # 将布局字典写入临时文件
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                json.dump(layout_json, tmp_file, indent=2, ensure_ascii=False)
                temp_layout_path = tmp_file.name
            
            try:
                # compute_metrics 的参数顺序：config_path, duration, layout_path=...
                result = compute_metrics(
                    self.factory_config_path,
                    self.simulation_duration,
                    layout_path=temp_layout_path,
                    detail=False,
                )
                
                # 从返回结果中提取指标
                summary = result.get('summary', {})
                static = summary.get('static', {})
                dynamic = summary.get('dynamic', {})
                
                metrics_list['distance'].append(static.get('average_route_distance', 0))
                metrics_list['logistics'].append(static.get('total_logistics_intensity', 0))
                metrics_list['throughput'].append(dynamic.get('finished_goods', 0))
                
                # 计算平均利用率
                station_util = dynamic.get('station_utilization', {})
                if station_util:
                    # station_util 是字典，每个值是一个包含 'utilization' 的字典
                    util_values = [v.get('utilization', 0) if isinstance(v, dict) else v for v in station_util.values()]
                    avg_util = sum(util_values) / len(util_values) if util_values else 0
                else:
                    avg_util = 0
                metrics_list['utilization'].append(avg_util)
                
            except Exception as e:
                if verbose:
                    print(f"仿真失败: {e}")
                continue
            finally:
                # 清理临时文件
                try:
                    Path(temp_layout_path).unlink()
                except:
                    pass
        
        # 计算分位数边界
        bounds = {}
        
        # 检查是否有足够的成功 episode
        successful_episodes = len(metrics_list['distance'])
        if successful_episodes == 0:
            if verbose:
                print(f"\n警告: 没有成功完成的 episode！当前配置可能导致所有布局无法完整放置。")
                print(f"建议：检查工厂尺寸和功能单元尺寸是否匹配。")
            return {}
        elif successful_episodes < n_episodes * 0.1:
            if verbose:
                print(f"\n警告: 只有 {successful_episodes}/{n_episodes} 个 episode 成功完成布局。")
                print(f"校准结果可能不够准确，建议增加 episode 数量或检查配置。")
        
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
        
        # 吞吐量的最优值始终固定为400（根据仿真配置：1天=20000时间单位，产能400个/天）
        # 不参与校准更新
        if 'throughput' in bounds:
            bounds['throughput']['best'] = 400.0
        
        # 如果用户指定了吞吐量目标，覆盖best值（但通常不建议）
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


def get_default_bounds() -> Dict[str, Tuple[float, float]]:
    """
    获取默认指标边界（不运行校准，直接使用预设值）
    
    Returns:
        bounds: 字典 {metric_name: (best, worst)}
    """
    return {
        'distance': (DEFAULT_METRIC_BOUNDS['distance']['best'], DEFAULT_METRIC_BOUNDS['distance']['worst']),
        'logistics': (DEFAULT_METRIC_BOUNDS['logistics']['best'], DEFAULT_METRIC_BOUNDS['logistics']['worst']),
        'throughput': (DEFAULT_METRIC_BOUNDS['throughput']['best'], DEFAULT_METRIC_BOUNDS['throughput']['worst']),
        'utilization': (DEFAULT_METRIC_BOUNDS['utilization']['best'], DEFAULT_METRIC_BOUNDS['utilization']['worst']),
    }


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

