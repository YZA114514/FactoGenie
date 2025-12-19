"""
配置加载器 - 从 simulation 配置文件中提取 environment 所需的数据
负责人：张毅

从 chair_factory.json 和 chair_layout.json 读取配置并转换为环境可用的格式
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


class ConfigLoader:
    """
    从 simulation 配置文件加载工厂数据并转换为 RL 环境格式
    """
    
    def __init__(self, config_path: str, layout_path: Optional[str] = None):
        """
        初始化配置加载器
        
        Args:
            config_path: simulation配置文件路径 (如 simulation/configs/chair_factory.json)
            layout_path: 布局文件路径 (如 simulation/layouts/chair_layout.json)
                        如果为None，则从config中读取
        """
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        
        # 加载仿真配置（使用 utf-8-sig 处理 BOM）
        with open(self.config_path, 'r', encoding='utf-8-sig') as f:
            self.sim_config = json.load(f)
        
        # 确定布局文件路径
        if layout_path is None:
            # 从配置中读取相对路径
            layout_rel_path = self.sim_config.get('layout', 'layouts/chair_layout.json')
            # 相对于 simulation 目录解析
            self.layout_path = (self.config_dir.parent / layout_rel_path).resolve()
        else:
            self.layout_path = Path(layout_path)
        
        # 加载布局配置（使用 utf-8-sig 处理 BOM）
        if self.layout_path.exists():
            with open(self.layout_path, 'r', encoding='utf-8-sig') as f:
                self.layout_config = json.load(f)
        else:
            self.layout_config = None
            print(f"警告: 布局文件 {self.layout_path} 不存在，将使用默认配置")
    
    def get_factory_size(self) -> Tuple[int, int]:
        """
        获取工厂尺寸（网格大小）
        
        Returns:
            (width, height) 工厂的长和宽
        """
        if self.layout_config and 'factory' in self.layout_config:
            factory = self.layout_config['factory']
            width = factory.get('length', 90)
            height = factory.get('width', 60)
            return (width, height)
        else:
            # 默认值
            return (90, 60)
    
    def get_functional_units(self) -> List[Dict]:
        """
        从布局配置中提取功能单元信息（包括 fus 和可移动的 obstacles）
        
        Returns:
            功能单元列表，每个元素为字典:
            {
                'id': str,           # 功能单元ID (如 'rec_dock', 'station_1', 'obstacle_2')
                'name': str,         # 显示名称
                'size': (int, int),  # (长度, 宽度)
                'rotatable': bool,   # 是否可旋转
                'notch': (int, int), # 缺口尺寸 (notch_length, notch_width)
                'is_obstacle': bool, # 是否为障碍物（用于导出时区分）
            }
        
        注意：
            - fus: 全部读取
            - obstacles: 只读取 constraints.movable_obstacles 中指定的障碍物
            - constraints.fixed_obstacles 中的障碍物固定不动，不参与摆放
        """
        functional_units = []
        
        # 1. 读取所有 fus（功能单元）
        if self.layout_config and 'fus' in self.layout_config:
            for idx, fu in enumerate(self.layout_config['fus']):
                unit = {
                    'id': fu['id'],
                    'name': fu['id'],
                    'size': (fu['length'], fu['width']),
                    'rotatable': True,  # 默认可旋转
                    'notch': (fu.get('notch_length', 0), fu.get('notch_width', 0)),
                    'is_obstacle': False,  # 标记为非障碍物
                    # 保存原始配置用于后续输出
                    'buffer_in': fu.get('buffer_in', {'capacity': 100, 'initial': 0}),
                    'buffer_out': fu.get('buffer_out', {'capacity': 100, 'initial': 0}),
                    'production_rate': fu.get('production_rate', 1),
                    'processing_time': fu.get('processing_time', 0.0),
                }
                functional_units.append(unit)
        
        # 2. 从配置读取可移动障碍物列表（不再硬编码）
        movable_obstacles = set()
        if self.layout_config and 'constraints' in self.layout_config:
            movable_obstacles = set(self.layout_config['constraints'].get('movable_obstacles', []))
        
        # 如果配置中没有指定，使用空集合（不读取任何障碍物）
        # 注意：为了向后兼容，如果没有 constraints 配置，则不读取任何可移动障碍物
        
        if self.layout_config and 'obstacles' in self.layout_config:
            for obs in self.layout_config['obstacles']:
                obs_id = obs['id']
                if obs_id in movable_obstacles:
                    unit = {
                        'id': obs_id,
                        'name': obs_id,
                        'size': (obs['length'], obs['width']),
                        'rotatable': True,  # 障碍物也可旋转
                        'notch': (obs.get('notch_length', 0), obs.get('notch_width', 0)),
                        'is_obstacle': True,  # 标记为障碍物
                        # 障碍物没有生产相关属性，使用默认值
                        'buffer_in': {'capacity': 0, 'initial': 0},
                        'buffer_out': {'capacity': 0, 'initial': 0},
                        'production_rate': 0,
                        'processing_time': 0.0,
                    }
                    functional_units.append(unit)
        
        # 按面积降序排序：大单元先放置，避免空间碎片化
        # 这是装箱问题（bin packing）的经典优化策略
        functional_units.sort(key=lambda u: u['size'][0] * u['size'][1], reverse=True)
        
        # 打印放置顺序（调试用）
        print(f"[放置顺序] 按面积降序排列:")
        for i, u in enumerate(functional_units):
            area = u['size'][0] * u['size'][1]
            print(f"  {i+1}. {u['id']}: {u['size']} (面积: {area})")
        
        return functional_units
    
    def get_material_flow(self, functional_units: List[Dict]) -> np.ndarray:
        """
        从routes配置中构建物料流矩阵
        
        Args:
            functional_units: 功能单元列表
            
        Returns:
            物料流矩阵 [N x N]，元素为0或1表示是否存在物料流
        """
        N = len(functional_units)
        material_flow = np.zeros((N, N), dtype=np.float32)
        
        # 创建 id 到索引的映射
        id_to_idx = {unit['id']: i for i, unit in enumerate(functional_units)}
        
        # 从 routes 中提取物料流关系
        if 'routes' in self.sim_config:
            for route in self.sim_config['routes']:
                from_id = route['from']
                to_id = route['to']
                
                if from_id in id_to_idx and to_id in id_to_idx:
                    i = id_to_idx[from_id]
                    j = id_to_idx[to_id]
                    material_flow[i, j] = 1.0
        
        return material_flow
    
    def get_placement_constraints(self) -> Dict:
        """
        获取摆放约束规则
        
        从布局配置文件读取约束规则，支持：
        - fixed_obstacles: 固定障碍物（不参与摆放）
        - default_wall_attach: 默认必须贴墙的单元
        - fixed_positions: 固定位置约束
        - adjacency: 相邻约束
        - wall_attach: 用户自定义贴墙约束
        
        Returns:
            约束规则字典
        """
        constraints = {
            'min_distance': 0,
            'wall_units': [],           # 默认贴墙单元（从配置读取）
            'restricted_areas': [],
            'fixed_obstacles': [],      # 固定障碍物ID列表
            'fixed_positions': [],      # [{unit_id, x, y, angle}, ...]
            'adjacency': [],            # [{unit_a, unit_b, direction}, ...]
            'wall_attach': [],          # [{unit_id, wall}, ...]
        }
        
        # 从布局配置中读取自定义约束
        if self.layout_config and 'constraints' in self.layout_config:
            user_constraints = self.layout_config['constraints']
            
            # 读取固定障碍物（不参与摆放）
            if 'fixed_obstacles' in user_constraints:
                constraints['fixed_obstacles'] = user_constraints['fixed_obstacles']
            
            # 读取默认贴墙单元
            if 'default_wall_attach' in user_constraints:
                constraints['wall_units'] = user_constraints['default_wall_attach']
            
            if 'fixed_positions' in user_constraints:
                constraints['fixed_positions'] = user_constraints['fixed_positions']
            
            if 'adjacency' in user_constraints:
                constraints['adjacency'] = user_constraints['adjacency']
            
            if 'wall_attach' in user_constraints:
                constraints['wall_attach'] = user_constraints['wall_attach']
        
        return constraints
    
    def get_objective_weights(self) -> Dict[str, float]:
        """
        获取目标权重配置
        
        Returns:
            目标权重字典
        """
        # 可以从配置文件读取，或使用默认值
        weights = {
            'transportation_intensity': 0.4,
            'throughput_time': 0.3,
            'material_flow_clarity': 0.2,
            'space_utilization': 0.1
        }
        
        return weights
    
    def get_simulation_config(self) -> Dict:
        """
        获取完整的仿真配置（用于后续调用仿真）
        
        Returns:
            仿真配置字典
        """
        return self.sim_config
    
    def get_layout_template(self) -> Dict:
        """
        获取布局配置模板（用于输出时参考）
        
        Returns:
            布局配置字典
        """
        return self.layout_config if self.layout_config else {}


def load_chair_factory_config(base_dir: str = "FactoGenie/simulation") -> Dict:
    """
    便捷函数：加载椅子工厂的配置
    
    Args:
        base_dir: simulation目录的路径
        
    Returns:
        包含所有必要信息的配置字典
    """
    base_path = Path(base_dir)
    config_path = base_path / "configs" / "chair_factory.json"
    
    loader = ConfigLoader(str(config_path))
    
    functional_units = loader.get_functional_units()
    
    config = {
        'factory_size': loader.get_factory_size(),
        'functional_units': functional_units,
        'material_flow': loader.get_material_flow(functional_units),
        'placement_constraints': loader.get_placement_constraints(),
        'objective_weights': loader.get_objective_weights(),
        'config_path': str(config_path),
        'layout_path': str(loader.layout_path),
    }
    
    return config


# ====================
# 测试代码
# ====================
if __name__ == "__main__":
    print("测试配置加载器...")
    
    try:
        # 加载椅子工厂配置
        config = load_chair_factory_config()
        
        print("\n✓ 配置加载成功！")
        print(f"\n工厂尺寸: {config['factory_size']}")
        print(f"功能单元数量: {len(config['functional_units'])}")
        print("\n功能单元列表:")
        for unit in config['functional_units']:
            print(f"  - {unit['id']}: {unit['size']}")
        
        print(f"\n物料流矩阵形状: {config['material_flow'].shape}")
        print(f"物料流连接数: {np.sum(config['material_flow'] > 0)}")
        
        print("\n摆放约束:")
        constraints = config['placement_constraints']
        print(f"  最小距离: {constraints['min_distance']}")
        print(f"  必须贴墙: {constraints['wall_units']}")
        
        print("\n目标权重:")
        for key, value in config['objective_weights'].items():
            print(f"  {key}: {value}")
        
        print("\n✓ 测试完成！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

