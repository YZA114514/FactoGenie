"""
奖励函数模块
负责人：张毅

实现多目标奖励计算，参考文献中的公式
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
# 注意：如果需要高级距离计算，可以取消下面的注释
# from scipy.spatial.distance import cdist


class RewardCalculator:
    """
    多目标奖励计算器
    
    支持的目标：
    1. 运输强度 (Transportation Intensity) - 解析
    2. 物料流清晰度 (Material Flow Clarity) - 解析
    3. 生产周期时间 (Throughput Time) - 仿真
    4. 功能单元利用率 (Utilization) - 仿真
    """
    
    def __init__(
        self,
        layout_grid: np.ndarray,
        placed_units: List[Tuple],
        material_flow: np.ndarray,
        functional_units: List[Dict],
        objective_weights: Dict[str, float]
    ):
        """
        初始化奖励计算器
        
        Args:
            layout_grid: 布局网格
            placed_units: 已放置单元列表 [(unit_id, x, y, rotation), ...]
            material_flow: 物料流矩阵 [N, N]
            functional_units: 功能单元配置
            objective_weights: 目标权重
        """
        self.layout_grid = layout_grid
        self.placed_units = placed_units
        self.material_flow = material_flow
        self.functional_units = functional_units
        self.objective_weights = objective_weights
        
        # 归一化权重
        total_weight = sum(objective_weights.values())
        if total_weight > 0:
            self.objective_weights = {
                k: v/total_weight for k, v in objective_weights.items()
            }
    
    def calculate_total_reward(
        self,
        current_unit_idx: int,
        run_simulation: bool = False
    ) -> float:
        """
        计算总奖励
        
        Args:
            current_unit_idx: 当前放置的单元索引
            run_simulation: 是否运行仿真
            
        Returns:
            总奖励值 (范围: -1 到 0)
        """
        total_reward = 0.0
        
        # 1. 运输强度奖励
        if 'transportation_intensity' in self.objective_weights:
            r_ti = self._calculate_transportation_intensity_reward(current_unit_idx)
            total_reward += self.objective_weights['transportation_intensity'] * r_ti
        
        # 2. 物料流清晰度奖励
        if 'material_flow_clarity' in self.objective_weights:
            r_clarity = self._calculate_clarity_reward(current_unit_idx)
            total_reward += self.objective_weights['material_flow_clarity'] * r_clarity
        
        # 3. 仿真相关奖励（仅在放置完所有单元后计算）
        if run_simulation and len(self.placed_units) == len(self.functional_units):
            # 生产周期时间
            if 'throughput_time' in self.objective_weights:
                r_tt = self._calculate_throughput_time_reward()
                total_reward += self.objective_weights['throughput_time'] * r_tt
            
            # 功能单元利用率
            if 'utilization' in self.objective_weights:
                r_util = self._calculate_utilization_reward()
                total_reward += self.objective_weights['utilization'] * r_util
        
        return total_reward
    
    def _calculate_transportation_intensity_reward(
        self,
        current_unit_idx: int
    ) -> float:
        """
        计算运输强度奖励（文献公式10-11）
        
        R_TI = d_mt / d_mt_max
        
        d_mt = sum(distance(mt, j) * flow(mt, j))
        
        Returns:
            奖励值 (-1 到 0)
        """
        if current_unit_idx >= len(self.placed_units):
            return 0.0
        
        current_unit_id, x, y, rotation = self.placed_units[current_unit_idx]
        
        # 获取当前单元的中心坐标
        current_center = self._get_unit_center(current_unit_id, x, y, rotation)
        
        # 计算与所有已放置单元的运输强度
        total_intensity = 0.0
        max_possible_intensity = 0.0
        
        for i, (other_id, ox, oy, orot) in enumerate(self.placed_units):
            if i == current_unit_idx:
                continue
            
            # 获取另一个单元的中心
            other_center = self._get_unit_center(other_id, ox, oy, orot)
            
            # 计算曼哈顿距离（Dijkstra的简化）
            distance = np.abs(current_center[0] - other_center[0]) + \
                      np.abs(current_center[1] - other_center[1])
            
            # 获取物料流量
            flow = self.material_flow[current_unit_id, other_id]
            
            # 累积运输强度
            total_intensity += distance * flow
            
            # 计算最大可能强度（假设距离为网格对角线）
            max_distance = np.sqrt(self.layout_grid.shape[0]**2 + 
                                  self.layout_grid.shape[1]**2)
            max_possible_intensity += max_distance * flow
        
        # 归一化到 [-1, 0]
        if max_possible_intensity > 0:
            normalized_intensity = total_intensity / max_possible_intensity
            reward = -normalized_intensity
        else:
            reward = 0.0
        
        return reward
    
    def _calculate_clarity_reward(self, current_unit_idx: int) -> float:
        """
        计算物料流清晰度奖励（文献公式12-15）
        
        最佳情况：所有连接的单元排成一条线
        
        Returns:
            奖励值 (-1 到 0)
        """
        if current_unit_idx >= len(self.placed_units):
            return 0.0
        
        current_unit_id, x, y, rotation = self.placed_units[current_unit_idx]
        current_center = self._get_unit_center(current_unit_id, x, y, rotation)
        
        # 找到所有与当前单元有物料流的已放置单元
        connected_units = []
        for i, (other_id, ox, oy, orot) in enumerate(self.placed_units):
            if i == current_unit_idx:
                continue
            
            flow = self.material_flow[current_unit_id, other_id]
            if flow > 0:
                other_center = self._get_unit_center(other_id, ox, oy, orot)
                connected_units.append((other_id, other_center, flow))
        
        if len(connected_units) == 0:
            return 0.0
        
        # 计算角度偏差
        total_deviation = 0.0
        total_flow = 0.0
        
        for i, (id1, center1, flow1) in enumerate(connected_units):
            for j, (id2, center2, flow2) in enumerate(connected_units):
                if i >= j:
                    continue
                
                # 计算两个向量的角度
                vec1 = np.array(center1) - np.array(current_center)
                vec2 = np.array(center2) - np.array(current_center)
                
                # 角度（弧度）
                angle1 = np.arctan2(vec1[1], vec1[0])
                angle2 = np.arctan2(vec2[1], vec2[0])
                
                # 角度差（期望180度，即排成一线）
                angle_diff = np.abs(np.abs(angle1 - angle2) - np.pi)
                
                # 加权偏差
                weight = flow1 * flow2
                total_deviation += angle_diff * weight
                total_flow += weight
        
        # 归一化
        if total_flow > 0:
            normalized_deviation = total_deviation / (np.pi * total_flow)
            reward = -normalized_deviation
        else:
            reward = 0.0
        
        return reward
    
    def _calculate_throughput_time_reward(self) -> float:
        """
        计算生产周期时间奖励（需要仿真）
        
        这里需要调用SimPy仿真接口
        人员A需要实现与现有仿真的对接
        
        Returns:
            奖励值 (-1 到 0)
        """
        # TODO: 集成SimPy仿真
        # from .simulation_interface import SimulationInterface
        # sim = SimulationInterface(self.layout_grid, self.placed_units, ...)
        # throughput_time = sim.run_simulation()
        # normalized_time = throughput_time / estimated_max_time
        # return -normalized_time
        
        # 暂时返回占位值
        return 0.0
    
    def _calculate_utilization_reward(self) -> float:
        """
        计算功能单元利用率奖励（需要仿真）
        
        Returns:
            奖励值 (-1 到 0)
        """
        # TODO: 集成SimPy仿真
        # utilization = sim.get_utilization()
        # return -(1 - utilization)  # 利用率越高越好
        
        return 0.0
    
    def _get_unit_center(
        self,
        unit_id: int,
        x: int,
        y: int,
        rotation: int
    ) -> Tuple[float, float]:
        """
        获取功能单元的中心坐标
        
        Args:
            unit_id: 单元ID
            x, y: 左下角坐标
            rotation: 旋转角度
            
        Returns:
            中心坐标 (cx, cy)
        """
        unit = self.functional_units[unit_id]
        width, height = unit['size']
        
        # 考虑旋转
        if rotation in [90, 270]:
            width, height = height, width
        
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        
        return (center_x, center_y)


# ====================
# 测试代码
# ====================
if __name__ == "__main__":
    print("测试奖励函数...")
    
    # 模拟数据
    layout_grid = np.zeros((10, 10))
    
    functional_units = [
        {'id': 0, 'size': (2, 2)},
        {'id': 1, 'size': (3, 2)},
        {'id': 2, 'size': (2, 3)},
    ]
    
    # 模拟放置
    placed_units = [
        (0, 0, 0, 0),    # Unit 0 at (0,0)
        (1, 5, 0, 0),    # Unit 1 at (5,0)
        (2, 0, 5, 0),    # Unit 2 at (0,5)
    ]
    
    # 物料流矩阵
    material_flow = np.array([
        [0, 5, 2],
        [5, 0, 3],
        [2, 3, 0]
    ])
    
    # 目标权重
    weights = {
        'transportation_intensity': 0.7,
        'material_flow_clarity': 0.3
    }
    
    # 创建计算器
    calculator = RewardCalculator(
        layout_grid=layout_grid,
        placed_units=placed_units,
        material_flow=material_flow,
        functional_units=functional_units,
        objective_weights=weights
    )
    
    # 计算奖励
    for i in range(len(placed_units)):
        reward = calculator.calculate_total_reward(current_unit_idx=i)
        print(f"Unit {i} 奖励: {reward:.4f}")
    
    print("\n测试完成!")

