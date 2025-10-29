"""
工厂布局规划环境 - 主环境类
负责人：张毅

基于文献方法实现的RL环境，符合OpenAI Gym接口规范
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import copy


class LayoutEnvironment:
    """
    工厂布局规划强化学习环境
    
    状态空间：
        - 布局网格（占用情况、高度、介质供应等）
        - 物料流矩阵
        - 当前要放置的功能单元信息
    
    动作空间：
        - 位置(x, y)
        - 旋转角度(0°, 90°, 180°, 270°)
    
    奖励：
        - 多目标奖励函数（运输强度、生产周期等）
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int] = (20, 20),  # 网格尺寸 (nx, ny)
        functional_units: List[Dict] = None,     # 功能单元列表
        material_flow: np.ndarray = None,        # 物料流矩阵
        objective_weights: Dict[str, float] = None,  # 目标权重
        use_simulation: bool = False             # 是否使用仿真
    ):
        """
        初始化环境
        
        Args:
            grid_size: 布局网格大小 (宽度, 高度)
            functional_units: 功能单元配置列表
                [{
                    'id': 0,
                    'name': 'Machine_A',
                    'size': (3, 2),  # (宽, 高)
                    'rotatable': True
                }, ...]
            material_flow: 物料流矩阵 [N x N]，N为功能单元数量
            objective_weights: 目标权重字典
                {
                    'transportation_intensity': 0.5,
                    'throughput_time': 0.3,
                    'material_flow_clarity': 0.2
                }
            use_simulation: 是否集成SimPy仿真
        """
        self.grid_size = grid_size
        self.nx, self.ny = grid_size
        
        # 功能单元配置
        if functional_units is None:
            functional_units = self._create_default_units()
        self.functional_units = functional_units
        self.num_units = len(functional_units)
        
        # 物料流矩阵
        if material_flow is None:
            material_flow = self._create_default_material_flow()
        self.material_flow = material_flow
        
        # 目标权重
        if objective_weights is None:
            objective_weights = {'transportation_intensity': 1.0}
        self.objective_weights = objective_weights
        
        self.use_simulation = use_simulation
        
        # 环境状态
        self.reset()
        
    def reset(self) -> Dict:
        """
        重置环境到初始状态
        
        Returns:
            初始状态字典
        """
        # 布局网格 (0=空闲, >0=功能单元ID)
        self.layout_grid = np.zeros((self.nx, self.ny), dtype=np.int32)
        
        # 限制区域（墙壁、通道等）
        self.restricted_areas = np.zeros((self.nx, self.ny), dtype=bool)
        
        # 已放置的功能单元
        self.placed_units = []  # [(unit_id, x, y, rotation), ...]
        
        # 当前要放置的功能单元索引
        self.current_unit_idx = 0
        
        # 累积奖励
        self.episode_reward = 0.0
        
        return self._get_state()
    
    def step(self, action: Dict) -> Tuple[Dict, float, bool, Dict]:
        """
        执行一个动作
        
        Args:
            action: 动作字典 {'x': int, 'y': int, 'rotation': int}
                   rotation: 0, 90, 180, 270 度
        
        Returns:
            next_state: 下一个状态
            reward: 奖励值
            done: 是否结束
            info: 额外信息
        """
        # 获取当前功能单元
        unit = self.functional_units[self.current_unit_idx]
        
        # 解析动作
        x, y = action['x'], action['y']
        rotation = action.get('rotation', 0)
        
        # 检查动作有效性
        if not self._is_valid_action(unit, x, y, rotation):
            # 非法动作，给予惩罚
            reward = -1.0
            done = False
            info = {'error': 'Invalid action'}
            return self._get_state(), reward, done, info
        
        # 放置功能单元
        self._place_unit(unit, x, y, rotation)
        
        # 计算奖励（如果是最后一个单元，则运行仿真）
        is_last_unit = (self.current_unit_idx == self.num_units - 1)
        reward = self._calculate_reward(is_last_unit)
        
        # 更新状态
        self.current_unit_idx += 1
        done = (self.current_unit_idx >= self.num_units)
        
        # 获取下一个状态
        next_state = self._get_state()
        
        info = {
            'placed_units': len(self.placed_units),
            'total_units': self.num_units
        }
        
        return next_state, reward, done, info
    
    def get_valid_actions(self) -> List[Dict]:
        """
        获取当前状态下的所有有效动作（动作屏蔽）
        
        Returns:
            有效动作列表 [{'x': int, 'y': int, 'rotation': int}, ...]
        """
        valid_actions = []
        unit = self.functional_units[self.current_unit_idx]
        
        # 遍历所有可能的位置和旋转
        rotations = [0, 90, 180, 270] if unit.get('rotatable', True) else [0]
        
        for x in range(self.nx):
            for y in range(self.ny):
                for rotation in rotations:
                    if self._is_valid_action(unit, x, y, rotation):
                        valid_actions.append({
                            'x': x, 
                            'y': y, 
                            'rotation': rotation
                        })
        
        return valid_actions
    
    def _get_state(self) -> Dict:
        """
        获取当前状态表示
        
        Returns:
            状态字典，包含：
                - layout_grid: 布局网格 [nx, ny]
                - material_flow: 物料流矩阵 [N, N]
                - current_unit: 当前要放置的单元one-hot编码 [N]
                - placed_mask: 已放置单元的mask [N]
        """
        # 归一化布局网格 (0-1范围)
        layout_grid_norm = self.layout_grid.astype(np.float32) / max(self.num_units, 1)
        
        # 当前单元的one-hot编码
        current_unit_onehot = np.zeros(self.num_units, dtype=np.float32)
        if self.current_unit_idx < self.num_units:
            current_unit_onehot[self.current_unit_idx] = 1.0
        
        # 已放置单元的mask
        placed_mask = np.zeros(self.num_units, dtype=np.float32)
        for placed_unit in self.placed_units:
            unit_id = placed_unit[0]
            placed_mask[unit_id] = 1.0
        
        state = {
            'layout_grid': layout_grid_norm,
            'material_flow': self.material_flow.astype(np.float32),
            'current_unit': current_unit_onehot,
            'placed_mask': placed_mask,
            'restricted_areas': self.restricted_areas.astype(np.float32)
        }
        
        return state
    
    def _is_valid_action(
        self, 
        unit: Dict, 
        x: int, 
        y: int, 
        rotation: int
    ) -> bool:
        """
        检查动作是否有效
        
        Args:
            unit: 功能单元配置
            x, y: 左下角位置
            rotation: 旋转角度
            
        Returns:
            是否有效
        """
        # 获取旋转后的尺寸
        width, height = unit['size']
        if rotation in [90, 270]:
            width, height = height, width
        
        # 检查是否超出边界
        if x + width > self.nx or y + height > self.ny:
            return False
        
        # 检查是否与已放置单元或限制区域重叠
        for dx in range(width):
            for dy in range(height):
                pos_x, pos_y = x + dx, y + dy
                
                # 检查网格占用
                if self.layout_grid[pos_x, pos_y] != 0:
                    return False
                
                # 检查限制区域
                if self.restricted_areas[pos_x, pos_y]:
                    return False
        
        return True
    
    def _place_unit(
        self, 
        unit: Dict, 
        x: int, 
        y: int, 
        rotation: int
    ) -> None:
        """
        在布局中放置功能单元
        
        Args:
            unit: 功能单元配置
            x, y: 左下角位置
            rotation: 旋转角度
        """
        unit_id = unit['id']
        width, height = unit['size']
        
        # 旋转尺寸
        if rotation in [90, 270]:
            width, height = height, width
        
        # 在网格中标记
        for dx in range(width):
            for dy in range(height):
                self.layout_grid[x + dx, y + dy] = unit_id + 1  # +1避免与0混淆
        
        # 记录放置信息
        self.placed_units.append((unit_id, x, y, rotation))
    
    def _calculate_reward(self, run_simulation: bool = False) -> float:
        """
        计算当前步骤的奖励
        
        Args:
            run_simulation: 是否运行仿真获取动态目标
            
        Returns:
            奖励值 (范围: -1 到 0，越接近0越好)
        """
        # 这里调用RewardCalculator
        # 暂时返回简单的运输强度奖励
        
        from .reward_function import RewardCalculator
        
        calculator = RewardCalculator(
            layout_grid=self.layout_grid,
            placed_units=self.placed_units,
            material_flow=self.material_flow,
            functional_units=self.functional_units,
            objective_weights=self.objective_weights
        )
        
        reward = calculator.calculate_total_reward(
            current_unit_idx=self.current_unit_idx,
            run_simulation=run_simulation and self.use_simulation
        )
        
        return reward
    
    def _create_default_units(self) -> List[Dict]:
        """创建默认功能单元配置（用于测试）"""
        return [
            {'id': 0, 'name': 'Unit_0', 'size': (3, 3), 'rotatable': True},
            {'id': 1, 'name': 'Unit_1', 'size': (2, 2), 'rotatable': True},
            {'id': 2, 'name': 'Unit_2', 'size': (4, 2), 'rotatable': True},
            {'id': 3, 'name': 'Unit_3', 'size': (2, 3), 'rotatable': True},
            {'id': 4, 'name': 'Unit_4', 'size': (3, 2), 'rotatable': True},
        ]
    
    def _create_default_material_flow(self) -> np.ndarray:
        """创建默认物料流矩阵（用于测试）"""
        # 随机生成对称的物料流矩阵
        N = self.num_units
        flow = np.random.randint(0, 10, size=(N, N))
        flow = (flow + flow.T) / 2  # 对称化
        np.fill_diagonal(flow, 0)   # 对角线为0
        return flow
    
    def render(self, mode='console'):
        """
        可视化当前布局
        
        Args:
            mode: 'console' 或 'matplotlib'
        """
        if mode == 'console':
            print("\n当前布局:")
            print(self.layout_grid)
            print(f"\n已放置: {len(self.placed_units)}/{self.num_units}")
        elif mode == 'matplotlib':
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            fig, ax = plt.subplots(figsize=(10, 10))
            
            # 绘制网格
            ax.set_xlim(0, self.nx)
            ax.set_ylim(0, self.ny)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # 绘制已放置的功能单元
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_units))
            
            for unit_id, x, y, rotation in self.placed_units:
                unit = self.functional_units[unit_id]
                width, height = unit['size']
                
                # 考虑旋转
                if rotation in [90, 270]:
                    width, height = height, width
                
                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=2,
                    edgecolor='black',
                    facecolor=colors[unit_id],
                    alpha=0.6
                )
                ax.add_patch(rect)
                
                # 添加标签
                ax.text(
                    x + width/2, y + height/2,
                    unit['name'],
                    ha='center', va='center',
                    fontsize=10, fontweight='bold'
                )
            
            plt.title(f"工厂布局 - 已放置 {len(self.placed_units)}/{self.num_units}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.show()


# ====================
# 测试代码
# ====================
if __name__ == "__main__":
    print("测试布局环境...")
    
    # 创建环境
    env = LayoutEnvironment(
        grid_size=(15, 15),
        objective_weights={'transportation_intensity': 1.0}
    )
    
    print(f"环境创建成功！")
    print(f"网格大小: {env.grid_size}")
    print(f"功能单元数量: {env.num_units}")
    
    # 重置环境
    state = env.reset()
    print(f"\n初始状态形状:")
    for key, value in state.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
    
    # 获取有效动作
    valid_actions = env.get_valid_actions()
    print(f"\n有效动作数量: {len(valid_actions)}")
    print(f"示例动作: {valid_actions[:3]}")
    
    # 执行几步
    print("\n执行随机动作...")
    for i in range(3):
        valid_actions = env.get_valid_actions()
        if len(valid_actions) == 0:
            print("没有有效动作!")
            break
        
        # 随机选择一个动作
        action = valid_actions[np.random.randint(len(valid_actions))]
        next_state, reward, done, info = env.step(action)
        
        print(f"步骤 {i+1}: 动作={action}, 奖励={reward:.3f}, 完成={done}")
        env.render(mode='console')
        
        if done:
            break
    
    print("\n测试完成!")

