"""
工厂布局规划环境 - 主环境类
负责人：张毅

基于文献方法实现的RL环境，符合OpenAI Gym接口规范
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import copy
from pathlib import Path


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
        use_simulation: bool = True,            # 是否使用仿真
        config_path: Optional[str] = None,       # 仿真配置文件路径
        layout_path: Optional[str] = None,       # 布局输出路径
        simulation_duration: float = 120000,     # 仿真时长
        placement_constraints: Optional[Dict] = None  # 摆放约束
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
            config_path: 仿真配置文件路径
            layout_path: 布局输出文件路径
            simulation_duration: 仿真运行时长
            placement_constraints: 摆放约束规则
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
        
        # 摆放约束
        if placement_constraints is None:
            placement_constraints = {'min_distance': 0, 'wall_units': [], 'restricted_areas': []}
        self.placement_constraints = placement_constraints
        
        self.use_simulation = use_simulation
        self.config_path = config_path
        self.layout_path = layout_path
        self.simulation_duration = simulation_duration
        
        # 保存布局模板（用于导出）
        self.layout_template = None
        if config_path:
            try:
                from .config_loader import ConfigLoader
                loader = ConfigLoader(config_path, layout_path)
                self.layout_template = loader.get_layout_template()
            except Exception as e:
                print(f"警告: 无法加载布局模板: {e}")
        
        # 环境状态
        self.reset()
    
    @classmethod
    def from_config(cls, config_path: str, use_simulation: bool = True, 
                   simulation_duration: float = 120000) -> 'LayoutEnvironment':
        """
        从配置文件创建环境实例
        
        Args:
            config_path: 仿真配置文件路径 (如 'simulation/configs/chair_factory.json')
            use_simulation: 是否使用仿真计算奖励
            simulation_duration: 仿真时长
            
        Returns:
            LayoutEnvironment 实例
        """
        from .config_loader import ConfigLoader
        
        loader = ConfigLoader(config_path)
        functional_units = loader.get_functional_units()
        
        return cls(
            grid_size=loader.get_factory_size(),
            functional_units=functional_units,
            material_flow=loader.get_material_flow(functional_units),
            objective_weights=loader.get_objective_weights(),
            placement_constraints=loader.get_placement_constraints(),
            use_simulation=use_simulation,
            config_path=config_path,
            layout_path=str(loader.layout_path),
            simulation_duration=simulation_duration
        )
        
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
            unit_idx = placed_unit[0]  # 现在是索引
            placed_mask[unit_idx] = 1.0
        
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
        检查动作是否有效（包含摆放约束规则）
        
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
        if x < 0 or y < 0 or x + width > self.nx or y + height > self.ny:
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
        
        # 检查最小距离约束
        min_distance = self.placement_constraints.get('min_distance', 0)
        if min_distance > 0 and not self._check_min_distance(x, y, width, height, min_distance):
            return False
        
        # 检查贴墙约束
        wall_units = self.placement_constraints.get('wall_units', [])
        unit_id = unit.get('id', unit.get('name', ''))
        if unit_id in wall_units and not self._check_wall_constraint(x, y, width, height):
            return False
        
        # 检查禁止区域
        restricted_areas = self.placement_constraints.get('restricted_areas', [])
        for rx, ry, rw, rh in restricted_areas:
            if self._rectangles_overlap(x, y, width, height, rx, ry, rw, rh):
                return False
        
        return True
    
    def _check_min_distance(self, x: int, y: int, width: int, height: int, min_dist: int) -> bool:
        """
        检查与已放置单元的最小距离
        
        Args:
            x, y: 单元位置
            width, height: 单元尺寸
            min_dist: 最小距离
            
        Returns:
            是否满足最小距离约束
        """
        # 扩展检查区域
        check_x_start = max(0, x - min_dist)
        check_x_end = min(self.nx, x + width + min_dist)
        check_y_start = max(0, y - min_dist)
        check_y_end = min(self.ny, y + height + min_dist)
        
        # 检查扩展区域内是否有其他单元
        for cx in range(check_x_start, check_x_end):
            for cy in range(check_y_start, check_y_end):
                # 跳过单元自身区域
                if x <= cx < x + width and y <= cy < y + height:
                    continue
                # 检查是否被占用
                if self.layout_grid[cx, cy] != 0:
                    return False
        
        return True
    
    def _check_wall_constraint(self, x: int, y: int, width: int, height: int) -> bool:
        """
        检查是否贴墙（至少一边靠边界）
        
        Args:
            x, y: 单元位置
            width, height: 单元尺寸
            
        Returns:
            是否贴墙
        """
        # 检查是否至少有一边贴着工厂边界
        at_left = (x == 0)
        at_right = (x + width == self.nx)
        at_bottom = (y == 0)
        at_top = (y + height == self.ny)
        
        return at_left or at_right or at_bottom or at_top
    
    def _rectangles_overlap(self, x1: int, y1: int, w1: int, h1: int,
                           x2: int, y2: int, w2: int, h2: int) -> bool:
        """
        检查两个矩形是否重叠
        
        Args:
            x1, y1, w1, h1: 第一个矩形
            x2, y2, w2, h2: 第二个矩形
            
        Returns:
            是否重叠
        """
        # 检查是否不重叠的条件
        if x1 + w1 <= x2 or x2 + w2 <= x1:
            return False
        if y1 + h1 <= y2 or y2 + h2 <= y1:
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
        unit_id = unit['id']  # 这是字符串ID，如 'rec_dock'
        unit_idx = self.current_unit_idx  # 这是索引，用于导出
        
        width, height = unit['size']
        
        # 旋转尺寸
        if rotation in [90, 270]:
            width, height = height, width
        
        # 在网格中标记（使用索引+1）
        for dx in range(width):
            for dy in range(height):
                self.layout_grid[x + dx, y + dy] = unit_idx + 1  # +1避免与0混淆
        
        # 记录放置信息（使用索引，与 layout_exporter 期望一致）
        self.placed_units.append((unit_idx, x, y, rotation))
    
    def _calculate_reward(self, run_simulation: bool = False) -> float:
        """
        计算当前步骤的奖励
        
        Args:
            run_simulation: 是否运行仿真获取动态目标
            
        Returns:
            奖励值 (范围: -1 到 0，越接近0越好)
        """
        # 如果需要运行仿真（通常是最后一步）
        if run_simulation and self.use_simulation and self.config_path:
            reward = self._calculate_reward_with_simulation()
        else:
            # 使用静态奖励计算（中间步骤）
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
                run_simulation=False
            )
        
        return reward
    
    def _calculate_reward_with_simulation(self) -> float:
        """
        通过仿真计算奖励（仅在 episode 结束时调用）
        
        Returns:
            基于仿真结果的奖励值
        """
        try:
            print(f"\n[仿真] 开始运行仿真系统...")
            print(f"[仿真] 配置: {self.config_path}")
            print(f"[仿真] 时长: {self.simulation_duration}")
            
            # 1. 导出当前布局到 JSON
            self._export_current_layout()
            print(f"[仿真] 布局已导出到: {self.layout_path}")
            
            # 2. 运行仿真获取指标
            from simulation.interface import compute_metrics
            print(f"[仿真] 正在运行仿真（请稍候）...")
            
            metrics_result = compute_metrics(
                config_path=self.config_path,
                duration=self.simulation_duration,
                layout_path=self.layout_path,
                detail=True
            )
            
            # 3. 从指标中提取奖励
            summary = metrics_result['summary']
            static = summary.get('static', {})
            dynamic = summary.get('dynamic', {})
            
            # 提取关键指标（使用正确的字段名）
            avg_distance = static.get('average_route_distance', 0)
            total_logistics = static.get('total_logistics_intensity', 0)
            space_util = static.get('space_utilization', 0)
            
            finished_goods = dynamic.get('finished_goods', 0)
            throughput_rate = dynamic.get('throughput_rate', 0)
            
            # 计算平均工位利用率
            station_util_dict = dynamic.get('station_utilization', {})
            if station_util_dict:
                station_utils = [v.get('utilization', 0) for v in station_util_dict.values() if isinstance(v, dict)]
                avg_station_util = sum(station_utils) / len(station_utils) if station_utils else 0
            else:
                avg_station_util = 0
            
            print(f"[仿真指标] 平均距离:{avg_distance:.2f}, 物流强度:{total_logistics:.0f}, 空间利用:{space_util:.2%}")
            print(f"[仿真指标] 完成产品:{finished_goods:.0f}, 吞吐率:{throughput_rate:.6f}, 工位利用率:{avg_station_util:.4f}")
            
            # 计算多目标奖励（归一化到 [-1, 0] 范围）
            # 1. 运输距离越小越好（归一化：假设合理范围 30-100）
            distance_reward = -(avg_distance - 30) / 70.0  # 30最好=0, 100最差=-1
            distance_reward = max(distance_reward, -1.0)
            
            # 2. 物流强度越小越好（归一化：假设合理范围 1000-3000）
            logistics_reward = -(total_logistics - 1000) / 2000.0
            logistics_reward = max(logistics_reward, -1.0)
            
            # 3. 空间利用率越高越好（但不要太挤，0.15-0.3为佳）
            if space_util < 0.15:
                space_reward = (space_util - 0.15) / 0.15  # 低于15%不好
            elif space_util > 0.3:
                space_reward = (0.3 - space_util) / 0.3  # 高于30%太挤
            else:
                space_reward = 0.0  # 15-30%之间最好
            
            # 4. 吞吐量越大越好（归一化：假设目标100件）
            throughput_reward = (finished_goods - 100) / 100.0
            throughput_reward = min(max(throughput_reward, -1.0), 0.0)
            
            print(f"[奖励分量] 距离:{distance_reward:.3f}, 物流:{logistics_reward:.3f}, 空间:{space_reward:.3f}, 吞吐:{throughput_reward:.3f}")
            
            # 加权求和
            weights = self.objective_weights
            reward = (
                weights.get('transportation_intensity', 0.4) * distance_reward +
                weights.get('material_flow_clarity', 0.3) * logistics_reward +
                weights.get('space_utilization', 0.2) * space_reward +
                weights.get('throughput_time', 0.1) * throughput_reward
            )
            
            # 确保在 [-1, 0] 范围
            reward = np.clip(reward, -1.0, 0.0)
            
            print(f"[最终奖励] {reward:.6f}")
            
            return float(reward)
            
        except Exception as e:
            print(f"仿真运行失败: {e}")
            import traceback
            traceback.print_exc()
            # 失败时返回较大的惩罚
            return -1.0
    
    def _export_current_layout(self) -> None:
        """
        将当前布局导出到 JSON 文件
        """
        if not self.layout_path or not self.layout_template:
            print("警告: 未配置布局输出路径或模板")
            return
        
        from .layout_exporter import LayoutExporter
        
        exporter = LayoutExporter(self.layout_template, self.layout_path)
        exporter.export_layout(self.placed_units, self.functional_units)
    
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
            
            for unit_idx, x, y, rotation in self.placed_units:
                unit = self.functional_units[unit_idx]
                width, height = unit['size']
                
                # 考虑旋转
                if rotation in [90, 270]:
                    width, height = height, width
                
                rect = patches.Rectangle(
                    (x, y), width, height,
                    linewidth=2,
                    edgecolor='black',
                    facecolor=colors[unit_idx],
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

