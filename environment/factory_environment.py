"""
工厂布局规划环境 - 主环境类
负责人：张毅

基于文献方法实现的RL环境，符合OpenAI Gym接口规范
"""

import csv
import json
from datetime import datetime
import numpy as np
from typing import Dict, Tuple, List, Optional
import copy
from pathlib import Path


METRIC_FIELDS = [
    "average_route_distance",
    "total_route_distance",
    "max_route_distance",
    "min_route_distance",
    "total_logistics_intensity",
    "space_utilization",
    "factory_area",
    "occupied_area",
    "free_area",
    "finished_goods",
    "throughput_rate",
    "avg_station_utilization",
    "distance_reward",
    "logistics_reward",
    "flow_clarity_reward",
    "throughput_reward",
    "utilization_reward",
    "final_reward",
    "placed_units",
    "total_units",
    "use_simulation",
    "early_termination",
    "error",
    "station_utilization_detail",
    "transporter_utilization_detail",
    "summary_node_detail",
    "summary_material_detail",
    "static_summary_json",
    "dynamic_summary_json",
]


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
        simulation_duration: float = 20000,      # 仿真时长（1天=20000，产能400个/天）
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
        self.last_metrics = None
        self.metrics_log_path: Optional[Path] = None
        self.metrics_log_header_written = False
        self.episode_counter = 0
        
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

    def set_metrics_logger(self, log_path: str) -> None:
        """配置仿真指标日志输出文件"""
        if log_path:
            self.metrics_log_path = Path(log_path)
            self.metrics_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_log_header_written = self.metrics_log_path.exists()
    
    @classmethod
    def from_config(cls, config_path: str, use_simulation: bool = True, 
                   simulation_duration: float = 20000) -> 'LayoutEnvironment':
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
        # 首先检查是否还有有效动作（提前失败机制）
        valid_actions = self.get_valid_actions()
        if len(valid_actions) == 0:
            # 没有有效动作可用，直接结束episode，避免进入仿真
            print(f"⚠️ 步骤 {self.current_unit_idx + 1}: 没有有效动作，提前结束episode")
            reward = -10.0  # 严重惩罚，鼓励agent学习避免这种情况
            done = True
            
            info = {
                'error': 'No valid actions available',
                'placed_units': len(self.placed_units),
                'total_units': self.num_units,
                'early_termination': True
            }
            metrics_payload = self._record_episode_metrics(
                reward,
                extra={'early_termination': True, 'error': info['error']}
            )
            info['metrics'] = metrics_payload
            return self._get_state(), reward, done, info
        
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
        if done:
            extra_flags = {}
            if info.get('early_termination'):
                extra_flags['early_termination'] = True
            if 'error' in info:
                extra_flags['error'] = info['error']
            metrics_payload = self._record_episode_metrics(
                reward,
                extra=extra_flags if extra_flags else None
            )
            info['metrics'] = metrics_payload

        return next_state, reward, done, info
    
    def get_valid_actions(self) -> List[Dict]:
        """
        获取当前状态下的所有有效动作（动作屏蔽）
        
        Returns:
            有效动作列表 [{'x': int, 'y': int, 'rotation': int}, ...]
        """
        # 检查是否所有单元都已放置
        if self.current_unit_idx >= self.num_units:
            return []
        
        valid_actions = []
        unit = self.functional_units[self.current_unit_idx]
        
        # 遍历所有可能的位置和旋转
        rotations = [0, 90, 180, 270] if unit.get('rotatable', True) else [0]
        
        for x in range(self.nx):
            for y in range(self.ny):
                for rotation in rotations:
                    try:
                        if self._is_valid_action(unit, x, y, rotation):
                            valid_actions.append({
                                'x': x, 
                                'y': y, 
                                'rotation': rotation
                            })
                    except (IndexError, ValueError) as e:
                        # 跳过导致错误的动作
                        continue
        
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
        检查动作是否有效（简化约束规则）
        
        坐标系统（与Simulation统一）：
        - (x, y) 是旋转前矩形的左下角
        - 旋转方向：顺时针
        - 原始尺寸：length (长度), width (宽度)
        
        约束规则：
        1. 不超出边界
        2. 不与已放置单元重叠
        3. 接收/输出仓库（rec_dock, ship_dock）必须贴墙
        
        Args:
            unit: 功能单元配置
            x, y: 旋转前矩形的左下角位置
            rotation: 旋转角度（0, 90, 180, 270度，顺时针）
            
        Returns:
            是否有效
        """
        # 获取原始尺寸
        length, width = unit['size']  # 注意：这里length是第一个维度，width是第二个维度
        
        # 根据旋转角度计算实际占用的网格范围
        # 顺时针旋转，以(x, y)为旋转前矩形的左下角
        if rotation == 0:
            # 无旋转：占用[x, x+length) × [y, y+width)
            occupied_cells = [(x + dx, y + dy) for dx in range(length) for dy in range(width)]
            # 实际占用区域的边界（用于贴墙检查）
            actual_x_min, actual_x_max = x, x + length
            actual_y_min, actual_y_max = y, y + width
        elif rotation == 90:
            # 顺时针90°：占用[x, x+width) × [y-length, y)
            occupied_cells = [(x + dx, y - length + dy) for dx in range(width) for dy in range(length)]
            actual_x_min, actual_x_max = x, x + width
            actual_y_min, actual_y_max = y - length, y
        elif rotation == 180:
            # 顺时针180°：占用[x-length, x) × [y-width, y)
            occupied_cells = [(x - length + dx, y - width + dy) for dx in range(length) for dy in range(width)]
            actual_x_min, actual_x_max = x - length, x
            actual_y_min, actual_y_max = y - width, y
        elif rotation == 270:
            # 顺时针270°：占用[x-width, x) × [y, y+length)
            occupied_cells = [(x - width + dx, y + dy) for dx in range(width) for dy in range(length)]
            actual_x_min, actual_x_max = x - width, x
            actual_y_min, actual_y_max = y, y + length
        else:
            # 非标准角度，不支持
            return False
        
        # 1. 检查是否超出边界
        if actual_x_min < 0 or actual_y_min < 0 or actual_x_max > self.nx or actual_y_max > self.ny:
            return False
        
        # 2. 检查是否与已放置单元重叠
        for pos_x, pos_y in occupied_cells:
            # 双重检查边界（防止索引越界）
            if pos_x < 0 or pos_x >= self.nx or pos_y < 0 or pos_y >= self.ny:
                return False
            
            # 检查网格占用
            if self.layout_grid[pos_x, pos_y] != 0:
                return False
        
        # 3. 检查接收/输出仓库的贴墙约束
        unit_id = unit.get('id', unit.get('name', ''))
        if unit_id in ['rec_dock', 'ship_dock']:
            # 这两个单元必须至少有一边贴墙（使用实际占用区域的边界）
            at_left = (actual_x_min == 0)
            at_right = (actual_x_max == self.nx)
            at_bottom = (actual_y_min == 0)
            at_top = (actual_y_max == self.ny)
            
            if not (at_left or at_right or at_bottom or at_top):
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
        
        坐标系统（与Simulation统一）：
        - (x, y) 是旋转前矩形的左下角
        - 旋转方向：顺时针
        
        Args:
            unit: 功能单元配置
            x, y: 旋转前矩形的左下角位置
            rotation: 旋转角度（0, 90, 180, 270度，顺时针）
        """
        unit_id = unit['id']  # 这是字符串ID，如 'rec_dock'
        unit_idx = self.current_unit_idx  # 这是索引，用于导出
        
        # 获取原始尺寸
        length, width = unit['size']
        
        # 根据旋转角度计算实际占用的网格位置
        if rotation == 0:
            # 无旋转：占用[x, x+length) × [y, y+width)
            occupied_cells = [(x + dx, y + dy) for dx in range(length) for dy in range(width)]
        elif rotation == 90:
            # 顺时针90°：占用[x, x+width) × [y-length, y)
            occupied_cells = [(x + dx, y - length + dy) for dx in range(width) for dy in range(length)]
        elif rotation == 180:
            # 顺时针180°：占用[x-length, x) × [y-width, y)
            occupied_cells = [(x - length + dx, y - width + dy) for dx in range(length) for dy in range(width)]
        elif rotation == 270:
            # 顺时针270°：占用[x-width, x) × [y, y+length)
            occupied_cells = [(x - width + dx, y + dy) for dx in range(width) for dy in range(length)]
        else:
            occupied_cells = []
        
        # 在网格中标记（使用索引+1）
        for pos_x, pos_y in occupied_cells:
            self.layout_grid[pos_x, pos_y] = unit_idx + 1  # +1避免与0混淆
        
        # 记录放置信息（直接使用输入坐标，不需要转换）
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
            self.last_metrics = None

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
            
            # 提取关键指标（使用训练数据分析得到的真实范围）
            avg_distance = static.get('average_route_distance', 0)
            total_distance = static.get('total_route_distance', 0)
            max_distance = static.get('max_route_distance', 0)
            min_distance = static.get('min_route_distance', 0)
            total_logistics = static.get('total_logistics_intensity', 0)
            space_util = static.get('space_utilization', 0)
            factory_area = static.get('factory_area', 0)
            occupied_area = static.get('occupied_area', 0)
            free_area = static.get('free_area', 0)
            
            finished_goods = dynamic.get('finished_goods', 0)
            throughput_rate = dynamic.get('throughput_rate', 0)
            
            # 计算平均工位利用率
            station_util_dict = dynamic.get('station_utilization', {})
            if station_util_dict:
                station_utils = [v.get('utilization', 0) for v in station_util_dict.values() if isinstance(v, dict)]
                avg_station_util = sum(station_utils) / len(station_utils) if station_utils else 0
            else:
                avg_station_util = 0

            transporter_util_dict = dynamic.get('transporter_utilization', {})
            summary_node_detail = dynamic.get('summary_node', {})
            summary_material_detail = dynamic.get('summary_material', {})
            
            print(f"[仿真指标] 平均距离:{avg_distance:.2f}, 物流强度:{total_logistics:.0f}, 空间利用:{space_util:.2%}")
            print(f"[仿真指标] 完成产品:{finished_goods:.0f}, 吞吐率:{throughput_rate:.6f}, 工位利用率:{avg_station_util:.4f}")
            
            # 计算多目标奖励（归一化到 [-1, 0] 范围）
            # 1. 运输距离越小越好（基于随机布局 vs SLP 数据，范围约 7-24）
            distance_best, distance_worst = 6.5, 24.0
            distance_range = max(distance_worst - distance_best, 1e-6)
            distance_reward = -(avg_distance - distance_best) / distance_range
            distance_reward = np.clip(distance_reward, -1.0, 0.0)
            
            # 2. 物流强度越小越好（基于实测 230-830 的范围）
            logistics_best, logistics_worst = 230.0, 900.0
            logistics_range = max(logistics_worst - logistics_best, 1e-6)
            logistics_reward = -(total_logistics - logistics_best) / logistics_range
            logistics_reward = np.clip(logistics_reward, -1.0, 0.0)
            
            # 3. 物料流清晰度奖励（基于角度偏差，物料流路径直线化）
            # R_clarity = - Σ(角度偏差 × 流量权重) / (π × 总流量权重)
            flow_clarity_reward = self._calculate_flow_clarity_reward()
            flow_clarity_reward = np.clip(flow_clarity_reward, -1.0, 0.0)
            
            # 4. 吞吐量奖励（目标400件，允许超额）
            if finished_goods < 200:
                throughput_reward = -1.0  # 严重不足（低于50%）
            elif finished_goods < 400:
                throughput_reward = (finished_goods - 400) / 200.0  # 线性惩罚
            else:
                # 达到或超过目标，给予奖励（但有上限）
                throughput_reward = min((finished_goods - 400) / 800.0, 0.0)  # 最多+0分
            
            # 5. 工位利用率奖励（新增）
            if avg_station_util < 0.001:
                utilization_reward = -1.0  # 几乎不工作
            elif avg_station_util < 0.05:
                # 利用率1%-5%范围，鼓励提升
                utilization_reward = (avg_station_util - 0.05) / 0.05
            else:
                # 利用率>5%，逐渐接近0（最好）
                utilization_reward = min(-0.05 / avg_station_util + 1.0, 0.0)
            utilization_reward = np.clip(utilization_reward, -1.0, 0.0)
            
            print(f"[奖励分量] 距离:{distance_reward:.3f}, 物流:{logistics_reward:.3f}, 流清晰:{flow_clarity_reward:.3f}, 吞吐:{throughput_reward:.3f}, 利用率:{utilization_reward:.3f}")
            
            # 加权求和（移除空间利用项，重新归一化权重）
            weights = self.objective_weights
            weight_components = {
                'distance': weights.get('transportation_intensity', 0.20),
                'logistics': weights.get('material_flow_clarity', 0.30),
                'flow': weights.get('flow_clarity', weights.get('space_utilization', 0.20)),
                'throughput': weights.get('throughput_time', 0.25),
                'utilization': weights.get('utilization', 0.05)
            }
            total_weight = sum(weight_components.values()) or 1.0
            reward = (
                weight_components['distance'] * distance_reward +
                weight_components['logistics'] * logistics_reward +
                weight_components['flow'] * flow_clarity_reward +
                weight_components['throughput'] * throughput_reward +
                weight_components['utilization'] * utilization_reward
            ) / total_weight
            
            # 确保在 [-1, 0] 范围
            reward = np.clip(reward, -1.0, 0.0)
            
            print(f"[最终奖励] {reward:.6f}")

            self.last_metrics = {
                'average_route_distance': float(avg_distance),
                'total_route_distance': float(total_distance),
                'max_route_distance': float(max_distance),
                'min_route_distance': float(min_distance),
                'total_logistics_intensity': float(total_logistics),
                'space_utilization': float(space_util),
                'factory_area': float(factory_area),
                'occupied_area': float(occupied_area),
                'free_area': float(free_area),
                'finished_goods': float(finished_goods),
                'throughput_rate': float(throughput_rate),
                'avg_station_utilization': float(avg_station_util),
                'distance_reward': float(distance_reward),
                'logistics_reward': float(logistics_reward),
                'flow_clarity_reward': float(flow_clarity_reward),
                'throughput_reward': float(throughput_reward),
                'utilization_reward': float(utilization_reward),
                'final_reward': float(reward),
                'station_utilization_detail': json.dumps(station_util_dict, ensure_ascii=False),
                'transporter_utilization_detail': json.dumps(transporter_util_dict, ensure_ascii=False),
                'summary_node_detail': json.dumps(summary_node_detail, ensure_ascii=False),
                'summary_material_detail': json.dumps(summary_material_detail, ensure_ascii=False),
                'static_summary_json': json.dumps(static, ensure_ascii=False),
                'dynamic_summary_json': json.dumps(dynamic, ensure_ascii=False)
            }

            return float(reward)

        except Exception as e:
            print(f"仿真运行失败: {e}")
            import traceback
            traceback.print_exc()
            # 失败时返回较大的惩罚
            self.last_metrics = {
                'error': str(e),
                'final_reward': -1.0
            }
            return -1.0

    def _record_episode_metrics(self, reward: float, extra: Optional[Dict] = None) -> Dict:
        """整理并记录一次 episode 的指标，返回写入的内容"""
        metrics = {key: None for key in METRIC_FIELDS}
        if self.last_metrics:
            for key, value in self.last_metrics.items():
                if key in metrics:
                    metrics[key] = value

        metrics['final_reward'] = metrics.get('final_reward', float(reward))
        metrics['placed_units'] = len(self.placed_units)
        metrics['total_units'] = self.num_units
        metrics['use_simulation'] = self.use_simulation
        metrics['early_termination'] = metrics.get('early_termination', False)

        if extra:
            for key, value in extra.items():
                if key in metrics:
                    metrics[key] = value

        self.episode_counter += 1
        # self._log_metrics(metrics)
        self._maybe_save_episode_layout()
        self.last_metrics = None
        return metrics

    def _log_metrics(self, metrics: Dict) -> None:
        """将仿真指标写入CSV日志，便于后续分析"""
        # 用户要求暂时禁用每个 episode 的 CSV 写入，如需恢复可取消以下注释
        # if not self.metrics_log_path:
        #     return
        # row = {field: metrics.get(field) for field in METRIC_FIELDS}
        # row['episode'] = self.episode_counter
        # row['timestamp'] = datetime.now().isoformat()
        # fieldnames = ['episode', 'timestamp', *METRIC_FIELDS]
        # try:
        #     with self.metrics_log_path.open('a', newline='', encoding='utf-8') as f:
        #         writer = csv.DictWriter(f, fieldnames=fieldnames)
        #         if not self.metrics_log_header_written:
        #             writer.writeheader()
        #             self.metrics_log_header_written = True
        #         writer.writerow(row)
        # except Exception as exc:
        #     print(f"警告: 写入指标日志失败: {exc}")

    def _maybe_save_episode_layout(self) -> None:
        """仅在满足周期条件时保存布局及可视化数据"""
        if not self.layout_path or not self.layout_template:
            return
        if self.episode_counter % 100 != 0:
            return
        try:
            self._export_current_layout()
            print(f"[布局记录] 第 {self.episode_counter} 个 episode，保存当前布局")
        except Exception as exc:
            print(f"警告: 间隔保存布局失败: {exc}")
    
    def _calculate_flow_clarity_reward(self) -> float:
        """
        计算物料流清晰度奖励
        基于角度偏差，评估物料流路径的直线化程度
        
        公式: R_clarity = - Σ(角度偏差 × 流量权重) / (π × 总流量权重)
        理想情况：所有连接的单元排成一条直线（180°）
        
        Returns:
            float: 清晰度奖励 [-1, 0]，0为最优（直线），-1为最差（混乱）
        """
        if self.material_flow is None or len(self.placed_units) < 2:
            return 0.0  # 无物料流或单元太少时给予最好奖励
        
        total_deviation = 0.0
        total_flow_weight = 0.0
        
        # 遍历所有已放置的单元
        for i, placed_unit in enumerate(self.placed_units):
            unit_idx, x, y, rotation = placed_unit
            
            # 获取单元尺寸
            unit_info = self.functional_units[unit_idx]
            width, height = unit_info['size']
            
            # 根据旋转调整尺寸
            if rotation in [90, 270]:
                width, height = height, width
            
            current_center = np.array([
                x + width / 2.0,
                y + height / 2.0
            ])
            
            # 找到所有与当前单元有物料流的单元
            connected_units = []
            for j, other_placed_unit in enumerate(self.placed_units):
                if i == j:
                    continue
                
                other_unit_idx, other_x, other_y, other_rotation = other_placed_unit
                
                # 检查物料流连接（双向）
                flow1 = self.material_flow[unit_idx, other_unit_idx] if unit_idx < self.material_flow.shape[0] and other_unit_idx < self.material_flow.shape[1] else 0
                flow2 = self.material_flow[other_unit_idx, unit_idx] if other_unit_idx < self.material_flow.shape[0] and unit_idx < self.material_flow.shape[1] else 0
                total_flow = flow1 + flow2
                
                if total_flow > 0:
                    # 获取其他单元尺寸
                    other_unit_info = self.functional_units[other_unit_idx]
                    other_width, other_height = other_unit_info['size']
                    
                    # 根据旋转调整尺寸
                    if other_rotation in [90, 270]:
                        other_width, other_height = other_height, other_width
                    
                    other_center = np.array([
                        other_x + other_width / 2.0,
                        other_y + other_height / 2.0
                    ])
                    connected_units.append((j, other_center, total_flow))
            
            # 如果连接单元少于2个，无法计算角度偏差
            if len(connected_units) < 2:
                continue
            
            # 计算所有连接单元对之间的角度偏差
            for i in range(len(connected_units)):
                for j in range(i + 1, len(connected_units)):
                    id1, center1, flow1 = connected_units[i]
                    id2, center2, flow2 = connected_units[j]
                    
                    # 计算两个向量
                    vec1 = center1 - current_center
                    vec2 = center2 - current_center
                    
                    # 避免零向量
                    if np.linalg.norm(vec1) < 1e-6 or np.linalg.norm(vec2) < 1e-6:
                        continue
                    
                    # 计算角度
                    angle1 = np.arctan2(vec1[1], vec1[0])
                    angle2 = np.arctan2(vec2[1], vec2[0])
                    
                    # 角度差（期望180度，即π弧度，排成一线）
                    angle_diff = abs(angle1 - angle2)
                    # 将角度差映射到[0, π]范围
                    if angle_diff > np.pi:
                        angle_diff = 2 * np.pi - angle_diff
                    
                    # 计算与理想角度π的偏差
                    angle_deviation = abs(angle_diff - np.pi)
                    
                    # 流量权重（两个连接的流量乘积）
                    flow_weight = flow1 * flow2
                    
                    # 累积偏差
                    total_deviation += angle_deviation * flow_weight
                    total_flow_weight += flow_weight
        
        # 归一化到 [-1, 0] 范围
        if total_flow_weight > 0:
            # 最大可能偏差是π，所以除以π进行归一化
            normalized_deviation = total_deviation / (np.pi * total_flow_weight)
            reward = -min(normalized_deviation, 1.0)  # 确保不超过-1
        else:
            reward = 0.0  # 无有效连接时给予最好奖励
        
        return reward
    
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

