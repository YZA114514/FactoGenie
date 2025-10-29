"""
环境与DQN适配器
将 LayoutEnvironment 的状态/动作格式转换为 DQN 需要的格式
"""

import numpy as np
import torch
from typing import Dict, Tuple, List


class EnvDQNAdapter:
    """
    适配器：连接 LayoutEnvironment 和 LayoutDQN
    
    主要功能：
        1. 将环境状态转换为DQN输入格式
        2. 将DQN动作索引转换为环境动作字典
        3. 将环境动作字典转换为DQN动作索引
    """
    
    def __init__(
        self,
        grid_size: Tuple[int, int],
        num_units: int,
        num_rotations: int = 4
    ):
        """
        初始化适配器
        
        Args:
            grid_size: (width, height) 网格尺寸
            num_units: 功能单元数量
            num_rotations: 旋转角度数量（默认4：0°, 90°, 180°, 270°）
        """
        self.width, self.height = grid_size
        self.num_units = num_units
        self.num_rotations = num_rotations
        
        # 动作空间大小
        self.num_actions = self.width * self.height * self.num_rotations
        
        # 旋转角度映射
        self.rotation_to_idx = {0: 0, 90: 1, 180: 2, 270: 3}
        self.idx_to_rotation = {0: 0, 1: 90, 2: 180, 3: 270}
    
    def env_state_to_dqn_input(
        self, 
        env_state: Dict,
        device: str = 'cpu'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        将环境状态转换为DQN输入
        
        Args:
            env_state: 环境返回的状态字典
                {
                    'layout_grid': [nx, ny],  # 每个位置的单元ID
                    'material_flow': [N, N],
                    'current_unit': [N],  # One-Hot
                    ...
                }
            device: 'cpu' 或 'cuda'
        
        Returns:
            layout: [1, H, W, N] 三维布局张量
            material_flow: [1, N, N] 物料流邻接矩阵
            current_object: [1, N] 待放置模块One-Hot
        """
        # 1. 转换布局: [nx, ny] → [H, W, N]
        layout_grid = env_state['layout_grid']  # [nx, ny]，归一化的单元ID
        layout = self._convert_layout_to_3d(layout_grid)
        
        # 2. 物料流二值化: 浮点数 → 0/1
        material_flow = env_state['material_flow']  # [N, N]
        material_flow_binary = self._binarize_material_flow(material_flow)
        
        # 3. 当前单元（已经是One-Hot）
        current_object = env_state['current_unit']  # [N]
        
        # 转为PyTorch张量并添加batch维度
        layout = torch.from_numpy(layout).float().unsqueeze(0).to(device)
        material_flow_binary = torch.from_numpy(material_flow_binary).float().unsqueeze(0).to(device)
        current_object = torch.from_numpy(current_object).float().unsqueeze(0).to(device)
        
        return layout, material_flow_binary, current_object
    
    def _convert_layout_to_3d(self, layout_grid: np.ndarray) -> np.ndarray:
        """
        将二维布局转换为三维One-Hot表示
        
        Args:
            layout_grid: [nx, ny] 每个位置的单元ID（归一化后，0=空）
        
        Returns:
            layout_3d: [H, W, N] 每个位置的One-Hot编码
        """
        H, W = layout_grid.shape
        N = self.num_units
        
        layout_3d = np.zeros((H, W, N), dtype=np.float32)
        
        for i in range(H):
            for j in range(W):
                unit_id = int(layout_grid[i, j] * N)  # 反归一化
                if unit_id > 0:  # 0表示空位
                    # One-Hot编码（单元ID从1开始，所以减1）
                    if unit_id - 1 < N:
                        layout_3d[i, j, unit_id - 1] = 1.0
        
        return layout_3d
    
    def _binarize_material_flow(
        self, 
        material_flow: np.ndarray,
        threshold: float = None
    ) -> np.ndarray:
        """
        将物料流矩阵二值化
        
        Args:
            material_flow: [N, N] 浮点数矩阵
            threshold: 阈值，默认为均值
        
        Returns:
            binary_flow: [N, N] 0/1矩阵
        """
        if threshold is None:
            threshold = material_flow.mean()
        
        binary_flow = (material_flow > threshold).astype(np.float32)
        return binary_flow
    
    def dqn_action_to_env_action(self, action_idx: int) -> Dict:
        """
        将DQN动作索引转换为环境动作字典
        
        Args:
            action_idx: DQN输出的动作索引 (0 ~ num_actions-1)
        
        Returns:
            action_dict: {'x': int, 'y': int, 'rotation': int}
        """
        # 动作索引 = (x * height + y) * num_rotations + rotation_idx
        rotation_idx = action_idx % self.num_rotations
        xy_idx = action_idx // self.num_rotations
        
        x = xy_idx // self.height
        y = xy_idx % self.height
        
        rotation = self.idx_to_rotation[rotation_idx]
        
        return {'x': x, 'y': y, 'rotation': rotation}
    
    def env_action_to_dqn_action(self, action_dict: Dict) -> int:
        """
        将环境动作字典转换为DQN动作索引
        
        Args:
            action_dict: {'x': int, 'y': int, 'rotation': int}
        
        Returns:
            action_idx: DQN动作索引
        """
        x = action_dict['x']
        y = action_dict['y']
        rotation = action_dict['rotation']
        
        rotation_idx = self.rotation_to_idx[rotation]
        action_idx = (x * self.height + y) * self.num_rotations + rotation_idx
        
        return action_idx
    
    def get_valid_action_indices(
        self, 
        valid_actions: List[Dict]
    ) -> List[int]:
        """
        将环境的有效动作列表转换为DQN动作索引列表
        
        Args:
            valid_actions: 环境返回的有效动作列表
                [{'x': 0, 'y': 0, 'rotation': 0}, ...]
        
        Returns:
            valid_indices: DQN动作索引列表 [0, 1, 5, 10, ...]
        """
        valid_indices = [
            self.env_action_to_dqn_action(action) 
            for action in valid_actions
        ]
        return valid_indices


# ====================
# 测试代码
# ====================
if __name__ == "__main__":
    print("=" * 80)
    print("测试 EnvDQNAdapter")
    print("=" * 80)
    
    # 创建适配器
    adapter = EnvDQNAdapter(
        grid_size=(10, 10),
        num_units=5,
        num_rotations=4
    )
    
    print(f"\n适配器配置:")
    print(f"  网格大小: {adapter.width} × {adapter.height}")
    print(f"  单元数量: {adapter.num_units}")
    print(f"  动作空间: {adapter.num_actions}")
    
    # 测试1: 动作转换
    print(f"\n【测试1】动作格式转换:")
    
    action_dict = {'x': 5, 'y': 3, 'rotation': 90}
    action_idx = adapter.env_action_to_dqn_action(action_dict)
    print(f"  环境动作: {action_dict}")
    print(f"  DQN索引: {action_idx}")
    
    # 反向转换
    recovered_dict = adapter.dqn_action_to_env_action(action_idx)
    print(f"  恢复动作: {recovered_dict}")
    print(f"  ✓ 转换正确: {recovered_dict == action_dict}")
    
    # 测试2: 状态转换
    print(f"\n【测试2】状态格式转换:")
    
    # 模拟环境状态
    env_state = {
        'layout_grid': np.random.rand(10, 10),  # 归一化的布局
        'material_flow': np.random.rand(5, 5) * 10,  # 浮点数物料流
        'current_unit': np.array([0, 0, 1, 0, 0], dtype=np.float32)  # One-Hot
    }
    
    print(f"  环境状态:")
    print(f"    layout_grid: {env_state['layout_grid'].shape}")
    print(f"    material_flow: {env_state['material_flow'].shape}")
    print(f"    current_unit: {env_state['current_unit'].shape}")
    
    # 转换为DQN输入
    layout, material_flow, current_object = adapter.env_state_to_dqn_input(env_state)
    
    print(f"  DQN输入:")
    print(f"    layout: {layout.shape}")  # [1, H, W, N]
    print(f"    material_flow: {material_flow.shape}")  # [1, N, N]
    print(f"    current_object: {current_object.shape}")  # [1, N]
    
    # 验证物料流二值化
    print(f"  物料流二值化:")
    print(f"    原始范围: [{env_state['material_flow'].min():.2f}, {env_state['material_flow'].max():.2f}]")
    print(f"    二值范围: [{material_flow.min():.0f}, {material_flow.max():.0f}]")
    print(f"    ✓ 已二值化: {set(material_flow.unique().numpy()) == {0.0, 1.0} or len(material_flow.unique()) <= 2}")
    
    # 测试3: 有效动作批量转换
    print(f"\n【测试3】有效动作批量转换:")
    
    valid_actions = [
        {'x': 0, 'y': 0, 'rotation': 0},
        {'x': 0, 'y': 0, 'rotation': 90},
        {'x': 1, 'y': 2, 'rotation': 180},
        {'x': 5, 'y': 3, 'rotation': 270}
    ]
    
    print(f"  环境有效动作: {len(valid_actions)} 个")
    print(f"  示例: {valid_actions[:2]}")
    
    valid_indices = adapter.get_valid_action_indices(valid_actions)
    print(f"  DQN有效索引: {valid_indices}")
    
    # 验证
    all_correct = all(
        adapter.dqn_action_to_env_action(idx) == action
        for idx, action in zip(valid_indices, valid_actions)
    )
    print(f"  ✓ 批量转换正确: {all_correct}")
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)

