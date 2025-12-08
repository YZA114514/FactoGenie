"""
Gym风格包装器 - 使LayoutEnvironment符合OpenAI Gym接口
适配同学编写的训练代码
"""

import numpy as np
from typing import Tuple
from gym import spaces

from .factory_environment import LayoutEnvironment


class FactoryEnv:
    """
    Gym风格的环境包装器
    
    将LayoutEnvironment包装成符合OpenAI Gym接口的环境，
    以便与现有的训练代码兼容。
    
    主要适配内容：
    1. 状态：从字典格式转换为单一数组
    2. 动作：从字典格式转换为整数索引
    3. 添加observation_space和action_space属性
    """
    
    def __init__(
        self,
        config_path: str = "simulation/configs/chair_factory.json",
        use_simulation: bool = False,  # 默认关闭仿真，加速训练
        simulation_duration: float = 20000,  # 1天 = 20000时间单位（400个椅子/天）
        objective_weights: dict = None,  # 自定义奖励权重
        placement_order: str = "default",  # 摆放顺序策略
        layout_path: str = None  # 自定义布局文件路径（用于并行实验隔离）
    ):
        """
        初始化包装器环境
        
        Args:
            config_path: 仿真配置文件路径
            use_simulation: 是否使用仿真计算奖励
            simulation_duration: 仿真时长（1天 = 20000时间单位，产能400个椅子/天）
            objective_weights: 自定义奖励权重字典，如 {'transportation_intensity': 0.2, ...}
            placement_order: 摆放顺序策略
                - 'default': 配置文件中的顺序
                - 'size_desc': 按面积从大到小
                - 'size_asc': 按面积从小到大
                - 'flow_desc': 按物料流连接数从多到少
                - 'random': 随机顺序
            layout_path: 自定义布局文件路径（用于并行实验隔离）
        """
        # 创建底层环境
        self.env = LayoutEnvironment.from_config(
            config_path=config_path,
            use_simulation=use_simulation,
            simulation_duration=simulation_duration,
            objective_weights=objective_weights,
            placement_order=placement_order,
            layout_path=layout_path
        )
        
        # 保存环境参数
        self.nx, self.ny = self.env.grid_size
        self.num_units = self.env.num_units
        
        # 动作空间：(x, y, rotation) 的组合
        # x ∈ [0, nx), y ∈ [0, ny), rotation ∈ {0, 90, 180, 270}
        self.num_rotations = 4
        self._action_space_n = self.nx * self.ny * self.num_rotations
        
        # 定义动作空间（整数索引）
        self.action_space = spaces.Discrete(self._action_space_n)
        
        # 定义观测空间（flatten后的状态向量）
        # 状态包含：layout_grid + material_flow + current_unit + placed_mask
        layout_dim = self.nx * self.ny
        material_flow_dim = self.num_units * self.num_units
        current_unit_dim = self.num_units
        placed_mask_dim = self.num_units
        restricted_dim = self.nx * self.ny
        
        total_dim = (
            layout_dim + 
            material_flow_dim + 
            current_unit_dim + 
            placed_mask_dim +
            restricted_dim
        )
        
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_dim,),
            dtype=np.float32
        )
        
        # 保存状态维度信息（供后续使用）
        self.state_dims = {
            'layout': layout_dim,
            'material_flow': material_flow_dim,
            'current_unit': current_unit_dim,
            'placed_mask': placed_mask_dim,
            'restricted': restricted_dim
        }
        
        # 缓存有效动作映射（用于加速）
        self._valid_actions_cache = None
    
    def reset(self) -> np.ndarray:
        """
        重置环境
        
        Returns:
            state: 扁平化的状态向量 [total_dim]
        """
        state_dict = self.env.reset()
        self._valid_actions_cache = None  # 清空缓存
        return self._dict_to_array(state_dict)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行一步
        
        Args:
            action: 动作索引（整数）
        
        Returns:
            next_state: 下一个状态向量
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        # 将整数动作转换为字典格式
        action_dict = self._action_idx_to_dict(action)
        
        # 执行动作
        next_state_dict, reward, done, info = self.env.step(action_dict)
        
        # 转换状态为数组格式
        next_state = self._dict_to_array(next_state_dict)
        
        # 清空有效动作缓存（因为状态已改变）
        self._valid_actions_cache = None
        
        return next_state, reward, done, info
    
    def get_valid_actions(self) -> list:
        """
        获取当前状态下的有效动作索引
        
        Returns:
            valid_action_indices: 有效动作的索引列表
        """
        if self._valid_actions_cache is not None:
            return self._valid_actions_cache
        
        # 从底层环境获取有效动作（字典格式）
        valid_actions_dict = self.env.get_valid_actions()
        
        # 转换为索引格式
        valid_indices = [
            self._action_dict_to_idx(a) 
            for a in valid_actions_dict
        ]
        
        self._valid_actions_cache = valid_indices
        return valid_indices
    
    def _dict_to_array(self, state_dict: dict) -> np.ndarray:
        """
        将字典格式的状态转换为扁平数组
        
        Args:
            state_dict: 状态字典
                {
                    'layout_grid': [nx, ny],
                    'material_flow': [N, N],
                    'current_unit': [N],
                    'placed_mask': [N],
                    'restricted_areas': [nx, ny]
                }
        
        Returns:
            state_array: 扁平化的状态向量 [total_dim]
        """
        # 展平各个部分
        layout_flat = state_dict['layout_grid'].flatten()
        material_flow_flat = state_dict['material_flow'].flatten()
        current_unit_flat = state_dict['current_unit'].flatten()
        placed_mask_flat = state_dict['placed_mask'].flatten()
        restricted_flat = state_dict['restricted_areas'].flatten()
        
        # 拼接成一个向量
        state_array = np.concatenate([
            layout_flat,
            material_flow_flat,
            current_unit_flat,
            placed_mask_flat,
            restricted_flat
        ]).astype(np.float32)
        
        return state_array
    
    def _action_idx_to_dict(self, action_idx: int) -> dict:
        """
        将动作索引转换为字典格式
        
        动作编码：action_idx = x * (ny * 4) + y * 4 + rotation_idx
        其中 rotation_idx ∈ {0, 1, 2, 3} 对应 {0°, 90°, 180°, 270°}
        
        Args:
            action_idx: 动作索引
        
        Returns:
            action_dict: {'x': int, 'y': int, 'rotation': int}
        """
        # 解码动作
        rotation_idx = action_idx % 4
        temp = action_idx // 4
        y = temp % self.ny
        x = temp // self.ny
        
        rotation = rotation_idx * 90  # 0, 90, 180, 270
        
        return {'x': x, 'y': y, 'rotation': rotation}
    
    def _action_dict_to_idx(self, action_dict: dict) -> int:
        """
        将字典格式的动作转换为索引
        
        Args:
            action_dict: {'x': int, 'y': int, 'rotation': int}
        
        Returns:
            action_idx: 动作索引
        """
        x = action_dict['x']
        y = action_dict['y']
        rotation = action_dict['rotation']
        
        rotation_idx = rotation // 90  # 0, 90, 180, 270 -> 0, 1, 2, 3
        
        action_idx = x * (self.ny * 4) + y * 4 + rotation_idx
        
        return action_idx
    
    def render(self, mode='human'):
        """渲染环境（可选）"""
        return self.env.render()
    
    def close(self):
        """关闭环境（可选）"""
        pass


# 为了向后兼容，创建一个别名
class GymFactoryEnv(FactoryEnv):
    """别名，保持一致性"""
    pass


if __name__ == "__main__":
    """测试包装器"""
    print("="*70)
    print("测试 FactoryEnv Gym包装器")
    print("="*70)
    
    # 创建环境
    env = FactoryEnv(
        config_path="simulation/configs/chair_factory.json",
        use_simulation=False
    )
    
    print(f"\n环境信息：")
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")
    print(f"  动作空间大小: {env.action_space.n}")
    print(f"  网格尺寸: {env.nx} x {env.ny}")
    print(f"  功能单元数: {env.num_units}")
    
    # 重置环境
    print(f"\n重置环境...")
    state = env.reset()
    print(f"  初始状态形状: {state.shape}")
    print(f"  初始状态范围: [{state.min():.3f}, {state.max():.3f}]")
    
    # 获取有效动作
    print(f"\n获取有效动作...")
    valid_actions = env.get_valid_actions()
    print(f"  有效动作数量: {len(valid_actions)}")
    print(f"  总动作空间: {env.action_space.n}")
    print(f"  有效动作比例: {len(valid_actions) / env.action_space.n * 100:.1f}%")
    
    # 执行一些步骤
    print(f"\n执行测试步骤...")
    for i in range(3):
        if len(valid_actions) == 0:
            print(f"  步骤 {i+1}: 没有有效动作")
            break
        
        # 随机选择一个有效动作
        action = np.random.choice(valid_actions)
        action_dict = env._action_idx_to_dict(action)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        print(f"  步骤 {i+1}:")
        print(f"    动作索引: {action}")
        print(f"    动作详情: x={action_dict['x']}, y={action_dict['y']}, θ={action_dict['rotation']}°")
        print(f"    奖励: {reward:.3f}")
        print(f"    完成: {done}")
        print(f"    已放置: {info['placed_units']}/{info['total_units']}")
        
        if done:
            break
        
        # 更新有效动作
        valid_actions = env.get_valid_actions()
        state = next_state
    
    print(f"\n" + "="*70)
    print("✓ 包装器测试完成！")
    print("="*70)
    
    # 测试动作编码/解码
    print(f"\n测试动作编码/解码:")
    test_actions = [
        {'x': 0, 'y': 0, 'rotation': 0},
        {'x': 10, 'y': 5, 'rotation': 90},
        {'x': 20, 'y': 15, 'rotation': 180},
    ]
    
    for act_dict in test_actions:
        idx = env._action_dict_to_idx(act_dict)
        recovered = env._action_idx_to_dict(idx)
        match = (act_dict == recovered)
        print(f"  {act_dict} → idx={idx} → {recovered} [{'✓' if match else '✗'}]")

