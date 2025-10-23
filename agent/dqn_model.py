"""
深度Q网络 - 价值网络
基于文献《Transferable multi-objective factory layout planning》实现

输入：工厂布局状态
输出：所有动作的Q值
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DQN(nn.Module):
    """
    深度Q网络 - 基础版本
    
    适用于简单的状态表示（向量形式）
    """
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int,
        hidden_dims: list = [256, 256]
    ):
        """
        初始化DQN网络
        
        Args:
            state_dim: 状态维度（输入维度）
            action_dim: 动作数量（输出维度）
            hidden_dims: 隐藏层维度列表
        """
        super(DQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 构建全连接网络
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(input_dim, action_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 状态张量 [batch_size, state_dim]
        
        Returns:
            Q值 [batch_size, action_dim]
        """
        return self.network(x)


class DuelingDQN(nn.Module):
    """
    Dueling DQN网络
    
    将Q值分解为状态价值V(s)和优势函数A(s,a)
    Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
    
    参考：Wang et al. 2015, Dueling Network Architectures for Deep RL
    """
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256]
    ):
        """
        初始化Dueling DQN
        
        Args:
            state_dim: 状态维度
            action_dim: 动作数量
            hidden_dims: 隐藏层维度
        """
        super(DuelingDQN, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 共享特征提取层
        feature_layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims[:-1]:
            feature_layers.append(nn.Linear(input_dim, hidden_dim))
            feature_layers.append(nn.ReLU())
            input_dim = hidden_dim
        
        self.feature_extractor = nn.Sequential(*feature_layers)
        
        # 状态价值流 V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # 优势流 A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], action_dim)
        )
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 状态张量 [batch_size, state_dim]
        
        Returns:
            Q值 [batch_size, action_dim]
        """
        # 提取特征
        features = self.feature_extractor(x)
        
        # 计算状态价值和优势
        value = self.value_stream(features)  # [batch_size, 1]
        advantage = self.advantage_stream(features)  # [batch_size, action_dim]
        
        # 组合为Q值：Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        # 减去平均优势提高稳定性
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class LayoutDQN(nn.Module):
    """
    工厂布局专用DQN网络
    
    基于文献设计，使用CNN处理布局网格 + MLP处理物料流
    适用于复杂的状态表示
    """
    def __init__(
        self,
        grid_size: tuple,           # 网格尺寸 (height, width)
        num_units: int,             # 功能单元数量
        action_dim: int,            # 动作数量
        hidden_dim: int = 256,
        use_dueling: bool = True
    ):
        """
        初始化布局DQN网络
        
        Args:
            grid_size: 布局网格尺寸 (height, width)
            num_units: 功能单元数量
            action_dim: 动作空间大小
            hidden_dim: 隐藏层维度
            use_dueling: 是否使用Dueling架构
        """
        super(LayoutDQN, self).__init__()
        
        self.grid_size = grid_size
        self.num_units = num_units
        self.action_dim = action_dim
        self.use_dueling = use_dueling
        
        # ====================
        # 1. CNN处理布局网格
        # ====================
        self.conv_layers = nn.Sequential(
            # 输入: [batch, 1, height, width]
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # 计算CNN输出维度
        conv_output_size = 64 * grid_size[0] * grid_size[1]
        
        # ====================
        # 2. MLP处理物料流矩阵
        # ====================
        material_flow_size = num_units * num_units
        
        self.material_flow_mlp = nn.Sequential(
            nn.Linear(material_flow_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        
        # ====================
        # 3. MLP处理单元信息
        # ====================
        unit_info_size = num_units * 2  # current_unit + placed_mask
        
        self.unit_info_mlp = nn.Sequential(
            nn.Linear(unit_info_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # ====================
        # 4. 融合层
        # ====================
        fusion_input_size = conv_output_size + 128 + 64
        
        if use_dueling:
            # Dueling架构
            self.value_stream = nn.Sequential(
                nn.Linear(fusion_input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            self.advantage_stream = nn.Sequential(
                nn.Linear(fusion_input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        else:
            # 标准架构
            self.q_network = nn.Sequential(
                nn.Linear(fusion_input_size, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
    
    def forward(self, state):
        """
        前向传播
        
        Args:
            state: 可以是：
                1. 字典格式 {'layout_grid': tensor, 'material_flow': tensor, ...}
                2. 单一张量（自动处理）
        
        Returns:
            Q值 [batch_size, action_dim]
        """
        # 如果输入是字典（完整状态）
        if isinstance(state, dict):
            return self._forward_dict(state)
        # 如果输入是单一张量（简化状态）
        else:
            return self._forward_tensor(state)
    
    def _forward_dict(self, state):
        """处理字典格式的状态"""
        # 1. 处理布局网格
        layout_grid = state['layout_grid']
        if layout_grid.dim() == 3:
            layout_grid = layout_grid.unsqueeze(1)  # [B, 1, H, W]
        
        conv_features = self.conv_layers(layout_grid)
        conv_features = conv_features.view(conv_features.size(0), -1)
        
        # 2. 处理物料流
        material_flow = state['material_flow']
        material_flow_flat = material_flow.view(material_flow.size(0), -1)
        mf_features = self.material_flow_mlp(material_flow_flat)
        
        # 3. 处理单元信息
        current_unit = state['current_unit']
        placed_mask = state['placed_mask']
        unit_info = torch.cat([current_unit, placed_mask], dim=1)
        unit_features = self.unit_info_mlp(unit_info)
        
        # 4. 融合特征
        fused_features = torch.cat([conv_features, mf_features, unit_features], dim=1)
        
        # 5. 计算Q值
        return self._compute_q_values(fused_features)
    
    def _forward_tensor(self, state):
        """处理简化的张量状态（向量形式）"""
        # 如果是简化状态，直接通过融合层
        # 这里假设状态已经是展平的向量
        return self._compute_q_values(state)
    
    def _compute_q_values(self, features):
        """从融合特征计算Q值"""
        if self.use_dueling:
            # Dueling架构
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # 标准架构
            q_values = self.q_network(features)
        
        return q_values


# ====================
# 工具函数
# ====================
def create_dqn(
    state_dim=None,
    action_dim=None,
    grid_size=None,
    num_units=None,
    network_type='simple',
    use_dueling=False,
    **kwargs
):
    """
    工厂函数：创建DQN网络
    
    Args:
        state_dim: 状态维度（简单DQN）
        action_dim: 动作数量
        grid_size: 网格尺寸（布局DQN）
        num_units: 功能单元数量（布局DQN）
        network_type: 'simple', 'dueling', 'layout'
        use_dueling: 是否使用Dueling架构
        **kwargs: 其他参数
    
    Returns:
        DQN网络实例
    """
    if network_type == 'simple':
        return DQN(state_dim, action_dim, **kwargs)
    
    elif network_type == 'dueling':
        return DuelingDQN(state_dim, action_dim, **kwargs)
    
    elif network_type == 'layout':
        return LayoutDQN(
            grid_size=grid_size,
            num_units=num_units,
            action_dim=action_dim,
            use_dueling=use_dueling,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown network type: {network_type}")


# ====================
# 测试代码
# ====================
if __name__ == "__main__":
    print("测试DQN价值网络...\n")
    
    # 测试1: 简单DQN
    print("=" * 60)
    print("测试1: 简单DQN")
    print("=" * 60)
    
    simple_dqn = DQN(state_dim=100, action_dim=10)
    test_input = torch.randn(4, 100)  # batch_size=4
    q_values = simple_dqn(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出Q值形状: {q_values.shape}")
    print(f"Q值范围: [{q_values.min():.3f}, {q_values.max():.3f}]")
    print(f"网络参数量: {sum(p.numel() for p in simple_dqn.parameters()):,}\n")
    
    # 测试2: Dueling DQN
    print("=" * 60)
    print("测试2: Dueling DQN")
    print("=" * 60)
    
    dueling_dqn = DuelingDQN(state_dim=100, action_dim=10)
    q_values_dueling = dueling_dqn(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出Q值形状: {q_values_dueling.shape}")
    print(f"Q值范围: [{q_values_dueling.min():.3f}, {q_values_dueling.max():.3f}]")
    print(f"网络参数量: {sum(p.numel() for p in dueling_dqn.parameters()):,}\n")
    
    # 测试3: 布局DQN
    print("=" * 60)
    print("测试3: 布局DQN（完整状态）")
    print("=" * 60)
    
    layout_dqn = LayoutDQN(
        grid_size=(20, 20),
        num_units=5,
        action_dim=20*20*4,  # 位置 * 旋转
        use_dueling=True
    )
    
    # 创建测试状态（字典格式）
    batch_size = 4
    test_state = {
        'layout_grid': torch.randn(batch_size, 20, 20),
        'material_flow': torch.randn(batch_size, 5, 5),
        'current_unit': torch.randn(batch_size, 5),
        'placed_mask': torch.randn(batch_size, 5)
    }
    
    q_values_layout = layout_dqn(test_state)
    
    print(f"输入状态:")
    for key, value in test_state.items():
        print(f"  {key}: {value.shape}")
    print(f"输出Q值形状: {q_values_layout.shape}")
    print(f"Q值范围: [{q_values_layout.min():.3f}, {q_values_layout.max():.3f}]")
    print(f"网络参数量: {sum(p.numel() for p in layout_dqn.parameters()):,}\n")
    
    # 测试4: 使用工厂函数
    print("=" * 60)
    print("测试4: 使用工厂函数创建网络")
    print("=" * 60)
    
    net1 = create_dqn(state_dim=50, action_dim=5, network_type='simple')
    net2 = create_dqn(state_dim=50, action_dim=5, network_type='dueling')
    net3 = create_dqn(
        grid_size=(15, 15),
        num_units=5,
        action_dim=100,
        network_type='layout',
        use_dueling=True
    )
    
    print(f"简单DQN参数量: {sum(p.numel() for p in net1.parameters()):,}")
    print(f"Dueling DQN参数量: {sum(p.numel() for p in net2.parameters()):,}")
    print(f"布局DQN参数量: {sum(p.numel() for p in net3.parameters()):,}")
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
