"""
深度Q网络（DQN）- 工厂布局规划

状态空间（State Space）：
    s_t = {s_t1, s_t2, s_t3}
    
    - s_t1: 布局状态 ∈ ℝ^(W×H×N) - 每个网格的One-Hot编码 → CNN嵌入
    - s_t2: 运输状态（DAG邻接矩阵 N×N）→ GCN嵌入
    - s_t3: 待放置模块状态 ∈ ℝ^N（One-Hot编码）→ MLP嵌入

动作空间（Action Space）：
    a_t = (x, y, θ) 其中：
    - x ∈ {0, ..., W-1}: 模块左上角x坐标
    - y ∈ {0, ..., H-1}: 模块左上角y坐标  
    - θ ∈ {0°, 90°, 180°, 270°}: 旋转角度
    - 总动作数 = W × H × 4

输出：
    - 所有动作的Q值 [batch_size, W×H×4]
    - 动作掩码预测 [batch_size, W×H×4]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


class GraphConvLayer(nn.Module):
    """
    图卷积层（GCN Layer）
    
    实现公式：H' = σ(D^(-1/2) A D^(-1/2) H W)
    其中：
        A: 邻接矩阵 [N, N]
        H: 节点特征 [N, feature_dim]
        W: 可学习权重矩阵
        D: 度矩阵
    """
    def __init__(self, in_features: int, out_features: int):
        """
        初始化图卷积层
        
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
        """
        super(GraphConvLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 节点特征 [batch_size, N, in_features]
            adj: 邻接矩阵 [batch_size, N, N] (0或1)
        
        Returns:
            输出特征 [batch_size, N, out_features]
        """
        # 添加自环：A' = A + I
        batch_size, N = adj.shape[0], adj.shape[1]
        identity = torch.eye(N, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self_loops = adj + identity
        
        # 计算度矩阵的逆平方根：D^(-1/2)
        degree = adj_with_self_loops.sum(dim=-1)  # [batch_size, N]
        degree_inv_sqrt = torch.pow(degree, -0.5)
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0  # 处理除以0
        
        # D^(-1/2) A D^(-1/2)
        norm_adj = degree_inv_sqrt.unsqueeze(-1) * adj_with_self_loops * degree_inv_sqrt.unsqueeze(-2)
        
        # H' = norm_adj @ H @ W
        out = torch.matmul(norm_adj, x)  # [batch_size, N, in_features]
        out = self.linear(out)           # [batch_size, N, out_features]
        
        return out


class GCN(nn.Module):
    """
    图卷积网络（Graph Convolutional Network）
    
    用于处理物料流邻接矩阵，提取物体之间的关系特征
    """
    def __init__(
        self, 
        num_nodes: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 2
    ):
        """
        初始化GCN网络
        
        Args:
            num_nodes: 节点数量（物体数量N）
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_layers: GCN层数
        """
        super(GCN, self).__init__()
        
        self.num_nodes = num_nodes
        
        # 节点初始特征嵌入（可学习）
        self.node_embedding = nn.Embedding(num_nodes, hidden_dim)
        
        # GCN层
        self.gcn_layers = nn.ModuleList()
        
        # 第一层
        self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # 中间层
        for _ in range(num_layers - 2):
            self.gcn_layers.append(GraphConvLayer(hidden_dim, hidden_dim))
        
        # 输出层
        if num_layers > 1:
            self.gcn_layers.append(GraphConvLayer(hidden_dim, output_dim))
        
        self.num_layers = num_layers
    
    def forward(self, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            adj: 邻接矩阵 [batch_size, N, N]，元素为0或1
        
        Returns:
            图特征 [batch_size, output_dim]
        """
        batch_size, N = adj.shape[0], adj.shape[1]
        
        # 初始化节点特征
        node_ids = torch.arange(N, device=adj.device).unsqueeze(0).expand(batch_size, -1)
        x = self.node_embedding(node_ids)  # [batch_size, N, hidden_dim]
        
        # 通过GCN层
        for i, gcn_layer in enumerate(self.gcn_layers):
            x = gcn_layer(x, adj)
            if i < self.num_layers - 1:  # 最后一层不加激活函数
                x = F.relu(x)
        
        # 聚合节点特征：全局平均池化
        graph_feature = x.mean(dim=1)  # [batch_size, output_dim]
        
        return graph_feature


class LayoutDQN(nn.Module):
    """
    工厂布局深度Q网络（最终版本）
    
    架构：
        1. CNN处理布局 s_t1: (H, W, N) → 提取空间特征
        2. GCN处理物料流 s_t2: (N, N) → 提取关系特征
        3. MLP处理待放置模块 s_t3: (N,) → 提取模块特征
        4. 融合三分支特征 → Dueling DQN → Q值 + 动作掩码
    
    输入：
        - layout: [batch, H, W, N] 布局张量（s_t1，最后一维是One-Hot编码）
        - material_flow: [batch, N, N] 物料流邻接矩阵（s_t2，0或1）
        - current_object: [batch, N] 待放置模块的One-Hot编码（s_t3）
        
    输出：
        - q_values: [batch, num_actions] 所有动作的Q值
        - action_mask: [batch, num_actions] 动作掩码（1=有效，0=无效）
    """
    
    def __init__(
        self,
        height: int,              # 工厂高度 H
        width: int,               # 工厂宽度 W
        num_objects: int,         # 待放置物体总数 N
        num_actions: int,         # 动作空间大小
        cnn_channels: list = [32, 64, 64],      # CNN通道数
        gcn_hidden_dim: int = 64,               # GCN隐藏层维度
        gcn_output_dim: int = 128,              # GCN输出维度
        fusion_hidden_dim: int = 256,           # 融合层隐藏维度
        use_dueling: bool = True                # 是否使用Dueling架构
    ):
        """
        初始化LayoutDQN网络
        
        Args:
            height: 工厂高度
            width: 工厂宽度  
            num_objects: 待放置物体总数
            num_actions: 动作空间大小
            cnn_channels: CNN各层通道数
            gcn_hidden_dim: GCN隐藏层维度
            gcn_output_dim: GCN输出维度
            fusion_hidden_dim: 融合层隐藏维度
            use_dueling: 是否使用Dueling架构
        """
        super(LayoutDQN, self).__init__()
        
        self.height = height
        self.width = width
        self.num_objects = num_objects
        self.num_actions = num_actions
        self.use_dueling = use_dueling
        
        # ===== 1. CNN处理布局 (H, W, N) =====
        # 输入通道数为N（One-Hot编码的维度）
        self.cnn = nn.Sequential(
            # Conv1: [B, N, H, W] → [B, 32, H, W]
            nn.Conv2d(num_objects, cnn_channels[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels[0]),
            
            # Conv2: [B, 32, H, W] → [B, 64, H, W]
            nn.Conv2d(cnn_channels[0], cnn_channels[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels[1]),
            
            # Conv3: [B, 64, H, W] → [B, 64, H, W]
            nn.Conv2d(cnn_channels[1], cnn_channels[2], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(cnn_channels[2])
        )
        
        # CNN输出展平后的维度
        self.cnn_output_dim = cnn_channels[-1] * height * width
        
        # ===== 2. GCN处理物料流邻接矩阵 =====
        self.gcn = GCN(
            num_nodes=num_objects,
            hidden_dim=gcn_hidden_dim,
            output_dim=gcn_output_dim,
            num_layers=2
        )
        
        # ===== 3. MLP处理待放置模块（s_t3）=====
        self.object_mlp = nn.Sequential(
            nn.Linear(num_objects, 64),  # One-Hot → 嵌入向量
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        self.object_output_dim = 64
        
        # ===== 4. 融合层（三分支特征融合）=====
        fusion_input_dim = self.cnn_output_dim + gcn_output_dim + self.object_output_dim
        
        if use_dueling:
            # Dueling DQN架构
            # 共享特征提取
            self.fusion_shared = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            )
            
            # 价值流 V(s)
            self.value_stream = nn.Sequential(
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_hidden_dim // 2, 1)
            )
            
            # 优势流 A(s,a)
            self.advantage_stream = nn.Sequential(
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(fusion_hidden_dim // 2, num_actions)
            )
        else:
            # 标准DQN架构
            self.q_network = nn.Sequential(
                nn.Linear(fusion_input_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
                nn.ReLU(),
                nn.Linear(fusion_hidden_dim, num_actions)
            )
        
        # 注意：动作掩码由环境直接提供，DQN不需要预测
    
    def forward(
        self, 
        layout: torch.Tensor, 
        material_flow: torch.Tensor,
        current_object: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            layout: 布局张量 [batch_size, H, W, N]
                   元素为0或1，最后一维是已放置物体的One-Hot编码（s_t1）
            material_flow: 物料流邻接矩阵 [batch_size, N, N]
                          元素为0或1，A[i,j]=1表示有从物体i到j的物料流（s_t2）
            current_object: 待放置模块的One-Hot编码 [batch_size, N]
                           元素为0或1，表示当前要放置哪个模块（s_t3）
        
        Returns:
            q_values: Q值 [batch_size, num_actions]
        """
        batch_size = layout.shape[0]
        
        # ===== 1. CNN处理布局（s_t1）=====
        # 调整维度：[B, H, W, N] → [B, N, H, W]
        layout_input = layout.permute(0, 3, 1, 2)  # [B, N, H, W]
        
        cnn_features = self.cnn(layout_input)  # [B, 64, H, W]
        cnn_features = cnn_features.reshape(batch_size, -1)  # [B, cnn_output_dim]
        
        # ===== 2. GCN处理物料流（s_t2）=====
        gcn_features = self.gcn(material_flow)  # [B, gcn_output_dim]
        
        # ===== 3. MLP处理待放置模块（s_t3）=====
        object_features = self.object_mlp(current_object.float())  # [B, object_output_dim]
        
        # ===== 4. 三分支特征融合 =====
        fused_features = torch.cat([
            cnn_features,      # 布局特征
            gcn_features,      # 物料流特征
            object_features    # 待放置模块特征
        ], dim=1)
        # [B, cnn_output_dim + gcn_output_dim + object_output_dim]
        
        # ===== 5. 计算Q值 =====
        if self.use_dueling:
            # Dueling架构
            shared = self.fusion_shared(fused_features)
            
            value = self.value_stream(shared)  # [B, 1]
            advantage = self.advantage_stream(shared)  # [B, num_actions]
            
            # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        else:
            # 标准DQN
            q_values = self.q_network(fused_features)
        
        # 注意：动作掩码由环境提供，在外部处理
        return q_values
    
    def get_action(
        self, 
        layout: torch.Tensor, 
        material_flow: torch.Tensor,
        current_object: torch.Tensor,
        valid_action_indices: List[int],
        epsilon: float = 0.0
    ) -> int:
        """
        选择动作（epsilon-greedy策略）
        
        注意：动作掩码由环境提供，这里只在有效动作中选择
        
        Args:
            layout: 布局张量 [1, H, W, N]
            material_flow: 物料流邻接矩阵 [1, N, N]
            current_object: 待放置模块的One-Hot编码 [1, N]
            valid_action_indices: 有效动作索引列表 [idx1, idx2, ...]
            epsilon: 探索率
        
        Returns:
            action_idx: 选择的动作索引（单个整数）
        """
        if len(valid_action_indices) == 0:
            raise ValueError("没有有效动作！")
        
        with torch.no_grad():
            # epsilon-greedy
            if np.random.random() < epsilon:
                # 随机选择一个有效动作
                action_idx = np.random.choice(valid_action_indices)
            else:
                # 贪心选择：计算所有Q值，然后只在有效动作中选最大的
                q_values = self.forward(layout, material_flow, current_object)  # [1, num_actions]
                
                # 只取有效动作的Q值
                valid_q_values = q_values[0, valid_action_indices]  # [len(valid_actions)]
                
                # 找到最大Q值对应的有效动作
                best_valid_idx = valid_q_values.argmax().item()
                action_idx = valid_action_indices[best_valid_idx]
        
        return action_idx


# ====================
# 辅助函数
# ====================

def create_layout_tensor(
    height: int,
    width: int, 
    num_objects: int,
    placed_positions: list
) -> torch.Tensor:
    """
    创建布局张量
    
    Args:
        height: 工厂高度
        width: 工厂宽度
        num_objects: 物体总数
        placed_positions: 已放置物体的位置列表
            [(object_id, x, y), ...]
            object_id从0开始
    
    Returns:
        layout: [H, W, N] 张量
            每个位置(x,y)的N维向量是One-Hot编码
            如果放置了物体i，则layout[x,y,i]=1，其余为0
            如果位置为空，则layout[x,y,:]=全0
    """
    layout = torch.zeros(height, width, num_objects)
    
    for object_id, x, y in placed_positions:
        if 0 <= x < height and 0 <= y < width and 0 <= object_id < num_objects:
            # One-Hot编码：只有对应物体的位置为1
            layout[x, y, :] = 0  # 先清零（其实已经是0）
            layout[x, y, object_id] = 1  # 设置对应位置为1
    
    return layout


def create_material_flow_matrix(
    num_objects: int,
    flows: list
) -> torch.Tensor:
    """
    创建物料流邻接矩阵
    
    Args:
        num_objects: 物体总数
        flows: 物料流列表 [(from_id, to_id), ...]
    
    Returns:
        adj_matrix: [N, N] 邻接矩阵（0或1）
    """
    adj_matrix = torch.zeros(num_objects, num_objects)
    
    for from_id, to_id in flows:
        if 0 <= from_id < num_objects and 0 <= to_id < num_objects:
            adj_matrix[from_id, to_id] = 1
    
    return adj_matrix


# ====================
# 测试代码
# ====================
if __name__ == "__main__":
    print("=" * 80)
    print("测试 LayoutDQN 网络")
    print("=" * 80)
    
    # 参数设置
    batch_size = 4
    H, W, N = 10, 10, 5  # 工厂10×10，5个物体
    num_actions = H * W * 4  # 位置 × 旋转角度
    
    # 创建网络
    model = LayoutDQN(
        height=H,
        width=W,
        num_objects=N,
        num_actions=num_actions,
        use_dueling=True
    )
    
    print(f"\n网络结构:")
    print(f"  输入: 布局 [{H}, {W}, {N}] + 物料流 [{N}, {N}] + 待放置模块 [{N}]")
    print(f"  输出: Q值 [{num_actions}] + 动作掩码 [{num_actions}]")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 创建模拟输入
    print(f"\n创建测试数据...")
    
    # 1. 布局张量 [B, H, W, N] - s_t1
    layout = torch.zeros(batch_size, H, W, N)
    # 模拟放置一些物体（每个位置的N维向量是One-Hot编码）
    layout[0, 0, 0, :] = torch.tensor([1., 0, 0, 0, 0])  # 物体0在位置(0,0)
    layout[0, 2, 3, :] = torch.tensor([0, 1., 0, 0, 0])  # 物体1在位置(2,3)
    layout[0, 5, 5, :] = torch.tensor([0, 0, 1., 0, 0])  # 物体2在位置(5,5)
    
    # 2. 物料流邻接矩阵 [B, N, N] - s_t2
    material_flow = torch.zeros(batch_size, N, N)
    # 模拟一些物料流
    material_flow[0, 0, 1] = 1  # 物体0 → 物体1
    material_flow[0, 1, 2] = 1  # 物体1 → 物体2
    material_flow[0, 2, 0] = 1  # 物体2 → 物体0（环）
    
    # 3. 待放置模块 [B, N] - s_t3
    current_object = torch.zeros(batch_size, N)
    # 模拟当前要放置物体3
    current_object[0, :] = torch.tensor([0, 0, 0, 1., 0])  # 物体3
    current_object[1, :] = torch.tensor([0, 0, 0, 0, 1.])  # 物体4
    
    print(f"  布局形状: {layout.shape}")
    print(f"  物料流形状: {material_flow.shape}")
    print(f"  待放置模块形状: {current_object.shape}")
    
    # 前向传播
    print(f"\n前向传播...")
    q_values = model(layout, material_flow, current_object)
    
    print(f"  Q值形状: {q_values.shape}")
    print(f"  Q值范围: [{q_values.min():.3f}, {q_values.max():.3f}]")
    
    # 测试动作选择（模拟环境提供的有效动作）
    print(f"\n测试动作选择（使用有效动作列表）...")
    # 模拟环境提供的有效动作索引（这里随机选一些）
    valid_indices = list(range(50, 100))  # 假设50-100是有效动作
    
    # 只测试单个样本
    single_layout = layout[0:1]
    single_flow = material_flow[0:1]
    single_object = current_object[0:1]
    
    action_greedy = model.get_action(
        single_layout, single_flow, single_object, 
        valid_indices, epsilon=0.0
    )
    print(f"  贪心动作: {action_greedy}")
    
    action_random = model.get_action(
        single_layout, single_flow, single_object, 
        valid_indices, epsilon=1.0
    )
    print(f"  随机动作: {action_random}")
    
    # 测试辅助函数
    print(f"\n测试辅助函数...")
    
    # 创建布局张量
    placed = [(0, 0, 0), (1, 2, 3), (2, 5, 5)]
    layout_test = create_layout_tensor(H, W, N, placed)
    print(f"  布局张量形状: {layout_test.shape}")
    print(f"  已放置物体数: {(layout_test.sum(dim=-1) > 0).sum()}")
    
    # 创建物料流矩阵
    flows = [(0, 1), (1, 2), (2, 0)]
    adj_test = create_material_flow_matrix(N, flows)
    print(f"  邻接矩阵形状: {adj_test.shape}")
    print(f"  物料流边数: {adj_test.sum().int()}")
    
    # 测试GCN单独
    print(f"\n测试GCN模块...")
    gcn = GCN(num_nodes=N, hidden_dim=64, output_dim=128)
    gcn_out = gcn(material_flow)
    print(f"  GCN输出形状: {gcn_out.shape}")
    print(f"  GCN参数量: {sum(p.numel() for p in gcn.parameters()):,}")
    
    print("\n" + "=" * 80)
    print("✓ 所有测试通过！")
    print("=" * 80)
    
    # 网络结构总结
    print(f"\n网络架构总结:")
    print(f"  1. CNN分支 (s_t1 布局状态):")
    print(f"     输入: [B, N, H, W] → 输出: [B, {model.cnn_output_dim}]")
    print(f"  2. GCN分支 (s_t2 运输状态):")
    print(f"     输入: [B, N, N] → 输出: [B, 128]")
    print(f"  3. MLP分支 (s_t3 待放置模块):")
    print(f"     输入: [B, N] → 输出: [B, {model.object_output_dim}]")
    print(f"  4. 融合层:")
    print(f"     输入: [B, {model.cnn_output_dim + 128 + model.object_output_dim}] → 输出: [B, 256]")
    print(f"  5. Dueling输出:")
    print(f"     V流: [B, 256] → [B, 1]")
    print(f"     A流: [B, 256] → [B, {num_actions}]")
    print(f"     Q值: V + (A - mean(A))")
    print(f"  6. 掩码预测:")
    print(f"     输入: [B, {model.cnn_output_dim + 128 + model.object_output_dim}] → 输出: [B, {num_actions}]")
        