# 🧠 LayoutDQN 网络使用指南

## 📋 概述

`LayoutDQN` 是专门为工厂布局规划设计的深度Q网络，融合了**CNN**、**GCN**和**MLP**三种深度学习架构，完整实现了状态空间 `s_t = {s_t1, s_t2, s_t3}`。

### **核心特性**

✅ **CNN处理空间布局 (s_t1)** - 使用卷积神经网络提取工厂布局的空间特征  
✅ **GCN处理物料流 (s_t2)** - 使用图卷积网络学习物体间的物料流关系  
✅ **MLP处理待放置模块 (s_t3)** - 使用多层感知机编码当前待放置模块信息  
✅ **Dueling架构** - 分离状态价值和动作优势，提升学习效率  
✅ **动作掩码** - 自动预测并过滤无效动作  

---

## 🏗️ 网络架构

```
输入层（三分支）
  ├─ s_t1: 布局张量 [B, H, W, N]
  │    └─ CNN提取空间特征 → [B, 64*H*W]
  │
  ├─ s_t2: 物料流矩阵 [B, N, N]
  │    └─ GCN提取关系特征 → [B, 128]
  │
  └─ s_t3: 待放置模块 [B, N]
       └─ MLP提取模块特征 → [B, 64]

融合层
  └─ 三分支特征拼接 → [B, 64*H*W + 128 + 64]

Dueling DQN
  ├─ 价值流 V(s) → [B, 1]
  └─ 优势流 A(s,a) → [B, num_actions]
       └─ Q(s,a) = V(s) + (A(s,a) - mean(A))

输出层
  ├─ Q值 [B, num_actions]
  └─ 动作掩码 [B, num_actions]
```

---

## 📊 输入数据格式

### 1️⃣ **布局张量** (Layout Tensor)

**形状**: `[batch_size, H, W, N]`

**含义**: 
- `H`: 工厂高度（网格行数）
- `W`: 工厂宽度（网格列数）
- `N`: 待放置物体总数
- 每个位置 `(x, y)` 的最后一维 `N` 是**One-Hot编码向量**
- `layout[b, x, y, :]` 是一个N维向量，只有一个元素为1，其余为0
- 如果位置为空，则该向量全为0

**示例**:
```python
# 10×10工厂，5个物体
H, W, N = 10, 10, 5

# 创建空布局（所有位置都是全0向量）
layout = torch.zeros(1, H, W, N)

# 放置物体0在位置(0, 0)
layout[0, 0, 0, :] = torch.tensor([1, 0, 0, 0, 0])  # One-Hot: 物体0
# 或简写：
layout[0, 0, 0, 0] = 1  # 其余已经是0

# 放置物体1在位置(2, 3)
layout[0, 2, 3, :] = torch.tensor([0, 1, 0, 0, 0])  # One-Hot: 物体1

# 放置物体2在位置(5, 5)
layout[0, 5, 5, :] = torch.tensor([0, 0, 1, 0, 0])  # One-Hot: 物体2

# 检查位置(2,3)的One-Hot编码
print(layout[0, 2, 3, :])  # tensor([0, 1, 0, 0, 0])

# 位置(1,1)为空
print(layout[0, 1, 1, :])  # tensor([0, 0, 0, 0, 0])
```

**重要**：
- ✅ 每个位置的One-Hot向量中，**最多只有一个1**（表示该位置放置的物体）
- ✅ 如果全为0，表示该位置**为空**
- ❌ 不能有多个1（一个位置不能同时放置多个物体）

**可视化理解**:

想象一个2×2的小工厂，3个物体：
```
工厂网格:
  (0,0)  (0,1)
  (1,0)  (1,1)

放置情况:
  物体0  空位
  物体1  物体2

布局张量 layout[0, :, :, :]:
  位置(0,0): [1, 0, 0]  ← 物体0的One-Hot
  位置(0,1): [0, 0, 0]  ← 空位（全0）
  位置(1,0): [0, 1, 0]  ← 物体1的One-Hot
  位置(1,1): [0, 0, 1]  ← 物体2的One-Hot
```

**使用辅助函数**:
```python
from agent.dqn_model import create_layout_tensor

# 定义已放置物体的位置
placed_positions = [
    (0, 0, 0),  # (物体ID, x坐标, y坐标)
    (1, 2, 3),
    (2, 5, 5)
]

# 自动创建布局张量（已处理One-Hot编码）
layout = create_layout_tensor(H, W, N, placed_positions)
# 添加batch维度
layout = layout.unsqueeze(0)  # [1, H, W, N]

# 验证One-Hot编码
print(layout[0, 0, 0, :])  # tensor([1., 0., 0., 0., 0.])
print(layout[0, 2, 3, :])  # tensor([0., 1., 0., 0., 0.])
print(layout[0, 1, 1, :])  # tensor([0., 0., 0., 0., 0.]) 空位
```

---

### 2️⃣ **物料流邻接矩阵** (Material Flow Adjacency Matrix)

**形状**: `[batch_size, N, N]`

**含义**:
- `N`: 物体总数
- 元素为 `0` 或 `1`
- `A[i, j] = 1` 表示有从物体 `i` 到物体 `j` 的物料流
- `A[i, j] = 0` 表示没有物料流

**示例**:
```python
N = 5

# 创建邻接矩阵
material_flow = torch.zeros(1, N, N)

# 物体0 → 物体1
material_flow[0, 0, 1] = 1

# 物体1 → 物体2
material_flow[0, 1, 2] = 1

# 物体2 → 物体3
material_flow[0, 2, 3] = 1

# 物体3 → 物体4
material_flow[0, 3, 4] = 1

# 物体4 → 物体0（形成环）
material_flow[0, 4, 0] = 1
```

**使用辅助函数**:
```python
from agent.dqn_model import create_material_flow_matrix

# 定义物料流
flows = [
    (0, 1),  # (源物体ID, 目标物体ID)
    (1, 2),
    (2, 3),
    (3, 4),
    (4, 0)
]

# 自动创建邻接矩阵
material_flow = create_material_flow_matrix(N, flows)
# 添加batch维度
material_flow = material_flow.unsqueeze(0)  # [1, N, N]
```

---

### 3️⃣ **待放置模块** (Current Object to Place)

**形状**: `[batch_size, N]`

**含义**:
- `N`: 物体总数
- 这是一个**One-Hot编码向量**
- `current_object[i] = 1` 表示当前要放置第 `i` 个物体
- 其余元素为 `0`

**示例**:
```python
N = 5

# 创建待放置模块向量
current_object = torch.zeros(1, N)

# 当前要放置物体3
current_object[0, :] = torch.tensor([0, 0, 0, 1., 0])

# 或简写：
current_object[0, 3] = 1  # 其余已经是0

# 验证
print(current_object)  # tensor([[0., 0., 0., 1., 0.]])
print(current_object.sum())  # tensor(1.) ✓ One-Hot编码的和为1
```

**在训练循环中的使用**:
```python
# 按顺序放置物体
for object_id in range(N):
    # 创建当前物体的One-Hot编码
    current_object = torch.zeros(1, N)
    current_object[0, object_id] = 1
    
    # 使用网络决策放置位置
    q_values, _ = model(layout, material_flow, current_object)
    action = q_values.argmax()
    
    # 执行动作，更新布局...
```

---

### 4️⃣ **动作掩码** (Action Mask)

**形状**: `[batch_size, num_actions]`

**含义**:
- `num_actions`: 所有可能的动作数量（通常 = H × W × 旋转角度数）
- 元素为 `0` 或 `1`
- `mask[i] = 1` 表示动作 `i` 有效
- `mask[i] = 0` 表示动作 `i` 无效（如位置被占用）

**示例**:
```python
num_actions = H * W * 4  # 10×10网格 × 4个旋转角度 = 400个动作

# 创建动作掩码（全部有效）
action_mask = torch.ones(1, num_actions)

# 将前50个动作设为无效
action_mask[0, :50] = 0

# 将某些特定位置设为无效
# 例如，位置(5, 5)的所有旋转都无效
for rotation in range(4):
    action_id = (5 * W + 5) * 4 + rotation
    action_mask[0, action_id] = 0
```

---

## 🚀 使用示例

### **示例1：创建网络**

```python
from agent.dqn_model import LayoutDQN

# 参数设置
H, W, N = 10, 10, 5  # 10×10工厂，5个物体
num_actions = H * W * 4  # 400个动作（位置×旋转）

# 创建网络
model = LayoutDQN(
    height=H,
    width=W,
    num_objects=N,
    num_actions=num_actions,
    cnn_channels=[32, 64, 64],     # CNN通道数
    gcn_hidden_dim=64,             # GCN隐藏层维度
    gcn_output_dim=128,            # GCN输出维度
    fusion_hidden_dim=256,         # 融合层维度
    use_dueling=True               # 使用Dueling架构
)

print(f"网络参数量: {sum(p.numel() for p in model.parameters()):,}")
```

---

### **示例2：前向传播（计算Q值）**

```python
import torch
from agent.dqn_model import LayoutDQN, create_layout_tensor, create_material_flow_matrix

# 创建网络
model = LayoutDQN(height=10, width=10, num_objects=5, num_actions=400)

# 准备输入数据
# 1. 布局 (s_t1)
placed = [(0, 0, 0), (1, 2, 3), (2, 5, 5)]
layout = create_layout_tensor(10, 10, 5, placed).unsqueeze(0)

# 2. 物料流 (s_t2)
flows = [(0, 1), (1, 2), (2, 0)]
material_flow = create_material_flow_matrix(5, flows).unsqueeze(0)

# 3. 待放置模块 (s_t3) - 假设当前要放置物体3
current_object = torch.zeros(1, 5)
current_object[0, 3] = 1  # 物体3的One-Hot编码

# 4. 动作掩码
action_mask = torch.ones(1, 400)
action_mask[0, :50] = 0  # 前50个动作无效

# 前向传播（三个状态输入）
q_values, predicted_mask = model(layout, material_flow, current_object, action_mask)

print(f"Q值形状: {q_values.shape}")  # [1, 400]
print(f"Q值范围: [{q_values.min():.3f}, {q_values.max():.3f}]")
print(f"预测掩码形状: {predicted_mask.shape}")  # [1, 400]
```

---

### **示例3：选择动作（epsilon-greedy）**

```python
# 贪心策略（epsilon=0）
action = model.get_action(
    layout=layout,
    material_flow=material_flow,
    current_object=current_object,  # ← 新增参数
    action_mask=action_mask,
    epsilon=0.0  # 完全贪心
)
print(f"贪心动作: {action.item()}")

# 随机策略（epsilon=1）
action_random = model.get_action(
    layout=layout,
    material_flow=material_flow,
    current_object=current_object,  # ← 新增参数
    action_mask=action_mask,
    epsilon=1.0  # 完全随机
)
print(f"随机动作: {action_random.item()}")

# epsilon-greedy策略
action_mixed = model.get_action(
    layout=layout,
    material_flow=material_flow,
    current_object=current_object,  # ← 新增参数
    action_mask=action_mask,
    epsilon=0.1  # 10%探索，90%利用
)
print(f"混合策略动作: {action_mixed.item()}")
```

---

### **示例4：训练循环（伪代码）**

```python
import torch.optim as optim

# 创建网络和优化器
policy_net = LayoutDQN(10, 10, 5, 400)
target_net = LayoutDQN(10, 10, 5, 400)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)

# 训练参数
gamma = 0.99
epsilon = 1.0
epsilon_decay = 0.995
epsilon_min = 0.01

for episode in range(1000):
    # 重置环境
    state = env.reset()  # 返回 (layout, material_flow, current_object)
    total_reward = 0
    
    for step in range(max_steps):
        # 获取当前状态（三个状态分量）
        layout, material_flow, current_object = state
        action_mask = env.get_valid_actions()
        
        # 选择动作
        action = policy_net.get_action(
            layout, material_flow, current_object, action_mask, epsilon
        )
        
        # 执行动作
        next_state, reward, done, info = env.step(action.item())
        
        # 存储经验到回放缓冲区
        replay_buffer.push(state, action, reward, next_state, done)
        
        # 学习
        if len(replay_buffer) >= batch_size:
            # 采样batch
            batch = replay_buffer.sample(batch_size)
            
            # 计算当前Q值
            q_values, _ = policy_net(
                batch.layout, 
                batch.material_flow,
                batch.current_object  # ← 新增
            )
            current_q = q_values.gather(1, batch.action)
            
            # 计算目标Q值（Double DQN）
            with torch.no_grad():
                next_q_values, _ = target_net(
                    batch.next_layout, 
                    batch.next_material_flow,
                    batch.next_current_object  # ← 新增
                )
                max_next_q = next_q_values.max(1)[0]
                target_q = batch.reward + gamma * max_next_q * (1 - batch.done)
            
            # 计算损失
            loss = F.mse_loss(current_q, target_q)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # 更新状态
        state = next_state
        total_reward += reward
        
        if done:
            break
    
    # 衰减epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
    # 定期更新目标网络
    if episode % 10 == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    print(f"Episode {episode}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.3f}")
```

---

## 🔧 参数说明

### **网络参数**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `height` | int | - | 工厂高度（网格行数） |
| `width` | int | - | 工厂宽度（网格列数） |
| `num_objects` | int | - | 待放置物体总数 |
| `num_actions` | int | - | 动作空间大小 |
| `cnn_channels` | list | [32,64,64] | CNN各层通道数 |
| `gcn_hidden_dim` | int | 64 | GCN隐藏层维度 |
| `gcn_output_dim` | int | 128 | GCN输出维度 |
| `fusion_hidden_dim` | int | 256 | 融合层隐藏维度 |
| `use_dueling` | bool | True | 是否使用Dueling架构 |

### **关键尺寸计算**

```python
# CNN输出维度
cnn_output_dim = cnn_channels[-1] * height * width
# 例如: 64 * 10 * 10 = 6400

# 融合层输入维度
fusion_input_dim = cnn_output_dim + gcn_output_dim
# 例如: 6400 + 128 = 6528

# 动作空间大小（示例）
num_actions = height * width * num_rotations
# 例如: 10 * 10 * 4 = 400
```

---

## 💡 设计细节

### **1. CNN处理布局**

**目的**: 提取布局的空间特征（如物体分布、空间关系）

**流程**:
```
输入: [B, H, W, N]
  ↓ permute → [B, N, H, W]
  ↓ Conv2d(N→32) + ReLU + BN
  ↓ Conv2d(32→64) + ReLU + BN  
  ↓ Conv2d(64→64) + ReLU + BN
  ↓ reshape → [B, 64*H*W]
输出: [B, 6400] (假设H=W=10)
```

**为什么用3层卷积？**
- 层1: 提取局部模式（相邻物体）
- 层2: 提取中等范围模式（物体组）
- 层3: 提取全局模式（整体布局）

---

### **2. GCN处理物料流**

**目的**: 学习物体之间的物料流关系

**流程**:
```
输入: [B, N, N] 邻接矩阵
  ↓ 节点嵌入 → [B, N, 64]
  ↓ GCN层1: H' = σ(D^(-1/2) A D^(-1/2) H W)
  ↓ ReLU
  ↓ GCN层2
  ↓ 全局平均池化 → [B, 128]
输出: [B, 128]
```

**GCN的优势**:
- ✅ 捕捉物体间的拓扑关系
- ✅ 处理不同数量的物料流边
- ✅ 学习远距离依赖（通过多层传播）

---

### **3. MLP处理待放置模块**

**目的**: 将待放置模块信息编码为特征向量

**流程**:
```
输入: [B, N] One-Hot编码
  ↓ 线性层1: N → 64
  ↓ ReLU
  ↓ 线性层2: 64 → 64
  ↓ ReLU
输出: [B, 64]
```

**为什么需要这个分支？**
- ✅ 告诉网络**当前要放置哪个模块**
- ✅ 让网络根据**模块特性**做出针对性决策
- ✅ 完整实现状态空间 `s_t = {s_t1, s_t2, s_t3}`

**与布局的区别**:
- **布局 (s_t1)**: 已放置的模块（过去）
- **待放置模块 (s_t3)**: 当前要放置的模块（现在）

---

### **4. Dueling DQN**

**目的**: 分离状态价值和动作优势

**公式**:
```
Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
```

**好处**:
- ✅ 更稳定的学习（V和A独立优化）
- ✅ 更快的收敛（V不依赖于具体动作）
- ✅ 更好的泛化（共享特征提取）

---

## ⚠️ 注意事项

### **1. 内存消耗**

```python
# 网络参数量（示例：10×10工厂，5个物体，400个动作）
# 约 2,775,393 参数 ≈ 10.6 MB (float32)

# 单个样本内存（不含batch）
layout_memory = H * W * N * 4 bytes      # 10*10*5*4 = 2KB
material_flow_memory = N * N * 4 bytes   # 5*5*4 = 100B
current_object_memory = N * 4 bytes      # 5*4 = 20B
# 总计 ≈ 2.1KB/样本

# 建议batch_size: 32-64（GPU显存允许下）
```

### **2. 数值稳定性**

```python
# GCN中的度矩阵逆平方根可能出现除以0
# 已在代码中处理：
degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0.0
```

### **3. 动作掩码**

```python
# 务必提供正确的动作掩码！
# 无效动作的Q值会被设为 -1e9

# 如果所有动作都无效，会选择动作0
# 建议在环境中提前检查
```

---

## 📚 与RL布局规划项目的对接

### **状态表示差异**

| 项目 | 布局 (s_t1) | 物料流 (s_t2) | 待放置模块 (s_t3) |
|------|-----------|-------------|----------------|
| **RL布局规划** | 二维网格 [H, W] | 物料流矩阵 [N, N]（浮点数） | 模块ID（整数） |
| **FactoGenie** | 三维张量 [H, W, N] | 邻接矩阵 [N, N]（0或1） | One-Hot向量 [N] |

### **转换示例**

```python
# RL布局规划 → FactoGenie
def convert_state(rl_state):
    """
    将RL布局规划的状态转换为FactoGenie格式
    
    Args:
        rl_state: dict with keys 'layout_grid', 'material_flow', 'current_unit_id', ...
    
    Returns:
        layout: [H, W, N]
        material_flow: [N, N]
        current_object: [N]
    """
    layout_grid = rl_state['layout_grid']  # [H, W]
    H, W = layout_grid.shape
    N = rl_state['num_units']
    
    # 1. 创建三维布局 (s_t1)
    layout = torch.zeros(H, W, N)
    for i in range(H):
        for j in range(W):
            unit_id = int(layout_grid[i, j])
            if unit_id > 0:  # 0表示空位
                layout[i, j, unit_id - 1] = 1
    
    # 2. 物料流二值化 (s_t2)
    material_flow_raw = rl_state['material_flow']  # [N, N]浮点数
    threshold = material_flow_raw.mean()
    material_flow = (material_flow_raw > threshold).float()
    
    # 3. 待放置模块One-Hot编码 (s_t3)
    current_unit_id = rl_state['current_unit_id']  # 整数 0~N-1
    current_object = torch.zeros(N)
    current_object[current_unit_id] = 1
    
    return layout, material_flow, current_object
```

---

## 🎓 进阶使用

### **自定义网络结构**

```python
# 修改CNN层数和通道
model = LayoutDQN(
    height=20, width=20, num_objects=10, num_actions=800,
    cnn_channels=[64, 128, 128, 256],  # 4层CNN
    gcn_hidden_dim=128,
    gcn_output_dim=256,
    fusion_hidden_dim=512
)

# 关闭Dueling架构
model_standard = LayoutDQN(
    ...,
    use_dueling=False
)
```

### **模型保存与加载**

```python
# 保存模型
torch.save(model.state_dict(), 'best_model.pth')

# 加载模型
model = LayoutDQN(height=10, width=10, num_objects=5, num_actions=400)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
```

---

## ❓ 常见问题

**Q: 为什么布局要用(H, W, N)而不是(H, W)?**

A: **One-Hot编码**使得网络能够区分不同物体，学习物体特定的特征。

- `(H, W)` 格式：只能表示"该位置是否有物体"（1个数字）
  ```python
  layout[x, y] = 1  # 有物体
  layout[x, y] = 0  # 无物体
  # 问题：不知道是哪个物体！
  ```

- `(H, W, N)` 格式：能表示"该位置放置的是哪个物体"（N维One-Hot向量）
  ```python
  layout[x, y, :] = [0, 1, 0, 0, 0]  # 物体1
  layout[x, y, :] = [0, 0, 1, 0, 0]  # 物体2
  layout[x, y, :] = [0, 0, 0, 0, 0]  # 空位
  # 网络能学习：物体1倾向于放在哪，物体2倾向于放在哪...
  ```

这样网络能学习**物体特定的布局模式**（如：仓库应该靠近入口，加工区应该靠近原料区等）。

**Q: 物料流为什么用邻接矩阵而不是权重矩阵?**

A: 简化模型，专注于"是否有物料流"的拓扑关系。权重信息可以通过GCN的多层传播隐式学习。

**Q: 动作掩码预测有什么用？**

A: 
1. 辅助训练：可以加入损失函数，提高网络对无效动作的识别能力
2. 调试：检查网络是否学会了物理约束
3. 实际使用中，仍建议由环境直接提供掩码

**Q: 如何调整网络大小？**

A: 根据问题规模：
- 小规模 (5×5, 3个物体): `cnn_channels=[16,32]`, `gcn_hidden_dim=32`
- 中规模 (10×10, 5个物体): 默认参数
- 大规模 (20×20, 10个物体): `cnn_channels=[64,128,256]`, `gcn_hidden_dim=128`

---

**祝你训练顺利！🚀**

