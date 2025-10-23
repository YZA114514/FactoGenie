# DQN价值网络使用说明

## 📚 概述

`dqn_model.py` 实现了三种DQN价值网络：
1. **简单DQN** - 基础全连接网络
2. **Dueling DQN** - 双流架构（V值 + A值）
3. **布局DQN** - 工厂布局专用（CNN + MLP融合）

---

## 🚀 快速开始

### 方式1：简单DQN（推荐开始）

```python
from agent.dqn_model import DQN

# 创建网络
net = DQN(
    state_dim=100,      # 状态维度
    action_dim=10,      # 动作数量
    hidden_dims=[256, 256]  # 隐藏层
)

# 前向传播
import torch
state = torch.randn(4, 100)  # batch_size=4
q_values = net(state)  # 输出: [4, 10]
```

### 方式2：Dueling DQN（提升性能）

```python
from agent.dqn_model import DuelingDQN

net = DuelingDQN(
    state_dim=100,
    action_dim=10,
    hidden_dims=[256, 256]
)

q_values = net(state)  # [batch_size, action_dim]
```

### 方式3：布局专用DQN（工厂布局问题）

```python
from agent.dqn_model import LayoutDQN

net = LayoutDQN(
    grid_size=(20, 20),    # 网格尺寸
    num_units=5,           # 功能单元数量
    action_dim=1600,       # 20*20*4 (位置*旋转)
    hidden_dim=256,
    use_dueling=True       # 使用Dueling架构
)

# 输入字典格式状态
state = {
    'layout_grid': torch.randn(4, 20, 20),
    'material_flow': torch.randn(4, 5, 5),
    'current_unit': torch.randn(4, 5),
    'placed_mask': torch.randn(4, 5)
}

q_values = net(state)  # [4, 1600]
```

### 方式4：使用工厂函数（最灵活）

```python
from agent.dqn_model import create_dqn

# 简单DQN
net1 = create_dqn(
    state_dim=50, 
    action_dim=5, 
    network_type='simple'
)

# Dueling DQN
net2 = create_dqn(
    state_dim=50, 
    action_dim=5, 
    network_type='dueling'
)

# 布局DQN
net3 = create_dqn(
    grid_size=(15, 15),
    num_units=5,
    action_dim=100,
    network_type='layout',
    use_dueling=True
)
```

---

## 🔧 与agent.py集成

修改 `agent.py` 中的 `Agent` 类：

```python
# agent.py
from agent.dqn_model import DQN, DuelingDQN, create_dqn

class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        
        # 创建价值网络
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        
        # 选择网络类型
        self.net = DQN(state_dim, action_dim)
        # 或
        self.net = DuelingDQN(state_dim, action_dim)
        
        self._reset()
    
    # ... 其他方法保持不变
```

---

## 📊 网络对比

| 网络类型 | 适用场景 | 参数量 | 性能 | 复杂度 |
|---------|---------|--------|------|--------|
| **简单DQN** | 快速原型、简单问题 | ~67K | ⭐⭐⭐ | 低 |
| **Dueling DQN** | 一般RL问题 | ~134K | ⭐⭐⭐⭐ | 中 |
| **布局DQN** | 工厂布局规划 | ~26M | ⭐⭐⭐⭐⭐ | 高 |

---

## 🎓 核心概念

### Dueling DQN

将Q值分解：
```
Q(s,a) = V(s) + [A(s,a) - mean(A(s,a))]
```

- **V(s)**: 状态价值（在这个状态有多好）
- **A(s,a)**: 优势函数（选这个动作比平均好多少）

**优点**：
- 更稳定的训练
- 更好的泛化
- 学习速度更快

### 布局DQN架构

```
输入状态
  ├── 布局网格 → CNN → 特征1
  ├── 物料流矩阵 → MLP → 特征2
  └── 单元信息 → MLP → 特征3
       ↓
  融合特征
       ↓
  Dueling架构（可选）
       ↓
  Q值输出
```

---

## 🧪 测试网络

直接运行模块测试：

```bash
cd FactoGenie
python -m agent.dqn_model
```

应该看到：
```
测试DQN价值网络...

============================================================
测试1: 简单DQN
============================================================
输入形状: torch.Size([4, 100])
输出Q值形状: torch.Size([4, 10])
Q值范围: [-0.123, 0.456]
网络参数量: 67,850

... 更多测试输出 ...

============================================================
所有测试通过！✓
============================================================
```

---

## 📖 API文档

### DQN

```python
class DQN(nn.Module):
    def __init__(
        self, 
        state_dim: int,           # 状态维度
        action_dim: int,          # 动作数量
        hidden_dims: list = [256, 256]  # 隐藏层维度
    )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, state_dim]
        Returns:
            q_values: [batch_size, action_dim]
        """
```

### DuelingDQN

```python
class DuelingDQN(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [256, 256]
    )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """同DQN"""
```

### LayoutDQN

```python
class LayoutDQN(nn.Module):
    def __init__(
        self,
        grid_size: tuple,         # (height, width)
        num_units: int,           # 功能单元数量
        action_dim: int,          # 动作数量
        hidden_dim: int = 256,
        use_dueling: bool = True  # 是否使用Dueling
    )
    
    def forward(self, state: dict or torch.Tensor) -> torch.Tensor:
        """
        Args:
            state: 字典或张量
                字典包含：
                    'layout_grid': [B, H, W] 或 [B, 1, H, W]
                    'material_flow': [B, N, N]
                    'current_unit': [B, N]
                    'placed_mask': [B, N]
        Returns:
            q_values: [B, action_dim]
        """
```

---

## 💡 使用建议

### 第一阶段：简化开发

```python
# 使用简单DQN快速验证流程
net = DQN(state_dim=50, action_dim=10)
```

### 第二阶段：提升性能

```python
# 改用Dueling DQN
net = DuelingDQN(state_dim=50, action_dim=10)
```

### 第三阶段：完整实现

```python
# 使用布局专用网络
net = LayoutDQN(
    grid_size=(20, 20),
    num_units=5,
    action_dim=1600,
    use_dueling=True
)
```

---

## 🔍 调试技巧

### 检查网络输出

```python
# 1. 检查形状
print(f"Input: {state.shape}")
print(f"Output: {q_values.shape}")

# 2. 检查数值范围
print(f"Q值范围: [{q_values.min():.3f}, {q_values.max():.3f}]")

# 3. 检查梯度
print(f"是否需要梯度: {q_values.requires_grad}")
```

### 检查参数量

```python
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"总参数: {total_params:,}")
print(f"可训练: {trainable_params:,}")
```

---

## 📝 常见问题

**Q: 为什么有三种网络？**
A: 
- 简单DQN：快速开始
- Dueling DQN：更好性能
- 布局DQN：专门优化工厂布局问题

**Q: 我应该用哪个？**
A: 
- 刚开始：简单DQN
- 提升效果：Dueling DQN
- 最终版本：布局DQN

**Q: 网络太大怎么办？**
A: 
```python
# 减少隐藏层维度
net = DQN(state_dim=100, action_dim=10, hidden_dims=[128, 128])

# 或减少CNN通道数（修改源码）
```

---

## 🎯 下一步

1. ✅ 价值网络已完成
2. ⏭️ 在 `agent.py` 中集成网络
3. ⏭️ 实现训练循环
4. ⏭️ 测试训练

---

**网络已经准备好，开始训练吧！** 🚀

