import torch.nn as nn


class DQN(nn.Module):
    """
    深度Q网络
    输入状态，输出所有动作的Q值
    """
    def __init__(self,):
        super(DQN, self).__init__()
        
    def forward(self, x):
        return x

class DuelingDQN(nn.Module):
    """
    对偶深度Q网络
    输入状态，输出所有动作的Q值
    包含价值和优势两个分支
    """
    def __init__(self,):
        super(DuelingDQN, self).__init__()
        
    def forward(self, x):
        return x