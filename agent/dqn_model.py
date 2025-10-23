import torch.nn as nn


class DQN(nn.Module):
    """
    深度Q网络
    输入状态，输出所有动作的Q值
    """
    def __init__(self,):
        super(DQN, self).__init__()
        
    def forward(self, x):
        