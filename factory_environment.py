import gym


class FactoryEnv(gym.Env):
    """
    工厂环境
    """
    def __init__(self):
        super(FactoryEnv, self).__init__()

    def step(self, action):
        """
        执行一步动作，返回下一个状态，奖励，是否结束，额外信息
        """

    def reset(self):
        """
        重置环境，返回初始状态
        """