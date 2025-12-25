import numpy as np
import torch

from agent.replay_buffer import Experience


GAMMA = 0.99



class Agent:
    def __init__(self, env, buffer, reward_decompose=None, reward_gamma=0.99):
        """
        reward_decompose: None（默认，原逻辑），'mean'（平均分配），'discount'（折扣分配）
        reward_gamma: 折扣分配时的gamma
        """
        self.env = env
        self.buffer = buffer
        self.reward_decompose = reward_decompose
        self.reward_gamma = reward_gamma
        self._reset_episode()

    def _reset_episode(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        self._episode_transitions = []

    def _capture_layout_snapshot(self):
        """在 episode 结束时捕获布局快照（reset 之前调用）"""
        try:
            # 获取底层环境
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, 'layout_template') and hasattr(env, 'placed_units'):
                from environment.layout_exporter import LayoutExporter
                exporter = LayoutExporter(env.layout_template, None)
                return exporter.export_layout_dict(
                    env.placed_units,
                    env.functional_units
                )
        except Exception as e:
            print(f"捕获布局快照失败: {e}")
        return None

    @torch.no_grad()
    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None

        action = self._select_action(net, epsilon, device)
        new_state, reward_payload, is_done, info = self.env.step(action)
        self._episode_transitions.append((self.state, action, new_state))
        self.state = new_state

        if is_done:
            rewards = self._resolve_episode_rewards(reward_payload, info)
            done_reward = self._commit_episode(rewards)
            # 在 reset 之前保存布局快照
            self.last_layout_snapshot = self._capture_layout_snapshot()
            self._reset_episode()

        return done_reward

    def _select_action(self, net, epsilon, device):
        # 获取有效动作（如果环境支持）
        valid_actions = None
        if hasattr(self.env, 'get_valid_actions'):
            try:
                valid_actions = self.env.get_valid_actions()
            except Exception as e:
                print(f"⚠️ 警告: 获取有效动作失败: {e}")
                valid_actions = None
            
            if valid_actions is not None and len(valid_actions) == 0:
                # 如果没有有效动作，返回默认动作0
                # 环境的step方法会检测到这种情况并提前结束episode
                print("⚠️ 当前状态没有有效动作，环境将提前结束episode")
                return 0
        
        if np.random.random() < epsilon:
            # 随机探索
            if valid_actions is not None:
                return np.random.choice(valid_actions)
            else:
                return self.env.action_space.sample()
        
        # 如果是 NoisyNet，确保在每次选择动作前重新采样噪声
        if hasattr(net, 'reset_noise') and getattr(net, 'use_noisy', False):
            try:
                net.reset_noise()
            except Exception:
                pass

        # 贪心选择
        state_v = torch.tensor(
            np.asarray([self.state]), dtype=torch.float32, device=device
        )
        q_vals_v = net(state_v)
        
        if valid_actions is not None:
            # 只在有效动作中选择
            valid_q_vals = q_vals_v[0, valid_actions]
            best_valid_idx = valid_q_vals.argmax().item()
            return valid_actions[best_valid_idx]
        else:
            # 标准贪心选择
            _, act_v = torch.max(q_vals_v, dim=1)
            return int(act_v.item())

    def _resolve_episode_rewards(self, reward_payload, info):
        """
        奖励分解：支持三种方式
        - None: 原逻辑（最后一步奖励，其余小惩罚）
        - 'mean': 平均分配
        - 'discount': 折扣分配
        """
        num_steps = len(self._episode_transitions)
        if self.reward_decompose is None:
            # 原逻辑
            rewards = []
            step_penalty = -0.01
            for i in range(num_steps):
                if i < num_steps - 1:
                    rewards.append(step_penalty)
                else:
                    rewards.append(reward_payload)
            if len(rewards) != num_steps:
                raise RuntimeError(
                    f"Reward length {len(rewards)} does not match transition count {num_steps}."
                )
            return rewards
        elif self.reward_decompose == 'mean':
            # 平均分配
            per_step_reward = reward_payload / num_steps
            rewards = [per_step_reward] * num_steps
            return rewards
        elif self.reward_decompose == 'discount':
            # 归一化折扣分配，所有步奖励之和等于最终奖励
            gamma = self.reward_gamma
            T = num_steps
            # 先算分母S
            S = sum([gamma ** k for k in range(T)])
            rewards = []
            for i in range(T):
                # t=0,...,T-1; r_t = gamma**(T-1-i)/S * reward_payload
                weight = gamma ** (T - 1 - i) / S
                rewards.append(weight * reward_payload)
            return rewards
        else:
            raise ValueError(f"Unknown reward_decompose mode: {self.reward_decompose}")

    def _commit_episode(self, rewards):
        total_reward = 0.0
        for idx, ((state, action, next_state), reward) in enumerate(
            zip(self._episode_transitions, rewards)
        ):
            done_flag = idx == len(self._episode_transitions) - 1
            experience = Experience(state, action, float(reward), done_flag, next_state)
            self.buffer.append(experience)
            total_reward += float(reward)
        self.total_reward = total_reward
        return total_reward


def calc_loss_prio(batch, weights, net, tgt_net, device="cpu", double_dqn=True):
    states, actions, rewards, dones, next_states = batch

    states_v = torch.tensor(np.asarray(states), dtype=torch.float32, device=device)
    next_states_v = torch.tensor(
        np.asarray(next_states), dtype=torch.float32, device=device
    )
    actions_v = torch.tensor(actions, dtype=torch.int64, device=device)
    rewards_v = torch.tensor(rewards, dtype=torch.float32, device=device)
    done_mask = torch.tensor(dones, dtype=torch.bool, device=device)
    weights_v = torch.tensor(weights, dtype=torch.float32, device=device)

    state_action_values = net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)

    with torch.no_grad():
        if double_dqn:
            next_state_actions = net(next_states_v).max(1)[1]
            next_state_values = tgt_net(next_states_v).gather(
                1, next_state_actions.unsqueeze(-1)
            ).squeeze(-1)
        else:
            next_state_values = tgt_net(next_states_v).max(1)[0]
        next_state_values = next_state_values.masked_fill(done_mask, 0.0)
        expected_state_action_values = rewards_v + GAMMA * next_state_values

    td_errors = (state_action_values - expected_state_action_values).abs()

    loss = (state_action_values - expected_state_action_values) ** 2
    loss_v = weights_v * loss
    return loss_v.mean(), td_errors.detach().cpu().numpy()


def get_q_values_for_state(net, state, device="cpu"):
    """
    获取指定状态的所有动作 Q 值（用于热力图）
    
    Args:
        net: DQN 网络
        state: 状态数组
        device: 计算设备
        
    Returns:
        Q 值数组 [num_actions]
    """
    net.eval()
    with torch.no_grad():
        state_v = torch.tensor(
            np.asarray([state]), dtype=torch.float32, device=device
        )
        q_vals = net(state_v).squeeze().cpu().numpy()
    return q_vals


def get_q_values_heatmap(net, env, state, device="cpu", actual_action=None):
    """
    获取当前状态的 Q 值热力图数据
    
    Args:
        net: DQN 网络
        env: 环境实例
        state: 当前状态
        device: 计算设备
        actual_action: 实际执行的动作索引（可选，如果提供则显示实际动作而不是最佳动作）
        
    Returns:
        热力图数据字典
    """
    # 获取所有 Q 值
    q_values = get_q_values_for_state(net, state, device)
    
    # 获取有效动作
    valid_actions = []
    if hasattr(env, 'get_valid_actions'):
        valid_actions = env.get_valid_actions()
    
    # 获取网格尺寸
    grid_size = getattr(env, 'grid_size', (20, 20))
    if hasattr(env, 'env'):
        grid_size = getattr(env.env, 'grid_size', (20, 20))
    
    nx, ny = grid_size
    num_rotations = 4  # 0, 90, 180, 270
    
    # 重塑 Q 值为 [rotation, y, x] 的形式
    # 假设动作编码为: action = rotation * (nx * ny) + y * nx + x
    # 使用 None 代替 -np.inf，因为 JSON 不支持 Infinity
    q_values_3d = np.full((num_rotations, ny, nx), None, dtype=object)
    
    valid_actions_set = set(valid_actions)
    for action_idx, q_val in enumerate(q_values):
        rotation = action_idx // (nx * ny)
        remaining = action_idx % (nx * ny)
        y = remaining // nx
        x = remaining % nx
        
        if rotation < num_rotations and y < ny and x < nx:
            # 只有有效动作才填充Q值，无效动作保持为 None
            if action_idx in valid_actions_set:
                q_values_3d[rotation, y, x] = float(q_val)
    
    # 确定要显示的动作：如果提供了实际动作，使用它；否则使用Q值最大的动作
    if actual_action is not None and 0 <= actual_action < len(q_values):
        display_action_idx = actual_action
    else:
        # 在有效动作中选择Q值最大的
        if valid_actions:
            valid_q = [(a, q_values[a]) for a in valid_actions if 0 <= a < len(q_values)]
            if valid_q:
                display_action_idx = max(valid_q, key=lambda x: x[1])[0]
            else:
                display_action_idx = int(np.argmax(q_values))
        else:
            display_action_idx = int(np.argmax(q_values))
    
    display_rotation = display_action_idx // (nx * ny)
    remaining = display_action_idx % (nx * ny)
    display_y = remaining // nx
    display_x = remaining % nx
    
    return {
        'grid_width': nx,
        'grid_height': ny,
        'angle_options': [0, 90, 180, 270],
        'q_values': q_values_3d.tolist(),
        'q_values_flat': q_values.tolist(),
        'valid_actions': valid_actions,
        'selected_action': {
            'x': display_x,
            'y': display_y,
            'angle': display_rotation * 90,
            'q_value': float(q_values[display_action_idx]),
        },
        'q_min': float(np.min(q_values[q_values > -1e9])) if np.any(q_values > -1e9) else 0,
        'q_max': float(np.max(q_values)),
    }