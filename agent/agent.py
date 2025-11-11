import numpy as np
import torch

from agent.replay_buffer import Experience


GAMMA = 0.99


class Agent:
    def __init__(self, env, buffer):
        self.env = env
        self.buffer = buffer
        self._reset_episode()

    def _reset_episode(self):
        self.state = self.env.reset()
        self.total_reward = 0.0
        self._episode_transitions = []

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
        
        # 贪心选择
        state_v = torch.tensor(
            np.array([self.state], copy=False), dtype=torch.float32, device=device
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
        奖励分解：将episode的最终奖励分配到各个步骤
        
        策略：
        - 中间步骤：给予小的时间惩罚（鼓励高效放置）
        - 最后一步：给予最终奖励（来自仿真或启发式计算）
        """
        num_steps = len(self._episode_transitions)
        rewards = []
        
        # 中间步骤的时间惩罚（可选，鼓励快速完成）
        step_penalty = -0.01
        
        for i in range(num_steps):
            if i < num_steps - 1:
                # 中间步骤：小惩罚或0
                rewards.append(step_penalty)
            else:
                # 最后一步：使用最终奖励
                # reward_payload 应该是最后一步的实际奖励
                rewards.append(reward_payload)
        
        if len(rewards) != num_steps:
            raise RuntimeError(
                f"Reward length {len(rewards)} does not match transition count {num_steps}."
            )
        
        return rewards

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

    states_v = torch.tensor(np.array(states, copy=False), dtype=torch.float32, device=device)
    next_states_v = torch.tensor(
        np.array(next_states, copy=False), dtype=torch.float32, device=device
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