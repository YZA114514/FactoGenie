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
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        state_v = torch.tensor(
            np.array([self.state], copy=False), dtype=torch.float32, device=device
        )
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        return int(act_v.item())

    def _resolve_episode_rewards(self, reward_payload, info):
        rewards = None
        
        # ==========分解奖励逻辑===========
        #
        # ================================

        if rewards is None:
            raise RuntimeError(
                "Failed to decompose rewards."
            )

        if len(rewards) != len(self._episode_transitions):
            raise RuntimeError(
                f"Reward length {len(rewards)} does not match transition count {len(self._episode_transitions)}."
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