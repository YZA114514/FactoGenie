import csv
from pathlib import Path
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent.dqn_model import DQN, DuelingDQN
from agent.agent import Agent, calc_loss_prio
from agent.replay_buffer import ExperienceBuffer, PrioReplayBuffer
from environment.gym_wrapper import FactoryEnv


def train(params):
    if params.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{params.gpu_id}")
        print(f"Using GPU: {torch.cuda.get_device_name(params.gpu_id)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    env = FactoryEnv()
    model_cls = DuelingDQN if params.use_dueling else DQN

    net = model_cls(
        env.observation_space.shape, 
        env.action_space.n,
        use_noisy=params.use_noisy
    ).to(device)
    tgt_net = model_cls(
        env.observation_space.shape, 
        env.action_space.n,
        use_noisy=params.use_noisy
    ).to(device)
    tgt_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=params.lr)

    buffer_cls = PrioReplayBuffer if params.use_prior else ExperienceBuffer
    buffer = buffer_cls(buf_size=params.replay_size)
    agent = Agent(env, buffer)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    reward_log_path = log_dir / "rewards.csv"
    loss_log_path = log_dir / "losses.csv"
    reward_plot_path = log_dir / "rewards.png"
    loss_plot_path = log_dir / "losses.png"

    reward_file = reward_log_path.open("w", newline="")
    loss_file = loss_log_path.open("w", newline="")
    reward_writer = csv.writer(reward_file)
    loss_writer = csv.writer(loss_file)
    reward_writer.writerow(["frame_idx", "reward"])
    loss_writer.writerow(["frame_idx", "loss"])
    reward_history = []
    loss_history = []
    reward_fig, reward_ax = plt.subplots()
    loss_fig, loss_ax = plt.subplots()

    def update_plot(history, ax, fig, title, xlabel, ylabel, output_path):
        if not history:
            return
        xs, ys = zip(*history)
        ax.clear()
        ax.plot(xs, ys, color="tab:blue")
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", linewidth=0.5)
        fig.tight_layout()
        fig.savefig(output_path)

    def log_reward(step, value):
        reward_history.append((step, value))
        reward_writer.writerow([step, value])
        reward_file.flush()
        update_plot(
            reward_history,
            reward_ax,
            reward_fig,
            "Episode Reward",
            "Frame",
            "Reward",
            reward_plot_path,
        )

    def log_loss(step, value):
        loss_history.append((step, value))
        loss_writer.writerow([step, value])
        loss_file.flush()
        update_plot(
            loss_history,
            loss_ax,
            loss_fig,
            "Training Loss",
            "Frame",
            "Loss",
            loss_plot_path,
        )

    epsilon = params.epsilon_start
    total_rewards = []
    frame_idx = 0
    best_m_reward = -float("inf")

    try:
        with tqdm(total=params.epsilon_decay_last_frame, desc="Training") as pbar:
            while frame_idx < params.epsilon_decay_last_frame:
                frame_idx += 1
                pbar.update(1)

                epsilon = max(
                    params.epsilon_final,
                    params.epsilon_start - frame_idx / params.epsilon_decay_last_frame,
                )
                reward = agent.play_step(net, epsilon, device=device)
                if reward is not None:
                    total_rewards.append(reward)
                    log_reward(frame_idx, reward)
                    m_reward = np.mean(total_rewards[-100:])
                    pbar.set_postfix_str(
                        f"Mean reward: {m_reward:.2f}, Epsilon: {epsilon:.2f}"
                    )
                    if best_m_reward < m_reward:
                        print(f"Best mean reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                        best_m_reward = m_reward

                if len(buffer) < params.replay_start_size:
                    continue

                buffer.update_beta(frame_idx)
                
                # 如果使用Noisy Net，重置噪声
                if params.use_noisy:
                    net.reset_noise()
                    tgt_net.reset_noise()

                optimizer.zero_grad()
                samples, batch_indices, weights = buffer.sample(params.batch_size)
                states, actions, rewards, dones, next_states = zip(*samples)
                batch = (
                    np.array(states, copy=False),
                    np.array(actions, copy=False),
                    np.array(rewards, dtype=np.float32),
                    np.array(dones, dtype=bool),
                    np.array(next_states, copy=False),
                )
                weights = np.array(weights, dtype=np.float32)

                loss_v, sample_prios_v = calc_loss_prio(
                    batch,
                    weights,
                    net,
                    tgt_net,
                    device=device,
                    double_dqn=params.use_double,
                )

                loss_value = loss_v.item()
                loss_v.backward()
                optimizer.step()

                if params.use_prior:
                    buffer.update_priorities(batch_indices, sample_prios_v)

                log_loss(frame_idx, loss_value)

                if frame_idx % params.sync_target_frames == 0:
                    tgt_net.load_state_dict(net.state_dict())
    finally:
        reward_file.close()
        loss_file.close()
        plt.close(reward_fig)
    plt.close(loss_fig)

