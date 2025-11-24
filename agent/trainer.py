import csv
import hashlib
import re
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
# 导入布局可视化工具
import sys
sys.path.append(str(Path(__file__).parent.parent))
from visualize_layouts import LayoutVisualizer


def train(params):
    if params.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{params.gpu_id}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 创建环境（默认不使用仿真以加速训练）
    simulation_duration = getattr(params, 'simulation_duration', 20000)
    env = FactoryEnv(
        use_simulation=params.use_simulation,
        simulation_duration=simulation_duration
    )
    print(f"Environment: use_simulation={params.use_simulation}, simulation_duration={simulation_duration}")
    
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
    # 处理奖励分解参数
    reward_decompose = getattr(params, 'reward_decompose', 'none')
    if reward_decompose == 'none':
        reward_decompose = None
    reward_gamma = getattr(params, 'reward_gamma', 0.9)
    agent = Agent(env, buffer, reward_decompose=reward_decompose, reward_gamma=reward_gamma)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    def format_param_value(value):
        if isinstance(value, float):
            formatted = f"{value:.6g}"  # compact scientific/decimal format
            return formatted.replace("+", "").replace("-", "m").replace(".", "p")
        return str(value)

    def sanitize_component(text):
        return re.sub(r"[^0-9A-Za-z_\-]", "_", text)

    alias_pairs = [
        ("lr", "lr"),
        ("batch_size", "bs"),
        ("replay_size", "rs"),
        ("replay_start_size", "rss"),
        ("sync_target_frames", "sync"),
        ("epsilon_decay_last_frame", "edf"),
        ("epsilon_start", "es"),
        ("epsilon_final", "ef"),
        ("use_prior", "prio"),
        ("use_double", "dbl"),
        ("use_dueling", "duel"),
        ("use_noisy", "noisy"),
        ("use_simulation", "sim"),
        ("reward_decompose", "rdc"),
        ("reward_gamma", "rg"),
    ]

    run_components = []
    for param_name, alias in alias_pairs:
        if not hasattr(params, param_name):
            continue
        value = getattr(params, param_name)
        if isinstance(value, bool):
            value_token = "1" if value else "0"
        else:
            value_token = format_param_value(value)
        component = sanitize_component(f"{alias}{value_token}")
        run_components.append(component)

    if not run_components:
        run_components.append("run")

    run_name = "__".join(run_components)
    if len(run_name) > 180:
        digest = hashlib.md5(run_name.encode("utf-8")).hexdigest()[:8]
        run_name = f"{run_name[:180]}__{digest}"

    run_dir = log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Logging outputs to: {run_dir}")

    # 配置仿真指标日志
    metrics_log_path = run_dir / "metrics.csv"
    if hasattr(env, 'env') and hasattr(env.env, 'set_metrics_logger'):
        env.env.set_metrics_logger(str(metrics_log_path))
        print(f"Metrics will be logged to: {metrics_log_path}")

    reward_log_path = run_dir / "rewards.csv"
    loss_log_path = run_dir / "losses.csv"
    reward_plot_path = run_dir / "rewards.png"
    loss_plot_path = run_dir / "losses.png"

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
    
    # 初始化布局可视化工具
    layout_visualizer = LayoutVisualizer(output_dir=str(run_dir / "layouts"))
    layout_save_interval = 400
    episode_counter = 0
    print(f"布局将保存到: {run_dir / 'layouts'}，每 {layout_save_interval} 次 episode 保存一次")

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
        nonlocal episode_counter
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

        # 保存布局和可视化
        episode_counter += 1
        if episode_counter % layout_save_interval == 0:
            try:
                layout_path = getattr(env.env, "layout_path", None)
                if layout_path and Path(layout_path).exists():
                    saved_json = layout_visualizer.save_layout_json(layout_path, episode_counter)
                    layout_visualizer.visualize_layout(
                        saved_json,
                        episode_num=episode_counter,
                        save=True,
                        show=False,
                    )
                    print(f"  [Episode {episode_counter}] 布局已保存 (奖励: {value:.2f})")
            except Exception as e:
                print(f"  [Episode {episode_counter}] 保存布局时出错: {e}")

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
                    m_reward = np.mean(total_rewards[-200:])
                    log_reward(frame_idx, m_reward)
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

