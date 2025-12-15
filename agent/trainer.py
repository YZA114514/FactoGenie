import csv
import hashlib
import json
import re
import shutil
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


# 默认布局文件路径
DEFAULT_LAYOUT_PATH = Path("simulation/layouts/chair_layout.json")
EXPERIMENT_LAYOUTS_DIR = Path("simulation/layouts/experiments")


def _create_experiment_layout(params) -> str:
    """
    为当前实验创建专用的布局文件，避免并行实验时的冲突。
    
    Args:
        params: 实验参数
    
    Returns:
        实验专用布局文件的路径
    """
    # 生成实验唯一标识符（基于关键参数）
    key_params = [
        f"po_{getattr(params, 'placement_order', 'default')}",
        f"wd_{getattr(params, 'weight_distance', 0.20):.2f}",
        f"wl_{getattr(params, 'weight_logistics', 0.30):.2f}",
        f"wf_{getattr(params, 'weight_flow', 0.20):.2f}",
        f"wt_{getattr(params, 'weight_throughput', 0.25):.2f}",
        f"wu_{getattr(params, 'weight_utilization', 0.05):.2f}",
        f"sim_{getattr(params, 'use_simulation', True)}",
        f"dur_{getattr(params, 'simulation_duration', 2000)}",
    ]
    # 添加时间戳和进程ID确保唯一性
    import os
    import time
    key_params.append(f"pid_{os.getpid()}")
    key_params.append(f"ts_{int(time.time() * 1000) % 100000}")
    
    experiment_id = "__".join(key_params)
    # 对长名称进行哈希处理
    if len(experiment_id) > 100:
        digest = hashlib.md5(experiment_id.encode("utf-8")).hexdigest()[:12]
        experiment_id = f"{experiment_id[:80]}_{digest}"
    
    # 清理文件名中的非法字符
    experiment_id = re.sub(r"[^0-9A-Za-z_\-\.]", "_", experiment_id)
    
    # 创建实验专用布局目录
    EXPERIMENT_LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 实验专用布局文件路径
    experiment_layout_path = EXPERIMENT_LAYOUTS_DIR / f"{experiment_id}.json"
    
    # 复制默认布局文件作为初始值
    if DEFAULT_LAYOUT_PATH.exists():
        shutil.copy(DEFAULT_LAYOUT_PATH, experiment_layout_path)
        print(f"Created experiment layout: {experiment_layout_path}")
    else:
        raise FileNotFoundError(f"Default layout file not found: {DEFAULT_LAYOUT_PATH}")
    
    return str(experiment_layout_path)


def _cleanup_experiment_layout(layout_path: str) -> None:
    """
    清理实验结束后的布局文件（可选）。
    
    Args:
        layout_path: 实验专用布局文件路径
    """
    try:
        layout_file = Path(layout_path)
        if layout_file.exists() and EXPERIMENT_LAYOUTS_DIR in layout_file.parents:
            layout_file.unlink()
            print(f"Cleaned up experiment layout: {layout_path}")
    except Exception as e:
        print(f"Warning: Failed to cleanup layout file: {e}")


def train(params):
    if params.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{params.gpu_id}")
        print(f"Using GPU: {device}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # 创建环境（默认不使用仿真以加速训练）
    simulation_duration = getattr(params, 'simulation_duration', 20000)
    
    # 构建奖励权重字典
    objective_weights = {
        'transportation_intensity': getattr(params, 'weight_distance', 0.20),
        'material_flow_clarity': getattr(params, 'weight_logistics', 0.30),
        'space_utilization': getattr(params, 'weight_flow', 0.20),  # flow_clarity uses space_utilization key
        'throughput_time': getattr(params, 'weight_throughput', 0.25),
        'utilization': getattr(params, 'weight_utilization', 0.05),
    }
    
    # 获取摆放顺序
    placement_order = getattr(params, 'placement_order', 'default')
    
    # 创建实验专用的布局文件（用于并行实验隔离）
    experiment_layout_path = _create_experiment_layout(params)
    
    # 校准指标边界（如果需要）
    metric_bounds = None
    calibrate_episodes = getattr(params, 'calibrate_episodes', 0)  # 0表示不校准，使用默认边界
    throughput_target = getattr(params, 'throughput_target', None)  # 用户指定的吞吐量目标
    
    if calibrate_episodes > 0:
        print(f"\n{'='*50}")
        print(f"开始校准指标边界 ({calibrate_episodes} 个随机布局)...")
        print(f"{'='*50}")
        
        from calibration.bounds import BoundsManager
        bounds_manager = BoundsManager()
        
        # 获取工厂配置路径
        factory_config = "simulation/configs/chair_factory.json"
        layout_config = experiment_layout_path
        
        bounds = bounds_manager.load_or_calibrate(
            factory_config_path=factory_config,
            layout_config_path=layout_config,
            n_episodes=calibrate_episodes,
            simulation_duration=simulation_duration,
            throughput_target=throughput_target,
        )
        
        # 提取边界用于环境
        metric_bounds = bounds_manager.get_bounds_for_reward(
            factory_config_path=factory_config,
            layout_config_path=layout_config,
        )
        print(f"\n使用校准边界: {metric_bounds}")
    
    env = FactoryEnv(
        use_simulation=params.use_simulation,
        simulation_duration=simulation_duration,
        objective_weights=objective_weights,
        placement_order=placement_order,
        layout_path=experiment_layout_path,
        metric_bounds=metric_bounds,
    )
    print(f"Environment: use_simulation={params.use_simulation}, simulation_duration={simulation_duration}")
    print(f"Objective weights: {objective_weights}")
    print(f"Placement order: {placement_order}")
    print(f"Layout file: {experiment_layout_path}")
    
    model_cls = DuelingDQN if params.use_dueling else DQN

    net = model_cls(
        env.observation_space.shape, 
        env.action_space.n,
        use_noisy=params.use_noisy,
        sigma_init=getattr(params, 'sigma_init', 0.5),
    ).to(device)
    tgt_net = model_cls(
        env.observation_space.shape, 
        env.action_space.n,
        use_noisy=params.use_noisy,
        sigma_init=getattr(params, 'sigma_init', 0.5),
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

    log_dir = Path("true_logs")
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
        ("total_steps", "ts"),
        ("epsilon_start", "es"),
        ("epsilon_final", "ef"),
        ("use_prior", "prio"),
        ("use_double", "dbl"),
        ("use_dueling", "duel"),
        ("use_noisy", "noisy"),
        ("sigma_init", "sinit"),
        ("use_simulation", "sim"),
        ("reward_decompose", "rdc"),
        ("reward_gamma", "rg"),
        # 奖励权重参数
        ("weight_distance", "wd"),
        ("weight_logistics", "wl"),
        ("weight_flow", "wf"),
        ("weight_throughput", "wt"),
        ("weight_utilization", "wu"),
        # 摆放顺序参数
        ("placement_order", "po"),
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
    
    # 检查点保存配置
    checkpoint_interval = getattr(params, 'checkpoint_interval', 0)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_reward = -float("inf")
    
    if checkpoint_interval > 0:
        print(f"检查点将保存到: {checkpoint_dir}，每 {checkpoint_interval} 次 episode 保存一次")
    
    def save_checkpoint(episode: int, reward: float, is_best: bool = False):
        """保存检查点：模型权重 + 布局 + 指标"""
        prefix = "best" if is_best else f"ep{episode}"
        
        # 1. 保存模型权重
        model_path = checkpoint_dir / f"model_{prefix}.pth"
        torch.save({
            'episode': episode,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'reward': reward,
            'epsilon': epsilon,
        }, model_path)
        
        # 2. 保存布局快照
        layout_path = getattr(env.env, "layout_path", None)
        if layout_path and Path(layout_path).exists():
            layout_snapshot_path = checkpoint_dir / f"layout_{prefix}.json"
            shutil.copy(layout_path, layout_snapshot_path)
        
        # 3. 保存指标
        metrics_path = checkpoint_dir / f"metrics_{prefix}.json"
        metrics_data = {
            'episode': episode,
            'reward': reward,
            'mean_reward_200': np.mean(total_rewards[-200:]) if total_rewards else 0,
            'epsilon': epsilon,
            'frame_idx': frame_idx,
            'is_best': is_best,
        }
        # 如果有仿真指标，也保存
        if hasattr(env.env, 'last_metrics') and env.env.last_metrics:
            metrics_data['simulation_metrics'] = env.env.last_metrics
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        tag = "🏆 BEST" if is_best else ""
        print(f"  [Checkpoint] {prefix} saved (reward: {reward:.3f}) {tag}")

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
        
        # 检查点保存
        nonlocal best_reward
        if checkpoint_interval > 0 and episode_counter % checkpoint_interval == 0:
            save_checkpoint(episode_counter, value, is_best=False)
        
        # 保存最佳检查点
        if value > best_reward:
            best_reward = value
            if checkpoint_interval > 0:
                save_checkpoint(episode_counter, value, is_best=True)

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
        with tqdm(total=params.total_steps, desc="Training") as pbar:
            while frame_idx < params.total_steps:
                frame_idx += 1
                pbar.update(1)

                # If NoisyNet is used, epsilon should be disabled (0.0)
                if params.use_noisy:
                    epsilon = 0.0
                else:
                    epsilon = max(
                        params.epsilon_final,
                        params.epsilon_start - frame_idx / params.epsilon_decay_last_frame,
                    )
                # If using NoisyNet, resample noise before selecting actions
                if params.use_noisy:
                    net.reset_noise()
                    tgt_net.reset_noise()
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
                # Ensure noise is resampled again before training/backprop
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


def train_with_callbacks(
    params,
    progress_callback=None,
    result_callback=None,
    stop_event=None,
    db_session=None,
    project_id=None,
):
    """
    支持回调的训练函数，供后端服务调用
    
    Args:
        params: 训练参数
        progress_callback: 进度回调函数 (step, episode, reward, best_reward, metrics) -> None
        result_callback: 布局结果回调函数 (layout_data) -> None
        stop_event: threading.Event 停止信号
        db_session: 数据库会话（用于保存检查点）
        project_id: 项目ID
        
    Returns:
        训练结果字典
    """
    if params.device == "cuda" and torch.cuda.is_available():
        device = torch.device(f"cuda:{params.gpu_id}")
    else:
        device = torch.device("cpu")
    
    simulation_duration = getattr(params, 'simulation_duration', 20000)
    
    # 构建奖励权重字典
    objective_weights = {
        'transportation_intensity': getattr(params, 'weight_distance', 0.20),
        'material_flow_clarity': getattr(params, 'weight_logistics', 0.30),
        'space_utilization': getattr(params, 'weight_flow', 0.20),
        'throughput_time': getattr(params, 'weight_throughput', 0.25),
        'utilization': getattr(params, 'weight_utilization', 0.05),
    }
    
    placement_order = getattr(params, 'placement_order', 'default')
    
    # 创建实验专用的布局文件
    experiment_layout_path = _create_experiment_layout(params)
    
    # 校准指标边界
    metric_bounds = None
    calibrate_episodes = getattr(params, 'calibrate_episodes', 0)
    throughput_target = getattr(params, 'throughput_target', None)
    
    if calibrate_episodes > 0:
        from calibration.bounds import BoundsManager
        bounds_manager = BoundsManager()
        factory_config = getattr(params, 'factory_config', "simulation/configs/chair_factory.json")
        
        bounds_manager.load_or_calibrate(
            factory_config_path=factory_config,
            layout_config_path=experiment_layout_path,
            n_episodes=calibrate_episodes,
            simulation_duration=simulation_duration,
            throughput_target=throughput_target,
        )
        
        metric_bounds = bounds_manager.get_bounds_for_reward(
            factory_config_path=factory_config,
            layout_config_path=experiment_layout_path,
        )
    
    env = FactoryEnv(
        use_simulation=params.use_simulation,
        simulation_duration=simulation_duration,
        objective_weights=objective_weights,
        placement_order=placement_order,
        layout_path=experiment_layout_path,
        metric_bounds=metric_bounds,
    )
    
    model_cls = DuelingDQN if params.use_dueling else DQN
    
    net = model_cls(
        env.observation_space.shape,
        env.action_space.n,
        use_noisy=params.use_noisy,
        sigma_init=getattr(params, 'sigma_init', 0.5),
    ).to(device)
    tgt_net = model_cls(
        env.observation_space.shape,
        env.action_space.n,
        use_noisy=params.use_noisy,
        sigma_init=getattr(params, 'sigma_init', 0.5),
    ).to(device)
    tgt_net.load_state_dict(net.state_dict())
    optimizer = optim.Adam(net.parameters(), lr=params.lr)
    
    buffer_cls = PrioReplayBuffer if params.use_prior else ExperienceBuffer
    buffer = buffer_cls(buf_size=params.replay_size)
    
    reward_decompose = getattr(params, 'reward_decompose', 'none')
    if reward_decompose == 'none':
        reward_decompose = None
    reward_gamma = getattr(params, 'reward_gamma', 0.9)
    agent = Agent(env, buffer, reward_decompose=reward_decompose, reward_gamma=reward_gamma)
    
    epsilon = params.epsilon_start
    total_rewards = []
    frame_idx = 0
    best_reward = -float("inf")
    best_layout = None
    episode_counter = 0
    
    checkpoint_interval = getattr(params, 'checkpoint_interval', 1000)
    
    try:
        while frame_idx < params.total_steps:
            # 检查停止信号
            if stop_event and stop_event.is_set():
                print("Training stopped by user")
                break
            
            frame_idx += 1
            
            if params.use_noisy:
                epsilon = 0.0
                net.reset_noise()
                tgt_net.reset_noise()
            else:
                epsilon = max(
                    params.epsilon_final,
                    params.epsilon_start - frame_idx / params.epsilon_decay_last_frame,
                )
            
            reward = agent.play_step(net, epsilon, device=device)
            
            if reward is not None:
                total_rewards.append(reward)
                episode_counter += 1
                m_reward = np.mean(total_rewards[-200:])
                
                # 更新最优布局
                if m_reward > best_reward:
                    best_reward = m_reward
                    # 保存布局
                    layout_path = getattr(env.env, 'layout_path', None)
                    if layout_path and Path(layout_path).exists():
                        with open(layout_path, 'r', encoding='utf-8') as f:
                            best_layout = json.load(f)
                        
                        # 布局结果回调
                        if result_callback:
                            result_callback(best_layout)
                
                # 进度回调
                if progress_callback and frame_idx % 100 == 0:
                    metrics = {}
                    if hasattr(env.env, 'last_metrics') and env.env.last_metrics:
                        metrics = env.env.last_metrics
                    
                    progress_callback(
                        step=frame_idx,
                        episode=episode_counter,
                        current_reward=reward,
                        mean_reward=m_reward,
                        best_reward=best_reward,
                        progress_pct=frame_idx / params.total_steps * 100,
                        epsilon=epsilon,
                        metrics=metrics,
                    )
            
            if len(buffer) < params.replay_start_size:
                continue
            
            buffer.update_beta(frame_idx)
            
            if params.use_noisy:
                net.reset_noise()
                tgt_net.reset_noise()
            
            optimizer.zero_grad()
            samples, batch_indices, weights = buffer.sample(params.batch_size)
            states, actions, rewards_batch, dones, next_states = zip(*samples)
            batch = (
                np.array(states, copy=False),
                np.array(actions, copy=False),
                np.array(rewards_batch, dtype=np.float32),
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
            
            loss_v.backward()
            optimizer.step()
            
            if params.use_prior:
                buffer.update_priorities(batch_indices, sample_prios_v)
            
            if frame_idx % params.sync_target_frames == 0:
                tgt_net.load_state_dict(net.state_dict())
    
    finally:
        # 清理
        _cleanup_experiment_layout(experiment_layout_path)
    
    return {
        'best_reward': best_reward,
        'best_layout': best_layout,
        'total_episodes': episode_counter,
        'final_mean_reward': np.mean(total_rewards[-200:]) if total_rewards else 0,
        'total_steps': frame_idx,
    }

