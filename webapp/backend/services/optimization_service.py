"""
优化服务：封装核心算法调用
"""
import sys
from pathlib import Path
from typing import Dict, Optional, Callable, Any
import json
import threading
import torch

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class OptimizationService:
    """优化服务：封装训练和评估功能"""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.stop_event = threading.Event()
        self._env = None
        self._agent = None
    
    def create_environment(
        self,
        factory_config_path: str,
        layout_config_path: str,
        training_params: Dict,
    ):
        """创建训练环境"""
        from environment.gym_wrapper import FactoryEnv
        
        # 构建权重
        weights = training_params.get('weights', {})
        objective_weights = {
            'transportation_intensity': weights.get('distance', 0.20),
            'material_flow_clarity': weights.get('logistics', 0.30),
            'space_utilization': weights.get('flow', 0.20),
            'throughput_time': weights.get('throughput', 0.25),
            'utilization': weights.get('utilization', 0.05),
        }
        
        # 校准指标边界（如果需要）
        metric_bounds = None
        calibrate_episodes = training_params.get('calibrate_episodes', 0)
        throughput_target = training_params.get('throughput_target', None)
        
        if calibrate_episodes > 0:
            print(f"\n{'='*50}")
            print(f"开始校准指标边界 ({calibrate_episodes} 个随机布局)...")
            print(f"{'='*50}")
            
            from calibration.bounds import BoundsManager
            bounds_manager = BoundsManager()
            
            bounds = bounds_manager.load_or_calibrate(
                factory_config_path=factory_config_path,
                layout_config_path=layout_config_path,
                n_episodes=calibrate_episodes,
                simulation_duration=training_params.get('simulation_duration', 2000),
                throughput_target=throughput_target,
            )
            
            # 提取边界用于环境
            metric_bounds = bounds_manager.get_bounds_for_reward(
                factory_config_path=factory_config_path,
                layout_config_path=layout_config_path,
            )
            print(f"\n使用校准边界: {metric_bounds}")
        
        self._env = FactoryEnv(
            config_path=factory_config_path,
            use_simulation=training_params.get('use_simulation', True),
            simulation_duration=training_params.get('simulation_duration', 2000),
            objective_weights=objective_weights,
            placement_order=training_params.get('placement_order', 'default'),
            layout_path=layout_config_path,
            metric_bounds=metric_bounds,  # 传递校准的边界
        )
        
        return self._env
    
    def run_training(
        self,
        training_params: Dict,
        progress_callback: Callable[[Dict], None] = None,
        checkpoint_callback: Callable[[int, float, str], None] = None,
    ) -> Dict:
        """
        运行训练
        
        Args:
            training_params: 训练参数
            progress_callback: 进度回调 (step, episode, reward, loss, epsilon)
            checkpoint_callback: 检查点回调 (episode, reward, model_path)
            
        Returns:
            训练结果
        """
        if self._env is None:
            return {'success': False, 'error': 'Environment not created'}
        
        from agent.dqn_model import DQN, DuelingDQN
        from agent.agent import Agent
        from agent.replay_buffer import ExperienceBuffer, PrioReplayBuffer
        import torch.optim as optim
        import numpy as np
        
        device = torch.device('cpu')
        
        # 创建网络
        use_dueling = training_params.get('dueling', False)
        use_noisy = training_params.get('noisy_net', False)
        model_cls = DuelingDQN if use_dueling else DQN
        
        net = model_cls(
            self._env.observation_space.shape,
            self._env.action_space.n,
            use_noisy=use_noisy,
        ).to(device)
        
        tgt_net = model_cls(
            self._env.observation_space.shape,
            self._env.action_space.n,
            use_noisy=use_noisy,
        ).to(device)
        tgt_net.load_state_dict(net.state_dict())
        
        optimizer = optim.Adam(net.parameters(), lr=training_params.get('learning_rate', 2e-5))
        
        # 创建经验回放
        use_prior = training_params.get('prioritized', False)
        buffer_cls = PrioReplayBuffer if use_prior else ExperienceBuffer
        buffer = buffer_cls(buf_size=training_params.get('replay_size', 50000))
        
        agent = Agent(self._env, buffer)
        
        # 训练参数
        total_steps = training_params.get('total_steps', 50000)
        epsilon_start = training_params.get('epsilon_start', 1.0)
        epsilon_final = training_params.get('epsilon_final', 0.05)
        epsilon_decay = training_params.get('epsilon_decay_frames', 150000)
        sync_target = training_params.get('sync_target_every', 2000)
        replay_start = training_params.get('replay_start_size', 5000)
        batch_size = training_params.get('batch_size', 32)
        checkpoint_interval = training_params.get('checkpoint_interval', 1000)
        
        # 初始化 CSV 文件用于保存训练指标
        import csv
        import time
        metrics_dir = self.project_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        rewards_csv = metrics_dir / "rewards.csv"
        losses_csv = metrics_dir / "losses.csv"
        
        # 写入 CSV 头
        with open(rewards_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['episode', 'step', 'reward', 'mean_reward_100', 'epsilon'])
        
        with open(losses_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'loss'])
        
        # 布局快照目录
        layouts_dir = self.project_dir / "layouts"
        layouts_dir.mkdir(parents=True, exist_ok=True)
        
        # 训练循环
        epsilon = epsilon_start
        total_rewards = []
        total_losses = []
        best_reward = -float('inf')
        episode_count = 0
        start_time = time.time()
        
        for frame_idx in range(1, total_steps + 1):
            # 检查停止信号
            if self.stop_event.is_set():
                break
            
            # 更新 epsilon
            if not use_noisy:
                epsilon = max(epsilon_final, epsilon_start - frame_idx / epsilon_decay)
            else:
                epsilon = 0.0
                net.reset_noise()
                tgt_net.reset_noise()
            
            # 执行一步
            reward = agent.play_step(net, epsilon, device=device)
            
            if reward is not None:
                total_rewards.append(reward)
                episode_count += 1
                mean_reward = np.mean(total_rewards[-100:])
                
                # 计算预计剩余时间
                elapsed = time.time() - start_time
                steps_per_sec = frame_idx / elapsed if elapsed > 0 else 0
                remaining_steps = total_steps - frame_idx
                estimated_remaining = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
                
                # 保存到 rewards.csv
                with open(rewards_csv, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([episode_count, frame_idx, reward, mean_reward, epsilon])
                
                # 进度回调（包含预计剩余时间和探索率）
                if progress_callback:
                    progress_callback({
                        'step': frame_idx,
                        'episode': episode_count,
                        'reward': reward,
                        'mean_reward': mean_reward,
                        'epsilon': epsilon,
                        'estimated_remaining': estimated_remaining,
                        'progress_pct': frame_idx / total_steps * 100,
                    })
                
                # 检查点保存
                if checkpoint_interval > 0 and episode_count % checkpoint_interval == 0:
                    model_path = self.project_dir / "checkpoints" / f"model_ep{episode_count}.pth"
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    torch.save({
                        'episode': episode_count,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'reward': reward,
                    }, model_path)
                    
                    # 保存布局快照
                    try:
                        layout_snapshot = self._get_layout_snapshot()
                        if layout_snapshot:
                            layout_path = layouts_dir / f"layout_ep{episode_count}.json"
                            with open(layout_path, 'w', encoding='utf-8') as f:
                                json.dump(layout_snapshot, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"保存布局快照失败: {e}")
                    
                    if checkpoint_callback:
                        checkpoint_callback(episode_count, reward, str(model_path))
                
                # 最佳模型和布局
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_path = self.project_dir / "checkpoints" / "model_best.pth"
                    best_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({
                        'episode': episode_count,
                        'model_state_dict': net.state_dict(),
                        'reward': reward,
                    }, best_path)
                    
                    # 保存最佳布局快照
                    try:
                        layout_snapshot = self._get_layout_snapshot()
                        if layout_snapshot:
                            best_layout_path = layouts_dir / "layout_best.json"
                            with open(best_layout_path, 'w', encoding='utf-8') as f:
                                json.dump(layout_snapshot, f, indent=2, ensure_ascii=False)
                    except Exception as e:
                        print(f"保存最佳布局快照失败: {e}")
            
            # 训练
            if len(buffer) >= replay_start:
                optimizer.zero_grad()
                samples, batch_indices, weights = buffer.sample(batch_size)
                
                from agent.agent import calc_loss_prio
                loss_v, sample_prios = calc_loss_prio(
                    self._prepare_batch(samples),
                    np.array(weights, dtype=np.float32),
                    net, tgt_net, device=device,
                    double_dqn=training_params.get('double_dqn', False),
                )
                
                loss_v.backward()
                optimizer.step()
                
                # 保存损失值到 losses.csv（每100步保存一次减少IO）
                loss_val = loss_v.item()
                total_losses.append(loss_val)
                if frame_idx % 100 == 0:
                    with open(losses_csv, 'a', newline='', encoding='utf-8') as f:
                        writer = csv.writer(f)
                        writer.writerow([frame_idx, loss_val])
                
                if use_prior:
                    buffer.update_priorities(batch_indices, sample_prios)
            
            # 同步目标网络
            if frame_idx % sync_target == 0:
                tgt_net.load_state_dict(net.state_dict())
        
        return {
            'success': True,
            'total_episodes': episode_count,
            'best_reward': best_reward,
            'final_reward': total_rewards[-1] if total_rewards else 0,
        }
    
    def _prepare_batch(self, samples):
        """准备批次数据"""
        import numpy as np
        states, actions, rewards, dones, next_states = zip(*samples)
        return (
            np.array(states, copy=False),
            np.array(actions, copy=False),
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            np.array(next_states, copy=False),
        )
    
    def _get_layout_snapshot(self) -> Optional[Dict]:
        """获取当前布局快照"""
        if self._env is None:
            return None
        
        try:
            # 获取底层环境
            env = self._env
            while hasattr(env, 'env'):
                env = env.env
            
            # 获取布局模板和已放置单元
            if hasattr(env, 'layout_template') and hasattr(env, 'placed_units'):
                from environment.layout_exporter import LayoutExporter
                exporter = LayoutExporter(env.layout_template, None)
                return exporter.export_layout_dict(
                    env.placed_units,
                    env.functional_units
                )
        except Exception as e:
            print(f"获取布局快照失败: {e}")
        
        return None
    
    def stop(self):
        """停止训练"""
        self.stop_event.set()
    
    def get_q_values(self, model_path: str, state) -> Dict:
        """
        获取指定状态的 Q 值（用于热力图）
        
        Args:
            model_path: 模型路径
            state: 当前状态
            
        Returns:
            Q 值字典
        """
        from agent.dqn_model import DQN
        
        if self._env is None:
            return {'error': 'Environment not created'}
        
        # 加载模型
        checkpoint = torch.load(model_path, map_location='cpu')
        
        net = DQN(
            self._env.observation_space.shape,
            self._env.action_space.n,
        )
        net.load_state_dict(checkpoint['model_state_dict'])
        net.eval()
        
        # 计算 Q 值
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = net(state_tensor).squeeze().numpy()
        
        return {
            'q_values': q_values.tolist(),
            'max_q': float(q_values.max()),
            'best_action': int(q_values.argmax()),
        }


def evaluate_layout(layout_path: str, factory_config_path: str, duration: float = 2000) -> Dict:
    """
    评估布局
    
    Args:
        layout_path: 布局文件路径
        factory_config_path: 工厂配置路径
        duration: 仿真时长
        
    Returns:
        评估结果
    """
    from simulation.interface import compute_metrics
    
    with open(layout_path, 'r') as f:
        layout = json.load(f)
    
    results = compute_metrics(layout, factory_config_path, duration=duration, detail=True)
    
    return {
        'average_route_distance': results.get('average_route_distance', 0),
        'total_logistics_intensity': results.get('total_logistics_intensity', 0),
        'finished_goods': results.get('finished_goods', 0),
        'throughput_rate': results.get('throughput_rate', 0),
        'space_utilization': results.get('space_utilization', 0),
        'station_utilization': results.get('station_utilization', {}),
    }








