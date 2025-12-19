"""
回放服务：加载检查点并计算热力图
"""
import sys
from pathlib import Path
from typing import Dict, Optional, List
import json
import torch
import numpy as np

# 添加项目根目录
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


class ReplayService:
    """回放服务"""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self._env = None
        self._net = None
        self._current_step = 0
        self._placement_history = []
    
    def load_checkpoint(
        self,
        model_path: str,
        layout_config_path: str,
        factory_config_path: str,
        training_params: Dict = None,
    ) -> bool:
        """
        加载检查点准备回放
        
        Returns:
            是否成功
        """
        from environment.gym_wrapper import FactoryEnv
        from agent.dqn_model import DQN, DuelingDQN
        
        try:
            # 创建环境
            training_params = training_params or {}
            weights = training_params.get('weights', {})
            objective_weights = {
                'transportation_intensity': weights.get('distance', 0.20),
                'material_flow_clarity': weights.get('logistics', 0.30),
                'space_utilization': weights.get('flow', 0.20),
                'throughput_time': weights.get('throughput', 0.25),
                'utilization': weights.get('utilization', 0.05),
            }
            
            self._env = FactoryEnv(
                config_path=factory_config_path,
                use_simulation=False,  # 回放时不需要仿真
                objective_weights=objective_weights,
                placement_order=training_params.get('placement_order', 'default'),
                layout_path=layout_config_path,
            )
            
            # 加载模型
            checkpoint = torch.load(model_path, map_location='cpu')
            
            use_dueling = training_params.get('dueling', False)
            model_cls = DuelingDQN if use_dueling else DQN
            
            self._net = model_cls(
                self._env.observation_space.shape,
                self._env.action_space.n,
                use_noisy=training_params.get('noisy_net', False),
            )
            self._net.load_state_dict(checkpoint['model_state_dict'])
            self._net.eval()
            
            # 重置环境
            self._env.reset()
            self._current_step = 0
            self._placement_history = []
            
            return True
            
        except Exception as e:
            print(f"加载检查点失败: {e}")
            return False
    
    def get_total_steps(self) -> int:
        """获取总步数（功能单元数量）"""
        if self._env is None:
            return 0
        return getattr(self._env.env, 'num_units', 0)
    
    def get_current_step(self) -> int:
        """获取当前步数"""
        return self._current_step
    
    def get_step_data(self, step: int = None) -> Dict:
        """
        获取指定步骤的数据
        
        Args:
            step: 步骤索引（None 表示当前步）
            
        Returns:
            步骤数据
        """
        if self._env is None or self._net is None:
            return {'error': 'Not initialized'}
        
        if step is not None and step != self._current_step:
            # 需要重新执行到目标步骤
            self._seek_to_step(step)
        
        # 获取当前状态
        state = self._env.env._get_state() if hasattr(self._env, 'env') else None
        
        # 获取当前单元信息
        current_unit = None
        if hasattr(self._env, 'env'):
            env = self._env.env
            if env.current_unit_idx < env.num_units:
                current_unit = env.functional_units[env.current_unit_idx]
        
        # 获取热力图
        heatmap = self.get_heatmap()
        
        return {
            'step': self._current_step,
            'total_steps': self.get_total_steps(),
            'current_unit': current_unit,
            'placed_units': self._placement_history.copy(),
            'heatmap': heatmap,
        }
    
    def step_forward(self) -> Dict:
        """
        执行一步（使用模型选择动作）
        
        Returns:
            步骤结果
        """
        if self._env is None or self._net is None:
            return {'error': 'Not initialized'}
        
        if self._current_step >= self.get_total_steps():
            return {'done': True}
        
        # 获取当前状态
        state = self._get_current_state()
        
        # 使用模型选择动作
        from agent.agent import get_q_values_for_state
        q_values = get_q_values_for_state(self._net, state)
        
        # 获取有效动作
        valid_actions = self._env.get_valid_actions()
        
        if not valid_actions:
            return {'error': 'No valid actions', 'done': True}
        
        # 在有效动作中选择最佳
        valid_q = [(a, q_values[a]) for a in valid_actions]
        best_action = max(valid_q, key=lambda x: x[1])[0]
        
        # 执行动作
        next_state, reward, done, info = self._env.step(best_action)
        
        # 记录放置历史
        self._placement_history.append({
            'step': self._current_step,
            'action': best_action,
            'q_value': float(q_values[best_action]),
        })
        
        self._current_step += 1
        
        return {
            'step': self._current_step,
            'action': best_action,
            'reward': reward,
            'done': done,
        }
    
    def _seek_to_step(self, target_step: int):
        """跳转到指定步骤"""
        # 重置环境
        self._env.reset()
        self._current_step = 0
        self._placement_history = []
        
        # 执行到目标步骤
        while self._current_step < target_step:
            result = self.step_forward()
            if result.get('done') or result.get('error'):
                break
    
    def _get_current_state(self):
        """获取当前状态的 numpy 数组"""
        if hasattr(self._env, 'env'):
            state_dict = self._env.env._get_state()
            # 转换为模型输入格式
            return self._env._dict_to_array(state_dict)
        return None
    
    def get_heatmap(self) -> Dict:
        """
        获取当前状态的 Q 值热力图
        
        Returns:
            热力图数据
        """
        if self._env is None or self._net is None:
            return {'error': 'Not initialized'}
        
        state = self._get_current_state()
        if state is None:
            return {'error': 'Cannot get state'}
        
        from agent.agent import get_q_values_heatmap
        
        return get_q_values_heatmap(self._net, self._env, state)
    
    def get_layout_state(self) -> Dict:
        """获取当前布局状态（用于可视化）"""
        if self._env is None:
            return {'error': 'Not initialized'}
        
        env = self._env.env if hasattr(self._env, 'env') else self._env
        
        placed_units = []
        for unit_id, x, y, rotation in getattr(env, 'placed_units', []):
            placed_units.append({
                'unit_id': unit_id,
                'x': x,
                'y': y,
                'angle': rotation,
            })
        
        return {
            'grid_size': env.grid_size,
            'placed_units': placed_units,
            'current_unit_idx': env.current_unit_idx,
        }


# 全局实例缓存
_replay_instances: Dict[str, ReplayService] = {}


def get_replay_service(project_id: str) -> ReplayService:
    """获取或创建回放服务实例"""
    if project_id not in _replay_instances:
        project_dir = Path(__file__).parent.parent.parent.parent / "data" / "projects" / project_id
        _replay_instances[project_id] = ReplayService(project_dir)
    return _replay_instances[project_id]


def clear_replay_service(project_id: str):
    """清除回放服务实例"""
    if project_id in _replay_instances:
        del _replay_instances[project_id]










