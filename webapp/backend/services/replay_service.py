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
        self._state_history = []  # 保存历史状态以便回退
        self._initial_state = None  # 保存初始状态
        self._saved_layout = None  # 保存的布局JSON数据（训练时保存的最终布局）
        self._current_episode = None  # 当前加载的episode
    
    def load_checkpoint(
        self,
        model_path: str,
        layout_config_path: str,
        factory_config_path: str,
        training_params: Dict = None,
        layout_path: str = None,  # 训练时保存的布局JSON路径
    ) -> bool:
        """
        加载检查点准备回放
        
        Args:
            model_path: 模型文件路径
            layout_config_path: 布局配置文件路径（初始配置）
            factory_config_path: 工厂配置文件路径
            training_params: 训练参数
            layout_path: 训练时保存的布局JSON路径（用于显示布局快照）
        
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
            
            # 加载训练时保存的布局JSON（用于显示布局快照）
            self._saved_layout = None
            if layout_path and Path(layout_path).exists():
                try:
                    with open(layout_path, 'r', encoding='utf-8') as f:
                        self._saved_layout = json.load(f)
                    print(f"[INFO] 已加载保存的布局: {layout_path}")
                except Exception as e:
                    print(f"[WARN] 加载布局文件失败: {layout_path}, 错误: {e}")
            
            # 重置环境
            self._env.reset()
            self._current_step = 0
            self._placement_history = []
            self._state_history = []
            # 保存初始状态
            self._initial_state = self._get_env_state_snapshot()
            self._state_history.append(self._initial_state)
            
            return True
            
        except Exception as e:
            print(f"加载检查点失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_total_steps(self) -> int:
        """
        获取总步数（可移动单元数量，排除固定位置的单元）
        
        优先从保存的布局JSON获取，以确保与回放进度一致
        """
        # 优先使用保存布局的可移动单元数量
        if self._saved_layout is not None:
            saved_layout_data = self.get_saved_layout(max_step=None)
            return saved_layout_data.get('total_movable_units', 0)
        
        # 回退到从环境获取
        if self._env is None:
            return 0
        env = self._env.env if hasattr(self._env, 'env') else self._env
        num_units = getattr(env, 'num_units', 0)
        # 排除固定位置的单元（这些单元会在 reset 时自动放置，不参与 Agent 决策）
        fixed_position_map = getattr(env, '_fixed_position_map', {})
        if fixed_position_map:
            functional_units = getattr(env, 'functional_units', [])
            fixed_count = sum(1 for u in functional_units 
                            if u.get('id') in fixed_position_map or u.get('name') in fixed_position_map)
            return num_units - fixed_count
        return num_units
    
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
        
        # 获取当前状态（确保状态是最新的）
        state = self._get_current_state()
        if state is None:
            return {'error': 'Cannot get current state'}
        
        # 获取当前单元信息
        current_unit = None
        if hasattr(self._env, 'env'):
            env = self._env.env
            if hasattr(env, 'current_unit_idx') and env.current_unit_idx < len(getattr(env, 'functional_units', [])):
                current_unit = env.functional_units[env.current_unit_idx]
        
        # 获取当前步骤实际执行的动作（用于显示在热力图中）
        # 逻辑说明：
        # - placement_history[i] 记录的是步骤i执行的动作
        # - 执行步骤i的动作后，_current_step变成i+1
        # - 如果当前在步骤N（_current_step=N），说明已经执行了步骤0到N-1的动作
        # - 想要查看步骤N的热力图，应该显示"在步骤N的状态下，模型会选择什么动作"
        # - 如果步骤N已经执行了动作，应该显示步骤N实际执行的动作（placement_history[N]）
        # - 如果步骤N还没有执行动作，显示Q值最大的动作
        actual_action = None
        if self._placement_history and len(self._placement_history) > 0:
            # 检查当前步骤是否已经执行了动作
            # placement_history的长度表示已执行的步骤数
            # 如果_current_step < len(placement_history)，说明当前步骤已经执行了动作
            if self._current_step < len(self._placement_history):
                # 当前步骤已经执行了动作，显示实际执行的动作
                actual_action = self._placement_history[self._current_step].get('action')
                print(f"[DEBUG] 步骤 {self._current_step} 已执行动作: {actual_action}")
            else:
                print(f"[DEBUG] 步骤 {self._current_step} 尚未执行动作，显示Q值最大的动作")
        
        # 获取热力图（每次都重新计算，确保是最新的，并传入实际执行的动作）
        # 如果actual_action为None，则显示Q值最大的动作
        heatmap = self.get_heatmap(actual_action=actual_action)
        
        # 获取回放进度布局：从保存的布局中截取前N个单元（N=当前步骤+1，因为步骤从0开始）
        # 这样可以保证回放进度的最终结果与保存的布局一致
        replay_layout = self.get_saved_layout(max_step=self._current_step + 1)
        
        # 同时返回保存的最终布局（用于对比）
        saved_layout = self.get_saved_layout(max_step=None)
        
        # 获取已摆放单元列表（从环境获取，包含单元ID）
        placed_units_list = []
        if hasattr(self._env, 'env'):
            env = self._env.env
            functional_units = getattr(env, 'functional_units', [])
            placed_units_env = getattr(env, 'placed_units', [])
            for unit_idx, x, y, rotation in placed_units_env:
                if unit_idx < len(functional_units):
                    unit = functional_units[unit_idx]
                    placed_units_list.append({
                        'unit_id': unit.get('id', unit.get('name', f'unit_{unit_idx}')),
                        'name': unit.get('name', unit.get('id', f'unit_{unit_idx}')),
                        'x': x,
                        'y': y,
                        'angle': rotation,
                    })
            
            # 添加固定障碍物（它们不在placed_units中，但应该显示在已摆放列表中）
            layout_template = getattr(env, 'layout_template', None)
            if layout_template:
                fixed_obstacle_ids = set(env.placement_constraints.get('fixed_obstacles', []))
                obstacles = layout_template.get('obstacles', [])
                for obs in obstacles:
                    obs_id = obs.get('id', '')
                    if obs_id in fixed_obstacle_ids:
                        placed_units_list.append({
                            'unit_id': obs_id,
                            'name': obs_id,
                            'x': obs.get('x', 0),
                            'y': obs.get('y', 0),
                            'angle': obs.get('angle', 0),
                            'is_fixed_obstacle': True,  # 标记为固定障碍物
                        })
        
        return {
            'step': self._current_step,
            'total_steps': self.get_total_steps(),
            'current_unit': current_unit,
            'placed_units': placed_units_list,  # 使用从环境获取的已摆放单元列表
            'heatmap': heatmap,
            'layout': replay_layout,  # 回放进度布局（从保存的布局截取前N个单元）
            'saved_layout': saved_layout,  # 训练时保存的最终布局（用于对比）
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
        
        # 调试：打印执行的动作信息
        action_dict = None
        if hasattr(self._env, '_action_idx_to_dict'):
            action_dict = self._env._action_idx_to_dict(best_action)
        print(f"[DEBUG] 步骤 {self._current_step} 执行动作: idx={best_action}, dict={action_dict}, reward={reward}")
        
        # 记录放置历史
        self._placement_history.append({
            'step': self._current_step,
            'action': best_action,
            'q_value': float(q_values[best_action]),
        })
        
        self._current_step += 1
        
        # 保存当前状态到历史
        current_state = self._get_env_state_snapshot()
        self._state_history.append(current_state)
        
        return {
            'step': self._current_step,
            'action': best_action,
            'reward': reward,
            'done': done,
        }
    
    def step_backward(self) -> Dict:
        """
        回退一步（通过重新执行到上一步）
        
        Returns:
            步骤结果
        """
        if self._env is None or self._net is None:
            return {'error': 'Not initialized'}
        
        if self._current_step <= 0:
            return {'error': 'Already at step 0', 'step': 0}
        
        # 通过跳转到上一步来实现回退
        target_step = self._current_step - 1
        self._seek_to_step(target_step)
        
        return {
            'step': self._current_step,
            'message': 'Stepped backward',
        }
    
    def _seek_to_step(self, target_step: int):
        """跳转到指定步骤"""
        if target_step < 0:
            target_step = 0
        if target_step > self.get_total_steps():
            target_step = self.get_total_steps()
        
        # 如果已经在目标步骤，不需要操作
        if target_step == self._current_step:
            return
        
        # 如果目标步骤小于当前步骤，需要重置并重新执行
        if target_step < self._current_step:
            # 重置环境
            self._env.reset()
            self._current_step = 0
            self._placement_history = []
            self._state_history = []
            if self._initial_state:
                self._state_history.append(self._initial_state)
        
        # 执行到目标步骤（从当前步骤继续）
        while self._current_step < target_step:
            result = self.step_forward()
            if result.get('done') or result.get('error'):
                break
    
    def _get_current_state(self):
        """获取当前状态的 numpy 数组"""
        if hasattr(self._env, 'env'):
            state_dict = self._env.env._get_state()
            # 转换为模型输入格式
            state_array = self._env._dict_to_array(state_dict)
            # 调试：打印状态信息
            if hasattr(self._env.env, 'current_unit_idx'):
                print(f"[DEBUG] 获取状态: current_step={self._current_step}, current_unit_idx={self._env.env.current_unit_idx}, placed_units={len(getattr(self._env.env, 'placed_units', []))}")
            return state_array
        return None
    
    def get_heatmap(self, actual_action=None) -> Dict:
        """
        获取当前状态的 Q 值热力图
        
        Args:
            actual_action: 实际执行的动作索引（可选，用于显示实际动作而不是最佳动作）
        
        Returns:
            热力图数据
        """
        if self._env is None or self._net is None:
            return {'error': 'Not initialized'}
        
        state = self._get_current_state()
        if state is None:
            return {'error': 'Cannot get state'}
        
        from agent.agent import get_q_values_heatmap
        
        return get_q_values_heatmap(self._net, self._env, state, actual_action=actual_action)
    
    def _get_env_state_snapshot(self) -> Dict:
        """获取环境状态的快照（用于回退）"""
        if self._env is None:
            return None
        
        env = self._env.env if hasattr(self._env, 'env') else self._env
        
        placed_units = []
        for unit_id, x, y, rotation in getattr(env, 'placed_units', []):
            placed_units.append((unit_id, x, y, rotation))
        
        return {
            'current_unit_idx': getattr(env, 'current_unit_idx', 0),
            'placed_units': placed_units.copy(),
        }
    
    def _restore_env_state(self, state_snapshot: Dict):
        """恢复环境状态"""
        if self._env is None or state_snapshot is None:
            return
        
        env = self._env.env if hasattr(self._env, 'env') else self._env
        
        # 重置环境
        self._env.reset()
        
        # 恢复已放置的单元
        for unit_id, x, y, rotation in state_snapshot.get('placed_units', []):
            # 找到对应的单元并放置
            unit_idx = None
            for idx, unit in enumerate(getattr(env, 'functional_units', [])):
                if unit.get('id') == unit_id:
                    unit_idx = idx
                    break
            
            if unit_idx is not None:
                # 计算动作索引（需要根据环境的具体实现）
                # 这里假设有一个方法可以直接放置单元
                try:
                    # 尝试通过环境的方法放置
                    if hasattr(env, '_place_unit_at_position'):
                        env._place_unit_at_position(unit_idx, x, y, rotation)
                    else:
                        # 如果没有直接方法，需要通过动作来放置
                        # 这里需要根据实际环境实现来调整
                        pass
                except Exception as e:
                    print(f"恢复状态时放置单元失败: {e}")
        
        # 恢复当前单元索引
        if hasattr(env, 'current_unit_idx'):
            env.current_unit_idx = state_snapshot.get('current_unit_idx', 0)
    
    def get_saved_layout(self, max_step: int = None) -> Dict:
        """
        获取训练时保存的布局（用于显示"布局快照"）
        
        Args:
            max_step: 最大步骤数，只返回前max_step个单元。None表示返回全部。
        
        这是训练时该检查点实际保存的布局，不是回放模拟的结果
        """
        if self._saved_layout is None:
            return {'error': 'No saved layout', 'placed_units': [], 'grid_size': [36, 18]}
        
        # 从保存的布局JSON中提取数据
        factory = self._saved_layout.get('factory', {})
        grid_size = [factory.get('length', 36), factory.get('width', 18)]
        
        all_units = []
        fixed_units = []  # 固定位置的单元（始终显示）
        movable_units = []  # 可移动的单元（按步骤显示）
        
        # 获取固定障碍物ID列表
        constraints = self._saved_layout.get('constraints', {})
        fixed_obstacle_ids = set(constraints.get('fixed_obstacles', []))
        fixed_position_ids = set(constraints.get('fixed_positions', {}).keys() if isinstance(constraints.get('fixed_positions'), dict) else [])
        
        # 处理功能单元（注意：布局JSON使用 'fus' 而不是 'functional_units'）
        fus = self._saved_layout.get('fus', self._saved_layout.get('functional_units', []))
        for unit in fus:
            unit_id = unit.get('id', unit.get('name', unit.get('label', '')))
            unit_data = {
                'unit_id': unit_id,
                'x': unit.get('x', 0),
                'y': unit.get('y', 0),
                'angle': unit.get('angle', 0),
                'width': unit.get('width', 4),
                'length': unit.get('length', unit.get('height', 4)),  # 兼容 height 字段
                'notch_length': unit.get('notch_length', 0),
                'notch_width': unit.get('notch_width', 0),
                'label': unit.get('label', unit.get('name', unit.get('id', ''))),
                'typeLabel': 'FU',
            }
            # 固定位置的单元始终显示
            if unit_id in fixed_position_ids:
                fixed_units.append(unit_data)
            else:
                movable_units.append(unit_data)
        
        # 处理障碍物
        for obs in self._saved_layout.get('obstacles', []):
            obs_id = obs.get('id', obs.get('name', obs.get('label', '')))
            obs_data = {
                'unit_id': obs_id,
                'x': obs.get('x', 0),
                'y': obs.get('y', 0),
                'angle': obs.get('angle', 0),
                'width': obs.get('width', 4),
                'length': obs.get('length', obs.get('height', 4)),  # 兼容 height 字段
                'notch_length': obs.get('notch_length', 0),
                'notch_width': obs.get('notch_width', 0),
                'label': obs.get('label', obs.get('id', obs.get('name', ''))),
                'typeLabel': 'Obstacle',
            }
            # 固定障碍物始终显示
            if obs_id in fixed_obstacle_ids:
                fixed_units.append(obs_data)
            else:
                movable_units.append(obs_data)
        
        # 根据max_step截取可移动单元
        if max_step is not None:
            # 只显示前max_step个可移动单元
            displayed_movable = movable_units[:max_step]
        else:
            displayed_movable = movable_units
        
        # 合并固定单元和可移动单元
        all_units = fixed_units + displayed_movable
        
        return {
            'grid_size': grid_size,
            'placed_units': all_units,
            'total_movable_units': len(movable_units),
        }
    
    def get_layout_state(self) -> Dict:
        """获取当前回放状态的布局（用于显示回放进度）"""
        if self._env is None:
            return {'error': 'Not initialized'}
        
        env = self._env.env if hasattr(self._env, 'env') else self._env
        
        placed_units = []
        functional_units = getattr(env, 'functional_units', [])
        
        # 环境的placed_units存储的是(unit_idx, x, y, rotation)
        for unit_idx, x, y, rotation in getattr(env, 'placed_units', []):
            # 从functional_units中查找单元信息
            if unit_idx < len(functional_units):
                unit_info = functional_units[unit_idx]
                unit_id = unit_info.get('id', unit_info.get('name', f'unit_{unit_idx}'))
                
                # 获取尺寸（注意：size是(length, width)，即(高, 宽)）
                size = unit_info.get('size', [5, 4])
                length, width = size[0], size[1]
                
                # 获取缺口信息（如station_4有缺口）
                notch = unit_info.get('notch', [0, 0])
                notch_length = notch[0] if len(notch) > 0 else 0
                notch_width = notch[1] if len(notch) > 1 else 0
                
                # 判断是否是障碍物
                is_obstacle = unit_info.get('is_obstacle', False)
                
                unit_data = {
                    'unit_id': unit_id,
                    'x': x,
                    'y': y,
                    'angle': rotation,
                    'width': width,  # 宽度
                    'length': length,  # 使用length而不是height，与ResultsPage一致
                    'notch_length': notch_length,
                    'notch_width': notch_width,
                    'label': unit_info.get('name', unit_info.get('id', unit_id)),
                    'typeLabel': 'Obstacle' if is_obstacle else 'FU',
                }
                
                placed_units.append(unit_data)
        
        # 添加固定障碍物（从layout_template中读取）
        fixed_obstacles = []
        layout_template = getattr(env, 'layout_template', None)
        if layout_template:
            fixed_obstacle_ids = set(env.placement_constraints.get('fixed_obstacles', []))
            obstacles = layout_template.get('obstacles', [])
            for obs in obstacles:
                obs_id = obs.get('id', '')
                if obs_id in fixed_obstacle_ids:
                    fixed_obstacles.append({
                        'unit_id': obs_id,
                        'x': obs.get('x', 0),
                        'y': obs.get('y', 0),
                        'angle': obs.get('angle', 0),
                        'width': obs.get('width', 4),
                        'length': obs.get('length', 4),  # 使用length
                        'notch_length': obs.get('notch_length', 0),
                        'notch_width': obs.get('notch_width', 0),
                        'label': obs_id,
                        'typeLabel': 'Obstacle',
                    })
        
        # 合并已放置单元和固定障碍物
        all_units = placed_units + fixed_obstacles
        
        return {
            'grid_size': env.grid_size,
            'placed_units': all_units,  # 包含固定障碍物
            'current_unit_idx': getattr(env, 'current_unit_idx', 0),
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











