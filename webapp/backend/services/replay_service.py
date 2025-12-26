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
        print(f"[DEBUG] load_checkpoint: model_path={model_path}")
        print(f"[DEBUG] load_checkpoint: layout_config_path={layout_config_path}")
        print(f"[DEBUG] load_checkpoint: layout_path={layout_path}")
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
            self._net.eval()  # 设置为评估模式，这会禁用NoisyNet的噪声
            
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
            
            # 调试：打印环境和布局的单元数量
            env = self._env.env if hasattr(self._env, 'env') else self._env
            print(f"[DEBUG] Environment num_units: {getattr(env, 'num_units', 'N/A')}")
            print(f"[DEBUG] Environment functional_units count: {len(getattr(env, 'functional_units', []))}")
            if hasattr(env, 'functional_units'):
                for i, u in enumerate(env.functional_units):
                    print(f"[DEBUG]   Unit {i}: {u.get('id', 'N/A')}")
            print(f"[DEBUG] get_total_steps(): {self.get_total_steps()}")
            if self._saved_layout:
                saved_data = self.get_saved_layout(max_step=None)
                print(f"[DEBUG] Saved layout total_movable_units: {saved_data.get('total_movable_units', 'N/A')}")
            
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

        # 当请求的步骤与当前不同，使用模型重新推理到目标步
        if step is not None and step != self._current_step:
            self._seek_to_step(step)

        # 获取当前状态用于热力图推理
        state = self._get_current_state()
        if state is None:
            return {'error': 'Cannot get current state'}

        env = self._env.env if hasattr(self._env, 'env') else self._env

        # 直接从环境获取当前待放置的单元信息（与实时推理一致）
        current_unit = None
        if hasattr(env, 'functional_units') and hasattr(env, 'current_unit_idx'):
            idx = getattr(env, 'current_unit_idx', -1)
            functional_units = getattr(env, 'functional_units', [])
            if 0 <= idx < len(functional_units):
                unit = functional_units[idx]
                size = unit.get('size', (1, 1))
                current_unit = {
                    'id': unit.get('id', unit.get('name', f'unit_{idx}')),
                    'name': unit.get('name', unit.get('id', f'unit_{idx}')),
                    'size': tuple(size) if isinstance(size, (list, tuple)) else (size, size),
                    'is_obstacle': unit.get('is_obstacle', False),
                }

        # 基于当前环境状态实时计算热力图
        heatmap = self.get_heatmap(actual_action=None)

        # 当前回放进度布局（基于实时推理结果）
        replay_layout = self.get_layout_state()

        # 保存的最终布局仅用于对比展示，不再驱动回放
        saved_layout = self.get_saved_layout(max_step=None)

        placed_units_list = replay_layout.get('placed_units', [])

        return {
            'step': self._current_step,
            'total_steps': self.get_total_steps(),
            'current_unit': current_unit,
            'placed_units': placed_units_list,
            'heatmap': heatmap,
            'layout': replay_layout,
            'saved_layout': saved_layout,
        }
    
    def step_forward(self) -> Dict:
        """
        执行一步回放
        
        回放模式：
        始终使用模型选择动作（实时计算），不依赖保存的布局记录。
        这样可以观察模型在当前状态下的真实决策。
        
        Returns:
            步骤结果
        """
        total_steps = self.get_total_steps()
        print(f"[DEBUG] step_forward: current_step={self._current_step}, total_steps={total_steps}")
        
        if self._current_step >= total_steps:
            print(f"[DEBUG] step_forward: Done! current_step >= total_steps")
            return {'done': True}
        
        # 始终使用模型选择动作
        if self._env is None or self._net is None:
            return {'error': 'Not initialized'}
        
        # 获取当前状态
        state = self._get_current_state()
        
        # 使用模型选择动作
        from agent.agent import get_q_values_for_state
        q_values = get_q_values_for_state(self._net, state)
        
        # 获取有效动作
        valid_actions = self._env.get_valid_actions()
        
        if not valid_actions:
            print(f"[DEBUG] step_forward: No valid actions at step {self._current_step}")
            # 如果没有有效动作，尝试跳过（或者结束）
            self._current_step += 1
            return {
                'step': self._current_step,
                'done': self._current_step >= total_steps,
                'warning': 'No valid actions in environment'
            }
        
        # 在有效动作中选择最佳
        valid_q = [(a, q_values[a]) for a in valid_actions]
        best_action = max(valid_q, key=lambda x: x[1])[0]
        
        # 执行动作
        next_state, reward, done, info = self._env.step(best_action)
        
        # 调试：打印执行的动作信息
        action_dict = None
        if hasattr(self._env, '_action_idx_to_dict'):
            action_dict = self._env._action_idx_to_dict(best_action)
        print(f"[DEBUG] 步骤 {self._current_step} 执行动作: idx={best_action}, dict={action_dict}, reward={reward}, done={done}")
        
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
            'done': self._current_step >= total_steps,
        }
    
    def step_backward(self) -> Dict:
        """
        回退一步
        
        Returns:
            步骤结果
        """
        if self._current_step <= 0:
            return {'error': 'Already at step 0', 'step': 0}

        if self._env is None or self._net is None:
            return {'error': 'Not initialized'}

        target_step = self._current_step - 1
        self._seek_to_step(target_step)

        return {
            'step': self._current_step,
            'message': 'Stepped backward',
        }
    
    def _seek_to_step(self, target_step: int):
        """跳转到指定步骤"""
        total_steps = self.get_total_steps()
        
        if target_step < 0:
            target_step = 0
        if target_step > total_steps:
            target_step = total_steps
        
        # 如果已经在目标步骤，不需要操作
        if target_step == self._current_step:
            return
        
        # 如果目标步骤小于当前步骤，需要重置并重新执行
        if target_step < self._current_step:
            # 重置环境
            if self._env is not None:
                self._env.reset()
            self._current_step = 0
            self._placement_history = []
            self._state_history = []
            if self._initial_state:
                self._state_history.append(self._initial_state)
        
        # 执行到目标步骤（从当前步骤继续）
        # 注意：step_forward 现在始终使用模型进行实时决策
        while self._current_step < target_step:
            result = self.step_forward()
            if result.get('error'):
                print(f"[DEBUG] _seek_to_step: Error at step {self._current_step}, stopping")
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
    
    def _find_unit_in_saved_layout(self, unit_id: str) -> Optional[Dict]:
        """
        在保存的布局中查找指定ID的单元
        
        Args:
            unit_id: 单元ID
            
        Returns:
            单元数据（包含x, y, angle等）或None
        """
        if self._saved_layout is None:
            return None
        
        # 在fus中查找
        for unit in self._saved_layout.get('fus', []):
            if unit.get('id') == unit_id:
                return unit
        
        # 在obstacles中查找
        for obs in self._saved_layout.get('obstacles', []):
            if obs.get('id') == unit_id:
                return obs
        
        return None
    
    def _position_to_action(self, x: int, y: int, angle: int) -> int:
        """
        将位置和角度转换为动作索引
        
        Args:
            x: x坐标
            y: y坐标  
            angle: 角度（0, 90, 180, 270）
            
        Returns:
            动作索引
        """
        if self._env is None:
            return 0
        
        env = self._env.env if hasattr(self._env, 'env') else self._env
        
        # 优先使用 grid_size 或 nx/ny
        if hasattr(env, 'grid_size'):
            nx, ny = env.grid_size
        elif hasattr(env, 'nx') and hasattr(env, 'ny'):
            nx, ny = env.nx, env.ny
        else:
            nx = getattr(env, 'grid_cols', 36)
            ny = getattr(env, 'grid_rows', 18)
        
        # 角度转rotation索引（0, 90, 180, 270 -> 0, 1, 2, 3）
        rotation = (angle // 90) % 4
        
        # action = rotation * (nx * ny) + y * nx + x
        action_idx = rotation * (nx * ny) + y * nx + x
        return action_idx
    
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
    
    def _sync_env_to_step(self, target_step: int):
        """
        将环境状态同步到指定步骤（基于保存的布局）
        
        这个方法用于在保存布局模式下，让环境状态与当前步骤保持一致，
        以便正确计算热力图和获取当前单元信息。
        
        Args:
            target_step: 目标步骤（0-based）
        """
        if self._env is None or self._saved_layout is None:
            return
        
        # 重置环境
        self._env.reset()
        env = self._env.env if hasattr(self._env, 'env') else self._env
        
        # 从保存的布局中提取可移动单元
        constraints = self._saved_layout.get('constraints', {})
        fixed_position_ids = set(constraints.get('fixed_positions', {}).keys() if isinstance(constraints.get('fixed_positions'), dict) else [])
        fixed_obstacle_ids = set(constraints.get('fixed_obstacles', []))
        
        movable_units_data = []
        
        # 提取功能单元
        fus = self._saved_layout.get('fus', [])
        for unit in fus:
            unit_id = unit.get('id', '')
            if unit_id not in fixed_position_ids:
                movable_units_data.append({
                    'id': unit_id,
                    'x': unit.get('x', 0),
                    'y': unit.get('y', 0),
                    'angle': unit.get('angle', 0),
                    'is_obstacle': False,
                })
        
        # 提取可移动障碍物
        obstacles = self._saved_layout.get('obstacles', [])
        for obs in obstacles:
            obs_id = obs.get('id', '')
            if obs_id not in fixed_obstacle_ids:
                movable_units_data.append({
                    'id': obs_id,
                    'x': obs.get('x', 0),
                    'y': obs.get('y', 0),
                    'angle': obs.get('angle', 0),
                    'is_obstacle': True,
                })
        
        # 将前target_step个单元放置到环境中
        functional_units = getattr(env, 'functional_units', [])
        id_to_idx = {u.get('id', u.get('name', '')): i for i, u in enumerate(functional_units)}
        
        for i in range(min(target_step, len(movable_units_data))):
            unit_data = movable_units_data[i]
            unit_id = unit_data['id']
            
            if unit_id in id_to_idx:
                unit_idx = id_to_idx[unit_id]
                x = unit_data['x']
                y = unit_data['y']
                rotation = unit_data['angle']
                
                # 将动作转换为字典格式并执行
                action_dict = {'x': x, 'y': y, 'rotation': rotation}
                try:
                    # 直接调用环境的step方法
                    env.step(action_dict)
                except Exception as e:
                    print(f"[WARN] 同步环境状态时放置单元 {unit_id} 失败: {e}")
                    # 如果step失败，尝试直接更新placed_units（不推荐，但作为fallback）
                    if hasattr(env, 'placed_units'):
                        env.placed_units.append((unit_idx, x, y, rotation))
                        # 更新布局网格
                        if hasattr(env, '_place_unit'):
                            unit = functional_units[unit_idx]
                            env._place_unit(unit, x, y, rotation)
        
        # 更新current_unit_idx到目标步骤
        if hasattr(env, 'current_unit_idx'):
            env.current_unit_idx = target_step
            # 跳过固定位置的单元
            if hasattr(env, '_skip_fixed_units'):
                env._skip_fixed_units()
        
        print(f"[DEBUG] _sync_env_to_step: 已同步环境到步骤 {target_step}")
    
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











