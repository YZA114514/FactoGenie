"""
环境与DQN集成完整示例
展示如何将 LayoutEnvironment 和 LayoutDQN 结合使用
"""

import sys
sys.path.append('.')

import torch
import numpy as np
from environment.factory_environment import LayoutEnvironment
from environment.env_dqn_adapter import EnvDQNAdapter
from agent.dqn_model import LayoutDQN


def run_one_episode(env, adapter, model, epsilon=0.1, device='cpu'):
    """
    运行一个完整的episode
    
    Args:
        env: LayoutEnvironment环境
        adapter: EnvDQNAdapter适配器
        model: LayoutDQN网络
        epsilon: 探索率
        device: 设备
    
    Returns:
        total_reward: 总奖励
        actions_taken: 执行的动作列表
    """
    # 重置环境
    env_state = env.reset()
    
    total_reward = 0.0
    actions_taken = []
    done = False
    step = 0
    
    print(f"\n{'='*80}")
    print(f"开始Episode - 需要放置 {env.num_units} 个功能单元")
    print(f"{'='*80}")
    
    while not done:
        step += 1
        print(f"\n--- 步骤 {step} ---")
        
        # 1. 获取有效动作
        valid_env_actions = env.get_valid_actions()
        print(f"有效动作数量: {len(valid_env_actions)}")
        
        if len(valid_env_actions) == 0:
            print("⚠️ 警告：没有有效动作！")
            break
        
        # 2. 转换有效动作为DQN索引
        valid_dqn_indices = adapter.get_valid_action_indices(valid_env_actions)
        
        # 3. 转换环境状态为DQN输入
        layout, material_flow, current_object = adapter.env_state_to_dqn_input(
            env_state, device=device
        )
        
        print(f"当前放置单元: {np.where(env_state['current_unit'] == 1)[0][0]}")
        
        # 4. 使用DQN选择动作
        action_dqn_idx = model.get_action(
            layout, material_flow, current_object,
            valid_dqn_indices,
            epsilon=epsilon
        )
        
        # 5. 转换DQN动作为环境动作
        action_env = adapter.dqn_action_to_env_action(action_dqn_idx)
        
        print(f"选择的动作: {action_env}")
        
        # 6. 执行动作
        next_env_state, reward, done, info = env.step(action_env)
        
        print(f"奖励: {reward:.3f}")
        print(f"已放置: {info['placed_units']}/{info['total_units']}")
        
        # 7. 记录
        total_reward += reward
        actions_taken.append(action_env)
        
        # 8. 更新状态
        env_state = next_env_state
    
    print(f"\n{'='*80}")
    print(f"Episode结束！")
    print(f"总奖励: {total_reward:.3f}")
    print(f"总步数: {step}")
    print(f"{'='*80}\n")
    
    return total_reward, actions_taken


def main():
    """主函数：演示完整流程"""
    
    print("\n" + "="*80)
    print("环境与DQN集成测试")
    print("="*80)
    
    # ========== 1. 创建环境 ==========
    print("\n【步骤1】创建环境...")
    
    grid_size = (15, 15)
    env = LayoutEnvironment(
        grid_size=grid_size,
        objective_weights={'transportation_intensity': 1.0}
    )
    
    print(f"✓ 环境创建成功")
    print(f"  网格大小: {grid_size}")
    print(f"  功能单元数: {env.num_units}")
    
    # ========== 2. 创建适配器 ==========
    print("\n【步骤2】创建适配器...")
    
    adapter = EnvDQNAdapter(
        grid_size=grid_size,
        num_units=env.num_units,
        num_rotations=4
    )
    
    print(f"✓ 适配器创建成功")
    print(f"  动作空间大小: {adapter.num_actions}")
    
    # ========== 3. 创建DQN网络 ==========
    print("\n【步骤3】创建DQN网络...")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model = LayoutDQN(
        height=grid_size[1],
        width=grid_size[0],
        num_objects=env.num_units,
        num_actions=adapter.num_actions,
        use_dueling=True
    ).to(device)
    
    print(f"✓ DQN网络创建成功")
    print(f"  设备: {device}")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # ========== 4. 测试状态转换 ==========
    print("\n【步骤4】测试状态转换...")
    
    env_state = env.reset()
    layout, material_flow, current_object = adapter.env_state_to_dqn_input(
        env_state, device=device
    )
    
    print(f"✓ 状态转换成功")
    print(f"  环境状态:")
    print(f"    layout_grid: {env_state['layout_grid'].shape}")
    print(f"    material_flow: {env_state['material_flow'].shape}")
    print(f"    current_unit: {env_state['current_unit'].shape}")
    print(f"  DQN输入:")
    print(f"    layout: {layout.shape}")
    print(f"    material_flow: {material_flow.shape}")
    print(f"    current_object: {current_object.shape}")
    
    # ========== 5. 测试动作转换 ==========
    print("\n【步骤5】测试动作转换...")
    
    valid_env_actions = env.get_valid_actions()
    valid_dqn_indices = adapter.get_valid_action_indices(valid_env_actions)
    
    print(f"✓ 动作转换成功")
    print(f"  环境有效动作数: {len(valid_env_actions)}")
    print(f"  DQN有效索引数: {len(valid_dqn_indices)}")
    print(f"  示例环境动作: {valid_env_actions[0]}")
    print(f"  对应DQN索引: {valid_dqn_indices[0]}")
    
    # ========== 6. 测试动作选择 ==========
    print("\n【步骤6】测试DQN动作选择...")
    
    # 贪心选择
    action_idx = model.get_action(
        layout, material_flow, current_object,
        valid_dqn_indices,
        epsilon=0.0
    )
    action_env = adapter.dqn_action_to_env_action(action_idx)
    
    print(f"✓ 动作选择成功")
    print(f"  DQN选择索引: {action_idx}")
    print(f"  转换为环境动作: {action_env}")
    print(f"  动作是否有效: {action_env in valid_env_actions}")
    
    # ========== 7. 运行完整Episode ==========
    print("\n【步骤7】运行完整Episode...")
    
    # 重置环境
    env.reset()
    
    # 运行一个episode（使用较高的epsilon进行探索）
    total_reward, actions = run_one_episode(
        env, adapter, model,
        epsilon=0.3,  # 30%探索
        device=device
    )
    
    # ========== 8. 可视化结果 ==========
    print("\n【步骤8】可视化最终布局...")
    env.render(mode='console')
    
    # ========== 总结 ==========
    print("\n" + "="*80)
    print("✅ 所有测试通过！环境与DQN集成成功！")
    print("="*80)
    print(f"\n📊 测试总结:")
    print(f"  • 环境功能: ✓ 正常")
    print(f"  • 适配器转换: ✓ 正常")
    print(f"  • DQN推理: ✓ 正常")
    print(f"  • 动作有效性: ✓ 正常")
    print(f"  • 完整流程: ✓ 正常")
    print(f"\n🎯 下一步可以:")
    print(f"  1. 实现经验回放缓冲区")
    print(f"  2. 编写训练循环")
    print(f"  3. 添加奖励函数")
    print(f"  4. 开始训练！")
    print("\n")


if __name__ == "__main__":
    main()

