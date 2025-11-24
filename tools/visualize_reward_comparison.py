"""
奖励函数对比可视化工具

展示修改前后奖励函数的差异，帮助理解非线性映射的效果
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

def old_distance_reward(distance, best=6.5, worst=24.0):
    """旧版距离奖励（线性）"""
    normalized = (distance - best) / (worst - best)
    return -np.clip(normalized, 0.0, 1.0)

def new_distance_reward(distance, best=6.0, worst=28.0, k= 3.5):
    """新版距离奖励（Tanh非线性）"""
    normalized = (distance - best) / (worst - best)
    normalized = np.clip(normalized, 0.0, 1.0)
    return -np.tanh(k * normalized) / np.tanh(k)

def old_logistics_reward(logistics, best=230.0, worst=900.0):
    """旧版物流强度奖励（线性）"""
    normalized = (logistics - best) / (worst - best)
    return -np.clip(normalized, 0.0, 1.0)

def new_logistics_reward(logistics, best=200.0, worst=1050.0, k=3.5):
    """新版物流强度奖励（Tanh非线性）"""
    normalized = (logistics - best) / (worst - best)
    normalized = np.clip(normalized, 0.0, 1.0)
    return -np.tanh(k * normalized) / np.tanh(k)

def old_throughput_reward(finished_goods):
    """旧版吞吐量奖励（分段线性）"""
    if finished_goods < 200:
        return -1.0
    elif finished_goods < 400:
        return (finished_goods - 400) / 200.0
    else:
        return min((finished_goods - 400) / 800.0, 0.0)

def new_throughput_reward(finished_goods, best=400, worst=120.0, k=3.0):
    """新版吞吐量奖励（Tanh非线性，反向）"""
    if finished_goods < worst:
        return -1.0
    elif finished_goods >= best:
        return 0.0
    else:
        # 反向：产量低惩罚重
        normalized = 1.0 - (finished_goods - worst) / (best - worst)
        normalized = np.clip(normalized, 0.0, 1.0)
        return -np.tanh(k * normalized) / np.tanh(k)

def old_utilization_reward(util):
    """旧版利用率奖励（分段线性）"""
    if util < 0.001:
        return -1.0
    elif util < 0.05:
        return (util - 0.05) / 0.05
    else:
        return min(-0.05 / util + 1.0, 0.0)

def new_utilization_reward(util, best=0.07,worst=0.01, k=3.0):
    """新版利用率奖励（Tanh非线性，反向）"""
    if util < 0.001:
        return -1.0
    elif util >= best:
        return 0.0
    else:
        # 反向：利用率低惩罚重
        normalized = 1.0 - (util - worst) / (best - worst)
        normalized = np.clip(normalized, 0.0, 1.0)
        return -np.tanh(k * normalized) / np.tanh(k)


def plot_comparison():
    """绘制新旧奖励函数对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 距离奖励对比
    ax1 = axes[0, 0]
    distances = np.linspace(6, 28, 200)
    old_rewards = [old_distance_reward(d) for d in distances]
    new_rewards = [new_distance_reward(d) for d in distances]
    
    ax1.plot(distances, old_rewards, 'b-', label='旧版(线性)', linewidth=2)
    ax1.plot(distances, new_rewards, 'r-', label='新版(Tanh k=2.5)', linewidth=2)
    ax1.fill_between(distances, old_rewards, new_rewards, alpha=0.3, color='green', 
                      where=np.array(new_rewards) > np.array(old_rewards), label='改善区域')
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.axvline(x=7.0, color='g', linestyle=':', alpha=0.5, label='SLP优化值')
    ax1.axvline(x=16.18, color='orange', linestyle=':', alpha=0.5, label='随机平均值')
    ax1.set_xlabel('平均运输距离')
    ax1.set_ylabel('奖励值')
    ax1.set_title('距离奖励对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 物流强度奖励对比
    ax2 = axes[0, 1]
    logistics = np.linspace(200, 1000, 200)
    old_rewards = [old_logistics_reward(l) for l in logistics]
    new_rewards = [new_logistics_reward(l) for l in logistics]
    
    ax2.plot(logistics, old_rewards, 'b-', label='旧版(线性)', linewidth=2)
    ax2.plot(logistics, new_rewards, 'r-', label='新版(Tanh k=3.0)', linewidth=2)
    ax2.fill_between(logistics, old_rewards, new_rewards, alpha=0.3, color='green',
                      where=np.array(new_rewards) > np.array(old_rewards), label='改善区域')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.axvline(x=232, color='g', linestyle=':', alpha=0.5, label='SLP优化值')
    ax2.axvline(x=512, color='orange', linestyle=':', alpha=0.5, label='随机平均值')
    ax2.set_xlabel('总物流强度')
    ax2.set_ylabel('奖励值')
    ax2.set_title('物流强度奖励对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 吞吐量奖励对比
    ax3 = axes[1, 0]
    throughputs = np.linspace(120, 420, 200)
    old_rewards = [old_throughput_reward(t) for t in throughputs]
    new_rewards = [new_throughput_reward(t) for t in throughputs]
    
    ax3.plot(throughputs, old_rewards, 'b-', label='旧版(分段线性)', linewidth=2)
    ax3.plot(throughputs, new_rewards, 'r-', label='新版(Tanh k=2.0)', linewidth=2)
    ax3.fill_between(throughputs, old_rewards, new_rewards, alpha=0.3, color='green',
                      where=np.array(new_rewards) > np.array(old_rewards), label='改善区域')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3.axvline(x=400, color='g', linestyle=':', alpha=0.5, label='目标值')
    ax3.axvline(x=325, color='orange', linestyle=':', alpha=0.5, label='随机平均值')
    ax3.set_xlabel('完成产品数量')
    ax3.set_ylabel('奖励值')
    ax3.set_title('吞吐量奖励对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. 利用率奖励对比
    ax4 = axes[1, 1]
    utils = np.linspace(0.01, 0.10, 200)
    old_rewards = [old_utilization_reward(u) for u in utils]
    new_rewards = [new_utilization_reward(u) for u in utils]
    
    ax4.plot(utils, old_rewards, 'b-', label='旧版(分段线性)', linewidth=2)
    ax4.plot(utils, new_rewards, 'r-', label='新版(Tanh k=2.0)', linewidth=2)
    ax4.fill_between(utils, old_rewards, new_rewards, alpha=0.3, color='green',
                      where=np.array(new_rewards) > np.array(old_rewards), label='改善区域')
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0.061, color='g', linestyle=':', alpha=0.5, label='SLP优化值')
    ax4.axvline(x=0.054, color='orange', linestyle=':', alpha=0.5, label='随机平均值')
    ax4.set_xlabel('平均工位利用率')
    ax4.set_ylabel('奖励值')
    ax4.set_title('利用率奖励对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tools/reward_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✓ 奖励对比图已保存到: tools/reward_comparison.png")
    plt.show()


def print_comparison_table():
    """打印数值对比表"""
    print("\n" + "="*80)
    print("距离奖励对比 (range: 6.0-28.0)")
    print("="*80)
    print(f"{'距离':>8} | {'归一化':>8} | {'旧奖励(线性)':>14} | {'新奖励(Tanh)':>14} | {'差异':>10}")
    print("-"*80)
    
    distances = [6.0, 8.2, 11.5, 17.0, 22.5, 25.8, 28.0]
    for d in distances:
        old_r = old_distance_reward(d)
        new_r = new_distance_reward(d)
        normalized = (d - 6.0) / 22.0
        diff = new_r - old_r
        print(f"{d:8.1f} | {normalized:8.3f} | {old_r:14.3f} | {new_r:14.3f} | {diff:+10.3f}")
    
    print("\n" + "="*80)
    print("物流强度奖励对比 (range: 200.0-1050.0)")
    print("="*80)
    print(f"{'物流强度':>10} | {'归一化':>8} | {'旧奖励(线性)':>14} | {'新奖励(Tanh)':>14} | {'差异':>10}")
    print("-"*80)
    
    logistics_values = [200, 300, 450, 625, 800, 950, 1050]
    for l in logistics_values:
        old_r = old_logistics_reward(l)
        new_r = new_logistics_reward(l)
        normalized = (l - 200.0) / 850.0
        diff = new_r - old_r
        print(f"{l:10.0f} | {normalized:8.3f} | {old_r:14.3f} | {new_r:14.3f} | {diff:+10.3f}")
    
    print("\n" + "="*80)
    print("吞吐量奖励对比 (range: 120-420)")
    print("="*80)
    print(f"{'完成数量':>10} | {'归一化':>8} | {'旧奖励':>14} | {'新奖励':>14} | {'差异':>10}")
    print("-"*80)
    
    throughputs = [120, 180, 240, 300, 360, 400, 420]
    for t in throughputs:
        old_r = old_throughput_reward(t)
        new_r = new_throughput_reward(t)
        normalized = (t - 120) / 300 if t >= 120 else 0
        diff = new_r - old_r
        print(f"{t:10.0f} | {normalized:8.3f} | {old_r:14.3f} | {new_r:14.3f} | {diff:+10.3f}")
    
    print("\n" + "="*80)
    print("利用率奖励对比 (range: 0.01-0.08)")
    print("="*80)
    print(f"{'利用率':>10} | {'归一化':>8} | {'旧奖励':>14} | {'新奖励':>14} | {'差异':>10}")
    print("-"*80)
    
    utils = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08]
    for u in utils:
        old_r = old_utilization_reward(u)
        new_r = new_utilization_reward(u)
        normalized = (u - 0.01) / 0.07 if u >= 0.01 else 0
        diff = new_r - old_r
        print(f"{u:10.3f} | {normalized:8.3f} | {old_r:14.3f} | {new_r:14.3f} | {diff:+10.3f}")


def analyze_improvement():
    """分析改进效果"""
    print("\n" + "="*80)
    print("改进效果分析")
    print("="*80)
    
    # 距离分析
    good_distance = 7.0  # SLP优化结果
    avg_distance = 16.18  # 随机平均
    bad_distance = 24.0   # 较差结果
    
    print("\n1. 距离奖励改进:")
    print(f"   好的布局 (d={good_distance}):")
    print(f"      旧奖励: {old_distance_reward(good_distance):.4f}")
    print(f"      新奖励: {new_distance_reward(good_distance):.4f}")
    print(f"      改善: {new_distance_reward(good_distance) - old_distance_reward(good_distance):+.4f}")
    
    print(f"   平均布局 (d={avg_distance}):")
    print(f"      旧奖励: {old_distance_reward(avg_distance):.4f}")
    print(f"      新奖励: {new_distance_reward(avg_distance):.4f}")
    print(f"      改善: {new_distance_reward(avg_distance) - old_distance_reward(avg_distance):+.4f}")
    
    print(f"   差的布局 (d={bad_distance}):")
    print(f"      旧奖励: {old_distance_reward(bad_distance):.4f}")
    print(f"      新奖励: {new_distance_reward(bad_distance):.4f}")
    print(f"      改善: {new_distance_reward(bad_distance) - old_distance_reward(bad_distance):+.4f}")
    
    # 物流强度分析
    good_logistics = 232  # SLP优化结果
    avg_logistics = 512   # 随机平均
    bad_logistics = 829   # 较差结果
    
    print("\n2. 物流强度奖励改进:")
    print(f"   好的布局 (l={good_logistics}):")
    print(f"      旧奖励: {old_logistics_reward(good_logistics):.4f}")
    print(f"      新奖励: {new_logistics_reward(good_logistics):.4f}")
    print(f"      改善: {new_logistics_reward(good_logistics) - old_logistics_reward(good_logistics):+.4f}")
    
    print(f"   平均布局 (l={avg_logistics}):")
    print(f"      旧奖励: {old_logistics_reward(avg_logistics):.4f}")
    print(f"      新奖励: {new_logistics_reward(avg_logistics):.4f}")
    print(f"      改善: {new_logistics_reward(avg_logistics) - old_logistics_reward(avg_logistics):+.4f}")
    
    print(f"   差的布局 (l={bad_logistics}):")
    print(f"      旧奖励: {old_logistics_reward(bad_logistics):.4f}")
    print(f"      新奖励: {new_logistics_reward(bad_logistics):.4f}")
    print(f"      改善: {new_logistics_reward(bad_logistics) - old_logistics_reward(bad_logistics):+.4f}")
    
    # 计算整体区别度
    print("\n3. 好坏动作区别度对比:")
    
    # 距离
    old_gap_dist = abs(old_distance_reward(good_distance) - old_distance_reward(bad_distance))
    new_gap_dist = abs(new_distance_reward(good_distance) - new_distance_reward(bad_distance))
    print(f"   距离奖励:")
    print(f"      旧版区别度: {old_gap_dist:.4f}")
    print(f"      新版区别度: {new_gap_dist:.4f}")
    print(f"      提升: {(new_gap_dist/old_gap_dist - 1)*100:.1f}%")
    
    # 物流
    old_gap_log = abs(old_logistics_reward(good_logistics) - old_logistics_reward(bad_logistics))
    new_gap_log = abs(new_logistics_reward(good_logistics) - new_logistics_reward(bad_logistics))
    print(f"   物流强度奖励:")
    print(f"      旧版区别度: {old_gap_log:.4f}")
    print(f"      新版区别度: {new_gap_log:.4f}")
    print(f"      提升: {(new_gap_log/old_gap_log - 1)*100:.1f}%")


if __name__ == "__main__":
    print("="*80)
    print("奖励函数对比可视化")
    print("="*80)
    
    # 打印对比表
    print_comparison_table()
    
    # 分析改进效果
    analyze_improvement()
    
    # 绘制对比图
    print("\n正在生成对比图...")
    plot_comparison()
    
    print("\n" + "="*80)
    print("✓ 分析完成!")
    print("="*80)

