"""
工厂实际数据配置模块
用于存储真实工厂的布局参数、单元配置、物流信息等

使用方法:
1. 填写本文件中的配置参数
2. 在创建环境时导入: from factory_data_config import FACTORY_CONFIG
3. 传入环境: env = LayoutEnvironment(**FACTORY_CONFIG)
"""

import numpy as np
from typing import List, Dict, Tuple


# ==================== 厂房配置 ====================

# 厂房网格尺寸 (单位: 米或网格单元)
GRID_SIZE = (15, 15)  # (长度nx, 宽度ny)

# 每个网格单元的实际尺寸 (单位: 米)
CELL_SIZE = 1.0  # 例如: 每个网格单元代表1米×1米


# ==================== 功能单元配置 ====================

FUNCTIONAL_UNITS = [
    {
        'id': 0,
        'name': '接收区',  # 功能单元名称
        'size': (3, 2),    # (宽度, 高度) 单位: 网格数
        'actual_size': (3.0, 2.0),  # 实际尺寸 (米)
        'rotatable': False,  # 是否可旋转
        'must_on_wall': True,  # 是否必须贴墙
        'wall_side': 'left',  # 贴哪面墙: 'left', 'right', 'top', 'bottom'
        'description': '原材料接收区域'
    },
    {
        'id': 1,
        'name': '切割工位',
        'size': (2, 2),
        'actual_size': (2.0, 2.0),
        'rotatable': True,
        'must_on_wall': False,
        'description': '木材切割加工区'
    },
    {
        'id': 2,
        'name': '打磨工位',
        'size': (2, 2),
        'actual_size': (2.0, 2.0),
        'rotatable': True,
        'must_on_wall': False,
        'description': '零件打磨区'
    },
    {
        'id': 3,
        'name': '组装工位',
        'size': (3, 2),
        'actual_size': (3.0, 2.0),
        'rotatable': True,
        'must_on_wall': False,
        'description': '椅子组装区'
    },
    {
        'id': 4,
        'name': '发货区',
        'size': (3, 2),
        'actual_size': (3.0, 2.0),
        'rotatable': False,
        'must_on_wall': True,
        'wall_side': 'right',
        'description': '成品发货区域'
    },
]


# ==================== 物料流配置 ====================

# 物料流矩阵 [N×N]
# MATERIAL_FLOW[i][j] 表示从单元i到单元j的物料流量
# 可以是：
#   - 单位时间内的运输次数
#   - 单位时间内的物料重量（kg）
#   - 单位时间内的物料体积（m³）

MATERIAL_FLOW = np.array([
    #   接收  切割  打磨  组装  发货
    [   0,   10,    0,    0,    0  ],  # 从接收区
    [   0,    0,    8,    0,    0  ],  # 从切割工位
    [   0,    0,    0,    6,    0  ],  # 从打磨工位
    [   0,    0,    0,    0,    5  ],  # 从组装工位
    [   0,    0,    0,    0,    0  ],  # 从发货区
], dtype=np.float32)

# 物料流描述 (可选)
MATERIAL_FLOW_DESCRIPTION = {
    (0, 1): "原木 -> 切割 (10次/小时)",
    (1, 2): "切割件 -> 打磨 (8次/小时)",
    (2, 3): "打磨件 -> 组装 (6次/小时)",
    (3, 4): "成品 -> 发货 (5次/小时)",
}


# ==================== 生产工艺流程 ====================

# 工艺流程顺序（用于SimPy仿真）
PROCESS_SEQUENCE = [
    {'unit_id': 0, 'name': '接收', 'processing_time': 5},   # 单位: 分钟
    {'unit_id': 1, 'name': '切割', 'processing_time': 15},
    {'unit_id': 2, 'name': '打磨', 'processing_time': 20},
    {'unit_id': 3, 'name': '组装', 'processing_time': 25},
    {'unit_id': 4, 'name': '发货', 'processing_time': 10},
]

# 各工位处理能力（单位: 件/小时）
PROCESSING_CAPACITY = {
    0: 12,  # 接收区
    1: 8,   # 切割工位
    2: 6,   # 打磨工位
    3: 5,   # 组装工位
    4: 10,  # 发货区
}


# ==================== 布局约束条件 ====================

# 限制区域（障碍物、柱子、通道等）
# 格式: [(x, y, width, height), ...]
RESTRICTED_AREAS = [
    # (5, 5, 1, 1),  # 示例: (5,5)位置有一个柱子
    # (0, 7, 15, 1), # 示例: 横向通道
]

# 最小间距要求（单位: 网格单元）
MIN_DISTANCE_BETWEEN_UNITS = 1  # 单元之间至少间隔1个网格

# 特定单元之间的最小/最大距离约束
DISTANCE_CONSTRAINTS = [
    # {
    #     'unit_a': 1,  # 切割工位
    #     'unit_b': 2,  # 打磨工位
    #     'min_distance': 1,  # 最小距离
    #     'max_distance': 5,  # 最大距离（用于保证物流效率）
    # },
]

# 相邻性要求（哪些单元必须相邻）
ADJACENCY_REQUIREMENTS = [
    # (1, 2),  # 切割和打磨必须相邻
    # (2, 3),  # 打磨和组装必须相邻
]


# ==================== 优化目标权重 ====================

OBJECTIVE_WEIGHTS = {
    'transportation_intensity': 0.4,   # 运输强度（距离×流量）
    'material_flow_clarity': 0.2,     # 物料流清晰度
    'throughput_time': 0.3,           # 生产周期（需要仿真）
    'utilization': 0.1,               # 设备利用率（需要仿真）
}


# ==================== 仿真参数 ====================

SIMULATION_CONFIG = {
    'use_simulation': False,  # 是否启用SimPy仿真
    'simulation_time': 480,   # 仿真时长（单位: 分钟，480=8小时）
    'warmup_time': 60,        # 预热时间（单位: 分钟）
    'num_replications': 3,    # 仿真重复次数
}


# ==================== 完整配置字典 ====================

FACTORY_CONFIG = {
    'grid_size': GRID_SIZE,
    'functional_units': FUNCTIONAL_UNITS,
    'material_flow': MATERIAL_FLOW,
    'objective_weights': OBJECTIVE_WEIGHTS,
    'use_simulation': SIMULATION_CONFIG['use_simulation'],
}


# ==================== 辅助函数 ====================

def validate_config() -> bool:
    """
    验证配置的有效性
    
    Returns:
        配置是否有效
    """
    errors = []
    
    # 检查功能单元数量与物料流矩阵维度是否匹配
    num_units = len(FUNCTIONAL_UNITS)
    if MATERIAL_FLOW.shape != (num_units, num_units):
        errors.append(f"物料流矩阵维度 {MATERIAL_FLOW.shape} 与功能单元数量 {num_units} 不匹配")
    
    # 检查单元ID是否连续
    unit_ids = [u['id'] for u in FUNCTIONAL_UNITS]
    if unit_ids != list(range(num_units)):
        errors.append(f"功能单元ID必须从0开始连续: {unit_ids}")
    
    # 检查单元尺寸是否超出网格
    nx, ny = GRID_SIZE
    for unit in FUNCTIONAL_UNITS:
        w, h = unit['size']
        if w > nx or h > ny:
            errors.append(f"单元 {unit['name']} 尺寸 ({w},{h}) 超出网格 ({nx},{ny})")
    
    # 检查权重和是否为1
    weight_sum = sum(OBJECTIVE_WEIGHTS.values())
    if abs(weight_sum - 1.0) > 0.01:
        errors.append(f"目标权重之和应为1.0，当前为 {weight_sum}")
    
    # 打印错误
    if errors:
        print("❌ 配置验证失败:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("✅ 配置验证通过!")
        return True


def print_config_summary():
    """打印配置摘要"""
    print("=" * 60)
    print("📋 工厂配置摘要")
    print("=" * 60)
    print(f"厂房网格: {GRID_SIZE[0]} × {GRID_SIZE[1]}")
    print(f"功能单元数量: {len(FUNCTIONAL_UNITS)}")
    print(f"\n功能单元列表:")
    for unit in FUNCTIONAL_UNITS:
        print(f"  [{unit['id']}] {unit['name']:<12} - 尺寸: {unit['size']}")
    
    print(f"\n物料流总量: {MATERIAL_FLOW.sum():.1f}")
    print(f"主要物流路径:")
    for (i, j), desc in MATERIAL_FLOW_DESCRIPTION.items():
        print(f"  {desc}")
    
    print(f"\n优化目标权重:")
    for obj, weight in OBJECTIVE_WEIGHTS.items():
        print(f"  {obj:<28}: {weight:.2f}")
    
    print(f"\n仿真: {'启用' if SIMULATION_CONFIG['use_simulation'] else '禁用'}")
    print("=" * 60)


# ==================== 示例：如何使用配置 ====================

if __name__ == "__main__":
    # 1. 验证配置
    validate_config()
    
    # 2. 打印配置摘要
    print_config_summary()
    
    # 3. 示例: 创建环境
    print("\n📌 使用示例:")
    print("```python")
    print("from factory_data_config import FACTORY_CONFIG")
    print("from factory_environment import LayoutEnvironment")
    print()
    print("# 使用实际工厂数据创建环境")
    print("env = LayoutEnvironment(**FACTORY_CONFIG)")
    print("```")




