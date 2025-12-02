"""
布局输出器 - 将 Environment 的布局转换为 Simulation 需要的格式
负责人：张毅

将 RL environment 的布局结果输出到 chair_layout.json，供仿真系统使用
"""

import json
import copy
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np


class LayoutExporter:
    """
    将环境的布局状态转换为仿真系统所需的 JSON 格式
    """
    
    def __init__(self, layout_template: Dict, output_path: str):
        """
        初始化布局输出器
        
        Args:
            layout_template: 原始布局配置模板（从 config_loader 获取）
            output_path: 输出文件路径 (如 simulation/layouts/chair_layout.json)
        """
        self.layout_template = layout_template
        self.output_path = Path(output_path)
    
    def export_layout(
        self, 
        placed_units: List[Tuple],
        functional_units: List[Dict]
    ) -> None:
        """
        导出布局到 JSON 文件（同时更新 fus 和 obstacles）
        
        Args:
            placed_units: 已放置的功能单元列表 [(unit_id, x, y, rotation), ...]
                         其中 unit_id 是在 functional_units 中的索引
            functional_units: 功能单元配置列表（从 config_loader 获取，包含 fus 和可移动 obstacles）
        """
        # 创建新的布局配置（基于模板）
        new_layout = copy.deepcopy(self.layout_template)
        
        # 如果模板为空，创建基础结构
        if not new_layout:
            new_layout = {
                'factory': {
                    'length': 90,
                    'width': 60,
                    'grid_spacing': 1
                },
                'fus': [],
                'obstacles': [],
                'transporters': [],
                'product_flows': []
            }
        
        # 获取 fus 和 obstacles 列表
        fus_list = new_layout.get('fus', [])
        obstacles_list = new_layout.get('obstacles', [])
        
        # 创建 id 到索引的映射（分别对 fus 和 obstacles）
        id_to_fus_idx = {fu['id']: idx for idx, fu in enumerate(fus_list)}
        id_to_obs_idx = {obs['id']: idx for idx, obs in enumerate(obstacles_list)}
        
        # 更新每个已放置单元的位置
        for unit_idx, x, y, rotation in placed_units:
            unit = functional_units[unit_idx]
            unit_id = unit['id']
            is_obstacle = unit.get('is_obstacle', False)
            
            if is_obstacle:
                # 更新 obstacles 列表中的位置
                if unit_id in id_to_obs_idx:
                    obs_idx = id_to_obs_idx[unit_id]
                    obstacles_list[obs_idx]['x'] = int(x)
                    obstacles_list[obs_idx]['y'] = int(y)
                    obstacles_list[obs_idx]['angle'] = int(rotation)
            else:
                # 更新 fus 列表中的位置
            if unit_id in id_to_fus_idx:
                fus_idx = id_to_fus_idx[unit_id]
                # 坐标系统已统一：Environment和Simulation都使用旋转前矩形的左下角
                # 直接使用Environment的坐标，无需转换
                fus_list[fus_idx]['x'] = int(x)
                fus_list[fus_idx]['y'] = int(y)
                fus_list[fus_idx]['angle'] = int(rotation)
                # length和width保持不变（Simulation会根据angle旋转）
        
        new_layout['fus'] = fus_list
        new_layout['obstacles'] = obstacles_list
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入 JSON 文件（使用 utf-8 不带 BOM）
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(new_layout, f, indent=2, ensure_ascii=False)
        
        # 性能优化：减少打印输出（训练时会产生大量输出）
        # print(f"✓ 布局已导出到: {self.output_path}")  # 已注释，减少I/O开销
    
    def export_layout_dict(
        self, 
        placed_units: List[Tuple],
        functional_units: List[Dict]
    ) -> Dict:
        """
        生成布局配置字典但不写入文件（用于测试或临时使用）
        
        Args:
            placed_units: 已放置的功能单元列表
            functional_units: 功能单元配置列表（包含 fus 和可移动 obstacles）
            
        Returns:
            布局配置字典
        """
        new_layout = copy.deepcopy(self.layout_template)
        
        if not new_layout:
            return {}
        
        # 获取 fus 和 obstacles 列表
        fus_list = new_layout.get('fus', [])
        obstacles_list = new_layout.get('obstacles', [])
        
        # 创建 id 到索引的映射
        id_to_fus_idx = {fu['id']: idx for idx, fu in enumerate(fus_list)}
        id_to_obs_idx = {obs['id']: idx for idx, obs in enumerate(obstacles_list)}
        
        for unit_idx, x, y, rotation in placed_units:
            unit = functional_units[unit_idx]
            unit_id = unit['id']
            is_obstacle = unit.get('is_obstacle', False)
            
            if is_obstacle:
                # 更新 obstacles 列表中的位置
                if unit_id in id_to_obs_idx:
                    obs_idx = id_to_obs_idx[unit_id]
                    obstacles_list[obs_idx]['x'] = int(x)
                    obstacles_list[obs_idx]['y'] = int(y)
                    obstacles_list[obs_idx]['angle'] = int(rotation)
            else:
                # 更新 fus 列表中的位置
            if unit_id in id_to_fus_idx:
                fus_idx = id_to_fus_idx[unit_id]
                # 坐标系统已统一，直接使用Environment的坐标
                fus_list[fus_idx]['x'] = int(x)
                fus_list[fus_idx]['y'] = int(y)
                fus_list[fus_idx]['angle'] = int(rotation)
        
        new_layout['fus'] = fus_list
        new_layout['obstacles'] = obstacles_list
        return new_layout


def export_environment_layout(
    placed_units: List[Tuple],
    functional_units: List[Dict],
    layout_template: Dict,
    output_path: str
) -> None:
    """
    便捷函数：直接导出环境布局
    
    Args:
        placed_units: 已放置的功能单元列表
        functional_units: 功能单元配置列表
        layout_template: 布局模板
        output_path: 输出文件路径
    """
    exporter = LayoutExporter(layout_template, output_path)
    exporter.export_layout(placed_units, functional_units)


# ====================
# 测试代码
# ====================
if __name__ == "__main__":
    print("测试布局导出器...")
    
    try:
        # 加载配置
        from config_loader import ConfigLoader
        
        config_path = "FactoGenie/simulation/configs/chair_factory.json"
        loader = ConfigLoader(config_path)
        
        functional_units = loader.get_functional_units()
        layout_template = loader.get_layout_template()
        
        # 模拟一些放置结果（使用原始位置作为测试）
        placed_units = []
        for idx, unit in enumerate(functional_units):
            # 从模板中读取原始位置
            if layout_template and 'fus' in layout_template:
                original_fu = layout_template['fus'][idx]
                x = original_fu.get('x', 0)
                y = original_fu.get('y', 0)
                angle = original_fu.get('angle', 0)
                placed_units.append((idx, x, y, angle))
        
        # 导出到临时文件
        output_path = "FactoGenie/simulation/layouts/test_output.json"
        exporter = LayoutExporter(layout_template, output_path)
        exporter.export_layout(placed_units, functional_units)
        
        # 验证输出
        with open(output_path, 'r', encoding='utf-8-sig') as f:
            exported = json.load(f)
        
        print("\n✓ 布局导出成功！")
        print(f"导出的功能单元数量: {len(exported.get('fus', []))}")
        
        # 清理测试文件
        Path(output_path).unlink()
        print("\n✓ 测试完成！")
        
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()

