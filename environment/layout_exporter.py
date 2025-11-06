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
    
    def _convert_coordinates(
        self, 
        env_x: int, 
        env_y: int, 
        rotation: int, 
        original_length: float, 
        original_width: float
    ) -> tuple:
        """
        将Environment坐标转换为Simulation坐标
        
        Environment: (x,y)是实际占用区域的左下角
        Simulation: (x,y)是旋转前矩形的左下角，通过angle进行顺时针旋转
        
        旋转变换（顺时针）：
        - 0°:   实际占用 [x, x+L) × [y, y+W)
        - 90°:  实际占用 [x, x+W) × [y-L, y)   → 左下角在 (x, y-L)
        - 180°: 实际占用 [x-L, x) × [y-W, y)   → 左下角在 (x-L, y-W)
        - 270°: 实际占用 [x-W, x) × [y, y+L)   → 左下角在 (x-W, y)
        
        Args:
            env_x, env_y: Environment中的坐标（实际占用的左下角）
            rotation: 旋转角度
            original_length: 原始长度（未旋转）
            original_width: 原始宽度（未旋转）
        
        Returns:
            (sim_x, sim_y): Simulation期望的坐标（旋转前矩形的左下角）
        """
        if rotation == 0:
            # 无旋转，坐标相同
            return env_x, env_y
        elif rotation == 90:
            # 旋转90°后，占用区域是 [x, x+W) × [y, y+L)
            # Simulation旋转后占用 [sx, sx+W) × [sy-L, sy)
            # 匹配条件：sx=x, sy-L=y → sy=y+L
            return env_x, env_y + original_length
        elif rotation == 180:
            # 旋转180°后，占用区域是 [x, x+L) × [y, y+W)
            # Simulation旋转后占用 [sx-L, sx) × [sy-W, sy)
            # 匹配条件：sx-L=x, sy-W=y → sx=x+L, sy=y+W
            return env_x + original_length, env_y + original_width
        elif rotation == 270:
            # 旋转270°后，占用区域是 [x, x+W) × [y, y+L)
            # Simulation旋转后占用 [sx-W, sx) × [sy, sy+L)
            # 匹配条件：sx-W=x, sy=y → sx=x+W, sy=y
            return env_x + original_width, env_y
        else:
            # 非标准角度，保持原样
            return env_x, env_y
        
    def export_layout(
        self, 
        placed_units: List[Tuple],
        functional_units: List[Dict]
    ) -> None:
        """
        导出布局到 JSON 文件
        
        Args:
            placed_units: 已放置的功能单元列表 [(unit_id, x, y, rotation), ...]
                         其中 unit_id 是在 functional_units 中的索引
            functional_units: 功能单元配置列表（从 config_loader 获取）
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
                'transporters': [],
                'product_flows': []
            }
        
        # 更新功能单元的位置和角度
        fus_list = new_layout.get('fus', [])
        
        # 创建 id 到 fus 索引的映射
        id_to_fus_idx = {}
        for idx, fu in enumerate(fus_list):
            id_to_fus_idx[fu['id']] = idx
        
        # 更新每个已放置单元的位置
        for unit_idx, x, y, rotation in placed_units:
            unit = functional_units[unit_idx]
            unit_id = unit['id']
            
            # 如果该单元在 fus 列表中，更新其位置
            if unit_id in id_to_fus_idx:
                fus_idx = id_to_fus_idx[unit_id]
                
                # 获取原始尺寸（未旋转）
                original_length = fus_list[fus_idx]['length']
                original_width = fus_list[fus_idx]['width']
                
                # 坐标转换：Environment -> Simulation
                # Environment: (x,y)是实际占用区域的左下角，rotation已应用到尺寸
                # Simulation: (x,y)是旋转前矩形的左下角，通过angle变换旋转
                #
                # 需要调整(x,y)，使得Simulation旋转后得到Environment的占用区域
                sim_x, sim_y = self._convert_coordinates(
                    x, y, rotation, original_length, original_width
                )
                
                fus_list[fus_idx]['x'] = int(sim_x)
                fus_list[fus_idx]['y'] = int(sim_y)
                fus_list[fus_idx]['angle'] = int(rotation)
                
                # 保持原始的length和width不变（Simulation会根据angle旋转）
        
        new_layout['fus'] = fus_list
        
        # 确保输出目录存在
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入 JSON 文件（使用 utf-8 不带 BOM）
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(new_layout, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 布局已导出到: {self.output_path}")
    
    def export_layout_dict(
        self, 
        placed_units: List[Tuple],
        functional_units: List[Dict]
    ) -> Dict:
        """
        生成布局配置字典但不写入文件（用于测试或临时使用）
        
        Args:
            placed_units: 已放置的功能单元列表
            functional_units: 功能单元配置列表
            
        Returns:
            布局配置字典
        """
        # 创建临时文件导出，然后读取返回
        new_layout = copy.deepcopy(self.layout_template)
        
        if not new_layout or 'fus' not in new_layout:
            return {}
        
        fus_list = new_layout['fus']
        id_to_fus_idx = {fu['id']: idx for idx, fu in enumerate(fus_list)}
        
        for unit_idx, x, y, rotation in placed_units:
            unit = functional_units[unit_idx]
            unit_id = unit['id']
            
            if unit_id in id_to_fus_idx:
                fus_idx = id_to_fus_idx[unit_id]
                
                # 坐标转换
                original_length = fus_list[fus_idx]['length']
                original_width = fus_list[fus_idx]['width']
                sim_x, sim_y = self._convert_coordinates(
                    x, y, rotation, original_length, original_width
                )
                
                fus_list[fus_idx]['x'] = int(sim_x)
                fus_list[fus_idx]['y'] = int(sim_y)
                fus_list[fus_idx]['angle'] = int(rotation)
        
        new_layout['fus'] = fus_list
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

