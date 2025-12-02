"""
可视化和记录每个episode的布局
支持：功能单元(fus)、障碍物(obstacles)、缺角(notch)
"""

import json
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from datetime import datetime

def _rotate_point(px, py, angle_deg, ox=0, oy=0):
    """顺时针旋转点 (px, py) 绕 (ox, oy)"""
    rad = math.radians(angle_deg)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    dx, dy = px - ox, py - oy
    rx = cos_a * dx + sin_a * dy
    ry = -sin_a * dx + cos_a * dy
    return ox + rx, oy + ry

def _get_polygon_vertices(x, y, length, width, angle, notch_length=0, notch_width=0):
    """
    计算带缺角的矩形多边形顶点
    
    缺角位置：右上角（旋转前）
    坐标系：(x, y) 是旋转前矩形的左下角
    """
    notch_length = notch_length or 0
    notch_width = notch_width or 0
    
    # 构建基础多边形（局部坐标，左下角在原点）
    if notch_length > 0 and notch_width > 0:
        # L形多边形（右上角有缺口）
        base = [
            (0, 0),                              # 左下角
            (length - notch_length, 0),          # 缺口左下
            (length - notch_length, notch_width), # 缺口左上
            (length, notch_width),               # 缺口右下
            (length, width),                     # 右上角
            (0, width),                          # 左上角
        ]
    else:
        # 普通矩形
        base = [
            (0, 0),
            (length, 0),
            (length, width),
            (0, width),
        ]
    
    # 旋转并平移到世界坐标
    vertices = []
    for px, py in base:
        rx, ry = _rotate_point(px, py, angle, 0, 0)
        vertices.append((x + rx, y + ry))
    
    return vertices

class LayoutVisualizer:
    """布局可视化工具"""
    
    def __init__(self, output_dir="logs/layouts"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_layout(self, layout_path, episode_num=None, save=True, show=False):
        """
        可视化布局JSON文件
        
        Args:
            layout_path: 布局JSON文件路径
            episode_num: Episode编号（用于文件命名）
            save: 是否保存图片
            show: 是否显示图片
        """
        # 读取布局数据
        with open(layout_path, 'r', encoding='utf-8-sig') as f:
            layout_data = json.load(f)
        
        factory = layout_data.get('factory', {})
        factory_length = factory.get('length', 100)
        factory_width = factory.get('width', 100)
        fus = layout_data.get('fus', [])
        obstacles = layout_data.get('obstacles', [])
        
        # 创建图形
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        # 绘制工厂边界
        factory_rect = patches.Rectangle(
            (0, 0), factory_length, factory_width,
            linewidth=2, edgecolor='black', facecolor='white', zorder=0
        )
        ax.add_patch(factory_rect)
        
        # 颜色列表（为 fus 分配颜色）
        colors = plt.cm.Set3(np.linspace(0, 1, max(len(fus), 1)))
        
        # 绘制每个功能单元（支持缺角）
        for idx, fu in enumerate(fus):
            fu_id = fu.get('id', f'FU-{idx}')
            x = fu.get('x', 0)
            y = fu.get('y', 0)
            length = fu.get('length', 0)
            width = fu.get('width', 0)
            angle = fu.get('angle', 0)
            notch_length = fu.get('notch_length', 0)
            notch_width = fu.get('notch_width', 0)
            
            # 计算多边形顶点（支持缺角）
            vertices = _get_polygon_vertices(x, y, length, width, angle, notch_length, notch_width)
            
            # 绘制多边形
            poly = patches.Polygon(
                vertices,
                closed=True,
                edgecolor='black',
                facecolor=colors[idx],
                alpha=0.7,
                linewidth=1.5,
                zorder=2
            )
            ax.add_patch(poly)
            
            # 计算中心点（用于放置标签）
            center_x = np.mean([v[0] for v in vertices])
            center_y = np.mean([v[1] for v in vertices])
            
            # 添加标签
            ax.text(
                center_x, center_y, fu_id,
                ha='center', va='center',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                zorder=3
            )
            
            # 添加角度标注（如果有旋转）
            if angle != 0:
                ax.text(
                    center_x, center_y - 1.5, f'{angle}°',
                    ha='center', va='top',
                    fontsize=7, color='red',
                    zorder=3
                )
            
            # 标记参考点(x,y)
            ax.plot(x, y, 'ro', markersize=4, zorder=4)
        
        # 绘制障碍物
        for obs in obstacles:
            obs_id = obs.get('id', '')
            x = obs.get('x', 0)
            y = obs.get('y', 0)
            length = obs.get('length', 0)
            width = obs.get('width', 0)
            angle = obs.get('angle', 0)
            notch_length = obs.get('notch_length', 0)
            notch_width = obs.get('notch_width', 0)
            
            # 计算多边形顶点
            vertices = _get_polygon_vertices(x, y, length, width, angle, notch_length, notch_width)
            
            # 绘制障碍物（灰色，带斜线填充）
            poly = patches.Polygon(
                vertices,
                closed=True,
                edgecolor='black',
                facecolor='#9a9a9a',
                alpha=0.6,
                linewidth=1.5,
                hatch='///',
                zorder=2
            )
            ax.add_patch(poly)
            
            # 添加标签
            if obs_id:
                center_x = np.mean([v[0] for v in vertices])
                center_y = np.mean([v[1] for v in vertices])
                ax.text(
                    center_x, center_y, obs_id,
                    ha='center', va='center',
                    fontsize=8, fontweight='bold',
                    color='black',
                    zorder=3
                )
        
        # 设置坐标轴
        ax.set_xlim(-5, factory_length + 5)
        ax.set_ylim(-5, factory_width + 5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        
        # 标题
        title = f'Factory Layout'
        if episode_num is not None:
            title += f' - Episode {episode_num}'
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # 添加图例
        legend_text = f'Factory: {factory_length}m × {factory_width}m\n'
        legend_text += f'FUs: {len(fus)}, Obstacles: {len(obstacles)}'
        ax.text(
            0.02, 0.98, legend_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # 保存图片
        if save:
            if episode_num is not None:
                filename = f'layout_episode_{episode_num:05d}.png'
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'layout_{timestamp}.png'
            
            output_path = self.output_dir / filename
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ 布局图已保存到: {output_path}")
        
        # 显示图片
        if show:
            plt.show()
        else:
            plt.close()
    
    def save_layout_json(self, layout_path, episode_num, output_subdir="json"):
        """
        保存布局JSON的副本
        
        Args:
            layout_path: 原始布局JSON路径
            episode_num: Episode编号
            output_subdir: 输出子目录
        """
        output_dir = self.output_dir / output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(layout_path, 'r', encoding='utf-8-sig') as f:
            layout_data = json.load(f)
        
        output_path = output_dir / f'layout_episode_{episode_num:05d}.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(layout_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 布局JSON已保存到: {output_path}")
        
        return output_path

def visualize_current_layout(layout_path="simulation/layouts/chair_layout.json"):
    """可视化当前的布局文件"""
    
    print("="*80)
    print("可视化布局")
    print("="*80)
    
    visualizer = LayoutVisualizer()
    
    if Path(layout_path).exists():
        print(f"\n读取布局: {layout_path}")
        visualizer.visualize_layout(layout_path, save=True, show=True)
    else:
        print(f"\n❌ 布局文件不存在: {layout_path}")
        print("请先运行一次训练生成布局文件")

def visualize_all_layouts(layout_dir="logs/layouts/json"):
    """可视化所有保存的布局"""
    
    layout_dir = Path(layout_dir)
    if not layout_dir.exists():
        print(f"目录不存在: {layout_dir}")
        return
    
    json_files = sorted(layout_dir.glob("layout_episode_*.json"))
    
    if not json_files:
        print(f"在 {layout_dir} 中没有找到布局文件")
        return
    
    print(f"\n找到 {len(json_files)} 个布局文件")
    visualizer = LayoutVisualizer()
    
    for json_file in json_files:
        # 从文件名提取episode编号
        try:
            episode_num = int(json_file.stem.split('_')[-1])
        except:
            episode_num = None
        
        print(f"\n处理: {json_file.name}")
        visualizer.visualize_layout(json_file, episode_num=episode_num, save=True, show=False)
    
    print(f"\n✓ 所有布局已可视化")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='可视化工厂布局')
    parser.add_argument('--layout', type=str, default="simulation/layouts/chair_layout.json",
                      help='布局JSON文件路径')
    parser.add_argument('--all', action='store_true',
                      help='可视化所有保存的布局')
    parser.add_argument('--show', action='store_true',
                      help='显示图片窗口')
    
    args = parser.parse_args()
    
    if args.all:
        visualize_all_layouts()
    else:
        visualizer = LayoutVisualizer()
        if Path(args.layout).exists():
            visualizer.visualize_layout(args.layout, save=True, show=args.show)
        else:
            print(f"❌ 文件不存在: {args.layout}")







