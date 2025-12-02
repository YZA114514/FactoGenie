# Multi-Material Demo（生产物流仿真示例）

基于 **SimPy** 的多物料装配与运输仿真工具链，覆盖布局校验、路径规划、仿真运行、指标与可视化，支持布局内显式障碍物。

## 组件与调用链
- 高层入口：`run_simulation.py` / `interface.py`  
  1) 读取 `config` → 2) `planning.compute_route_plans`：解析/校验布局（`layout_validation`，含 obstacles），计算最短路并补充 `travel_time`/`path_points` → 3) `model.build_model` 组装 `AssemblySim`（`engine.py`）→ 4) 运行并生成事件/库存 → 5) 可交由 `metrics.py` 汇总指标。  
- 路径与布局：`planning.py` 依赖 `path_planner.py`（网格 BFS，避开 FU 与 obstacles）、`geometry_utils.py`、`visual_utils.py`（统一绘制），校验由 `layout_validation.py` 执行。  
- 可视化：`path_planner --visualize`（路线）、`transporter_viz.py`（路线+时间轴）、`animation.py`（库存动画）、`visualize_inventory.py`（库存曲线）、`layout.py`（纯布局）；均复用 `visual_utils.draw_layout`，在绘制前会做布局合法性检查。  
- 数据与报表：`metrics.py`（静态/动态指标），`bom.py`（物料平衡）。

## 配置与布局
- Config（示例见 `configs/`）  
  - `routes`: from/to/material/batch_size/travel_time/transporter_id（travel_time 会被规划覆盖）  
  - `assemblies`: station/inputs/output/output_quantity/process_time  
  - `transporters`: id/count/speed  
  - `initial_inventory`、`summary`、`layout`（指向布局文件）  
- Layout（示例见 `layouts/`）  
  - `factory`: length/width/grid_spacing  
  - `fus`: id/length/width/notch_length/notch_width/x/y/angle  
  - `obstacles`: 同上字段，始终视为障碍；校验和寻路都会考虑

## 常用命令
- 安装依赖：`pip install -r requirements.txt`
- 运行仿真  
  ```powershell
  python -m multi_material_demo.run_simulation `
    --config multi_material_demo/configs/chair_factory.json `
    --duration 120000
  ```
- 计算/可视化最短路（并写回 config）  
  ```powershell
  python -m multi_material_demo.path_planner `
    --config multi_material_demo/configs/chair_factory.json `
    --layout multi_material_demo/layouts/layout_episode_00002.json `
    --visualize --write
  ```
- 运输可视化（路线+时间轴）  
  ```powershell
  python -m multi_material_demo.transporter_viz `
    --config multi_material_demo/configs/chair_factory.json `
    --layout multi_material_demo/layouts/layout_episode_00002.json `
    --duration 1200
  ```
- 库存可视化  
  - 动画：`python -m multi_material_demo.animation --config ... --layout ... --duration 200`
  - 曲线：`python -m multi_material_demo.visualize_inventory --config ... --layout ... --duration 200`
- 指标/报表  
  - 摘要：`python -m multi_material_demo.metrics --config ... --layout ... --duration 120000`
  - JSON 全量：加 `--json --detail`  
  - BOM：`python -m multi_material_demo.bom --config ... --layout ... --duration 120000`

## Python 接口示例
```python
from multi_material_demo.interface import run_simulation, compute_metrics

res = run_simulation("multi_material_demo/configs/chair_factory.json", duration=120000)
print(res.finished_goods, res.summary_node, res.summary_material)

metrics = compute_metrics("multi_material_demo/configs/chair_factory.json", duration=120000, detail=True)
print(metrics["summary"]["dynamic"]["throughput_rate"])
```

## 目录速览
- `engine.py`：SimPy 仿真核心（库存/装配/运输/车辆重定位）  
- `model.py`：从 config 构建 `AssemblySim`  
- `planning.py`：布局加载、校验、最短路补全  
- `path_planner.py`：CLI 最短路计算与可视化  
- `metrics.py`：静态/动态指标  
- `transporter_viz.py`：运输路径 + 时间轴  
- `animation.py` / `visualize_inventory.py`：库存动画 / 曲线  
- `bom.py`：BOM 平衡  
- `visual_utils.py`：布局/障碍绘制基座  
- `layouts/`、`configs/`：示例布局与配置
