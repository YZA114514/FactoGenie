# Multi-Material Demo（生产物流仿真示例）

本仓库包含一个基于 **SimPy** 的多物料装配与运输仿真引擎，以及一系列布局规划、可视化、指标评估工具。适用于课程设计、方案评估或与外部优化/强化学习模块的集成实验。

## 准备工作

- Python ≥ 3.9 建议使用与项目相同的 Conda 环境。
- 主要依赖：`simpy`、`matplotlib`、`numpy`（如需 Pandas/Seaborn 可自行添加）。
- 进入仓库根目录后执行环境安装：
  ```bash
  pip install -r requirements.txt   # 若尚未维护，可自行 pip install simpy matplotlib numpy
  ```

## 快速体验

### 运行仿真

```powershell
C:/Anaconda/envs/ml-new/python.exe -m multi_material_demo.run_simulation `
    --config multi_material_demo/configs/simple_four_station.json `
    --duration 1200
```

要换成椅子工厂案例：
```powershell
C:/Anaconda/envs/ml-new/python.exe -m multi_material_demo.run_simulation `
    --config multi_material_demo/configs/chair_factory.json `
    --duration 120000
```

脚本会自动解析配置中的 `layout` 路径，先执行几何合法性检查（防止工位重叠或越界），再预先计算运输最短路与行驶时间，最后输出成品节点的库存快照。

### 运输路径规划 / 可视化

```powershell
# 计算所有路线最短路并可选写回配置
python -m multi_material_demo.path_planner --config multi_material_demo/configs/simple_four_station.json --write

# 查看运输车辆路径与时间轴
python -m multi_material_demo.transporter_viz --config multi_material_demo/configs/simple_four_station.json --duration 200
```

### 库存与动画

- `python -m multi_material_demo.visualize_inventory --config ... --duration 200`
- `python -m multi_material_demo.animation --config ... --duration 200`

上述脚本会绘制各节点库存曲线或在布局上实时展示库存变化。

### BOM 校验

```powershell
python -m multi_material_demo.bom --config multi_material_demo/configs/chair_factory.json --duration 120000
```

输出装配、消耗与运输的物料平衡报表，用于对账或方案比较。

## 指标评估（metrics）

`multi_material_demo.metrics` 聚合了静态与动态指标，默认输出简明摘要：

```powershell
python -m multi_material_demo.metrics --config multi_material_demo/configs/chair_factory.json --duration 120000
```

若需结构化数据供外部系统（如强化学习模块）读取，使用 `--json`，并可通过 `--detail` 获取路线级明细：

```powershell
python -m multi_material_demo.metrics `
    --config multi_material_demo/configs/chair_factory.json `
    --duration 120000 `
    --json --detail
```

指标覆盖：
- **静态**：总/平均/最大/最小运输距离、物流强度（距离×批量）、空间利用率。
- **动态**：产出与吞吐率、工位利用率、运输工具利用率与载荷情况。

## Python 接口（外部模块可直接调用）

为了便于程序化集成，`multi_material_demo.interface` 提供了高层函数：

```python
from multi_material_demo.interface import run_simulation, compute_metrics

# 运行仿真
result = run_simulation("multi_material_demo/configs/chair_factory.json", duration=120000)
print(result.finished_goods, result.summary_node)

# 获取指标摘要（detail=True 可返回完整明细）
metrics = compute_metrics("multi_material_demo/configs/chair_factory.json", duration=120000, detail=True)
summary = metrics["summary"]
transporters = summary["dynamic"]["transporter_utilization"]
```

`run_simulation` 返回包含成品数量、节点信息以及可选事件列表的 `SimulationResult`；`compute_metrics` 直接复用 `metrics.py` 中的统计逻辑，适合与策略优化、强化学习或看板系统对接。

## 目录结构速览

| 路径 | 说明 |
| ---- | ---- |
| `multi_material_demo/engine.py` | SimPy 仿真核心：多物料装配、库存、运输 |
| `multi_material_demo/planning.py` | 布局读取、最短路规划、路线补全 |
| `multi_material_demo/path_planner.py` | CLI 最短路计算与可视化 |
| `multi_material_demo/run_simulation.py` | 仿真命令行入口 |
| `multi_material_demo/metrics.py` | 静态/动态指标采集与摘要 |
| `multi_material_demo/transporter_viz.py` | 运输路径 + 时间轴可视化 |
| `multi_material_demo/bom.py` | BOM 平衡分析 |
| `multi_material_demo/interface.py` | 对外 API，供其他模块直接调用 |
| `multi_material_demo/layouts/` | 布局 JSON 示例 |
| `multi_material_demo/configs/` | 仿真配置示例 |

## 后续扩展建议

- 丰富指标库：等待时间、在制品峰值、拥堵热力图等。
- 补充自动化测试或数据导出功能（CSV/Parquet）。
- 与策略/强化学习模块结合时，可直接调用 `interface.py`，无需关心内部细节。

如有问题或需要增强的功能，可在 README 后续章节补充或提 Issue。祝使用顺利！
