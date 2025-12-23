# FactoGenie 前端待完成功能清单

## 概述

本文档列出前端需要补齐的功能，供前端开发人员参照实现。后端 API 已准备就绪。

---

## 🔴 高优先级（核心功能）

### 1. 实时训练曲线

**功能描述**：使用图表库（echarts/recharts）绘制训练过程中的奖励、损失曲线

**后端 API**：
```
GET /api/results/{project_id}/metrics?metric=reward&start=0&end=1000
```

**数据格式**：
```typescript
{
  code: 0,
  data: {
    metric: "reward",
    values: [
      { episode: 0, value: -0.5 },
      { episode: 1, value: -0.3 },
      ...
    ]
  }
}
```

**可选 metric 值**：`reward`, `loss`, `distance_score`, `logistics_score`, `throughput`, `utilization`, `flow_clarity`

**文件位置建议**：`ResultsPage.tsx` 中现有的指标曲线区域

---

### 2. 摆放过程回放

**功能描述**：逐步展示每个功能单元的放置过程，支持暂停/快进/后退

**后端 API**：
```
POST /api/replay/{project_id}/start       # 启动回放会话
GET  /api/replay/{project_id}/step/{step} # 获取指定步骤数据
POST /api/replay/{project_id}/forward     # 执行一步
DELETE /api/replay/{project_id}/session   # 关闭会话
```

**响应示例**：
```typescript
// GET /api/replay/{id}/step/{step}
{
  code: 0,
  data: {
    step: 3,
    total_steps: 10,
    current_unit: { id: "station_1", width: 7, height: 4, ... },
    placed_units: [
      { step: 0, unit_id: "rec_dock", x: 30, y: 0, angle: 270 },
      ...
    ],
    heatmap: ActionHeatmap  // Q值热力图数据
  }
}
```

**UI 建议**：
- 新建 `ReplayPage.tsx` 或在 `ResultsPage.tsx` 中添加回放区域
- 播放控制栏：播放/暂停、上一步、下一步、进度条
- 布局画布：显示当前已放置的单元和正在放置的单元

---

### 3. Q值决策热力图

**功能描述**：显示 Agent 对每个位置的 Q 值评估，用热力图形式展示

**后端 API**：
```
GET /api/replay/{project_id}/heatmap
```

**响应格式**：
```typescript
interface ActionHeatmap {
  grid_width: number;
  grid_height: number;
  angle_options: number[];      // [0, 90, 180, 270]
  q_values: number[][][];       // [angle][y][x] Q值矩阵
  valid_actions: number[];      // 有效动作索引列表
  selected_action: {
    x: number;
    y: number;
    angle: number;
    q_value: number;
  };
  q_min: number;
  q_max: number;
}
```

**UI 建议**：
- 使用 Canvas 或 SVG 绘制热力图网格
- 支持切换不同旋转角度（0°, 90°, 180°, 270°）的 Tab
- 用颜色渐变表示 Q 值高低（红高蓝低）
- 高亮显示最终选择的位置

---

### 4. 结果导出按钮

**功能描述**：下载最佳布局 JSON 文件

**后端 API**：
```
GET /api/results/{project_id}/best
```

**实现方式**：
```typescript
const downloadBestLayout = async (projectId: string) => {
  const res = await resultsApi.getBestLayout(projectId);
  if (res.code === 0 && res.data?.layout) {
    const blob = new Blob([JSON.stringify(res.data.layout, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `layout_best_${projectId}.json`;
    a.click();
  }
};
```

**UI 建议**：在 `ResultsPage.tsx` 最佳布局卡片中添加"导出布局"按钮

---

## 🟡 中优先级（用户体验）

### 5. 预计剩余时间显示

**后端字段**：`trainingStatus.estimated_remaining`（秒）

**显示位置**：`TrainingPage.tsx` 训练进度区域

**格式化示例**：
```typescript
const formatTime = (seconds: number | null) => {
  if (seconds === null) return "-";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
};

// UI
<Text>预计剩余: {formatTime(trainingStatus.estimated_remaining)}</Text>
```

---

### 6. 探索率(ε)显示

**后端字段**：`trainingStatus.current_epsilon`

**显示位置**：`TrainingPage.tsx` 训练进度区域

**格式化**：
```typescript
<Text>探索率: {(trainingStatus.current_epsilon ?? 0).toFixed(4)}</Text>
```

---

### 7. 约束设置 UI

**功能描述**：在配置页面添加约束设置表单

**约束类型**：
```typescript
interface Constraints {
  // 障碍物分类（简化配置）
  fixed_obstacles?: string[];        // 固定障碍物ID列表
  movable_obstacles?: string[];      // 可移动障碍物ID列表
  default_wall_attach?: string[];    // 默认贴墙的单元ID列表
  
  // 详细约束配置
  fixed_positions?: { unit_id: string; x: number; y: number; angle: number }[];
  adjacency?: { unit_a: string; unit_b: string; direction?: string }[];
  wall_attach?: { unit_id: string; wall: "top" | "bottom" | "left" | "right" }[];
}
```

**UI 建议**：
- 在 `TrainingPage.tsx` 或 `BuilderPage.tsx` 中添加约束配置区域
- 障碍物分类：从 `obstacles` 列表中勾选固定/可移动
- 默认贴墙：勾选需要贴墙的单元（自动选择最近墙壁）
- 固定位置：选择单元 → 输入坐标和角度
- 相邻约束：选择两个单元 → 选择方向
- 贴墙约束：选择单元 → 选择具体墙壁（top/bottom/left/right）

---

### 8. 物料流线箭头

**功能描述**：在布局渲染时显示物料流向箭头

**数据来源**：`factory_config.routes` 数组

**实现方式**：
- 在 ReactFlow 或 SVG 布局中，根据 routes 数据绘制从 `from` 到 `to` 的箭头
- 箭头粗细可表示 `batch_size`

---

## 🟢 低优先级（锦上添花）

### 9. CSV 指标下载

**功能描述**：下载训练过程的 rewards.csv 和 losses.csv

**文件位置**：`data/projects/{project_id}/metrics/`

**实现方式**：后端可新增下载接口，或前端直接访问静态文件路径

---

### 10. 多布局对比

**功能描述**：对比不同检查点的布局指标

**实现方式**：
- 选择多个检查点
- 并排显示布局和指标对比表

---

## 后端已更新的数据结构

### TrainingProgress（状态API返回）

```typescript
interface TrainingProgress {
  project_id: string;
  status: "pending" | "running" | "paused" | "completed" | "failed" | "stopped";
  current_step: number;
  total_steps: number;
  current_episode: number;
  best_reward: number | null;
  current_epsilon: number | null;        // 新增：当前探索率
  estimated_remaining: number | null;    // 新增：预计剩余时间（秒）
}
```

### 文件存储结构

```
data/projects/{project_id}/
├── metrics/
│   ├── rewards.csv   # episode,step,reward,mean_reward_100,epsilon
│   └── losses.csv    # step,loss
├── layouts/
│   ├── layout_ep{N}.json
│   └── layout_best.json
└── checkpoints/
    ├── model_ep{N}.pth
    ├── model_best.pth
    └── model_final.pth
```

---

## 实现建议

1. **优先级排序**：按 🔴 → 🟡 → 🟢 顺序实现
2. **复用组件**：热力图和布局渲染可抽取为独立组件
3. **状态管理**：回放功能建议使用 Zustand 管理回放状态
4. **图表库**：推荐使用 `recharts`（已在项目依赖中）或 `echarts-for-react`

---

## 更新日志

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2024-12-19 | 初始版本 |


