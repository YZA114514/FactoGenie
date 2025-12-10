# FactoGenie API Contract / 前后端接口约定

## 概述

本文档定义前后端数据传输格式和API接口规范，供前后端开发人员参照。

---

## 1. 通用约定

### 1.1 基础URL
```
REST API: http://localhost:8000/api
WebSocket: ws://localhost:8000/api/ws
```

### 1.2 响应格式
```json
{
  "code": 0,           // 0=成功, 非0=错误码
  "message": "success",
  "data": { ... }      // 实际数据
}
```

### 1.3 错误码
| code | 说明 |
|------|------|
| 0 | 成功 |
| 1001 | 参数错误 |
| 1002 | 资源不存在 |
| 1003 | 文件格式错误 |
| 2001 | 训练任务已存在 |
| 2002 | 训练任务不存在 |
| 5000 | 服务器内部错误 |

---

## 2. 数据结构定义

### 2.1 工厂配置 (FactoryConfig)

对应 `simulation/configs/chair_factory.json`

```typescript
interface FactoryConfig {
  initial_inventory: {
    [node_id: string]: {
      [material: string]: number
    }
  };
  routes: Route[];
  assemblies: Assembly[];
  summary: {
    finished_node: string;
    finished_material: string;
  };
  monitor: MonitorItem[];
  transporters: Transporter[];
  layout: string;  // 关联的布局文件路径
}

interface Route {
  from: string;
  to: string;
  material: string;
  batch_size: number;
  travel_time: number;
  transporter_id: string;
}

interface Assembly {
  station: string;
  inputs: { [material: string]: number };
  output: string;
  process_time: number;
}

interface MonitorItem {
  node: string;
  material: string;
}

interface Transporter {
  id: string;
  count: number;
  speed: number;
}
```

### 2.2 布局配置 (LayoutConfig)

对应 `simulation/layouts/chair_layout.json`

```typescript
interface LayoutConfig {
  canvas: {
    width: number;
    height: number;
  };
  fus: FunctionalUnit[];      // 功能单元
  obstacles: Obstacle[];       // 障碍物
}

interface FunctionalUnit {
  id: string;
  label: string;
  width: number;
  height: number;
  movable: boolean;
  x?: number;       // 初始/固定位置
  y?: number;
  angle?: number;   // 0, 90, 180, 270
}

interface Obstacle {
  id: string;
  label: string;
  width: number;
  height: number;
  movable: boolean;
  x?: number;
  y?: number;
  angle?: number;
}
```

### 2.3 特殊约束 (Constraints)

```typescript
interface Constraints {
  fixed_positions: FixedPosition[];   // 固定位置（硬约束）
  adjacency: AdjacencyConstraint[];   // 相邻约束（硬约束）
  wall_attach: WallAttachConstraint[]; // 贴墙约束（硬约束）
}

interface FixedPosition {
  unit_id: string;   // 功能单元ID
  x: number;         // 固定X坐标
  y: number;         // 固定Y坐标
  angle: number;     // 固定角度 (0, 90, 180, 270)
}

interface AdjacencyConstraint {
  unit_a: string;    // 单元A的ID
  unit_b: string;    // 单元B的ID
  direction?: "horizontal" | "vertical" | "any";  // 相邻方向，默认any
}

interface WallAttachConstraint {
  unit_id: string;   // 功能单元ID
  wall: "top" | "bottom" | "left" | "right";  // 必须贴的墙
}
```

**约束行为说明**：

| 约束类型 | 实现方式 | 不满足时的行为 |
|----------|----------|----------------|
| 固定位置 | 训练开始前预先放置，Agent不参与 | - |
| 相邻约束 | 布局完成后检查 | 奖励=-5，标记为约束违反 |
| 贴墙约束 | 在有效动作筛选时检查 | 不满足的位置不可选 |

**示例**：
```json
{
  "constraints": {
    "fixed_positions": [
      { "unit_id": "rec_dock", "x": 0, "y": 10, "angle": 0 }
    ],
    "adjacency": [
      { "unit_a": "station_1", "unit_b": "station_2", "direction": "horizontal" }
    ],
    "wall_attach": [
      { "unit_id": "ship_dock", "wall": "right" }
    ]
  }
}
```

### 2.4 训练参数 (TrainingParams)

对应 `main.py` 参数

```typescript
interface TrainingParams {
  // 基础参数
  total_steps: number;           // 默认 50000
  learning_rate: number;         // 默认 0.00002
  batch_size: number;            // 默认 32
  
  // 经验回放
  replay_size: number;           // 默认 50000
  replay_start_size: number;     // 默认 5000
  
  // 探索参数
  epsilon_start: number;         // 默认 1.0
  epsilon_final: number;         // 默认 0.05
  epsilon_decay_frames: number;  // 默认 150000
  
  // 网络参数
  sync_target_every: number;     // 默认 2000
  double_dqn: boolean;           // 默认 false
  dueling: boolean;              // 默认 false
  noisy_net: boolean;            // 默认 false
  prioritized: boolean;          // 默认 false
  
  // 奖励权重
  weights: ObjectiveWeights;
  
  // 摆放顺序
  placement_order: PlacementOrder;
  
  // 仿真参数
  simulation_duration: number;   // 默认 2000
  use_simulation: boolean;       // 默认 true
  
  // 检查点参数
  checkpoint_interval: number;   // 每多少episode保存一次权重，默认 1000，0表示不保存中间过程
  
  // 校准参数
  calibrate_episodes: number;    // 校准回合数，默认 0（不校准），建议 100
  throughput_target?: number;    // 用户指定的吞吐量目标，可选
}

interface ObjectiveWeights {
  distance: number;      // 默认 0.25
  logistics: number;     // 默认 0.25
  flow: number;          // 默认 0.15
  throughput: number;    // 默认 0.20
  utilization: number;   // 默认 0.15
}

type PlacementOrder = 
  | "size_desc"        // 面积降序（默认）
  | "size_asc"         // 面积升序
  | "flow_desc"        // 物流强度降序
  | "random"           // 随机
  | "process_flow"     // 工艺流程顺序
  | "logistics_intensity"; // 物流强度顺序
```

### 2.5 训练进度 (TrainingProgress)

```typescript
interface TrainingProgress {
  task_id: string;
  status: "pending" | "running" | "paused" | "completed" | "failed" | "stopped";
  current_step: number;
  total_steps: number;
  current_episode: number;
  elapsed_time: number;      // 秒
  estimated_remaining: number; // 秒
  latest_metrics: EpisodeMetrics;
}

interface EpisodeMetrics {
  episode: number;
  reward: number;
  loss: number;
  epsilon: number;
  metrics: {
    distance_score: number;
    logistics_score: number;
    throughput: number;
    utilization: number;
    flow_clarity: number;
  };
}
```

### 2.6 布局结果 (LayoutResult)

```typescript
interface LayoutResult {
  episode: number;
  timestamp: string;
  reward: number;
  placements: {
    [unit_id: string]: {
      x: number;
      y: number;
      angle: number;
    }
  };
  metrics: {
    distance_score: number;
    logistics_score: number;
    throughput: number;
    utilization: number;
    flow_clarity: number;
    finished_goods: number;
  };
}
```

### 2.7 动作热力图 (ActionHeatmap)

```typescript
interface ActionHeatmap {
  step: number;
  unit_id: string;
  unit_label: string;
  grid_width: number;
  grid_height: number;
  angle_options: number[];    // [0, 90, 180, 270]
  q_values: number[][][];     // [angle][y][x] Q值矩阵
  selected_action: {
    x: number;
    y: number;
    angle: number;
    q_value: number;
  };
}
```

### 2.8 训练记录 (TrainingRecord)

```typescript
interface TrainingRecord {
  id: string;
  name: string;
  created_at: string;
  status: "running" | "completed" | "failed" | "stopped";
  factory_config: FactoryConfig;
  layout_config: LayoutConfig;
  constraints: Constraints;
  training_params: TrainingParams;
  checkpoints: Checkpoint[];
  final_result?: LayoutResult;
}

interface Checkpoint {
  episode: number;
  timestamp: string;
  model_path: string;      // 模型权重文件路径
  reward: number;
  is_best: boolean;
}
```

---

## 3. REST API 接口

### 3.1 配置管理

#### 上传工厂配置文件
```
POST /api/config/factory/upload
Content-Type: multipart/form-data

Request:
  file: File (JSON)

Response:
{
  "code": 0,
  "data": {
    "config": FactoryConfig,
    "validation": {
      "valid": true,
      "errors": []
    }
  }
}
```

#### 保存工厂配置
```
POST /api/config/factory
Content-Type: application/json

Request: FactoryConfig

Response:
{
  "code": 0,
  "data": {
    "config_id": "string"
  }
}
```

#### 上传布局配置文件
```
POST /api/config/layout/upload
Content-Type: multipart/form-data

Request:
  file: File (JSON)

Response:
{
  "code": 0,
  "data": {
    "config": LayoutConfig,
    "validation": {
      "valid": true,
      "errors": []
    }
  }
}
```

#### 保存布局配置
```
POST /api/config/layout
Content-Type: application/json

Request: LayoutConfig

Response:
{
  "code": 0,
  "data": {
    "config_id": "string"
  }
}
```

#### 保存约束配置
```
POST /api/config/constraints
Content-Type: application/json

Request: Constraints

Response:
{
  "code": 0,
  "data": {
    "constraints_id": "string"
  }
}
```

### 3.2 训练任务

#### 启动训练
```
POST /api/training/start
Content-Type: application/json

Request:
{
  "name": "string",                  // 训练任务名称
  "factory_config": FactoryConfig,
  "layout_config": LayoutConfig,
  "constraints": Constraints,
  "training_params": TrainingParams
}

Response:
{
  "code": 0,
  "data": {
    "task_id": "string"
  }
}
```

#### 停止训练
```
POST /api/training/{task_id}/stop

Response:
{
  "code": 0,
  "data": {
    "status": "stopped"
  }
}
```

#### 暂停训练
```
POST /api/training/{task_id}/pause

Response:
{
  "code": 0,
  "data": {
    "status": "paused"
  }
}
```

#### 恢复训练
```
POST /api/training/{task_id}/resume

Response:
{
  "code": 0,
  "data": {
    "status": "running"
  }
}
```

#### 获取训练状态
```
GET /api/training/{task_id}/status

Response:
{
  "code": 0,
  "data": TrainingProgress
}
```

#### 获取训练记录列表
```
GET /api/training/records?page=1&size=20

Response:
{
  "code": 0,
  "data": {
    "total": 100,
    "records": TrainingRecord[]
  }
}
```

#### 获取单个训练记录详情
```
GET /api/training/records/{record_id}

Response:
{
  "code": 0,
  "data": TrainingRecord
}
```

#### 删除训练记录
```
DELETE /api/training/records/{record_id}

Response:
{
  "code": 0,
  "message": "deleted"
}
```

### 3.3 校准管理

#### 触发校准
```
POST /api/calibration/run
Content-Type: application/json

Request:
{
  "factory_config": FactoryConfig,
  "layout_config": LayoutConfig,
  "n_episodes": 100,
  "simulation_duration": 2000,
  "throughput_target": null    // 可选，用户指定的吞吐量目标
}

Response:
{
  "code": 0,
  "data": {
    "config_hash": "abc123def456",
    "bounds": {
      "distance": { "best": 12.5, "worst": 25.3 },
      "logistics": { "best": 4200, "worst": 11000 },
      "throughput": { "best": 380, "worst": 150 },
      "utilization": { "best": 0.75, "worst": 0.35 }
    }
  }
}
```

#### 查询校准缓存
```
GET /api/calibration/cache?factory_hash={hash}

Response:
{
  "code": 0,
  "data": {
    "exists": true,
    "bounds": { ... },
    "created_at": "2024-12-10T10:00:00Z"
  }
}
```

#### 清除校准缓存
```
DELETE /api/calibration/cache/{config_hash}

Response:
{
  "code": 0,
  "message": "deleted"
}
```

### 3.4 结果查询

#### 获取布局历史
```
GET /api/results/{task_id}/layouts?page=1&size=20

Response:
{
  "code": 0,
  "data": {
    "total": 500,
    "layouts": LayoutResult[]
  }
}
```

#### 获取最佳布局
```
GET /api/results/{task_id}/best

Response:
{
  "code": 0,
  "data": LayoutResult
}
```

#### 获取指标曲线数据
```
GET /api/results/{task_id}/metrics?metric=reward&start=0&end=1000

Response:
{
  "code": 0,
  "data": {
    "metric": "reward",
    "values": [
      { "episode": 0, "value": -0.5 },
      { "episode": 1, "value": -0.3 },
      ...
    ]
  }
}
```

#### 获取动作热力图
```
GET /api/results/{task_id}/heatmap/{episode}/{step}

Response:
{
  "code": 0,
  "data": ActionHeatmap
}
```

### 3.4 模型检查点

#### 获取检查点列表
```
GET /api/checkpoints/{task_id}

Response:
{
  "code": 0,
  "data": Checkpoint[]
}
```

#### 加载检查点回放
```
POST /api/checkpoints/{task_id}/{episode}/replay

Response:
{
  "code": 0,
  "data": {
    "replay_task_id": "string"
  }
}
```

---

## 4. WebSocket 接口

### 4.1 训练进度实时推送

```
连接: ws://localhost:8000/api/ws/training/{task_id}

服务端推送消息类型:
```

#### 进度更新
```json
{
  "type": "progress",
  "data": {
    "current_step": 1000,
    "total_steps": 50000,
    "current_episode": 50,
    "epsilon": 0.95,
    "loss": 0.023
  }
}
```

#### Episode完成
```json
{
  "type": "episode_complete",
  "data": EpisodeMetrics
}
```

#### 新布局生成
```json
{
  "type": "layout",
  "data": LayoutResult
}
```

#### 动作决策（用于可视化摆放过程）
```json
{
  "type": "action",
  "data": {
    "step": 3,
    "unit_id": "station_1",
    "placement": { "x": 10, "y": 5, "angle": 90 },
    "heatmap": ActionHeatmap  // 可选，仅在启用详细模式时
  }
}
```

#### 训练完成
```json
{
  "type": "complete",
  "data": {
    "best_episode": 450,
    "best_reward": 0.85,
    "total_time": 3600
  }
}
```

#### 错误
```json
{
  "type": "error",
  "data": {
    "code": 5000,
    "message": "Training failed: out of memory"
  }
}
```

### 4.2 回放WebSocket

```
连接: ws://localhost:8000/api/ws/replay/{replay_task_id}

客户端发送:
{
  "action": "play" | "pause" | "step" | "seek",
  "step"?: number  // seek时指定
}

服务端推送:
{
  "type": "step",
  "data": {
    "step": 3,
    "unit_id": "station_1",
    "placement": { "x": 10, "y": 5, "angle": 90 },
    "heatmap": ActionHeatmap,
    "current_layout": { ... }  // 当前所有已放置单元
  }
}
```

---

## 5. 文件存储约定

### 5.1 目录结构
```
data/
├── uploads/           # 用户上传的配置文件
│   └── {task_id}/
│       ├── factory_config.json
│       └── layout_config.json
├── checkpoints/       # 模型检查点
│   └── {task_id}/
│       ├── model_ep100.pth
│       ├── model_ep200.pth
│       └── model_best.pth
└── results/           # 训练结果
    └── {task_id}/
        ├── metrics.csv
        ├── rewards.csv
        ├── layouts/
        │   ├── layout_ep100.json
        │   └── layout_ep200.json
        └── heatmaps/     # 可选
            └── ep100_step3.json
```

### 5.2 文件命名规则
- task_id: UUID格式，如 `a1b2c3d4-e5f6-7890-abcd-ef1234567890`
- 检查点: `model_ep{episode}.pth`, `model_best.pth`
- 布局: `layout_ep{episode}.json`
- 热力图: `ep{episode}_step{step}.json`

---

## 6. 默认参数值

```json
{
  "training_params": {
    "total_steps": 50000,
    "learning_rate": 0.00002,
    "batch_size": 32,
    "replay_size": 50000,
    "replay_start_size": 5000,
    "epsilon_start": 1.0,
    "epsilon_final": 0.05,
    "epsilon_decay_frames": 150000,
    "sync_target_every": 2000,
    "double_dqn": false,
    "dueling": false,
    "noisy_net": false,
    "prioritized": false,
    "simulation_duration": 2000,
    "use_simulation": true,
    "weights": {
      "distance": 0.25,
      "logistics": 0.25,
      "flow": 0.15,
      "throughput": 0.20,
      "utilization": 0.15
    },
    "placement_order": "size_desc",
    "checkpoint_interval": 1000,
    "calibrate_episodes": 100,
    "throughput_target": null
  }
}
```

---

## 7. 验证规则

### 7.1 工厂配置验证
- `routes` 中的 `from`/`to` 必须在布局中存在
- `assemblies` 中的 `station` 必须在布局中存在
- `transporters` 中的 `id` 必须被 `routes` 引用

### 7.2 布局配置验证
- `canvas.width` 和 `canvas.height` 必须 > 0
- 所有单元的 `width` 和 `height` 必须 > 0
- `movable=false` 的单元必须有 `x`, `y` 坐标

### 7.3 约束验证
- `fixed_positions` 中的 `unit_id` 必须存在
- `adjacency` 中的 `unit_a` 和 `unit_b` 必须存在
- `wall_attach` 中的 `unit_id` 必须存在

### 7.4 训练参数验证
- 所有权重之和必须等于 1.0（允许0.001误差）
- `total_steps` >= 1000
- `learning_rate` 在 (0, 1) 范围内
- `batch_size` >= 1

---

## 更新日志

| 版本 | 日期 | 说明 |
|------|------|------|
| 1.0 | 2024-12-10 | 初始版本 |

