/**
 * FactoGenie 前端类型定义
 * 与后端 API Contract 保持一致
 */

// ========== 通用响应 ==========
export interface ApiResponse<T> {
  code: number;
  message?: string;
  data: T;
}

// ========== 工厂配置 ==========
export interface Route {
  from: string;
  to: string;
  material: string;
  batch_size: number;
  travel_time: number;
  transporter_id: string;
}

export interface Assembly {
  station: string;
  inputs: Record<string, number>;
  output: string;
  process_time: number;
}

export interface MonitorItem {
  node: string;
  material: string;
}

export interface Transporter {
  id: string;
  count: number;
  speed: number;
}

export interface FactoryConfig {
  initial_inventory: Record<string, Record<string, number>>;
  routes: Route[];
  assemblies: Assembly[];
  summary: {
    finished_node: string;
    finished_material: string;
  };
  monitor: MonitorItem[];
  transporters: Transporter[];
  layout: string;
}

// ========== 布局配置 ==========
export interface FunctionalUnit {
  id: string;
  label: string;
  width: number;
  height: number;
  movable: boolean;
  x?: number;
  y?: number;
  angle?: number;
}

export interface Obstacle {
  id: string;
  label: string;
  width: number;
  height: number;
  movable: boolean;
  x?: number;
  y?: number;
  angle?: number;
}

export interface LayoutConfig {
  canvas: {
    width: number;
    height: number;
  };
  fus: FunctionalUnit[];
  obstacles: Obstacle[];
}

// ========== 约束 ==========
export interface FixedPosition {
  unit_id: string;
  x: number;
  y: number;
  angle: number;
}

export interface AdjacencyConstraint {
  unit_a: string;
  unit_b: string;
  direction?: 'horizontal' | 'vertical' | 'any';
}

export interface WallAttachConstraint {
  unit_id: string;
  wall: 'top' | 'bottom' | 'left' | 'right';
}

export interface Constraints {
  fixed_positions: FixedPosition[];
  adjacency: AdjacencyConstraint[];
  wall_attach: WallAttachConstraint[];
}

// ========== 训练参数 ==========
export interface ObjectiveWeights {
  distance: number;
  logistics: number;
  flow: number;
  throughput: number;
  utilization: number;
}

export type PlacementOrder = 
  | 'size_desc'
  | 'size_asc'
  | 'flow_desc'
  | 'random'
  | 'process_flow'
  | 'logistics_intensity';

export interface TrainingParams {
  total_steps: number;
  learning_rate: number;
  batch_size: number;
  replay_size: number;
  replay_start_size: number;
  epsilon_start: number;
  epsilon_final: number;
  epsilon_decay_frames: number;
  sync_target_every: number;
  double_dqn: boolean;
  dueling: boolean;
  noisy_net: boolean;
  prioritized: boolean;
  simulation_duration: number;
  use_simulation: boolean;
  weights: ObjectiveWeights;
  placement_order: PlacementOrder;
  checkpoint_interval: number;  // 每多少episode保存一次权重，0表示不保存
}

// ========== 训练进度 ==========
export interface EpisodeMetrics {
  episode: number;
  reward: number;
  loss?: number;
  epsilon?: number;
  metrics?: {
    distance_score: number;
    logistics_score: number;
    throughput: number;
    utilization: number;
    flow_clarity: number;
  };
}

export interface TrainingProgress {
  task_id: string;
  status: 'pending' | 'running' | 'paused' | 'completed' | 'failed' | 'stopped';
  current_step: number;
  total_steps: number;
  current_episode: number;
  elapsed_time: number;
  estimated_remaining: number;
  latest_metrics?: EpisodeMetrics;
}

// ========== 布局结果 ==========
export interface Placement {
  x: number;
  y: number;
  angle: number;
}

export interface LayoutMetrics {
  distance_score: number;
  logistics_score: number;
  throughput: number;
  utilization: number;
  flow_clarity: number;
  finished_goods?: number;
}

export interface LayoutResult {
  episode: number;
  timestamp: string;
  reward: number;
  placements: Record<string, Placement>;
  metrics: LayoutMetrics;
}

// ========== 动作热力图 ==========
export interface ActionHeatmap {
  step: number;
  unit_id: string;
  unit_label: string;
  grid_width: number;
  grid_height: number;
  angle_options: number[];
  q_values: number[][][]; // [angle][y][x]
  selected_action: {
    x: number;
    y: number;
    angle: number;
    q_value: number;
  };
}

// ========== 训练记录 ==========
export interface Checkpoint {
  episode: number;
  timestamp: string;
  model_path: string;
  reward: number;
  is_best: boolean;
}

export interface TrainingRecord {
  id: string;
  name: string;
  created_at: string;
  status: 'running' | 'completed' | 'failed' | 'stopped';
  factory_config: FactoryConfig;
  layout_config: LayoutConfig;
  constraints?: Constraints;
  training_params: TrainingParams;
  checkpoints: Checkpoint[];
  final_result?: LayoutResult;
}

// ========== WebSocket 消息 ==========
export type WSMessageType = 
  | 'progress'
  | 'episode_complete'
  | 'layout'
  | 'action'
  | 'complete'
  | 'error';

export interface WSMessage<T = unknown> {
  type: WSMessageType;
  data: T;
}

// ========== 默认值 ==========
export const DEFAULT_TRAINING_PARAMS: TrainingParams = {
  total_steps: 50000,
  learning_rate: 0.00002,
  batch_size: 32,
  replay_size: 50000,
  replay_start_size: 5000,
  epsilon_start: 1.0,
  epsilon_final: 0.05,
  epsilon_decay_frames: 150000,
  sync_target_every: 2000,
  double_dqn: false,
  dueling: false,
  noisy_net: false,
  prioritized: false,
  simulation_duration: 2000,
  use_simulation: true,
  weights: {
    distance: 0.25,
    logistics: 0.25,
    flow: 0.15,
    throughput: 0.20,
    utilization: 0.15,
  },
  placement_order: 'size_desc',
  checkpoint_interval: 1000,
};

