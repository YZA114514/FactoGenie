/**
 * API 服务封装
 */
import axios from 'axios';
import type {
  ApiResponse,
  FactoryConfig,
  LayoutConfig,
  Constraints,
  TrainingParams,
  TrainingProgress,
  TrainingRecord,
  LayoutResult,
  ActionHeatmap,
} from '../types';

const BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 30000,
});

// ========== 配置管理 ==========

export const configApi = {
  uploadFactoryConfig: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post<ApiResponse<{ config: FactoryConfig; validation: { valid: boolean; errors: string[] } }>>(
      '/config/factory/upload',
      formData
    );
    return res.data;
  },

  saveFactoryConfig: async (config: FactoryConfig) => {
    const res = await api.post<ApiResponse<{ config_id: string }>>('/config/factory', config);
    return res.data;
  },

  uploadLayoutConfig: async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    const res = await api.post<ApiResponse<{ config: LayoutConfig; validation: { valid: boolean; errors: string[] } }>>(
      '/config/layout/upload',
      formData
    );
    return res.data;
  },

  saveLayoutConfig: async (config: LayoutConfig) => {
    const res = await api.post<ApiResponse<{ config_id: string }>>('/config/layout', config);
    return res.data;
  },

  saveConstraints: async (constraints: Constraints) => {
    const res = await api.post<ApiResponse<{ constraints_id: string }>>('/config/constraints', constraints);
    return res.data;
  },
};

// ========== 训练任务 ==========

export interface StartTrainingRequest {
  name: string;
  factory_config: FactoryConfig;
  layout_config: LayoutConfig;
  constraints?: Constraints;
  training_params: TrainingParams;
}

export const trainingApi = {
  start: async (request: StartTrainingRequest) => {
    const res = await api.post<ApiResponse<{ task_id: string }>>('/training/start', request);
    return res.data;
  },

  stop: async (taskId: string) => {
    const res = await api.post<ApiResponse<{ status: string }>>(`/training/${taskId}/stop`);
    return res.data;
  },

  pause: async (taskId: string) => {
    const res = await api.post<ApiResponse<{ status: string }>>(`/training/${taskId}/pause`);
    return res.data;
  },

  resume: async (taskId: string) => {
    const res = await api.post<ApiResponse<{ status: string }>>(`/training/${taskId}/resume`);
    return res.data;
  },

  getStatus: async (taskId: string) => {
    const res = await api.get<ApiResponse<TrainingProgress>>(`/training/${taskId}/status`);
    return res.data;
  },

  getRecords: async (page = 1, size = 20) => {
    const res = await api.get<ApiResponse<{ total: number; records: TrainingRecord[] }>>(
      '/training/records',
      { params: { page, size } }
    );
    return res.data;
  },

  getRecord: async (recordId: string) => {
    const res = await api.get<ApiResponse<TrainingRecord>>(`/training/records/${recordId}`);
    return res.data;
  },

  deleteRecord: async (recordId: string) => {
    const res = await api.delete<ApiResponse<null>>(`/training/records/${recordId}`);
    return res.data;
  },
};

// ========== 结果查询 ==========

export const resultsApi = {
  getLayouts: async (taskId: string, page = 1, size = 20) => {
    const res = await api.get<ApiResponse<{ total: number; layouts: LayoutResult[] }>>(
      `/results/${taskId}/layouts`,
      { params: { page, size } }
    );
    return res.data;
  },

  getBestLayout: async (taskId: string) => {
    const res = await api.get<ApiResponse<LayoutResult>>(`/results/${taskId}/best`);
    return res.data;
  },

  getMetrics: async (taskId: string, metric: string, start = 0, end?: number) => {
    const res = await api.get<ApiResponse<{ metric: string; values: { episode: number; value: number }[] }>>(
      `/results/${taskId}/metrics`,
      { params: { metric, start, end } }
    );
    return res.data;
  },

  getHeatmap: async (taskId: string, episode: number, step: number) => {
    const res = await api.get<ApiResponse<ActionHeatmap>>(`/results/${taskId}/heatmap/${episode}/${step}`);
    return res.data;
  },
};

// ========== WebSocket ==========

export const createTrainingWebSocket = (taskId: string): WebSocket => {
  const wsUrl = BASE_URL.replace('http', 'ws') + `/ws/training/${taskId}`;
  return new WebSocket(wsUrl);
};

export default api;

