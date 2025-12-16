import { create } from "zustand";
import type {
  Material,
  MaterialId,
  TransporterDefinition,
  TransporterId,
  FactoryNode,
  NodeId,
  Assembly,
  Route,
  RouteId,
  RouteConfiguration,
  FactoryStateSnapshot,
} from "../types/factory";

// 训练状态类型
interface TrainingStatus {
  status: string;
  current_step?: number;
  current_episode?: number;
  best_reward?: number;
  total_steps?: number;
}

interface FactoryStoreState extends FactoryStateSnapshot {
  // 训练相关状态
  currentProjectId: string | null;
  trainingStatus: TrainingStatus | null;
  setCurrentProjectId: (id: string | null) => void;
  setTrainingStatus: (status: TrainingStatus | null) => void;

  addMaterial: (material: Material) => void;
  removeMaterial: (id: MaterialId) => void;
  upsertMaterial: (material: Material) => void;
  setSnapshot: (snapshot: FactoryStateSnapshot) => void;

  addTransporter: (t: TransporterDefinition) => void;
  removeTransporter: (id: TransporterId) => void;
  upsertTransporter: (t: TransporterDefinition) => void;

  addNode: (node: FactoryNode) => void;
  removeNode: (id: NodeId) => void;
  upsertNode: (node: FactoryNode) => void;

  addAssembly: (assembly: Assembly) => void;
  removeAssembly: (nodeId: NodeId) => void; // one per node assumption; otherwise extend to id
  upsertAssembly: (assembly: Assembly) => void;

  addRoute: (route: Route, config?: RouteConfiguration) => void;
  removeRoute: (routeId: RouteId) => void;
  upsertRoute: (route: Route) => void;
  upsertRouteConfig: (config: RouteConfiguration) => void;

  reset: () => void;
  setCanvasSize: (w: number, h: number) => void;
}

const initialState: FactoryStateSnapshot = {
  materials: [],
  transporters: [],
  nodes: [],
  assemblies: [],
  routes: [],
  routeConfigs: [],
  canvasSize: { width: 100, height: 100 },
};

const cleanupForMaterial = (state: FactoryStoreState, id: MaterialId) => {
  const assemblies = state.assemblies
    .map((a) => {
      const inputs = { ...a.inputs };
      if (inputs[id] !== undefined) {
        delete inputs[id];
      }
      const output = a.output === id ? undefined : a.output;
      if (!output && Object.keys(inputs).length === 0) {
        return null;
      }
      return { ...a, inputs, output: output ?? a.output };
    })
    .filter(Boolean) as Assembly[];
  const routes = state.routes.filter((r) => r.materialId !== id);
  const routeIds = new Set(routes.map((r) => r.id));
  const routeConfigs = state.routeConfigs.filter((c) => routeIds.has(c.routeId));
  return { assemblies, routes, routeConfigs };
};

const cleanupForTransporter = (state: FactoryStoreState, id: TransporterId) => ({
  routeConfigs: state.routeConfigs.filter((c) => c.transporterId !== id),
});

const cleanupForNode = (state: FactoryStoreState, id: NodeId) => {
  const nodes = state.nodes.filter((n) => n.id !== id);
  const assemblies = state.assemblies.filter((a) => a.nodeId !== id);
  const routes = state.routes.filter((r) => r.fromNodeId !== id && r.toNodeId !== id);
  const routeIds = new Set(routes.map((r) => r.id));
  const routeConfigs = state.routeConfigs.filter((c) => routeIds.has(c.routeId));
  return { nodes, assemblies, routes, routeConfigs };
};

export const useFactoryStore = create<FactoryStoreState>((set) => ({
  ...initialState,
  currentProjectId: null,
  trainingStatus: null,

  setCurrentProjectId: (id) => set({ currentProjectId: id }),
  setTrainingStatus: (status) => set({ trainingStatus: status }),

  setSnapshot: (snapshot) =>
    set(() => ({
      ...initialState,
      ...snapshot,
      canvasSize: snapshot.canvasSize ?? initialState.canvasSize,
    })),

  reset: () => set({ ...initialState, currentProjectId: null, trainingStatus: null }),
  setCanvasSize: (w, h) =>
    set((state) => ({
      ...state,
      canvasSize: { width: w, height: h },
    })),

  addMaterial: (material) =>
    set((state) => ({
      materials: state.materials.find((m) => m.id === material.id)
        ? state.materials
        : [...state.materials, material],
    })),

  upsertMaterial: (material) =>
    set((state) => ({
      materials: state.materials.some((m) => m.id === material.id)
        ? state.materials.map((m) => (m.id === material.id ? material : m))
        : [...state.materials, material],
    })),

  removeMaterial: (id) =>
    set((state) => {
      const cleaned = cleanupForMaterial(state, id);
      return {
        ...cleaned,
        materials: state.materials.filter((m) => m.id !== id),
      };
    }),

  addTransporter: (t) =>
    set((state) => ({
      transporters: state.transporters.find((x) => x.id === t.id)
        ? state.transporters
        : [...state.transporters, t],
    })),

  upsertTransporter: (t) =>
    set((state) => ({
      transporters: state.transporters.some((x) => x.id === t.id)
        ? state.transporters.map((x) => (x.id === t.id ? t : x))
        : [...state.transporters, t],
    })),

  removeTransporter: (id) =>
    set((state) => ({
      ...cleanupForTransporter(state, id),
      transporters: state.transporters.filter((t) => t.id !== id),
    })),

  addNode: (node) =>
    set((state) => ({
      nodes: state.nodes.find((n) => n.id === node.id) ? state.nodes : [...state.nodes, node],
    })),

  upsertNode: (node) =>
    set((state) => ({
      nodes: state.nodes.some((n) => n.id === node.id)
        ? state.nodes.map((n) => (n.id === node.id ? node : n))
        : [...state.nodes, node],
    })),

  removeNode: (id) =>
    set((state) => ({
      ...cleanupForNode(state, id),
    })),

  addAssembly: (assembly) =>
    set((state) => ({
      assemblies: [...state.assemblies.filter((a) => a.nodeId !== assembly.nodeId), assembly],
    })),

  upsertAssembly: (assembly) =>
    set((state) => ({
      assemblies: state.assemblies.some((a) => a.nodeId === assembly.nodeId)
        ? state.assemblies.map((a) => (a.nodeId === assembly.nodeId ? assembly : a))
        : [...state.assemblies, assembly],
    })),

  removeAssembly: (nodeId) =>
    set((state) => ({
      assemblies: state.assemblies.filter((a) => a.nodeId !== nodeId),
    })),

  addRoute: (route, config) =>
    set((state) => {
      const routes = state.routes.find((r) => r.id === route.id) ? state.routes : [...state.routes, route];
      const routeConfigs = config
        ? state.routeConfigs.find((c) => c.routeId === config.routeId)
          ? state.routeConfigs.map((c) => (c.routeId === config.routeId ? config : c))
          : [...state.routeConfigs, config]
        : state.routeConfigs;
      return { routes, routeConfigs };
    }),

  upsertRoute: (route) =>
    set((state) => ({
      routes: state.routes.some((r) => r.id === route.id)
        ? state.routes.map((r) => (r.id === route.id ? route : r))
        : [...state.routes, route],
    })),

  upsertRouteConfig: (config) =>
    set((state) => ({
      routeConfigs: state.routeConfigs.some((c) => c.routeId === config.routeId)
        ? state.routeConfigs.map((c) => (c.routeId === config.routeId ? config : c))
        : [...state.routeConfigs, config],
    })),

  removeRoute: (routeId) =>
    set((state) => ({
      routes: state.routes.filter((r) => r.id !== routeId),
      routeConfigs: state.routeConfigs.filter((c) => c.routeId !== routeId),
    })),
}));
