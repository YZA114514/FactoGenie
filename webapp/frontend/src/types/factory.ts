// Core data models for factory layout and logistics planning

export type MaterialId = string;
export type NodeId = string;
export type TransporterId = string;
export type RouteId = string;

export interface Material {
  id: MaterialId;
  color?: string;
  description?: string;
}

export interface TransporterDefinition {
  id: TransporterId;
  speed: number;
  count: number;
}

export type FactoryNodeType = "FU" | "Obstacle";

export interface FactoryNode {
  id: NodeId;
  type: FactoryNodeType;
  dimensions: {
    length: number;
    width: number;
    notchLength?: number;
    notchWidth?: number;
  };
  angle?: number;
  x?: number;
  y?: number;
  initialInventory?: Record<MaterialId, number>;
}

export interface Assembly {
  nodeId: NodeId;
  inputs: Record<MaterialId, number>;
  output: MaterialId;
  processTime: number;
  outputCount?: number; // 默认1，可选
}

export interface Route {
  id: RouteId;
  fromNodeId: NodeId;
  toNodeId: NodeId;
  materialId: MaterialId;
}

export interface RouteConfiguration {
  routeId: RouteId;
  transporterId: TransporterId;
  batchSize: number;
  travelTime: number;
}

// Aggregate state for convenience (used in store)
export interface FactoryStateSnapshot {
  materials: Material[];
  transporters: TransporterDefinition[];
  nodes: FactoryNode[];
  assemblies: Assembly[];
  routes: Route[];
  routeConfigs: RouteConfiguration[];
  canvasSize: { width: number; height: number };
}
