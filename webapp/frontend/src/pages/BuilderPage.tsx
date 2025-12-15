import { useEffect, useMemo, useState } from "react";
import {
  Typography,
  Space,
  Alert,
  List,
  Tag,
  Collapse,
  Form,
  Input,
  InputNumber,
  Select,
  Button,
  Row,
  Col,
  Modal,
  Card,
  message,
  Divider,
} from "antd";
import { PlusOutlined } from "@ant-design/icons";
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  addEdge,
  useNodesState,
  useEdgesState,
  type Node as RFNode,
  type Edge as RFEdge,
  type Connection,
} from "reactflow";
import "reactflow/dist/style.css";
import { useFactoryStore } from "../store/factoryStore";
import type { FactoryNode, Assembly, Route, RouteConfiguration, FactoryStateSnapshot } from "../types/factory";
import { configApi } from "../services/api";
import type { LayoutConfig, FactoryConfig } from "../types";

const { Title, Paragraph, Text } = Typography;
const { Panel } = Collapse;
const INLINE_LAYOUT_NAME = "inline_layout.json";

const BuilderPage = () => {
  const {
    materials,
    transporters,
    nodes,
    assemblies,
    routes,
    routeConfigs,
    canvasSize,
    setCanvasSize,
    upsertNode,
    upsertAssembly,
    addRoute,
    addMaterial,
    addTransporter,
  } = useFactoryStore();

  const [nodeForm] = Form.useForm<FactoryNode & { inventory?: { material: string; quantity: number }[] }>();
  const [assemblyForm] = Form.useForm<Assembly & { inputsList?: { material: string; quantity: number }[] }>();
  const [routeForm] = Form.useForm<{ materialId: string; transporterId: string; batchSize: number }>();
  const [metaForm] = Form.useForm<{ finishedNode?: string; finishedMaterial?: string; monitors?: { node: string; material: string }[] }>();
  const nodeTypeWatch = Form.useWatch("type", nodeForm);

  const [rfNodes, setRfNodes, onNodesChange] = useNodesState<RFNode>([]);
  const [rfEdges, setRfEdges, onEdgesChange] = useEdgesState<RFEdge>([]);
  const [pendingConn, setPendingConn] = useState<Connection | null>(null);
  const [routeModalOpen, setRouteModalOpen] = useState(false);
  const [materialForm] = Form.useForm<{ id: string; color?: string; description?: string }>();
  const [transporterForm] = Form.useForm<{ id: string; speed: number; count: number }>();
  const [savingBackend, setSavingBackend] = useState(false);
  const [savingLocal, setSavingLocal] = useState(false);
  const [lastImport, setLastImport] = useState<{ type: "layout" | "factory"; name: string } | null>(null);

  const fuNodes = useMemo(() => nodes.filter((n) => n.type === "FU"), [nodes]);
  const obstacleNodes = useMemo(() => nodes.filter((n) => n.type === "Obstacle"), [nodes]);
  const summary = [
    { label: "功能单元 (FU)", value: fuNodes.length },
    { label: "障碍物 (Obstacle)", value: obstacleNodes.length },
    { label: "工序 (assemblies)", value: assemblies.length },
    { label: "物料 (materials)", value: materials.length },
    { label: "运输工具 (transporters)", value: transporters.length },
    { label: "路线 (routes)", value: routes.length },
    { label: "线路配置 (route configs)", value: routeConfigs.length },
  ];

  const handleAddNode = async () => {
    const values = await nodeForm.validateFields();
    const inventoryObj: FactoryNode["initialInventory"] = {};
    (values.inventory || []).forEach((row) => {
      if (row.material) inventoryObj[row.material] = Number(row.quantity) || 0;
    });
    const payload: FactoryNode = {
      id: values.id,
      type: values.type,
      dimensions: values.dimensions,
      angle: values.angle,
      initialInventory: values.type === "FU" && Object.keys(inventoryObj).length ? inventoryObj : undefined,
    };
    upsertNode(payload);
    nodeForm.resetFields();
  };

  const handleAddAssembly = async () => {
    const values = await assemblyForm.validateFields();
    const inputs: Assembly["inputs"] = {};
    (values.inputsList || []).forEach((row) => {
      if (row.material) inputs[row.material] = Number(row.quantity) || 0;
    });
    const payload: Assembly = {
      nodeId: values.nodeId,
      output: values.output,
      outputCount: values.outputCount ? Number(values.outputCount) : 1,
      processTime: values.processTime,
      inputs,
    };
    upsertAssembly(payload);
    assemblyForm.resetFields();
  };

  useEffect(() => {
    setRfNodes((prev) => {
      const prevMap = new Map(prev.map((n) => [n.id, n]));
      const next: RFNode[] = nodes.map((n, idx) => {
        const existing = prevMap.get(n.id);
        if (existing) return existing;
        const fallback: RFNode = {
          id: n.id,
          data: { label: `${n.id} (${n.type})` },
          position: { x: (idx % 4) * 180, y: Math.floor(idx / 4) * 160 },
          type: "default",
        };
        return fallback;
      });
      return next;
    });
  }, [nodes, setRfNodes]);

  useEffect(() => {
    const cfgMap = new Map(routeConfigs.map((c) => [c.routeId, c]));
    const edgeList: RFEdge[] = routes.map((r) => {
      const cfg = cfgMap.get(r.id);
      const labelParts = [r.materialId, cfg?.transporterId].filter(Boolean).join(" | ");
      return {
        id: r.id,
        source: r.fromNodeId,
        target: r.toNodeId,
        label: labelParts || "route",
        animated: false,
      };
    });
    setRfEdges(edgeList);
  }, [routes, routeConfigs, setRfEdges]);

  const handleConnect = (connection: Connection) => {
    if (!connection.source || !connection.target) return;
    const src = nodes.find((n) => n.id === connection.source);
    const tgt = nodes.find((n) => n.id === connection.target);
    if (src?.type === "Obstacle" || tgt?.type === "Obstacle") {
      message.warning("路线仅支持在 FU 之间连线");
      return;
    }
    setPendingConn(connection);
    setRouteModalOpen(true);
  };

  const handleRouteSubmit = async () => {
    if (!pendingConn?.source || !pendingConn?.target) {
      setRouteModalOpen(false);
      return;
    }
    const values = await routeForm.validateFields();
    const routeId = `route-${Date.now()}`;
    addRoute(
      {
        id: routeId,
        fromNodeId: pendingConn.source,
        toNodeId: pendingConn.target,
        materialId: values.materialId,
      },
      {
        routeId,
        transporterId: values.transporterId,
        batchSize: values.batchSize,
        travelTime: 0,
      }
    );
    setRfEdges((eds) =>
      addEdge(
        {
          id: routeId,
          source: pendingConn.source!,
          target: pendingConn.target!,
          label: `${values.materialId} | ${values.transporterId}`,
        },
        eds
      )
    );
    setRouteModalOpen(false);
    setPendingConn(null);
    routeForm.resetFields();
  };

  const materialOptions = useMemo(() => materials.map((m) => ({ label: m.id, value: m.id })), [materials]);
  const transporterOptions = useMemo(() => transporters.map((t) => ({ label: t.id, value: t.id })), [transporters]);
  const fuOptions = useMemo(() => fuNodes.map((n) => ({ label: `${n.id} (${n.type})`, value: n.id })), [fuNodes]);
  const inventoryPreview = useMemo(
    () =>
      fuNodes.flatMap((n) =>
        Object.entries(n.initialInventory || {}).map(([mat, qty]) => ({
          node: n.id,
          material: mat,
          qty,
        }))
      ),
    [fuNodes]
  );
  const routeCfgMap = useMemo(() => new Map(routeConfigs.map((c) => [c.routeId, c])), [routeConfigs]);

  const handleAddMaterial = async () => {
    const v = await materialForm.validateFields();
    addMaterial(v);
    materialForm.resetFields();
  };

  const handleAddTransporter = async () => {
    try {
      const v = await transporterForm.validateFields();
      addTransporter({
        ...v,
        speed: Number(v.speed),
        count: Number(v.count),
      });
      transporterForm.resetFields();
    } catch (err) {
      // antd validation rejection (e.g. speed/count empty)
      message.warning("请先填写完整运输工具的速度和数量");
    }
  };

  const buildLayoutConfig = (): LayoutConfig => {
    const fus = fuNodes.map((n) => ({
      id: n.id,
      label: n.id,
      width: n.dimensions.width,
      height: n.dimensions.length, // keep height for兼容
      length: n.dimensions.length, // 兼容后端 loader 需要 length
      notch_length: n.dimensions.notchLength,
      notch_width: n.dimensions.notchWidth,
      movable: true,
      angle: n.angle,
    }));
    const obstacles = obstacleNodes.map((n) => ({
      id: n.id,
      label: n.id,
      width: n.dimensions.width,
      height: n.dimensions.length,
      length: n.dimensions.length,
      notch_length: n.dimensions.notchLength,
      notch_width: n.dimensions.notchWidth,
      movable: true,
      angle: n.angle,
    }));
    const size = { width: canvasSize.width, height: canvasSize.height };
    return {
      // 兼容老后端需要的 canvas 字段，同时保留新版 factory
      canvas: size,
      factory: { length: size.width, width: size.height, grid_spacing: 1 },
      fus,
      obstacles,
    };
  };

  const buildFactoryConfig = (): FactoryConfig => {
    const meta = metaForm.getFieldsValue();
    const initial_inventory: FactoryConfig["initial_inventory"] = {};
    fuNodes.forEach((n) => {
      if (n.initialInventory) initial_inventory[n.id] = n.initialInventory;
    });
    const assembliesMapped = assemblies.map((a) => ({
      station: a.nodeId,
      inputs: a.inputs,
      output: a.output,
      process_time: a.processTime,
    }));
    const cfgMap = new Map(routeConfigs.map((c) => [c.routeId, c]));
    const routesMapped = routes
      .map((r) => {
        const cfg = cfgMap.get(r.id);
        if (!cfg) return null;
        return {
          from: r.fromNodeId,
          to: r.toNodeId,
          material: r.materialId,
          batch_size: cfg.batchSize,
          travel_time: cfg.travelTime ?? 0,
          transporter_id: cfg.transporterId,
        };
      })
      .filter(Boolean) as FactoryConfig["routes"];

    return {
      initial_inventory,
      routes: routesMapped,
      assemblies: assembliesMapped as any,
      summary: { finished_node: meta.finishedNode || "", finished_material: meta.finishedMaterial || "" },
      monitor:
        (meta.monitors || []).map((m) => ({ node: m.node, material: m.material })).filter((m) => m.node && m.material) ||
        [],
      transporters: transporters,
      layout: INLINE_LAYOUT_NAME,
    };
  };

  const importLayoutConfig = async (file: File) => {
    const text = await file.text();
    const data = JSON.parse(text) as LayoutConfig;
    const fus = data.fus || [];
    const obstacles = data.obstacles || [];
    const newCanvas = (() => {
      if ((data as any).factory) {
        const f = (data as any).factory;
        return {
          width: Number(f.length) || canvasSize.width,
          height: Number(f.width) || canvasSize.height,
        };
      }
      if (data.canvas) {
        return {
          width: Number(data.canvas.width) || canvasSize.width,
          height: Number(data.canvas.height) || canvasSize.height,
        };
      }
      return canvasSize;
    })();
    setCanvasSize(newCanvas.width, newCanvas.height);
    const nodeMap = new Map<string, FactoryNode>();
    nodes.forEach((n) => nodeMap.set(n.id, n));
    fus.forEach((f) => {
      const existing = nodeMap.get(f.id);
      nodeMap.set(f.id, {
        id: f.id,
        type: "FU",
        dimensions: {
          length: f.height ?? (f as any).length ?? existing?.dimensions.length ?? 10,
          width: f.width ?? existing?.dimensions.width ?? 10,
          notchLength: f.notch_length ?? existing?.dimensions.notchLength,
          notchWidth: f.notch_width ?? existing?.dimensions.notchWidth,
        },
        angle: f.angle ?? existing?.angle,
        initialInventory: existing?.initialInventory,
      });
    });
    obstacles.forEach((o) => {
      const existing = nodeMap.get(o.id);
      nodeMap.set(o.id, {
        id: o.id,
        type: "Obstacle",
        dimensions: {
          length: o.height ?? (o as any).length ?? existing?.dimensions.length ?? 10,
          width: o.width ?? existing?.dimensions.width ?? 10,
          notchLength: o.notch_length ?? existing?.dimensions.notchLength,
          notchWidth: o.notch_width ?? existing?.dimensions.notchWidth,
        },
        angle: o.angle ?? existing?.angle,
      });
    });
    const snapshot: FactoryStateSnapshot = {
      materials,
      transporters,
      nodes: Array.from(nodeMap.values()),
      assemblies,
      routes,
      routeConfigs,
      canvasSize: newCanvas,
    };
    useFactoryStore.setState(snapshot);
    setLastImport({ type: "layout", name: file.name });
    message.success("布局已导入至建模");
  };

  const importFactoryConfig = async (file: File) => {
    const text = await file.text();
    const data = JSON.parse(text) as FactoryConfig;
    const nodesMap = new Map<string, FactoryNode>();
    // keep existing障碍物
    nodes
      .filter((n) => n.type === "Obstacle")
      .forEach((n) => nodesMap.set(n.id, n));

    const addNodeIfMissing = (id: string) => {
      if (!nodesMap.has(id)) nodesMap.set(id, { id, type: "FU", dimensions: { length: 10, width: 10 } });
    };
    Object.entries(data.initial_inventory || {}).forEach(([nodeId, inv]) => {
      addNodeIfMissing(nodeId);
      const node = nodesMap.get(nodeId)!;
      node.initialInventory = Object.fromEntries(
        Object.entries(inv as Record<string, number>).map(([m, q]) => [m, Number(q) || 0])
      );
    });
    (data.assemblies || []).forEach((a) => addNodeIfMissing(a.station));
    (data.routes || []).forEach((r) => {
      addNodeIfMissing(r.from);
      addNodeIfMissing(r.to);
    });
    const assembliesMapped: Assembly[] = (data.assemblies || []).map((a) => ({
      nodeId: a.station,
      inputs: a.inputs || {},
      output: a.output,
      processTime: a.process_time || 0,
      outputCount: 1,
    }));
    const routesMapped: Route[] = (data.routes || []).map((r: any, idx: number) => ({
      id: r.id || `r_${idx}`,
      fromNodeId: r.from,
      toNodeId: r.to,
      materialId: r.material,
    }));
    const routeCfgMapped: RouteConfiguration[] = (data.routes || []).map((r: any, idx: number) => ({
      routeId: r.id || `r_${idx}`,
      transporterId: r.transporter_id,
      batchSize: r.batch_size,
      travelTime: r.travel_time ?? 0,
    }));
    const materialSet = new Set<string>();
    routesMapped.forEach((r) => materialSet.add(r.materialId));
    assembliesMapped.forEach((a) => {
      materialSet.add(a.output);
      Object.keys(a.inputs || {}).forEach((m) => materialSet.add(m));
    });
    (data.monitor || []).forEach((m) => {
      materialSet.add(m.material);
      addNodeIfMissing(m.node);
    });
    if (data.summary?.finished_material) materialSet.add(data.summary.finished_material);

    const snapshot: FactoryStateSnapshot = {
      materials: Array.from(materialSet).map((id) => ({ id })),
      transporters:
        data.transporters?.map((t: any) => ({
          id: t.id,
          speed: Number(t.speed) || 0,
          count: Number(t.count) || 1,
        })) || [],
      nodes: Array.from(nodesMap.values()),
      assemblies: assembliesMapped,
      routes: routesMapped,
      routeConfigs: routeCfgMapped,
      canvasSize,
    };
    useFactoryStore.setState(snapshot);
    metaForm.setFieldsValue({
      finishedNode: data.summary?.finished_node,
      finishedMaterial: data.summary?.finished_material,
      monitors: (data.monitor || []).map((m) => ({ node: m.node, material: m.material })),
    });
    setLastImport({ type: "factory", name: file.name });
    message.success("工厂配置已导入至建模");
  };

  const saveToLocal = async () => {
    try {
      setSavingLocal(true);
      const layoutCfg = buildLayoutConfig();
      const factoryCfg = buildFactoryConfig();
      const download = (data: any, filename: string) => {
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = filename;
        link.click();
        URL.revokeObjectURL(url);
      };
      download(layoutCfg, "layout.json");
      download(factoryCfg, "factory.json");
      message.success("已导出 layout.json 和 factory.json");
    } catch (e) {
      message.error(e instanceof Error ? e.message : "导出失败");
    } finally {
      setSavingLocal(false);
    }
  };

  const handleSaveBackend = async () => {
    try {
      setSavingBackend(true);
      const layoutCfg = buildLayoutConfig();
      const factoryCfg = buildFactoryConfig();
      const formatErrors = (res: any) => {
        const errs = res?.data?.validation?.errors;
        if (Array.isArray(errs) && errs.length) return errs.join("; ");
        return res?.message;
      };
      const [resLayout, resFactory] = await Promise.all([
        configApi.saveLayoutConfig(layoutCfg),
        configApi.saveFactoryConfig(factoryCfg),
      ]);
      if (resLayout.code !== 0) throw new Error(formatErrors(resLayout) || "布局保存失败");
      if (resFactory.code !== 0) throw new Error(formatErrors(resFactory) || "工厂配置保存失败");
      message.success("布局与工厂配置已保存到后端");
    } catch (e) {
      message.error(e instanceof Error ? e.message : "保存失败");
    } finally {
      setSavingBackend(false);
    }
  };

  return (
    <Space direction="vertical" size="large" style={{ width: "100%" }}>
      <div>
        <Title level={3} style={{ margin: 0 }}>
          节点与物流（Builder）
        </Title>
        <Paragraph type="secondary">
          顺序：先定义节点（尺寸、初始库存），再定义工序（输入/输出、加工时间），然后在画布连线并配置物料、运输工具、批量。
        </Paragraph>
      </div>

      <Card
        title={
          <Space>
            <Text strong>导入 / 预览</Text>
            {lastImport ? (
              <Tag color="green">
                已导入{lastImport.type === "layout" ? "布局" : "工厂"}: {lastImport.name}
              </Tag>
            ) : (
              <Tag>未导入文件</Tag>
            )}
          </Space>
        }
      >
        <Space style={{ marginBottom: 12 }} wrap>
          <Button>
            <label style={{ cursor: "pointer" }}>
              导入 factory.json
              <input
                type="file"
                accept=".json"
                style={{ display: "none" }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) importFactoryConfig(file);
                  e.target.value = "";
                }}
              />
            </label>
          </Button>
          <Button>
            <label style={{ cursor: "pointer" }}>
              导入 layout.json
              <input
                type="file"
                accept=".json"
                style={{ display: "none" }}
                onChange={(e) => {
                  const file = e.target.files?.[0];
                  if (file) importLayoutConfig(file);
                  e.target.value = "";
                }}
              />
            </label>
          </Button>
          <Space size="small">
            <Text>工厂长</Text>
            <InputNumber
              min={1}
              value={canvasSize.width}
              onChange={(v) => setCanvasSize(Number(v) || canvasSize.width, canvasSize.height)}
            />
            <Text>宽</Text>
            <InputNumber
              min={1}
              value={canvasSize.height}
              onChange={(v) => setCanvasSize(canvasSize.width, Number(v) || canvasSize.height)}
            />
          </Space>
        </Space>
        <List
          size="small"
          header="节点/初始库存"
          dataSource={inventoryPreview}
          locale={{ emptyText: "无初始库存" }}
          renderItem={(item) => (
            <List.Item>
              <Text code>{item.node}</Text> · <Text>{item.material}</Text> · 数量 {item.qty}
            </List.Item>
          )}
        />
        <Divider />
        <List
          size="small"
          header="工序"
          dataSource={assemblies}
          locale={{ emptyText: "无工序" }}
          renderItem={(a) => (
            <List.Item>
              <Text code>{a.nodeId}</Text> · 产出 {a.output} (x{a.outputCount || 1}) · 加工时间 {a.processTime} · 投入：
              {Object.entries(a.inputs || {})
                .map(([m, q]) => `${m}x${q}`)
                .join(", ")}
            </List.Item>
          )}
        />
        <Divider />
        <List
          size="small"
          header="路线"
          dataSource={routes}
          locale={{ emptyText: "无路线" }}
          renderItem={(r) => {
            const cfg = routeCfgMap.get(r.id);
            return (
              <List.Item>
                <Text code>{r.fromNodeId}</Text> → <Text code>{r.toNodeId}</Text> · 物料 {r.materialId} · 批量 {cfg?.batchSize ?? "-"} · 车辆{" "}
                {cfg?.transporterId ?? "-"}
              </List.Item>
            );
          }}
        />
        <Divider />
        <List
          size="small"
          header="监控点"
          dataSource={metaForm.getFieldValue("monitors") || []}
          locale={{ emptyText: "无监控配置" }}
          renderItem={(m: any) => (
            <List.Item>
              <Text code={m?.node || "-"}></Text> · 物料 {m?.material || "-"}
            </List.Item>
          )}
        />
        <Divider />
        <Space direction="vertical">
          <Text>成品节点：{metaForm.getFieldValue("finishedNode") || "-"}</Text>
          <Text>成品物料：{metaForm.getFieldValue("finishedMaterial") || "-"}</Text>
        </Space>
      </Card>

      <Collapse defaultActiveKey={["overview"]}>
        <Panel header="当前资源概览" key="overview">
          <List
            dataSource={summary}
            renderItem={(item) => (
              <List.Item>
                <Text>{item.label}</Text>
                <Tag color="blue">{item.value}</Tag>
              </List.Item>
            )}
          />
          <Alert
            style={{ marginTop: 12 }}
            type="info"
            showIcon
            message="提示"
            description="当前为占位版画布，可继续拖拽连线配置路线。"
          />
        </Panel>

        <Panel header="基础资源（物料/运输工具）" key="resources">
          <Form layout="inline" form={materialForm} style={{ marginBottom: 12 }}>
            <Form.Item name="id" label="物料ID" rules={[{ required: true }]}>
              <Input placeholder="如 steel" />
            </Form.Item>
            <Form.Item name="color" label="颜色">
              <Input placeholder="可选" />
            </Form.Item>
            <Form.Item>
              <Button icon={<PlusOutlined />} type="primary" onClick={handleAddMaterial}>
                添加物料
              </Button>
            </Form.Item>
          </Form>
          <List
            size="small"
            bordered
            dataSource={materials}
            renderItem={(m) => (
              <List.Item>
                <Text code>{m.id}</Text> {m.color ? <Text type="secondary">· {m.color}</Text> : null}
              </List.Item>
            )}
          />
          <Divider />
          <Form layout="inline" form={transporterForm} style={{ marginBottom: 12 }}>
            <Form.Item name="id" label="车辆ID" rules={[{ required: true }]}>
              <Input placeholder="如 forklift" />
            </Form.Item>
            <Form.Item name="speed" label="速度" rules={[{ required: true, type: "number", min: 0 }]}>
              <InputNumber min={0} />
            </Form.Item>
            <Form.Item name="count" label="数量" rules={[{ required: true, type: "number", min: 1 }]}>
              <InputNumber min={1} />
            </Form.Item>
            <Form.Item>
              <Button icon={<PlusOutlined />} type="primary" onClick={handleAddTransporter}>
                添加车辆
              </Button>
            </Form.Item>
          </Form>
          <List
            size="small"
            bordered
            dataSource={transporters}
            renderItem={(t) => (
              <List.Item>
                <Text code>{t.id}</Text> <Text type="secondary">· 速度 {t.speed} · 数量 {t.count}</Text>
              </List.Item>
            )}
          />
        </Panel>

        <Panel header="节点构建（FU / Obstacle，支持缺角）" key="nodes">
          <Form layout="vertical" form={nodeForm}>
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name="id" label="节点 ID" rules={[{ required: true }]}>
                  <Input placeholder="如 store_A" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="type" label="类型" rules={[{ required: true }]}>
                  <Select
                    options={[
                      { label: "功能单元 (FU)", value: "FU" },
                      { label: "障碍物 (Obstacle)", value: "Obstacle" },
                    ]}
                    placeholder="选择节点类型"
                  />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name={["dimensions", "length"]} label="长度" rules={[{ required: true, type: "number", min: 0.1 }]}>
                  <InputNumber min={0.1} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name={["dimensions", "width"]} label="宽度" rules={[{ required: true, type: "number", min: 0.1 }]}>
                  <InputNumber min={0.1} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name={["dimensions", "notchLength"]} label="缺角长 (可选)" rules={[{ type: "number", min: 0 }]}>
                  <InputNumber min={0} style={{ width: "100%" }} placeholder="缺角长度" />
                </Form.Item>
              </Col>
            </Row>
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name={["dimensions", "notchWidth"]} label="缺角宽 (可选)" rules={[{ type: "number", min: 0 }]}>
                  <InputNumber min={0} style={{ width: "100%" }} placeholder="缺角宽度" />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="angle" label="角度 (可选)" rules={[{ type: "number" }]}>
                  <InputNumber min={0} style={{ width: "100%" }} placeholder="0-360" />
                </Form.Item>
              </Col>
            </Row>
            {nodeTypeWatch !== "Obstacle" && (
              <Form.List name="inventory">
                {(fields, { add, remove }) => (
                  <Space direction="vertical" style={{ width: "100%" }}>
                    <Text strong>初始库存（仅 FU，按物料）</Text>
                    {fields.map(({ key, name }) => (
                      <Row gutter={12} key={key}>
                        <Col span={10}>
                          <Form.Item name={[name, "material"]} rules={[{ required: true }]}>
                            <Select mode="tags" options={materialOptions} placeholder="物料" />
                          </Form.Item>
                        </Col>
                        <Col span={10}>
                          <Form.Item name={[name, "quantity"]} rules={[{ required: true, type: "number", min: 0 }]}>
                            <InputNumber min={0} style={{ width: "100%" }} placeholder="数量" />
                          </Form.Item>
                        </Col>
                        <Col span={4}>
                          <Button danger onClick={() => remove(name)} block>
                            删除
                          </Button>
                        </Col>
                      </Row>
                    ))}
                    <Button type="dashed" icon={<PlusOutlined />} onClick={() => add()}>
                      添加库存项
                    </Button>
                  </Space>
                )}
              </Form.List>
            )}
            <Space style={{ marginTop: 12 }}>
              <Button type="primary" icon={<PlusOutlined />} onClick={handleAddNode}>
                保存/添加节点
              </Button>
              <Text type="secondary">保存即覆盖同 ID 节点</Text>
            </Space>
          </Form>
        </Panel>

        <Panel header="工序配置（绑定节点）" key="assemblies">
          <Form layout="vertical" form={assemblyForm}>
            <Row gutter={16}>
              <Col span={8}>
                <Form.Item name="nodeId" label="节点" rules={[{ required: true }]}>
                  <Select showSearch placeholder="选择节点 (仅 FU)" options={fuOptions} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="output" label="产出物料" rules={[{ required: true }]}>
                  <Select options={materialOptions} placeholder="选择或输入" showSearch allowClear />
                </Form.Item>
              </Col>
              <Col span={4}>
                <Form.Item
                  name="outputCount"
                  label="产出数量"
                  initialValue={1}
                  rules={[{ required: true, type: "number", min: 0.0001 }]}
                >
                  <InputNumber min={0.0001} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
              <Col span={8}>
                <Form.Item name="processTime" label="加工时间" rules={[{ required: true, type: "number", min: 0 }]}>
                  <InputNumber min={0} style={{ width: "100%" }} />
                </Form.Item>
              </Col>
            </Row>
            <Form.List name="inputsList">
              {(fields, { add, remove }) => (
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Text strong>输入物料 (配方)</Text>
                  {fields.map(({ key, name }) => (
                    <Row gutter={12} key={key}>
                      <Col span={10}>
                        <Form.Item name={[name, "material"]} rules={[{ required: true }]}>
                          <Select options={materialOptions} placeholder="物料" showSearch allowClear />
                        </Form.Item>
                      </Col>
                      <Col span={10}>
                        <Form.Item name={[name, "quantity"]} rules={[{ required: true, type: "number", min: 0.0001 }]}>
                          <InputNumber min={0.0001} style={{ width: "100%" }} placeholder="数量" />
                        </Form.Item>
                      </Col>
                      <Col span={4}>
                        <Button danger onClick={() => remove(name)} block>
                          删除
                        </Button>
                      </Col>
                    </Row>
                  ))}
                  <Button type="dashed" icon={<PlusOutlined />} onClick={() => add()}>
                    添加输入
                  </Button>
                </Space>
              )}
            </Form.List>
            <Space style={{ marginTop: 12 }}>
              <Button type="primary" icon={<PlusOutlined />} onClick={handleAddAssembly}>
                保存/添加工序
              </Button>
              <Text type="secondary">同一节点再次保存即覆盖其工序</Text>
            </Space>
          </Form>
        </Panel>

        <Panel header="成品与监控" key="meta">
          <Form layout="vertical" form={metaForm}>
            <Row gutter={16}>
              <Col span={12}>
                <Form.Item name="finishedNode" label="成品节点 (finished_node)">
                  <Select showSearch placeholder="选择成品节点" options={fuOptions} allowClear />
                </Form.Item>
              </Col>
              <Col span={12}>
                <Form.Item name="finishedMaterial" label="成品物料 (finished_material)">
                  <Select mode="tags" placeholder="选择或输入成品物料" options={materialOptions} allowClear />
                </Form.Item>
              </Col>
            </Row>
            <Form.List name="monitors">
              {(fields, { add, remove }) => (
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Text strong>监控点 (monitor) — 节点 + 物料</Text>
                  {fields.map(({ key, name }) => (
                    <Row gutter={12} key={key}>
                      <Col span={10}>
                        <Form.Item name={[name, "node"]}>
                          <Select showSearch placeholder="节点" options={fuOptions} allowClear />
                        </Form.Item>
                      </Col>
                      <Col span={10}>
                        <Form.Item name={[name, "material"]}>
                          <Select mode="tags" options={materialOptions} placeholder="物料" allowClear />
                        </Form.Item>
                      </Col>
                      <Col span={4}>
                        <Button danger block onClick={() => remove(name)}>
                          删除
                        </Button>
                      </Col>
                    </Row>
                  ))}
                  <Button type="dashed" icon={<PlusOutlined />} onClick={() => add()}>
                    添加监控点
                  </Button>
                </Space>
              )}
            </Form.List>
          </Form>
        </Panel>
      </Collapse>

      <Card title="画布（React Flow，占位版）" bodyStyle={{ height: 400 }}>
        <div style={{ height: "100%" }}>
          <ReactFlow
            nodes={rfNodes}
            edges={rfEdges}
            onNodesChange={onNodesChange}
            onEdgesChange={onEdgesChange}
            onConnect={handleConnect}
            fitView
          >
            <MiniMap />
            <Controls />
            <Background />
          </ReactFlow>
        </div>
        <Space style={{ marginTop: 12 }}>
          <Button type="primary" onClick={handleSaveBackend} loading={savingBackend}>
            保存当前模型到后端
          </Button>
          <Button onClick={saveToLocal} loading={savingLocal}>
            导出为 JSON
          </Button>
          <Text type="secondary">保存/导出包含布局和工厂配置，布局名使用内嵌占位 {INLINE_LAYOUT_NAME}</Text>
        </Space>
      </Card>

      <Modal
        title="配置路线"
        open={routeModalOpen}
        onOk={handleRouteSubmit}
        onCancel={() => {
          setRouteModalOpen(false);
          setPendingConn(null);
        }}
        okText="保存路线"
      >
        <Form layout="vertical" form={routeForm}>
          <Form.Item label="物料" name="materialId" rules={[{ required: true }]}>
            <Select mode="tags" options={materialOptions} placeholder="选择或输入物料" />
          </Form.Item>
          <Form.Item label="运输工具" name="transporterId" rules={[{ required: true }]}>
            <Select options={transporterOptions} placeholder="选择车辆" />
          </Form.Item>
          <Form.Item label="批量 (batch size)" name="batchSize" rules={[{ required: true, type: "number", min: 1 }]}>
            <InputNumber min={1} style={{ width: "100%" }} />
          </Form.Item>
          {pendingConn?.source && pendingConn?.target && (
            <Alert
              type="info"
              showIcon
              message={`从 ${pendingConn.source} 到 ${pendingConn.target}`}
              style={{ marginTop: 8 }}
            />
          )}
        </Form>
      </Modal>
    </Space>
  );
};

export default BuilderPage;
