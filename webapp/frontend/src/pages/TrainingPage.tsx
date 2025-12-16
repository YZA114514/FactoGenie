import { useMemo, useState, useEffect } from "react";
import {
  Card,
  Typography,
  Alert,
  Form,
  Input,
  Select,
  Button,
  Space,
  message,
  Row,
  Col,
} from "antd";
import { PlayCircleOutlined, StopOutlined, ReloadOutlined } from "@ant-design/icons";
import { useFactoryStore } from "../store/factoryStore";
import type { LayoutConfig, FactoryConfig } from "../types";
import { trainingApi } from "../services/api";

const { Title, Paragraph, Text } = Typography;

const TrainingPage = () => {
  const { 
    nodes, assemblies, routes, routeConfigs, materials, transporters, canvasSize,
    currentProjectId, trainingStatus, setCurrentProjectId, setTrainingStatus
  } = useFactoryStore();
  const [form] = Form.useForm<{
    name: string;
    description?: string;
    finishedNode?: string;
    finishedMaterial?: string;
    trainingParams?: string;
    projectId?: string;
  }>();
  const [loading, setLoading] = useState(false);

  // 页面加载时恢复之前的项目ID
  useEffect(() => {
    if (currentProjectId) {
      form.setFieldsValue({ projectId: currentProjectId });
      // 自动刷新状态
      refreshStatus(currentProjectId);
    }
  }, []);

  // 刷新状态的辅助函数
  const refreshStatus = async (projectId?: string) => {
    const id = projectId || form.getFieldValue("projectId") || currentProjectId;
    if (!id) return;
    try {
      const res = await trainingApi.getStatus(id);
      if (res.data) {
        setTrainingStatus(res.data);
      }
    } catch (e) {
      console.error("Failed to refresh status", e);
    }
  };

  const fuNodes = useMemo(() => nodes.filter((n) => n.type === "FU"), [nodes]);
  const fuOptions = fuNodes.map((n) => ({ label: n.id, value: n.id }));
  const materialOptions = materials.map((m) => ({ label: m.id, value: m.id }));

  const buildLayoutConfig = (): LayoutConfig => {
    const fus = nodes
      .filter((n) => n.type === "FU")
      .map((n) => ({
        id: n.id,
        label: n.id,
        width: n.dimensions.width,
        height: n.dimensions.length,
        length: n.dimensions.length, // env.loader 需要 length
        notch_length: n.dimensions.notchLength,
        notch_width: n.dimensions.notchWidth,
        movable: true,
        angle: n.angle,
        x: n.x,
        y: n.y,
      }));
    const obstacles = nodes
      .filter((n) => n.type === "Obstacle")
      .map((n) => ({
        id: n.id,
        label: n.id,
        width: n.dimensions.width,
        height: n.dimensions.length,
        length: n.dimensions.length,
        notch_length: n.dimensions.notchLength,
        notch_width: n.dimensions.notchWidth,
        movable: true,
        angle: n.angle,
        x: n.x,
        y: n.y,
      }));
    return {
      factory: { length: canvasSize.width, width: canvasSize.height, grid_spacing: 1 },
      canvas: { width: canvasSize.width, height: canvasSize.height }, // 兼容旧字段
      fus,
      obstacles,
    };
  };

  const buildFactoryConfig = (finishedNode?: string, finishedMaterial?: string): FactoryConfig => {
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
      .map((r, idx) => {
        const cfg = cfgMap.get(r.id);
        if (!cfg) return null;
        return {
          id: r.id || `r_${idx}`,
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
      summary: { finished_node: finishedNode || "", finished_material: finishedMaterial || "" },
      monitor: [],
      transporters: transporters,
      layout: "inline_layout.json",
    };
  };

  const handleCreateAndStart = async () => {
    try {
      const values = await form.validateFields(["name"]);
      setLoading(true);
      const tp = form.getFieldValue("trainingParams");
      const parsedParams = tp ? JSON.parse(tp) : {};
      const finishedNode = form.getFieldValue("finishedNode");
      const finishedMaterial = form.getFieldValue("finishedMaterial");

      const payload = {
        name: values.name,
        description: form.getFieldValue("description"),
        factory_config: buildFactoryConfig(finishedNode, finishedMaterial),
        layout_config: buildLayoutConfig(),
        constraints: undefined,
        training_params: parsedParams,
      };
      const resCreate = await trainingApi.createProject(payload);
      if (resCreate.code !== 0) throw new Error(resCreate.message || "创建项目失败");
      const projectId = resCreate.data.project_id;
      form.setFieldsValue({ projectId });
      setCurrentProjectId(projectId);  // 保存到全局状态
      message.success(`项目已创建：${projectId}`);
      const resStart = await trainingApi.startProject(projectId);
      if (resStart.code !== 0) throw new Error(resStart.message || "启动失败");
      message.success("训练已启动");
      await refreshStatus(projectId);
    } catch (e) {
      message.error(e instanceof Error ? e.message : "操作失败，检查训练参数 JSON 是否有效");
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    const projectId = form.getFieldValue("projectId") || currentProjectId;
    if (!projectId) {
      message.warning("请先创建/填写项目ID");
      return;
    }
    setLoading(true);
    try {
      await trainingApi.stopProject(projectId);
      message.success("训练已停止");
      await refreshStatus(projectId);
    } catch (e) {
      message.error(e instanceof Error ? e.message : "操作失败");
    } finally {
      setLoading(false);
    }
  };

  const handleLoadProject = async () => {
    const projectId = form.getFieldValue("projectId");
    if (!projectId) {
      message.warning("请输入项目ID");
      return;
    }
    setLoading(true);
    try {
      setCurrentProjectId(projectId);
      await refreshStatus(projectId);
      message.success("已加载项目状态");
    } catch (e) {
      message.error(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <Title level={3} style={{ marginTop: 0 }}>
        训练控制
      </Title>
      <Paragraph type="secondary">
        直接用当前 Builder 的配置创建项目并启动训练。训练参数可粘贴 JSON，不填则用后端默认。
      </Paragraph>

      <Form layout="vertical" form={form}>
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item name="name" label="项目名称" rules={[{ required: true }]}>
              <Input placeholder="如 demo-project" />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="projectId" label="项目ID（创建后回填，可用于控制/查询）">
              <Input placeholder="自动填充，可手动输入已存在ID" />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="description" label="描述">
              <Input placeholder="可选" />
            </Form.Item>
          </Col>
        </Row>
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item name="finishedNode" label="成品节点 (finished_node)">
              <Select showSearch allowClear placeholder="选择 FU" options={fuOptions} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="finishedMaterial" label="成品物料 (finished_material)">
              <Select showSearch allowClear options={materialOptions} placeholder="选择或输入" />
            </Form.Item>
          </Col>
        </Row>
        <Form.Item
          name="trainingParams"
          label="训练参数 JSON（可选，留空用后端默认）"
          tooltip='示例：{ "total_steps": 1000, "learning_rate": 0.0002 }'
        >
          <Input.TextArea rows={4} placeholder="{}" />
        </Form.Item>
      </Form>

      <Space style={{ marginBottom: 16 }} wrap>
        <Button type="primary" icon={<PlayCircleOutlined />} loading={loading} onClick={handleCreateAndStart}>
          创建并启动训练
        </Button>
        <Button danger icon={<StopOutlined />} disabled={loading} onClick={handleStop}>
          停止
        </Button>
        <Button icon={<ReloadOutlined />} onClick={() => refreshStatus()}>刷新状态</Button>
        <Button onClick={handleLoadProject}>加载已有项目</Button>
      </Space>

      {trainingStatus ? (
        <Alert
          showIcon
          type={trainingStatus.status === 'running' ? 'success' : trainingStatus.status === 'completed' ? 'info' : 'warning'}
          message={`当前状态: ${trainingStatus.status}`}
          description={
            <Space direction="vertical">
              <Text>项目ID: {currentProjectId || form.getFieldValue("projectId") || "-"}</Text>
              <Text>当前步数: {trainingStatus.current_step ?? "-"} / {trainingStatus.total_steps ?? "-"}</Text>
              <Text>当前回合: {trainingStatus.current_episode ?? "-"}</Text>
              <Text>最佳奖励: {trainingStatus.best_reward?.toFixed(4) ?? "-"}</Text>
            </Space>
          }
        />
      ) : (
        <Alert type="warning" showIcon message="尚未查询到状态，请创建项目或输入已有项目ID后点击加载" />
      )}

      <Alert
        type="info"
        showIcon
        style={{ marginTop: 16 }}
        message="提示"
        description="训练状态会在页面切换后保留。输入已有项目ID并点击「加载已有项目」可恢复查看之前的训练。"
      />
    </Card>
  );
};

export default TrainingPage;
