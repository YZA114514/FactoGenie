import { useMemo, useState, useEffect } from "react";
import {
  Card,
  Typography,
  Alert,
  Form,
  Input,
  InputNumber,
  Select,
  Button,
  Space,
  message,
  Row,
  Col,
  Collapse,
  Divider,
  Switch,
  Tooltip,
} from "antd";
import { PlayCircleOutlined, StopOutlined, ReloadOutlined, SyncOutlined, UndoOutlined, QuestionCircleOutlined } from "@ant-design/icons";
import { useFactoryStore } from "../store/factoryStore";
import type { LayoutConfig, FactoryConfig } from "../types";
import { trainingApi, calibrationApi } from "../services/api";

const { Title, Paragraph, Text } = Typography;
const { Panel } = Collapse;

// 默认训练参数
const DEFAULT_TRAINING_PARAMS = {
  total_steps: 150000,
  learning_rate: 0.00002,
  batch_size: 32,
  replay_size: 50000,
  replay_start_size: 5000,
  epsilon_start: 1.0,
  epsilon_final: 0.05,
  epsilon_decay_frames: 150000,
  sync_target_every: 4000,
  double_dqn: false,
  dueling: true,
  noisy_net: true,
  prioritized: true,
  simulation_duration: 2000,
  use_simulation: true,
  checkpoint_interval: 100,
  calibrate_episodes: 100,
  weights: {
    distance: 0.20,
    logistics: 0.30,
    flow: 0.20,
    throughput: 0.25,
    utilization: 0.05,
  },
  placement_order: 'logistics_intensity',
};

const TrainingPage = () => {
  const { 
    nodes, assemblies, routes, routeConfigs, materials, transporters, canvasSize,
    currentProjectId, trainingStatus, setCurrentProjectId, setTrainingStatus,
    constraints,
  } = useFactoryStore();
  const [form] = Form.useForm<{
    name: string;
    description?: string;
    finishedNode?: string;
    finishedMaterial?: string;
    projectId?: string;
  }>();
  const [loading, setLoading] = useState(false);
  const [calibrating, setCalibrating] = useState(false);
  
  // 训练参数状态（表单输入）
  const [trainingParams, setTrainingParams] = useState({ ...DEFAULT_TRAINING_PARAMS });

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

  // 重置训练参数为默认值
  const resetTrainingParams = () => {
    setTrainingParams({ ...DEFAULT_TRAINING_PARAMS });
    message.success("训练参数已重置为默认值");
  };

  // 更新单个训练参数
  const updateParam = (key: string, value: any) => {
    setTrainingParams((prev) => ({ ...prev, [key]: value }));
  };

  // 更新权重参数
  const updateWeight = (key: string, value: number) => {
    setTrainingParams((prev) => ({
      ...prev,
      weights: { ...prev.weights, [key]: value },
    }));
  };

  const buildLayoutConfig = (): LayoutConfig => {
    const fus = nodes
      .filter((n) => n.type === "FU")
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
        movable: true, // 默认都是可移动的
        angle: n.angle,
        x: n.x,
        y: n.y,
      }));
    
    // 从store读取固定/可移动障碍物配置（保留从JSON导入的配置）
    const fixedObstacleIds = constraints?.fixed_obstacles?.length > 0 
      ? constraints.fixed_obstacles 
      : [];  // 如果没有配置，默认为空（所有障碍物可移动）
    const movableObstacleIds = constraints?.movable_obstacles?.length > 0 
      ? constraints.movable_obstacles 
      : obstacles.map((o) => o.id).filter(id => !fixedObstacleIds.includes(id));
    
    // 默认需要贴墙的单元（优先使用store配置，否则自动识别dock类型）
    const defaultWallAttach = constraints?.default_wall_attach?.length > 0
      ? constraints.default_wall_attach
      : fus
          .filter((f) => f.id.toLowerCase().includes('dock') || f.id.toLowerCase().includes('rec') || f.id.toLowerCase().includes('ship'))
          .map((f) => f.id);
    
    return {
      factory: { length: canvasSize.width, width: canvasSize.height, grid_spacing: 1 },
      canvas: { width: canvasSize.width, height: canvasSize.height },
      fus,
      obstacles,
      constraints: {
        fixed_obstacles: fixedObstacleIds,
        movable_obstacles: movableObstacleIds,
        default_wall_attach: defaultWallAttach,
        fixed_positions: (constraints?.fixed_positions || []) as any,
        adjacency: (constraints?.adjacency || []).map(a => ({ ...a, direction: (a.direction || "any") as "any" | "horizontal" | "vertical" })) as any,
        wall_attach: (constraints?.wall_attach || []).map(w => ({ ...w, wall: (w.wall || "left") as "top" | "bottom" | "left" | "right" })) as any,
      } as any,
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
      const finishedNode = form.getFieldValue("finishedNode");
      const finishedMaterial = form.getFieldValue("finishedMaterial");

      const layoutConfig = buildLayoutConfig();
      const payload = {
        name: values.name,
        description: form.getFieldValue("description"),
        factory_config: buildFactoryConfig(finishedNode, finishedMaterial),
        layout_config: layoutConfig,
        constraints: layoutConfig.constraints as any,
        training_params: trainingParams as any, // 直接使用表单状态
      };
      const resCreate = await trainingApi.createProject(payload);
      if (resCreate.code !== 0) throw new Error(resCreate.message || "创建项目失败");
      const projectId = resCreate.data.project_id;
      form.setFieldsValue({ projectId });
      setCurrentProjectId(projectId);
      message.success(`项目已创建：${projectId}`);
      const resStart = await trainingApi.startProject(projectId);
      if (resStart.code !== 0) throw new Error(resStart.message || "启动失败");
      message.success("训练已启动");
      refreshStatus(projectId).catch(err => {
        console.warn("Failed to refresh status immediately:", err);
      });
    } catch (e) {
      message.error(e instanceof Error ? e.message : "操作失败");
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

  const handleForceRecalibrate = async () => {
    try {
      setCalibrating(true);
      const finishedNode = form.getFieldValue("finishedNode");
      const finishedMaterial = form.getFieldValue("finishedMaterial");
      
      const factoryConfig = buildFactoryConfig(finishedNode, finishedMaterial);
      const layoutConfig = buildLayoutConfig();
      
      message.info("开始强制重新校准指标边界，这可能需要几分钟...");
      
      // 减少校准episode数以加快速度（50个episode通常足够）
      const res = await calibrationApi.runCalibration({
        factory_config: factoryConfig,
        layout_config: layoutConfig,
        n_episodes: 50, // 减少到50个episode以加快速度
        simulation_duration: 2000,
        force_recalibrate: true, // 强制重新校准
      });
      
      if (res.code !== 0) {
        throw new Error(res.message || "校准失败");
      }
      
      message.success(
        `校准完成！指标边界已更新。配置哈希: ${res.data?.config_hash || "N/A"}`,
        5
      );
      
      // 显示校准结果
      if (res.data?.bounds) {
        const bounds = res.data.bounds;
        console.log("校准结果:", bounds);
        message.info(
          `距离: [${bounds.distance?.best?.toFixed(2) || "N/A"}, ${bounds.distance?.worst?.toFixed(2) || "N/A"}] | ` +
          `物流: [${bounds.logistics?.best?.toFixed(2) || "N/A"}, ${bounds.logistics?.worst?.toFixed(2) || "N/A"}] | ` +
          `吞吐: [${bounds.throughput?.best?.toFixed(2) || "N/A"}, ${bounds.throughput?.worst?.toFixed(2) || "N/A"}]`,
          8
        );
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : "校准失败");
    } finally {
      setCalibrating(false);
    }
  };

  return (
    <Card>
      <Title level={3} style={{ marginTop: 0 }}>
        训练控制
      </Title>
      <Paragraph type="secondary">
        使用当前 Builder 的配置创建项目并启动训练。在下方设置训练参数，或使用默认值。
      </Paragraph>

      <Form layout="vertical" form={form}>
        <Row gutter={16}>
          <Col span={8}>
            <Form.Item name="name" label="项目名称" rules={[{ required: true }]}>
              <Input placeholder="如 demo-project" />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="projectId" label="项目ID（创建后回填）">
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
            <Form.Item name="finishedNode" label="成品节点">
              <Select showSearch allowClear placeholder="选择 FU" options={fuOptions} />
            </Form.Item>
          </Col>
          <Col span={8}>
            <Form.Item name="finishedMaterial" label="成品物料">
              <Select showSearch allowClear options={materialOptions} placeholder="选择或输入" />
            </Form.Item>
          </Col>
        </Row>
      </Form>

      <Collapse defaultActiveKey={["params"]} style={{ marginBottom: 16 }}>
        <Panel 
          header={
            <Space>
              <span>训练参数设置</span>
              <Button size="small" icon={<UndoOutlined />} onClick={(e) => { e.stopPropagation(); resetTrainingParams(); }}>
                重置默认值
              </Button>
            </Space>
          } 
          key="params"
        >
          <Divider orientation="left">基础参数</Divider>
          <Row gutter={[16, 12]}>
            <Col span={6}>
              <Text>总训练步数</Text>
              <Tooltip title="训练的总步数，更多步数通常能获得更好的结果">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={1000}
                max={10000000}
                step={10000}
                value={trainingParams.total_steps}
                onChange={(v) => updateParam('total_steps', v || 150000)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={6}>
              <Text>学习率</Text>
              <Tooltip title="神经网络的学习率，较小的值训练更稳定但更慢">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={0.000001}
                max={0.01}
                step={0.00001}
                value={trainingParams.learning_rate}
                onChange={(v) => updateParam('learning_rate', v || 0.00002)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={6}>
              <Text>批量大小</Text>
              <Tooltip title="每次训练使用的样本数量">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={8}
                max={256}
                step={8}
                value={trainingParams.batch_size}
                onChange={(v) => updateParam('batch_size', v || 32)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={6}>
              <Text>检查点间隔</Text>
              <Tooltip title="每隔多少episode保存一次检查点，0表示只保存最终模型">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={0}
                max={10000}
                step={50}
                value={trainingParams.checkpoint_interval}
                onChange={(v) => updateParam('checkpoint_interval', v || 100)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
          </Row>

          <Divider orientation="left">探索参数</Divider>
          <Row gutter={[16, 12]}>
            <Col span={6}>
              <Text>初始探索率 (ε)</Text>
              <Tooltip title="训练初期随机探索的概率，通常设为1.0">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={0}
                max={1}
                step={0.1}
                value={trainingParams.epsilon_start}
                onChange={(v) => updateParam('epsilon_start', v ?? 1.0)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={6}>
              <Text>最终探索率 (ε)</Text>
              <Tooltip title="训练后期保持的最小探索概率">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={0}
                max={1}
                step={0.01}
                value={trainingParams.epsilon_final}
                onChange={(v) => updateParam('epsilon_final', v ?? 0.05)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={6}>
              <Text>探索衰减步数</Text>
              <Tooltip title="探索率从初始值衰减到最终值所需的步数">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={1000}
                max={1000000}
                step={10000}
                value={trainingParams.epsilon_decay_frames}
                onChange={(v) => updateParam('epsilon_decay_frames', v || 150000)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
          </Row>

          <Divider orientation="left">经验回放</Divider>
          <Row gutter={[16, 12]}>
            <Col span={6}>
              <Text>回放缓冲区大小</Text>
              <Tooltip title="存储历史经验的缓冲区容量">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={1000}
                max={500000}
                step={10000}
                value={trainingParams.replay_size}
                onChange={(v) => updateParam('replay_size', v || 50000)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={6}>
              <Text>回放启动大小</Text>
              <Tooltip title="开始训练前需要收集的最少经验数量">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={100}
                max={50000}
                step={1000}
                value={trainingParams.replay_start_size}
                onChange={(v) => updateParam('replay_start_size', v || 5000)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={6}>
              <Text>目标网络同步频率</Text>
              <Tooltip title="每隔多少步同步一次目标网络">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={100}
                max={20000}
                step={500}
                value={trainingParams.sync_target_every}
                onChange={(v) => updateParam('sync_target_every', v || 4000)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
          </Row>

          <Divider orientation="left">DQN增强功能</Divider>
          <Row gutter={[16, 12]}>
            <Col span={6}>
              <Space>
                <Switch checked={trainingParams.double_dqn} onChange={(v) => updateParam('double_dqn', v)} />
                <Text>Double DQN</Text>
                <Tooltip title="使用Double DQN减少Q值过估计"><QuestionCircleOutlined style={{ color: '#999' }} /></Tooltip>
              </Space>
            </Col>
            <Col span={6}>
              <Space>
                <Switch checked={trainingParams.dueling} onChange={(v) => updateParam('dueling', v)} />
                <Text>Dueling DQN</Text>
                <Tooltip title="使用Dueling架构分离状态价值和动作优势"><QuestionCircleOutlined style={{ color: '#999' }} /></Tooltip>
              </Space>
            </Col>
            <Col span={6}>
              <Space>
                <Switch checked={trainingParams.noisy_net} onChange={(v) => updateParam('noisy_net', v)} />
                <Text>Noisy Net</Text>
                <Tooltip title="使用噪声网络实现参数化探索"><QuestionCircleOutlined style={{ color: '#999' }} /></Tooltip>
              </Space>
            </Col>
            <Col span={6}>
              <Space>
                <Switch checked={trainingParams.prioritized} onChange={(v) => updateParam('prioritized', v)} />
                <Text>优先经验回放</Text>
                <Tooltip title="优先采样TD误差较大的经验"><QuestionCircleOutlined style={{ color: '#999' }} /></Tooltip>
              </Space>
            </Col>
          </Row>

          <Divider orientation="left">奖励权重（总和应为1.0）</Divider>
          <Row gutter={[16, 12]}>
            <Col span={4}>
              <Text>距离</Text>
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={trainingParams.weights.distance}
                onChange={(v) => updateWeight('distance', v ?? 0.2)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={4}>
              <Text>物流</Text>
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={trainingParams.weights.logistics}
                onChange={(v) => updateWeight('logistics', v ?? 0.3)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={4}>
              <Text>流向</Text>
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={trainingParams.weights.flow}
                onChange={(v) => updateWeight('flow', v ?? 0.2)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={4}>
              <Text>吞吐</Text>
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={trainingParams.weights.throughput}
                onChange={(v) => updateWeight('throughput', v ?? 0.25)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={4}>
              <Text>利用率</Text>
              <InputNumber
                min={0}
                max={1}
                step={0.05}
                value={trainingParams.weights.utilization}
                onChange={(v) => updateWeight('utilization', v ?? 0.05)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={4}>
              <Text type="secondary">
                当前总和: {(trainingParams.weights.distance + trainingParams.weights.logistics + 
                  trainingParams.weights.flow + trainingParams.weights.throughput + 
                  trainingParams.weights.utilization).toFixed(2)}
              </Text>
            </Col>
          </Row>

          <Divider orientation="left">仿真设置</Divider>
          <Row gutter={[16, 12]}>
            <Col span={6}>
              <Space>
                <Switch checked={trainingParams.use_simulation} onChange={(v) => updateParam('use_simulation', v)} />
                <Text>启用仿真</Text>
                <Tooltip title="是否在训练中使用仿真计算动态指标"><QuestionCircleOutlined style={{ color: '#999' }} /></Tooltip>
              </Space>
            </Col>
            <Col span={6}>
              <Text>仿真时长</Text>
              <InputNumber
                min={100}
                max={50000}
                step={500}
                value={trainingParams.simulation_duration}
                onChange={(v) => updateParam('simulation_duration', v || 2000)}
                style={{ width: "100%", marginTop: 4 }}
                disabled={!trainingParams.use_simulation}
              />
            </Col>
            <Col span={6}>
              <Text>校准回合数</Text>
              <Tooltip title="用于校准奖励指标边界的随机布局回合数">
                <QuestionCircleOutlined style={{ marginLeft: 4, color: '#999' }} />
              </Tooltip>
              <InputNumber
                min={10}
                max={500}
                step={10}
                value={trainingParams.calibrate_episodes}
                onChange={(v) => updateParam('calibrate_episodes', v || 100)}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
            <Col span={6}>
              <Text>摆放顺序策略</Text>
              <Select
                value={trainingParams.placement_order}
                onChange={(v) => updateParam('placement_order', v)}
                options={[
                  { label: "物流强度优先", value: "logistics_intensity" },
                  { label: "面积优先(降序)", value: "size_desc" },
                  { label: "默认顺序", value: "default" },
                ]}
                style={{ width: "100%", marginTop: 4 }}
              />
            </Col>
          </Row>
        </Panel>
      </Collapse>

      <Space style={{ marginBottom: 16 }} wrap>
        <Button type="primary" icon={<PlayCircleOutlined />} loading={loading} onClick={handleCreateAndStart}>
          创建并启动训练
        </Button>
        <Button danger icon={<StopOutlined />} disabled={loading} onClick={handleStop}>
          停止
        </Button>
        <Button icon={<ReloadOutlined />} onClick={() => refreshStatus()}>刷新状态</Button>
        <Button onClick={handleLoadProject}>加载已有项目</Button>
        <Button 
          icon={<SyncOutlined />} 
          loading={calibrating} 
          onClick={handleForceRecalibrate}
          title="强制重新计算奖励指标的上下界（基于当前配置）"
        >
          强制更新指标边界
        </Button>
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
