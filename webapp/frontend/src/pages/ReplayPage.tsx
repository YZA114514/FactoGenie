import { useEffect, useState } from "react";
import { Card, Typography, Form, InputNumber, Button, Space, message, Row, Col, Alert, Tag, Select, Descriptions, Progress } from "antd";
import { StepBackwardOutlined, StepForwardOutlined, ReloadOutlined } from "@ant-design/icons";
import { replayApi, trainingApi } from "../services/api";
import type { ActionHeatmap } from "../types";
import ReactECharts from "echarts-for-react";
import ReactFlow, { Background, Controls, MiniMap, Node as RFNode } from "reactflow";
import "reactflow/dist/style.css";

const { Title, Paragraph, Text } = Typography;

const ReplayPage = () => {
  const [form] = Form.useForm<{ projectId: string; episode: number }>();
  const [info, setInfo] = useState<any>(null);
  const [stepData, setStepData] = useState<any>(null);
  const [heatmap, setHeatmap] = useState<ActionHeatmap | null>(null);
  const [loading, setLoading] = useState(false);
  const [projectOptions, setProjectOptions] = useState<{ label: string; value: string }[]>([]);
  const [checkpointOptions, setCheckpointOptions] = useState<{ label: string; value: number }[]>([]);
  const [jumpStep, setJumpStep] = useState<number | null>(null);  // 跳转步骤输入值
  const [inventoryData, setInventoryData] = useState<any>(null);  // 物料量变化数据
  const [inventoryLoading, setInventoryLoading] = useState(false);

  useEffect(() => {
    trainingApi
      .listProjects(1, 50)
      .then((res) => {
        if (res.code === 0) {
          setProjectOptions((res.data.projects || []).map((p: any) => ({ label: `${p.name || p.id} (${p.status})`, value: p.id })));
        }
      })
      .catch(() => {});
  }, []);

  const loadCheckpoints = async (pid: string) => {
    try {
      const res = await trainingApi.getCheckpoints(pid);
      if (res.code === 0) {
        setCheckpointOptions((res.data || []).map((cp: any) => ({ label: `Ep ${cp.episode} ${cp.is_best ? "(best)" : ""}`, value: cp.episode })));
      } else {
        setCheckpointOptions([]);
      }
    } catch {
      setCheckpointOptions([]);
    }
  };

  const startReplay = async () => {
    const { projectId, episode } = await form.validateFields(["projectId", "episode"]);
    setLoading(true);
    try {
      const res = await replayApi.start(projectId, Number(episode));
      if (res.code !== 0) throw new Error(res.message || "启动失败");
      setInfo(res.data);
      message.success("回放已启动");
      await fetchStep(0);
      // 同时获取物料量变化数据（传递episode参数以使用对应检查点的布局）
      fetchInventoryData(projectId, Number(episode));
    } catch (e) {
      message.error(e instanceof Error ? e.message : "启动失败");
    } finally {
      setLoading(false);
    }
  };

  const fetchInventoryData = async (projectId: string, episode?: number) => {
    setInventoryLoading(true);
    try {
      const res = await replayApi.getInventoryChart(projectId, episode);
      if (res.code === 0) {
        setInventoryData(res.data);
      } else {
        console.warn("获取物料量数据失败:", res.message);
      }
    } catch (e) {
      console.error("获取物料量数据失败:", e);
    } finally {
      setInventoryLoading(false);
    }
  };

  const fetchStep = async (stepNum?: number) => {
    const projectId = form.getFieldValue("projectId");
    if (!projectId) return;
    // 优先使用传入的参数，否则使用跳转输入框的值，最后使用当前步骤
    const stepVal = stepNum !== undefined ? stepNum : (jumpStep ?? stepData?.step ?? 0);
    try {
      const res = await replayApi.step(projectId, Number(stepVal));
      if (res.code === 0) {
        setStepData(res.data);
        setHeatmap(res.data?.heatmap || null);
        setJumpStep(res.data?.step ?? null);  // 同步更新跳转输入框
      } else {
        message.warning(res.message || "跳转失败");
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : "获取步数据失败");
    }
  };

  const forward = async () => {
    const projectId = form.getFieldValue("projectId");
    if (!projectId) return message.warning("先启动回放");
    try {
      const res = await replayApi.forward(projectId);
      if (res.code === 0 && res.data) {
        const stepRes = await replayApi.step(projectId, res.data?.step ?? 0);
        if (stepRes.code === 0) {
          setStepData(stepRes.data);
          setHeatmap(stepRes.data?.heatmap || null);
        }
        setJumpStep(res.data?.step ?? null);
      } else {
        message.info(res.message || "已到末尾");
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : "前进一步失败");
    }
  };

  const backward = async () => {
    const projectId = form.getFieldValue("projectId");
    if (!projectId) return message.warning("先启动回放");
    try {
      const res = await replayApi.backward(projectId);
      if (res.code === 0 && res.data) {
        const stepRes = await replayApi.step(projectId, res.data?.step ?? 0);
        if (stepRes.code === 0) {
          setStepData(stepRes.data);
          setHeatmap(stepRes.data?.heatmap || null);
        }
        setJumpStep(res.data?.step ?? null);
      } else {
        message.info(res.message || "已在第一步");
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : "回退一步失败");
    }
  };

  const closeSession = async () => {
    const projectId = form.getFieldValue("projectId");
    if (!projectId) return;
    await replayApi.close(projectId).catch(() => {});
    setInfo(null);
    setStepData(null);
    setHeatmap(null);
    message.success("已关闭回放会话");
  };

  // 当前步骤和总步骤
  const currentStep = stepData?.step ?? 0;
  const totalSteps = stepData?.total_steps ?? info?.total_steps ?? 0;
  const progress = totalSteps > 0 ? ((currentStep / totalSteps) * 100) : 0;

  return (
    <Card>
      <Title level={3} style={{ marginTop: 0 }}>回放 / 热力图</Title>
      <Paragraph type="secondary">选择项目和检查点，逐步回放训练过程，查看每一步的布局和Q值热力图。</Paragraph>

      <Form layout="inline" form={form} style={{ marginBottom: 16 }}>
        <Form.Item name="projectId" label="项目" rules={[{ required: true }]}>
          <Select
            style={{ width: 260 }}
            placeholder="选择项目"
            showSearch
            allowClear
            options={projectOptions}
            onChange={(v) => {
              form.setFieldsValue({ projectId: v, episode: undefined });
              setInfo(null);
              setStepData(null);
              setHeatmap(null);
              setCheckpointOptions([]);
              if (v) loadCheckpoints(v);
            }}
          />
        </Form.Item>
        <Form.Item name="episode" label="检查点" rules={[{ required: true }]}>
          <Select style={{ width: 140 }} placeholder="选择Episode" options={checkpointOptions} />
        </Form.Item>
        <Form.Item>
          <Button type="primary" onClick={startReplay} loading={loading}>启动回放</Button>
        </Form.Item>
        <Form.Item>
          <Button onClick={closeSession}>关闭会话</Button>
        </Form.Item>
      </Form>

      {info && (
        <Card size="small" style={{ marginBottom: 16, background: "#f6ffed", border: "1px solid #b7eb8f" }}>
          <Row gutter={16} align="middle">
            <Col span={8}>
              <Text strong style={{ fontSize: 16 }}>当前步骤: {currentStep} / {totalSteps}</Text>
              <Progress percent={progress} size="small" showInfo={false} style={{ marginTop: 4 }} />
            </Col>
            <Col span={16}>
              <Space size="large">
                <InputNumber 
                  style={{ width: 80 }} 
                  placeholder="步骤" 
                  min={0}
                  max={totalSteps}
                  value={jumpStep}
                  onChange={(val) => setJumpStep(val)}
                  onPressEnter={() => fetchStep()}
                />
                <Button onClick={() => fetchStep()}>跳转</Button>
                <Button icon={<StepBackwardOutlined />} onClick={backward} disabled={currentStep <= 0}>上一步</Button>
                <Button type="primary" icon={<StepForwardOutlined />} onClick={forward} disabled={currentStep >= totalSteps}>下一步</Button>
                <Button icon={<ReloadOutlined />} onClick={() => fetchStep(currentStep)}>刷新</Button>
              </Space>
            </Col>
          </Row>
        </Card>
      )}

      {!info && <Alert type="info" showIcon message="请选择项目和检查点，然后点击「启动回放」开始实时推理回放" style={{ marginBottom: 16 }} />}

      <Row gutter={16}>
        {/* 左侧：实时推理区域 */}
        <Col span={11}>
          {/* 当前步骤详情 */}
          <Card 
            size="small" 
            title={
              <Space>
                <span style={{ color: '#1677ff', fontWeight: 600 }}>📊 当前步骤详情</span>
                <Tag color="blue">实时推理</Tag>
              </Space>
            }
            style={{ borderColor: '#91caff' }}
          >
            {stepData ? (
              <Descriptions column={1} size="small" bordered>
                <Descriptions.Item label="当前步骤">
                  <Text strong>{currentStep} / {totalSteps}</Text>
                </Descriptions.Item>
                <Descriptions.Item label="待摆放单元">
                  {stepData.current_unit ? (
                    <Tag color="processing">{stepData.current_unit.id || stepData.current_unit.name}</Tag>
                  ) : <Tag color="success">全部完成</Tag>}
                </Descriptions.Item>
                <Descriptions.Item label={`已摆放 (${stepData.placed_units?.length || 0})`}>
                  <div style={{ maxHeight: 60, overflowY: "auto" }}>
                    {stepData.placed_units && stepData.placed_units.length > 0 ? (
                      stepData.placed_units.map((p: any, i: number) => (
                        <Tag key={i} style={{ margin: 2 }} color="blue">
                          {p.unit_id || p.name || `单元${i}`}
                        </Tag>
                      ))
                    ) : (
                      <Text type="secondary">无</Text>
                    )}
                  </div>
                </Descriptions.Item>
                <Descriptions.Item label="模型选择">
                  {heatmap?.selected_action ? (
                    <Space direction="vertical" size={0}>
                      <Text code style={{ color: '#fa8c16' }}>
                        位置 ({heatmap.selected_action.x}, {heatmap.selected_action.y})
                      </Text>
                      <Text code>
                        角度 {heatmap.selected_action.angle}° · Q值 {heatmap.selected_action.q_value?.toFixed(4)}
                      </Text>
                    </Space>
                  ) : <Text type="secondary">无</Text>}
                </Descriptions.Item>
              </Descriptions>
            ) : (
              <div style={{ textAlign: 'center', padding: '20px 0', color: '#999' }}>
                请启动回放查看实时推理结果
              </div>
            )}
          </Card>

          {/* 实时回放进度 */}
          <Card 
            size="small" 
            title={
              <Space>
                <span style={{ color: '#1677ff', fontWeight: 600 }}>🔄 实时回放进度</span>
                <Tag color="blue">步骤 {currentStep}/{totalSteps}</Tag>
              </Space>
            }
            style={{ marginTop: 12, borderColor: '#91caff' }}
          >
            {stepData?.layout?.placed_units && stepData.layout.placed_units.length > 0 ? (
              <LayoutFlow key={`layout-${currentStep}`} layout={stepData.layout} height={240} />
            ) : (
              <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fafafa', borderRadius: 4 }}>
                <Text type="secondary">点击"下一步"开始逐步摆放</Text>
              </div>
            )}
          </Card>

          {/* 训练保存的最终布局 */}
          <Card 
            size="small" 
            title={
              <Space>
                <span style={{ color: '#52c41a', fontWeight: 600 }}>📁 训练保存的最终布局</span>
                <Tag color="green">参考对比</Tag>
              </Space>
            }
            style={{ marginTop: 12, borderColor: '#b7eb8f' }}
          >
            {stepData?.saved_layout?.placed_units && stepData.saved_layout.placed_units.length > 0 ? (
              <LayoutFlow key={`saved-layout-${form.getFieldValue('episode')}`} layout={stepData.saved_layout} height={240} />
            ) : (
              <div style={{ height: 240, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f6ffed', borderRadius: 4 }}>
                <Text type="secondary">暂无保存的布局数据</Text>
              </div>
            )}
          </Card>
        </Col>

        {/* 右侧：Q值热力图和物料量 */}
        <Col span={13}>
          {/* Q值热力图 */}
          <Card 
            size="small" 
            title={
              <Space>
                <span style={{ color: '#1677ff', fontWeight: 600 }}>🔥 Q值热力图</span>
                <Tag color="blue">实时计算</Tag>
                <Text type="secondary" style={{ fontSize: 12 }}>所有角度取最大值</Text>
              </Space>
            }
            style={{ borderColor: '#91caff' }}
          >
            {heatmap && heatmap.q_values && heatmap.selected_action ? (
              <QValueHeatmap heatmap={heatmap} />
            ) : (
              <div style={{ height: 380, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#fafafa', borderRadius: 4 }}>
                <Text type="secondary">启动回放后显示Q值热力图</Text>
              </div>
            )}
          </Card>

          {/* 物料量变化图 */}
          <Card 
            size="small" 
            title={
              <Space>
                <span style={{ color: '#52c41a', fontWeight: 600 }}>📈 物料量变化图</span>
                <Tag color="green">最终布局仿真</Tag>
              </Space>
            }
            style={{ marginTop: 12, borderColor: '#b7eb8f' }}
          >
            {inventoryLoading ? (
              <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                <Space direction="vertical" align="center">
                  <div className="ant-spin ant-spin-spinning"><span className="ant-spin-dot ant-spin-dot-spin"><i></i><i></i><i></i><i></i></span></div>
                  <Text type="secondary">正在运行仿真获取数据...</Text>
                </Space>
              </div>
            ) : inventoryData?.series?.length > 0 ? (
              <InventoryChart data={inventoryData} />
            ) : (
              <div style={{ height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f6ffed', borderRadius: 4 }}>
                <Text type="secondary">启动回放后自动加载物料量数据</Text>
              </div>
            )}
          </Card>
        </Col>
      </Row>
    </Card>
  );
};

export default ReplayPage;

// 布局快照ReactFlow组件（可交互、可拖拽、可缩放）
// 完全复用ResultsPage的渲染逻辑
const LayoutFlow = ({ layout, height = 400 }: { layout: any; height?: number }) => {
  if (!layout || !layout.placed_units || !Array.isArray(layout.placed_units) || layout.placed_units.length === 0) {
    return <Text type="secondary">暂无布局数据</Text>;
  }
  
  // 使用与ResultsPage相同的缩放计算
  const gridSize = layout.grid_size || [36, 18];
  const canvasW = gridSize[0];
  const canvasH = gridSize[1];
  const scale = 360 / Math.max(canvasW, canvasH, 1);
  
  // 完全复用ResultsPage的rotateCW函数
  const rotateCW = (dx: number, dy: number, angleDeg: number) => {
    const rad = (angleDeg * Math.PI) / 180;
    const cos = Math.cos(rad);
    const sin = Math.sin(rad);
    return { x: cos * dx + sin * dy, y: -sin * dx + cos * dy };
  };

  // 转换placed_units为ReactFlow节点（与ResultsPage完全一致的逻辑）
  const nodes: RFNode[] = layout.placed_units.map((u: any, idx: number) => {
    const baseX = u.x ?? (idx % 5) * (canvasW / 5);
    const baseY = u.y ?? Math.floor(idx / 5) * (canvasH / 5);
    // 使用length和width（与ResultsPage一致）
    const len = u.length || u.height || 10;  // length对应高度
    const wid = u.width || 10;  // width对应宽度
    const nl = u.notch_length || 0;  // 缺口长度
    const nw = u.notch_width || 0;  // 缺口宽度
    const angle = u.angle || 0;

    // 构建多边形（左下角为原点）- 支持缺口形状
    const pts: { x: number; y: number }[] = nl && nw
      ? [
          { x: 0, y: 0 },
          { x: len - nl, y: 0 },
          { x: len - nl, y: nw },
          { x: len, y: nw },
          { x: len, y: wid },
          { x: 0, y: wid },
        ]
      : [
          { x: 0, y: 0 },
          { x: len, y: 0 },
          { x: len, y: wid },
          { x: 0, y: wid },
        ];
    
    // 旋转多边形并应用缩放
    const rotated = pts.map((p) => {
      const r = rotateCW(p.x, p.y, angle);
      return { x: (baseX + r.x) * scale, y: (baseY + r.y) * scale };
    });
    
    // 计算边界框
    const minX = Math.min(...rotated.map((p) => p.x));
    const minY = Math.min(...rotated.map((p) => p.y));
    const maxX = Math.max(...rotated.map((p) => p.x));
    const maxY = Math.max(...rotated.map((p) => p.y));
    
    // 翻转y轴：环境的y向上，ReactFlow的y向下
    const rfMinY = canvasH * scale - maxY;
    const rfMaxY = canvasH * scale - minY;
    
    // 转换为相对坐标（翻转y轴后）
    const relPoly = rotated.map((p) => ({ 
      x: p.x - minX, 
      y: (canvasH * scale - p.y) - rfMinY  // 翻转y轴
    }));
    
    // 使用后端返回的typeLabel，或根据ID判断
    const typeLabel = u.typeLabel || (
      (u.unit_id || '').toLowerCase().includes('obstacle') || 
      (u.unit_id || '').toLowerCase().includes('cafeteria') ? 'Obstacle' : 'FU'
    );
    
    return {
      id: u.unit_id || `unit-${idx}`,
      type: "layoutNode",
      position: { x: minX + 10, y: rfMinY + 10 },  // 使用翻转后的y坐标
      data: {
        id: u.unit_id || u.label || `U${idx}`,
        typeLabel: typeLabel,
        width: maxX - minX,
        height: rfMaxY - rfMinY,
        polygon: relPoly,
      },
      draggable: false,
    };
  });

  // 自定义节点组件（与ResultsPage的LayoutNode完全一致）
  const LayoutNode = ({ data }: any) => {
    const { id, typeLabel, polygon, width, height } = data;
    const stroke = typeLabel === "FU" ? "#1677ff" : "#ff4d4f";
    const fill = typeLabel === "FU" ? "#e8f0ff" : "#ffeaea";
    const pathD = polygon && polygon.length
      ? `M ${polygon.map((p: any) => `${p.x} ${p.y}`).join(" L ")} Z`
      : `M0 0 H ${width} V ${height} H 0 Z`;
    
    return (
      <div style={{ width, height, position: "relative" }}>
        <svg width={width} height={height} style={{ position: "absolute", top: 0, left: 0 }}>
          <path d={pathD} fill={fill} stroke={stroke} strokeWidth={2} />
        </svg>
        <div style={{ position: "absolute", top: 4, left: 4, fontSize: 12, fontWeight: 600, color: "#000" }}>
          {id}
        </div>
      </div>
    );
  };

  // 工厂边界节点
  const BoundaryNode = ({ data }: any) => {
    const { width, height, label } = data;
    return (
      <div style={{ width, height, position: "relative", pointerEvents: "none" }}>
        <svg width={width} height={height} style={{ position: "absolute", top: 0, left: 0 }}>
          <rect x={0} y={0} width={width} height={height} fill="none" stroke="#333" strokeWidth={3} strokeDasharray="8 4" />
        </svg>
        <div style={{ position: "absolute", bottom: -20, left: 0, fontSize: 11, color: "#666" }}>{label}</div>
      </div>
    );
  };

  const factoryBoundaryNode = {
    id: "factory-boundary",
    position: { x: 10, y: 10 },
    data: { 
      width: canvasW * scale, 
      height: canvasH * scale,
      label: `工厂 ${canvasW}×${canvasH}`
    },
    type: "boundaryNode",
    draggable: false,
    selectable: false,
  };

  const nodeTypes = { layoutNode: LayoutNode, boundaryNode: BoundaryNode };

  return (
    <div style={{ height, border: "1px solid #d9d9d9", background: "#fafafa" }}>
      <ReactFlow
        nodes={[factoryBoundaryNode, ...nodes]}
        edges={[]}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2, maxZoom: 3, minZoom: 0.1 }}
        panOnDrag={true}
        zoomOnScroll={true}
        zoomOnPinch={true}
        preventScrolling={false}
        nodesDraggable={false}
        nodesConnectable={false}
      >
        <Background color="#f0f0f0" gap={scale} size={1} />
        <Controls />
        <MiniMap 
          nodeColor={(node) => node.data?.color || "#1677ff"}
          style={{ background: "#fff" }}
        />
      </ReactFlow>
    </div>
  );
};

// Q值热力图组件（使用ECharts绘制等高线图/热力图）
const QValueHeatmap = ({ heatmap }: { heatmap: ActionHeatmap }) => {
  // 合并四个角度的Q值，取每个位置的最大值
  const gridW = heatmap.grid_width || 0;
  const gridH = heatmap.grid_height || 0;
  const qValues = heatmap.q_values || [];
  
  // 计算每个位置四个角度的最大Q值
  const maxQGrid: number[][] = [];
  const bestAngleGrid: number[][] = []; // 记录每个位置的最佳角度

  for (let y = 0; y < gridH; y++) {
    const row: number[] = [];
    const angleRow: number[] = [];
    for (let x = 0; x < gridW; x++) {
      let maxQ = -Infinity;
      let bestAngle = 0;
      let hasValid = false;
      for (let angle = 0; angle < 4 && angle < qValues.length; angle++) {
        const val = qValues[angle]?.[y]?.[x];
        // 后端现在返回 null 表示无效动作
        if (val !== undefined && val !== null && val > -1e9) {
          if (val > maxQ) {
            maxQ = val;
            bestAngle = angle * 90; // 转换为角度值
          }
          hasValid = true;
        }
      }
      row.push(hasValid ? maxQ : -Infinity);
      angleRow.push(hasValid ? bestAngle : -1);
    }
    maxQGrid.push(row);
    bestAngleGrid.push(angleRow);
  }

  // 第一遍：收集有效Q值，计算范围
  let minVal = Infinity, maxVal = -Infinity;
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const v = maxQGrid[y][x];
      if (v !== -Infinity) {
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
      }
    }
  }
  
  // 如果全是无效值，给默认范围
  if (minVal === Infinity) {
    minVal = -1;
    maxVal = 0;
  }

  // 非线性变换函数：增强高低Q值的对比度
  // 将Q值归一化到[0,1]，应用幂函数变换，再映射回[0,1]
  // 使用 gamma = 2.5，让高Q值更突出，低Q值更压缩
  const gamma = 2.5;
  const range = maxVal - minVal;
  const transformQ = (q: number | null): number | null => {
    if (q === null) return null;
    if (range < 1e-9) return 0.5; // 避免除零
    const normalized = (q - minVal) / range; // [0, 1]
    const transformed = Math.pow(normalized, gamma); // 非线性变换
    return transformed; // 返回变换后的值 [0, 1]
  };

  // 转换为ECharts格式 [x, y, transformedValue, angle, originalValue]
  // transformedValue用于颜色映射，originalValue用于tooltip显示
  const data: [number, number, number | null, number, number | null][] = [];
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW; x++) {
      const v = maxQGrid[y][x];
      const angle = bestAngleGrid[y][x];
      const originalVal = v === -Infinity ? null : v;
      const transformedVal = transformQ(originalVal);
      data.push([x, y, transformedVal, angle, originalVal]);
    }
  }

  // 选中的动作位置
  const selected = heatmap.selected_action;

  // 图例范围：变换后的值范围是[0, 1]
  const visualMapMin = 0;
  const visualMapMax = 1;

  const option = {
    tooltip: {
      position: "top",
      backgroundColor: 'rgba(50, 50, 50, 0.95)',
      borderColor: '#333',
      borderWidth: 1,
      padding: [8, 12],
      textStyle: {
        color: '#fff',
        fontSize: 13,
      },
      formatter: (params: any) => {
        // 处理markPoint的tooltip
        if (params.componentType === 'markPoint') {
          const coord = params.data.coord;
          const qValue = params.data.value;
          const angle = params.data.angle;
          return `
            <div style="font-weight:600;color:#fa8c16;margin-bottom:4px">🎯 模型选择的动作</div>
            <div style="display:flex;justify-content:space-between;gap:16px">
              <span style="color:#aaa">位置</span>
              <span style="font-family:monospace">(${coord[0]}, ${coord[1]})</span>
            </div>
            <div style="display:flex;justify-content:space-between;gap:16px">
              <span style="color:#aaa">角度</span>
              <span style="font-family:monospace">${angle}°</span>
            </div>
            <div style="display:flex;justify-content:space-between;gap:16px">
              <span style="color:#aaa">Q值</span>
              <span style="font-family:monospace;color:#52c41a">${qValue?.toFixed(4) ?? 'N/A'}</span>
            </div>
          `;
        }
        // 处理heatmap数据点的tooltip
        // data格式: [x, y, transformedValue, angle, originalValue]
        const originalVal = params.data[4];
        const angle = params.data[3];
        if (originalVal === null) {
          return `
            <div style="color:#999">
              <div style="margin-bottom:4px">📍 位置 (${params.data[0]}, ${params.data[1]})</div>
              <div style="color:#ff7875">⛔ 无效动作</div>
            </div>
          `;
        }
        return `
          <div style="margin-bottom:4px">📍 位置 <span style="font-family:monospace">(${params.data[0]}, ${params.data[1]})</span></div>
          <div style="display:flex;justify-content:space-between;gap:16px">
            <span style="color:#aaa">最佳角度</span>
            <span style="font-family:monospace">${angle}°</span>
          </div>
          <div style="display:flex;justify-content:space-between;gap:16px">
            <span style="color:#aaa">Q值</span>
            <span style="font-family:monospace;color:#69b1ff">${originalVal.toFixed(4)}</span>
          </div>
        `;
      },
    },
    grid: { 
      top: 20, 
      right: 80, 
      bottom: 40, 
      left: 50,
      containLabel: false,
    },
    xAxis: { 
      type: "category", 
      data: Array.from({ length: gridW }, (_, i) => i),
      name: 'X',
      nameLocation: 'center',
      nameGap: 25,
      nameTextStyle: { color: '#666', fontSize: 12 },
      axisLine: { lineStyle: { color: '#ddd' } },
      axisTick: { show: false },
      axisLabel: { 
        color: '#888', 
        fontSize: 10,
        interval: (index: number) => index % 5 === 0, // 每5格显示一个标签
      },
      splitLine: { show: false },
    },
    yAxis: { 
      type: "category", 
      data: Array.from({ length: gridH }, (_, i) => i),
      name: 'Y',
      nameLocation: 'center',
      nameGap: 35,
      nameTextStyle: { color: '#666', fontSize: 12 },
      axisLine: { lineStyle: { color: '#ddd' } },
      axisTick: { show: false },
      axisLabel: { 
        color: '#888', 
        fontSize: 10,
        interval: (index: number) => index % 2 === 0, // 每2格显示一个标签
      },
      splitLine: { show: false },
    },
    visualMap: {
      min: visualMapMin,
      max: visualMapMax,
      calculable: true,
      orient: "vertical",
      right: 5,
      top: "center",
      itemWidth: 12,
      itemHeight: 140,
      precision: 3,
      textStyle: { color: '#666', fontSize: 11 },
      inRange: { 
        color: [
          '#f0f5ff',  // 最低 - 非常浅蓝
          '#adc6ff',  // 低
          '#597ef7',  // 中
          '#2f54eb',  // 高
          '#10239e',  // 最高 - 深蓝
        ] 
      },
      dimension: 2,
      // 图例显示真实Q值范围，将变换后的[0,1]映射回原始范围
      formatter: (value: number) => {
        const realValue = minVal + value * range;
        return realValue.toFixed(2);
      },
    },
    series: [
      {
        name: "Q值",
        type: "heatmap",
        data: data,
        itemStyle: {
          borderColor: '#fff',
          borderWidth: 0.5,
          borderRadius: 2,
        },
        emphasis: { 
          itemStyle: { 
            shadowBlur: 15, 
            shadowColor: "rgba(47, 84, 235, 0.5)",
            borderColor: '#2f54eb',
            borderWidth: 2,
          } 
        },
        markPoint: selected ? {
          data: [{
            coord: [selected.x, selected.y],
            symbolSize: 24,
            symbolOffset: [0, 0],
            itemStyle: { 
              color: '#fa8c16', 
              borderColor: '#fff', 
              borderWidth: 3,
              shadowBlur: 10,
              shadowColor: 'rgba(250, 140, 22, 0.6)',
            },
            value: selected.q_value,
            angle: selected.angle,
          }],
          symbol: "circle",
          animation: true,
          label: {
            show: true,
            formatter: '',
            position: 'inside',
            fontSize: 12,
            offset: [0, 1],
          },
        } : undefined,
      },
    ],
  };

  return (
    <div style={{ position: 'relative' }}>
      <ReactECharts option={option} style={{ height: 420 }} />
      <div style={{ 
        display: 'flex', 
        justifyContent: 'center', 
        alignItems: 'center',
        gap: 24,
        marginTop: 8,
        padding: '8px 16px',
        background: 'linear-gradient(to right, #f8faff, #fff, #f8faff)',
        borderRadius: 6,
        fontSize: 12,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 12, height: 12, background: 'linear-gradient(135deg, #f0f5ff, #10239e)', borderRadius: 2 }}></div>
          <span style={{ color: '#666' }}>Q值范围</span>
          <span style={{ fontFamily: 'monospace', color: '#2f54eb', fontWeight: 500 }}>
            [{minVal.toFixed(3)}, {maxVal.toFixed(3)}]
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 12, height: 12, background: '#fa8c16', borderRadius: '50%', border: '2px solid #fff', boxShadow: '0 0 4px rgba(250,140,22,0.5)' }}></div>
          <span style={{ color: '#666' }}>模型选择</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <div style={{ width: 12, height: 12, background: '#f5f5f5', borderRadius: 2, border: '1px solid #ddd' }}></div>
          <span style={{ color: '#999' }}>无效区域</span>
        </div>
      </div>
    </div>
  );
};

// 物料量变化图表组件
const InventoryChart = ({ data }: { data: { series: Array<{ name: string; data: Array<[number, number]> }>; duration: number } }) => {
  const colorPalette = [
    '#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de',
    '#3ba272', '#fc8452', '#9a60b4', '#ea7ccc', '#48b8d0'
  ];

  const option = {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        let result = `时间: ${params[0]?.axisValue?.toFixed(1) || 0}<br/>`;
        params.forEach((p: any) => {
          result += `${p.marker} ${p.seriesName}: ${p.value?.[1] ?? 0}<br/>`;
        });
        return result;
      },
    },
    legend: {
      type: 'scroll',
      top: 0,
      data: data.series.map(s => s.name),
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: 40,
      containLabel: true,
    },
    xAxis: {
      type: 'value',
      name: '时间',
      max: data.duration,
    },
    yAxis: {
      type: 'value',
      name: '数量',
    },
    series: data.series.map((s, idx) => ({
      name: s.name,
      type: 'line',
      step: 'end',
      data: s.data,
      showSymbol: false,
      lineStyle: { width: 2 },
      itemStyle: { color: colorPalette[idx % colorPalette.length] },
    })),
  };

  return <ReactECharts option={option} style={{ height: 300 }} />;
};
