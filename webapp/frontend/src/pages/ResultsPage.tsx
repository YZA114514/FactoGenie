import { useEffect, useState } from "react";
import { Card, Typography, Form, Button, Space, List, Alert, Select, message, Tag, Row, Col, Tabs } from "antd";
import { resultsApi, trainingApi } from "../services/api";
import ReactFlow, { Background, Controls, MiniMap } from "reactflow";
import "reactflow/dist/style.css";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

const { Title, Paragraph, Text } = Typography;

const ResultsPage = () => {
  const [form] = Form.useForm<{ projectId: string }>();
  const [best, setBest] = useState<any>(null);
  const [layouts, setLayouts] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<{ metric?: string; values?: any[] }>({});
  const [losses, setLosses] = useState<{ values: { step: number; loss: number }[]; count: number }>({ values: [], count: 0 });
  const [rewardsCsv, setRewardsCsv] = useState<{ values: any[]; count: number }>({ values: [], count: 0 });
  const [loading, setLoading] = useState(false);
  const [projects, setProjects] = useState<{ label: string; value: string }[]>([]);
  const [projectInfo, setProjectInfo] = useState<any>(null);  // 项目信息（包含训练参数）

  useEffect(() => {
    // 预取项目列表供选择
    trainingApi
      .listProjects(1, 50)
      .then((res) => {
        if (res.code === 0) {
          const opts = (res.data.projects || []).map((p: any) => ({ label: `${p.name || p.id} (${p.status})`, value: p.id }));
          setProjects(opts);
        } else {
          message.warning(res.message || "加载项目列表失败");
        }
      })
      .catch(() => message.warning("加载项目列表失败"));
  }, []);

  const loadResults = async () => {
    const projectId = form.getFieldValue("projectId");
    if (!projectId) {
      message.warning("请先填写或选择项目ID");
      return;
    }
    try {
      setLoading(true);
      const [bestRes, layoutsRes, metricsRes, projectRes, lossesRes, rewardsCsvRes] = await Promise.allSettled([
        resultsApi.getBestLayout(projectId),
        resultsApi.getLayouts(projectId, 1, 20),
        resultsApi.getMetrics(projectId, "reward", 0),
        trainingApi.getProject(projectId),
        resultsApi.getLosses(projectId),
        resultsApi.getRewardsCsv(projectId),
      ]);
      if (bestRes.status === "fulfilled" && bestRes.value.code === 0) {
        setBest(bestRes.value.data);
      } else if (bestRes.status === "fulfilled") {
        setBest(null);
        message.info(bestRes.value.message || "暂无最佳布局");
      }
      if (layoutsRes.status === "fulfilled" && layoutsRes.value.code === 0) {
        const layoutsData = layoutsRes.value.data.layouts || [];
        setLayouts(layoutsData);
        if (layoutsData.length === 0) {
          message.info("暂无检查点。检查点会在训练达到 checkpoint_interval 设置的 episode 时保存（默认每 1000 个 episode）。");
        } else {
          console.log(`加载了 ${layoutsData.length} 个检查点`, layoutsData);
        }
      } else if (layoutsRes.status === "fulfilled") {
        setLayouts([]);
        const errorMsg = layoutsRes.value.message || "暂无检查点";
        message.warning(errorMsg);
        console.warn("获取检查点失败:", layoutsRes.value);
      } else if (layoutsRes.status === "rejected") {
        setLayouts([]);
        message.error("获取检查点失败: " + (layoutsRes.reason?.message || "未知错误"));
        console.error("获取检查点异常:", layoutsRes.reason);
      }
      if (metricsRes.status === "fulfilled" && metricsRes.value.code === 0) {
        setMetrics(metricsRes.value.data);
      } else if (metricsRes.status === "fulfilled") {
        setMetrics({});
      }
      // 处理项目信息
      if (projectRes.status === "fulfilled" && projectRes.value.code === 0) {
        setProjectInfo(projectRes.value.data);
      } else {
        setProjectInfo(null);
      }
      // 处理losses数据
      if (lossesRes.status === "fulfilled" && lossesRes.value.code === 0) {
        setLosses(lossesRes.value.data);
      } else {
        setLosses({ values: [], count: 0 });
      }
      // 处理rewards CSV数据
      if (rewardsCsvRes.status === "fulfilled" && rewardsCsvRes.value.code === 0) {
        setRewardsCsv(rewardsCsvRes.value.data);
      } else {
        setRewardsCsv({ values: [], count: 0 });
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Card>
      <Title level={3} style={{ marginTop: 0 }}>
        结果查看
      </Title>
      <Paragraph type="secondary">选择项目，查看最佳布局、指标曲线、历史布局。</Paragraph>

      <Form layout="inline" form={form} style={{ marginBottom: 12 }}>
        <Form.Item name="projectId" label="项目ID" style={{ minWidth: 320 }}>
          <Select
            showSearch
            allowClear
            placeholder="输入或选择项目ID"
            options={projects}
            onChange={() => {
              setBest(null);
              setLayouts([]);
              setMetrics({});
              setLosses({ values: [], count: 0 });
              setRewardsCsv({ values: [], count: 0 });
              setProjectInfo(null);
            }}
          />
        </Form.Item>
        <Form.Item>
          <Button type="primary" onClick={loadResults} loading={loading}>
            加载结果
          </Button>
        </Form.Item>
      </Form>

      {best ? (
        <Card
          size="small"
          title={`最佳布局（Episode ${best.episode}，reward ${best.reward ?? "-"})`}
          style={{ marginBottom: 12 }}
          extra={best.is_best ? <Tag color="green">best</Tag> : null}
        >
          <Row gutter={16}>
            <Col span={12}>
              <BestLayoutPreview layout={best.layout} />
            </Col>
            <Col span={12}>
              <List
                size="small"
                header="功能单元/障碍物"
                dataSource={[
                  ...(best.layout?.fus || []),
                  ...(best.layout?.obstacles || []),
                ].map((u: any) => ({
                  id: u.id,
                  type: (best.layout?.fus || []).some((x: any) => x.id === u.id) ? "FU" : "Obstacle",
                  width: u.width,
                  height: u.height ?? u.length,
                  x: u.x,
                  y: u.y,
                  angle: u.angle,
                }))}
                renderItem={(item: any) => (
                  <List.Item>
                    <Space>
                      <Tag color={item.type === "FU" ? "blue" : "volcano"}>{item.type}</Tag>
                      <Text code>{item.id}</Text>
                      <Text type="secondary">
                        {item.width} x {item.height} {item.x !== undefined ? `@ (${item.x}, ${item.y})` : ""}
                        {item.angle !== undefined ? ` · angle ${item.angle}` : ""}
                      </Text>
                    </Space>
                  </List.Item>
                )}
              />
            </Col>
          </Row>
        </Card>
      ) : (
        <Alert type="info" showIcon message="最佳布局：暂无数据" style={{ marginBottom: 12 }} />
      )}

      <Card size="small" title="指标曲线（reward）" style={{ marginBottom: 12 }}>
        {metrics.values && metrics.values.length ? (
          <Tabs
            defaultActiveKey="chart"
            items={[
              {
                key: "chart",
                label: "折线图",
                children: (
                  <div style={{ width: "100%", height: 300 }}>
                    <ResponsiveContainer>
                      <LineChart data={metrics.values} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="episode" label={{ value: "Episode", position: "insideBottom", offset: -5 }} />
                        <YAxis label={{ value: "Reward", angle: -90, position: "insideLeft" }} />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="value" stroke="#1677ff" name="Reward" dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ),
              },
              {
                key: "table",
                label: "数据列表",
                children: (
                  <div style={{ maxHeight: 300, overflowY: "auto" }}>
                    <List
                      size="small"
                      dataSource={metrics.values}
                      renderItem={(v: any) => (
                        <List.Item>
                          <Text code>Episode {v.episode}</Text>
                          <Text style={{ marginLeft: 12 }}>{typeof v.value === "number" ? v.value.toFixed(4) : v.value}</Text>
                        </List.Item>
                      )}
                    />
                  </div>
                ),
              },
            ]}
          />
        ) : (
          <Text type="secondary">暂无指标数据</Text>
        )}
      </Card>

      {/* Loss曲线 */}
      <Card size="small" title="训练损失曲线（Loss）" style={{ marginBottom: 12 }}>
        {losses.values && losses.values.length > 0 ? (
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={losses.values} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="step" label={{ value: "Step", position: "insideBottom", offset: -5 }} />
                <YAxis label={{ value: "Loss", angle: -90, position: "insideLeft" }} />
                <Tooltip />
                <Legend />
                <Line type="monotone" dataKey="loss" stroke="#ff4d4f" name="Loss" dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <Alert 
            type="info" 
            showIcon 
            message="暂无Loss数据" 
            description={
              <span>
                Loss数据会在训练开始后记录。训练需要先填充经验回放池（默认5000步），之后才会开始实际训练并产生Loss值。
                当前项目可能训练步数不足，或尚未开始实际训练。
              </span>
            }
          />
        )}
      </Card>

      {/* 详细奖励曲线（来自CSV，包含mean_reward和epsilon） */}
      {rewardsCsv.values && rewardsCsv.values.length > 0 && (
        <Card size="small" title="详细训练曲线（Reward + Epsilon）" style={{ marginBottom: 12 }}>
          <div style={{ width: "100%", height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={rewardsCsv.values} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="episode" label={{ value: "Episode", position: "insideBottom", offset: -5 }} />
                <YAxis yAxisId="left" label={{ value: "Reward", angle: -90, position: "insideLeft" }} />
                <YAxis yAxisId="right" orientation="right" domain={[0, 1]} label={{ value: "Epsilon", angle: 90, position: "insideRight" }} />
                <Tooltip />
                <Legend />
                <Line yAxisId="left" type="monotone" dataKey="reward" stroke="#1677ff" name="Reward" dot={false} />
                <Line yAxisId="left" type="monotone" dataKey="mean_reward" stroke="#52c41a" name="Mean Reward (100)" dot={false} strokeWidth={2} />
                <Line yAxisId="right" type="monotone" dataKey="epsilon" stroke="#faad14" name="Epsilon" dot={false} strokeDasharray="5 5" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* 训练参数配置 */}
      <Card size="small" title="训练参数配置" style={{ marginBottom: 12 }}>
        {projectInfo?.training_params ? (
          <>
            <Title level={5} style={{ marginTop: 0, marginBottom: 8 }}>基础参数</Title>
            <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Text type="secondary">总训练步数:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.total_steps?.toLocaleString() || '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">学习率:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.learning_rate || '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">批量大小:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.batch_size || '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">检查点间隔:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.checkpoint_interval || '-'}</Text>
              </Col>
            </Row>

            <Title level={5} style={{ marginBottom: 8 }}>探索参数</Title>
            <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Text type="secondary">Epsilon初始:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.epsilon_start ?? '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">Epsilon最终:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.epsilon_final ?? '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">探索衰减步数:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.epsilon_decay_frames?.toLocaleString() || '-'}</Text>
              </Col>
            </Row>

            <Title level={5} style={{ marginBottom: 8 }}>经验回放</Title>
            <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Text type="secondary">回放缓冲区大小:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.replay_size?.toLocaleString() || '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">回放启动大小:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.replay_start_size?.toLocaleString() || '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">目标网络同步频率:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.sync_target_every?.toLocaleString() || '-'}</Text>
              </Col>
            </Row>

            <Title level={5} style={{ marginBottom: 8 }}>DQN增强功能</Title>
            <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Text type="secondary">Double DQN:</Text>
                <Tag color={projectInfo.training_params.double_dqn ? 'green' : 'default'} style={{ marginLeft: 8 }}>
                  {projectInfo.training_params.double_dqn ? '开启' : '关闭'}
                </Tag>
              </Col>
              <Col span={6}>
                <Text type="secondary">Dueling DQN:</Text>
                <Tag color={projectInfo.training_params.dueling ? 'green' : 'default'} style={{ marginLeft: 8 }}>
                  {projectInfo.training_params.dueling ? '开启' : '关闭'}
                </Tag>
              </Col>
              <Col span={6}>
                <Text type="secondary">Noisy Net:</Text>
                <Tag color={projectInfo.training_params.noisy_net ? 'green' : 'default'} style={{ marginLeft: 8 }}>
                  {projectInfo.training_params.noisy_net ? '开启' : '关闭'}
                </Tag>
              </Col>
              <Col span={6}>
                <Text type="secondary">优先经验回放:</Text>
                <Tag color={projectInfo.training_params.prioritized ? 'green' : 'default'} style={{ marginLeft: 8 }}>
                  {projectInfo.training_params.prioritized ? '开启' : '关闭'}
                </Tag>
              </Col>
            </Row>

            <Title level={5} style={{ marginBottom: 8 }}>仿真设置</Title>
            <Row gutter={[16, 8]} style={{ marginBottom: 16 }}>
              <Col span={6}>
                <Text type="secondary">启用仿真:</Text>
                <Tag color={projectInfo.training_params.use_simulation ? 'green' : 'default'} style={{ marginLeft: 8 }}>
                  {projectInfo.training_params.use_simulation ? '是' : '否'}
                </Tag>
              </Col>
              <Col span={6}>
                <Text type="secondary">仿真时长:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.simulation_duration || '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">校准回合数:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.calibrate_episodes || '-'}</Text>
              </Col>
              <Col span={6}>
                <Text type="secondary">摆放顺序策略:</Text>
                <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.placement_order || '-'}</Text>
              </Col>
            </Row>

            {projectInfo.training_params.weights && (
              <>
                <Title level={5} style={{ marginBottom: 8 }}>奖励权重</Title>
                <Row gutter={[16, 8]}>
                  <Col span={4}>
                    <Text type="secondary">距离:</Text>
                    <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.weights.distance ?? '-'}</Text>
                  </Col>
                  <Col span={4}>
                    <Text type="secondary">物流:</Text>
                    <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.weights.logistics ?? '-'}</Text>
                  </Col>
                  <Col span={4}>
                    <Text type="secondary">流向:</Text>
                    <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.weights.flow ?? '-'}</Text>
                  </Col>
                  <Col span={4}>
                    <Text type="secondary">吞吐:</Text>
                    <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.weights.throughput ?? '-'}</Text>
                  </Col>
                  <Col span={4}>
                    <Text type="secondary">利用率:</Text>
                    <Text strong style={{ marginLeft: 8 }}>{projectInfo.training_params.weights.utilization ?? '-'}</Text>
                  </Col>
                </Row>
              </>
            )}
          </>
        ) : (
          <Text type="secondary">暂无训练参数数据</Text>
        )}
      </Card>

      <Card size="small" title={`布局/检查点历史 (${layouts.length} 条)`}>
        <div style={{ maxHeight: 400, overflowY: "auto" }}>
          <List
            size="small"
            dataSource={layouts}
            bordered
            locale={{ emptyText: "暂无检查点" }}
            renderItem={(item: any, index: number) => (
              <List.Item
                key={item.episode ?? index}
                style={{ background: item.is_best ? "#f6ffed" : undefined }}
              >
                <Space direction="vertical" style={{ width: "100%" }}>
                  <Space wrap>
                    <Tag color="blue">Episode {item.episode}</Tag>
                    <Text strong>reward: {typeof item.reward === "number" ? item.reward.toFixed(4) : (item.reward ?? "-")}</Text>
                    {item.is_best && <Tag color="green">最佳</Tag>}
                  </Space>
                  <Text type="secondary">
                    工厂尺寸：{item.layout?.factory?.length || item.layout?.canvas?.width || "-"} ×{" "}
                    {item.layout?.factory?.width || item.layout?.canvas?.height || "-"}
                  </Text>
                  <details style={{ marginTop: 4 }}>
                    <summary style={{ cursor: "pointer", color: "#1677ff" }}>查看布局 JSON</summary>
                    <pre style={{ whiteSpace: "pre-wrap", margin: 0, fontSize: 12, maxHeight: 200, overflowY: "auto", background: "#fafafa", padding: 8, borderRadius: 4 }}>
                      {JSON.stringify(item.layout, null, 2)}
                    </pre>
                  </details>
                </Space>
              </List.Item>
            )}
          />
        </div>
      </Card>
    </Card>
  );
};

export default ResultsPage;

// 简易布局预览组件（SVG）
const BestLayoutPreview = ({ layout }: { layout: any }) => {
  if (!layout || !layout.fus) return <Text type="secondary">暂无布局数据</Text>;

  const nodesRF = () => {
    const items = [
      ...(layout.fus || []).map((u: any) => ({ ...u, typeLabel: "FU" })),
      ...(layout.obstacles || []).map((u: any) => ({ ...u, typeLabel: "Obstacle" })),
    ];
    const canvasW = layout.factory?.length || layout.canvas?.width || 100;
    const canvasH = layout.factory?.width || layout.canvas?.height || 100;
    const scale = 360 / Math.max(canvasW, canvasH, 1);

    const rotateCW = (dx: number, dy: number, angleDeg: number) => {
      const rad = (angleDeg * Math.PI) / 180;
      const cos = Math.cos(rad);
      const sin = Math.sin(rad);
      return { x: cos * dx + sin * dy, y: -sin * dx + cos * dy };
    };

    return items.map((u: any, idx: number) => {
      const baseX = u.x ?? (idx % 5) * (canvasW / 5);
      const baseY = u.y ?? Math.floor(idx / 5) * (canvasH / 5);
      const len = u.length || u.height || 10;  // length对应高度
      const wid = u.width || 10;  // width对应宽度
      const nl = u.notch_length || 0;
      const nw = u.notch_width || 0;
      const angle = u.angle || 0;

      // 构建多边形（左下角为原点）
      // 注意：环境坐标系统y轴向上，ReactFlow坐标系统y轴向下
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
      
      // 旋转多边形
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
      // 转换公式：rfY = canvasH * scale - envY
      const rfMinY = canvasH * scale - maxY;
      const rfMaxY = canvasH * scale - minY;
      
      // 转换为相对坐标（翻转y轴后）
      const relPoly = rotated.map((p) => ({ 
        x: p.x - minX, 
        y: (canvasH * scale - p.y) - rfMinY  // 翻转y轴
      }));

      return {
        id: u.id || `node-${idx}`,
        position: { x: minX + 10, y: rfMinY + 10 },  // 使用翻转后的y坐标
        data: { id: u.id, typeLabel: u.typeLabel, polygon: relPoly, width: maxX - minX, height: rfMaxY - rfMinY },
        type: "layoutNode",
        draggable: false,
      };
    });
  };

  // 创建工厂边界节点
  const canvasW = layout.factory?.length || layout.canvas?.width || 100;
  const canvasH = layout.factory?.width || layout.canvas?.height || 100;
  const scale = 360 / Math.max(canvasW, canvasH, 1);
  
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

  return (
    <div style={{ height: 380, border: "1px solid #eee", position: "relative" }}>
      <ReactFlow
        nodes={[factoryBoundaryNode, ...nodesRF()]}
        edges={[]}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodeTypes={{ layoutNode: LayoutNode, boundaryNode: BoundaryNode }}
      >
        <MiniMap />
        <Controls />
        <Background />
      </ReactFlow>
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

// 自定义节点：用路径绘制旋转/缺角形状
const LayoutNode = ({ data }: any) => {
  const { id, typeLabel, polygon, width, height } = data;
  const stroke = typeLabel === "FU" ? "#1677ff" : "#ff4d4f";
  const fill = typeLabel === "FU" ? "#e8f0ff" : "#ffeaea";
  const pathD =
    polygon && polygon.length
      ? `M ${polygon.map((p: any) => `${p.x} ${p.y}`).join(" L ")} Z`
      : `M0 0 H ${width} V ${height} H 0 Z`;
  return (
    <div style={{ width, height, position: "relative" }}>
      <svg width={width} height={height} style={{ position: "absolute", top: 0, left: 0 }}>
        <path d={pathD} fill={fill} stroke={stroke} strokeWidth={2} />
      </svg>
      <div style={{ position: "absolute", top: 4, left: 4, fontSize: 12, fontWeight: 600, color: "#000" }}>{id}</div>
    </div>
  );
};
