import { useEffect, useState } from "react";
import { Card, Typography, Form, Button, Space, List, Alert, Select, message, Tag, Row, Col } from "antd";
import { resultsApi, trainingApi } from "../services/api";
import ReactFlow, { Background, Controls, MiniMap } from "reactflow";
import "reactflow/dist/style.css";

const { Title, Paragraph, Text } = Typography;

const ResultsPage = () => {
  const [form] = Form.useForm<{ projectId: string }>();
  const [best, setBest] = useState<any>(null);
  const [layouts, setLayouts] = useState<any[]>([]);
  const [metrics, setMetrics] = useState<{ metric?: string; values?: any[] }>({});
  const [loading, setLoading] = useState(false);
  const [projects, setProjects] = useState<{ label: string; value: string }[]>([]);

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
      const [bestRes, layoutsRes, metricsRes] = await Promise.allSettled([
        resultsApi.getBestLayout(projectId),
        resultsApi.getLayouts(projectId, 1, 20),
        resultsApi.getMetrics(projectId, "reward", 0),
      ]);
      if (bestRes.status === "fulfilled" && bestRes.value.code === 0) {
        setBest(bestRes.value.data);
      } else if (bestRes.status === "fulfilled") {
        setBest(null);
        message.info(bestRes.value.message || "暂无最佳布局");
      }
      if (layoutsRes.status === "fulfilled" && layoutsRes.value.code === 0) {
        setLayouts(layoutsRes.value.data.layouts || []);
      } else if (layoutsRes.status === "fulfilled") {
        setLayouts([]);
        message.info(layoutsRes.value.message || "暂无检查点");
      }
      if (metricsRes.status === "fulfilled" && metricsRes.value.code === 0) {
        setMetrics(metricsRes.value.data);
      } else if (metricsRes.status === "fulfilled") {
        setMetrics({});
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
          <List
            size="small"
            dataSource={metrics.values}
            renderItem={(v: any) => (
              <List.Item>
                Episode {v.episode}: {v.value}
              </List.Item>
            )}
          />
        ) : (
          <Text type="secondary">暂无指标数据</Text>
        )}
      </Card>

      <Card size="small" title="布局/检查点历史">
        <List
          size="small"
          dataSource={layouts}
          bordered
          locale={{ emptyText: "暂无检查点" }}
          renderItem={(item: any) => (
            <List.Item>
              <Space direction="vertical">
                <Text>
                  Episode {item.episode} · reward {item.reward ?? "-"} {item.is_best ? <Tag color="green">best</Tag> : null}
                </Text>
                <Text type="secondary">
                  工厂尺寸：{item.layout?.factory?.length || item.layout?.canvas?.width || "-"} ×{" "}
                  {item.layout?.factory?.width || item.layout?.canvas?.height || "-"} · grid_spacing{" "}
                  {item.layout?.factory?.grid_spacing ?? "-"}
                </Text>
                <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(item.layout, null, 2)}</pre>
              </Space>
            </List.Item>
          )}
        />
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
      const len = u.length || u.width || 10;
      const wid = u.width || u.length || 10;
      const nl = u.notch_length || 0;
      const nw = u.notch_width || 0;
      const angle = u.angle || 0;

      // build polygon (lower-left origin)
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
      const rotated = pts.map((p) => {
        const r = rotateCW(p.x, p.y, angle);
        return { x: (baseX + r.x) * scale, y: (baseY + r.y) * scale };
      });
      const minX = Math.min(...rotated.map((p) => p.x));
      const minY = Math.min(...rotated.map((p) => p.y));
      const maxX = Math.max(...rotated.map((p) => p.x));
      const maxY = Math.max(...rotated.map((p) => p.y));
      const relPoly = rotated.map((p) => ({ x: p.x - minX, y: p.y - minY }));

      return {
        id: u.id || `node-${idx}`,
        position: { x: minX + 10, y: minY + 10 },
        data: { id: u.id, typeLabel: u.typeLabel, polygon: relPoly, width: maxX - minX, height: maxY - minY },
        type: "layoutNode",
        draggable: false,
      };
    });
  };

  return (
    <div style={{ height: 380, border: "1px solid #eee", position: "relative" }}>
      <ReactFlow
        nodes={nodesRF()}
        edges={[]}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        nodeTypes={{ layoutNode: LayoutNode }}
      >
        <MiniMap />
        <Controls />
        <Background />
      </ReactFlow>
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
