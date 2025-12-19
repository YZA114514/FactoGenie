import { useEffect, useState } from "react";
import { Card, Typography, Form, Input, InputNumber, Button, Space, message, Row, Col, Alert, Tag, Select } from "antd";
import { replayApi, trainingApi } from "../services/api";
import type { ActionHeatmap } from "../types";

const { Title, Paragraph, Text } = Typography;

const ReplayPage = () => {
  const [form] = Form.useForm<{ projectId: string; episode: number; step?: number }>();
  const [info, setInfo] = useState<any>(null);
  const [stepData, setStepData] = useState<any>(null);
  const [heatmap, setHeatmap] = useState<ActionHeatmap | null>(null);
  const [angleIndex, setAngleIndex] = useState<number>(0);
  const [loading, setLoading] = useState(false);
  const [projectOptions, setProjectOptions] = useState<{ label: string; value: string }[]>([]);
  const [checkpointOptions, setCheckpointOptions] = useState<{ label: string; value: number }[]>([]);

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
      await fetchHeatmap();
    } catch (e) {
      message.error(e instanceof Error ? e.message : "启动失败");
    } finally {
      setLoading(false);
    }
  };

  const fetchStep = async (stepNum?: number) => {
    const projectId = form.getFieldValue("projectId");
    const stepVal = stepNum ?? form.getFieldValue("step") ?? 0;
    if (!projectId) return;
    try {
      const res = await replayApi.step(projectId, Number(stepVal));
      if (res.code === 0) {
        setStepData(res.data);
        form.setFieldsValue({ step: res.data?.step });
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
      if (res.code === 0) {
        setStepData(res.data);
        form.setFieldsValue({ step: res.data?.step });
        await fetchHeatmap();
      } else {
        message.info(res.message || "已到末尾");
      }
    } catch (e) {
      message.error(e instanceof Error ? e.message : "前进一步失败");
    }
  };

  const fetchHeatmap = async () => {
    const projectId = form.getFieldValue("projectId");
    if (!projectId) return;
    try {
      const res = await replayApi.heatmap(projectId);
      if (res.code === 0) {
        setHeatmap(res.data?.heatmap || null);
      }
    } catch {
      /* ignore */
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

  return (
    <Card>
      <Title level={3} style={{ marginTop: 0 }}>
        回放 / 热力图
      </Title>
      <Paragraph type="secondary">按 episode 启动回放，单步或前进查看放置过程的 Q 值热力图。</Paragraph>

      <Form layout="inline" form={form} style={{ marginBottom: 12 }}>
        <Form.Item name="projectId" label="项目ID" rules={[{ required: true }]}>
          <Select
            style={{ width: 260 }}
            placeholder="选择或输入项目ID"
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
        <Form.Item name="episode" label="Episode" rules={[{ required: true, type: "number", min: 0 }]}>
          <Select
            style={{ width: 160 }}
            placeholder="选择检查点 Episode，或手动输入"
            showSearch
            allowClear
            options={checkpointOptions}
            onChange={(v) => form.setFieldsValue({ episode: v as number })}
            dropdownRender={(menu) => (
              <>
                {menu}
                <div style={{ padding: 8 }}>
                  <InputNumber
                    style={{ width: "100%" }}
                    placeholder="手动输入 Episode"
                    onPressEnter={(e) => {
                      const num = Number((e.target as HTMLInputElement).value);
                      if (!Number.isNaN(num)) form.setFieldsValue({ episode: num });
                    }}
                  />
                </div>
              </>
            )}
          />
        </Form.Item>
        <Form.Item>
          <Button type="primary" onClick={startReplay} loading={loading}>
            启动回放
          </Button>
        </Form.Item>
        <Form.Item>
          <Button onClick={closeSession}>关闭会话</Button>
        </Form.Item>
      </Form>

      <Space style={{ marginBottom: 12 }} wrap>
        <Button onClick={() => fetchStep()}>跳转到步</Button>
        <Form.Item name="step" style={{ marginBottom: 0 }}>
          <InputNumber style={{ width: 100 }} placeholder="step" />
        </Form.Item>
        <Button type="primary" onClick={forward}>
          下一步
        </Button>
        <Button onClick={fetchHeatmap}>刷新热力图</Button>
      </Space>

      {info ? (
        <Alert
          type="success"
          showIcon
          message={
            <Space>
              <Text>项目 {info.project_id}</Text>
              <Tag color="blue">Episode {info.episode}</Tag>
              <Tag>总步数 {info.total_steps}</Tag>
            </Space>
          }
        />
      ) : (
        <Alert type="info" showIcon message="尚未启动回放" />
      )}

      <Row gutter={16} style={{ marginTop: 12 }}>
        <Col span={12}>
          <Card size="small" title="当前步数据">
            {stepData ? (
              <pre style={{ whiteSpace: "pre-wrap", margin: 0 }}>{JSON.stringify(stepData, null, 2)}</pre>
            ) : (
              <Text type="secondary">暂无</Text>
            )}
          </Card>
        </Col>
        <Col span={12}>
          <Card
            size="small"
            title="Q 值热力图"
            extra={
              heatmap?.angle_options?.map((ang, idx) => (
                <Button
                  key={ang}
                  type={idx === angleIndex ? "primary" : "default"}
                  size="small"
                  style={{ marginLeft: 4 }}
                  onClick={() => setAngleIndex(idx)}
                >
                  角度 {ang}
                </Button>
              )) || null
            }
          >
            {heatmap ? (
              <HeatmapGrid heatmap={heatmap} angleIndex={angleIndex} />
            ) : (
              <Text type="secondary">暂无热力图</Text>
            )}
          </Card>
        </Col>
      </Row>
    </Card>
  );
};

export default ReplayPage;

const HeatmapGrid = ({ heatmap, angleIndex }: { heatmap: ActionHeatmap; angleIndex: number }) => {
  const angIdx = Math.min(angleIndex, (heatmap.angle_options?.length || 1) - 1);
  const qGrid = heatmap.q_values?.[angIdx] || [];
  const w = heatmap.grid_width || (qGrid[0]?.length ?? 0);
  const h = heatmap.grid_height || qGrid.length;
  const min = heatmap.q_min ?? 0;
  const max = heatmap.q_max ?? 1;
  const scale = (v: number) => {
    if (max === min) return 0.5;
    return (v - min) / (max - min);
  };
  const cellSize = 18;

  const selected =
    heatmap.selected_action &&
    heatmap.selected_action.angle === heatmap.angle_options?.[angIdx]
      ? { x: heatmap.selected_action.x, y: heatmap.selected_action.y }
      : null;

  return (
    <div style={{ position: "relative", width: w * cellSize, height: h * cellSize, border: "1px solid #f0f0f0" }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: `repeat(${w}, ${cellSize}px)`,
          gridTemplateRows: `repeat(${h}, ${cellSize}px)`,
          gap: 1,
        }}
      >
        {qGrid.map((row, y) =>
          row.map((val: number, x: number) => {
            const t = scale(val ?? 0);
            const color = `rgba(22, 119, 255, ${t})`;
            const isSel = selected && selected.x === x && selected.y === y;
            return (
              <div
                key={`${x}-${y}`}
                style={{
                  width: cellSize,
                  height: cellSize,
                  background: color,
                  border: isSel ? "2px solid #fa8c16" : "1px solid #f0f0f0",
                }}
                title={`(${x},${y})=${val?.toFixed(3) ?? "N/A"}`}
              />
            );
          })
        )}
      </div>
    </div>
  );
};
