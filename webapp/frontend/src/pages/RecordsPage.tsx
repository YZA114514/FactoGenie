import { useEffect, useState } from "react";
import { Card, Typography, Table, Button, Space, message, Tag } from "antd";
import type { ColumnsType, TablePaginationConfig } from "antd/es/table";
import { trainingApi } from "../services/api";

const { Title, Paragraph } = Typography;

interface ProjectRow {
  id: string;
  name: string;
  status: string;
  current_episode?: number;
  total_steps?: number;
  best_reward?: number;
  created_at?: string;
}

const statusColor: Record<string, string> = {
  running: "green",
  stopped: "red",
  paused: "orange",
  completed: "blue",
  failed: "volcano",
  interrupted: "orange",  // 服务器重启等原因导致的中断
};

const statusText: Record<string, string> = {
  running: "运行中",
  stopped: "已停止",
  paused: "已暂停",
  completed: "已完成",
  failed: "失败",
  interrupted: "已中断",
};

const RecordsPage = () => {
  const [data, setData] = useState<ProjectRow[]>([]);
  const [loading, setLoading] = useState(false);
  const [pagination, setPagination] = useState<TablePaginationConfig>({ current: 1, pageSize: 10, total: 0 });

  const fetchProjects = async (page = 1, size = 10) => {
    try {
      setLoading(true);
      const res = await trainingApi.listProjects(page, size);
      if (res.code !== 0) throw new Error(res.message || "加载失败");
      setData(res.data.projects || []);
      setPagination({ current: res.data.page, pageSize: res.data.size, total: res.data.total });
    } catch (e) {
      message.error(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProjects(pagination.current, pagination.pageSize);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleTableChange = (pager: TablePaginationConfig) => {
    fetchProjects(pager.current || 1, pager.pageSize || 10);
  };

  const handleDelete = async (id: string) => {
    try {
      await trainingApi.deleteProject(id);
      message.success("已删除");
      fetchProjects(pagination.current, pagination.pageSize);
    } catch (e) {
      message.error(e instanceof Error ? e.message : "删除失败");
    }
  };

  const columns: ColumnsType<ProjectRow> = [
    { title: "ID", dataIndex: "id", key: "id", width: 180, ellipsis: true },
    { title: "名称", dataIndex: "name", key: "name" },
    {
      title: "状态",
      dataIndex: "status",
      key: "status",
      render: (v: string) => <Tag color={statusColor[v] || "default"}>{statusText[v] || v || "-"}</Tag>,
    },
    { title: "Episode", dataIndex: "current_episode", key: "current_episode" },
    { title: "最佳奖励", dataIndex: "best_reward", key: "best_reward" },
    { title: "创建时间", dataIndex: "created_at", key: "created_at" },
    {
      title: "操作",
      key: "action",
      render: (_, record) => (
        <Space>
          <Button size="small" onClick={() => navigator.clipboard.writeText(record.id || "")}>
            复制ID
          </Button>
          <Button size="small" danger onClick={() => handleDelete(record.id)}>
            删除
          </Button>
        </Space>
      ),
    },
  ];

  return (
    <Card>
      <Title level={3} style={{ marginTop: 0 }}>
        训练记录
      </Title>
      <Paragraph type="secondary">列出已创建的训练项目，支持删除，复制 ID 用于结果查看页。</Paragraph>
      <Table
        rowKey="id"
        loading={loading}
        columns={columns}
        dataSource={data}
        pagination={pagination}
        onChange={handleTableChange}
        size="small"
      />
    </Card>
  );
};

export default RecordsPage;
