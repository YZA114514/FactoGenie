import { Card, List, Typography } from 'antd';

const { Title, Paragraph, Text } = Typography;

const HomePage = () => {
  const steps = [
    '在 .env 设置后端地址：VITE_API_BASE_URL=http://localhost:8000/api',
    '先在“配置上传”上传/校验工厂配置与布局配置',
    '在“训练控制”创建项目并启动训练（后端跑起来后可联调）',
    '在“训练记录”查看项目列表与状态',
    '在“结果查看”查看最佳布局、指标曲线、热力图',
  ];

  return (
    <Card>
      <Title level={3} style={{ marginTop: 0 }}>
        欢迎使用 FactoGenie
      </Title>
      <Paragraph>这是一个最小可运行的前端骨架，已接好接口封装，方便快速联调和迭代。</Paragraph>
      <List
        header={<Text strong>快速上手</Text>}
        bordered
        dataSource={steps}
        renderItem={(item) => <List.Item>{item}</List.Item>}
      />
    </Card>
  );
};

export default HomePage;
