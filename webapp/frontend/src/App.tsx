import { Layout, Menu, Typography, Space } from 'antd';
import { Link, Route, Routes, useLocation } from 'react-router-dom';
import {
  PlayCircleOutlined,
  LineChartOutlined,
  DatabaseOutlined,
  HomeOutlined,
  ProjectOutlined,
  EyeOutlined,
} from '@ant-design/icons';
import HomePage from './pages/HomePage';
import TrainingPage from './pages/TrainingPage';
import ResultsPage from './pages/ResultsPage';
import RecordsPage from './pages/RecordsPage';
import BuilderPage from './pages/BuilderPage';
import ReplayPage from './pages/ReplayPage';

const { Header, Content, Footer } = Layout;
const { Title } = Typography;

const navItems = [
  { key: 'home', label: <Link to="/">概览</Link>, icon: <HomeOutlined /> },
  { key: 'builder', label: <Link to="/builder">配置 / 建模</Link>, icon: <ProjectOutlined /> },
  { key: 'training', label: <Link to="/training">训练控制</Link>, icon: <PlayCircleOutlined /> },
  { key: 'results', label: <Link to="/results">结果查看</Link>, icon: <LineChartOutlined /> },
  { key: 'records', label: <Link to="/records">训练记录</Link>, icon: <DatabaseOutlined /> },
  { key: 'replay', label: <Link to="/replay">回放 / 热力图</Link>, icon: <EyeOutlined /> },
];

const App = () => {
  const { pathname } = useLocation();
  const activeKey = navItems.find((item) => {
    if (item.key === 'home') {
      return pathname === '/';
    }
    return pathname.startsWith(`/${item.key}`);
  })?.key;

  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Header
        style={{
          position: 'sticky',
          top: 0,
          zIndex: 10,
          display: 'flex',
          alignItems: 'center',
          gap: 16,
        }}
      >
        <Title level={4} style={{ color: '#fff', margin: 0 }}>
          FactoGenie
        </Title>
        <Menu
          theme="dark"
          mode="horizontal"
          selectedKeys={activeKey ? [activeKey] : []}
          items={navItems}
          style={{ flex: 1 }}
        />
      </Header>
      <Content style={{ padding: 24 }}>
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/builder" element={<BuilderPage />} />
          <Route path="/training" element={<TrainingPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/records" element={<RecordsPage />} />
          <Route path="/replay" element={<ReplayPage />} />
        </Routes>
      </Content>
      <Footer style={{ textAlign: 'center' }}>
        <Space size="small">前端 React + Vite · 接口基于 docs/api/API_CONTRACT.md</Space>
      </Footer>
    </Layout>
  );
};

export default App;
