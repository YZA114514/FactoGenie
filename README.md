# FactoGenie

FactoGenie 是一个使用深度强化学习（Deep Reinforcement Learning）进行工厂布局放置优化的研究/原型项目，包含仿真环境、训练服务以及一个用于构建/启动/查看训练的前端管理界面。

## 主要特性：
- 基于 FastAPI 的后端接口与训练控制
- 基于 PyTorch 的强化学习核心（可自定义训练参数）
- 可视化前端（React + Vite + Ant Design + React Flow）用于构建工厂、启动训练并查看结果

## 目录结构概览：
- `webapp/` — 前端与后端代码（前端: `webapp/frontend`; 后端: `webapp/backend`）
- `simulation/` — 仿真与环境实现
- `agent/` — 训练/强化学习代理实现
- `data/` — 输入输出数据、检查点与结果

## 快速上手

### 0. 环境要求
- 安装 Python 3.8+（推荐 3.11）
- 安装 Node.js（推荐 LTS）和 npm

### 1. 克隆仓库并进入目录：

```bash
git clone <your-repo-url>
cd FactoGenie
```

### 2. 在虚拟环境中安装依赖并运行后端：

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python webapp/backend/run.py
```

后端启动后默认监听 http://localhost:8000（API 前缀 /api）。

### 3. 启动前端开发服务器（在另一个终端）：

```bash
cd webapp/frontend
npm install
echo "VITE_API_BASE_URL=http://localhost:8002/api" > .env
npm run dev
# 在浏览器打开 http://localhost:5173
```

### 4. 简单操作（在浏览器中）：
- 打开 Builder 页面：导入或创建 `layout.json` / `factory.json`，添加节点/工序/路线，保存或导出 JSON。
- 打开 Training 页面：填写项目名称，点击“创建并启动训练”，等待任务运行并查看状态。
- 打开 Results 页面：输入项目 ID，加载并查看最佳布局与指标。
- 详细使用说明请阅读`系统使用手册.txt`

## 常见问题快速解答：
- 如果后端无法启动：确认 `.venv` 已激活且 `pip install -r requirements.txt` 成功。
- 如果前端加载空白：确认 `VITE_API_BASE_URL` 指向后端地址，或在浏览器控制台查看错误。

## 项目树结构（概要）
下面列出项目的主要目录和关键文件，便于快速定位：

```
FactoGenie-develop/
├── LICENSE
├── README.md
├── requirements.txt
├── requirements-dev.txt
├── main.py
├── agent/
│   ├── agent.py
│   ├── dqn_model.py
│   ├── trainer.py
│   └── replay_buffer.py
├── simulation/
│   ├── engine.py
│   ├── layout.py
│   ├── planning.py
│   ├── metrics.py
│   └── visual_utils.py
├── environment/
│   ├── factory_environment.py
│   ├── gym_wrapper.py
│   └── reward_function.py
├── calibration/
│   ├── calibrator.py
│   └── bounds.py
├── data/
│   └── (输入/输出、模型检查点、结果等，不建议提交大文件)
├── webapp/
│   ├── backend/
│   │   ├── run.py
│   │   ├── api/
│   │   │   ├── main.py
│   │   │   └── routes/
│   │   ├── db/
│   │   └── workers/
│   └── frontend/
│       ├── package.json
│       ├── src/
│       │   ├── pages/
│       │   │   ├── BuilderPage.tsx
│       │   │   ├── TrainingPage.tsx
│       │   │   ├── ResultsPage.tsx
│       │   │   └── RecordsPage.tsx
│       │   └── store/
│       └── README.md
├── docs/
└── 系统使用手册.txt
```
