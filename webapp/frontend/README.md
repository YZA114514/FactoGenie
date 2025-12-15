# 启动与使用备忘（本机环境，PowerShell）

## 前端（Vite）
```
cd "E:\清华大学\2025秋 工业工程课程设计\code\git test\FactoGenie\webapp\frontend"
$env:Path="C:\Program Files\nodejs;" + $env:Path
npm install              # 首次需要
npm run dev              # 启动开发服务器，浏览器打开提示的 http://localhost:5173
```

## 后端（FastAPI，Anaconda 环境 ml-new）
```
cd "E:\清华大学\2025秋 工业工程课程设计\code\git test\FactoGenie\webapp\backend"
& "C:\Anaconda\envs\ml-new\python.exe" -m pip install -r requirements.txt   # 缺依赖时运行
& "C:\Anaconda\envs\ml-new\python.exe" run.py                               # 启动后端
```
默认监听 `http://localhost:8000/api`，前端也指向该地址。

## 建模与导出
- 在 Builder 导入 layout.json（含 factory.length/width/grid_spacing）和 factory.json；工厂尺寸会带入画布与导出。
- 编辑 FU/Obstacle（尺寸、缺角、坐标）、工序、路线；保存/导出时会生成 layout.json（使用 factory 字段）和 factory.json。
- “导出为 JSON”可本地下载；“保存到后端”调用后端接口。

## 训练
- 打开“训练控制”，填写项目名称，可选训练参数 JSON（如 `{"checkpoint_interval":5,"total_steps":200}`），选择成品节点/物料，点击“创建并启动训练”。
- 后端本地线程运行训练；可用暂停/恢复/停止按钮，刷新状态查看进度。

## 结果与记录
- “训练记录”页列出项目，可复制项目ID。
- “结果查看”页选择项目ID后点击“加载结果”，可查看最佳布局（支持旋转/缺角）、指标列表、检查点历史。

常见问题：
- `npm` 被执行策略拦截：用 `& "C:\Program Files\nodejs\npm.cmd" run dev`。
- 坐标/尺寸异常：确保 layout.json 使用 `factory.length/width`，节点带 `x/y`。*** End Patch`**
