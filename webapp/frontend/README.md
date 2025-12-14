# 启动备忘录（本机环境）

前端（Vite，PowerShell 命令逐行输入）  
```
cd "E:\清华大学\2025秋 工业工程课程设计\code\git test\FactoGenie\webapp\frontend"
$env:Path="C:\Program Files\nodejs;" + $env:Path      
npm run dev          # 启动开发服务器，浏览器打开提示的 http://localhost:5173
```
cd "E:\清华大学\2025秋 工业工程课程设计\code\git test\FactoGenie\webapp\frontend"
$env:Path="C:\Program Files\nodejs;" + $env:Path; & "C:\Program Files\nodejs\npm.cmd" run dev


后端（FastAPI，Anaconda 环境 ml-new，PowerShell 命令逐行输入）  
```
cd "E:\清华大学\2025秋 工业工程课程设计\code\git test\FactoGenie\webapp\backend"
& "C:\Anaconda\envs\ml-new\python.exe" -m pip install -r requirements.txt   # 依赖（已有可跳过）
& "C:\Anaconda\envs\ml-new\python.exe" run.py                               # 启动后端
```
后端默认监听 `http://localhost:8000/api`，前端默认也是这个地址。

训练/结果要点  
- Builder 导入 layout/factory 后保存到后端。  
- 训练页：填写项目名，可选训练参数 JSON（如 `{"checkpoint_interval":5,"total_steps":200}`），点击“创建并启动训练”。  
- 结果页：选择项目 ID，点击“加载结果”查看最佳布局、检查点历史；训练记录页可复制项目 ID。

{ "checkpoint_interval": 5, "total_steps": 5000 }