"""
后端启动脚本
"""
import sys
from pathlib import Path

# 添加路径
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",  # 本地开发使用 127.0.0.1，如需从其他设备访问可改为 0.0.0.0
        port=8001,
        reload=True,
        reload_dirs=[str(backend_dir)],
    )








