"""
后端启动脚本
"""
import sys
from pathlib import Path

# 添加路径
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

import uvicorn
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FactoGenie Backend")
    parser.add_argument("--reset", type=str, default="true", help="Whether to initialize the database (true/false)")
    args = parser.parse_args()

    # 设置环境变量供 api/main.py 使用
    os.environ["FACTOGENIE_INIT_DB"] = args.reset.lower()

    uvicorn.run(
        "api.main:app",
        host="127.0.0.1",  # 本地开发使用 127.0.0.1，如需从其他设备访问可改为 0.0.0.0
        port=8002,
        reload=True,
        reload_dirs=[str(backend_dir)],
    )








