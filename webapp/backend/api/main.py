"""
FactoGenie Backend - FastAPI Entry Point
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
backend_dir = Path(__file__).parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from api.routes import config, training, results, calibration, replay
from db.database import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化数据库
    init_db()
    yield
    # 关闭时清理（如果需要）


app = FastAPI(
    title="FactoGenie API",
    description="Factory Layout Optimization Backend",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(config.router, prefix="/api/config", tags=["Config"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])
app.include_router(calibration.router, prefix="/api/calibration", tags=["Calibration"])
app.include_router(replay.router, prefix="/api/replay", tags=["Replay"])


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"code": 0, "data": {"status": "ok"}}


@app.get("/")
async def root():
    """根路径"""
    return {
        "name": "FactoGenie API",
        "version": "1.0.0",
        "docs": "/docs",
    }

