"""
FactoGenie Backend - FastAPI Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import config, training, results

app = FastAPI(
    title="FactoGenie API",
    description="Factory Layout Optimization Backend",
    version="1.0.0"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # 前端地址
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(config.router, prefix="/api/config", tags=["Config"])
app.include_router(training.router, prefix="/api/training", tags=["Training"])
app.include_router(results.router, prefix="/api/results", tags=["Results"])


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

