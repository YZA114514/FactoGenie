"""
后端配置
"""
import os
from pathlib import Path

# 项目目录
PROJECT_ROOT = Path(__file__).parent.parent.parent
BACKEND_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"

# 数据库
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    f"sqlite:///{DATA_DIR / 'factogenie.db'}"
)

# Redis (Celery)
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')

# 训练默认参数
DEFAULT_TRAINING_PARAMS = {
    'total_steps': 150000,
    'learning_rate': 0.00002,
    'batch_size': 32,
    'replay_size': 50000,
    'replay_start_size': 5000,
    'epsilon_start': 1.0,
    'epsilon_final': 0.05,
    'epsilon_decay_frames': 150000,
    'sync_target_every': 4000,
    'double_dqn': False,
    'dueling': True,
    'noisy_net': True,
    'prioritized': True,
    'simulation_duration': 2000,
    'use_simulation': True,
    'weights': {
        'distance': 0.20,
        'logistics': 0.30,
        'flow': 0.20,
        'throughput': 0.25,
        'utilization': 0.05,
    },
    'placement_order': 'logistics_intensity',
    'checkpoint_interval': 1000,
    'calibrate_episodes': 100,
}

# CORS
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
]

# 本地/调试模式下用于限制单次运行步数（默认 0 表示不限制，生产/集成测试可通过环境变量设置为 2000 等）
SHORT_TRAINING_MAX_STEPS = int(os.getenv('SHORT_TRAINING_MAX_STEPS', '0'))

