"""
后端 API 快速测试脚本
运行: python test_api.py
"""
import requests
import json

BASE_URL = "http://localhost:8000/api"

def test_create_project():
    """测试创建项目"""
    # 加载示例配置
    with open("../../simulation/configs/chair_factory.json") as f:
        factory_config = json.load(f)
    with open("../../simulation/layouts/chair_layout.json") as f:
        layout_config = json.load(f)
    
    resp = requests.post(f"{BASE_URL}/training/projects", json={
        "name": "测试项目",
        "factory_config": factory_config,
        "layout_config": layout_config,
        "training_params": {
            "total_steps": 1000,
            "checkpoint_interval": 100
        }
    })
    print("创建项目:", resp.status_code)
    data = resp.json()
    print(json.dumps(data, indent=2, ensure_ascii=False))
    return data.get("data", {}).get("project_id")

def test_list_projects():
    """测试获取项目列表"""
    resp = requests.get(f"{BASE_URL}/training/projects")
    print("\n项目列表:", resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

def test_upload_config():
    """测试上传配置文件"""
    with open("../../simulation/configs/chair_factory.json", "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/config/factory/upload",
            files={"file": ("factory.json", f, "application/json")}
        )
    print("\n上传工厂配置:", resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

def test_validate_constraints():
    """测试验证约束"""
    resp = requests.post(f"{BASE_URL}/config/constraints/validate", json={
        "fixed_positions": [
            {"unit_id": "recdock", "x": 0, "y": 10, "angle": 0}
        ],
        "adjacency": [
            {"unit_a": "station1", "unit_b": "station2"}
        ],
        "wall_attach": [
            {"unit_id": "shipdock", "wall": "right"}
        ]
    })
    print("\n验证约束:", resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

def test_start_training(project_id: str):
    """测试启动训练"""
    resp = requests.post(f"{BASE_URL}/training/projects/{project_id}/start")
    print("\n启动训练:", resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

def test_get_status(project_id: str):
    """测试获取状态"""
    resp = requests.get(f"{BASE_URL}/training/projects/{project_id}/status")
    print("\n训练状态:", resp.status_code)
    print(json.dumps(resp.json(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    print("=" * 50)
    print("FactoGenie 后端 API 测试")
    print("=" * 50)
    
    # 测试配置上传
    test_upload_config()
    
    # 测试约束验证
    test_validate_constraints()
    
    # 测试创建项目
    project_id = test_create_project()
    
    # 测试项目列表
    test_list_projects()
    
    if project_id:
        # 测试获取状态
        test_get_status(project_id)
        
        # 测试启动训练 (需要 Celery 运行)
        # test_start_training(project_id)
    
    print("\n" + "=" * 50)
    print("测试完成!")





