"""测试公共配置：在导入任何 app 模块前设置测试期环境变量。"""

import os
import sys
import types

import pytest
from fastapi.testclient import TestClient

# Transformers 即使未使用本地模型，也可能在导入时探测 torch；Windows 测试环境禁用该可选后端。
os.environ.setdefault("USE_TORCH", "0")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
# langchain_text_splitters 会在包导入时加载 sentence_transformers；测试不使用该可选分词器，
# 用轻量模块替身阻止 Windows 上损坏的 torch 原生运行时在收集阶段崩溃。
if "sentence_transformers" not in sys.modules:
    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = type("SentenceTransformer", (), {})
    sys.modules["sentence_transformers"] = sentence_transformers_stub
# 测试日志用文本格式，便于排查
os.environ.setdefault("LOG_FORMAT", "text")
# 测试认证链路时启用 users 表 + JWT；接口测试通过真实登录拿 token。
os.environ.setdefault("USER_AUTH_ENABLED", "true")
os.environ.setdefault("JWT_SECRET", "test_job_agent_secret")


@pytest.fixture(scope="session")
def client():
    # TestClient 保持 session 级，避免 asyncpg 全局连接池跨多个已关闭事件循环复用。
    from app.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def auth_headers(client):
    import uuid

    username = f"test_{uuid.uuid4().hex[:8]}"
    password = f"pw_{uuid.uuid4().hex[:8]}"
    r = client.post("/api/auth/register", json={"username": username, "password": password})
    assert r.status_code == 200
    token = r.json()["data"]["access_token"]
    return {"Authorization": f"Bearer {token}"}
