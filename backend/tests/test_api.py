"""API 集成测试（基于 FastAPI TestClient，复用真实 PostgreSQL）。

覆盖：健康检查、指标端点、知识库上传/列表/删除、可选鉴权。
聊天等需调用 LLM 的接口不在此处测（留给人工/评测脚本）。

注意：kb 上传的后台索引任务会真实调用 embedding，测试中以 no-op 替身避免
外部依赖与耗时；仅验证接口契约（上传即时返回、列表可见、删除生效）。
"""

import io

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert "database" in body


def test_metrics(client):
    r = client.get("/metrics")
    assert r.status_code == 200
    body = r.json()
    assert "counters" in body and "latency" in body
    assert "uptime_seconds" in body


def test_kb_upload_list_delete(client, auth_headers, monkeypatch):
    import app.api.knowledge as kb_api

    async def fake_indexing(*args, **kwargs):
        return None

    monkeypatch.setattr(kb_api.kb_service, "run_indexing", fake_indexing)
    content = "测试知识库条目：熟悉 Java 与 Spring Boot 后端开发。".encode("utf-8")
    files = {"file": ("test_upload.txt", io.BytesIO(content), "text/plain")}
    up = client.post("/api/kb/upload", files=files, headers=auth_headers)
    assert up.status_code == 200
    data = up.json()
    assert data["code"] == 0
    fid = data["data"]["id"]

    lst = client.get("/api/kb/files", headers=auth_headers).json()
    ids = [f["id"] for f in lst["data"]]
    assert fid in ids

    dele = client.delete(f"/api/kb/files/{fid}", headers=auth_headers)
    assert dele.status_code == 200
    lst2 = client.get("/api/kb/files", headers=auth_headers).json()
    assert fid not in [f["id"] for f in lst2["data"]]


def test_kb_upload_rejects_bad_type(client, auth_headers):
    content = b"x"
    files = {"file": ("evil.exe", io.BytesIO(content), "application/octet-stream")}
    r = client.post("/api/kb/upload", files=files, headers=auth_headers)
    assert r.status_code == 400


def test_kb_reindex_starts_background_job(client, auth_headers, monkeypatch):
    import app.api.knowledge as kb_api

    captured = {}

    async def fake_prepare(file_id, user_id):
        captured["file_id"] = file_id
        captured["user_id"] = user_id
        return file_id, "西游记.txt", ".txt", user_id, True

    async def fake_run(*args, **kwargs):
        captured["run"] = (args, kwargs)

    monkeypatch.setattr(kb_api.kb_service, "prepare_reindex", fake_prepare)
    monkeypatch.setattr(kb_api.kb_service, "run_indexing", fake_run)

    response = client.post("/api/kb/files/file-1/reindex", headers=auth_headers)

    assert response.status_code == 202
    assert response.json()["data"]["status"] == "indexing"
    assert captured["file_id"] == "file-1"
    assert "run" in captured


def test_kb_reindex_conflict_returns_409(client, auth_headers, monkeypatch):
    import app.api.knowledge as kb_api

    async def conflict(*args, **kwargs):
        raise kb_api.kb_service.ReindexConflict("文件正在索引")

    monkeypatch.setattr(kb_api.kb_service, "prepare_reindex", conflict)
    response = client.post("/api/kb/files/file-1/reindex", headers=auth_headers)

    assert response.status_code == 409
    assert "正在索引" in response.json()["detail"]
