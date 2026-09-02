"""用户认证与多租户隔离契约测试。"""

import uuid


def test_password_hash_roundtrip():
    from app.core.security import hash_password, verify_password

    hashed = hash_password("secret123")
    assert hashed.startswith("pbkdf2_sha256$")
    assert verify_password("secret123", hashed)
    assert not verify_password("wrong-password", hashed)


def test_jwt_roundtrip():
    from app.core.security import create_access_token, decode_access_token

    token = create_access_token("u1", "alice")
    payload = decode_access_token(token)
    assert payload["sub"] == "u1"
    assert payload["username"] == "alice"
    assert payload["typ"] == "access"


def test_auth_register_login_me(client):
    suffix = uuid.uuid4().hex[:8]
    username = f"u_{suffix}"
    password = f"pw_{uuid.uuid4().hex[:8]}"

    reg = client.post(
        "/api/auth/register",
        json={"username": username, "password": password, "display_name": "测试用户"},
    )
    assert reg.status_code == 200
    reg_body = reg.json()
    assert reg_body["code"] == 0
    token = reg_body["data"]["access_token"]
    assert reg_body["data"]["user"]["username"] == username

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["data"]["username"] == username

    login = client.post("/api/auth/login", json={"username": username, "password": password})
    assert login.status_code == 200
    assert login.json()["data"]["access_token"]


def test_auth_rejects_missing_token_when_enabled(client):
    r = client.get("/api/chat/sessions")
    assert r.status_code == 401


def test_sessions_are_isolated_by_user(client):
    suffix = uuid.uuid4().hex[:8]
    password = f"pw_{uuid.uuid4().hex[:8]}"

    def signup(name: str) -> str:
        r = client.post("/api/auth/register", json={"username": name, "password": password})
        assert r.status_code == 200
        return r.json()["data"]["access_token"]

    token_a = signup(f"alice_{suffix}")
    token_b = signup(f"bob_{suffix}")
    headers_a = {"Authorization": f"Bearer {token_a}"}
    headers_b = {"Authorization": f"Bearer {token_b}"}

    created = client.post("/api/chat/sessions", headers=headers_a)
    assert created.status_code == 200
    sid = created.json()["data"]["id"]

    a_sessions = client.get("/api/chat/sessions", headers=headers_a).json()["data"]
    assert sid in [item["id"] for item in a_sessions]

    b_sessions = client.get("/api/chat/sessions", headers=headers_b).json()["data"]
    assert sid not in [item["id"] for item in b_sessions]

    b_messages = client.get(f"/api/chat/sessions/{sid}/messages", headers=headers_b)
    assert b_messages.status_code == 404
