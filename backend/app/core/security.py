"""密码哈希、JWT 令牌创建与解析等认证安全工具。"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timedelta
from typing import Any

from app.config import settings


def hash_password(password: str) -> str:
    """Hash password with PBKDF2-HMAC-SHA256.

    Format: pbkdf2_sha256$iterations$salt$hash
    Uses only Python stdlib to keep deployment small.
    """
    if len(password) < 6:
        raise ValueError("密码长度至少 6 位")
    iterations = 260_000
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    """用恒定时间比较验证明文密码与已存哈希是否匹配。"""
    try:
        scheme, iter_s, salt, expected = password_hash.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt.encode("utf-8"),
            int(iter_s),
        ).hex()
        return hmac.compare_digest(digest, expected)
    except Exception:
        return False


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


def create_access_token(user_id: str, username: str, expires_delta: timedelta | None = None) -> str:
    """创建包含用户标识和过期时间的 JWT。"""
    now = int(time.time())
    exp = now + int((expires_delta or timedelta(minutes=settings.jwt_access_token_minutes)).total_seconds())
    if settings.jwt_algorithm != "HS256":
        raise ValueError("当前内置 JWT 实现仅支持 HS256")
    header = {"alg": settings.jwt_algorithm, "typ": "JWT"}
    payload: dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "iat": now,
        "exp": exp,
        "typ": "access",
    }
    signing_input = f"{_b64url_encode(json.dumps(header, separators=(',', ':')).encode())}.{_b64url_encode(json.dumps(payload, separators=(',', ':')).encode())}"
    signature = hmac.new(settings.jwt_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_access_token(token: str) -> dict[str, Any]:
    """验证 JWT 签名、类型和过期时间，返回可信载荷。"""
    try:
        header_b64, payload_b64, sig_b64 = token.split(".", 2)
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "HS256":
            raise ValueError("unsupported algorithm")
        signing_input = f"{header_b64}.{payload_b64}"
        expected = hmac.new(settings.jwt_secret.encode("utf-8"), signing_input.encode("ascii"), hashlib.sha256).digest()
        actual = _b64url_decode(sig_b64)
        if not hmac.compare_digest(actual, expected):
            raise ValueError("invalid signature")
        payload = json.loads(_b64url_decode(payload_b64))
        if payload.get("typ") != "access":
            raise ValueError("invalid token type")
        if int(payload.get("exp", 0)) < int(time.time()):
            raise ValueError("token expired")
        if not payload.get("sub"):
            raise ValueError("missing subject")
        return payload
    except Exception as exc:
        raise ValueError("invalid token") from exc


def token_expires_at(token: str) -> datetime | None:
    try:
        payload = decode_access_token(token)
        return datetime.fromtimestamp(int(payload["exp"]))
    except Exception:
        return None
