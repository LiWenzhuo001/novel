"""结构化日志配置——std 库实现，零额外依赖。

目标：
- 统一以 JSON 行输出（便于接入 Loki / ELK / 云端日志），字段包含
  timestamp / level / logger / event / 以及调用方附加的上下文（request_id、latency_ms 等）。
- 开发态（LOG_FORMAT=text）退化为可读文本，方便本地排查。

用法：
    from app.core.logging_config import get_logger
    log = get_logger("kb")
    log.info("file.indexed", file_id=..., chunks=...)   # 关键字参数即结构化上下文字段
    log.error("request.failed", request_id=...)          # 异常用 exc_info=True
"""

import json
import logging
import os
import sys
import time
import uuid


class JsonFormatter(logging.Formatter):
    """把 LogRecord 渲染成单行 JSON。msg 作为 event 字段，上下文（_ctx_*）平铺透传。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "level": record.levelname.lower(),
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, val in record.__dict__.items():
            if key.startswith("_ctx_"):
                payload[key[5:]] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    """开发态可读文本：HH:MM:SS LEVEL [logger] event key=val ..."""

    def format(self, record: logging.LogRecord) -> str:
        parts = [
            time.strftime("%H:%M:%S", time.localtime(record.created)),
            record.levelname,
            f"[{record.name}]",
            record.getMessage(),
        ]
        ctx = " ".join(
            f"{k[5:]}={v}" for k, v in record.__dict__.items() if k.startswith("_ctx_")
        )
        if ctx:
            parts.append(ctx)
        if record.exc_info:
            parts.append(self.formatException(record.exc_info))
        return " ".join(parts)


class StructuredLogger:
    """对标准库 Logger 的轻包装：允许 log.info("event", key=val)，
    关键字参数自动作为结构化上下文字段（写入 extra 的 _ctx_ 前缀）。"""

    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
        if not self._logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            fmt = os.getenv("LOG_FORMAT", "json").lower()
            handler.setFormatter(TextFormatter() if fmt == "text" else JsonFormatter())
            self._logger.addHandler(handler)
            self._logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
            self._logger.propagate = False

    def _emit(self, level: int, event: str, **ctx):
        extra = {f"_ctx_{k}": v for k, v in ctx.items()}
        self._logger.log(level, event, extra=extra)

    def info(self, event: str, **ctx):
        self._emit(logging.INFO, event, **ctx)

    def warning(self, event: str, **ctx):
        self._emit(logging.WARNING, event, **ctx)

    def error(self, event: str, **ctx):
        self._emit(logging.ERROR, event, **ctx)

    def debug(self, event: str, **ctx):
        self._emit(logging.DEBUG, event, **ctx)


def get_logger(name: str) -> StructuredLogger:
    """创建带结构化字段支持的应用日志器。"""
    return StructuredLogger(name)


def new_request_id() -> str:
    """生成当前请求使用的短请求 ID。"""
    return uuid.uuid4().hex[:16]
