"""轻量内存指标——进程内计数与延迟采样，供 /metrics 端点暴露。

设计取舍：
- 不引入 prometheus_client 等依赖，用线程安全的 dict + 最近 N 条延迟样本，
  足够覆盖单实例个人应用的观测需求；后续要接 Prometheus 只需替换此模块实现。
- 指标包含：聊天请求数、知识库上传数、检索调用数、错误数，以及聊天/检索的
  最近延迟样本（用于算 p50/p95/avg）。
"""

from threading import Lock
from time import time


class Metrics:
    def __init__(self, latency_window: int = 200):
        self._lock = Lock()
        self.counters = {
            "chat_requests": 0,
            "kb_uploads": 0,
            "retrieval_calls": 0,
            "errors": 0,
            "sse_cancellations": 0,
        }
        self._latency = {  # name -> list[float(ms)]
            "chat": [],
            "retrieval": [],
            "request": [],
        }
        self._window = latency_window
        self.started_at = time()

    # ---- 计数 ----
    def incr(self, name: str, by: int = 1):
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + by

    def error(self):
        self.incr("errors")

    # ---- 延迟采样 ----
    def record_latency(self, name: str, ms: float):
        with self._lock:
            buf = self._latency.setdefault(name, [])
            buf.append(ms)
            if len(buf) > self._window:
                buf.pop(0)

    def _stats(self, name: str) -> dict:
        buf = sorted(self._latency.get(name, []))
        n = len(buf)
        if n == 0:
            return {"count": 0, "avg_ms": 0, "p50_ms": 0, "p95_ms": 0}
        avg = sum(buf) / n
        p50 = buf[int(n * 0.50)]
        p95 = buf[min(n - 1, int(n * 0.95))]
        return {
            "count": n,
            "avg_ms": round(avg, 1),
            "p50_ms": round(p50, 1),
            "p95_ms": round(p95, 1),
        }

    def snapshot(self) -> dict:
        """返回进程内计数器和延迟采样的快照。"""
        with self._lock:
            return {
                "uptime_seconds": round(time() - self.started_at, 1),
                "counters": dict(self.counters),
                "latency": {k: self._stats(k) for k in self._latency},
            }


metrics = Metrics()
