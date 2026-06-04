"""HTTP 请求日志中间件。

每次请求写入 logs/uburnode.log：入站 / 出站 / 耗时 / 状态码。
X-Request-Id 支持上游传入或自动生成，便于与业务日志关联。
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from fastapi import Request, Response
from loguru import logger
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

MAX_LOG_BODY_LEN = 512


class RequestLogMiddleware(BaseHTTPMiddleware):
    """在 ASGI 边界记录请求生命周期，不解析 body（避免大 payload 拖慢日志）。"""

    def __init__(self, app: ASGIApp) -> None:
        super().__init__(app)

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        client_host = request.client.host if request.client else "unknown"
        started_at = time.perf_counter()

        bound = logger.bind(request_id=request_id)
        bound.info(
            "请求开始，方法={}，路径={}，查询参数={}，客户端={}",
            request.method,
            request.url.path,
            _truncate(request.url.query),
            client_host,
        )

        try:
            response = await call_next(request)
        except Exception:
            duration_ms = _elapsed_ms(started_at)
            bound.exception(
                "请求失败，方法={}，路径={}，耗时={}毫秒",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = _elapsed_ms(started_at)
        bound.info(
            "请求完成，方法={}，路径={}，状态码={}，耗时={}毫秒",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-Id"] = request_id
        return response


def _elapsed_ms(started_at: float) -> int:
    return int((time.perf_counter() - started_at) * 1000)


def _truncate(value: str, max_len: int = MAX_LOG_BODY_LEN) -> str:
    if len(value) <= max_len:
        return value
    return f"{value[:max_len]}…（已截断）"
