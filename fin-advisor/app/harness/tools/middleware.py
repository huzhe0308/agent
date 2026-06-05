"""
工具执行中间件链（洋葱模型）。

执行顺序（由外到内）：
  Audit → DuplicateGuard → HighRiskGate → ParamValidation → Execution
"""

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Awaitable, Callable

from langchain_core.tools import BaseTool
from pydantic import ValidationError

from app.harness.enums import ToolCallStatus, ToolType
from app.harness.models import RunContext, ToolCallRecord
from app.harness.tools.security import approval_gate


def _resolve_tool_type(tool: BaseTool) -> ToolType:
    meta = getattr(tool, "metadata", None) or {}
    raw = meta.get("tool_type", ToolType.STRUCTURED.value)
    try:
        return ToolType(raw)
    except ValueError:
        return ToolType.STRUCTURED


@dataclass
class ToolCallContext:
    """单次工具调用的管道上下文。"""

    tool_name: str
    arguments: dict[str, Any]
    tool: BaseTool | None = None
    tool_type: ToolType = ToolType.STRUCTURED
    run_ctx: RunContext | None = None
    session_id: str = ""
    approval_token: str | None = None

    validated_args: dict[str, Any] = field(default_factory=dict)
    status: ToolCallStatus = ToolCallStatus.SUCCESS
    block_reason: str | None = None
    approval_id: str | None = None
    result: str = ""
    error: str | None = None
    duration_ms: float = 0.0
    skipped_execution: bool = False


class ToolMiddleware(ABC):
    """中间件抽象基类。"""

    @abstractmethod
    def before(self, ctx: ToolCallContext) -> None:
        """前置处理；设置 ctx.skipped_execution=True 可短路后续执行。"""

    def after(self, ctx: ToolCallContext, record: ToolCallRecord) -> ToolCallRecord:
        """后置处理（可选覆盖）。"""
        return record


class ParamValidationMiddleware(ToolMiddleware):
    """参数严格校验：基于 StructuredTool 的 args_schema 做 Pydantic 校验。"""

    def before(self, ctx: ToolCallContext) -> None:
        if ctx.skipped_execution or not ctx.tool:
            return

        schema = getattr(ctx.tool, "args_schema", None)
        if schema is None:
            ctx.validated_args = dict(ctx.arguments)
            return

        try:
            if hasattr(schema, "model_validate"):
                validated = schema.model_validate(ctx.arguments)
                ctx.validated_args = validated.model_dump()
            else:
                validated = schema(**ctx.arguments)  # type: ignore[misc]
                ctx.validated_args = dict(validated.__dict__)
        except (ValidationError, TypeError, ValueError) as e:
            ctx.status = ToolCallStatus.VALIDATION_ERROR
            ctx.block_reason = f"参数校验失败: {e}"
            ctx.skipped_execution = True


class HighRiskGateMiddleware(ToolMiddleware):
    """高风险工具阻断 + 人工审批门控。"""

    def before(self, ctx: ToolCallContext) -> None:
        if ctx.skipped_execution:
            return

        allowed, approval_req, block_reason = approval_gate.check_high_risk(
            tool_name=ctx.tool_name,
            arguments=ctx.validated_args or ctx.arguments,
            session_id=ctx.session_id or (ctx.run_ctx.manifest.chat_id if ctx.run_ctx else ""),
            run_id=ctx.run_ctx.manifest.run_id if ctx.run_ctx else None,
            approval_token=ctx.approval_token,
        )

        if allowed:
            return

        if approval_req:
            ctx.status = ToolCallStatus.PENDING_APPROVAL
            ctx.approval_id = approval_req.approval_id
            ctx.block_reason = (
                f"高风险工具 '{ctx.tool_name}' 需人工审批，"
                f"approval_id={approval_req.approval_id}"
            )
        else:
            ctx.status = ToolCallStatus.BLOCKED_HIGH_RISK
            ctx.block_reason = block_reason or f"高风险工具 '{ctx.tool_name}' 已被阻断"

        ctx.skipped_execution = True


class DuplicateCallGuardMiddleware(ToolMiddleware):
    """短时间重复调用拦截（相同工具+相同参数）。"""

    def __init__(self, window_seconds: float = 30.0, max_entries: int = 1000) -> None:
        self._window = window_seconds
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._lock = Lock()
        self._max_entries = max_entries

    def _fingerprint(self, ctx: ToolCallContext) -> str:
        payload = json.dumps(
            {"tool": ctx.tool_name, "args": ctx.validated_args or ctx.arguments},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    def before(self, ctx: ToolCallContext) -> None:
        if ctx.skipped_execution:
            return

        fp = self._fingerprint(ctx)
        now = time.monotonic()

        with self._lock:
            # 清理过期条目
            expired = [k for k, ts in self._cache.items() if now - ts > self._window]
            for k in expired:
                del self._cache[k]

            if fp in self._cache:
                ctx.status = ToolCallStatus.DUPLICATE_BLOCKED
                ctx.block_reason = (
                    f"短时间重复调用已拦截（窗口 {self._window}s）: {ctx.tool_name}"
                )
                ctx.skipped_execution = True
                return

            self._cache[fp] = now
            if len(self._cache) > self._max_entries:
                self._cache.popitem(last=False)


class ExecutionMiddleware(ToolMiddleware):
    """实际工具执行 + 异常隔离（捕获所有异常，不向上抛出）。"""

    def __init__(self, invoke_fn: Callable[[ToolCallContext], str]) -> None:
        self._invoke_fn = invoke_fn

    def before(self, ctx: ToolCallContext) -> None:
        if ctx.skipped_execution:
            return

        start = time.perf_counter()
        try:
            ctx.result = self._invoke_fn(ctx)
            ctx.status = ToolCallStatus.SUCCESS
        except Exception as e:
            ctx.status = ToolCallStatus.EXECUTION_ERROR
            ctx.error = str(e)
            ctx.result = ""
        finally:
            ctx.duration_ms = (time.perf_counter() - start) * 1000


class AsyncExecutionMiddleware(ToolMiddleware):
    """异步执行 + 异常隔离。"""

    def __init__(self, invoke_fn: Callable[[ToolCallContext], Awaitable[str]]) -> None:
        self._invoke_fn = invoke_fn

    async def run(self, ctx: ToolCallContext) -> None:
        if ctx.skipped_execution:
            return

        start = time.perf_counter()
        try:
            ctx.result = await self._invoke_fn(ctx)
            ctx.status = ToolCallStatus.SUCCESS
        except Exception as e:
            ctx.status = ToolCallStatus.EXECUTION_ERROR
            ctx.error = str(e)
            ctx.result = ""
        finally:
            ctx.duration_ms = (time.perf_counter() - start) * 1000


class AuditMiddleware(ToolMiddleware):
    """审计后置：写 trace、更新 manifest。"""

    def after(self, ctx: ToolCallContext, record: ToolCallRecord) -> ToolCallRecord:
        if not ctx.run_ctx:
            return record

        from app.harness.artifacts.store import artifact_store

        artifact_store.append_trace(
            ctx.run_ctx,
            event_type="tool_call",
            payload={
                "tool_name": record.tool_name,
                "tool_type": record.tool_type.value,
                "status": record.status.value if hasattr(record.status, "value") else record.status,
                "arguments": record.arguments,
                "success": record.success,
                "duration_ms": record.duration_ms,
                "error": record.error,
                "block_reason": record.block_reason,
                "approval_id": record.approval_id,
                "result_preview": record.result[:500] if record.result else "",
            },
        )
        if record.success and record.tool_name not in ctx.run_ctx.manifest.tools_used:
            ctx.run_ctx.manifest.tools_used.append(record.tool_name)
        return record


def ctx_to_record(ctx: ToolCallContext) -> ToolCallRecord:
    """将管道上下文转为审计记录。"""
    success = ctx.status == ToolCallStatus.SUCCESS
    return ToolCallRecord(
        tool_name=ctx.tool_name,
        tool_type=ctx.tool_type,
        arguments=ctx.validated_args or ctx.arguments,
        result=ctx.result,
        duration_ms=ctx.duration_ms,
        success=success,
        error=ctx.error or ctx.block_reason,
        status=ctx.status,
        block_reason=ctx.block_reason,
        approval_id=ctx.approval_id,
    )


def build_middleware_chain(
    invoke_fn: Callable[[ToolCallContext], str],
    *,
    duplicate_window: float = 30.0,
) -> list[ToolMiddleware]:
    """构建同步中间件链（不含 Audit，由 Executor 在最外层调用）。"""
    return [
        DuplicateCallGuardMiddleware(window_seconds=duplicate_window),
        HighRiskGateMiddleware(),
        ParamValidationMiddleware(),
        ExecutionMiddleware(invoke_fn),
    ]


def run_middleware_chain(
    ctx: ToolCallContext,
    chain: list[ToolMiddleware],
) -> ToolCallContext:
    """顺序执行 before 链；ExecutionMiddleware 在链末实际调用工具。"""
    for mw in chain:
        mw.before(ctx)
        if ctx.skipped_execution:
            break
    return ctx
