"""
统一工具执行器 —— 标准化安全边界。

通过中间件链实现：
  1. 参数严格校验（Pydantic args_schema）
  2. 高风险工具阻断 + 人工审批预留
  3. 短时间重复调用拦截
  4. 执行异常隔离（不向上抛出）
  5. 审计 trace 落盘
"""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool

from app.config import get_settings
from app.harness.enums import ToolCallStatus, ToolType
from app.harness.models import RunContext, ToolCallRecord
from app.harness.tools.middleware import (
    AsyncExecutionMiddleware,
    AuditMiddleware,
    ParamValidationMiddleware,
    ToolCallContext,
    _resolve_tool_type,
    build_middleware_chain,
    ctx_to_record,
    run_middleware_chain,
)
from app.harness.tools.middleware import (
    DuplicateCallGuardMiddleware,
    HighRiskGateMiddleware,
)


class ToolExecutor:
    """
    统一工具执行入口（中间件模式）。

    调用示例:
        executor = ToolExecutor(tools, run_ctx, session_id="chat-001")
        record = executor.execute("compound_interest", {"principal": 10000, ...})
        if record.status == ToolCallStatus.PENDING_APPROVAL:
            # 引导用户完成人工审批后，携带 approval_token 重试
            record = executor.execute(..., approval_token=record.approval_id)
    """

    def __init__(
        self,
        tools: list[BaseTool],
        run_ctx: RunContext | None = None,
        session_id: str = "",
    ) -> None:
        self._tools = {t.name: t for t in tools}
        self._run_ctx = run_ctx
        self._session_id = session_id or (run_ctx.manifest.chat_id if run_ctx else "")
        self.records: list[ToolCallRecord] = []
        self._audit = AuditMiddleware()

        settings = get_settings()
        self._duplicate_window = settings.tool_duplicate_window_seconds

    @property
    def tool_names(self) -> list[str]:
        return list(self._tools.keys())

    def get_langchain_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    def _build_ctx(
        self,
        name: str,
        arguments: dict[str, Any],
        approval_token: str | None = None,
    ) -> ToolCallContext:
        tool = self._tools.get(name)
        return ToolCallContext(
            tool_name=name,
            arguments=arguments,
            tool=tool,
            tool_type=_resolve_tool_type(tool) if tool else ToolType.BUILTIN,
            run_ctx=self._run_ctx,
            session_id=self._session_id,
            approval_token=approval_token,
        )

    def _invoke_tool(self, ctx: ToolCallContext) -> str:
        """实际调用 LangChain 工具（由 ExecutionMiddleware 调用）。"""
        if not ctx.tool:
            raise ValueError(f"工具不存在: {ctx.tool_name}")
        args = ctx.validated_args or ctx.arguments
        result = ctx.tool.invoke(args)
        return result if isinstance(result, str) else str(result)

    async def _ainvoke_tool(self, ctx: ToolCallContext) -> str:
        """异步调用 LangChain 工具。"""
        if not ctx.tool:
            raise ValueError(f"工具不存在: {ctx.tool_name}")
        args = ctx.validated_args or ctx.arguments
        if hasattr(ctx.tool, "ainvoke"):
            result = await ctx.tool.ainvoke(args)
        else:
            result = ctx.tool.invoke(args)
        return result if isinstance(result, str) else str(result)

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        approval_token: str | None = None,
    ) -> ToolCallRecord:
        """同步执行工具（完整安全管道）。"""
        ctx = self._build_ctx(name, arguments, approval_token)

        if not ctx.tool:
            record = ToolCallRecord(
                tool_name=name,
                tool_type=ToolType.BUILTIN,
                arguments=arguments,
                success=False,
                status=ToolCallStatus.NOT_FOUND,
                error=f"工具不存在: {name}",
                block_reason=f"工具不存在: {name}",
            )
            self.records.append(record)
            self._audit.after(ctx, record)
            return record

        chain = build_middleware_chain(
            self._invoke_tool,
            duplicate_window=self._duplicate_window,
        )
        run_middleware_chain(ctx, chain)
        record = ctx_to_record(ctx)
        record = self._audit.after(ctx, record)
        self.records.append(record)
        self._update_manifest_count()
        return record

    async def aexecute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        approval_token: str | None = None,
    ) -> ToolCallRecord:
        """异步执行工具（完整安全管道）。"""
        ctx = self._build_ctx(name, arguments, approval_token)

        if not ctx.tool:
            record = ToolCallRecord(
                tool_name=name,
                tool_type=ToolType.BUILTIN,
                arguments=arguments,
                success=False,
                status=ToolCallStatus.NOT_FOUND,
                error=f"工具不存在: {name}",
                block_reason=f"工具不存在: {name}",
            )
            self.records.append(record)
            self._audit.after(ctx, record)
            return record

        # 前置中间件（校验 / 高风险 / 去重）
        pre_chain = [
            DuplicateCallGuardMiddleware(window_seconds=self._duplicate_window),
            HighRiskGateMiddleware(),
            ParamValidationMiddleware(),
        ]
        run_middleware_chain(ctx, pre_chain)

        # 异步执行 + 异常隔离
        if not ctx.skipped_execution:
            exec_mw = AsyncExecutionMiddleware(self._ainvoke_tool)
            await exec_mw.run(ctx)

        record = ctx_to_record(ctx)
        record = self._audit.after(ctx, record)
        self.records.append(record)
        self._update_manifest_count()
        return record

    def _update_manifest_count(self) -> None:
        if self._run_ctx:
            self._run_ctx.manifest.tool_calls_count = len(self.records)
