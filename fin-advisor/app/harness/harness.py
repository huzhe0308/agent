"""
Agent Harness 主编排器。

这是整个本地 Agent 的统一入口，负责串联：
  模型接入 → 工具执行 → 分层记忆 → Checkpoint → 运行工件落盘

上层（chat_service）只需调用 start_run / finish_run，
本模块自动完成 trace 记录和 checkpoint 快照。
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from app.harness.artifacts.store import artifact_store
from app.harness.enums import RunStatus, ToolType
from app.harness.memory.manager import agent_memory
from app.harness.models import RunContext
from app.harness.providers.registry import get_llm
from app.harness.state.checkpoint import get_checkpoint_manager
from app.harness.tools.executor import ToolExecutor
from app.harness.tools.registry import load_all_tools, load_sync_tools

FIN_MANUS_SYSTEM = """你是 FinManus 金融超级智能体，具备 ReAct 推理与工具调用能力。
职责：理解用户金融理财诉求，调用工具完成测算与检索，给出专业结构化回答。
涉及投资建议时声明"仅供参考，不构成投资建议"。"""


class AgentHarness:
    """
    本地 Agent 统一 Harness。

    核心能力：
    - get_llm()：2 类模型后端统一接入
    - create_tool_executor()：7 类工具统一执行 + 审计
    - start_run() / finish_run()：运行生命周期管理
    - trace_node()：节点级 trace + checkpoint
    - memory：四层 Agent 记忆
    """

    def __init__(self) -> None:
        self.checkpoint_mgr = get_checkpoint_manager()
        self.memory = agent_memory
        self.artifacts = artifact_store

    def get_llm(self) -> BaseChatModel:
        """获取当前配置的 LLM 实例。"""
        return get_llm()

    async def create_tool_executor(
        self,
        run_ctx: RunContext | None = None,
        enabled_types: list[ToolType] | None = None,
        session_id: str = "",
    ) -> ToolExecutor:
        """创建异步工具执行器（含 MCP 工具）。"""
        tools = await load_all_tools(enabled_types)
        sid = session_id or (run_ctx.manifest.chat_id if run_ctx else "")
        return ToolExecutor(tools, run_ctx, session_id=sid)

    def create_sync_tool_executor(
        self,
        run_ctx: RunContext | None = None,
        enabled_types: list[ToolType] | None = None,
        session_id: str = "",
    ) -> ToolExecutor:
        """创建同步工具执行器（不含 MCP，用于 workflow 节点）。"""
        tools = load_sync_tools(enabled_types)
        sid = session_id or (run_ctx.manifest.chat_id if run_ctx else "")
        return ToolExecutor(tools, run_ctx, session_id=sid)

    def start_run(
        self,
        chat_id: str,
        mode: str = "advisor",
        metadata: dict | None = None,
    ) -> RunContext:
        """
        开始一次 Agent 运行。

        创建 run_id、初始化 manifest.json、写入 run_start 轨迹事件。
        """
        ctx = self.artifacts.create_run(chat_id, mode=mode, metadata=metadata)
        self.artifacts.append_trace(ctx, event_type="run_start", payload={"mode": mode})
        return ctx

    def finish_run(
        self,
        ctx: RunContext,
        status: RunStatus = RunStatus.COMPLETED,
        error: str | None = None,
    ) -> None:
        """结束运行：写入 run_end 事件，更新 manifest 状态。"""
        self.artifacts.append_trace(
            ctx,
            event_type="run_end",
            payload={"status": status.value, "error": error},
        )
        self.artifacts.finalize_run(ctx, status=status, error=error)

    def trace_node(self, ctx: RunContext, node: str, state: dict) -> None:
        """
        记录 LangGraph 节点进入事件 + 保存状态快照。

        同时将会话的最后 checkpoint_id 写入记忆系统。
        """
        self.artifacts.append_trace(ctx, event_type="node_enter", node=node)
        self.checkpoint_mgr.snapshot_node(ctx, node, state)
        cp_id = self.checkpoint_mgr.get_latest_checkpoint_id(ctx.manifest.run_id)
        if cp_id:
            self.memory.set_checkpoint(ctx.manifest.chat_id, cp_id)

    async def run_react_agent(
        self,
        message: str,
        chat_id: str,
        run_ctx: RunContext | None = None,
    ) -> str:
        """通过 Harness 执行 ReAct Agent（非流式完整路径）。"""
        ctx = run_ctx or self.start_run(chat_id, mode="manus")
        history = self.memory.format_for_prompt(chat_id)

        executor = await self.create_tool_executor(ctx)
        llm = self.get_llm()
        agent = create_react_agent(
            llm,
            executor.get_langchain_tools(),
            prompt=SystemMessage(content=FIN_MANUS_SYSTEM),
            checkpointer=self.checkpoint_mgr.checkpointer,
        )

        user_content = message
        if history:
            user_content = f"{history}\n\n当前问题：{message}"

        config = self.checkpoint_mgr.thread_config(chat_id, ctx.manifest.run_id)
        self.artifacts.append_trace(ctx, event_type="react_invoke", payload={"message": message[:200]})

        try:
            result = await agent.ainvoke(
                {"messages": [HumanMessage(content=user_content)]},
                config=config,
            )
            messages = result.get("messages", [])
            answer = messages[-1].content if messages else "未能生成回答。"
            self.finish_run(ctx, RunStatus.COMPLETED)
            return answer
        except Exception as e:
            self.finish_run(ctx, RunStatus.FAILED, error=str(e))
            raise

    def resume_from_checkpoint(self, run_id: str, checkpoint_id: str) -> dict | None:
        """从工件级 checkpoint 快照恢复状态（只读）。"""
        return self.checkpoint_mgr.restore_state(run_id, checkpoint_id)

    def restore_execution_chain(self, session_id: str, checkpoint_id: str):
        """根据 session_id 从 checkpoint 恢复完整执行链路。"""
        return self.checkpoint_mgr.restore_execution_chain(session_id, checkpoint_id)

    def get_run_manifest(self, run_id: str):
        """获取运行清单。"""
        return self.artifacts.load_manifest(run_id)

    def list_runs(self, chat_id: str | None = None):
        """列出历史运行记录。"""
        return self.artifacts.list_runs(chat_id)

    def get_tool_catalog(self) -> list[dict[str, Any]]:
        """返回 7 类工具的元信息目录。"""
        from app.harness.tools.registry import get_tool_specs

        return [s.model_dump() for s in get_tool_specs()]


@lru_cache
def get_harness() -> AgentHarness:
    """单例获取 Harness 实例。"""
    return AgentHarness()
