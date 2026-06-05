"""
7 类工具统一注册与加载。

工具类型与加载函数对应关系：
  1. STRUCTURED  → load_structured_tools()  本地金融计算
  2. MCP         → load_mcp_tools()         MCP 协议外部工具
  3. RAG         → load_rag_tools()         百炼知识库检索
  4. CHAIN       → load_chain_tools()       LCEL 链封装
  5. HTTP        → load_http_tools()        外部 HTTP API
  6. SUBAGENT    → load_subagent_tools()    子 Agent 委托
  7. BUILTIN     → load_builtin_tools()     内置控制工具

对外接口：
  - load_sync_tools()：同步加载（不含 MCP）
  - load_all_tools()：异步加载全部（含 MCP）
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import httpx
from langchain_core.tools import StructuredTool

from app.chains.intent import classify_intent
from app.config import get_settings
from app.harness.enums import ToolType
from app.harness.models import ToolSpec
from app.rag.bailian_retriever import retrieve_context
from app.tools.financial import get_financial_tools


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ------------------------------------------------------------------ #
# 1. STRUCTURED
# ------------------------------------------------------------------ #
def load_structured_tools() -> list[StructuredTool]:
    return get_financial_tools()


# ------------------------------------------------------------------ #
# 2. MCP
# ------------------------------------------------------------------ #
async def load_mcp_tools() -> list[StructuredTool]:
    settings = get_settings()
    if not settings.mcp_enable:
        return []

    from pathlib import Path

    servers_file = Path(settings.mcp_servers_file)
    if not servers_file.exists():
        return []

    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
    except ImportError:
        return []

    config = json.loads(servers_file.read_text(encoding="utf-8"))
    mcp_tools: list[StructuredTool] = []

    for name, server_cfg in config.get("mcpServers", {}).items():
        command = server_cfg.get("command")
        args = server_cfg.get("args", [])
        if not command:
            continue
        try:
            params = StdioServerParameters(command=command, args=args)
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    for tool in tools_result.tools:
                        mcp_tools.append(
                            StructuredTool.from_function(
                                coroutine=_make_mcp_caller(session, tool.name),
                                name=f"mcp_{name}_{tool.name}",
                                description=tool.description or f"MCP tool: {tool.name}",
                                metadata={"tool_type": ToolType.MCP.value},
                            )
                        )
        except Exception:
            continue

    return mcp_tools


def _make_mcp_caller(session, tool_name: str):
    async def _call(**kwargs):
        result = await session.call_tool(tool_name, arguments=kwargs)
        texts = [c.text for c in result.content if hasattr(c, "text")]
        return "\n".join(texts) if texts else str(result)

    return _call


# ------------------------------------------------------------------ #
# 3. RAG
# ------------------------------------------------------------------ #
def _rag_search(query: str, top_k: int = 5) -> str:
    """从百炼知识库检索理财相关内容。"""
    return retrieve_context([query]) or "未检索到相关资料。"


def load_rag_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_rag_search,
            name="knowledge_search",
            description="从理财知识库检索产品文档、监管政策、理财知识",
            metadata={"tool_type": ToolType.RAG.value},
        )
    ]


# ------------------------------------------------------------------ #
# 4. CHAIN（LCEL 链封装为工具）
# ------------------------------------------------------------------ #
def _classify_user_intent(question: str) -> str:
    result = classify_intent(question)
    return f"意图: {result.intent.value}, 置信度: {result.confidence}, 理由: {result.reasoning}"


def load_chain_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_classify_user_intent,
            name="intent_classifier",
            description="对用户问题进行意图分类（qa/calculation/report/general）",
            metadata={"tool_type": ToolType.CHAIN.value},
        )
    ]


# ------------------------------------------------------------------ #
# 5. HTTP（外部 API 工具）
# ------------------------------------------------------------------ #
def _http_fetch(url: str, method: str = "GET") -> str:
    """调用外部 HTTP API 获取数据（如行情、利率等）。"""
    try:
        with httpx.Client(timeout=15.0) as client:
            if method.upper() == "POST":
                resp = client.post(url)
            else:
                resp = client.get(url)
            resp.raise_for_status()
            return resp.text[:4000]
    except Exception as e:
        return f"HTTP 请求失败: {e}"


def load_http_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_http_fetch,
            name="http_fetch",
            description="通过 HTTP 获取外部金融数据（URL 需为公开可访问接口）",
            metadata={"tool_type": ToolType.HTTP.value},
        )
    ]


# ------------------------------------------------------------------ #
# 6. SUBAGENT（子 Agent 委托）
# ------------------------------------------------------------------ #
def _invoke_calc_agent(question: str) -> str:
    """委托计算子 Agent 处理数值测算问题。"""
    from langchain_core.messages import HumanMessage, SystemMessage
    from langgraph.prebuilt import create_react_agent

    from app.harness.providers.registry import get_llm

    system = "你是金融计算子 Agent，专注调用计算工具完成测算。"
    agent = create_react_agent(
        get_llm(),
        load_structured_tools(),
        prompt=SystemMessage(content=system),
    )
    result = agent.invoke(
        {"messages": [HumanMessage(content=question)]},
        config={"recursion_limit": 10},
    )
    messages = result.get("messages", [])
    return messages[-1].content if messages else "子 Agent 未返回结果"


def load_subagent_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_invoke_calc_agent,
            name="calc_subagent",
            description="委托专业计算子 Agent 处理复杂数值测算任务",
            metadata={"tool_type": ToolType.SUBAGENT.value},
        )
    ]


# ------------------------------------------------------------------ #
# 7. BUILTIN（内置控制工具）
# ------------------------------------------------------------------ #
def _get_current_time() -> str:
    return _utc_now()


def _do_terminate(reason: str = "任务完成") -> str:
    return f"TERMINATE: {reason}"


def load_builtin_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=_get_current_time,
            name="get_current_time",
            description="获取当前 UTC 时间",
            metadata={"tool_type": ToolType.BUILTIN.value},
        ),
        StructuredTool.from_function(
            func=_do_terminate,
            name="do_terminate",
            description="标记当前 Agent 任务已完成并终止循环",
            metadata={"tool_type": ToolType.BUILTIN.value},
        ),
    ]


# ------------------------------------------------------------------ #
# Registry
# ------------------------------------------------------------------ #
_TOOL_LOADERS: dict[ToolType, Any] = {
    ToolType.STRUCTURED: load_structured_tools,
    ToolType.RAG: load_rag_tools,
    ToolType.CHAIN: load_chain_tools,
    ToolType.HTTP: load_http_tools,
    ToolType.SUBAGENT: load_subagent_tools,
    ToolType.BUILTIN: load_builtin_tools,
}


def get_tool_specs() -> list[ToolSpec]:
    specs: list[ToolSpec] = []
    for tool_type, loader in _TOOL_LOADERS.items():
        for tool in loader():
            specs.append(
                ToolSpec(
                    name=tool.name,
                    tool_type=tool_type,
                    description=tool.description or "",
                )
            )
    specs.append(
        ToolSpec(
            name="mcp_*",
            tool_type=ToolType.MCP,
            description="MCP 服务器动态发现工具（MCP_ENABLE=true）",
            enabled=get_settings().mcp_enable,
        )
    )
    return specs


def load_sync_tools(enabled_types: list[ToolType] | None = None) -> list[StructuredTool]:
    """加载除 MCP 外的同步工具。"""
    types = enabled_types or list(_TOOL_LOADERS.keys())
    tools: list[StructuredTool] = []
    for tool_type in types:
        loader = _TOOL_LOADERS.get(tool_type)
        if loader:
            tools.extend(loader())
    return tools


async def load_all_tools(enabled_types: list[ToolType] | None = None) -> list[StructuredTool]:
    """加载全部 7 类工具（含 MCP）。"""
    tools = load_sync_tools(enabled_types)
    if get_settings().mcp_enable:
        tools.extend(await load_mcp_tools())
    return tools
