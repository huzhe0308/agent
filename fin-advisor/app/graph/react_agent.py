"""
ReAct 超级智能体（FinManus）。

与 workflow.py 的固定流水线不同，本模块提供开放式 ReAct Agent：
- 自主决定何时调用工具、调用哪个工具
- 加载全部 7 类同步工具
- 挂载 LangGraph Checkpointer 支持多轮状态恢复
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from app.harness.harness import get_harness
from app.harness.models import RunContext
from app.harness.providers.registry import get_llm
from app.harness.tools.registry import load_sync_tools

FIN_MANUS_SYSTEM = """你是 FinManus 金融超级智能体，具备 ReAct 推理与工具调用能力。

职责：
1. 理解用户金融理财诉求，必要时调用工具进行数值测算与知识检索
2. 给出专业、结构化的回答
3. 涉及投资建议时声明"仅供参考，不构成投资建议"
4. 复杂任务分步推理：思考 → 调用工具 → 汇总结论"""


@lru_cache
def get_react_agent():
    """
    创建并缓存 ReAct Agent 实例。

    使用 langgraph.prebuilt.create_react_agent 实现
    Think → Act → Observe 循环。
    """
    harness = get_harness()
    llm = get_llm()
    tools = load_sync_tools()  # 加载 7 类同步工具
    return create_react_agent(
        llm,
        tools,
        prompt=SystemMessage(content=FIN_MANUS_SYSTEM),
        checkpointer=harness.checkpoint_mgr.checkpointer,
    )


async def run_react(
    message: str,
    history: str = "",
    chat_id: str = "manus_default",
    run_ctx: RunContext | None = None,
) -> str:
    """同步调用 ReAct Agent（非流式，供内部或测试使用）。"""
    harness = get_harness()
    ctx = run_ctx or harness.start_run(chat_id, mode="manus")

    user_content = message
    if history:
        user_content = f"{history}\n\n当前问题：{message}"

    agent = get_react_agent()
    config = harness.checkpoint_mgr.thread_config(chat_id, ctx.manifest.run_id)

    result = await agent.ainvoke(
        {"messages": [HumanMessage(content=user_content)]},
        config=config,
    )
    messages = result.get("messages", [])
    return messages[-1].content if messages else "未能生成回答，请稍后重试。"
