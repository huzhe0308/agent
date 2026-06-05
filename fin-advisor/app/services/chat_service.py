"""
对话服务层 —— 整个系统的编排中枢。

职责：
1. 接收 API 层传入的用户消息
2. 通过 AgentMemoryManager 加载/更新四层记忆
3. 通过 AgentHarness 创建运行上下文（run_id + trace）
4. 调用 LangGraph 工作流或 ReAct Agent 生成回答
5. 流式输出（SSE）并落盘运行工件
"""

from collections.abc import AsyncIterator

from langchain_core.messages import HumanMessage

from app.chains.generate import build_qa_chain, build_report_chain
from app.config import get_settings
from app.graph.react_agent import get_react_agent
from app.graph.workflow import run_advisor_workflow
from app.harness.enums import RunStatus
from app.harness.harness import get_harness
from app.harness.memory.manager import agent_memory
from app.models.schemas import IntentType


async def _stream_llm_chain(chain, inputs: dict) -> AsyncIterator[str]:
    """将 LCEL 链的流式输出逐 token 透传。"""
    async for chunk in chain.astream(inputs):
        if chunk:
            yield chunk


async def chat_stream(
    message: str,
    chat_id: str,
    risk_profile: str | None = None,
) -> AsyncIterator[str]:
    """
    金融理财咨询 SSE 流式对话（主路径）。

    执行流程：
    1. 加载记忆 → 写入用户消息
    2. harness.start_run() 创建 run_id
    3. run_advisor_workflow() 执行 LangGraph 流水线
    4. 分块流式输出答案
    5. 写入助手消息 → 更新摘要 → finish_run()
    """
    harness = get_harness()
    memory = agent_memory

    # 若前端传入风险偏好，写入语义记忆（用户画像）
    if risk_profile:
        memory.upsert_semantic_fact(chat_id, "risk_profile", risk_profile)

    # 组装四层记忆为 Prompt 文本
    history = memory.format_for_prompt(chat_id)
    memory.append_turn(chat_id, "user", message)

    # 创建本次运行上下文，开始记录 trace
    run_ctx = harness.start_run(chat_id, mode="advisor", metadata={"risk_profile": risk_profile})

    try:
        # 执行 LangGraph 主工作流（意图→重写→拆解→检索→生成）
        result = run_advisor_workflow(
            message=message,
            chat_id=chat_id,
            history=history,
            risk_profile=risk_profile,
            run_ctx=run_ctx,
        )

        answer = result.get("answer", "")
        intent = result.get("intent", IntentType.GENERAL.value)
        rewritten = result.get("rewritten_query", message)

        if answer:
            # 工作流节点已生成完整答案，按 8 字符分块模拟流式
            chunk_size = 8
            for i in range(0, len(answer), chunk_size):
                yield answer[i : i + chunk_size]
        else:
            # 回退：工作流未产出答案时，直接流式调用 LCEL 生成链
            is_report = intent == IntentType.REPORT_GENERATION.value
            chain = build_report_chain() if is_report else build_qa_chain()
            inputs = {
                "question": rewritten,
                "history": history or "（无历史）",
                "context": result.get("context", "") or "（未检索到相关资料）",
                "extra_instruction": "",
            }
            answer_parts: list[str] = []
            async for token in _stream_llm_chain(chain, inputs):
                answer_parts.append(token)
                yield token
            answer = "".join(answer_parts)

        # 持久化助手回复，并记录意图（用于 Episodic Memory）
        memory.append_turn(chat_id, "assistant", answer, intent=intent)
        harness.finish_run(run_ctx, RunStatus.COMPLETED)

        # 滚动摘要：压缩长对话，控制 Token
        settings = get_settings()
        if settings.enable_summary:
            memory.refresh_summary(chat_id, recent_turns=3)

    except Exception as e:
        harness.finish_run(run_ctx, RunStatus.FAILED, error=str(e))
        raise


async def manus_stream(message: str, chat_id: str) -> AsyncIterator[str]:
    """
    ReAct 超级智能体 SSE 流式对话。

    与 chat_stream 的区别：
    - 不走 LangGraph 固定流水线，而是 ReAct 自由推理
    - 可自主决定调用哪些工具（7 类工具全部可用）
    - 通过 astream_events 实现真正的 LLM token 级流式
    """
    harness = get_harness()
    memory = agent_memory

    history = memory.format_for_prompt(chat_id)
    memory.append_turn(chat_id, "user", message)

    run_ctx = harness.start_run(chat_id, mode="manus")
    agent = get_react_agent()
    config = harness.checkpoint_mgr.thread_config(chat_id, run_ctx.manifest.run_id)

    user_content = message
    if history:
        user_content = f"{history}\n\n当前问题：{message}"

    answer_parts: list[str] = []
    try:
        # 监听 LangGraph 事件流，捕获 LLM 逐 token 输出
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=user_content)]},
            version="v2",
            config=config,
        ):
            kind = event.get("event")
            if kind == "on_chat_model_stream":
                chunk = event.get("data", {}).get("chunk")
                if chunk and hasattr(chunk, "content") and chunk.content:
                    answer_parts.append(chunk.content)
                    yield chunk.content

        answer = "".join(answer_parts)
        if not answer:
            answer = "未能生成回答，请稍后重试。"
            yield answer

        memory.append_turn(chat_id, "assistant", answer)
        harness.finish_run(run_ctx, RunStatus.COMPLETED)

        settings = get_settings()
        if settings.enable_summary:
            memory.refresh_summary(chat_id, recent_turns=2)

    except Exception as e:
        harness.finish_run(run_ctx, RunStatus.FAILED, error=str(e))
        raise
