"""
LangGraph 金融咨询主工作流（升级版）。

流水线：
  intent → rewrite → [条件 decompose] → retrieve(多域RAG) → [路由] → 生成/计算
"""

from functools import lru_cache

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

from app.chains.decompose import decompose_question
from app.chains.generate import generate_answer
from app.chains.intent import classify_intent
from app.chains.rewrite import rewrite_query
from app.graph.state import AdvisorState
from app.harness.harness import get_harness
from app.harness.models import RunContext
from app.harness.providers.registry import get_llm
from app.models.schemas import IntentResult, IntentType, KnowledgeDomain
from app.rag.bailian_retriever import retrieve_context

CALC_SYSTEM = """你是金融理财计算助手。根据用户问题调用合适的金融工具完成测算，
并用通俗易懂的语言解释结果。若参数不足，请说明需要补充的信息。"""

_harness = get_harness()
_run_ctx: RunContext | None = None


def _set_run_ctx(ctx: RunContext | None) -> None:
    global _run_ctx
    _run_ctx = ctx


def _trace(node: str, state: AdvisorState) -> None:
    if _run_ctx:
        _harness.trace_node(_run_ctx, node, dict(state))


def _intent_from_state(state: AdvisorState) -> IntentResult:
    domains = []
    for d in state.get("knowledge_domains") or []:
        try:
            domains.append(KnowledgeDomain(d))
        except ValueError:
            pass
    return IntentResult(
        intent=IntentType(state.get("intent", IntentType.GENERAL.value)),
        intents=state.get("intents") or [],
        reasoning=state.get("intent_reasoning", ""),
        requires_decompose=state.get("requires_decompose", False),
        knowledge_domains=domains,
    )


def _node_intent(state: AdvisorState) -> dict:
    _trace("intent", state)
    summary = (state.get("history") or "")[:300]
    result = classify_intent(state["message"], history_summary=summary)
    return {
        "intent": result.intent.value,
        "intents": result.intents,
        "intent_reasoning": result.reasoning,
        "requires_decompose": result.requires_decompose,
        "knowledge_domains": [d.value for d in result.knowledge_domains],
        "_intent_result": result,  # 节点间传递（不持久化到 checkpoint 外）
    }


def _node_rewrite(state: AdvisorState) -> dict:
    _trace("rewrite", state)
    intent_hint = ",".join(state.get("intents") or [state.get("intent", "")])
    result = rewrite_query(state["message"], state.get("history", ""), intent_hint=intent_hint)
    return {
        "rewritten_query": result.rewritten_query,
        "compressed_context": result.compressed_context,
        "extracted_entities": [e.model_dump() for e in result.extracted_entities],
    }


def _node_decompose(state: AdvisorState) -> dict:
    _trace("decompose", state)
    intent_result = _intent_from_state(state)
    entities = "；".join(
        f"{e.get('entity_type')}:{e.get('value')}"
        for e in (state.get("extracted_entities") or [])
    )
    query = state.get("rewritten_query") or state["message"]
    result = decompose_question(query, intent_result=intent_result, entities=entities)
    return {
        "sub_questions": result.sub_questions,
        "sub_tasks": [t.model_dump() for t in result.sub_tasks],
    }


def _node_retrieve(state: AdvisorState) -> dict:
    _trace("retrieve", state)
    intent_result = _intent_from_state(state)
    questions = state.get("sub_questions") or [state.get("rewritten_query") or state["message"]]
    context = retrieve_context(questions, intent_result=intent_result)
    return {"context": context}


def _node_generate_qa(state: AdvisorState) -> dict:
    _trace("generate_qa", state)
    enriched_history = state.get("history", "")
    if state.get("compressed_context"):
        enriched_history = f"{state['compressed_context']}\n{enriched_history}"
    answer = generate_answer(
        question=state.get("rewritten_query") or state["message"],
        history=enriched_history,
        context=state.get("context", ""),
        is_report=False,
        risk_profile=state.get("risk_profile"),
    )
    return {"answer": answer}


def _node_generate_report(state: AdvisorState) -> dict:
    _trace("generate_report", state)
    answer = generate_answer(
        question=state.get("rewritten_query") or state["message"],
        history=state.get("history", ""),
        context=state.get("context", ""),
        is_report=True,
        risk_profile=state.get("risk_profile"),
    )
    return {"answer": answer}


def _node_calc_react(state: AdvisorState) -> dict:
    _trace("calc_react", state)
    executor = _harness.create_sync_tool_executor(
        _run_ctx, session_id=state.get("chat_id", "default"),
    )
    agent = create_react_agent(
        get_llm(), executor.get_langchain_tools(),
        prompt=SystemMessage(content=CALC_SYSTEM),
    )
    user_content = state["message"]
    if state.get("context"):
        user_content = f"参考资料：\n{state['context']}\n\n用户问题：{state['message']}"
    if state.get("compressed_context"):
        user_content = f"关键上下文：{state['compressed_context']}\n\n{user_content}"
    config = _harness.checkpoint_mgr.thread_config(
        state.get("chat_id", "default"),
        _run_ctx.manifest.run_id if _run_ctx else None,
    )
    result = agent.invoke({"messages": [HumanMessage(content=user_content)]}, config=config)
    messages = result.get("messages", [])
    answer = messages[-1].content if messages else "计算失败，请检查参数后重试。"
    return {"answer": answer}


def _route_after_rewrite(state: AdvisorState) -> str:
    if state.get("requires_decompose"):
        return "decompose"
    return "retrieve"


def _route_after_retrieve(state: AdvisorState) -> str:
    intent = state.get("intent", IntentType.GENERAL.value)
    if intent in (IntentType.CALCULATION.value, IntentType.COMPLEX_PLANNING.value):
        return "calc_react"
    if intent == IntentType.REPORT_GENERATION.value:
        return "generate_report"
    return "generate_qa"


def _node_skip_decompose(state: AdvisorState) -> dict:
    """不拆解时，将重写后问题作为唯一子问题。"""
    q = state.get("rewritten_query") or state["message"]
    return {"sub_questions": [q], "sub_tasks": []}


def _build_graph():
    graph = StateGraph(AdvisorState)

    graph.add_node("intent", _node_intent)
    graph.add_node("rewrite", _node_rewrite)
    graph.add_node("decompose", _node_decompose)
    graph.add_node("skip_decompose", _node_skip_decompose)
    graph.add_node("retrieve", _node_retrieve)
    graph.add_node("generate_qa", _node_generate_qa)
    graph.add_node("generate_report", _node_generate_report)
    graph.add_node("calc_react", _node_calc_react)

    graph.set_entry_point("intent")
    graph.add_edge("intent", "rewrite")
    graph.add_conditional_edges(
        "rewrite",
        _route_after_rewrite,
        {"decompose": "decompose", "retrieve": "skip_decompose"},
    )
    graph.add_edge("decompose", "retrieve")
    graph.add_edge("skip_decompose", "retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _route_after_retrieve,
        {
            "calc_react": "calc_react",
            "generate_qa": "generate_qa",
            "generate_report": "generate_report",
        },
    )
    graph.add_edge("generate_qa", END)
    graph.add_edge("generate_report", END)
    graph.add_edge("calc_react", END)
    return graph


@lru_cache
def build_advisor_graph():
    return _build_graph().compile(checkpointer=_harness.checkpoint_mgr.checkpointer)


def run_advisor_workflow(
    message: str,
    chat_id: str,
    history: str = "",
    risk_profile: str | None = None,
    run_ctx: RunContext | None = None,
) -> dict:
    _set_run_ctx(run_ctx)
    graph = build_advisor_graph()
    config = _harness.checkpoint_mgr.thread_config(
        chat_id, run_ctx.manifest.run_id if run_ctx else None,
    )
    result = graph.invoke(
        {
            "chat_id": chat_id,
            "message": message,
            "history": history,
            "risk_profile": risk_profile,
        },
        config=config,
    )
    return result
