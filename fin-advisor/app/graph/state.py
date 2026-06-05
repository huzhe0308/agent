"""LangGraph 工作流状态定义。"""

from typing import Annotated, Any

from typing_extensions import TypedDict

from langgraph.graph.message import add_messages


class AdvisorState(TypedDict, total=False):
    chat_id: str
    message: str
    history: str
    risk_profile: str | None

    # 理解阶段
    intent: str
    intents: list[str]
    intent_reasoning: str
    requires_decompose: bool
    knowledge_domains: list[str]

    rewritten_query: str
    compressed_context: str
    extracted_entities: list[dict[str, Any]]

    sub_questions: list[str]
    sub_tasks: list[dict[str, Any]]

    context: str
    answer: str
    messages: Annotated[list, add_messages]
