"""
API 请求/响应 与 核心业务数据模型。
"""

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class IntentType(str, Enum):
    """主意图枚举，决定 LangGraph 工作流路由。"""

    QA = "qa"
    CALCULATION = "calculation"
    COMPLEX_PLANNING = "complex_planning"
    REPORT_GENERATION = "report_generation"
    REGULATORY = "regulatory"
    PRODUCT = "product"
    GENERAL = "general"


class KnowledgeDomain(str, Enum):
    """RAG 知识库域：对应百炼不同 Index。"""

    KNOWLEDGE = "knowledge"
    PRODUCT = "product"
    REGULATORY = "regulatory"


class ChatRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    message: str = Field(..., min_length=1, description="用户消息")
    chat_id: str = Field(..., alias="chatId", description="会话 ID")
    risk_profile: str | None = Field(default=None, description="风险偏好：保守/稳健/积极")


class IntentResult(BaseModel):
    """多意图识别结果。"""

    intent: IntentType = Field(description="主意图（兼容旧路由）")
    intents: list[str] = Field(default_factory=list, description="多标签意图列表")
    confidence: float = 0.0
    reasoning: str = ""
    requires_decompose: bool = Field(
        default=False,
        description="是否需拆解为多步子任务（复杂计算/规划）",
    )
    knowledge_domains: list[KnowledgeDomain] = Field(
        default_factory=list,
        description="建议检索的知识域及权重依据",
    )


class FinancialEntity(BaseModel):
    """从对话中提取的金融实体。"""

    entity_type: str = Field(description="stock_code|metric|product|amount|policy|other")
    value: str
    normalized: str = ""


class RewriteResult(BaseModel):
    """上下文感知重写结果。"""

    rewritten_query: str
    compressed_context: str = Field(default="", description="压缩后的关键上下文（≤120字）")
    extracted_entities: list[FinancialEntity] = Field(default_factory=list)
    needs_context: bool = False
    dropped_noise: list[str] = Field(default_factory=list, description="被剔除的寒暄/无效内容")


class SubTask(BaseModel):
    """可执行子任务（拆解链输出）。"""

    step: int
    action: Literal["retrieve", "calculate", "analyze", "compare"] = "retrieve"
    description: str
    query: str
    knowledge_domain: KnowledgeDomain | None = None


class DecomposeResult(BaseModel):
    """复杂问题拆解结果。"""

    sub_questions: list[str] = Field(default_factory=list)
    sub_tasks: list[SubTask] = Field(default_factory=list)
    is_complex: bool = False
    execution_order: Literal["sequential", "parallel"] = "sequential"


class SessionMessage(BaseModel):
    role: str
    content: str
    turn_id: int = 0
    timestamp: str = ""


class SemanticFact(BaseModel):
    key: str
    value: str
    confidence: float = 1.0
    source_turn: int = 0


class EpisodicTurn(BaseModel):
    turn_id: int
    role: str
    content: str
    timestamp: str = ""
    importance: float = 0.5
    intent: str | None = None


class SessionData(BaseModel):
    chat_id: str
    messages: list[SessionMessage] = Field(default_factory=list)
    episodes: list[EpisodicTurn] = Field(default_factory=list)
    semantic_facts: list[SemanticFact] = Field(default_factory=list)
    summary: str = ""
    turn_counter: int = 0
    last_checkpoint_id: str | None = None
    updated_at: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryContext(BaseModel):
    chat_id: str
    summary: str = ""
    working_memory: list[SessionMessage] = Field(default_factory=list)
    episodic_memory: list[EpisodicTurn] = Field(default_factory=list)
    semantic_facts: list[SemanticFact] = Field(default_factory=list)
    last_checkpoint_id: str | None = None


class ChatResponse(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    chat_id: str = Field(alias="chatId")
    answer: str
    intent: str | None = None
    run_id: str | None = None


class HealthResponse(BaseModel):
    status: str
    service: str


class RunSummary(BaseModel):
    run_id: str
    chat_id: str
    mode: str
    status: str
    started_at: str
    finished_at: str | None = None
    tools_used: list[str] = Field(default_factory=list)
    tool_calls_count: int = 0


class ResumeRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    run_id: str = Field(alias="runId")
    checkpoint_id: str = Field(alias="checkpointId")
    message: str | None = None
