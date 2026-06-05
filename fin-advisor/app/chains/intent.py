"""
LCEL 多意图识别链。

支持多标签分类 + 主意图路由 + 复杂规划判定 + RAG 知识域建议。
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.llm.factory import get_llm
from app.models.schemas import IntentResult, IntentType, KnowledgeDomain

INTENT_SYSTEM = """你是金融理财咨询平台的「多意图分类器」。请分析用户最新问题，输出结构化 JSON。

## 意图标签（可多选，填入 intents 数组）
| 标签 | 含义 |
|------|------|
| qa | 理财概念、市场常识、投资原理 |
| calculation | 单一数值测算（复利/定投/收益率/贷款） |
| complex_planning | **复杂计算或规划**：需多步推理，如先查财报再算市盈率、先检索再对比再测算 |
| report_generation | 需要输出结构化投资/理财分析报告 |
| regulatory | 监管政策、合规、投资者保护 |
| product | 具体产品文档、费率、条款 |
| general | 闲聊或与理财无关 |

## 主意图 intent（单选，用于路由）
优先级：complex_planning > report_generation > calculation > regulatory > product > qa > general

## requires_decompose
当且仅当 intents 含 complex_planning，或问题明确需要「先A再B再C」多步执行时，设为 true。

## knowledge_domains（建议检索域，可多选）
- knowledge：理财知识、概念、策略
- product：产品说明书、费率、条款
- regulatory：法规、监管文件、合规要求

## 输出约束
- 仅输出 JSON，不要解释
- reasoning 不超过 40 字
- 对含股票代码(如600519)、财务指标(PE/PB/ROE)、金额、产品名的句子提高 qa/product 置信度

{format_instructions}"""

INTENT_HUMAN = """对话摘要（可选）：
{history_summary}

用户最新问题：
{question}"""


def build_intent_chain() -> Runnable:
    parser = PydanticOutputParser(pydantic_object=IntentResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", INTENT_SYSTEM),
        ("human", INTENT_HUMAN),
    ])
    return (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | get_llm()
        | parser
    )


def _normalize_intent(result: IntentResult) -> IntentResult:
    """后处理：根据多标签修正 requires_decompose 与 knowledge_domains。"""
    labels = set(result.intents or [result.intent.value])

    if IntentType.COMPLEX_PLANNING.value in labels:
        result.requires_decompose = True
        if result.intent == IntentType.GENERAL:
            result.intent = IntentType.COMPLEX_PLANNING

    if not result.knowledge_domains:
        domains: list[KnowledgeDomain] = []
        if "regulatory" in labels:
            domains.append(KnowledgeDomain.REGULATORY)
        if "product" in labels:
            domains.append(KnowledgeDomain.PRODUCT)
        if "qa" in labels or "complex_planning" in labels or "calculation" in labels:
            domains.append(KnowledgeDomain.KNOWLEDGE)
        result.knowledge_domains = domains or [KnowledgeDomain.KNOWLEDGE]

    if not result.intents:
        result.intents = [result.intent.value]

    return result


def classify_intent(question: str, history_summary: str = "") -> IntentResult:
    """多意图分类；失败时回退 general。"""
    chain = build_intent_chain()
    try:
        result = chain.invoke({
            "question": question,
            "history_summary": _trim(history_summary, 200),
        })
        return _normalize_intent(result)
    except Exception:
        return IntentResult(
            intent=IntentType.GENERAL,
            intents=["general"],
            confidence=0.5,
            reasoning="fallback",
        )


def _trim(text: str, max_chars: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= max_chars else text[:max_chars] + "…"
