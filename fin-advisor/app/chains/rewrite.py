"""
LCEL 上下文感知问题重写链。

核心能力：
- 消解指代，生成可独立检索的完整问句
- 提取金融实体（股票代码、指标、产品、金额）
- 压缩历史为关键上下文，剔除寒暄废话
- 严格控制输出 Token（JSON 字段有字数上限）
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.llm.factory import get_llm
from app.models.schemas import FinancialEntity, RewriteResult

REWRITE_SYSTEM = """你是金融理财咨询系统的「上下文感知问题重写器」。

## 任务
1. 结合对话历史，将用户最新问题改写为**可独立理解**的完整问句（rewritten_query，≤80字）
2. 从历史中提炼**金融关键信息**写入 compressed_context（≤120字），仅保留：
   - 股票/基金代码、产品名称
   - 财务指标（PE、PB、ROE、收益率等）及数值
   - 金额、期限、风险偏好
   - 已达成共识的结论
3. 提取结构化实体 extracted_entities（最多 8 个）
4. 记录被剔除的寒暄/废话 dropped_noise（如「你好」「谢谢」「天气不错」）

## 禁止
- 不要复述整段历史
- 不要保留无效社交寒暄
- 不要编造历史中不存在的数字或代码
- 各字符串字段严格遵守字数上限

## 实体类型 entity_type
stock_code | metric | product | amount | policy | risk_profile | other

{format_instructions}"""

REWRITE_HUMAN = """对话历史（可能含噪声）：
{history}

用户最新问题：
{question}

意图提示（可选）：{intent_hint}"""


def build_rewrite_chain() -> Runnable:
    parser = PydanticOutputParser(pydantic_object=RewriteResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", REWRITE_SYSTEM),
        ("human", REWRITE_HUMAN),
    ])
    return (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | get_llm()
        | parser
    )


def _fallback_rewrite(question: str, history: str) -> RewriteResult:
    return RewriteResult(
        rewritten_query=question,
        compressed_context="",
        needs_context=bool(history and history.strip()),
    )


def _post_process(result: RewriteResult, question: str) -> RewriteResult:
    """硬截断，防止 LLM 超长输出。"""
    result.rewritten_query = result.rewritten_query.strip()[:120] or question
    result.compressed_context = result.compressed_context.strip()[:150]
    result.dropped_noise = [n[:30] for n in (result.dropped_noise or [])[:5]]
    result.extracted_entities = result.extracted_entities[:8]
    if not result.rewritten_query:
        result.rewritten_query = question
    return result


def rewrite_query(
    question: str,
    history: str = "",
    intent_hint: str = "",
) -> RewriteResult:
    """执行上下文感知重写。"""
    chain = build_rewrite_chain()
    try:
        result = chain.invoke({
            "question": question,
            "history": _compress_history_input(history),
            "intent_hint": intent_hint or "无",
        })
        return _post_process(result, question)
    except Exception:
        return _fallback_rewrite(question, history)


def _compress_history_input(history: str, max_chars: int = 2000) -> str:
    """送入 LLM 前截断历史，控制输入 Token。"""
    h = (history or "（无历史）").strip()
    if len(h) <= max_chars:
        return h
    # 保留尾部（近期对话更重要）
    return "…（前文已省略）\n" + h[-max_chars:]


def format_rewrite_for_prompt(result: RewriteResult) -> str:
    """将重写结果格式化为下游链可用的紧凑上下文。"""
    parts: list[str] = []
    if result.compressed_context:
        parts.append(f"[关键上下文] {result.compressed_context}")
    if result.extracted_entities:
        ents = "；".join(f"{e.entity_type}:{e.value}" for e in result.extracted_entities)
        parts.append(f"[实体] {ents}")
    parts.append(f"[问题] {result.rewritten_query}")
    return "\n".join(parts)


def estimate_rewrite_tokens(result: RewriteResult) -> int:
    """粗估重写输出字符数（用于评测压缩率）。"""
    text = result.rewritten_query + result.compressed_context
    text += "".join(e.value for e in result.extracted_entities)
    return len(text)
