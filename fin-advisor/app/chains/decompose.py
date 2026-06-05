"""
LCEL 复杂问题拆解链。

当意图为 complex_planning 或 requires_decompose=true 时，
将单问题拆解为有序子任务（如：先查财报 → 再算市盈率 → 再对比行业）。
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from app.llm.factory import get_llm
from app.models.schemas import DecomposeResult, IntentResult, SubTask

DECOMPOSE_SYSTEM = """你是金融理财「复杂任务规划器」。将用户复杂问题拆解为 2-5 个**可独立执行**的子任务。

## 子任务 action 类型
| action | 适用场景 |
|--------|----------|
| retrieve | 需查知识库/财报/政策/产品文档 |
| calculate | 需数值测算（复利、定投、市盈率等） |
| analyze | 需综合分析、风险评估 |
| compare | 需对比多个产品/策略 |

## knowledge_domain（retrieve 类任务必填）
knowledge | product | regulatory

## 拆解原则
1. 多步依赖用 sequential（如先 retrieve 再 calculate）
2. 独立检索用 parallel
3. 每步 description ≤ 30字，query ≤ 60字
4. sub_questions 与 sub_tasks 的 query 保持一致（兼容旧接口）
5. 简单单一问题：is_complex=false，sub_tasks 仅 1 个

## 示例
问题：「帮我查一下贵州茅台最近财报，算一下市盈率，和白酒行业比怎么样」
→ step1 retrieve(regulatory/knowledge) 查财报
→ step2 calculate 算 PE
→ step3 compare 与行业对比

{format_instructions}"""

DECOMPOSE_HUMAN = """主意图：{primary_intent}
多标签：{intents}
已重写问题：{question}
已提取实体：{entities}"""


def build_decompose_chain() -> Runnable:
    parser = PydanticOutputParser(pydantic_object=DecomposeResult)
    prompt = ChatPromptTemplate.from_messages([
        ("system", DECOMPOSE_SYSTEM),
        ("human", DECOMPOSE_HUMAN),
    ])
    return (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | get_llm()
        | parser
    )


def _fallback(question: str) -> DecomposeResult:
    return DecomposeResult(
        sub_questions=[question],
        sub_tasks=[SubTask(step=1, action="retrieve", description="直接回答", query=question)],
        is_complex=False,
    )


def _sync_sub_questions(result: DecomposeResult) -> DecomposeResult:
    """确保 sub_questions 与 sub_tasks 同步。"""
    if result.sub_tasks:
        result.sub_questions = [t.query for t in result.sub_tasks]
    elif result.sub_questions:
        result.sub_tasks = [
            SubTask(step=i + 1, action="retrieve", description=f"子问题{i+1}", query=q)
            for i, q in enumerate(result.sub_questions)
        ]
    result.is_complex = len(result.sub_tasks) > 1
    return result


def decompose_question(
    question: str,
    intent_result: IntentResult | None = None,
    entities: str = "",
) -> DecomposeResult:
    """
    执行问题拆解。

    若 intent 不要求拆解且问题较短，快速返回单任务。
    """
    if intent_result and not intent_result.requires_decompose and len(question) < 40:
        return _fallback(question)

    primary = intent_result.intent.value if intent_result else "qa"
    intents = ",".join(intent_result.intents) if intent_result else primary

    chain = build_decompose_chain()
    try:
        result = chain.invoke({
            "primary_intent": primary,
            "intents": intents,
            "question": question,
            "entities": entities or "无",
        })
        return _sync_sub_questions(result)
    except Exception:
        return _fallback(question)


def should_decompose(intent_result: IntentResult | None) -> bool:
    """判断是否进入拆解节点。"""
    if not intent_result:
        return False
    return intent_result.requires_decompose or intent_result.intent.value == "complex_planning"
