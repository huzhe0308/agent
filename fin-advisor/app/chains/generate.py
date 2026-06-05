"""
LCEL 答案生成链。

提供两种生成模式：
  - build_qa_chain()：普通理财问答（基于 RAG 上下文 + 记忆）
  - build_report_chain()：投资/理财分析报告（Markdown 格式）

由 workflow 的 generate_qa / generate_report 节点调用。
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

from app.llm.factory import get_llm

QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是专业的金融理财顾问助手。基于检索到的知识库内容与对话历史回答用户问题。

要求：
1. 回答准确、专业、易懂，必要时给出风险提示
2. 若知识库无相关信息，诚实说明并给出通用建议
3. 涉及具体投资建议时，声明"仅供参考，不构成投资建议"
4. 使用 Markdown 格式，结构清晰

{extra_instruction}""",
        ),
        (
            "human",
            """对话历史：
{history}

检索到的参考资料：
{context}

用户问题：{question}

请回答：""",
        ),
    ]
)

REPORT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是专业理财报告撰写助手。根据用户需求与参考资料，生成 Markdown 格式投资/理财分析报告。

报告结构建议：
# 报告标题
## 一、客户诉求摘要
## 二、市场与产品分析
## 三、风险评估
## 四、配置建议
## 五、免责声明

声明：本报告仅供参考，不构成投资建议。""",
        ),
        (
            "human",
            """对话历史：
{history}

参考资料：
{context}

报告需求：{question}""",
        ),
    ]
)


def build_qa_chain(extra_instruction: str = "") -> Runnable:
    return (
        QA_PROMPT.partial(extra_instruction=extra_instruction)
        | get_llm()
        | StrOutputParser()
    )


def build_report_chain() -> Runnable:
    return REPORT_PROMPT | get_llm() | StrOutputParser()


def generate_answer(
    question: str,
    history: str,
    context: str,
    *,
    is_report: bool = False,
    risk_profile: str | None = None,
) -> str:
    extra = ""
    if risk_profile:
        extra = f"用户风险偏好：{risk_profile}，请在建议中考虑该偏好。"

    if is_report:
        chain = build_report_chain()
    else:
        chain = build_qa_chain(extra_instruction=extra)

    return chain.invoke(
        {
            "question": question,
            "history": history or "（无历史）",
            "context": context or "（未检索到相关资料）",
        }
    )
