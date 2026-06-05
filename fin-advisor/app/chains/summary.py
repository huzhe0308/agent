"""
LCEL 滚动摘要链。

长对话时，将新增回合合并到已有摘要中，
压缩 Token 消耗，保留关键信息（产品、建议、测算结果等）。
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.llm.factory import get_llm

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """你是对话摘要助手。将金融理财咨询对话压缩为简洁摘要（200字以内），保留：
- 用户关注的产品/主题
- 已给出的关键建议与测算结果
- 用户风险偏好等关键信息""",
        ),
        (
            "human",
            """已有摘要：
{existing_summary}

新增对话：
{new_messages}

请输出更新后的完整摘要：""",
        ),
    ]
)


def summarize(existing_summary: str, new_messages: str) -> str:
    """生成/更新滚动摘要，失败时返回原摘要。"""
    chain = SUMMARY_PROMPT | get_llm() | StrOutputParser()
    try:
        return chain.invoke(
            {
                "existing_summary": existing_summary or "（无）",
                "new_messages": new_messages,
            }
        )
    except Exception:
        return existing_summary
