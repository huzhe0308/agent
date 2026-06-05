"""委托 Harness 统一模型接入。"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.harness.providers.registry import get_llm as _get_llm


def get_llm() -> BaseChatModel:
    return _get_llm()
