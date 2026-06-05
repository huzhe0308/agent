"""
统一模型接入层 —— 2 类后端。

通过 MODEL_BACKEND 环境变量切换：
  - dashscope：阿里云通义千问（ChatTongyi）
  - openai_compat：OpenAI 兼容 API（Ollama / vLLM / OpenAI）

所有 LCEL 链、LangGraph 节点、ReAct Agent 均通过 get_llm() 获取模型。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import get_settings
from app.harness.enums import ModelBackend


class BaseModelProvider(ABC):
    """模型提供者抽象基类。"""

    backend: ModelBackend

    @abstractmethod
    def create(self) -> BaseChatModel:
        """创建并返回 ChatModel 实例。"""
        ...


class DashScopeProvider(BaseModelProvider):
    """后端1：阿里云 DashScope 通义千问。"""

    backend = ModelBackend.DASHSCOPE

    def create(self) -> BaseChatModel:
        from langchain_community.chat_models.tongyi import ChatTongyi

        settings = get_settings()
        return ChatTongyi(
            model=settings.llm_model,
            dashscope_api_key=settings.dashscope_api_key or None,
            streaming=True,
            temperature=settings.llm_temperature,
        )


class OpenAICompatProvider(BaseModelProvider):
    """后端2：OpenAI 兼容 API（适用于 Ollama、vLLM 等）。"""

    backend = ModelBackend.OPENAI_COMPAT

    def create(self) -> BaseChatModel:
        from langchain_openai import ChatOpenAI

        settings = get_settings()
        return ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key or "not-needed",
            base_url=settings.openai_base_url,
            streaming=True,
            temperature=settings.llm_temperature,
        )


# 后端注册表
_PROVIDERS: dict[ModelBackend, BaseModelProvider] = {
    ModelBackend.DASHSCOPE: DashScopeProvider(),
    ModelBackend.OPENAI_COMPAT: OpenAICompatProvider(),
}


def get_provider(backend: ModelBackend | None = None) -> BaseModelProvider:
    """根据配置或指定后端获取 Provider。"""
    settings = get_settings()
    resolved = backend or ModelBackend(settings.model_backend)
    return _PROVIDERS[resolved]


@lru_cache
def get_llm(backend: ModelBackend | None = None) -> BaseChatModel:
    """单例获取 LLM 实例（进程内缓存，避免重复创建连接）。"""
    return get_provider(backend).create()
