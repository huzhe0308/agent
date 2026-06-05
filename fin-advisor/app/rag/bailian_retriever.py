"""
百炼向量知识库 Retrieve API — 多域路由 + 加权合并。

支持三类内容召回：
  - knowledge：理财知识
  - product：产品文档
  - regulatory：监管政策

根据上游意图识别结果动态分配检索权重与 Index ID。
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from app.config import get_settings
from app.models.schemas import IntentResult, KnowledgeDomain

# 意图标签 → 知识域权重（归一化前）
DEFAULT_DOMAIN_WEIGHTS: dict[KnowledgeDomain, float] = {
    KnowledgeDomain.KNOWLEDGE: 1.0,
    KnowledgeDomain.PRODUCT: 0.0,
    KnowledgeDomain.REGULATORY: 0.0,
}

INTENT_DOMAIN_WEIGHTS: dict[str, dict[KnowledgeDomain, float]] = {
    "qa": {KnowledgeDomain.KNOWLEDGE: 0.7, KnowledgeDomain.PRODUCT: 0.2, KnowledgeDomain.REGULATORY: 0.1},
    "calculation": {KnowledgeDomain.KNOWLEDGE: 0.8, KnowledgeDomain.PRODUCT: 0.15, KnowledgeDomain.REGULATORY: 0.05},
    "complex_planning": {KnowledgeDomain.KNOWLEDGE: 0.5, KnowledgeDomain.PRODUCT: 0.3, KnowledgeDomain.REGULATORY: 0.2},
    "report_generation": {KnowledgeDomain.KNOWLEDGE: 0.4, KnowledgeDomain.PRODUCT: 0.35, KnowledgeDomain.REGULATORY: 0.25},
    "regulatory": {KnowledgeDomain.KNOWLEDGE: 0.1, KnowledgeDomain.PRODUCT: 0.1, KnowledgeDomain.REGULATORY: 0.8},
    "product": {KnowledgeDomain.KNOWLEDGE: 0.15, KnowledgeDomain.PRODUCT: 0.75, KnowledgeDomain.REGULATORY: 0.1},
    "general": {KnowledgeDomain.KNOWLEDGE: 0.5, KnowledgeDomain.PRODUCT: 0.25, KnowledgeDomain.REGULATORY: 0.25},
}

DOMAIN_LABELS = {
    KnowledgeDomain.KNOWLEDGE: "理财知识",
    KnowledgeDomain.PRODUCT: "产品文档",
    KnowledgeDomain.REGULATORY: "监管政策",
}


class BailianRetriever(BaseRetriever):
    """单 Index 百炼检索器。"""

    workspace_id: str = Field(default="")
    index_id: str = Field(default="")
    api_key: str = Field(default="")
    top_k: int = Field(default=5)
    domain: KnowledgeDomain = KnowledgeDomain.KNOWLEDGE
    endpoint: str = Field(default="https://bailian.aliyuncs.com/v2/index/retrieve")

    def _retrieve(self, query: str) -> list[Document]:
        settings = get_settings()
        if not settings.bailian_enable:
            return []

        workspace_id = self.workspace_id or settings.bailian_workspace_id
        index_id = self.index_id
        api_key = self.api_key or settings.dashscope_api_key

        if not workspace_id or not index_id or not api_key:
            return []

        payload = {
            "WorkspaceId": workspace_id,
            "IndexId": index_id,
            "Query": query,
            "DenseSimilarityTopK": self.top_k,
        }

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.post(
                    self.endpoint,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
        except Exception:
            return []

        docs = self._parse_documents(data)
        for d in docs:
            d.metadata["domain"] = self.domain.value
            d.metadata["domain_label"] = DOMAIN_LABELS[self.domain]
        return docs

    def _parse_documents(self, data: dict[str, Any]) -> list[Document]:
        docs: list[Document] = []
        nodes = data.get("Data", {}).get("Nodes") or data.get("nodes") or []
        for item in nodes:
            if isinstance(item, str):
                docs.append(Document(page_content=item))
                continue
            text = (
                item.get("Text") or item.get("text")
                or item.get("Content") or item.get("content") or ""
            )
            if not text:
                text = json.dumps(item, ensure_ascii=False)
            metadata = {k: v for k, v in item.items() if k not in ("Text", "text", "Content", "content")}
            docs.append(Document(page_content=text, metadata=metadata))
        return docs

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return self._retrieve(query)

    async def _aget_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None,
    ) -> list[Document]:
        return self._retrieve(query)


class MultiDomainRetriever:
    """多知识库路由检索 + 加权合并。"""

    def __init__(self) -> None:
        settings = get_settings()
        self._workspace = settings.bailian_workspace_id
        self._api_key = settings.dashscope_api_key
        self._index_map = {
            KnowledgeDomain.KNOWLEDGE: settings.bailian_index_knowledge_id or settings.bailian_index_id,
            KnowledgeDomain.PRODUCT: settings.bailian_index_product_id or settings.bailian_index_id,
            KnowledgeDomain.REGULATORY: settings.bailian_index_regulatory_id or settings.bailian_index_id,
        }

    def _get_retriever(self, domain: KnowledgeDomain, top_k: int) -> BailianRetriever:
        return BailianRetriever(
            workspace_id=self._workspace,
            index_id=self._index_map[domain],
            api_key=self._api_key,
            top_k=top_k,
            domain=domain,
        )

    def resolve_weights(self, intent_result: IntentResult | None) -> dict[KnowledgeDomain, float]:
        """根据意图结果计算各域检索权重。"""
        if not intent_result:
            return dict(DEFAULT_DOMAIN_WEIGHTS)

        weights = INTENT_DOMAIN_WEIGHTS.get(
            intent_result.intent.value,
            DEFAULT_DOMAIN_WEIGHTS,
        ).copy()

        # 多标签叠加：命中域权重 +0.15
        for label in intent_result.intents:
            if label == "regulatory":
                weights[KnowledgeDomain.REGULATORY] = weights.get(KnowledgeDomain.REGULATORY, 0) + 0.15
            if label == "product":
                weights[KnowledgeDomain.PRODUCT] = weights.get(KnowledgeDomain.PRODUCT, 0) + 0.15

        # 显式 knowledge_domains 优先
        if intent_result.knowledge_domains:
            boosted = {d: 0.1 for d in KnowledgeDomain}
            for d in intent_result.knowledge_domains:
                boosted[d] = 1.0
            for d in KnowledgeDomain:
                weights[d] = max(weights.get(d, 0), boosted[d])

        total = sum(weights.values()) or 1.0
        return {d: w / total for d, w in weights.items()}

    def retrieve(
        self,
        queries: list[str],
        intent_result: IntentResult | None = None,
        base_top_k: int = 5,
    ) -> list[Document]:
        """多 query × 多域加权检索，返回带 score 的 Document 列表。"""
        settings = get_settings()
        if not settings.bailian_enable:
            return []

        weights = self.resolve_weights(intent_result)
        scored: list[tuple[float, Document]] = []
        seen: set[str] = set()

        for query in queries:
            for domain, weight in weights.items():
                if weight < 0.05:
                    continue
                top_k = max(1, int(base_top_k * weight) + 1)
                retriever = self._get_retriever(domain, top_k)
                for rank, doc in enumerate(retriever.invoke(query)):
                    content = doc.page_content.strip()
                    if not content or content in seen:
                        continue
                    seen.add(content)
                    # 加权得分：域权重 × 排名衰减
                    score = weight * (1.0 / (rank + 1))
                    doc.metadata["retrieval_score"] = round(score, 4)
                    doc.metadata["domain"] = domain.value
                    doc.metadata["domain_label"] = DOMAIN_LABELS[domain]
                    scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in scored[: base_top_k * 2]]


def get_multi_domain_retriever() -> MultiDomainRetriever:
    return MultiDomainRetriever()


def merge_and_format_documents(docs: list[Document]) -> str:
    """召回后合并与格式化。"""
    if not docs:
        return ""

    by_domain: dict[str, list[Document]] = {}
    for doc in docs:
        label = doc.metadata.get("domain_label", "综合")
        by_domain.setdefault(label, []).append(doc)

    sections: list[str] = []
    for label, domain_docs in by_domain.items():
        lines = [f"## {label}"]
        for i, doc in enumerate(domain_docs, 1):
            score = doc.metadata.get("retrieval_score", "")
            prefix = f"[{i}]" + (f"(相关度{score})" if score else "")
            snippet = doc.page_content.strip()
            if len(snippet) > 500:
                snippet = snippet[:500] + "…"
            lines.append(f"{prefix} {snippet}")
        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def retrieve_context(
    queries: list[str],
    intent_result: IntentResult | None = None,
) -> str:
    """对外统一入口：多域检索 + 格式化。"""
    retriever = get_multi_domain_retriever()
    docs = retriever.retrieve(queries, intent_result=intent_result)
    return merge_and_format_documents(docs)


# 向后兼容
def get_retriever() -> BailianRetriever:
    settings = get_settings()
    return BailianRetriever(
        workspace_id=settings.bailian_workspace_id,
        index_id=settings.bailian_index_id,
        api_key=settings.dashscope_api_key,
        domain=KnowledgeDomain.KNOWLEDGE,
    )
