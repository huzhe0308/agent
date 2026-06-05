"""
RAG 多域路由单元评测（独立于三维度，验证权重分配逻辑）。
"""

from __future__ import annotations

from app.models.schemas import IntentResult, IntentType, KnowledgeDomain
from app.rag.bailian_retriever import MultiDomainRetriever, merge_and_format_documents
from langchain_core.documents import Document


def test_resolve_weights_for_regulatory_intent():
    retriever = MultiDomainRetriever()
    intent = IntentResult(
        intent=IntentType.REGULATORY,
        intents=["regulatory"],
        knowledge_domains=[KnowledgeDomain.REGULATORY],
    )
    weights = retriever.resolve_weights(intent)
    assert weights[KnowledgeDomain.REGULATORY] >= weights[KnowledgeDomain.KNOWLEDGE]
    assert abs(sum(weights.values()) - 1.0) < 0.01


def test_resolve_weights_multi_domain_boost():
    retriever = MultiDomainRetriever()
    intent = IntentResult(
        intent=IntentType.COMPLEX_PLANNING,
        intents=["qa", "product", "complex_planning"],
        knowledge_domains=[KnowledgeDomain.KNOWLEDGE, KnowledgeDomain.PRODUCT],
    )
    weights = retriever.resolve_weights(intent)
    assert weights[KnowledgeDomain.KNOWLEDGE] > 0
    assert weights[KnowledgeDomain.PRODUCT] > 0


def test_merge_and_format_groups_by_domain():
    docs = [
        Document(page_content="货币基金适合保守投资者", metadata={"domain_label": "理财知识", "retrieval_score": 0.8}),
        Document(page_content="某理财产品费率0.5%", metadata={"domain_label": "产品文档", "retrieval_score": 0.6}),
    ]
    formatted = merge_and_format_documents(docs)
    assert "## 理财知识" in formatted
    assert "## 产品文档" in formatted
    assert "货币基金" in formatted
