"""
维度 2：上下文治理

输入 10 轮长对话，评测重写模块能否在压缩 Token 50%+ 的同时保留关键金融实体。
可与 LLM-as-Judge 结合（EVAL_USE_LLM=true）。
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from app.chains.rewrite import estimate_rewrite_tokens, rewrite_query
from app.models.schemas import FinancialEntity, RewriteResult
from eval.judge import EntityRetentionCase, judge_entity_retention


def _build_history(turns: list[dict]) -> str:
    return "\n".join(f"{t['role']}: {t['content']}" for t in turns)


def _mock_rewrite_result(question: str, history: str) -> RewriteResult:
    """离线模拟：压缩历史、保留实体（用于 CI 无 LLM 场景）。"""
    return RewriteResult(
        rewritten_query="请对比贵州茅台（600519）当前市盈率与白酒行业平均 PE",
        compressed_context="关注600519贵州茅台；风险偏好稳健型；期望年化25%；已了解PE概念",
        extracted_entities=[
            FinancialEntity(entity_type="stock_code", value="600519"),
            FinancialEntity(entity_type="metric", value="PE"),
            FinancialEntity(entity_type="product", value="贵州茅台"),
            FinancialEntity(entity_type="risk_profile", value="稳健型"),
            FinancialEntity(entity_type="amount", value="25%"),
            FinancialEntity(entity_type="other", value="白酒行业"),
        ],
        needs_context=True,
        dropped_noise=["你好", "天气不错", "谢谢"],
    )


@pytest.fixture
def dialog_history(long_dialog: dict) -> str:
    return _build_history(long_dialog["turns"])


def test_compression_ratio_at_least_50_percent(long_dialog: dict, dialog_history: str):
    """重写输出相对原始历史压缩 ≥50%（字符粗估）。"""
    result = _mock_rewrite_result(long_dialog["final_question"], dialog_history)
    original_len = len(dialog_history)
    compressed_len = estimate_rewrite_tokens(result)

    ratio = 1.0 - (compressed_len / original_len)
    assert ratio >= 0.5, (
        f"压缩率 {ratio:.1%} 未达 50%，"
        f"原始={original_len} 压缩后={compressed_len}"
    )


def test_golden_entities_preserved_in_rewrite(long_dialog: dict, dialog_history: str):
    """黄金实体应出现在重写输出中。"""
    result = _mock_rewrite_result(long_dialog["final_question"], dialog_history)
    combined = (
        result.rewritten_query
        + result.compressed_context
        + "".join(e.value for e in result.extracted_entities)
    )
    for entity in long_dialog["golden_entities"]:
        assert entity in combined or entity.lower() in combined.lower(), (
            f"实体 '{entity}' 未保留"
        )


def test_noise_dropped(long_dialog: dict, dialog_history: str):
    result = _mock_rewrite_result(long_dialog["final_question"], dialog_history)
    assert result.dropped_noise
    # 寒暄不应进入 compressed_context
    noise_terms = ["天气不错", "你好"]
    for term in noise_terms:
        assert term not in result.compressed_context


def test_entity_retention_judge_passes(long_dialog: dict, dialog_history: str):
    """LLM-as-Judge / 规则裁判：实体保留评测。"""
    result = _mock_rewrite_result(long_dialog["final_question"], dialog_history)
    rewrite_text = (
        f"{result.rewritten_query}\n{result.compressed_context}\n"
        + " ".join(e.value for e in result.extracted_entities)
    )
    verdict = judge_entity_retention(
        EntityRetentionCase(
            golden_entities=long_dialog["golden_entities"],
            rewrite_text=rewrite_text,
            min_recall=0.8,
        )
    )
    assert verdict.passed, verdict.reasoning
    assert verdict.score >= 0.8


@pytest.mark.llm
@pytest.mark.skipif(
    os.getenv("EVAL_USE_LLM", "").lower() not in ("1", "true", "yes"),
    reason="设置 EVAL_USE_LLM=true 以运行真实重写链",
)
def test_live_rewrite_chain_compression(long_dialog: dict, dialog_history: str):
    """真实 LLM 重写链：压缩率 + 实体保留。"""
    result = rewrite_query(
        long_dialog["final_question"],
        dialog_history,
        intent_hint="qa,complex_planning",
    )
    original_len = len(dialog_history)
    compressed_len = estimate_rewrite_tokens(result)
    ratio = 1.0 - (compressed_len / original_len)

    assert ratio >= 0.5, f"真实链压缩率仅 {ratio:.1%}"

    rewrite_text = (
        f"{result.rewritten_query}\n{result.compressed_context}\n"
        + " ".join(e.value for e in result.extracted_entities)
    )
    verdict = judge_entity_retention(
        EntityRetentionCase(
            golden_entities=long_dialog["golden_entities"],
            rewrite_text=rewrite_text,
            min_recall=0.7,
        )
    )
    assert verdict.passed, verdict.reasoning


def test_rewrite_chain_with_mocked_llm(long_dialog: dict, dialog_history: str):
    """Mock LLM 验证 rewrite_query 管道与后处理截断。"""
    mock_result = _mock_rewrite_result(long_dialog["final_question"], dialog_history)

    with patch("app.chains.rewrite.build_rewrite_chain") as mock_chain:
        runnable = MagicMock()
        runnable.invoke.return_value = mock_result
        mock_chain.return_value = runnable

        result = rewrite_query(long_dialog["final_question"], dialog_history)

    assert len(result.rewritten_query) <= 120
    assert len(result.compressed_context) <= 150
    assert len(result.extracted_entities) <= 8
