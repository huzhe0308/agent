"""
LLM-as-Judge 评测辅助。

默认使用规则裁判（离线可跑）；设置 EVAL_USE_LLM=true 时调用真实 LLM。
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, Field


class JudgeVerdict(BaseModel):
    """裁判输出。"""

    passed: bool
    score: float = Field(ge=0.0, le=1.0, description="0-1 置信分")
    reasoning: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


@dataclass
class EntityRetentionCase:
    """实体保留评测用例。"""

    golden_entities: list[str]
    rewrite_text: str
    min_recall: float = 0.8


@dataclass
class RecoveryConsistencyCase:
    """恢复一致性评测用例。"""

    original_answer: str
    resumed_answer: str
    key_facts: list[str]


def use_llm_judge() -> bool:
    return os.getenv("EVAL_USE_LLM", "").lower() in ("1", "true", "yes")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").lower())


def _entity_recall(golden: list[str], text: str) -> float:
    if not golden:
        return 1.0
    norm_text = _normalize(text)
    hits = sum(1 for e in golden if _normalize(e) in norm_text)
    return hits / len(golden)


def rule_judge_entity_retention(case: EntityRetentionCase) -> JudgeVerdict:
    recall = _entity_recall(case.golden_entities, case.rewrite_text)
    passed = recall >= case.min_recall
    return JudgeVerdict(
        passed=passed,
        score=recall,
        reasoning=f"实体召回率 {recall:.0%}（阈值 {case.min_recall:.0%}）",
        details={"recall": recall, "golden": case.golden_entities},
    )


def rule_judge_recovery_consistency(case: RecoveryConsistencyCase) -> JudgeVerdict:
    """规则裁判：关键事实是否同时出现在两次回答中。"""
    if not case.key_facts:
        sim = 1.0 if _normalize(case.original_answer) == _normalize(case.resumed_answer) else 0.5
        return JudgeVerdict(
            passed=sim >= 0.9,
            score=sim,
            reasoning="无关键事实列表，使用全文相似度粗判",
        )

    orig = _normalize(case.original_answer)
    resumed = _normalize(case.resumed_answer)
    hits = sum(1 for f in case.key_facts if _normalize(f) in orig and _normalize(f) in resumed)
    score = hits / len(case.key_facts)
    return JudgeVerdict(
        passed=score >= 0.8,
        score=score,
        reasoning=f"关键事实一致率 {score:.0%}",
        details={"matched": hits, "total": len(case.key_facts)},
    )


def llm_judge(prompt: str, system: str = "") -> JudgeVerdict:
    """调用 LLM 裁判，要求返回 JSON。"""
    from langchain_core.messages import HumanMessage, SystemMessage

    from app.harness.providers.registry import get_llm

    llm = get_llm()
    messages = []
    if system:
        messages.append(SystemMessage(content=system))
    messages.append(HumanMessage(content=prompt))

    raw = llm.invoke(messages).content
    try:
        # 提取 JSON 块
        match = re.search(r"\{[\s\S]*\}", raw)
        data = json.loads(match.group()) if match else json.loads(raw)
        return JudgeVerdict.model_validate(data)
    except Exception as e:
        return JudgeVerdict(
            passed=False,
            score=0.0,
            reasoning=f"Judge 解析失败: {e}",
            details={"raw": raw[:500]},
        )


JUDGE_ENTITY_SYSTEM = """你是金融对话评测裁判。评估「重写输出」是否保留了历史中的关键金融实体。
输出 JSON：{"passed": bool, "score": 0-1, "reasoning": "≤60字", "details": {}}"""


def judge_entity_retention(case: EntityRetentionCase) -> JudgeVerdict:
    if not use_llm_judge():
        return rule_judge_entity_retention(case)

    prompt = f"""黄金实体列表：{case.golden_entities}
重写输出：
{case.rewrite_text}

请判断实体保留率是否 ≥ {case.min_recall:.0%}。仅输出 JSON。"""
    verdict = llm_judge(prompt, JUDGE_ENTITY_SYSTEM)
    verdict.passed = verdict.passed and verdict.score >= case.min_recall
    return verdict


JUDGE_RECOVERY_SYSTEM = """你是金融智能体恢复一致性裁判。比较中断前与恢复后的回答是否语义一致。
输出 JSON：{"passed": bool, "score": 0-1, "reasoning": "≤60字", "details": {}}"""


def judge_recovery_consistency(case: RecoveryConsistencyCase) -> JudgeVerdict:
    if not use_llm_judge():
        return rule_judge_recovery_consistency(case)

    prompt = f"""关键事实：{case.key_facts}

【中断前回答】
{case.original_answer}

【恢复后回答】
{case.resumed_answer}

评估恢复后是否保留关键事实且语义一致。仅输出 JSON。"""
    return llm_judge(prompt, JUDGE_RECOVERY_SYSTEM)
