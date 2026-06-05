"""
维度 3：恢复正确性

随机中断状态机，从 Checkpoint 重启，验证输出/状态一致性。
"""

from __future__ import annotations

import os
import random
from typing import Any
from unittest.mock import patch

import pytest

from app.harness.artifacts.store import artifact_store
from app.harness.enums import RunStatus
from app.harness.models import RunContext
from app.harness.state.checkpoint import get_checkpoint_manager
from eval.judge import RecoveryConsistencyCase, judge_recovery_consistency

WORKFLOW_NODES = ["intent", "rewrite", "decompose", "retrieve", "generate_qa", "calc_react"]


def _sample_state(node: str, chat_id: str = "eval-recovery") -> dict[str, Any]:
    return {
        "chat_id": chat_id,
        "message": "对比贵州茅台600519与白酒行业平均PE",
        "history": "用户关注600519，风险偏好稳健型",
        "intent": "complex_planning",
        "intents": ["qa", "complex_planning"],
        "requires_decompose": True,
        "knowledge_domains": ["knowledge", "product"],
        "rewritten_query": "贵州茅台(600519)市盈率与白酒行业均值对比",
        "compressed_context": "600519贵州茅台；稳健型；PE",
        "extracted_entities": [{"entity_type": "stock_code", "value": "600519"}],
        "sub_questions": ["查贵州茅台财报", "计算PE", "对比行业"],
        "context": "## 理财知识\n[1] 白酒行业平均PE约25倍",
        "answer": "贵州茅台当前PE约30倍，高于白酒行业平均25倍，估值偏贵。",
        "_checkpoint_node": node,
    }


@pytest.fixture
def recovery_run(tmp_path, monkeypatch) -> RunContext:
    runs_dir = tmp_path / "recovery_runs"
    runs_dir.mkdir()
    monkeypatch.setenv("RUNS_DIR", str(runs_dir))
    from app.config import get_settings

    get_settings.cache_clear()
    get_checkpoint_manager.cache_clear()
    ctx = artifact_store.create_run(chat_id="eval-recovery", mode="advisor")
    ctx.manifest.status = RunStatus.INTERRUPTED
    artifact_store.save_manifest(ctx.manifest)
    return ctx


def test_restore_execution_chain_at_random_nodes(recovery_run: RunContext, rng: random.Random):
    """在每个节点快照后恢复，验证状态字段一致。"""
    mgr = get_checkpoint_manager()
    session_id = recovery_run.manifest.chat_id

    nodes = WORKFLOW_NODES.copy()
    rng.shuffle(nodes)

    for node in nodes[:3]:  # 随机抽 3 个节点
        state = _sample_state(node)
        cid = f"cp_test_{node}"
        artifact_store.save_checkpoint(recovery_run, node, state, checkpoint_id=cid)

        restored = mgr.restore_execution_chain(session_id, cid)
        assert restored is not None, f"节点 {node} 恢复失败"
        assert restored.restored_node == node
        assert restored.restored_state.get("rewritten_query") == state["rewritten_query"]
        assert restored.restored_state.get("answer") == state["answer"]
        assert restored.langgraph_config["configurable"]["thread_id"] == session_id
        assert restored.langgraph_config["configurable"]["checkpoint_id"] == cid


def test_checkpoint_state_immutable_after_restore(recovery_run: RunContext):
    """恢复后修改内存状态不应影响已落盘快照。"""
    state = _sample_state("retrieve")
    cid = "cp_immutable_test"
    artifact_store.save_checkpoint(recovery_run, "retrieve", state, checkpoint_id=cid)

    loaded = artifact_store.load_checkpoint(recovery_run.manifest.run_id, cid)
    assert loaded is not None
    original_answer = loaded.state["answer"]

    loaded.state["answer"] = "被篡改的回答"
    reloaded = artifact_store.load_checkpoint(recovery_run.manifest.run_id, cid)
    assert reloaded.state["answer"] == original_answer


def test_recovery_answer_consistency_judge():
    """规则/LLM 裁判：中断前后回答关键事实一致。"""
    original = "贵州茅台(600519)当前PE约30倍，高于白酒行业平均25倍。"
    resumed = "从检查点恢复：600519贵州茅台市盈率约30倍，行业均值约25倍，估值偏高。"

    verdict = judge_recovery_consistency(
        RecoveryConsistencyCase(
            original_answer=original,
            resumed_answer=resumed,
            key_facts=["600519", "30倍", "25倍", "白酒行业"],
        )
    )
    assert verdict.passed, verdict.reasoning
    assert verdict.score >= 0.8


def test_workflow_resume_produces_consistent_answer(recovery_run: RunContext):
    """模拟工作流：中断于 retrieve 后恢复，generate 输出一致。"""
    deterministic_answer = (
        "贵州茅台(600519)市盈率约30倍，白酒行业平均约25倍，当前估值高于行业均值。"
    )

    state = _sample_state("retrieve")
    cid = "cp_resume_generate"
    artifact_store.save_checkpoint(recovery_run, "retrieve", state, checkpoint_id=cid)

    with patch("app.graph.workflow.generate_answer", return_value=deterministic_answer):
        from app.graph.workflow import _node_generate_qa

        out1 = _node_generate_qa(state)
        # 模拟从 checkpoint 恢复后再次执行 generate
        restored = get_checkpoint_manager().restore_execution_chain(
            recovery_run.manifest.chat_id, cid,
        )
        out2 = _node_generate_qa(restored.restored_state)

    assert out1["answer"] == out2["answer"]
    assert "600519" in out1["answer"]


@pytest.mark.llm
@pytest.mark.skipif(
    os.getenv("EVAL_USE_LLM", "").lower() not in ("1", "true", "yes"),
    reason="设置 EVAL_USE_LLM=true 以运行完整恢复集成评测",
)
def test_full_workflow_checkpoint_resume_integration():
    """完整 LangGraph 中断恢复（需 LLM）。"""
    from app.graph.workflow import build_advisor_graph
    from app.harness.harness import get_harness

    harness = get_harness()
    chat_id = "eval-full-resume"
    config = harness.checkpoint_mgr.thread_config(chat_id)

    graph = build_advisor_graph()
    # 首次运行
    result1 = graph.invoke(
        {
            "chat_id": chat_id,
            "message": "10000元按5%复利投资10年终值多少？",
            "history": "",
        },
        config=config,
    )
    answer1 = result1.get("answer", "")

    # 同 thread 再次 invoke（应从 checkpoint 续跑上下文）
    result2 = graph.invoke(
        {
            "chat_id": chat_id,
            "message": "10000元按5%复利投资10年终值多少？",
            "history": "",
        },
        config=config,
    )
    answer2 = result2.get("answer", "")

    verdict = judge_recovery_consistency(
        RecoveryConsistencyCase(
            original_answer=answer1,
            resumed_answer=answer2,
            key_facts=["10000", "5%", "10"],
        )
    )
    assert verdict.score >= 0.5, verdict.reasoning
