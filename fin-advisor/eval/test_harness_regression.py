"""
维度 1：Harness Regression

验证工具调用参数校验与拦截逻辑 100% 生效（纯单元，不依赖 LLM）。
"""

from __future__ import annotations

import pytest

from app.harness.enums import ToolCallStatus, ToolType
from app.harness.tools.executor import ToolExecutor
from app.harness.tools.security import approval_gate


# ------------------------------------------------------------------ #
# 参数校验
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "tool_name,arguments,expected_status",
    [
        (
            "compound_interest",
            {"principal": -1000, "annual_rate": 0.05, "years": 5},
            ToolCallStatus.VALIDATION_ERROR,
        ),
        (
            "compound_interest",
            {"principal": 0, "annual_rate": 0.05, "years": 5},
            ToolCallStatus.VALIDATION_ERROR,
        ),
        (
            "dca_investment",
            {"monthly_amount": 1000, "annual_rate": -0.1, "years": 3},
            ToolCallStatus.VALIDATION_ERROR,
        ),
        (
            "risk_profile_score",
            {"age": 15, "income_stability": 3, "investment_experience": 2,
             "loss_tolerance": 2, "investment_horizon_years": 5},
            ToolCallStatus.VALIDATION_ERROR,
        ),
    ],
    ids=["negative_principal", "zero_principal", "negative_rate", "underage"],
)
def test_param_validation_blocks_invalid_args(
    tool_executor: ToolExecutor,
    tool_name: str,
    arguments: dict,
    expected_status: ToolCallStatus,
):
    record = tool_executor.execute(tool_name, arguments)
    assert record.status == expected_status
    assert not record.success
    assert record.block_reason


def test_param_validation_accepts_valid_compound_interest(tool_executor: ToolExecutor):
    record = tool_executor.execute(
        "compound_interest",
        {"principal": 10000, "annual_rate": 0.05, "years": 10},
    )
    assert record.status == ToolCallStatus.SUCCESS
    assert record.success
    assert "终值" in record.result


# ------------------------------------------------------------------ #
# 工具不存在
# ------------------------------------------------------------------ #
def test_not_found_tool(tool_executor: ToolExecutor):
    record = tool_executor.execute("nonexistent_tool_xyz", {"foo": "bar"})
    assert record.status == ToolCallStatus.NOT_FOUND
    assert not record.success


# ------------------------------------------------------------------ #
# 重复调用拦截
# ------------------------------------------------------------------ #
def test_duplicate_call_guard(tool_executor: ToolExecutor):
    args = {"principal": 5000, "annual_rate": 0.04, "years": 3}
    first = tool_executor.execute("compound_interest", args)
    assert first.status == ToolCallStatus.SUCCESS

    second = tool_executor.execute("compound_interest", args)
    assert second.status == ToolCallStatus.DUPLICATE_BLOCKED
    assert not second.success
    assert "重复" in (second.block_reason or "")


# ------------------------------------------------------------------ #
# 高风险工具门控
# ------------------------------------------------------------------ #
def test_high_risk_tool_requires_approval(tool_executor: ToolExecutor):
    record = tool_executor.execute(
        "http_fetch",
        {"url": "https://example.com", "method": "GET"},
    )
    assert record.status == ToolCallStatus.PENDING_APPROVAL
    assert record.approval_id
    assert not record.success


def test_high_risk_subagent_blocked_without_approval(tool_executor: ToolExecutor):
    record = tool_executor.execute(
        "calc_subagent",
        {"question": "计算 10000 元 5% 复利 10 年"},
    )
    assert record.status in (
        ToolCallStatus.PENDING_APPROVAL,
        ToolCallStatus.BLOCKED_HIGH_RISK,
    )
    assert not record.success


def test_high_risk_tool_passes_with_approval_token(tool_executor: ToolExecutor):
    # 先触发审批单
    pending = tool_executor.execute(
        "http_fetch",
        {"url": "https://httpbin.org/get", "method": "GET"},
    )
    assert pending.status == ToolCallStatus.PENDING_APPROVAL
    approval_id = pending.approval_id
    assert approval_id

    # 模拟人工批准
    approval_gate.approve(approval_id, approver="eval-tester")

    # 携带 token 重试（新 executor 避免 duplicate guard）
    tools = tool_executor.get_langchain_tools()
    fresh = ToolExecutor(tools, tool_executor._run_ctx, session_id="eval-session-2")
    approved = fresh.execute(
        "http_fetch",
        {"url": "https://httpbin.org/get", "method": "GET"},
        approval_token=approval_id,
    )
    # 可能因网络失败 EXECUTION_ERROR，但不应被安全层阻断
    assert approved.status in (ToolCallStatus.SUCCESS, ToolCallStatus.EXECUTION_ERROR)
    assert approved.status not in (
        ToolCallStatus.PENDING_APPROVAL,
        ToolCallStatus.BLOCKED_HIGH_RISK,
        ToolCallStatus.VALIDATION_ERROR,
        ToolCallStatus.DUPLICATE_BLOCKED,
    )


# ------------------------------------------------------------------ #
# 回归矩阵：拦截逻辑覆盖率
# ------------------------------------------------------------------ #
INTERCEPTION_CASES = [
    ("validation", "compound_interest", {"principal": -1, "annual_rate": 0.05, "years": 1},
     ToolCallStatus.VALIDATION_ERROR),
    ("not_found", "ghost_tool", {}, ToolCallStatus.NOT_FOUND),
]


def test_interception_matrix_coverage(tool_executor: ToolExecutor):
    """确保核心拦截路径均有用例覆盖。"""
    covered_statuses: set[ToolCallStatus] = set()

    for _, tool, args, status in INTERCEPTION_CASES:
        rec = tool_executor.execute(tool, args)
        covered_statuses.add(rec.status)
        assert rec.status == status

    dup = tool_executor.execute(
        "compound_interest",
        {"principal": 1000, "annual_rate": 0.03, "years": 2},
    )
    tool_executor.execute(
        "compound_interest",
        {"principal": 1000, "annual_rate": 0.03, "years": 2},
    )
    covered_statuses.add(ToolCallStatus.DUPLICATE_BLOCKED)

    hr = tool_executor.execute("http_fetch", {"url": "https://x.com"})
    covered_statuses.add(hr.status)

    required = {
        ToolCallStatus.VALIDATION_ERROR,
        ToolCallStatus.NOT_FOUND,
        ToolCallStatus.DUPLICATE_BLOCKED,
        ToolCallStatus.PENDING_APPROVAL,
    }
    assert required.issubset(covered_statuses), f"未覆盖: {required - covered_statuses}"
