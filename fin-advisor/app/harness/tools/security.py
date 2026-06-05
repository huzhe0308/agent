"""
工具调用安全策略与人工审批门控。

提供：
- 高风险工具识别
- 高风险白名单（预批准，跳过审批）
- ApprovalGate 人工审批接口（预留，可对接 HTTP/Webhook/消息队列）
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from app.harness.enums import ApprovalStatus
from app.harness.models import ApprovalRequest, utc_now

# ------------------------------------------------------------------ #
# 高风险工具定义
# ------------------------------------------------------------------ #

# 明确列为高风险的工具名（调用前需审批或命中白名单）
HIGH_RISK_TOOLS: frozenset[str] = frozenset({
    "http_fetch",       # 外部 HTTP 请求，存在 SSRF 风险
    "calc_subagent",    # 子 Agent 委托，执行链不可控
})

# 高风险工具名前缀（如 mcp_* 均为外部进程工具）
HIGH_RISK_TOOL_PREFIXES: tuple[str, ...] = ("mcp_",)

# 高风险白名单：即使属于高风险类别，也允许直接执行（无需审批）
# 可在运行时通过 ApprovalGate.extend_whitelist() 动态扩展
HIGH_RISK_WHITELIST: frozenset[str] = frozenset({
    # 示例：若某 MCP 工具已审计可加入白名单
    # "mcp_fin-calc_compound_interest",
})


def is_high_risk_tool(tool_name: str) -> bool:
    """判断工具是否属于高风险类别。"""
    if tool_name in HIGH_RISK_TOOLS:
        return True
    return any(tool_name.startswith(p) for p in HIGH_RISK_TOOL_PREFIXES)


def is_whitelisted(tool_name: str, extra_whitelist: frozenset[str] | None = None) -> bool:
    """判断是否在高风险白名单中。"""
    combined = HIGH_RISK_WHITELIST | (extra_whitelist or frozenset())
    return tool_name in combined


class ApprovalGate:
    """
    人工审批门控（接口预留）。

    生产环境可替换为：
    - REST API 轮询 / Webhook 回调
    - 企业微信 / 钉钉审批机器人
    - 内部审批工单系统
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._pending: dict[str, ApprovalRequest] = {}
        self._extra_whitelist: set[str] = set()
        self._ttl_seconds = ttl_seconds
        self._lock = Lock()

    def extend_whitelist(self, tool_names: list[str]) -> None:
        """运行时扩展高风险白名单。"""
        with self._lock:
            self._extra_whitelist.update(tool_names)

    def request_approval(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str,
        run_id: str | None = None,
    ) -> ApprovalRequest:
        """
        创建审批请求并挂起（预留接口）。

        返回 ApprovalRequest，调用方应阻断执行并告知 approval_id。
        """
        approval_id = uuid4().hex
        request = ApprovalRequest(
            approval_id=approval_id,
            tool_name=tool_name,
            arguments=arguments,
            session_id=session_id,
            run_id=run_id,
            status=ApprovalStatus.PENDING,
        )
        with self._lock:
            self._pending[approval_id] = request
        # 预留：self._notify_approvers(request)
        return request

    def approve(self, approval_id: str, approver: str = "system") -> bool:
        """人工批准（预留接口，供管理 API 调用）。"""
        with self._lock:
            req = self._pending.get(approval_id)
            if not req or req.status != ApprovalStatus.PENDING:
                return False
            req.status = ApprovalStatus.APPROVED
            req.approver = approver
            req.resolved_at = utc_now()
            return True

    def reject(self, approval_id: str, approver: str = "system", reason: str = "") -> bool:
        """人工拒绝。"""
        with self._lock:
            req = self._pending.get(approval_id)
            if not req or req.status != ApprovalStatus.PENDING:
                return False
            req.status = ApprovalStatus.REJECTED
            req.approver = approver
            req.reject_reason = reason
            req.resolved_at = utc_now()
            return True

    def is_approved(self, approval_id: str) -> bool:
        """检查审批是否已通过且未过期。"""
        with self._lock:
            req = self._pending.get(approval_id)
            if not req:
                return False
            if req.status == ApprovalStatus.APPROVED:
                return True
            if req.status == ApprovalStatus.PENDING:
                created = datetime.fromisoformat(req.created_at)
                if datetime.now(timezone.utc) - created > timedelta(seconds=self._ttl_seconds):
                    req.status = ApprovalStatus.EXPIRED
            return False

    def get_request(self, approval_id: str) -> ApprovalRequest | None:
        """查询审批单。"""
        with self._lock:
            return self._pending.get(approval_id)

    def list_pending(self, session_id: str | None = None) -> list[ApprovalRequest]:
        """列出待审批请求。"""
        with self._lock:
            reqs = list(self._pending.values())
        if session_id:
            reqs = [r for r in reqs if r.session_id == session_id]
        return [r for r in reqs if r.status == ApprovalStatus.PENDING]

    def check_high_risk(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        session_id: str,
        run_id: str | None = None,
        approval_token: str | None = None,
    ) -> tuple[bool, ApprovalRequest | None, str | None]:
        """
        高风险检查。

        返回:
            (allowed, approval_request, block_reason)
            - allowed=True：可直接执行
            - allowed=False + approval_request：已创建审批单，等待人工批准
            - allowed=False + block_reason：被阻断（已拒绝等）
        """
        if not is_high_risk_tool(tool_name):
            return True, None, None

        extra = frozenset(self._extra_whitelist)
        if is_whitelisted(tool_name, extra):
            return True, None, None

        if approval_token and self.is_approved(approval_token):
            return True, None, None

        req = self.request_approval(tool_name, arguments, session_id, run_id)
        return False, req, None

    def _notify_approvers(self, request: ApprovalRequest) -> None:
        """预留：通知审批人（Webhook / IM / 邮件）。"""
        pass


# 进程内单例
approval_gate = ApprovalGate()
