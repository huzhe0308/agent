"""
Agent Harness 核心枚举定义。
"""

from enum import Enum


class ModelBackend(str, Enum):
    """2 类模型后端。"""

    DASHSCOPE = "dashscope"
    OPENAI_COMPAT = "openai_compat"


class ToolType(str, Enum):
    """7 类工具。"""

    STRUCTURED = "structured"
    MCP = "mcp"
    RAG = "rag"
    CHAIN = "chain"
    HTTP = "http"
    SUBAGENT = "subagent"
    BUILTIN = "builtin"


class ArtifactType(str, Enum):
    """3 类运行工件。"""

    TRACE = "trace"
    CHECKPOINT = "checkpoint"
    MANIFEST = "manifest"


class RunStatus(str, Enum):
    """单次 Agent 运行的生命周期状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class ToolCallStatus(str, Enum):
    """工具调用结果状态（安全边界审计）。"""

    SUCCESS = "success"
    VALIDATION_ERROR = "validation_error"       # 参数校验失败
    BLOCKED_HIGH_RISK = "blocked_high_risk"     # 高风险阻断
    PENDING_APPROVAL = "pending_approval"       # 等待人工审批
    DUPLICATE_BLOCKED = "duplicate_blocked"     # 短时间重复调用拦截
    NOT_FOUND = "not_found"                     # 工具不存在
    EXECUTION_ERROR = "execution_error"         # 执行异常（已隔离）


class ApprovalStatus(str, Enum):
    """人工审批状态（预留接口）。"""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
