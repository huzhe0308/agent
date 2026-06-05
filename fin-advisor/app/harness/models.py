"""
Agent Harness 运行时数据模型。

定义单次 Agent 运行过程中产生的核心数据结构：
- RunManifest / RunContext：运行清单与上下文句柄
- TraceEvent：轨迹事件（写入 trace.jsonl）
- CheckpointSnapshot：节点状态快照
- ToolCallRecord / ToolSpec：工具调用审计
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from app.harness.enums import (
    ApprovalStatus,
    ArtifactType,
    ModelBackend,
    RunStatus,
    ToolCallStatus,
    ToolType,
)


def utc_now() -> str:
    """返回 UTC ISO 时间戳字符串。"""
    return datetime.now(timezone.utc).isoformat()


class ToolSpec(BaseModel):
    """工具元信息（用于 /api/harness/tools 目录展示）。"""

    name: str
    tool_type: ToolType
    description: str = ""
    enabled: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolCallRecord(BaseModel):
    """单次工具调用的审计记录。"""

    tool_name: str
    tool_type: ToolType
    arguments: dict[str, Any] = Field(default_factory=dict)
    result: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None
    status: ToolCallStatus = ToolCallStatus.SUCCESS
    block_reason: str | None = None
    approval_id: str | None = None


class ApprovalRequest(BaseModel):
    """高风险工具人工审批单（接口预留）。"""

    approval_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    session_id: str
    run_id: str | None = None
    status: ApprovalStatus = ApprovalStatus.PENDING
    approver: str | None = None
    reject_reason: str = ""
    created_at: str = Field(default_factory=utc_now)
    resolved_at: str | None = None


class TraceEvent(BaseModel):
    """运行轨迹事件，每行写入 trace.jsonl 一条。"""

    run_id: str                   # 所属运行 ID
    seq: int                      # 事件序号（单调递增）
    timestamp: str = Field(default_factory=utc_now)
    event_type: str               # 如 run_start / node_enter / tool_call / checkpoint
    node: str | None = None       # LangGraph 节点名
    payload: dict[str, Any] = Field(default_factory=dict)


class CheckpointSnapshot(BaseModel):
    """LangGraph 工作流节点的状态快照，用于复盘与恢复。"""

    run_id: str
    checkpoint_id: str            # 如 cp_0003_retrieve
    session_id: str = ""          # 会话 ID（同 chat_id / thread_id）
    thread_id: str                # LangGraph thread_id
    node: str                     # 快照时的节点名
    state: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=utc_now)


class RunManifest(BaseModel):
    """
    运行清单（manifest.json）。

    记录一次对话执行的元信息：模型、工具、状态、时间、工件路径。
    """

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    chat_id: str
    thread_id: str                # LangGraph thread_id，用于 Checkpoint 关联
    mode: str = "advisor"         # advisor | manus
    status: RunStatus = RunStatus.PENDING
    model_backend: ModelBackend = ModelBackend.DASHSCOPE
    model_name: str = ""
    tools_used: list[str] = Field(default_factory=list)   # 本次实际调用的工具
    tool_calls_count: int = 0
    started_at: str = Field(default_factory=utc_now)
    finished_at: str | None = None
    error: str | None = None
    artifact_paths: dict[ArtifactType, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------------------ #
# 三类基础运行工件（结构化定义）
# ------------------------------------------------------------------ #

class ManifestArtifact(BaseModel):
    """工件1：运行清单 manifest.json"""

    artifact_type: Literal[ArtifactType.MANIFEST] = ArtifactType.MANIFEST
    run_id: str
    session_id: str
    manifest: RunManifest


class TraceArtifact(BaseModel):
    """工件2：执行轨迹 trace.jsonl"""

    artifact_type: Literal[ArtifactType.TRACE] = ArtifactType.TRACE
    run_id: str
    session_id: str
    events: list[TraceEvent] = Field(default_factory=list)
    event_count: int = 0


class CheckpointArtifact(BaseModel):
    """工件3：节点快照 checkpoints/*.json"""

    artifact_type: Literal[ArtifactType.CHECKPOINT] = ArtifactType.CHECKPOINT
    run_id: str
    session_id: str
    snapshots: list[CheckpointSnapshot] = Field(default_factory=list)
    snapshot_count: int = 0


class StructuredRunArtifacts(BaseModel):
    """三类工件的聚合视图，用于结构化落盘与恢复。"""

    manifest: ManifestArtifact
    trace: TraceArtifact
    checkpoint: CheckpointArtifact


class ExecutionChainRestore(BaseModel):
    """从 checkpoint 恢复的执行链路包。"""

    session_id: str
    run_id: str
    checkpoint_id: str
    restored_node: str
    restored_state: dict[str, Any] = Field(default_factory=dict)
    execution_chain: list[TraceEvent] = Field(default_factory=list)
    langgraph_config: dict[str, Any] = Field(default_factory=dict)
    artifacts: StructuredRunArtifacts | None = None
    resumable: bool = True
    message: str = ""


class RunContext(BaseModel):
    """
    单次运行的上下文句柄。

    在 chat_service 中创建，贯穿 workflow → trace → finish_run。
    """

    model_config = {"arbitrary_types_allowed": True}

    manifest: RunManifest
    seq: int = 0                  # 轨迹事件计数器

    def next_seq(self) -> int:
        """获取并递增事件序号。"""
        self.seq += 1
        return self.seq
