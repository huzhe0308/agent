"""
LangGraph Checkpoint 持久化与执行链路恢复。

两层 Checkpoint 机制：
1. LangGraph Checkpointer（SqliteSaver / MemorySaver）—— 图运行时状态
2. 工件级快照（ArtifactStore）—— 三类结构化工件落盘

核心恢复接口：
  restore_execution_chain(session_id, checkpoint_id)
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.config import get_settings
from app.harness.artifacts.store import artifact_store
from app.harness.enums import ArtifactType, RunStatus
from app.harness.models import (
    CheckpointArtifact,
    ExecutionChainRestore,
    ManifestArtifact,
    RunContext,
    StructuredRunArtifacts,
    TraceArtifact,
    TraceEvent,
)


class CheckpointManager:
    """管理 LangGraph checkpointer 与三类工件的恢复。"""

    def __init__(self) -> None:
        self._sqlite_saver = None
        self._memory_saver = MemorySaver()

    @property
    def checkpointer(self):
        settings = get_settings()
        if settings.checkpoint_backend == "sqlite":
            return self._get_sqlite_saver()
        return self._memory_saver

    def _get_sqlite_saver(self):
        if self._sqlite_saver is None:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver

                conn_str = str(get_settings().checkpoint_db_path)
                self._sqlite_saver = SqliteSaver.from_conn_string(conn_str)
            except Exception:
                self._sqlite_saver = self._memory_saver
        return self._sqlite_saver

    def thread_config(
        self,
        thread_id: str,
        run_id: str | None = None,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """构建 LangGraph 调用配置；恢复时可注入 checkpoint_id。"""
        cfg: dict[str, Any] = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": 25,
        }
        if run_id:
            cfg["configurable"]["run_id"] = run_id
        if checkpoint_id:
            cfg["configurable"]["checkpoint_id"] = checkpoint_id
        return cfg

    def snapshot_node(self, ctx: RunContext, node: str, state: dict) -> None:
        """节点进入时保存工件级 checkpoint 快照。"""
        artifact_store.save_checkpoint(ctx, node, state, session_id=ctx.manifest.chat_id)

    def get_latest_checkpoint_id(self, run_id: str) -> str | None:
        checkpoints = artifact_store.list_checkpoints(run_id)
        return checkpoints[-1].checkpoint_id if checkpoints else None

    def restore_state(self, run_id: str, checkpoint_id: str) -> dict | None:
        """从工件快照读取状态（只读）。"""
        snapshot = artifact_store.load_checkpoint(run_id, checkpoint_id)
        return snapshot.state if snapshot else None

    def load_structured_artifacts(self, run_id: str) -> StructuredRunArtifacts | None:
        """加载某次运行的三类结构化工件。"""
        return artifact_store.load_structured_artifacts(run_id)

    def persist_structured_artifacts(self, run_id: str) -> StructuredRunArtifacts | None:
        """
        将三类工件聚合后结构化落盘到 run 目录 artifacts/ 子目录。

        落盘路径:
          data/runs/{run_id}/artifacts/manifest.json
          data/runs/{run_id}/artifacts/trace.json
          data/runs/{run_id}/artifacts/checkpoints.json
        """
        return artifact_store.persist_artifact_bundle(run_id)

    def restore_execution_chain(
        self,
        session_id: str,
        checkpoint_id: str,
    ) -> ExecutionChainRestore | None:
        """
        根据 session_id 从指定 checkpoint 恢复执行链路。

        步骤:
        1. 定位包含该 checkpoint 的 run_id
        2. 加载三类工件（manifest / trace / checkpoint）
        3. 截取 checkpoint 之前的执行轨迹
        4. 返回可续跑的状态包（含 LangGraph config）
        """
        located = artifact_store.find_run_by_session_and_checkpoint(session_id, checkpoint_id)
        if not located:
            return None

        run_id, snapshot = located
        artifacts = artifact_store.load_structured_artifacts(run_id)
        if not artifacts:
            return None

        # 截取到目标 checkpoint 为止的执行链（含 checkpoint 事件）
        execution_chain = _slice_trace_to_checkpoint(
            artifacts.trace.events,
            checkpoint_id,
            snapshot.node,
        )

        manifest = artifacts.manifest.manifest
        resumable = manifest.status in (RunStatus.RUNNING, RunStatus.INTERRUPTED, RunStatus.FAILED)

        return ExecutionChainRestore(
            session_id=session_id,
            run_id=run_id,
            checkpoint_id=checkpoint_id,
            restored_node=snapshot.node,
            restored_state=snapshot.state,
            execution_chain=execution_chain,
            langgraph_config=self.thread_config(
                thread_id=session_id,
                run_id=run_id,
                checkpoint_id=checkpoint_id,
            ),
            artifacts=artifacts,
            resumable=resumable,
            message=(
                f"已从节点 '{snapshot.node}' 恢复执行链路，"
                f"共 {len(execution_chain)} 条轨迹事件"
            ),
        )


def _slice_trace_to_checkpoint(
    events: list[TraceEvent],
    checkpoint_id: str,
    node: str,
) -> list[TraceEvent]:
    """截取 trace 中到目标 checkpoint 为止的事件链。"""
    chain: list[TraceEvent] = []
    for event in events:
        chain.append(event)
        if event.event_type == "checkpoint":
            payload_cp = event.payload.get("checkpoint_id")
            if payload_cp == checkpoint_id:
                break
        if event.event_type == "node_enter" and event.node == node:
            # 兼容无 checkpoint 事件标记的情况：停在目标节点
            if not any(
                e.event_type == "checkpoint"
                and e.payload.get("checkpoint_id") == checkpoint_id
                for e in chain
            ):
                pass
    return chain


@lru_cache
def get_checkpoint_manager() -> CheckpointManager:
    return CheckpointManager()


def restore_execution_chain(session_id: str, checkpoint_id: str) -> ExecutionChainRestore | None:
    """
    模块级恢复函数（对外统一入口）。

    根据 session_id（即 chat_id）和 checkpoint_id 恢复完整执行链路。
    """
    return get_checkpoint_manager().restore_execution_chain(session_id, checkpoint_id)
