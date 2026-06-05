"""3 类运行工件落盘：Trace / Checkpoint / Manifest。"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings
from app.harness.enums import ArtifactType, RunStatus
from app.harness.models import (
    CheckpointArtifact,
    CheckpointSnapshot,
    ManifestArtifact,
    RunContext,
    RunManifest,
    StructuredRunArtifacts,
    TraceArtifact,
    TraceEvent,
    utc_now,
)


class ArtifactStore:
    """运行工件统一存储，支持结构化落盘与按 session 恢复。"""

    def __init__(self) -> None:
        self._base = get_settings().runs_path

    def run_dir(self, run_id: str) -> Path:
        path = self._base / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def artifacts_bundle_dir(self, run_id: str) -> Path:
        """结构化聚合工件目录 data/runs/{run_id}/artifacts/"""
        path = self.run_dir(run_id) / "artifacts"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "manifest.json"

    def trace_path(self, run_id: str) -> Path:
        return self.run_dir(run_id) / "trace.jsonl"

    def checkpoint_dir(self, run_id: str) -> Path:
        path = self.run_dir(run_id) / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def session_index_path(self) -> Path:
        """session_id → [(run_id, checkpoint_ids)] 索引。"""
        return self._base / "session_index.json"

    def create_run(
        self,
        chat_id: str,
        mode: str = "advisor",
        metadata: dict | None = None,
    ) -> RunContext:
        settings = get_settings()
        manifest = RunManifest(
            chat_id=chat_id,
            thread_id=chat_id,
            mode=mode,
            status=RunStatus.RUNNING,
            model_backend=settings.model_backend_enum,
            model_name=settings.llm_model,
            metadata=metadata or {},
        )
        self.run_dir(manifest.run_id)
        manifest.artifact_paths = {
            ArtifactType.MANIFEST: str(self.manifest_path(manifest.run_id)),
            ArtifactType.TRACE: str(self.trace_path(manifest.run_id)),
            ArtifactType.CHECKPOINT: str(self.checkpoint_dir(manifest.run_id)),
        }
        self.save_manifest(manifest)
        self._index_session_run(chat_id, manifest.run_id)
        return RunContext(manifest=manifest)

    def save_manifest(self, manifest: RunManifest) -> None:
        path = self.manifest_path(manifest.run_id)
        path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")

    def load_manifest(self, run_id: str) -> RunManifest | None:
        path = self.manifest_path(run_id)
        if not path.exists():
            return None
        return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))

    def finalize_run(
        self,
        ctx: RunContext,
        status: RunStatus = RunStatus.COMPLETED,
        error: str | None = None,
    ) -> None:
        ctx.manifest.status = status
        ctx.manifest.finished_at = utc_now()
        ctx.manifest.error = error
        self.save_manifest(ctx.manifest)
        self.persist_artifact_bundle(ctx.manifest.run_id)

    def append_trace(
        self,
        ctx: RunContext,
        event_type: str,
        node: str | None = None,
        payload: dict | None = None,
    ) -> TraceEvent:
        event = TraceEvent(
            run_id=ctx.manifest.run_id,
            seq=ctx.next_seq(),
            event_type=event_type,
            node=node,
            payload=payload or {},
        )
        path = self.trace_path(ctx.manifest.run_id)
        with path.open("a", encoding="utf-8") as f:
            f.write(event.model_dump_json() + "\n")
        return event

    def save_checkpoint(
        self,
        ctx: RunContext,
        node: str,
        state: dict,
        checkpoint_id: str | None = None,
        session_id: str | None = None,
    ) -> CheckpointSnapshot:
        cid = checkpoint_id or f"cp_{ctx.seq:04d}_{node}"
        sid = session_id or ctx.manifest.chat_id
        snapshot = CheckpointSnapshot(
            run_id=ctx.manifest.run_id,
            checkpoint_id=cid,
            session_id=sid,
            thread_id=ctx.manifest.thread_id,
            node=node,
            state=_serialize_state(state),
        )
        path = self.checkpoint_dir(ctx.manifest.run_id) / f"{snapshot.checkpoint_id}.json"
        path.write_text(snapshot.model_dump_json(indent=2), encoding="utf-8")

        self._index_session_checkpoint(sid, ctx.manifest.run_id, cid)
        self.append_trace(
            ctx,
            event_type="checkpoint",
            node=node,
            payload={"checkpoint_id": cid, "session_id": sid},
        )
        return snapshot

    def load_checkpoint(self, run_id: str, checkpoint_id: str) -> CheckpointSnapshot | None:
        path = self.checkpoint_dir(run_id) / f"{checkpoint_id}.json"
        if not path.exists():
            return None
        return CheckpointSnapshot.model_validate_json(path.read_text(encoding="utf-8"))

    def list_checkpoints(self, run_id: str) -> list[CheckpointSnapshot]:
        cp_dir = self.checkpoint_dir(run_id)
        snapshots: list[CheckpointSnapshot] = []
        for path in sorted(cp_dir.glob("*.json")):
            snapshots.append(
                CheckpointSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
            )
        return snapshots

    def read_trace(self, run_id: str) -> list[TraceEvent]:
        path = self.trace_path(run_id)
        if not path.exists():
            return []
        events: list[TraceEvent] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                events.append(TraceEvent.model_validate_json(line))
        return events

    def list_runs(self, chat_id: str | None = None) -> list[RunManifest]:
        manifests: list[RunManifest] = []
        if not self._base.exists():
            return manifests
        for run_dir in sorted(self._base.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir.name == "session_index.json":
                continue
            manifest = self.load_manifest(run_dir.name)
            if manifest and (chat_id is None or manifest.chat_id == chat_id):
                manifests.append(manifest)
        return manifests

    def load_structured_artifacts(self, run_id: str) -> StructuredRunArtifacts | None:
        """加载三类结构化工件聚合视图。"""
        manifest = self.load_manifest(run_id)
        if not manifest:
            return None

        session_id = manifest.chat_id
        events = self.read_trace(run_id)
        snapshots = self.list_checkpoints(run_id)

        return StructuredRunArtifacts(
            manifest=ManifestArtifact(
                run_id=run_id,
                session_id=session_id,
                manifest=manifest,
            ),
            trace=TraceArtifact(
                run_id=run_id,
                session_id=session_id,
                events=events,
                event_count=len(events),
            ),
            checkpoint=CheckpointArtifact(
                run_id=run_id,
                session_id=session_id,
                snapshots=snapshots,
                snapshot_count=len(snapshots),
            ),
        )

    def persist_artifact_bundle(self, run_id: str) -> StructuredRunArtifacts | None:
        """
        将三类工件聚合结构化落盘到 artifacts/ 子目录。

        便于外部系统一次性拉取完整运行快照。
        """
        bundle = self.load_structured_artifacts(run_id)
        if not bundle:
            return None

        out = self.artifacts_bundle_dir(run_id)
        (out / "manifest.json").write_text(
            bundle.manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        (out / "trace.json").write_text(
            bundle.trace.model_dump_json(indent=2), encoding="utf-8"
        )
        (out / "checkpoints.json").write_text(
            bundle.checkpoint.model_dump_json(indent=2), encoding="utf-8"
        )
        (out / "bundle.json").write_text(
            bundle.model_dump_json(indent=2), encoding="utf-8"
        )
        return bundle

    def find_run_by_session_and_checkpoint(
        self,
        session_id: str,
        checkpoint_id: str,
    ) -> tuple[str, CheckpointSnapshot] | None:
        """根据 session_id + checkpoint_id 定位 run 与快照。"""
        index = self._load_session_index()
        run_ids = index.get(session_id, [])

        # 优先从索引查找
        for run_id in reversed(run_ids):
            snapshot = self.load_checkpoint(run_id, checkpoint_id)
            if snapshot:
                return run_id, snapshot

        # 回退：全量扫描
        for manifest in self.list_runs(session_id):
            snapshot = self.load_checkpoint(manifest.run_id, checkpoint_id)
            if snapshot:
                return manifest.run_id, snapshot

        return None

    # ---------- session 索引 ----------

    def _load_session_index(self) -> dict[str, list[str]]:
        path = self.session_index_path()
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}

    def _save_session_index(self, index: dict[str, list[str]]) -> None:
        self.session_index_path().write_text(
            json.dumps(index, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _index_session_run(self, session_id: str, run_id: str) -> None:
        index = self._load_session_index()
        runs = index.setdefault(session_id, [])
        if run_id not in runs:
            runs.append(run_id)
        self._save_session_index(index)

    def _index_session_checkpoint(
        self, session_id: str, run_id: str, checkpoint_id: str
    ) -> None:
        self._index_session_run(session_id, run_id)
        cp_index_path = self.run_dir(run_id) / "checkpoint_index.json"
        cp_index: dict[str, list[str]] = {}
        if cp_index_path.exists():
            cp_index = json.loads(cp_index_path.read_text(encoding="utf-8"))
        ids = cp_index.setdefault(session_id, [])
        if checkpoint_id not in ids:
            ids.append(checkpoint_id)
        cp_index_path.write_text(json.dumps(cp_index, indent=2), encoding="utf-8")


def _serialize_state(state: dict) -> dict:
    result: dict = {}
    for key, value in state.items():
        try:
            json.dumps(value, default=str)
            result[key] = value
        except (TypeError, ValueError):
            result[key] = str(value)
    return result


artifact_store = ArtifactStore()
