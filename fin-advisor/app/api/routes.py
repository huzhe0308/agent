"""
API 路由层。

分为两组：
1. /api/ai/*     —— 对话接口（SSE 流式 / 同步 REST）
2. /api/harness/* —— Harness 管理接口（运行记录 / trace / checkpoint / 记忆查看）
"""

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.harness.harness import get_harness
from app.harness.memory.manager import agent_memory
from app.memory.session_store import session_store
from app.models.schemas import (
    ChatRequest,
    ChatResponse,
    MemoryContext,
    ResumeRequest,
    RunSummary,
    SessionData,
)
from app.services.chat_service import chat_stream, manus_stream

router = APIRouter(prefix="/ai", tags=["AI"])
harness_router = APIRouter(prefix="/harness", tags=["Agent Harness"])


def _sse_response(stream: AsyncIterator[str]) -> StreamingResponse:
    """将异步字符串流包装为 SSE（Server-Sent Events）响应。"""
    async def event_generator():
        async for token in stream:
            yield f"data: {token}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ======================== Harness 管理 API ========================

@harness_router.get("/tools")
async def list_tools():
    """查看 7 类工具的元信息目录。"""
    return {"tools": get_harness().get_tool_catalog()}


@harness_router.get("/runs")
async def list_runs(chatId: str | None = Query(default=None)):
    """列出历史运行记录，可按 chatId 过滤。"""
    manifests = get_harness().list_runs(chatId)
    return {
        "runs": [
            RunSummary(
                run_id=m.run_id,
                chat_id=m.chat_id,
                mode=m.mode,
                status=m.status.value,
                started_at=m.started_at,
                finished_at=m.finished_at,
                tools_used=m.tools_used,
                tool_calls_count=m.tool_calls_count,
            ).model_dump()
            for m in manifests
        ]
    }


@harness_router.get("/runs/{run_id}/manifest")
async def get_run_manifest(run_id: str):
    """获取某次运行的 manifest.json 内容。"""
    manifest = get_harness().get_run_manifest(run_id)
    if not manifest:
        raise HTTPException(404, "运行记录不存在")
    return manifest


@harness_router.get("/runs/{run_id}/trace")
async def get_run_trace(run_id: str):
    """获取某次运行的完整 trace.jsonl 轨迹。"""
    from app.harness.artifacts.store import artifact_store

    events = artifact_store.read_trace(run_id)
    return {"run_id": run_id, "events": [e.model_dump() for e in events]}


@harness_router.get("/runs/{run_id}/checkpoints")
async def get_run_checkpoints(run_id: str):
    """获取某次运行的所有 checkpoint 快照。"""
    from app.harness.artifacts.store import artifact_store

    snapshots = artifact_store.list_checkpoints(run_id)
    return {"run_id": run_id, "checkpoints": [s.model_dump() for s in snapshots]}


@harness_router.post("/runs/{run_id}/resume")
async def resume_run(run_id: str, body: ResumeRequest):
    """从 checkpoint 恢复状态（只读，用于复盘 / 调试）。"""
    state = get_harness().resume_from_checkpoint(run_id, body.checkpoint_id)
    if state is None:
        raise HTTPException(404, "Checkpoint 不存在")
    return {"run_id": run_id, "checkpoint_id": body.checkpoint_id, "state": state}


@harness_router.post("/sessions/{session_id}/restore")
async def restore_session_chain(session_id: str, body: ResumeRequest):
    """根据 session_id + checkpoint_id 恢复完整执行链路。"""
    result = get_harness().restore_execution_chain(session_id, body.checkpoint_id)
    if result is None:
        raise HTTPException(404, "未找到对应的 session/checkpoint")
    return result


@harness_router.get("/approvals/pending")
async def list_pending_approvals(sessionId: str | None = Query(default=None)):
    """列出待审批的高风险工具调用（人工审批接口预留）。"""
    from app.harness.tools.security import approval_gate

    pending = approval_gate.list_pending(sessionId)
    return {"pending": [p.model_dump() for p in pending]}


@harness_router.post("/approvals/{approval_id}/approve")
async def approve_tool_call(approval_id: str, approver: str = Query(default="admin")):
    """批准高风险工具调用（审批通过后可用 approval_id 作为 approval_token 重试）。"""
    from app.harness.tools.security import approval_gate

    if not approval_gate.approve(approval_id, approver):
        raise HTTPException(404, "审批单不存在或已处理")
    return {"approval_id": approval_id, "status": "approved"}


@harness_router.post("/approvals/{approval_id}/reject")
async def reject_tool_call(
    approval_id: str,
    approver: str = Query(default="admin"),
    reason: str = Query(default=""),
):
    """拒绝高风险工具调用。"""
    from app.harness.tools.security import approval_gate

    if not approval_gate.reject(approval_id, approver, reason):
        raise HTTPException(404, "审批单不存在或已处理")
    return {"approval_id": approval_id, "status": "rejected"}


@harness_router.get("/memory/{chat_id}", response_model=MemoryContext)
async def get_memory_context(chat_id: str):
    """查看指定会话的四层记忆组装视图。"""
    return agent_memory.build_memory_context(chat_id)


# ======================== 对话 API ========================

@router.get("/fin_advisor/session/{chat_id}", response_model=SessionData)
async def get_session(chat_id: str):
    """获取会话历史数据。"""
    return session_store.load(chat_id)


@router.delete("/fin_advisor/session/{chat_id}")
async def clear_session(chat_id: str):
    """清除会话（删除记忆文件）。"""
    agent_memory.clear(chat_id)
    return {"chatId": chat_id, "cleared": True}


@router.get("/fin_advisor/chat/sse")
async def fin_advisor_chat_sse(
    message: str = Query(..., description="用户消息"),
    chatId: str = Query(..., description="会话 ID"),
    riskProfile: str | None = Query(default=None, description="风险偏好"),
):
    """金融理财咨询 SSE 流式对话（主入口）。"""
    return _sse_response(chat_stream(message, chatId, riskProfile))


@router.post("/fin_advisor/chat", response_model=ChatResponse)
async def fin_advisor_chat_post(body: ChatRequest):
    """REST 同步对话（JSON Body）。"""
    parts: list[str] = []
    async for token in chat_stream(body.message, body.chat_id, body.risk_profile):
        parts.append(token)
    return ChatResponse(chatId=body.chat_id, answer="".join(parts))


@router.get("/fin_advisor/chat/sync")
async def fin_advisor_chat_sync(message: str = Query(...), chatId: str = Query(...)):
    """同步对话（非流式，便于调试）。"""
    parts: list[str] = []
    async for token in chat_stream(message, chatId):
        parts.append(token)
    return {"chatId": chatId, "answer": "".join(parts)}


@router.get("/manus/chat")
async def manus_chat_sse(
    message: str = Query(...),
    chatId: str = Query(default="manus_default"),
):
    """ReAct 超级智能体 SSE 流式对话。"""
    return _sse_response(manus_stream(message, chatId))


# 兼容旧前端路由
@router.get("/love_app/chat/sse")
async def love_app_chat_sse_alias(message: str = Query(...), chatId: str = Query(...)):
    return _sse_response(chat_stream(message, chatId))


@router.get("/love_app/chat/sync")
async def love_app_chat_sync_alias(message: str = Query(...), chatId: str = Query(...)):
    parts: list[str] = []
    async for token in chat_stream(message, chatId):
        parts.append(token)
    return {"chatId": chatId, "answer": "".join(parts)}
