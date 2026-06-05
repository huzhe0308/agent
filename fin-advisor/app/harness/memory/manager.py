"""分层 Agent 记忆：Working / Episodic / Semantic / Summary。"""

from __future__ import annotations

import pickle
import re
from datetime import datetime, timezone
from pathlib import Path

from app.chains.summary import summarize
from app.config import get_settings
from app.models.schemas import (
    EpisodicTurn,
    MemoryContext,
    SemanticFact,
    SessionData,
    SessionMessage,
)


class AgentMemoryManager:
    """
    四层记忆架构：
    - Working Memory：当前窗口内的原始消息
    - Episodic Memory：带时间戳/重要性/意图的结构化回合
    - Semantic Memory：从对话中提取的用户事实（风险偏好、目标等）
    - Summary Memory：滚动 LLM 摘要（长对话压缩）
    """

    def __init__(self) -> None:
        self._dir = get_settings().session_path  # 默认 ./data/sessions

    def _pickle_path(self, chat_id: str) -> Path:
        """Pickle 文件路径（高性能读写）。"""
        return self._dir / f"{chat_id}.pkl"

    def _json_path(self, chat_id: str) -> Path:
        """JSON 镜像路径（人类可读，用于备份/调试）。"""
        return self._dir / f"{chat_id}.json"

    def load(self, chat_id: str) -> SessionData:
        """加载会话：优先 pickle，回退 JSON，均不存在则新建。"""
        pickle_path = self._pickle_path(chat_id)
        if pickle_path.exists():
            with pickle_path.open("rb") as f:
                return pickle.load(f)

        json_path = self._json_path(chat_id)
        if json_path.exists():
            with json_path.open("r", encoding="utf-8") as f:
                return SessionData.model_validate_json(f.read())

        return SessionData(chat_id=chat_id)

    def save(self, session: SessionData) -> None:
        """双写持久化：pickle（快）+ JSON（可读）。"""
        session.updated_at = datetime.now(timezone.utc).isoformat()
        with self._pickle_path(session.chat_id).open("wb") as f:
            pickle.dump(session, f)
        with self._json_path(session.chat_id).open("w", encoding="utf-8") as f:
            f.write(session.model_dump_json(indent=2))

    def append_turn(
        self,
        chat_id: str,
        role: str,
        content: str,
        *,
        intent: str | None = None,
        importance: float | None = None,
    ) -> SessionData:
        """
        追加一个对话回合。

        同时更新 Working Memory（messages）和 Episodic Memory（episodes），
        用户消息还会触发语义事实提取。
        """
        session = self.load(chat_id)
        turn_id = session.turn_counter + 1
        session.turn_counter = turn_id
        now = datetime.now(timezone.utc).isoformat()

        msg = SessionMessage(role=role, content=content, turn_id=turn_id, timestamp=now)
        session.messages.append(msg)

        imp = importance if importance is not None else _estimate_importance(content, role)
        episode = EpisodicTurn(
            turn_id=turn_id,
            role=role,
            content=content,
            timestamp=now,
            intent=intent,
            importance=imp,
        )
        session.episodes.append(episode)

        settings = get_settings()
        max_msgs = settings.max_history_turns * 2
        if len(session.messages) > max_msgs:
            session.messages = session.messages[-max_msgs:]

        max_episodes = settings.max_episodic_turns
        if len(session.episodes) > max_episodes:
            session.episodes = _prune_episodes(session.episodes, max_episodes)

        if role == "user":
            self._extract_semantic_facts(session, content)

        self.save(session)
        return session

    def update_summary(self, chat_id: str, summary: str) -> SessionData:
        session = self.load(chat_id)
        session.summary = summary
        self.save(session)
        return session

    def upsert_semantic_fact(
        self,
        chat_id: str,
        key: str,
        value: str,
        confidence: float = 1.0,
        source_turn: int = 0,
    ) -> SessionData:
        session = self.load(chat_id)
        for fact in session.semantic_facts:
            if fact.key == key:
                fact.value = value
                fact.confidence = confidence
                fact.source_turn = source_turn
                self.save(session)
                return session

        session.semantic_facts.append(
            SemanticFact(
                key=key,
                value=value,
                confidence=confidence,
                source_turn=source_turn,
            )
        )
        self.save(session)
        return session

    def build_memory_context(self, chat_id: str) -> MemoryContext:
        """组装四层记忆的结构化视图（供 API 查看或内部调试）。"""
        session = self.load(chat_id)
        settings = get_settings()

        working = session.messages[-settings.working_memory_turns * 2 :]
        episodic = sorted(
            session.episodes,
            key=lambda e: e.importance,
            reverse=True,
        )[: settings.episodic_recall_count]

        return MemoryContext(
            chat_id=chat_id,
            summary=session.summary,
            working_memory=working,
            episodic_memory=episodic,
            semantic_facts=session.semantic_facts,
            last_checkpoint_id=session.last_checkpoint_id,
        )

    def format_for_prompt(self, chat_id: str) -> str:
        """
        将四层记忆组装为 Prompt 文本，注入 LLM 上下文。

        组装顺序：摘要 → 用户画像 → 高重要性历史 → 近期对话
        """
        ctx = self.build_memory_context(chat_id)
        lines: list[str] = []

        if ctx.summary:
            lines.append(f"[对话摘要]\n{ctx.summary}")

        if ctx.semantic_facts:
            facts = "；".join(f"{f.key}={f.value}" for f in ctx.semantic_facts)
            lines.append(f"[用户画像]\n{facts}")

        important_episodes = [
            e for e in ctx.episodic_memory if e.importance >= 0.6 and e not in _in_working(e, ctx.working_memory)
        ]
        if important_episodes:
            ep_lines = [
                f"- [{e.role}|重要度{e.importance:.1f}] {e.content[:200]}"
                for e in important_episodes[:5]
            ]
            lines.append("[关键历史回合]\n" + "\n".join(ep_lines))

        if ctx.working_memory:
            turn_lines = []
            for msg in ctx.working_memory:
                label = "用户" if msg.role == "user" else "助手"
                turn_lines.append(f"{label}: {msg.content}")
            lines.append("[近期对话]\n" + "\n".join(turn_lines))

        return "\n\n".join(lines)

    def refresh_summary(self, chat_id: str, recent_turns: int = 3) -> str:
        session = self.load(chat_id)
        recent = session.messages[-recent_turns * 2 :]
        if not recent:
            return session.summary

        new_msgs = "\n".join(
            f"{'用户' if m.role == 'user' else '助手'}: {m.content}" for m in recent
        )
        updated = summarize(session.summary, new_msgs)
        self.update_summary(chat_id, updated)
        return updated

    def set_checkpoint(self, chat_id: str, checkpoint_id: str) -> None:
        session = self.load(chat_id)
        session.last_checkpoint_id = checkpoint_id
        self.save(session)

    def clear(self, chat_id: str) -> None:
        for path in (self._pickle_path(chat_id), self._json_path(chat_id)):
            if path.exists():
                path.unlink()

    def _extract_semantic_facts(self, session: SessionData, content: str) -> None:
        """规则 + 模式提取语义记忆（轻量，无额外 LLM 调用）。"""
        patterns = [
            (r"风险偏好[是为：:\s]*([^\s，,。.]+)", "risk_profile"),
            (r"(保守|稳健|积极|激进)型?", "risk_profile"),
            (r"(\d+)\s*岁", "age"),
            (r"月收入[约为：:\s]*(\d+)", "monthly_income"),
            (r"投资目标[是为：:\s]*([^，,。.]+)", "investment_goal"),
            (r"可投资[金资]?金[约为：:\s]*(\d+)", "investable_amount"),
            (r"投资期限[约为：:\s]*(\d+)\s*年", "investment_horizon"),
        ]
        for pattern, key in patterns:
            match = re.search(pattern, content)
            if match:
                value = match.group(1)
                self._upsert_fact_in_session(session, key, value, session.turn_counter)

    def _upsert_fact_in_session(
        self,
        session: SessionData,
        key: str,
        value: str,
        source_turn: int,
    ) -> None:
        for fact in session.semantic_facts:
            if fact.key == key:
                fact.value = value
                fact.source_turn = source_turn
                return
        session.semantic_facts.append(
            SemanticFact(key=key, value=value, source_turn=source_turn)
        )


def _estimate_importance(content: str, role: str) -> float:
    score = 0.4
    if role == "user":
        score += 0.1
    keywords = ["风险", "收益", "定投", "复利", "报告", "预算", "目标", "配置"]
    score += min(0.4, sum(0.08 for kw in keywords if kw in content))
    if len(content) > 100:
        score += 0.1
    return min(1.0, score)


def _prune_episodes(episodes: list[EpisodicTurn], max_count: int) -> list[EpisodicTurn]:
    """保留高重要性 + 最近回合。"""
    by_importance = sorted(episodes, key=lambda e: e.importance, reverse=True)
    keep_ids = {e.turn_id for e in by_importance[: max_count // 2]}
    keep_ids.update(e.turn_id for e in episodes[-(max_count // 2) :])
    return [e for e in episodes if e.turn_id in keep_ids][-max_count:]


def _in_working(episode: EpisodicTurn, working: list[SessionMessage]) -> bool:
    return any(m.turn_id == episode.turn_id for m in working)


agent_memory = AgentMemoryManager()
