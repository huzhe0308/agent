"""兼容层：委托给 AgentMemoryManager。"""

from app.harness.memory.manager import agent_memory
from app.models.schemas import SessionData, SessionMessage


class SessionStore:
    """向后兼容的 SessionStore 门面。"""

    def load(self, chat_id: str) -> SessionData:
        return agent_memory.load(chat_id)

    def save(self, session: SessionData) -> None:
        agent_memory.save(session)

    def append_message(self, chat_id: str, role: str, content: str) -> SessionData:
        return agent_memory.append_turn(chat_id, role, content)

    def update_summary(self, chat_id: str, summary: str) -> SessionData:
        return agent_memory.update_summary(chat_id, summary)

    def get_recent_history(self, chat_id: str, turns: int | None = None) -> list[SessionMessage]:
        session = agent_memory.load(chat_id)
        from app.config import get_settings

        limit = (turns or get_settings().max_history_turns) * 2
        return session.messages[-limit:]

    def format_history(self, chat_id: str) -> str:
        return agent_memory.format_for_prompt(chat_id)

    def _pickle_path(self, chat_id: str):
        return agent_memory._pickle_path(chat_id)

    def _json_path(self, chat_id: str):
        return agent_memory._json_path(chat_id)


session_store = SessionStore()
