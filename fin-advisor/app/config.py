"""
全局配置模块。

从 .env 文件加载所有运行时参数，包括：
- 模型后端（DashScope / OpenAI 兼容）
- 记忆与 Checkpoint 策略
- RAG / MCP / 服务端口等
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from app.harness.enums import ModelBackend


class Settings(BaseSettings):
    """应用配置，字段名与 .env 环境变量一一对应。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---------- 模型后端（2 类）----------
    model_backend: str = "dashscope"          # dashscope | openai_compat
    dashscope_api_key: str = ""               # 阿里云 DashScope API Key
    llm_model: str = "qwen-plus"              # 通义千问模型名
    llm_temperature: float = 0.7

    openai_api_key: str = ""                  # OpenAI 兼容后端 API Key
    openai_base_url: str = "http://localhost:11434/v1"  # Ollama/vLLM 地址
    openai_model: str = "llama3"

    # ---------- RAG（百炼知识库，三域 Index）----------
    bailian_workspace_id: str = ""
    bailian_index_id: str = ""              # 默认/回退 Index
    bailian_index_knowledge_id: str = ""    # 理财知识库
    bailian_index_product_id: str = ""      # 产品文档库
    bailian_index_regulatory_id: str = ""   # 监管政策库
    bailian_enable: bool = False

    # ---------- 分层记忆 ----------
    session_dir: str = "./data/sessions"      # 会话持久化目录
    max_history_turns: int = 10               # Working Memory 最大轮次
    working_memory_turns: int = 6             # 注入 Prompt 的近期轮次
    max_episodic_turns: int = 50              # Episodic Memory 上限
    episodic_recall_count: int = 8            # 召回的高重要性回合数
    enable_summary: bool = True               # 是否启用滚动 LLM 摘要

    # ---------- LangGraph Checkpoint ----------
    checkpoint_backend: str = "sqlite"        # sqlite | memory
    checkpoint_dir: str = "./data/checkpoints"

    # ---------- 运行工件（可复盘）----------
    runs_dir: str = "./data/runs"

    # ---------- 工具安全边界 ----------
    tool_duplicate_window_seconds: float = 30.0  # 重复调用拦截窗口（秒）

    # ---------- MCP 外部工具 ----------
    mcp_enable: bool = False
    mcp_servers_file: str = "./mcp-servers.json"

    # ---------- HTTP 服务 ----------
    host: str = "0.0.0.0"
    port: int = 8123
    api_prefix: str = "/api"

    @property
    def model_backend_enum(self) -> ModelBackend:
        """将字符串配置转为枚举。"""
        return ModelBackend(self.model_backend)

    @property
    def session_path(self) -> Path:
        """会话/记忆存储路径（自动创建）。"""
        path = Path(self.session_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def checkpoint_db_path(self) -> Path:
        """LangGraph SqliteSaver 数据库路径。"""
        path = Path(self.checkpoint_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path / "checkpoints.db"

    @property
    def runs_path(self) -> Path:
        """运行工件根目录（每次对话生成一个 run_id 子目录）。"""
        path = Path(self.runs_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache
def get_settings() -> Settings:
    """单例获取配置（进程内缓存）。"""
    return Settings()
