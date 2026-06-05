"""
评测公共 fixtures。

维度独立：各 test_*.py 仅共享基础配置，不合并总分。
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest

from app.harness.artifacts.store import artifact_store
from app.harness.enums import RunStatus
from app.harness.models import RunContext
from app.harness.tools.executor import ToolExecutor
from app.harness.tools.registry import load_sync_tools
from app.harness.enums import ToolType

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture(scope="session")
def long_dialog(fixtures_dir: Path) -> dict:
    path = fixtures_dir / "long_dialog.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def eval_run_ctx(tmp_path, monkeypatch) -> RunContext:
    """隔离评测运行目录，避免污染生产 data/。"""
    runs_dir = tmp_path / "runs"
    runs_dir.mkdir()
    monkeypatch.setenv("RUNS_DIR", str(runs_dir))
    from app.config import get_settings

    get_settings.cache_clear()
    ctx = artifact_store.create_run(chat_id="eval-session", mode="eval")
    return ctx


@pytest.fixture
def tool_executor(eval_run_ctx: RunContext) -> ToolExecutor:
    tools = load_sync_tools([
        ToolType.STRUCTURED,
        ToolType.HTTP,
        ToolType.SUBAGENT,
        ToolType.BUILTIN,
    ])
    return ToolExecutor(tools, eval_run_ctx, session_id="eval-session")


@pytest.fixture
def rng() -> random.Random:
    seed = int(os.getenv("EVAL_RANDOM_SEED", "42"))
    return random.Random(seed)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "llm: 需要真实 LLM（EVAL_USE_LLM=true）",
    )
