"""
FastAPI 应用入口。

职责：
1. 创建 FastAPI 实例并注册路由（对话 API + Harness API）
2. 启动时初始化数据目录（会话、运行工件、Checkpoint）
3. 提供健康检查端点，展示 Harness 能力概览
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import harness_router, router as ai_router
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时创建必要的数据目录。"""
    settings = get_settings()
    settings.session_path.mkdir(parents=True, exist_ok=True)       # 会话/记忆存储
    settings.runs_path.mkdir(parents=True, exist_ok=True)          # 运行工件（trace/manifest）
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)  # LangGraph Checkpoint
    yield


def create_app() -> FastAPI:
    """工厂函数：组装 FastAPI 应用。"""
    settings = get_settings()
    app = FastAPI(
        title="FinAdvisor",
        description="面向金融理财咨询场景的企业级智能平台（Agent Harness）",
        version="2.0.0",
        lifespan=lifespan,
    )

    # 允许前端跨域访问（开发环境）
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health():
        """健康检查：返回服务状态及 Harness 支持的模型/工具/工件类型。"""
        from app.harness.enums import ModelBackend, ToolType

        return {
            "status": "ok",
            "service": "fin-advisor",
            "harness": {
                "model_backends": [b.value for b in ModelBackend],
                "tool_types": [t.value for t in ToolType],
                "artifact_types": ["trace", "checkpoint", "manifest"],
            },
        }

    # 注册路由：/api/ai/* 对话接口，/api/harness/* Harness 管理接口
    app.include_router(ai_router, prefix=settings.api_prefix)
    app.include_router(harness_router, prefix=settings.api_prefix)
    return app


app = create_app()
