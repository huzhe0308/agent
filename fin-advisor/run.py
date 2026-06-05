#!/usr/bin/env python
"""启动 FinAdvisor 服务：uvicorn app.main:app --reload"""

import uvicorn

from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
    )
