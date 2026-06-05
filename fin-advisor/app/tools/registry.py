"""委托 Harness 工具注册。"""

from langchain_core.tools import StructuredTool

from app.harness.tools.registry import load_all_tools, load_mcp_tools, load_sync_tools
from app.tools.financial import get_financial_tools


def get_all_tools() -> list[StructuredTool]:
    return load_sync_tools()


async def get_all_tools_async() -> list[StructuredTool]:
    return await load_all_tools()
