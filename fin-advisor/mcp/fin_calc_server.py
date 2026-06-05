#!/usr/bin/env python
"""MCP 金融计算工具服务器（复利 / 定投 / 风险偏好评分）。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from app.tools.financial import calc_compound_interest, calc_dca, calc_risk_profile

mcp = FastMCP("fin-calc")


@mcp.tool()
def compound_interest(
    principal: float,
    annual_rate: float,
    years: float,
    compound_per_year: int = 12,
) -> str:
    """计算复利终值。"""
    return calc_compound_interest(principal, annual_rate, years, compound_per_year)


@mcp.tool()
def dca_investment(monthly_amount: float, annual_rate: float, years: float) -> str:
    """计算定投终值。"""
    return calc_dca(monthly_amount, annual_rate, years)


@mcp.tool()
def risk_profile_score(
    age: int,
    income_stability: int,
    investment_experience: int,
    loss_tolerance: int,
    investment_horizon_years: int,
) -> str:
    """评估用户风险偏好类型。"""
    return calc_risk_profile(
        age,
        income_stability,
        investment_experience,
        loss_tolerance,
        investment_horizon_years,
    )


if __name__ == "__main__":
    mcp.run()
