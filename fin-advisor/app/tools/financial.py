"""金融理财 StructuredTool 工具集。"""

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


class CompoundInterestInput(BaseModel):
    principal: float = Field(..., description="本金（元）", gt=0)
    annual_rate: float = Field(..., description="年化利率（小数，如 0.05 表示 5%）", ge=0)
    years: float = Field(..., description="投资年限", gt=0)
    compound_per_year: int = Field(default=12, description="每年复利次数，月复利=12")


class DCAInput(BaseModel):
    monthly_amount: float = Field(..., description="每月定投金额（元）", gt=0)
    annual_rate: float = Field(..., description="预期年化收益率（小数）", ge=0)
    years: float = Field(..., description="定投年限", gt=0)


class RiskProfileInput(BaseModel):
    age: int = Field(..., description="年龄", ge=18, le=100)
    income_stability: int = Field(..., description="收入稳定性 1-5，5最稳定", ge=1, le=5)
    investment_experience: int = Field(..., description="投资经验 1-5，5最丰富", ge=1, le=5)
    loss_tolerance: int = Field(..., description="亏损承受度 1-5，5最高", ge=1, le=5)
    investment_horizon_years: int = Field(..., description="可投资期限（年）", ge=1)


def calc_compound_interest(
    principal: float,
    annual_rate: float,
    years: float,
    compound_per_year: int = 12,
) -> str:
    """复利终值计算。"""
    amount = principal * (1 + annual_rate / compound_per_year) ** (
        compound_per_year * years
    )
    profit = amount - principal
    return (
        f"复利测算结果：\n"
        f"- 本金：{principal:,.2f} 元\n"
        f"- 年化利率：{annual_rate * 100:.2f}%\n"
        f"- 投资年限：{years} 年\n"
        f"- 复利频率：每年 {compound_per_year} 次\n"
        f"- 终值：{amount:,.2f} 元\n"
        f"- 收益：{profit:,.2f} 元"
    )


def calc_dca(
    monthly_amount: float,
    annual_rate: float,
    years: float,
) -> str:
    """定投终值计算（期末一次性估值）。"""
    months = int(years * 12)
    monthly_rate = annual_rate / 12
    if monthly_rate == 0:
        total = monthly_amount * months
    else:
        total = monthly_amount * ((1 + monthly_rate) ** months - 1) / monthly_rate * (
            1 + monthly_rate
        )
    invested = monthly_amount * months
    profit = total - invested
    return (
        f"定投测算结果：\n"
        f"- 每月定投：{monthly_amount:,.2f} 元\n"
        f"- 预期年化：{annual_rate * 100:.2f}%\n"
        f"- 定投年限：{years} 年（共 {months} 期）\n"
        f"- 累计投入：{invested:,.2f} 元\n"
        f"- 预估终值：{total:,.2f} 元\n"
        f"- 预估收益：{profit:,.2f} 元"
    )


def calc_risk_profile(
    age: int,
    income_stability: int,
    investment_experience: int,
    loss_tolerance: int,
    investment_horizon_years: int,
) -> str:
    """风险偏好评分与类型判定。"""
    # 年龄因子：年轻可承受更高风险
    age_score = max(1, min(5, (65 - age) // 10 + 1))
    horizon_score = max(1, min(5, investment_horizon_years // 3 + 1))

    total = (
        age_score * 0.15
        + income_stability * 0.2
        + investment_experience * 0.25
        + loss_tolerance * 0.25
        + horizon_score * 0.15
    )
    score = round(total * 20, 1)  # 映射到 0-100

    if score < 40:
        profile = "保守型"
        advice = "建议以货币基金、国债、大额存单等低风险产品为主。"
    elif score < 60:
        profile = "稳健型"
        advice = "建议股债平衡，债券基金+指数定投组合，控制权益仓位 30%-50%。"
    elif score < 80:
        profile = "积极型"
        advice = "可适当提高权益类资产配置，关注行业 ETF 与优质主动基金。"
    else:
        profile = "激进型"
        advice = "可配置较高比例股票/权益基金，但需严格止损与分散投资。"

    return (
        f"风险偏好评分结果：\n"
        f"- 综合得分：{score}/100\n"
        f"- 类型判定：{profile}\n"
        f"- 配置建议：{advice}\n"
        f"- 评分维度：年龄={age_score}, 收入稳定性={income_stability}, "
        f"投资经验={investment_experience}, 亏损承受={loss_tolerance}, "
        f"投资期限={horizon_score}"
    )


def get_financial_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            func=calc_compound_interest,
            name="compound_interest",
            description="计算复利终值，适用于一次性投资场景",
            args_schema=CompoundInterestInput,
        ),
        StructuredTool.from_function(
            func=calc_dca,
            name="dca_investment",
            description="计算基金/理财定投终值",
            args_schema=DCAInput,
        ),
        StructuredTool.from_function(
            func=calc_risk_profile,
            name="risk_profile_score",
            description="根据用户特征评估风险偏好类型（保守/稳健/积极/激进）",
            args_schema=RiskProfileInput,
        ),
    ]
