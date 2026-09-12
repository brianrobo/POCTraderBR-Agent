from typing import Any, Dict, Optional

import anthropic

from .chart_input import ChartInput
from .models import ChartAnalysisResult

MODEL = "claude-opus-5"


def build_system_prompt(agent_cfg: Dict[str, Any]) -> str:
    role = agent_cfg["role"].strip()
    criteria = agent_cfg["criteria"]

    criteria_block = "\n".join(f"{i+1}. {c.strip()}" for i, c in enumerate(criteria))

    return (
        f"{role}\n\n"
        "다음 판단 기준들을 하나씩 확인하고, 제공된 차트 자료(이미지 또는 캔들 데이터)를 "
        "근거로 판단하라. 근거가 명확하지 않은 내용은 추측하지 말고 found=false로 표시하라.\n\n"
        f"[판단 기준]\n{criteria_block}\n\n"
        "각 기준에 대해 signals 항목을 하나씩 작성하고, "
        "하나 이상의 기준에서 신호가 발견되면 overall_found=true로 설정하라."
    )


def analyze_chart(
    client: anthropic.Anthropic,
    timeframe_key: str,
    agent_cfg: Dict[str, Any],
    chart_input: ChartInput,
    ticker: Optional[str] = None,
) -> ChartAnalysisResult:
    """하나의 타임프레임 차트 자료(이미지 또는 캔들 데이터)를 지정된 기준으로 분석.

    chart_input이 이미지에서 왔는지 증권사 API 캔들 데이터에서 왔는지는 여기서
    신경 쓰지 않는다 (ChartInput.content_blocks()/describe()가 흡수).
    """
    system_prompt = build_system_prompt(agent_cfg)

    ticker_note = f"분석 대상 종목: {ticker}\n\n" if ticker else ""
    user_text = (
        f"{ticker_note}{chart_input.describe(agent_cfg['label'])}를 분석 기준에 따라 평가하라."
    )

    response = client.messages.parse(
        model=MODEL,
        max_tokens=4096,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    *chart_input.content_blocks(),
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        output_format=ChartAnalysisResult,
    )

    result = response.parsed_output
    result.timeframe = timeframe_key
    return result
