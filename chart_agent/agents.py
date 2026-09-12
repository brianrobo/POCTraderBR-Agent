from typing import Any, Dict, Optional

import anthropic

from .models import ChartAnalysisResult
from .vision import encode_image

MODEL = "claude-opus-5"


def build_system_prompt(agent_cfg: Dict[str, Any]) -> str:
    role = agent_cfg["role"].strip()
    criteria = agent_cfg["criteria"]

    criteria_block = "\n".join(f"{i+1}. {c.strip()}" for i, c in enumerate(criteria))

    return (
        f"{role}\n\n"
        "다음 판단 기준들을 하나씩 확인하고, 첨부된 차트 이미지를 근거로 판단하라. "
        "차트에 명확히 나타나지 않는 내용은 추측하지 말고 found=false로 표시하라.\n\n"
        f"[판단 기준]\n{criteria_block}\n\n"
        "각 기준에 대해 signals 항목을 하나씩 작성하고, "
        "하나 이상의 기준에서 신호가 발견되면 overall_found=true로 설정하라."
    )


def analyze_chart(
    client: anthropic.Anthropic,
    timeframe_key: str,
    agent_cfg: Dict[str, Any],
    image_path: str,
    ticker: Optional[str] = None,
) -> ChartAnalysisResult:
    """하나의 타임프레임 차트 이미지를 지정된 기준으로 분석"""
    system_prompt = build_system_prompt(agent_cfg)

    ticker_note = f"분석 대상 종목: {ticker}\n\n" if ticker else ""
    user_text = (
        f"{ticker_note}첨부된 {agent_cfg['label']} 차트 이미지를 분석 기준에 따라 평가하라."
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
                    encode_image(image_path),
                    {"type": "text", "text": user_text},
                ],
            }
        ],
        output_format=ChartAnalysisResult,
    )

    result = response.parsed_output
    result.timeframe = timeframe_key
    return result
