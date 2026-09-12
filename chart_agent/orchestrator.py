import json
from typing import Dict, Optional

import anthropic

from .models import ChartAnalysisResult

MODEL = "claude-opus-5"

SYNTHESIS_SYSTEM_PROMPT = """\
너는 여러 타임프레임(일봉/30분봉/3분봉/화면)의 차트 분석 결과를 종합해서
최종 판단을 내리는 총괄 분석가다. 각 타임프레임 에이전트가 이미 개별 판단을
내렸으니, 그 결과들 사이의 일치/불일치를 근거로 신뢰도를 평가하라.

예를 들어 3분봉과 30분봉에서 같은 시점 근처에 거래량 급증 신호가 확인되고
일봉에서도 같은 방향의 추세가 보인다면 신뢰도가 높다고 판단하고, 타임프레임 간
신호가 서로 어긋나거나 일부만 존재한다면 그 점을 명시하라.

한국어로, 다음 구조로 답하라:
1. 종합 결론 (한두 문장)
2. 타임프레임별 일치/불일치 근거
3. 참고 사항 / 주의할 점
"""


def synthesize(
    client: anthropic.Anthropic,
    results: Dict[str, ChartAnalysisResult],
    ticker: Optional[str] = None,
) -> str:
    """여러 타임프레임의 분석 결과를 하나의 최종 리포트 텍스트로 종합"""
    ticker_note = f"분석 대상 종목: {ticker}\n\n" if ticker else ""

    payload = {
        key: result.model_dump() for key, result in results.items()
    }

    user_text = (
        f"{ticker_note}아래는 각 타임프레임 에이전트의 분석 결과(JSON)다.\n\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n\n"
        "위 내용을 종합해서 최종 리포트를 작성하라."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=SYNTHESIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_text}],
    )

    return next(b.text for b in response.content if b.type == "text")
