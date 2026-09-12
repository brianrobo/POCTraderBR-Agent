from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from .models import ChartAnalysisResult

TIMEFRAME_TITLES = {
    "daily": "일봉 (Daily)",
    "min30": "30분봉 (30min)",
    "min3": "3분봉 (3min)",
    "screen": "화면 캡처 (Screen)",
}


def render_report(
    results: Dict[str, ChartAnalysisResult],
    synthesis: str,
    ticker: Optional[str] = None,
) -> str:
    lines = []
    lines.append(f"# 차트 분석 리포트{f' - {ticker}' if ticker else ''}")
    lines.append(f"생성 시각: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append("## 종합 판단")
    lines.append(synthesis.strip())
    lines.append("")
    lines.append("## 타임프레임별 상세 결과")

    for key, result in results.items():
        title = TIMEFRAME_TITLES.get(key, key)
        lines.append(f"### {title}")
        lines.append(f"- 신호 발견 여부: {'O' if result.overall_found else 'X'}")
        lines.append(f"- 종합 코멘트: {result.overall_comment}")
        for i, sig in enumerate(result.signals, 1):
            lines.append(f"  {i}. [{'O' if sig.found else 'X'}] ({sig.confidence}) {sig.criterion}")
            lines.append(f"     - 근거: {sig.evidence}")
            lines.append(f"     - 가격: {sig.price_move_desc}")
            lines.append(f"     - 거래량: {sig.volume_move_desc}")
        lines.append("")

    return "\n".join(lines)


def save_report(
    results: Dict[str, ChartAnalysisResult],
    synthesis: str,
    out_path: str,
    ticker: Optional[str] = None,
) -> None:
    content = render_report(results, synthesis, ticker=ticker)
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
