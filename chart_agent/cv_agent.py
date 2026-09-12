"""OpenCV 검출 결과를 LLM 에이전트와 동일한 ChartAnalysisResult로 변환.

daily/min30/min3는 LLM 대신 이 모듈을 쓴다 (비용 없음, 즉시 결과).
화면 캡처(screen)는 호가창/뉴스 등 이미지 밖 맥락 판단이 필요해서 계속
LLM(agents.py)을 쓴다.
"""

from typing import Any, Dict, List, Optional, Tuple

from .chart_input import ChartInput
from .cv_breakout import (
    BreakoutSignal,
    Candle,
    RetestSignal,
    analyze_image,
    analyze_image_bytes,
    cluster_signal_indices,
    detect_breakouts_from_ohlcv,
    detect_retest_and_reject,
    draw_debug_bytes,
)
from .models import ChartAnalysisResult, CriterionSignal


def _confidence_from_ratio(ratio: float) -> str:
    if ratio >= 10:
        return "high"
    if ratio >= 5:
        return "medium"
    return "low"


def _signal_to_criterion_signal(sig: BreakoutSignal, criterion_text: str) -> CriterionSignal:
    return CriterionSignal(
        criterion=criterion_text,
        found=True,
        evidence=f"{sig.position_desc} 구간 (왼쪽에서 {sig.candle_index + 1}번째 / 전체 {sig.total_candles}개 캔들)",
        price_move_desc=f"가격 변동폭이 직전 구간 평균 대비 약 {sig.body_ratio:.1f}배",
        volume_move_desc=f"거래량이 직전 구간 중앙값 대비 약 {sig.volume_ratio:.1f}배",
        confidence=_confidence_from_ratio(min(sig.body_ratio, sig.volume_ratio)),
    )


def _retest_to_criterion_signal(
    r: RetestSignal, candles: List[Candle], total_candles: int
) -> CriterionSignal:
    peak, retest, decline = candles[r.peak_index], candles[r.retest_index], candles[r.decline_index]
    return CriterionSignal(
        criterion="물량 털기 정황: 거래량이 터진 가격대를 이후에 다시 찍고 하락 (개미 물량 흡수 추정)",
        found=True,
        evidence=(
            f"{peak.index + 1}번째 캔들(거래량 폭증 지점)의 가격대를 "
            f"{retest.index + 1}번째 캔들에서 재접근했다가 "
            f"{decline.index + 1}번째 캔들에서 그 밑으로 하락 (전체 {total_candles}개 중)"
        ),
        price_move_desc="재접근 캔들의 저가 밑으로 하락 마감 — 해당 가격대에서 매물 소화 후 이탈 정황",
        volume_move_desc="-",
        confidence="medium",
    )


def _build_result(
    timeframe_key: str,
    criterion_text: str,
    signals: List[BreakoutSignal],
    candles: Optional[List[Candle]] = None,
    retests: Optional[List[RetestSignal]] = None,
) -> ChartAnalysisResult:
    retests = retests or []
    if signals:
        criterion_signals = [_signal_to_criterion_signal(s, criterion_text) for s in signals]
        if candles is not None:
            criterion_signals += [
                _retest_to_criterion_signal(r, candles, len(candles)) for r in retests
            ]
        positions = sorted({s.position_desc for s in signals}, key=["과거", "중간", "최근"].index)
        comment = (
            f"OpenCV 분석 결과 총 {len(signals)}개 구간에서 물량 털기 의심 신호가 "
            f"발견되었습니다 ({', '.join(positions)})."
        )
        if retests:
            comment += f" 이 중 {len(retests)}개 구간은 이후 가격대 재접근 후 하락까지 확인됐습니다."
    else:
        criterion_signals = [
            CriterionSignal(
                criterion=criterion_text,
                found=False,
                evidence="OpenCV 분석 결과 조건(양봉 + 직전 대비 거래량·가격 변동폭 급증)을 만족하는 물량 털기 의심 구간을 찾지 못했습니다.",
                price_move_desc="-",
                volume_move_desc="-",
                confidence="high",
            )
        ]
        comment = "OpenCV 분석 결과 물량 털기 의심 구간을 찾지 못했습니다."

    return ChartAnalysisResult(
        timeframe=timeframe_key,
        signals=criterion_signals,
        overall_found=len(signals) > 0,
        overall_comment=comment,
    )


def analyze_chart_cv_annotated(
    timeframe_key: str, agent_cfg: Dict[str, Any], chart_input: ChartInput
) -> Tuple[ChartAnalysisResult, Optional[bytes]]:
    """analyze_chart_cv()와 같지만, 이미지 입력이면 검출 구간에 동그라미를
    그린 PNG 바이트도 함께 반환한다 (캔들 데이터 입력이면 None).

    현재는 criteria.yaml의 첫 번째 기준(물량 털기 = 가격 상승+거래량 폭증)
    전용이다. 이 기준과 다른 새 기준을 daily/min30/min3에 추가하면 별도
    로직이 필요하다 — agents.analyze_chart()(LLM)로 되돌리거나 새 검출기를
    추가.
    """
    criterion_text = agent_cfg["criteria"][0]
    annotated_png: Optional[bytes] = None

    if chart_input.kind == "image":
        if chart_input.image_bytes is not None:
            candles, volumes, signals, img, baseline_y = analyze_image_bytes(chart_input.image_bytes)
        elif chart_input.image_path:
            candles, volumes, signals, img, baseline_y = analyze_image(chart_input.image_path)
        else:
            raise ValueError("이미지 데이터가 없습니다.")
        annotated_png = draw_debug_bytes(img, candles, volumes, signals, baseline_y)
        clusters = cluster_signal_indices(signals)
        retests = detect_retest_and_reject(candles, volumes, clusters)
        return _build_result(timeframe_key, criterion_text, signals, candles, retests), annotated_png
    elif chart_input.kind == "candles":
        signals = detect_breakouts_from_ohlcv(chart_input.candles)
        # TODO: 재접근-하락(물량 털기) 판정은 아직 이미지 경로만 지원. 캔들 데이터 경로는
        # 실제 가격 단위로 같은 로직을 다시 짜야 함 (부호/방향이 픽셀 y좌표와 반대).
        return _build_result(timeframe_key, criterion_text, signals), annotated_png
    else:
        raise ValueError(f"알 수 없는 chart_input.kind: {chart_input.kind}")


def analyze_chart_cv(
    timeframe_key: str, agent_cfg: Dict[str, Any], chart_input: ChartInput
) -> ChartAnalysisResult:
    """agents.analyze_chart()와 같은 자리에 쓰는 비-LLM(OpenCV) 버전.

    동그라미 표시 이미지가 필요 없을 때 쓰는 간단한 버전.
    """
    result, _ = analyze_chart_cv_annotated(timeframe_key, agent_cfg, chart_input)
    return result
