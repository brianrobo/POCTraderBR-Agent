from typing import List, Literal

from pydantic import BaseModel, Field


class CriterionSignal(BaseModel):
    """하나의 판단 기준에 대한 분석 결과"""

    criterion: str = Field(description="어떤 기준에 대한 판단인지 (기준 원문 요약)")
    found: bool = Field(description="해당 기준에 부합하는 구간을 발견했는지 여부")
    evidence: str = Field(description="차트 상 근거 (위치, 캔들 개수 기준 최근/중간/과거 등)")
    price_move_desc: str = Field(description="가격 변화에 대한 설명 (대략적인 상승률 등)")
    volume_move_desc: str = Field(description="거래량 변화에 대한 설명 (평소 대비 증가 배수 추정 등)")
    confidence: Literal["high", "medium", "low"] = Field(description="판단 확신도")


class ChartAnalysisResult(BaseModel):
    """하나의 타임프레임(에이전트)에 대한 종합 분석 결과"""

    timeframe: str = Field(description="분석한 타임프레임 키 (daily/min30/min3/screen)")
    signals: List[CriterionSignal] = Field(description="기준별 판단 결과 목록")
    overall_found: bool = Field(description="하나 이상의 기준에서 신호가 발견되었는지 여부")
    overall_comment: str = Field(description="이 타임프레임에 대한 종합 코멘트 (한국어)")
