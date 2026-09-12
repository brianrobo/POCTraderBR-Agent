from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .vision import encode_image, encode_image_bytes


@dataclass
class Candle:
    """OHLCV 캔들 한 개. 증권사 REST API 응답을 이 형태로 매핑해서 사용한다."""

    time: str
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class ChartInput:
    """분석 대상 하나(이미지 또는 캔들 데이터)를 감싸는 공통 입력.

    지금은 화면 캡처 이미지만 쓰지만, 나중에 증권사 REST API로 봉 데이터를
    직접 받아오게 되면 ChartInput.from_candles(...)로 같은 analyze_chart()에
    바로 꽂을 수 있다. agents.py는 kind가 무엇이든 content_blocks()/describe()만
    호출하므로 수정할 필요가 없다.
    """

    kind: Literal["image", "candles"]
    image_path: Optional[str] = None
    image_bytes: Optional[bytes] = None
    image_media_type: Optional[str] = None
    candles: Optional[List[Candle]] = None

    @classmethod
    def from_image(cls, image_path: str) -> "ChartInput":
        return cls(kind="image", image_path=image_path)

    @classmethod
    def from_image_bytes(cls, data: bytes, media_type: str) -> "ChartInput":
        """메모리 상의 이미지 바이트로부터 생성 (예: 웹 업로드/클립보드 붙여넣기)."""
        return cls(kind="image", image_bytes=data, image_media_type=media_type)

    @classmethod
    def from_candles(cls, candles: List[Dict[str, Any]]) -> "ChartInput":
        """증권사 API 등에서 받은 캔들 목록(dict)으로부터 생성.

        각 dict는 time/open/high/low/close/volume 키를 가져야 한다.
        """
        if not candles:
            raise ValueError("candles가 비어 있습니다.")
        return cls(kind="candles", candles=[Candle(**c) for c in candles])

    def content_blocks(self) -> List[Dict[str, Any]]:
        """Claude Messages API의 user content 블록 목록으로 변환."""
        if self.kind == "image":
            if self.image_bytes is not None:
                return [encode_image_bytes(self.image_bytes, self.image_media_type or "image/png")]
            if not self.image_path:
                raise ValueError("image_path가 없습니다.")
            return [encode_image(self.image_path)]

        if not self.candles:
            raise ValueError("candles가 없습니다.")
        rows = "\n".join(
            f"{c.time},{c.open},{c.high},{c.low},{c.close},{c.volume}"
            for c in self.candles
        )
        table = "time,open,high,low,close,volume\n" + rows
        return [
            {
                "type": "text",
                "text": (
                    "[캔들 데이터 - 증권사 API로 수신한 정확한 수치, 이미지 아님]\n"
                    f"{table}"
                ),
            }
        ]

    def describe(self, label: str) -> str:
        """프롬프트에 넣을 안내 문구."""
        if self.kind == "image":
            return f"첨부된 {label} 차트 이미지"
        return f"위에 제공된 {label} 캔들 데이터"
