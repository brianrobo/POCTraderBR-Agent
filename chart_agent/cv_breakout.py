"""OpenCV로 '가격 상승 + 거래량 폭증' 캔들 구간을 직접 검출.

키움 종합차트(상단 캔들+이동평균선, 하단 거래대금 막대) 캡처 화면을
전제로 색상 기반 검출을 한다. LLM 호출 없이 결정론적으로 동작한다.

색상(이 사용자의 캡처 화면에서 실측):
  - 양봉(빨강): RGB 약 (216-230, 20-30, 10-25)
  - 음봉(파랑): RGB 약 (10-40, 80-95, 165-195)
  - 거래대금 막대(보라): RGB 약 (95-115, 90-105, 130-150)
  - 이동평균선(분홍5/파랑10/주황20/초록60/검정120)은 1~2px로 얇아서
    가로 방향 모폴로지 오프닝으로 제거하고 두꺼운 캔들 몸통/막대만 남긴다.
"""

import statistics
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


def _median(values: List[float]) -> float:
    return statistics.median(values) if values else 0.0

# --- 색상 범위 (BGR, cv2 기본 채널 순서) ---
RED_LOWER = np.array([0, 0, 140])
RED_UPPER = np.array([70, 70, 255])

BLUE_LOWER = np.array([140, 50, 0])
BLUE_UPPER = np.array([220, 130, 70])

PURPLE_LOWER = np.array([120, 80, 85])
PURPLE_UPPER = np.array([170, 120, 130])

OPEN_KERNEL = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))  # 얇은 이평선 제거용
MIN_CANDLE_WIDTH = 3  # 이보다 얇은 색상 그룹은 텍스트/노이즈로 간주해 제외


@dataclass
class Candle:
    index: int
    x_start: int
    x_end: int
    color: str  # "up" | "down"
    range_top: int
    range_bottom: int
    body_top: int
    body_bottom: int

    @property
    def range_height(self) -> int:
        return self.range_bottom - self.range_top + 1

    @property
    def body_height(self) -> int:
        return max(self.body_bottom - self.body_top + 1, 1)


@dataclass
class BreakoutSignal:
    candle_index: int
    total_candles: int
    volume_height: float
    volume_baseline: float
    volume_ratio: float
    body_height: float
    body_baseline: float
    body_ratio: float

    @property
    def position_desc(self) -> str:
        frac = self.candle_index / max(self.total_candles - 1, 1)
        if frac > 0.8:
            return "최근"
        if frac > 0.4:
            return "중간"
        return "과거"


def _mask(img_bgr: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    mask = cv2.inRange(img_bgr, lower, upper)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, OPEN_KERNEL)


def _mask_bounds(mask: np.ndarray) -> Optional[Tuple[int, int]]:
    ys, _ = np.where(mask > 0)
    if len(ys) == 0:
        return None
    return int(ys.min()), int(ys.max())


def _find_dense_band(mask: np.ndarray, merge_gap: int = 8) -> Tuple[int, int]:
    """캔들 창/거래량 창처럼 픽셀이 밀집된 y축 구간을 찾는다.

    툴바 아이콘, 상단 시세 배지, 우측 축 숫자 등 화면 다른 곳의 같은 색
    잔여 픽셀 때문에 mask 전체의 min/max만 쓰면 창 경계가 어긋난다. 픽셀이
    있는 row들을 구간으로 묶되(merge_gap 이하 간격은 이어붙임), 그중
    세로로 가장 길게 이어지는 구간을 실제 차트/거래량 창으로 판단한다
    (툴바 아이콘 등은 짧고 굵게, 차트는 얇고 길게 분포하기 때문).
    """
    row_has_pixel = (mask > 0).any(axis=1)
    runs = []
    y = 0
    h = len(row_has_pixel)
    while y < h:
        if row_has_pixel[y]:
            start = y
            while y < h and row_has_pixel[y]:
                y += 1
            runs.append([start, y - 1])
        else:
            y += 1

    if not runs:
        raise ValueError("색상 마스크에서 유효한 구간을 찾지 못했습니다.")

    merged = [runs[0]]
    for s, e in runs[1:]:
        if s - merged[-1][1] <= merge_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])

    best = max(merged, key=lambda r: r[1] - r[0])
    return best[0], best[1]


def segment_candles(red_mask: np.ndarray, blue_mask: np.ndarray) -> List[Candle]:
    candle_mask = (red_mask > 0) | (blue_mask > 0)
    col_has_pixel = candle_mask.any(axis=0)

    groups: List[Tuple[int, int]] = []
    x = 0
    w = len(col_has_pixel)
    while x < w:
        if col_has_pixel[x]:
            start = x
            while x < w and col_has_pixel[x]:
                x += 1
            groups.append((start, x - 1))
        else:
            x += 1

    candles = []
    for x0, x1 in groups:
        if x1 - x0 + 1 < MIN_CANDLE_WIDTH:
            continue  # 이평선/텍스트 잔여 노이즈 제거
        seg_red = red_mask[:, x0 : x1 + 1] > 0
        seg_blue = blue_mask[:, x0 : x1 + 1] > 0
        red_count = int(seg_red.sum())
        blue_count = int(seg_blue.sum())
        color = "up" if red_count >= blue_count else "down"
        seg = seg_red if color == "up" else seg_blue

        rows_any = seg.any(axis=1)
        ys = np.where(rows_any)[0]
        if len(ys) == 0:
            continue
        range_top, range_bottom = int(ys.min()), int(ys.max())

        width = x1 - x0 + 1
        row_widths = seg.sum(axis=1)
        body_rows = np.where(row_widths >= max(width * 0.6, 1))[0]
        if len(body_rows):
            body_top, body_bottom = int(body_rows.min()), int(body_rows.max())
        else:
            body_top, body_bottom = range_top, range_bottom

        candles.append(
            Candle(
                index=len(candles),
                x_start=x0,
                x_end=x1,
                color=color,
                range_top=range_top,
                range_bottom=range_bottom,
                body_top=body_top,
                body_bottom=body_bottom,
            )
        )
    return candles


def volume_heights_for_candles(
    purple_mask: np.ndarray, candles: List[Candle], baseline_y: int
) -> List[int]:
    heights = []
    for c in candles:
        seg = purple_mask[:, c.x_start : c.x_end + 1] > 0
        ys = np.where(seg.any(axis=1))[0]
        if len(ys) == 0:
            heights.append(0)
        else:
            heights.append(int(baseline_y - ys.min()))
    return heights


def detect_breakouts(
    colors: List[str],
    body_heights: List[float],
    volumes: List[float],
    lookback: int = 20,
    volume_ratio_threshold: float = 3.0,
    min_volume: float = 0,
    min_body_ratio: float = 1.5,
) -> List[BreakoutSignal]:
    """가격 상승(양봉) + 거래량이 직전 lookback개 대비 크게 튄 지점을 찾는다.

    픽셀 높이든(이미지) 실제 원화 거래대금이든(캔들 데이터) 같은 로직으로
    쓸 수 있도록 단위에 무관한 숫자 리스트를 입력으로 받는다.

    직전 구간의 '평균'이 아니라 '중앙값'을 기준으로 삼는다 — 겹쳐 있는
    다른 스파이크 하나 때문에 평균이 끌어올려져서 바로 다음 스파이크를
    놓치는 걸 막기 위함이다. 조용한 구간(거래량이 계속 0에 가까움) 뒤에
    갑자기 터지는, 가장 전형적인 경우도 잡을 수 있도록 중앙값에 최소
    바닥값(1.0)을 둔다.
    """
    signals = []
    n = len(colors)

    for i in range(n):
        if colors[i] != "up":
            continue
        if volumes[i] < min_volume:
            continue

        window_start = max(0, i - lookback)
        prior_vols = volumes[window_start:i]
        if not prior_vols:
            continue
        vol_baseline = max(_median(prior_vols), 1.0)
        vol_ratio = volumes[i] / vol_baseline
        if vol_ratio < volume_ratio_threshold:
            continue

        prior_bodies = [body_heights[j] for j in range(window_start, i) if body_heights[j] > 0]
        body_baseline = sum(prior_bodies) / len(prior_bodies) if prior_bodies else 1.0
        body_ratio = body_heights[i] / body_baseline if body_baseline else 0.0
        if body_ratio < min_body_ratio:
            continue  # 거래량만 튀고 가격 상승폭 자체는 평소 수준 이하인 경우 제외

        signals.append(
            BreakoutSignal(
                candle_index=i,
                total_candles=n,
                volume_height=volumes[i],
                volume_baseline=vol_baseline,
                volume_ratio=vol_ratio,
                body_height=body_heights[i],
                body_baseline=body_baseline,
                body_ratio=body_ratio,
            )
        )
    return signals


def analyze_image_array(
    img: np.ndarray, lookback: int = 20, volume_ratio_threshold: float = 3.0
) -> Tuple[List[Candle], List[int], List[BreakoutSignal], np.ndarray, int]:
    red_mask_full = _mask(img, RED_LOWER, RED_UPPER)
    blue_mask_full = _mask(img, BLUE_LOWER, BLUE_UPPER)
    purple_mask_full = _mask(img, PURPLE_LOWER, PURPLE_UPPER)

    vol_top, vol_bottom = _find_dense_band(purple_mask_full)
    candle_top, candle_bottom = _find_dense_band(red_mask_full | blue_mask_full)

    # 창 밖(툴바/축 라벨 등) 잔여 픽셀이 컬럼 분리에 섞이지 않도록 해당 창 범위로 마스크를 제한
    red_mask = np.zeros_like(red_mask_full)
    blue_mask = np.zeros_like(blue_mask_full)
    purple_mask = np.zeros_like(purple_mask_full)
    red_mask[candle_top : candle_bottom + 1] = red_mask_full[candle_top : candle_bottom + 1]
    blue_mask[candle_top : candle_bottom + 1] = blue_mask_full[candle_top : candle_bottom + 1]
    purple_mask[vol_top : vol_bottom + 1] = purple_mask_full[vol_top : vol_bottom + 1]

    baseline_y = vol_bottom

    candles = segment_candles(red_mask, blue_mask)
    volumes = volume_heights_for_candles(purple_mask, candles, baseline_y)
    signals = detect_breakouts(
        [c.color for c in candles],
        [c.body_height for c in candles],
        volumes,
        lookback,
        volume_ratio_threshold,
        min_volume=8,  # 이 정도보다 얇은 막대는 픽셀 노이즈로 간주
    )

    return candles, volumes, signals, img, baseline_y


def analyze_image(
    image_path: str, lookback: int = 20, volume_ratio_threshold: float = 3.0
) -> Tuple[List[Candle], List[int], List[BreakoutSignal], np.ndarray, int]:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")
    return analyze_image_array(img, lookback, volume_ratio_threshold)


def analyze_image_bytes(
    data: bytes, lookback: int = 20, volume_ratio_threshold: float = 3.0
) -> Tuple[List[Candle], List[int], List[BreakoutSignal], np.ndarray, int]:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("이미지 바이트를 디코딩하지 못했습니다.")
    return analyze_image_array(img, lookback, volume_ratio_threshold)


def detect_breakouts_from_ohlcv(
    candles,  # List[chart_agent.chart_input.Candle]
    lookback: int = 20,
    volume_ratio_threshold: float = 3.0,
    min_body_ratio: float = 1.5,
) -> List[BreakoutSignal]:
    """실제 OHLCV 캔들 데이터(증권사 API 등)로 같은 기준을 계산.

    픽셀 추정치 대신 실제 가격/거래량 숫자를 쓰므로 이미지 방식보다 정확하다.
    """
    colors = ["up" if c.close >= c.open else "down" for c in candles]
    body_heights = [abs(c.close - c.open) for c in candles]
    volumes = [c.volume for c in candles]
    return detect_breakouts(
        colors, body_heights, volumes, lookback, volume_ratio_threshold, min_volume=0, min_body_ratio=min_body_ratio
    )


def cluster_signal_indices(signals: List[BreakoutSignal], gap: int = 12) -> List[Tuple[int, int]]:
    """근처 캔들에서 연달아 나온 신호를 하나의 구간(클러스터)으로 묶는다.

    사용자가 차트를 눈으로 볼 때 "이 부근 전체가 한 번의 거래량 폭증"이라고
    묶어서 인식하는 것과 맞추기 위함 — 캔들 하나하나에 박스를 그리는 대신
    구간 전체에 동그라미 하나를 그린다.
    """
    idxs = sorted({s.candle_index for s in signals})
    if not idxs:
        return []
    clusters = [[idxs[0], idxs[0]]]
    for i in idxs[1:]:
        if i - clusters[-1][1] <= gap:
            clusters[-1][1] = i
        else:
            clusters.append([i, i])
    return [(s, e) for s, e in clusters]


def _draw_ellipse_around(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int = 10, thickness: int = 2
) -> None:
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax, ay = (x1 - x0) // 2 + pad, (y1 - y0) // 2 + pad
    cv2.ellipse(img, (cx, cy), (max(ax, 8), max(ay, 8)), 0, 0, 360, (0, 230, 255), thickness)


def draw_debug(
    img: np.ndarray,
    candles: List[Candle],
    volumes: List[int],
    signals: List[BreakoutSignal],
    baseline_y: int,
    out_path: str,
    cluster_gap: int = 12,
) -> None:
    """검출된 '가격 상승+거래량 폭증' 구간을 원본 이미지 위에 동그라미로 표시.

    사람이 차트를 보고 손으로 동그라미 치는 것과 같은 방식 — 캔들 쪽 동그라미
    하나 + 그 아래 거래량 막대 쪽 동그라미 하나를 구간별로 그린다.
    """
    debug = img.copy()

    for start_idx, end_idx in cluster_signal_indices(signals, gap=cluster_gap):
        group = candles[start_idx : end_idx + 1]
        x0 = min(c.x_start for c in group)
        x1 = max(c.x_end for c in group)

        price_y0 = min(c.range_top for c in group)
        price_y1 = max(c.range_bottom for c in group)
        _draw_ellipse_around(debug, x0, price_y0, x1, price_y1, pad=14)

        vol_ys = [baseline_y - volumes[i] for i in range(start_idx, end_idx + 1) if volumes[i] > 0]
        if vol_ys:
            _draw_ellipse_around(debug, x0, min(vol_ys), x1, baseline_y, pad=10)

    cv2.imwrite(out_path, debug)


def draw_debug_bytes(
    img: np.ndarray,
    candles: List[Candle],
    volumes: List[int],
    signals: List[BreakoutSignal],
    baseline_y: int,
    cluster_gap: int = 12,
) -> bytes:
    """draw_debug()와 동일하지만 PNG 바이트로 반환 (웹 응답에 바로 embed할 때 사용)"""
    debug = img.copy()
    for start_idx, end_idx in cluster_signal_indices(signals, gap=cluster_gap):
        group = candles[start_idx : end_idx + 1]
        x0 = min(c.x_start for c in group)
        x1 = max(c.x_end for c in group)
        price_y0 = min(c.range_top for c in group)
        price_y1 = max(c.range_bottom for c in group)
        _draw_ellipse_around(debug, x0, price_y0, x1, price_y1, pad=14)
        vol_ys = [baseline_y - volumes[i] for i in range(start_idx, end_idx + 1) if volumes[i] > 0]
        if vol_ys:
            _draw_ellipse_around(debug, x0, min(vol_ys), x1, baseline_y, pad=10)
    ok, buf = cv2.imencode(".png", debug)
    if not ok:
        raise ValueError("이미지 인코딩에 실패했습니다.")
    return buf.tobytes()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenCV 기반 가격+거래량 브레이크아웃 검출")
    parser.add_argument("image", help="분석할 차트 캡처 이미지 경로")
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--ratio", type=float, default=3.0)
    parser.add_argument("--debug-out", default=None, help="검출 결과를 그린 디버그 이미지 저장 경로")
    args = parser.parse_args()

    candles, volumes, signals, img, baseline_y = analyze_image(args.image, args.lookback, args.ratio)
    print(f"검출된 캔들 수: {len(candles)}")
    print(f"검출된 신호 수: {len(signals)}")
    for s in signals:
        print(
            f"  - #{s.candle_index} ({s.position_desc}) "
            f"거래량 {s.volume_ratio:.1f}배(기준 {s.volume_baseline:.0f}px) / "
            f"몸통 {s.body_ratio:.1f}배(기준 {s.body_baseline:.0f}px)"
        )

    if args.debug_out:
        draw_debug(img, candles, volumes, signals, baseline_y, args.debug_out)
        print(f"디버그 이미지 저장: {args.debug_out}")
