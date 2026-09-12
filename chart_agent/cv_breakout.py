"""OpenCV로 '물량 털기'(가격 상승 + 거래량 폭증) 캔들 구간을 직접 검출.

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
from PIL import Image, ImageDraw, ImageFont

KOREAN_FONT_PATH = "C:/Windows/Fonts/malgun.ttf"
_FONT_CACHE = {}


def _get_font(size: int):
    """cv2.putText는 한글을 못 그려서(깨짐) Pillow로 범례 텍스트를 렌더링한다."""
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype(KOREAN_FONT_PATH, size)
        except OSError:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


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


def _refine_wicks(
    candles: List[Candle],
    img: np.ndarray,
    band_top: int,
    band_bottom: int,
    threshold: float = 18.0,
) -> None:
    """윗꼬리/아랫꼬리 복원 — 색상이 아니라 '밝기'로 찾는다.

    처음엔 색상 마스크(오프닝 안 한 원본)로 꼬리를 복원해봤는데, 실제로는
    꼬리 선이 안티에일리어싱 때문에 몸통 채우기 색보다 훨씬 옅게(회색
    배경에 가깝게) 렌더링돼서 엄격한 색상 임계값을 통과하지 못하는 경우가
    많았다. 대신 몸통 색과 무관하게, 배경보다 확실히 어두운 픽셀이 있는
    동안 몸통 위/아래로 계속 늘려가는 방식으로 찾는다 — 이 좁은 x 범위
    (캔들 폭)에서만 스캔하므로 다른 캔들/이평선이 섞여 들어올 위험이 작고,
    배경 밝기로 돌아오면 바로 멈추므로 근처를 지나는 이평선까지 이어붙는
    것도 막는다.
    """
    gray = img.mean(axis=2)
    bg_level = float(np.median(gray[band_top : band_bottom + 1, :]))

    for c in candles:
        col = gray[:, c.x_start : c.x_end + 1]
        # 평균이 아니라 최솟값 — 꼬리는 캔들 폭 안에서 1~2px만 차지해서,
        # 나머지 배경 픽셀과 평균 내면 신호가 묻힌다.

        y = c.body_top
        while y - 1 >= band_top and bg_level - col[y - 1].min() >= threshold:
            y -= 1
        c.range_top = min(c.range_top, y)

        y = c.body_bottom
        while y + 1 <= band_bottom and bg_level - col[y + 1].min() >= threshold:
            y += 1
        c.range_bottom = max(c.range_bottom, y)


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
    _refine_wicks(candles, img, candle_top, candle_bottom)

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


def extend_cluster_to_breakdown(
    candles: List[Candle], volumes: List[float], start: int, end: int, pullback_frac: float = 0.03
) -> int:
    """클러스터 끝을, 개별 캔들 신호 기준이 아니라 '가격이 이 구간의 폭발
    캔들 시가 아래로 확실히 무너지기 전'까지로 늘린다.

    예: 제일 크게 터진 캔들 옆에 그보다 작지만(예: 절반 정도) 여전히 큰
    거래량 캔들이 붙어 있으면, 그 캔들 개별로는 신호 임계값을 못 넘겨도
    같은 물량 털기 구간으로 봐야 한다 — 가격이 아직 그 구간을 벗어나지
    않았기 때문.

    기준선은 (클러스터 전체가 아니라) 거래량이 가장 큰 캔들 하나의 시가로
    고정한다 — 클러스터에 섞여 들어온, 훨씬 이전의 작은 신호까지 포함해서
    평균/최저를 잡으면 기준선이 너무 낮아져 구간이 끝없이 늘어난다.
    """
    pane_top = min(c.range_top for c in candles)
    pane_bottom = max(c.range_bottom for c in candles)
    min_move = max((pane_bottom - pane_top) * pullback_frac, 5)

    peak_idx = find_peak_index(candles, volumes, start, end)
    zone_low = candles[peak_idx].body_bottom
    e = max(end, peak_idx)
    n = len(candles)
    while e + 1 < n:
        nxt = candles[e + 1]
        if nxt.body_bottom > zone_low + min_move:
            break  # 가격이 폭발 캔들 시가 아래로 확실히 무너짐 — 여기서 구간 종료
        e += 1
    return e


def _draw_ellipse_around(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad: int = 10, thickness: int = 2,
    color: Tuple[int, int, int] = (0, 230, 255),
) -> None:
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    ax, ay = (x1 - x0) // 2 + pad, (y1 - y0) // 2 + pad
    cv2.ellipse(img, (cx, cy), (max(ax, 8), max(ay, 8)), 0, 0, 360, color, thickness)


def find_peak_index(candles: List[Candle], volumes: List[float], start: int, end: int) -> int:
    """구간(start~end) 안에서 거래량이 가장 큰 캔들 — '실제로 물량이 터진 가격'."""
    return max(range(start, end + 1), key=lambda i: volumes[i])


@dataclass
class RetestSignal:
    """물량이 터진 가격대를 이후에 다시 찍고(재상승) 내려간(하락) 정황.

    개미들에게 '본전' 근처까지 가격을 다시 올려줘서 물량을 던지게 만들고
    (매집), 그 직후 가격이 다시 빠지는 패턴 — 첫 거래량 폭증만으로는 알 수
    없고, 그 뒤 캔들들을 봐야 확인된다.
    """

    peak_index: int
    retest_index: int
    decline_index: int


def detect_retest_and_reject(
    candles: List[Candle],
    volumes: List[float],
    clusters: List[Tuple[int, int]],
    pullback_frac: float = 0.03,
    tolerance_frac: float = 0.02,
    max_lookahead: int = 90,
    decline_window: int = 15,
) -> List[RetestSignal]:
    """각 물량 폭증 구간(cluster)의 최대 거래량 캔들(peak) 가격대를,
    이후 캔들이 (한 번 눌렸다가) 다시 찍고 나서 내려가는지 확인한다.

    픽셀 y좌표만 있고 실제 가격 축 눈금은 모르므로, 캔들 창 전체 높이에 대한
    비율로 '의미 있는 되돌림 폭(pullback_frac)'과 '같은 가격대로 볼 허용
    오차(tolerance_frac)'를 정한다.
    """
    if not candles:
        return []

    pane_top = min(c.range_top for c in candles)
    pane_bottom = max(c.range_bottom for c in candles)
    pane_height = max(pane_bottom - pane_top, 1)
    min_move = max(pane_height * pullback_frac, 5)
    tol = max(pane_height * tolerance_frac, 4)

    results = []
    for raw_start, raw_end in clusters:
        start, end = raw_start, extend_cluster_to_breakdown(candles, volumes, raw_start, raw_end, pullback_frac)
        peak_idx = find_peak_index(candles, volumes, start, end)
        # 기준 가격은 거래량 최대 캔들의 몸통이 아니라, 그 구간에서 실제로
        # 찍은 최고가('가격 상단' = 전고 돌파 지점) — 변동성이 거기서 나온다는
        # 사용자 설명 반영.
        zone_level = min(candles[i].range_top for i in range(start, end + 1))
        zone_top = zone_bottom = zone_level

        pulled_back = False
        retest_idx = None
        limit = min(end + 1 + max_lookahead, len(candles))
        for j in range(end + 1, limit):
            c = candles[j]
            if not pulled_back:
                if c.body_bottom > zone_bottom + min_move:  # 가격이 그 가격대 아래로 확실히 빠짐
                    pulled_back = True
                continue
            if zone_top - tol <= c.range_top <= zone_bottom + tol:  # 그 가격대를 다시 찍음
                retest_idx = j
                break

        if retest_idx is None:
            continue

        retest_low = candles[retest_idx].range_bottom
        decline_idx = None
        limit2 = min(retest_idx + 1 + decline_window, len(candles))
        for k in range(retest_idx + 1, limit2):
            if candles[k].body_top > retest_low - min_move * 0.3:  # 재접근 캔들의 저가 밑으로 이탈
                decline_idx = k
                break

        if decline_idx is not None:
            results.append(RetestSignal(peak_index=peak_idx, retest_index=retest_idx, decline_index=decline_idx))

    return results


def find_significant_volume_points(
    candles: List[Candle],
    volumes: List[float],
    start: int,
    end: int,
    min_frac_of_peak: float = 0.35,
    merge_gap: int = 1,
) -> List[int]:
    """구간(extend_cluster_to_breakdown로 늘린 범위) 안에서 '따로 표시할 만큼
    큰' 거래량 지점을 전부 찾는다 — 제일 큰 것 하나만이 아니라, 그 옆에 붙은
    절반 정도 크기의 캔들도 별도 매집 지점으로 봐야 한다는 사용자 설명 반영.

    양봉/음봉을 가리지 않는다 — 같은 구간 안에서 거래대금이 크게 터진
    음봉도 물량이 넘어간 지점일 수 있다 (오히려 음봉에서 크게 터지는 게
    투매/물량 소화 구간일 수 있음).

    같은 급등이 이틀에 걸쳐 찍힌 것처럼 바로 붙어있는 지점(merge_gap 이내)은
    거래량이 더 큰 쪽 하나로 합친다.
    """
    peak_vol = max(volumes[i] for i in range(start, end + 1))
    if peak_vol <= 0:
        return []
    threshold = peak_vol * min_frac_of_peak
    candidates = sorted(i for i in range(start, end + 1) if volumes[i] >= threshold)

    merged: List[int] = []
    for i in candidates:
        if merged and i - merged[-1] <= merge_gap:
            if volumes[i] > volumes[merged[-1]]:
                merged[-1] = i
        else:
            merged.append(i)
    return merged


def _add_legend(img: np.ndarray, show_retest: bool, show_accum: bool) -> np.ndarray:
    """이미지 맨 위에 마크 설명을 붙인다 (한글이라 Pillow로 렌더링).

    기존 내용을 가리지 않도록 캔버스 자체를 위로 늘려서 그 여백에 그린다.
    """
    legend_h = 40
    h, w = img.shape[:2]
    canvas = np.full((h + legend_h, w, 3), 255, dtype=np.uint8)
    canvas[legend_h:, :] = img

    pil_img = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(17)
    cy = legend_h // 2

    items = [("circle", (255, 230, 0), ": 물량 털기")]
    if show_retest:
        items.append(("circle", (255, 140, 0), ": 재접근 후 하락"))
    if show_accum:
        items.append(("box", (0, 160, 200), ": 매집 구간"))

    icon_d, gap_after_icon, gap_between = 18, 8, 20
    right_margin = 20
    widths = [icon_d + gap_after_icon + draw.textlength(text, font=font) for _, _, text in items]
    x = w - right_margin - (sum(widths) + gap_between * (len(items) - 1))

    for (shape, color, text), item_w in zip(items, widths):
        if shape == "circle":
            draw.ellipse((x, cy - 9, x + icon_d, cy + 9), outline=color, width=2)
        else:
            draw.rectangle((x, cy - 7, x + icon_d, cy + 7), outline=color, width=2)
        draw.text((x + icon_d + gap_after_icon, cy - 10), text, font=font, fill=(30, 30, 30))
        x += item_w + gap_between

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


@dataclass
class AccumulationZone:
    """물량 폭증(매집 완료 후 상승) 직전의 좁은 횡보 구간 — 항목 3(세력의
    매집 구간) 1차 구현. 폭발이 시작되는 캔들 바로 앞에서부터 거꾸로,
    가격이 좁은 밴드 안에 머무는 동안 계속 넓혀가다가 밴드를 벗어나면
    멈춘다. 저가 구간뿐 아니라 1차 상승 후 고점 부근의 눌림목도 같은
    방식으로 잡힌다 — 가격대가 아니라 '폭이 좁은지'만 본다.
    """

    start_index: int
    end_index: int  # inclusive, burst_start - 1


def find_accumulation_zone(
    candles: List[Candle],
    volumes: List[float],
    burst_start: int,
    boundary: int,
    peak_volume: float,
    quiet_frac: float = 0.2,
    min_length: int = 5,
    max_length: int = 40,
    single_candle_range_frac: float = 0.2,
) -> Optional[AccumulationZone]:
    """'거래량이 아직 낮게 유지되는지'로 매집 구간을 잡는다 (가격 자체의
    폭이 아니라).

    사용자 피드백: 매집 구간은 가격이 완만하게 우상향해도(횡보가 아니어도)
    거래량만 낮게 유지되면 하나의 매집 구간으로 본다. 처음엔 누적 가격 폭
    기준을 썼는데, 실제 사례에서 진짜 돌파 직전 완만한 상승 구간이 가격
    폭 기준에 걸려 통째로 빠지는 문제가 있어서 거래량 기준으로 바꿨다.

    다만 거래량만 보면, 거래량은 적은데 캔들 하나가 유난히 큰 폭으로
    움직인 경우(예: 직전 급등의 되돌림으로 하루 만에 크게 빠지는 캔들)까지
    "조용하다"고 잘못 포함시켜서 매집 구간 박스가 그 캔들 높이만큼 세로로
    커져버리는 문제가 있었다 — 그래서 캔들 하나의 고가-저가 폭이 전체
    차트 높이 대비 너무 크면(single_candle_range_frac) 거래량과 상관없이
    "아직 조용하지 않다"고 보고 거기서 구간을 끊는다.
    """
    if burst_start - 1 < boundary:
        return None

    pane_top = min(c.range_top for c in candles)
    pane_bottom = max(c.range_bottom for c in candles)
    max_single_range = (pane_bottom - pane_top) * single_candle_range_frac
    quiet_threshold = peak_volume * quiet_frac

    def _is_quiet(idx: int) -> bool:
        c = candles[idx]
        return volumes[idx] <= quiet_threshold and (c.range_bottom - c.range_top) <= max_single_range

    earliest = max(boundary, burst_start - max_length)
    i = burst_start - 1
    start = i
    while i - 1 >= earliest and _is_quiet(i - 1):
        i -= 1
        start = i

    if burst_start - start < min_length:
        return None
    return AccumulationZone(start_index=start, end_index=burst_start - 1)


def _draw_zone_box(
    img: np.ndarray, x0: int, y0: int, x1: int, y1: int, pad_x: int = -3, pad_y: int = 6
) -> None:
    # 가로 여백은 세로보다 훨씬 작게 — 캔들 사이 간격이 보통 4~5px밖에 안 돼서,
    # 넉넉한 여백을 주면 바로 다음(제외해야 할) 장대양봉을 살짝 덮어버린다.
    cv2.rectangle(img, (x0 - pad_x, y0 - pad_y), (x1 + pad_x, y1 + pad_y), (200, 160, 0), 2, cv2.LINE_AA)


def _render_annotations(
    img: np.ndarray,
    candles: List[Candle],
    volumes: List[float],
    signals: List[BreakoutSignal],
    baseline_y: int,
    cluster_gap: int,
) -> Tuple[np.ndarray, bool, bool]:
    debug = img.copy()
    raw_clusters = cluster_signal_indices(signals, gap=cluster_gap)
    extended_clusters = [
        (s, extend_cluster_to_breakdown(candles, volumes, s, e)) for s, e in raw_clusters
    ]
    retests = detect_retest_and_reject(candles, volumes, raw_clusters)
    retest_by_peak = {r.peak_index: r for r in retests}

    has_accum = False
    prev_end = 0
    for start_idx, end_idx in extended_clusters:
        # 매집 구간의 끝 경계는 "거래대금이 터지며 가격이 오른 장대양봉
        # 나오기 바로 전"이어야 한다 (사용자 확인) — 장대양봉 자체(그리고
        # 그 직후 붙어있는 또 다른 장대양봉까지)는 매집이 아니다.
        #
        # find_significant_volume_points()는 "동그라미 표시" 용도로 바로
        # 붙어있는 지점을 하나로 합치는데(예: 47·48번이 둘 다 큰 장대양봉인데
        # 47은 48에 합쳐져 사라짐), 그 합쳐진 결과를 여기 쓰면 47번이
        # 매집 구간에 잘못 포함된다. 그래서 병합 없이 직접 "그 구간 최대
        # 거래량의 35% 이상인 양봉"을 전부 찾아 그중 가장 이른 캔들을
        # 기준으로 삼는다. (양봉이 하나도 없으면 클러스터의 첫 신호
        # 캔들(start_idx)로 폴백 — detect_breakouts()는 항상 양봉만
        # 신호로 잡으므로 안전하다.)
        peak_idx = find_peak_index(candles, volumes, start_idx, end_idx)
        peak_volume = volumes[peak_idx]
        threshold = peak_volume * 0.35
        up_candidates = [
            i for i in range(start_idx, end_idx + 1)
            if candles[i].color == "up" and volumes[i] >= threshold
        ]
        ramp_start = min(up_candidates) if up_candidates else start_idx
        zone = find_accumulation_zone(candles, volumes, ramp_start, prev_end, peak_volume)
        if zone:
            has_accum = True
            group = candles[zone.start_index : zone.end_index + 1]
            _draw_zone_box(
                debug,
                min(c.x_start for c in group),
                min(c.range_top for c in group),
                max(c.x_end for c in group),
                max(c.range_bottom for c in group),
            )
        prev_end = end_idx + 1

    for start_idx, end_idx in extended_clusters:
        peak_idx = find_peak_index(candles, volumes, start_idx, end_idx)

        for i in find_significant_volume_points(candles, volumes, start_idx, end_idx):
            c = candles[i]
            # 윗꼬리(고가)까지 포함 — 몸통(종가)보다 더 위까지 찔렀다가 밀린
            # 부분이 실제로 물량을 던지게 만든 상단이라는 설명 반영
            _draw_ellipse_around(debug, c.x_start, c.range_top, c.x_end, c.body_bottom, pad=8)
            if volumes[i] > 0:
                vol_top = baseline_y - volumes[i]
                _draw_ellipse_around(debug, c.x_start, vol_top, c.x_end, baseline_y, pad=6)

        retest = retest_by_peak.get(peak_idx)
        if retest:
            r = candles[retest.retest_index]
            zone_y = min(candles[i].range_top for i in range(start_idx, end_idx + 1))
            cv2.line(debug, (candles[end_idx].x_end, zone_y), (r.x_start, zone_y), (0, 165, 255), 1, cv2.LINE_AA)
            _draw_ellipse_around(
                debug, r.x_start, r.range_top, r.x_end, r.range_bottom, pad=8, color=(0, 140, 255)
            )

    return debug, bool(retests), has_accum


def draw_debug(
    img: np.ndarray,
    candles: List[Candle],
    volumes: List[int],
    signals: List[BreakoutSignal],
    baseline_y: int,
    out_path: str,
    cluster_gap: int = 12,
) -> None:
    """검출된 '물량 털기'(가격 상승+거래량 폭증) 의심 구간을 원본 이미지 위에 동그라미로 표시.

    구간별 최대 거래량 캔들(실제 물량이 터진 가격)에 작은 동그라미를 치고,
    그 가격대를 이후에 다시 찍고 내려가는 재접근-하락 정황이 있으면 주황색
    동그라미와 연결선으로 함께 표시한다. 맨 위에는 마크 범례를 붙인다.
    """
    debug, has_retest, has_accum = _render_annotations(img, candles, volumes, signals, baseline_y, cluster_gap)
    debug = _add_legend(debug, has_retest, has_accum)
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
    debug, has_retest, has_accum = _render_annotations(img, candles, volumes, signals, baseline_y, cluster_gap)
    debug = _add_legend(debug, has_retest, has_accum)
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
