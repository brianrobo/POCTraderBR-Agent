from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULT_CRITERIA_PATH = Path(__file__).resolve().parent.parent / "config" / "criteria.yaml"

TIMEFRAME_KEYS = ["daily", "min30", "min3", "screen"]
CV_CAPABLE_KEYS = ["daily", "min30", "min3"]  # OpenCV(비-LLM)로 판단하는 타임프레임
CANDLE_CAPABLE_KEYS = CV_CAPABLE_KEYS  # 화면 캡처는 캔들 데이터로 대체 불가


def load_criteria(path: Path = DEFAULT_CRITERIA_PATH) -> Dict[str, Any]:
    """config/criteria.yaml을 읽어서 {timeframe_key: {label, role, criteria: [...]}} 형태로 반환"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    agents = data.get("agents", {})
    for key in TIMEFRAME_KEYS:
        if key not in agents:
            raise ValueError(f"criteria.yaml에 '{key}' 에이전트 설정이 없습니다: {path}")
        if not agents[key].get("criteria"):
            raise ValueError(f"'{key}' 에이전트에 criteria가 비어 있습니다: {path}")

    return agents
