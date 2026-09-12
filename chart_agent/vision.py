import base64
import mimetypes
from pathlib import Path
from typing import Any, Dict

SUPPORTED_MEDIA_TYPES = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def encode_image(image_path: str) -> Dict[str, Any]:
    """이미지 파일을 Claude Messages API의 image content block으로 변환"""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
    if path.suffix.lower() not in SUPPORTED_MEDIA_TYPES:
        raise ValueError(
            f"지원하지 않는 이미지 형식입니다: {path.suffix} "
            f"(지원: {', '.join(sorted(SUPPORTED_MEDIA_TYPES))})"
        )

    media_type = mimetypes.guess_type(str(path))[0] or "image/png"
    with open(path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")

    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": data,
        },
    }
