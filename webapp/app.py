import base64
import os
from pathlib import Path
from typing import Dict, List, Optional

import anthropic
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from chart_agent.agents import analyze_chart
from chart_agent.chart_input import ChartInput
from chart_agent.config import CANDLE_CAPABLE_KEYS, TIMEFRAME_KEYS, load_criteria
from chart_agent.cv_agent import analyze_chart_cv_annotated
from chart_agent.models import ChartAnalysisResult
from chart_agent.orchestrator import synthesize

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="브레이크아웃 시그널 데스크 (로컬)")

if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
    print(
        "[경고] ANTHROPIC_API_KEY(또는 ANTHROPIC_AUTH_TOKEN)가 설정되어 있지 않습니다. "
        "분석 요청 시 실패합니다. .env.example을 참고해 환경변수를 설정하세요."
    )

client = anthropic.Anthropic()
criteria_state: Dict = load_criteria()


def call_anthropic(fn, *args, **kwargs):
    """anthropic 호출을 감싸서 실패 유형별로 적절한 HTTP 오류로 변환"""
    try:
        return fn(*args, **kwargs)
    except anthropic.RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"요청이 많습니다. 잠시 후 다시 시도하세요. ({e.message})")
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=500, detail="서버의 ANTHROPIC_API_KEY가 올바르지 않습니다.")
    except anthropic.BadRequestError as e:
        raise HTTPException(status_code=400, detail=f"잘못된 요청입니다: {e.message}")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API 오류 ({e.status_code}): {e.message}")
    except anthropic.APIConnectionError:
        raise HTTPException(status_code=502, detail="Claude API에 연결하지 못했습니다. 네트워크를 확인하세요.")
    except TypeError as e:
        if "authentication" in str(e).lower():
            raise HTTPException(
                status_code=500,
                detail="ANTHROPIC_API_KEY가 설정되지 않았습니다. 서버를 재시작하기 전에 환경변수를 설정하세요.",
            )
        raise HTTPException(status_code=500, detail=f"예상치 못한 오류: {e}")


@app.get("/api/criteria")
def get_criteria():
    return {
        key: {"label": criteria_state[key]["label"], "criteria": criteria_state[key]["criteria"]}
        for key in TIMEFRAME_KEYS
    }


class CriteriaUpdate(BaseModel):
    criteria: List[str]


@app.patch("/api/criteria/{key}")
def update_criteria(key: str, body: CriteriaUpdate):
    if key not in TIMEFRAME_KEYS:
        raise HTTPException(status_code=404, detail=f"알 수 없는 타임프레임: {key}")
    cleaned = [c.strip() for c in body.criteria if c.strip()]
    if not cleaned:
        raise HTTPException(status_code=400, detail="criteria는 비어 있을 수 없습니다.")
    criteria_state[key]["criteria"] = cleaned
    return {"label": criteria_state[key]["label"], "criteria": criteria_state[key]["criteria"]}


class AnalyzeResponse(ChartAnalysisResult):
    annotated_image_data_url: Optional[str] = None  # 검출 구간에 동그라미 표시한 이미지 (CV 결과일 때만)


@app.post("/api/analyze/{key}", response_model=AnalyzeResponse)
def analyze(key: str, image: UploadFile = File(...), ticker: Optional[str] = Form(None)):
    if key not in TIMEFRAME_KEYS:
        raise HTTPException(status_code=404, detail=f"알 수 없는 타임프레임: {key}")

    data = image.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="이미지 데이터가 비어 있습니다.")

    chart_input = ChartInput.from_image_bytes(data, image.content_type or "image/png")

    try:
        if key in CANDLE_CAPABLE_KEYS:
            result, annotated_png = analyze_chart_cv_annotated(key, criteria_state[key], chart_input)
            annotated_url = None
            if annotated_png:
                annotated_url = "data:image/png;base64," + base64.b64encode(annotated_png).decode("ascii")
            return AnalyzeResponse(**result.model_dump(), annotated_image_data_url=annotated_url)
        else:
            result = call_anthropic(
                analyze_chart, client, key, criteria_state[key], chart_input, ticker=ticker or None
            )
            return AnalyzeResponse(**result.model_dump())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


class SynthesizeRequest(BaseModel):
    results: Dict[str, ChartAnalysisResult]
    ticker: Optional[str] = None


@app.post("/api/synthesize")
def synthesize_endpoint(body: SynthesizeRequest):
    if not body.results:
        raise HTTPException(status_code=400, detail="종합할 결과가 없습니다.")
    text = call_anthropic(synthesize, client, body.results, ticker=body.ticker or None)
    return {"synthesis": text}


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")
