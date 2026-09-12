# 주식 차트 분석 도구 (POC)

일봉 / 30분봉 / 3분봉 / 화면 캡처 이미지를 각각 전문화된 Claude 에이전트로
분석하고, 결과를 종합해서 하나의 리포트로 만드는 도구입니다.

검출 로직이 실제로 어떤 트레이딩 개념("물량 털기")을 잡으려는 건지,
임계값들이 왜 그렇게 정해졌는지는 [DOMAIN.md](DOMAIN.md) 참고. 지금까지의
변경 이력은 [HISTORY.md](HISTORY.md).

## 구조

- `config/criteria.yaml` — 타임프레임별 판단 기준. 여기에 기준 문장을 하나씩
  추가하면 됩니다.
- `chart_agent/chart_input.py` — 분석 대상을 감싸는 공통 입력 타입(`ChartInput`).
  지금은 화면 캡처 이미지(`from_image`)만 쓰지만, 나중에 증권사 REST API로
  봉 데이터를 직접 받아오게 되면 `from_candles`로 같은 파이프라인에 그대로
  꽂을 수 있습니다. 아래 "데이터 소스 확장" 참고.
- `chart_agent/agents.py` — `ChartInput` 하나 + 해당 타임프레임 기준으로
  분석하는 LLM(Claude Vision) 에이전트. 화면 캡처(screen)에서만 사용합니다.
- `chart_agent/cv_breakout.py` — OpenCV로 "물량 털기"(가격 상승+거래량
  폭증)를 직접 검출 (LLM 미사용, 무료·즉시). 캔들/거래량 막대 색상은 키움 종합차트
  기준으로 실측해 맞춰뒀습니다 — 다른 HTS/테마를 쓰면 `RED_*`/`BLUE_*`/
  `PURPLE_*` 색상 범위를 다시 잡아야 합니다.
- `chart_agent/cv_agent.py` — `cv_breakout`의 검출 결과를 `agents.py`와
  똑같은 `ChartAnalysisResult`로 변환. daily/min30/min3는 이걸 사용합니다.
- `chart_agent/orchestrator.py` — 4개 에이전트 결과를 종합해서 최종 판단을
  내리는 로직 (타임프레임 간 신호 일치 여부 확인).
- `chart_agent/report.py` — 마크다운 리포트 생성.
- `main.py` — CLI 진입점.

## 설치

```bash
pip install -r requirements.txt
```

`ANTHROPIC_API_KEY` 환경변수를 설정하세요 (`.env.example` 참고).

## 웹으로 실행하기 (로컬)

브라우저에서 업로드/붙여넣기로 바로 테스트하고 싶으면 로컬 웹 서버를 띄우면
됩니다. CLI와 같은 `chart_agent` 백엔드를 그대로 사용하고, 서버가 직접
`ANTHROPIC_API_KEY`로 Claude를 호출합니다 (Artifact 버전과 달리 뷰어의
Claude 사용량이 아니라 서버 소유자의 API 키를 씁니다).

```bash
python -m uvicorn webapp.app:app --reload --port 8000
```

`http://localhost:8000` 접속. 카드별로 이미지를 클릭/드래그 업로드하거나,
카드를 선택한 뒤 `Ctrl+V`로 캡처한 화면을 바로 붙여넣을 수 있습니다. 기준
편집("기준 저장")은 서버가 켜져 있는 동안만 유지되고, 재시작하면
`config/criteria.yaml` 값으로 돌아갑니다(원본은 항상 그 파일).

증권사 REST API를 붙일 때도 `webapp/app.py`의 `/api/analyze/{key}` 자리에
`ChartInput.from_candles(...)`를 쓰는 새 엔드포인트를 하나 추가하면 되고,
프런트엔드/다른 엔드포인트는 그대로 재사용할 수 있습니다.

## 사용법

```bash
python main.py --daily daily.png --min30 min30.png --min3 min3.png --screen screen.png --ticker "005930 삼성전자"
```

4개 중 필요한 것만 넘겨도 됩니다 (최소 1개 이상).

### 캔들 데이터로 분석하기 (이미지 없이)

일봉/30분봉/3분봉은 이미지 대신 캔들 데이터 JSON으로도 분석할 수 있습니다
(`examples/sample_candles.json` 형식 참고 — `time/open/high/low/close/volume`
목록):

```bash
python main.py --daily-candles examples/sample_candles.json --ticker "005930 삼성전자"
```

이미지와 캔들 데이터는 타임프레임당 하나만 지정할 수 있고(`--daily`와
`--daily-candles` 동시 사용 불가), 화면 캡처(`screen`)는 캔들 데이터로
대체할 수 없습니다(호가창 등 화면 전체 정보라서 캔들 수치만으로 대체가 안 됨).

## 데이터 소스 확장 (증권사 REST API 등)

지금은 사람이 캡처한 이미지를 붙여넣는 방식이지만, 나중에 키움 Open API+나
한국투자증권 KIS Developers 같은 REST API로 봉 데이터를 직접 받아오게 되면:

1. API 응답을 `[{"time":..,"open":..,"high":..,"low":..,"close":..,"volume":..}, ...]`
   형태로 변환한다.
2. `ChartInput.from_candles(candles)`로 감싼다.
3. `analyze_chart(client, key, criteria[key], chart_input, ticker=...)`에
   그대로 넘긴다 — `agents.py`/`orchestrator.py`/`report.py`는 수정할 필요가
   없습니다.

캔들 데이터를 쓰면 이미지에서 상승률·거래량 배수를 추정하는 대신 정확한
수치로 판단할 수 있다는 장점도 있습니다 — `chart_agent/cv_breakout.py`의
`detect_breakouts_from_ohlcv()`가 이미 이 경로를 구현해뒀고, `cv_agent.py`가
자동으로 이미지/캔들 데이터 중 들어온 쪽에 맞춰 처리합니다.

## 기준 추가하기

`config/criteria.yaml`의 각 타임프레임(`daily`/`min30`/`min3`/`screen`)
아래 `criteria` 리스트에 새 문장을 추가하면 다음 실행부터 바로 반영됩니다.

**단, `screen`(화면 캡처)만** 그렇습니다 — LLM이 기준 문장을 직접 읽고
판단하기 때문입니다. `daily`/`min30`/`min3`는 OpenCV(`cv_agent.py`)가
"물량 털기"(가격 상승+거래량 폭증) 한 가지만 코드로 직접 계산하므로, criteria.yaml에
새 문장을 추가해도 이 세 타임프레임에는 반영되지 않습니다. 이 세
타임프레임에 다른 종류의 기준을 추가하려면 (a) `cv_breakout.py`에 그
기준을 계산하는 로직을 추가하거나 (b) 해당 타임프레임을 다시
`agents.analyze_chart()`(LLM)로 되돌려야 합니다.
