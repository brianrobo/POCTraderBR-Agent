# 주식 차트 분석 도구 (POC)

일봉 / 30분봉 / 3분봉 / 화면 캡처 이미지를 각각 전문화된 Claude 에이전트로
분석하고, 결과를 종합해서 하나의 리포트로 만드는 도구입니다.

## 구조

- `config/criteria.yaml` — 타임프레임별 판단 기준. 여기에 기준 문장을 하나씩
  추가하면 됩니다.
- `chart_agent/agents.py` — 이미지 한 장 + 해당 타임프레임 기준으로 분석하는
  개별 에이전트 로직.
- `chart_agent/orchestrator.py` — 4개 에이전트 결과를 종합해서 최종 판단을
  내리는 로직 (타임프레임 간 신호 일치 여부 확인).
- `chart_agent/report.py` — 마크다운 리포트 생성.
- `main.py` — CLI 진입점.

## 설치

```bash
pip install -r requirements.txt
```

`ANTHROPIC_API_KEY` 환경변수를 설정하세요 (`.env.example` 참고).

## 사용법

```bash
python main.py --daily daily.png --min30 min30.png --min3 min3.png --screen screen.png --ticker "005930 삼성전자"
```

4개 중 필요한 것만 넘겨도 됩니다 (최소 1개 이상).

## 기준 추가하기

`config/criteria.yaml`의 각 타임프레임(`daily`/`min30`/`min3`/`screen`)
아래 `criteria` 리스트에 새 문장을 추가하면 다음 실행부터 바로 반영됩니다.
코드 수정이 필요 없습니다.
