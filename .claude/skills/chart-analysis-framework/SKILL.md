---
name: chart-analysis-framework
description: Use this skill whenever discussing, planning, or implementing chart-analysis criteria for this project (POCTraderBR_TradingAgent) — before adding a new detector or criterion, check which of the 4 big-picture analysis goals it belongs to. Holds the overall roadmap so a single detector (e.g. 물량 털기) is understood as one piece, not the whole tool.
---

# 차트 분석 큰 그림 (로드맵)

이 프로젝트가 궁극적으로 판단하려는 건 아래 4가지다. 사용자가 하나씩
구체적 판단 기준을 알려주면서 검증·고도화하는 방식으로 진행한다 — 지금
전부 구현돼 있는 게 아니라, 이 문서가 "전체 지도" 역할을 한다. 새로운
기준/검출 요청이 오면 먼저 이 중 어디에 해당하는지 확인하고, 기존에
이미 있는 개념(예: 물량 털기)의 연장인지 완전히 새 항목인지 판단한다.

## 1. 세력의 존재 확인
**상태: 방법론 미정 (TBD)**

이 종목에 자금력 있는 매매 주체(세력)가 개입해 있다는 걸 어떻게
판단할지 — 아직 사용자로부터 구체적 기준을 받지 못함.

## 2. 세력이 물량을 털고 나갔는지 확인
**상태: 1차 구현됨** — `chart_agent/cv_breakout.py` (+ `cv_agent.py`),
자세한 설명은 [DOMAIN.md](../../../DOMAIN.md).

가격을 크게 올려 거래량을 터뜨린 뒤(개미 매수 유인), 그 가격대(특히
고가/윗꼬리 — 종가가 아님)를 다시 찍게 만들어서 개미 물량을 받아내고
하락하는 패턴("물량 털기")을 탐지한다. 지금까지 검증된 부분:
- 폭발 구간을 "가격이 폭발 캔들 시가 아래로 무너지기 전까지"로 자동 확장
- 그 구간 안에서 양봉/음봉 상관없이 거래량이 큰 지점을 전부 개별 표시
- 종가가 아니라 실제 고가(윗꼬리)까지 포함해서 표시

재접근-하락(전고 돌파 후 하락) 판정 로직은 시도했다가 검증 부족 +
사용자 요청으로 제거함 (git 히스토리에 남아있음, DOMAIN.md 참고).

## 3. 세력의 매집 구간
**상태: 1차 구현됨** — `chart_agent/cv_breakout.py`의 `find_accumulation_zone()`,
자세한 설명은 [DOMAIN.md](../../../DOMAIN.md).

2번(물량 털기 = 고점에서 분산)과는 반대 개념 — 각 물량 털기 구간(장대양봉)
바로 전, 거래량이 낮게 유지되던 구간을 매집 구간으로 표시한다(하늘색
박스). 저가 횡보뿐 아니라 1차 상승 후 고점 눌림목에서도 같은 방식으로
잡힌다 — 가격대가 아니라 "거래량이 조용한지"만 본다.

## 4. 이동평균선 정배열/역배열 확인
**상태: 방법론 미정 (TBD)**

정배열(단기>중기>장기 이평선이 위에서부터 순서대로 배열 — 상승 추세) /
역배열(반대 순서 — 하락 추세) 여부를 어떻게 판단할지 아직 기준 없음.
이 프로젝트가 이미 구분해둔 이평선 5개(5/10/20/60/120일, 색상은
분홍/파랑/주황/초록/검정 — `chart_agent/cv_breakout.py` 모듈 docstring
참고)를 활용할 여지가 있다.

## 진행 방식

1. 사용자가 위 항목 중 하나를 짚어서 구체적 판단 기준(어떤 캔들/이평선/
   구간을 어떻게 보고 어떤 조건이면 해당하는지)을 알려준다.
2. 그 기준에 맞는 검출 로직을 짜고, [cv-detection-tuning 스킬](.claude/skills/cv-detection-tuning/SKILL.md)의
   워크플로우(실제 캡처로 확인 → 디버그 시각화 → SendUserFile로 전송 →
   사용자 확인 후 커밋)를 그대로 따른다.
3. 검증이 끝나면 이 문서의 해당 항목 "상태"를 갱신하고, 상세 로직/근거는
   `DOMAIN.md`에 추가한다. 커밋 후 `update-history-log` 스킬에 따라
   `HISTORY.md`도 갱신한다.
4. 네 항목은 서로 다른 목적이라 지금은 따로 개발한다 — 이 넷을 최종
   리포트에서 어떻게 종합할지(orchestrator)는 각 항목이 어느 정도
   갖춰진 뒤에 다시 논의한다.
