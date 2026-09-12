---
name: cv-detection-tuning
description: Use this skill whenever changing pixel-level detection logic in chart_agent/cv_breakout.py (colors, thresholds, wick/body extraction, clustering, retest logic) in this project (POCTraderBR_TradingAgent). Defines the required workflow — real screenshots plus a visual before/after — before any such change is considered done or committed.
---

# OpenCV 검출 로직 튜닝 워크플로우

`chart_agent/cv_breakout.py`의 색상/픽셀 임계값, 몸통·꼬리 추출, 클러스터링,
재접근-하락 판정 등을 건드릴 때는 반드시 아래 절차를 따른다. 설명만으로
"고쳤다"고 판단하지 않는다 — 이 프로젝트에서 여러 번, 코드상으로는 맞아
보이던 수정이 실제 캡처로 확인하고 나서야 틀린 게 드러났다.

## 절차

1. **실제 캡처 이미지를 구한다.** `examples/` 아래 이미 있는 샘플
   (`sample_daily_screen.png`, `sample_daily_screen_hires.png` 등)을 쓰거나,
   없으면 합성 이미지로 임의로 만들지 말고 사용자에게 실제 키움 HTS
   캡처를 요청한다. 색상/안티에일리어싱 특성은 실제 캡처에서만 정확히
   재현된다.
2. **수정 후 그 이미지로 직접 돌려서 디버그 시각화를 만든다**
   (`draw_debug()`로 PNG 생성 — 검출된 캔들/거래량/재접근 지점을 원본
   위에 동그라미로 표시).
3. **그 이미지를 `SendUserFile`로 사용자에게 보낸다.** 텍스트로 "이렇게
   고쳤습니다"라고만 설명하고 넘어가지 않는다.
4. 사용자가 실제로 보면서 맞는지 확인(또는 특정 지점을 짚어서 반박)하게
   한다 — 사용자는 종종 같은 캡처 위에 직접 손으로 동그라미를 그려서
   기대하는 결과를 보여준다.
5. 시각적으로 맞다고 확인된 뒤에만 커밋한다. 커밋 후에는
   `update-history-log` 스킬에 따라 `HISTORY.md`도 갱신하고, 변경한
   내용이 [DOMAIN.md](../../../DOMAIN.md)에 적힌 설명과 어긋나게 됐다면
   그 문서도 같이 고친다 (실제로 한 번, 음봉 포함 수정 후 문서를 안
   고쳐서 어긋난 채로 남아있었던 적이 있다 — 커밋 `576d32c`).

## 왜 이렇게 하는가

같은 수정이 코드만 보면 그럴듯해도 여러 번 조용히 틀렸었다:
- 이평선 제거용 모폴로지 오프닝이 얇은 캔들 꼬리까지 같이 지워버림
  (색상 마스크 기반 꼬리 복원 시도 1차 실패)
- 꼬리 복원을 "행 평균 밝기"로 계산했더니, 캔들 폭 전체로 평균을 내면서
  1~2px짜리 옅은 꼬리 신호가 배경 픽셀들 사이에 묻혀버림 (2차 실패)
- "행 내 최솟값 밝기"로 바꾸고 나서야 실제 63px/127px짜리 꼬리가
  정확히 잡힘 — 이것도 사용자가 더 고해상도로 재캡처해서 보내준
  이미지로 직접 픽셀을 찍어봐서 알아낸 것

가상의 픽셀 색을 추론하며 디버깅하는 것보다, 실제 이미지를 직접 찍어보는
쪽이 훨씬 빠르고 정확하다.

## 관련 문서

- 이 검출 로직이 잡으려는 실제 트레이딩 개념(물량 털기, 전고 돌파,
  윗꼬리)은 [DOMAIN.md](../../../DOMAIN.md)에 정리되어 있다. 새 튜닝
  요청이 들어오면 먼저 그 문서에 이미 있는 개념의 연장인지 확인한다.
