---
name: update-history-log
description: Use this skill every time you create a git commit in this project (POCTraderBR_TradingAgent). Keeps HISTORY.md up to date with a dated, user-facing summary of what each commit shipped, so anyone can see project history without reading diffs or git log.
---

# 커밋 히스토리 로그 유지

이 프로젝트는 루트의 `HISTORY.md`에 "언제 어떤 핵심 기능이 추가/변경됐는지"를
날짜별로 기록합니다. **이 저장소에서 커밋을 만들 때마다** (사용자가 따로
요청하지 않아도) 아래 절차를 따르세요.

## 절차

1. 평소대로 코드 변경사항을 커밋한다 (`git commit ...`).
2. 커밋 해시와 날짜를 확인한다:
   ```
   git log -1 --format="%h|%ad" --date=format:'%Y-%m-%d'
   ```
3. `HISTORY.md`를 연다.
   - 방금 커밋한 날짜와 같은 `## YYYY-MM-DD` 섹션이 이미 최상단에 있으면
     그 섹션 맨 위에 새 항목을 추가한다.
   - 없으면 파일 맨 위(제목 설명 바로 아래)에 새 `## YYYY-MM-DD` 섹션을
     만들고 그 아래에 항목을 추가한다. (최신 날짜가 항상 위로 오게 유지)
4. 항목 형식:
   ```
   - **핵심 기능을 한 문장으로 요약** (`짧은해시`) — 사용자 관점에서 무엇이
     가능해졌는지/무엇이 바뀌었는지 1~2문장.
   ```
5. `HISTORY.md`만 변경사항으로 스테이징해서 **별도의 작은 커밋**을 만든다
   (예: `Update HISTORY.md`). 방금 만든 기능 커밋에 amend로 끼워넣지 않는다
   — 이 프로젝트는 기존 커밋을 수정하지 않고 항상 새 커밋을 쌓는 방식을
   쓴다.

## 무엇을 적을지 판단하는 기준

- **코드 diff 요약이 아니라 기능 관점.** "함수 X를 리팩터링함" (X) →
  "이제 캔들 데이터로도 같은 판단을 돌릴 수 있음" (O).
- 오탈자 수정, 포맷팅, 사소한 리팩터링처럼 사용자가 체감할 기능 변화가
  없는 커밋은 HISTORY.md에 안 적어도 된다 — 판단이 애매하면 적는 쪽을
  택한다 (누락보다 사소한 항목 하나 더 있는 게 낫다).
- 한 커밋에 여러 핵심 기능이 섞여 있으면 항목을 여러 줄로 나눠 적어도 된다.
- 과거에 적은 항목은 절대 수정/삭제하지 않는다 (append-only 이력).

## 예시

```markdown
## 2026-09-13

- **키움 REST API 연동** (`a1b2c3d`) — 화면 캡처 대신 실제 API로 캔들
  데이터를 받아와서 바로 분석 가능해짐 (더 이상 스크린샷 필요 없음).

## 2026-09-12

- **이미지 클릭 시 확대 모달** (`39aee04`) — ...
```
