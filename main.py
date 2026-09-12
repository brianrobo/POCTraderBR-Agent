import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import anthropic

from chart_agent.chart_input import ChartInput
from chart_agent.config import CANDLE_CAPABLE_KEYS, load_criteria
from chart_agent.agents import analyze_chart
from chart_agent.cv_agent import analyze_chart_cv_annotated
from chart_agent.orchestrator import synthesize
from chart_agent.report import render_report, save_report

TIMEFRAME_KEYS = ["daily", "min30", "min3", "screen"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "주식 차트 다중 타임프레임 분석 도구. "
            "각 타임프레임은 --<key>(이미지) 또는 --<key>-candles(증권사 API 등에서 받은 "
            "캔들 JSON 파일) 중 하나로 지정한다."
        )
    )
    parser.add_argument("--daily", help="일봉 차트 이미지 경로")
    parser.add_argument("--min30", help="30분봉 차트 이미지 경로")
    parser.add_argument("--min3", help="3분봉 차트 이미지 경로")
    parser.add_argument("--screen", help="화면 캡처 이미지 경로")
    parser.add_argument(
        "--daily-candles",
        help="일봉 캔들 데이터 JSON 경로 (증권사 REST API 응답 등을 [{time,open,high,low,close,volume}, ...] 형태로 저장)",
    )
    parser.add_argument("--min30-candles", help="30분봉 캔들 데이터 JSON 경로")
    parser.add_argument("--min3-candles", help="3분봉 캔들 데이터 JSON 경로")
    parser.add_argument("--ticker", help="종목명/코드 (선택)")
    parser.add_argument("--out", help="리포트 저장 경로 (기본: outputs/report_<timestamp>.md)")
    return parser.parse_args()


def call_anthropic(fn, *args, **kwargs):
    """anthropic 호출 실패를 트레이스백 대신 안내 메시지로 바꿔서 종료"""
    try:
        return fn(*args, **kwargs)
    except anthropic.RateLimitError as e:
        print(f"요청이 많습니다. 잠시 후 다시 시도하세요. ({e.message})", file=sys.stderr)
    except anthropic.AuthenticationError:
        print("ANTHROPIC_API_KEY가 올바르지 않습니다.", file=sys.stderr)
    except anthropic.BadRequestError as e:
        print(f"잘못된 요청입니다: {e.message}", file=sys.stderr)
    except anthropic.APIStatusError as e:
        print(f"Claude API 오류 ({e.status_code}): {e.message}", file=sys.stderr)
    except anthropic.APIConnectionError:
        print("Claude API에 연결하지 못했습니다. 네트워크를 확인하세요.", file=sys.stderr)
    except TypeError as e:
        if "authentication" in str(e).lower():
            print(
                "ANTHROPIC_API_KEY가 설정되지 않았습니다. 환경변수를 설정한 뒤 다시 실행하세요 "
                "(.env.example 참고). daily/min30/min3만 쓰는 경우 OpenCV만 동작하고 이 단계는 "
                "건너뛸 수 없습니다 — 종합 판단은 항상 Claude를 호출합니다.",
                file=sys.stderr,
            )
        else:
            raise
    sys.exit(1)


def load_candles(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_chart_inputs(args) -> dict:
    chart_inputs = {}

    for key in TIMEFRAME_KEYS:
        image_path = getattr(args, key)
        candles_path = getattr(args, f"{key}_candles", None)

        if image_path and candles_path:
            print(f"--{key}와 --{key}-candles는 동시에 지정할 수 없습니다.", file=sys.stderr)
            sys.exit(1)
        if candles_path and key not in CANDLE_CAPABLE_KEYS:
            print(f"--{key}-candles는 지원하지 않습니다 (화면 캡처는 이미지만 가능).", file=sys.stderr)
            sys.exit(1)

        if image_path:
            chart_inputs[key] = ChartInput.from_image(image_path)
        elif candles_path:
            chart_inputs[key] = ChartInput.from_candles(load_candles(candles_path))

    return chart_inputs


def main():
    args = parse_args()
    chart_inputs = build_chart_inputs(args)

    if not chart_inputs:
        print(
            "최소 하나 이상의 차트 자료를 지정해야 합니다. "
            "(--daily/--min30/--min3/--screen 또는 --daily-candles/--min30-candles/--min3-candles)",
            file=sys.stderr,
        )
        sys.exit(1)

    criteria = load_criteria()
    client = anthropic.Anthropic()

    out_path = args.out or f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    annotated_paths = {}

    results = {}
    for key, chart_input in chart_inputs.items():
        if key in CANDLE_CAPABLE_KEYS:
            print(f"[{key}] OpenCV로 분석 중... ({chart_input.kind}, LLM 미사용)")
            results[key], annotated_png = analyze_chart_cv_annotated(key, criteria[key], chart_input)
            if annotated_png:
                annotated_path = Path(out_path).with_name(Path(out_path).stem + f"_{key}.png")
                annotated_path.write_bytes(annotated_png)
                annotated_paths[key] = str(annotated_path)
        else:
            print(f"[{key}] Claude로 분석 중... ({chart_input.kind})")
            results[key] = call_anthropic(
                analyze_chart, client, key, criteria[key], chart_input, ticker=args.ticker
            )
        print(f"[{key}] 완료 - 신호 발견: {results[key].overall_found}")

    print("종합 판단 생성 중...")
    synthesis = call_anthropic(synthesize, client, results, ticker=args.ticker)

    save_report(results, synthesis, out_path, ticker=args.ticker)

    print()
    print(render_report(results, synthesis, ticker=args.ticker))
    print()
    print(f"리포트 저장 완료: {out_path}")
    for key, path in annotated_paths.items():
        print(f"[{key}] 동그라미 표시 이미지: {path}")


if __name__ == "__main__":
    main()
