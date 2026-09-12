import argparse
import json
import sys
from datetime import datetime

import anthropic

from chart_agent.chart_input import ChartInput
from chart_agent.config import CANDLE_CAPABLE_KEYS, load_criteria
from chart_agent.agents import analyze_chart
from chart_agent.cv_agent import analyze_chart_cv
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

    results = {}
    for key, chart_input in chart_inputs.items():
        if key in CANDLE_CAPABLE_KEYS:
            print(f"[{key}] OpenCV로 분석 중... ({chart_input.kind}, LLM 미사용)")
            results[key] = analyze_chart_cv(key, criteria[key], chart_input)
        else:
            print(f"[{key}] Claude로 분석 중... ({chart_input.kind})")
            results[key] = analyze_chart(client, key, criteria[key], chart_input, ticker=args.ticker)
        print(f"[{key}] 완료 - 신호 발견: {results[key].overall_found}")

    print("종합 판단 생성 중...")
    synthesis = synthesize(client, results, ticker=args.ticker)

    out_path = args.out or f"outputs/report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    save_report(results, synthesis, out_path, ticker=args.ticker)

    print()
    print(render_report(results, synthesis, ticker=args.ticker))
    print()
    print(f"리포트 저장 완료: {out_path}")


if __name__ == "__main__":
    main()
