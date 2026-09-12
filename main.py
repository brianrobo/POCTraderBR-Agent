import argparse
import sys
from datetime import datetime

import anthropic

from chart_agent.config import load_criteria
from chart_agent.agents import analyze_chart
from chart_agent.orchestrator import synthesize
from chart_agent.report import render_report, save_report

TIMEFRAME_ARGS = {
    "daily": "daily",
    "min30": "min30",
    "min3": "min3",
    "screen": "screen",
}


def parse_args():
    parser = argparse.ArgumentParser(description="주식 차트 이미지 다중 타임프레임 분석 도구")
    parser.add_argument("--daily", help="일봉 차트 이미지 경로")
    parser.add_argument("--min30", help="30분봉 차트 이미지 경로")
    parser.add_argument("--min3", help="3분봉 차트 이미지 경로")
    parser.add_argument("--screen", help="화면 캡처 이미지 경로")
    parser.add_argument("--ticker", help="종목명/코드 (선택)")
    parser.add_argument("--out", help="리포트 저장 경로 (기본: outputs/report_<timestamp>.md)")
    return parser.parse_args()


def main():
    args = parse_args()

    image_paths = {
        key: getattr(args, key)
        for key in TIMEFRAME_ARGS
        if getattr(args, key)
    }

    if not image_paths:
        print("최소 하나 이상의 차트 이미지를 지정해야 합니다. (--daily, --min30, --min3, --screen)", file=sys.stderr)
        sys.exit(1)

    criteria = load_criteria()
    client = anthropic.Anthropic()

    results = {}
    for key, path in image_paths.items():
        print(f"[{key}] 분석 중... ({path})")
        results[key] = analyze_chart(client, key, criteria[key], path, ticker=args.ticker)
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
