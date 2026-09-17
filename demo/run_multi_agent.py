"""Run the minimal multi-agent test cluster."""

import argparse
import asyncio
import json

from multi_agent import run_cluster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirement", default="测试 item API 的功能、边界和状态一致性")
    parser.add_argument("--max-rounds", type=int, choices=range(1, 11), default=3)
    args = parser.parse_args()
    report = asyncio.run(run_cluster(args.requirement, max_rounds=args.max_rounds))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
