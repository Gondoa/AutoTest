"""Run the demonstrable dynamic multi-agent cluster."""

import argparse
import asyncio
import json

from dynamic_cluster import run_dynamic_cluster


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requirement", default="测试 item API 的功能、边界和状态一致性；必要时自动创建临时探测 Agent")
    parser.add_argument("--max-rounds", type=int, choices=range(1, 11), default=1)
    parser.add_argument("--offline", action="store_true", help="Do not call an LLM")
    args = parser.parse_args()
    report = asyncio.run(run_dynamic_cluster(args.requirement, args.max_rounds, args.offline))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
