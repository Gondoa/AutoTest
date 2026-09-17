"""Run the single-user LangGraph demo."""

import argparse
import asyncio
import json

from user_agent import run_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Use the scripted policy without calling an LLM")
    parser.add_argument("--max-steps", type=int, choices=range(1, 51), default=10, metavar="1-50")
    args = parser.parse_args()
    report = asyncio.run(run_agent(force_offline=args.offline, max_steps=args.max_steps))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
