# runners/run_cluster.py

import argparse
import asyncio
import json
import os
import sys

from core.protocol import AgentHandle, AgentSpec, AgentTask, AgentStatus
from agents.master_agent import MasterAgent


def build_master(offline: bool, max_rounds: int) -> MasterAgent:
    """
    组装主 Agent。

    这里是唯一需要知道"所有模块怎么拼在一起"的地方。
    """
    # 被测文件的绝对路径（子 Agent 的 CoverageTool 需要）
    source_path = os.path.join(
        os.path.dirname(__file__),
        "..", "targets", "fastapi_app", "app.py"
    )
    source_path = os.path.abspath(source_path)

    # 主 Agent 自己也需要一个 AgentSpec（描述它自己的身份）
    master_spec = AgentSpec(
        agent_type    = "master",
        objective     = "协调子 Agent 完成软件测试",
        prompt        = "你是测试主管，负责规划和协调测试流程。",
        allowed_paths = [],         # 主 Agent 不直接发 HTTP 请求
        allowed_tools = [],         # 主 Agent 不直接使用工具
        rules         = [],
        max_steps     = max_rounds,
    )
    master_handle = AgentHandle(spec=master_spec)

    return MasterAgent(
        handle      = master_handle,
        source_path = source_path,
        max_rounds  = max_rounds,
        offline     = offline,
    )


async def run(objective: str, offline: bool, max_rounds: int) -> int:
    """
    主流程：创建主 Agent，派发任务，打印报告，返回退出码。
    """
    master = build_master(offline=offline, max_rounds=max_rounds)

    task = AgentTask(
        agent_id    = master.agent_id,
        instruction = objective,
        context     = {},
    )

    print(f"[集群启动] 目标：{objective}")
    print(f"[集群启动] 模式：{'离线' if offline else '在线'} / 最大轮次：{max_rounds}")
    print("-" * 60)

    result = await master.run(task)

    # 打印最终报告
    print("\n[最终报告]")
    print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))

    # 退出码：passed=0，其他=1
    return 0 if result.status == "passed" else 1


def main():
    parser = argparse.ArgumentParser(description="AutoTest Agent 集群")
    parser.add_argument(
        "--objective",
        default = "测试 FastAPI item CRUD API，确保读写、错误处理和状态一致性正确",
        help    = "测试目标（自然语言描述）",
    )
    parser.add_argument(
        "--offline",
        action  = "store_true",
        help    = "使用离线（确定性）模式，不调用 LLM",
    )
    parser.add_argument(
        "--max-rounds",
        type    = int,
        default = 5,
        help    = "最大决策轮次（默认 5）",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(run(
        objective  = args.objective,
        offline    = args.offline,
        max_rounds = args.max_rounds,
    ))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()