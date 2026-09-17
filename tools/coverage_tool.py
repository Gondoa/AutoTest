# tools/coverage_tool.py

from __future__ import annotations
import os
import tempfile
from typing import Dict, Optional

from coverage import Coverage


class CoverageTool:
    """
    运行时代码覆盖率测量工具。

    每个子 Agent 持有一个独立实例，
    测量该 Agent 执行期间被测 App 的覆盖情况。
    """

    def __init__(self, source_path: str):
        """
        source_path: 被测文件的绝对路径，如
                     "C:/D/study/AutoTest/targets/fastapi_app/app.py"
        """
        self.source_path = source_path
        self._cov: Optional[Coverage] = None
        # 用临时文件存放 .coverage 数据，避免多个子 Agent 互相覆盖
        self._data_file = tempfile.mktemp(suffix=".coverage")

    # ── 生命周期 ───────────────────────────────────────────────────────────────

    def start(self) -> None:
        """开始测量，在子 Agent 执行第一个工具调用前调用"""
        self._cov = Coverage(
            data_file   = self._data_file,
            source      = [self.source_path],
            branch      = True,    # 同时测量分支覆盖率
        )
        self._cov.start()

    def stop(self) -> None:
        """停止测量，在子 Agent 调用 finish 工具后调用"""
        if self._cov is not None:
            self._cov.stop()
            self._cov.save()

    # ── 数据读取 ───────────────────────────────────────────────────────────────

    def get_report(self) -> Dict[str, float]:
        """
        返回覆盖率报告。

        返回格式：
          {
            "line_coverage":   0.85,   # 85% 的行被执行
            "branch_coverage": 0.70,   # 70% 的分支被执行
            "missing_lines":   [12, 15, 23],  # 未覆盖的行号
          }
        """
        if self._cov is None:
            return {"line_coverage": 0.0, "branch_coverage": 0.0, "missing_lines": []}

        self._cov.load()

        analysis = self._cov.analysis2(self.source_path)
        # analysis 返回：(filename, statements, excluded, missing, missing_branch_arcs, missing_branch_arcs_formatted)

        statements = analysis[1]   # 所有可执行行
        missing    = analysis[3]   # 未执行的行

        total    = len(statements)
        executed = total - len(missing)

        line_coverage = (executed / total) if total > 0 else 0.0

        # 分支覆盖率：coverage.py 内置计算
        try:
            branch_coverage = self._cov.report(
                include = [self.source_path],
                output_format = "total",
            ) / 100.0
        except Exception:
            branch_coverage = line_coverage   # fallback：用行覆盖率代替

        return {
            "line_coverage":   round(line_coverage,   2),
            "branch_coverage": round(branch_coverage, 2),
            "missing_lines":   list(missing),
        }

    # ── 清理 ───────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """删除临时 .coverage 文件，子 Agent 销毁时调用"""
        if os.path.exists(self._data_file):
            os.remove(self._data_file)