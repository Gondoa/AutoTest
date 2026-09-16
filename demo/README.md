# AI API 测试最小 Demo

这个 Demo 下载并参考了 [FastAPI](https://github.com/fastapi/fastapi) 官方仓库中的 body update 示例，使用一个等价的最小 API 作为被测对象。执行框架是 `pytest + httpx`。

## 安装

在项目根目录执行：

```bash
python -m pip install -r demo/requirements.txt
```

## 运行 Demo

```bash
python demo/run_demo.py
```

默认使用离线规划器，因此不需要 AI API Key。它模拟 AI 输出经过 schema 校验的测试计划，生成并执行读取、更新、404 和参数校验测试。

运行 pytest：

```bash
python -m pytest demo/test_generated_api.py -q
```

## AI 接入点

Demo 使用项目已有的 OpenRouter 配置：

```powershell
setx PROJECT_GUIDE_LLM_API_KEY "你的 OpenRouter API Key"
setx PROJECT_GUIDE_LLM_MODEL "~deepseek/deepseek-flash-latest"
setx PROJECT_GUIDE_LLM_URL "https://openrouter.ai/api/v1/chat/completions"
```

`setx` 只对新打开的终端生效。配置后重新打开 PowerShell，再运行：

```powershell
python demo/run_demo.py
```

`ai_planner.py` 会将需求发送到 Chat Completions 接口，并校验模型返回的测试计划。模型只能返回结构化测试计划，不能直接执行 shell 命令或任意代码。未设置 API Key 时自动使用离线规划器。

pytest 回归测试显式使用离线计划，避免测试结果受网络、模型版本和模型随机输出影响；`python demo/run_demo.py` 才是实际调用配置模型的入口。

## 当前 Demo 体现的 AI 闭环

```text
自然语言需求 -> AI 测试计划 -> schema 校验 -> API 执行 -> 结果报告
```

下一步可以在 `run_demo.py` 中加入接口覆盖率、状态码覆盖率和未覆盖场景补测，让流程变成：

```text
生成 -> 执行 -> 覆盖率分析 -> AI 补测 -> 再执行
```
