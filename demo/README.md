# AI API 测试最小 Demo

这个 Demo 下载并参考了 [FastAPI](https://github.com/fastapi/fastapi) 官方仓库中的 body update 示例，使用一个等价的最小 API 作为被测对象。执行框架是 `pytest + httpx`。

## LangGraph 单用户 Agent（简单版）

新增 `user_agent.py`，使用 **LangGraph StateGraph** 管理一个“谨慎编辑用户”的反馈循环：

```text
决策（模型选择下一步工具）
    ↓
执行（本地 API 请求）
    ↓
验证（状态码、字段值、已完成检查）
    ├─ 尚未完成 → 带上操作轨迹重新决策
    └─ 全部完成 / 断言失败 / 错误 / 步数耗尽 → 输出报告
```

安装依赖后，在项目根目录运行：

```bash
python demo/run_agent.py --offline
```

离线模式使用确定性的反馈策略，不调用大模型，用于演示和回归。预期执行 5 个操作：读取原值 → 保存目标值 → 再次读取确认 → 提交缺少 price 的请求 → 再次读取确认数据未被破坏。

配置下方的 `PROJECT_GUIDE_LLM_*` 环境变量后，运行真实模型模式：

```bash
python demo/run_agent.py
```

有 API Key 时，每个决策节点通过现有 OpenRouter 兼容接口调用模型，并传入之前的请求、响应、断言和已完成检查；未设置 Key 时自动离线。报告的 `mode` 明确区分两种模式。在线调用失败会报告 `error`，不会静默切换为离线。

这版角色、目标和四个工具固定在 `user_agent.py`：`read_item`、`save_item`、`invalid_save`、`finish`。模型选择操作顺序及下一步，不生成任意请求参数；目标值和正确性规则由场景定义。当前是一个本地 API 场景，尚未实现多用户协作、自动读取其他软件契约或轨迹重放。

每次运行创建独立的内存数据。成功必须验证全部五项检查，模型提前调用 `finish` 不算通过。默认最多执行 10 个 API 操作，可用 `--max-steps 1-50` 调整。报告包含 `status`、`completed_checks`、`missing_checks` 以及完整请求、响应和断言 `trace`；仅 `passed` 返回退出码 0，`failed` / `incomplete` / `error` 返回 1。

本机 Python 3.9 使用兼容的 `langgraph==0.6.11`。无需 LangGraph Server 或 LangSmith 服务。

## LangGraph Supervisor 集群（简单版）

新增 `multi_agent.py`，将单 Agent 扩展为一个最小的 Supervisor-Workers 集群：

```text
Supervisor：拆分测试任务
       ├─ Functional Agent：正常功能与合法更新
       ├─ Boundary Agent：不存在资源与非法输入
       └─ State Agent：多步骤状态与失败副作用
                    ↓
             Supervisor：汇总规则覆盖并判断结果
```

运行：

```bash
python demo/run_multi_agent.py
```

三个子 Agent 并行执行，但各自拥有隔离的被测应用数据。每个子 Agent 返回 `checks` 和 `evidence`，主 Agent 不凭自然语言总结判断通过，而是根据规则 ID 和确定性断言汇总 `covered_rules`、`missing_rules` 与总体 `status`。

集群现在支持主 Agent 反馈循环：第一轮发布全量任务，Review 节点识别缺口后，主 Agent 只为缺失规则创建补测任务；直到全部规则通过或达到轮次上限。命令行可用 `--max-rounds 1-10` 设置预算，默认 3 轮。`passed`、`incomplete` 和 `failed` 分开表示，达到预算不等于测试通过。

## 动态 Agent 集群演示

如果希望展示“主 Agent 根据需求创建子 Agent，并让每个子 Agent 独立调用模型”，运行：

```bash
python demo/run_dynamic_cluster.py
```

这个入口使用 `dynamic_cluster.py`：

- Supervisor 根据需求选择已注册的 Agent 类型，而不是固定写死调用顺序；
- 创建 Worker 时会下发项目上下文、最小读写权限和工具白名单；
- 每个 Worker 使用独立上下文、独立测试目标和独立的被测应用数据；
- 在线模式下，Supervisor 和每个 Worker 分别调用 OpenRouter 兼容模型；
- Worker 只能输出 `ToolCall`，工具仅允许 `http_request` 和 `finish`；
- 工具参数通过 Pydantic schema 校验，Agent 类型必须来自注册表；
- 每轮结果统一为 `WorkerResult`，包含任务 ID、规则、轨迹和证据；
- Supervisor 根据缺失规则向已有 Worker 发送定向补测任务；仅需求出现新类型时才创建 Worker，最多执行 `--max-rounds` 轮。

报告的 `metrics` 包含三类可审计指标：

- `scenario_coverage`：已验证的正常用户操作场景。当前契约基线是读取已有 item、合法更新并确认持久化；
- `exception_coverage`：已验证的异常/随意操作场景。当前基线是不存在资源、缺失必填字段、失败请求后的状态保持；
- `code_coverage`：对 `target_app.py` 实际运行时插桩得到的行覆盖率和分支覆盖率，以及未覆盖行号。

模型会看到场景和异常基线并负责规划 Worker；百分比的分母仍由代码固定，防止模型通过遗漏困难场景来虚高指标。代码覆盖率不是静态猜测，而是执行期间用 `coverage.py` 采集。任何未在 API 契约中列出的无限输入空间都不能声称“全部覆盖”，需要先把风险分区补充进基线。

没有 API Key 时是离线模式，但仍会执行动态注册表选择、协议校验、独立 Worker、工具调用和结果汇总，便于演示和 CI。配置 Key 后运行同一命令即可进入在线模式：

```powershell
python demo/run_dynamic_cluster.py
```

即使环境中已经配置了 Key，也可以显式使用 `python demo/run_dynamic_cluster.py --offline` 做确定性演示。

当前“自主添加 Agent”采用安全的演示实现：Supervisor 可以生成新的 `AgentSpec`，由 `create_ephemeral_agent()` 临时实例化并执行；它不需要预先注册名称，但规则和工具仍必须通过 schema 校验。Agent 执行后只保留结构化结果和轨迹，不保留运行实例。模型不能生成和执行任意 Python 代码。

默认命令包含“必要时自动创建临时探测 Agent”的需求，可以在报告的 `agents_created` 中看到 `temporary_contract_probe`。也可以自行传入：

```bash
python demo/run_dynamic_cluster.py --offline --requirement "检查 API，自动创建一个临时资源探测 Agent"
```

这个版本特意没有使用 `langgraph-supervisor` 额外包。LangChain 官方当前推荐使用 LangGraph/LangChain 的 Supervisor + 子 Agent 工具模式，独立 `langgraph-supervisor` 仓库也说明大多数场景推荐直接使用工具调用。本项目目前使用 `StateGraph`，便于显式控制状态、并行执行、证据合并和结束条件。

测试领域可参考 Playwright Test Agents 的 planner、generator、healer 三段式设计；通用多 Agent 可参考 LangChain 的 subagents 模式、CrewAI 的 hierarchical process 和 AutoGen 的 SelectorGroupChat。当前集群采用集中式 Supervisor，而不是让所有 Agent 广播聊天，原因是测试任务需要可审计的任务边界、数据隔离和结构化证据。

验证全部 Demo 测试：

```bash
python -m pytest demo -q
```

Agent 回归测试包含“返回成功但未保存”和“拒绝非法请求却改变数据”两个注入缺陷，以验证它能发现单纯状态码检查遗漏的问题。在线适配器通过模拟 HTTP 响应验证，不需要真实 Key。

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
