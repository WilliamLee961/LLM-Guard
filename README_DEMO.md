# LLM Guard 前后端演示版

这个 Demo 面向“可给甲方展示”的场景，包含：

- Web 前端控制台（输入问题、切换检测强度、展示输入/输出 Guard 过程）
- 后端 API（统一封装 Guard 流程）
- 双引擎模式
  - `demo`: 本地规则引擎，稳定、无外部依赖，适合现场演示
  - `real`: 调用仓库已有 `run_safe_qa/run_guardrails` 真实链路

如果你要跑论文实验（baseline 对比、阈值校准、Mac 启动流程），请看：

- `README_RESEARCH.md`

## 1. 安装依赖

```bash
pip install -r requirements-demo.txt
```

如果你要启用 `real` 模式，还需要安装仓库原有依赖（如 `langgraph/openai/faiss` 等）。
另外可通过环境变量配置真实模型地址（`llm_client.py` 已支持 OpenAI 与 Ark 兼容格式）：

```bash
set OPENAI_BASE_URL=http://your-host:8000/v1
set OPENAI_MODEL_NAME=Qwen
set OPENAI_API_KEY=EMPTY
```

### 使用火山方舟（Ark）API（推荐用于演示）

在 PowerShell 中设置（注意 `ARK_ENDPOINT_ID` 必须填你的“推理接入点 Endpoint ID”）：

```powershell
$env:ARK_API_KEY="你的API Key"
$env:ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
$env:ARK_ENDPOINT_ID="ep-xxxxxxxxxxxxxxxx"
$env:OPENAI_TRUST_ENV_PROXY="0"
```

如果曾经设置过 `$env:OPENAI_MODEL_NAME="Qwen"`，请清掉，避免模型名被旧值覆盖：

```powershell
Remove-Item Env:OPENAI_MODEL_NAME -ErrorAction SilentlyContinue
```

然后启动：

```powershell
python -m uvicorn demo_api:app --host 0.0.0.0 --port 8787 --reload
```

快速检查是否已连接到 Ark：

```powershell
Invoke-RestMethod http://127.0.0.1:8787/api/health
```

`/api/health` 返回里应看到你设置的 `llm_base_url` 和 `llm_model_name`。

#### 常见报错排查（Ark）

- 报错 `InvalidEndpointOrModel.NotFound`：
  - 当前 `model` 不存在，或你无权限访问。
  - 优先在控制台复制“推理接入点 Endpoint ID”（常见为 `ep-...`），设置到 `ARK_ENDPOINT_ID`。
  - 确认 `ARK_BASE_URL` 地域与接入点地域一致（例如北京区使用 `ark.cn-beijing.volces.com`）。
  - 确认 API Key 与该接入点在同一项目/账号下。
- 报错 `InvalidEndpointOrModel.ModelIDAccessDisabled`：
  - 说明账号未开通“按模型 ID 直连”能力，请改用你创建的 `Endpoint ID`。

## 2. 启动服务

```bash
uvicorn demo_api:app --host 0.0.0.0 --port 8787 --reload
```

或在 PowerShell 里直接：

```powershell
.\run_demo.ps1 -Port 8787
```

浏览器打开：

```text
http://127.0.0.1:8787
```

## 3. 关键接口

- `GET /api/health`：服务健康状态与真实引擎可用性
- `GET /api/examples`：前端示例提示词
- `POST /api/guarded-chat`：完整输入/输出 Guard 流程
- `POST /api/guard/analyze`：纯检测接口

## 4. 甲方演示建议脚本

1. 先用 `demo + normal` 演示“正常问题可放行”
2. 切到“越狱/违法示例”，展示输入侧拦截
3. 打开“模拟模型越权输出”，展示输出侧复检与重写
4. 切到 `strict`，展示更强拦截策略
5. 如果环境可用，再切 `real` 展示真实链路

## 5. 说明

- 当你请求 `real` 但真实链路不可用时，后端会自动降级到 `demo`，响应里有：
  - `used_fallback=true`
  - `fallback_reason=<错误原因>`
- 这样可以保证现场演示不中断。

## 6. 响应决策策略（与论文第3章对齐）

系统不再只做“二分类拦截”，而是使用动作集合：

- `ALLOW`：直接放行
- `REWRITE`：安全改写（保留可用信息，移除危险细节）
- `SAFE_ANSWER`：安全代答（拒绝危险目标 + 给可执行替代）
- `REFUSE`：明确拒绝并给替代路径
- `ESCALATE`：紧急风险升级处置（如明显自伤风险）

并按难度模式映射整体风险等级（L1-L4）到动作决策，用于提升 strict/normal 下的可用性与解释性。
