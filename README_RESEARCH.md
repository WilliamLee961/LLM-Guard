# LLM Guard 研究实验指南（CCF-A 方向）

本指南面向“侧载式 Input/Output Guard”研究与论文实验，目标是：

- 在不改底模权重的前提下，降低越狱/危险输出
- 同时控制 overrefusal（过度拒答）
- 形成可复现实验流程（baseline、消融、校准）

---

## 1. 代码改动要点

本仓库已加入三项研究化能力：

1. **边界校准策略（Boundary-aware calibration）**  
   文件：`response_policy.py`  
   - 支持通过 `GUARD_POLICY_CALIB_PATH` 外置 JSON 动态调整策略阈值  
   - 支持不确定性触发 `SAFE_ANSWER/REFUSE/REWRITE` 升级

2. **多视角风险检测（Multi-view uncertainty）**  
   文件：`risk_hub.py`  
   - 在可疑输入下进行二次“防守视角”检测  
   - 输出 `debug.multi_view.disagreement_count` 作为不确定性指标

3. **目标保持型安全改写（Goal-preserving rewrite）**  
   文件：`safe_qa_graph.py`  
   - 先提取“应保留的合法目标”再生成安全改写/代答，减少空话

---

## 2. Mac 启动（推荐）

在 macOS zsh/bash 中：

```bash
cd /path/to/LLM-Guard/LLM-Guard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-research.txt
```

设置 Ark API 环境变量：

```bash
export ARK_API_KEY="你的key"
export ARK_BASE_URL="https://ark.cn-beijing.volces.com/api/v3"
export ARK_ENDPOINT_ID="ep-xxxxxxxxxxxxxxxx"
export OPENAI_TRUST_ENV_PROXY="0"
unset OPENAI_MODEL_NAME
```

你也可以复制 `.env.example` 作为参考配置模板。

启动前后端演示：

```bash
chmod +x run_demo.sh
./run_demo.sh 8787
```

或直接：

```bash
python -m uvicorn demo_api:app --host 127.0.0.1 --port 8787 --reload
```

浏览器访问：

```text
http://127.0.0.1:8787
```

---

## 3. 运行 baseline 对比实验

### 3.1 样例数据集

样例文件：

`experiments/datasets/sample_guard_eval.jsonl`

格式（每行 JSON）：

```json
{"id":"x1","query":"...","is_harmful":true,"category":"..."}
```

### 3.2 一键跑对比

```bash
python experiments/run_guard_benchmark.py \
  --dataset experiments/datasets/sample_guard_eval.jsonl \
  --difficulty normal \
  --judge-difficulty strict \
  --baselines no_guard,input_only,output_only,full_sidecar \
  --output-dir experiments/results
```

输出：

- `experiments/results/*_{baseline}.jsonl`：逐样本结果
- `experiments/results/*_summary.json`：汇总指标

核心指标：

- `asr_harmful_output_rate`：有害样本最终仍输出有害内容的比例（越低越好）
- `benign_refusal_rate`：无害样本被拒/代答比例（越低越好）
- `benign_helpful_rate`：无害样本中“安全且有用”比例（越高越好）

---

## 4. 策略阈值校准（论文主实验建议）

你可以外置校准配置，例如：

`experiments/policy_calibration.sample.json`

启用方式：

```bash
export GUARD_POLICY_CALIB_PATH="experiments/policy_calibration.sample.json"
```

然后跑 benchmark，即可应用该策略。

自动网格搜索：

```bash
python experiments/tune_policy_calibration.py \
  --dataset experiments/datasets/sample_guard_eval.jsonl \
  --difficulty normal \
  --judge-difficulty strict \
  --output-dir experiments/results/calibration
```

输出：

- `calibration_ranking.json`：所有配置排序
- `best_calibration.json`：最佳配置

---

## 5. GitHub 上传建议流程

```bash
git checkout -b feat/sidecar-ccfa-experiments
git add .
git commit -m "feat(experiments): add sidecar guard benchmark, calibration workflow, and research docs"
git push -u origin feat/sidecar-ccfa-experiments
```

建议 PR 标题：

`feat: sidecar guard research pipeline for CCF-A experiments`

---

## 6. 注意事项

1. 当前评测脚本默认使用本系统 `run_guardrails` 作为风险裁判，论文终稿建议再加：
   - 外部 moderation 模型裁判（如 Llama Guard / WildGuard）
   - 人工抽样复核
2. 若 API 配额紧张，先用 `--max-samples 100` 做小规模试跑。
3. 若要对比不同策略版本，保留每次运行的 `summary.json` 与配置文件。
