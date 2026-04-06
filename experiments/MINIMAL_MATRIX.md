# 工程对比实验最小集合（4x6）

这个最小集合直接对齐当前 sidecar 评测脚本 `experiments/run_guard_benchmark.py`。

## 1) Guard 方法（4 个）

1. `no_guard`
2. `input_only`
3. `output_only`
4. `full_sidecar`

这四个 baseline 已在脚本中原生支持，无需改代码。

## 2) 数据集集合（6 个）

统一本地文件格式：JSONL，每行字段至少包含：

```json
{"id":"x-001","query":"...","is_harmful":true,"category":"..."}
```

建议放在 `experiments/datasets/` 下，文件名如下：

1. `jbb_mini.jsonl`（JailbreakBench 切片）
2. `harmbench_mini.jsonl`（HarmBench 切片）
3. `xstest_mini.jsonl`（XSTest 切片，重点看过拒）
4. `wildguard_mini.jsonl`（WildGuard 风格切片）
5. `toxicchat_mini.jsonl`（ToxicChat 切片）
6. `biz_benign_cn_mini.jsonl`（你的中文业务 benign 集）

建议每个切片先做 `200~500` 条用于快速迭代，后续扩到 `1k+`。

### 2.1 公开集一键准备（前 5 个）

新增脚本：`experiments/prepare_public_datasets.py`  
可自动下载并转换这 5 个公开集：

1. `jbb_mini.jsonl`
2. `harmbench_mini.jsonl`
3. `xstest_mini.jsonl`
4. `wildguard_mini.jsonl`
5. `toxicchat_mini.jsonl`

运行：

```bash
python experiments/prepare_public_datasets.py \
  --output-dir experiments/datasets \
  --max-samples-per-dataset 400 \
  --harmful-ratio 0.5
```

说明：`biz_benign_cn_mini.jsonl` 仍需要你提供本地中文业务集（脚本不会自动生成）。

## 3) 统一指标（直接复用现有 summary）

`run_guard_benchmark.py` 已输出以下核心指标：

1. `asr_harmful_output_rate`（有害样本越狱成功率，越低越好）
2. `harmful_blocked_rate`（有害样本被拦截比例，越高越好）
3. `benign_refusal_rate`（无害样本过拒率，越低越好）
4. `benign_helpful_rate`（无害样本有用回答率，越高越好）
5. `avg_latency_ms` / `p95_latency_ms`（时延）

## 4) 一键跑 4x6 矩阵

新增脚本：`experiments/run_minimal_matrix.py`

### macOS / Linux

```bash
python experiments/run_minimal_matrix.py \
  --dataset-dir experiments/datasets \
  --output-dir experiments/results/minimal_matrix \
  --difficulty normal \
  --judge-difficulty strict \
  --max-samples 300
```

### Windows PowerShell

```powershell
python experiments/run_minimal_matrix.py `
  --dataset-dir experiments/datasets `
  --output-dir experiments/results/minimal_matrix `
  --difficulty normal `
  --judge-difficulty strict `
  --max-samples 300
```

输出：

1. 每个数据集各自的 benchmark JSONL 和 summary JSON
2. 汇总表 `experiments/results/minimal_matrix/minimal_matrix_summary.csv`

## 5) 没准备齐 6 个数据集时的临时跑法

先用样例集占位，验证流程通路：

```bash
python experiments/run_minimal_matrix.py --allow-sample-fallback --max-samples 50
```

脚本会对缺失数据集自动回退到 `sample_guard_eval.jsonl`，用于联调，不用于论文结论。
