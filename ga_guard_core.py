import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

#######################################
# 配置区：按你的数据修改
#######################################

# 你的 jsonl 文件路径
JSONL_PATH = "./results/simply_Qwen_answer.jsonl"

# 文本字段名：比如 "question" 或 "prompt"
TEXT_FIELD = "response_with_qwen"      # 如果你字段叫 "prompt"，改成 "prompt"

# 标签字段名：比如 "label"
LABEL_FIELD = "label"

# 真实标签 -> 二分类 0/1 的映射
#   1 表示“违规”
#   0 表示“不违规”
# 如果你的 label 是中文，就改成 {"不违规": 0, "违规": 1} 这种
GOLD_LABEL_MAPPING = {
    "不违规": 0,
    "违规": 1,
}

# vLLM OpenAI 兼容接口配置
VLLM_API_BASE = "http://100.87.153.109:8001/v1"   # 记得带 /v1
VLLM_API_KEY  = "EMPTY"                      # 和 vLLM --api-key 一致
MODEL_NAME    = "GA_Guard_Core"

# 并发请求数：根据你机器情况调大/调小
MAX_CONCURRENT_REQUESTS = 60

#######################################
# GA Guard Core 输出类别定义
#######################################

# 模型输出的 slug -> 人类可读类别名
CATEGORY_SLUGS = {
    "illicit_activities": "Illicit Activities",
    "hate_and_abuse": "Hate & Abuse",
    "pii_and_ip": "PII & IP",
    "prompt_security": "Prompt Security",
    "sexual_content": "Sexual Content",
    "misinformation": "Misinformation",
    "violence_and_self_harm": "Violence & Self-Harm",
}

# 匹配 <hate_and_abuse_violation> / <hate_and_abuse_not_violation> 这种 token
TAG_PATTERN = re.compile(r"<([a-z_]+)_(violation|not_violation)>")

#######################################
# 初始化 OpenAI 客户端（指向 vLLM）
#######################################

client = OpenAI(
    base_url=VLLM_API_BASE,
    api_key=VLLM_API_KEY,
)

#######################################
# 工具函数
#######################################

def read_jsonl(path: str):
    """逐行读取 jsonl 文件，返回列表[dict, ...]"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def call_ga_guard_core(text: str, max_tokens: int = 64) -> str:
    """
    调 GA_Guard_Core 的 /v1/chat/completions 接口，
    返回原始字符串（包含 <..._violation> 这些标记）。
    """
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": text},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
        # 关键：不要跳过特殊 token，否则 <..._violation> 会被吃掉
        extra_body={
            "skip_special_tokens": False,
        },
    )
    return resp.choices[0].message.content


def parse_ga_guard_output(raw: str):
    """
    解析 GA Guard 输出，返回：
    - any_violation: bool，是否有任一类别违规
    - categories: dict[类别名 -> {violation: bool, raw_tag: str}]
    """
    matches = TAG_PATTERN.findall(raw or "")
    categories = {}

    for slug, status in matches:
        if slug not in CATEGORY_SLUGS:
            # 未来模型如果增加新类别，这里可以选择保留/打印日志
            continue
        name = CATEGORY_SLUGS[slug]
        is_violation = (status == "violation")
        categories[name] = {
            "violation": is_violation,
            "raw_tag": f"<{slug}_{status}>",
        }

    any_violation = any(info["violation"] for info in categories.values())

    return any_violation, categories


def predict_one(text: str):
    """
    对一条文本做审核，返回：
    - bin_label: 0/1 是否违规（只要任一类别 violation 即视为违规）
    - raw: 原始模型输出字符串
    - categories: 各类别细分结果
    """
    raw = call_ga_guard_core(text)
    any_violation, categories = parse_ga_guard_output(raw)
    bin_label = 1 if any_violation else 0
    # print(raw)
    # print(bin_label, categories)
    return bin_label, raw, categories


#######################################
# 主流程
#######################################

def main():
    jsonl_path = Path(JSONL_PATH)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到 jsonl 文件：{jsonl_path}")

    print("加载数据...")
    records = read_jsonl(str(jsonl_path))
    import random
    output_path = "./results/sampled_data_500.json"
    sample_size = 500
    selected_records = random.sample(records, sample_size)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in selected_records:
            # json.dumps 将字典转为字符串
            # ensure_ascii=False 保证中文不乱码
            line = json.dumps(record, ensure_ascii=False)
            # 写入文件并换行
            f.write(line + "\n")

    print(f"共读取 {len(records)} 条样本")

    texts = []
    y_true = []

    # 先把文本和真实标签整理出来
    for idx, item in enumerate(selected_records):
        if TEXT_FIELD not in item:
            raise KeyError(f"第 {idx} 条记录中找不到字段 '{TEXT_FIELD}'，请检查 TEXT_FIELD 设置")
        if LABEL_FIELD not in item:
            raise KeyError(f"第 {idx} 条记录中找不到字段 '{LABEL_FIELD}'，请检查 LABEL_FIELD 设置")

        text = item[TEXT_FIELD]
        gold_raw = item[LABEL_FIELD]

        if gold_raw not in GOLD_LABEL_MAPPING:
            raise ValueError(
                f"真实标签 {gold_raw!r} 不在 GOLD_LABEL_MAPPING 中，请修改 GOLD_LABEL_MAPPING"
            )

        gold = GOLD_LABEL_MAPPING[gold_raw]
        texts.append(text)
        y_true.append(gold)

    print("开始并行推理 GA_Guard_Core（通过 vLLM OpenAI 接口）...")

    y_pred = [None] * len(texts)
    detail_results = []  # 如果你想事后分析每条预测，可以用这个存详情

    # 线程池并发请求
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(predict_one, texts[i]): i
            for i in range(len(texts))
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                pred, raw, categories = future.result()
            except Exception as e:
                print(f"\n样本 {i} 推理失败: {e}")
                # 出错时当作不违规，避免整个评估中断；你也可以选择继续 raise
                pred = 0
                raw = ""
                categories = {}

            y_pred[i] = pred
            detail_results.append({
                "index": i,
                "text": texts[i],
                "pred_binary": pred,
                "raw_output": raw,
                "categories": categories,
            })

    # 安全检查
    assert None not in y_pred, "有样本没有生成预测，请检查并发逻辑"
    assert len(y_pred) == len(y_true), "预测数量和真实标签数量不一致！"

    print("\n====== GA_Guard_Core 二分类报告 (0: 不违规, 1: 违规) ======")
    print(classification_report(y_true, y_pred, digits=4))
    acc = accuracy_score(y_true, y_pred)
    print(f"整体准确率 ACC: {acc:.4f}")

    # 如果你想把详细预测结果保存下来，解开下面注释：
    # out_path = jsonl_path.with_suffix(".ga_guard_pred.jsonl")
    # with open(out_path, "w", encoding="utf-8") as f:
    #     for r in detail_results:
    #         f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # print(f\"详细预测结果已写入: {out_path}\")


if __name__ == "__main__":
    main()