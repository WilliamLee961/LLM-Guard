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

JSONL_PATH = "./results/results_en/Guard_FullPipeline_normal_20251215_123909.jsonl"

TEXT_FIELD = "question"
RESPONSE_FIELD = "response_with_qwen"   # 生成好的回复字段
LABEL_FIELD = "label"

GOLD_LABEL_MAPPING = {
    "不违规": 0,
    "违规": 1,
}

# vLLM OpenAI 兼容接口配置（审核模型）
VLLM_API_BASE = "http://100.87.153.109:8005/v1"
VLLM_API_KEY  = "EMPTY"
MODEL_NAME    = "models/GA_Guard_Core"

MAX_CONCURRENT_REQUESTS = 60

#######################################
# GA Guard Core 输出类别定义
#######################################

CATEGORY_SLUGS = {
    "illicit_activities": "Illicit Activities",
    "hate_and_abuse": "Hate & Abuse",
    "pii_and_ip": "PII & IP",
    "prompt_security": "Prompt Security",
    "sexual_content": "Sexual Content",
    "misinformation": "Misinformation",
    "violence_and_self_harm": "Violence & Self-Harm",
}

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
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def call_ga_guard_core(text: str, max_tokens: int = 64) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": text}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body={"skip_special_tokens": False},
    )
    return resp.choices[0].message.content


def parse_ga_guard_output(raw: str):
    matches = TAG_PATTERN.findall(raw or "")
    categories = {}

    for slug, status in matches:
        if slug not in CATEGORY_SLUGS:
            continue
        name = CATEGORY_SLUGS[slug]
        is_violation = (status == "violation")
        categories[name] = {
            "violation": is_violation,
            "raw_tag": f"<{slug}_{status}>",
        }

    any_violation = any(info["violation"] for info in categories.values())
    return any_violation, categories


def predict_guard(text: str):
    raw = call_ga_guard_core(text)
    any_violation, categories = parse_ga_guard_output(raw)
    bin_label = 1 if any_violation else 0
    return bin_label, raw, categories


def process_one(i: int, question: str, candidate_answer: str | None):
    """
    单条样本链路：
      1) 输入侧审核 question
      2) 若输入侧不违规：对 candidate_answer(来自 response_with_qwen) 做输出侧审核
      3) 仅当输入侧不违规 且 输出侧不违规：final_answer=candidate_answer
    """
    res = {
        "index": i,
        "question": question,

        # 输入侧
        "input_pred_binary": None,
        "input_raw_output": None,
        "input_categories": None,

        # 候选回答（来自文件）
        "candidate_answer": candidate_answer if candidate_answer else None,

        # 输出侧
        "output_pred_binary": None,
        "output_raw_output": None,
        "output_categories": None,

        # 最终输出
        "final_answer": None,
    }

    # ---------- 输入侧审核 ----------
    in_pred, in_raw, in_cats = predict_guard(question)
    res["input_pred_binary"] = in_pred
    res["input_raw_output"] = in_raw
    res["input_categories"] = in_cats

    # 若输入侧违规：输出侧全部 None（按你的要求）
    if in_pred == 1:
        return res

    # 输入侧不违规，但候选回答为空/缺失：输出侧也保持 None
    if not res["candidate_answer"]:
        return res

    # ---------- 输出侧审核 ----------
    out_pred, out_raw, out_cats = predict_guard(res["candidate_answer"])
    res["output_pred_binary"] = out_pred
    res["output_raw_output"] = out_raw
    res["output_categories"] = out_cats

    # ---------- 最终答案 ----------
    if out_pred == 0:
        res["final_answer"] = res["candidate_answer"]

    return res


#######################################
# 主流程
#######################################

def main():
    jsonl_path = Path(JSONL_PATH)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"找不到 jsonl 文件：{jsonl_path}")

    print("加载数据...")
    records = read_jsonl(str(jsonl_path))
    print(f"共读取 {len(records)} 条样本")

    questions = []
    candidates = []
    y_true = []
    eval_records = []

    for idx, item in enumerate(records):
        if TEXT_FIELD not in item:
            raise KeyError(f"第 {idx} 条记录中找不到字段 '{TEXT_FIELD}'，请检查 TEXT_FIELD 设置")
        if LABEL_FIELD not in item:
            raise KeyError(f"第 {idx} 条记录中找不到字段 '{LABEL_FIELD}'，请检查 LABEL_FIELD 设置")

        q = (item.get(TEXT_FIELD) or "").strip()
        if not q:
            continue

        gold_raw = (item.get(LABEL_FIELD) or "").strip()
        if gold_raw not in GOLD_LABEL_MAPPING:
            raise ValueError(f"真实标签 {gold_raw!r} 不在 GOLD_LABEL_MAPPING 中，请修改 GOLD_LABEL_MAPPING")

        # 候选回答允许缺失，用 None 补全
        cand = item.get(RESPONSE_FIELD)
        if isinstance(cand, str):
            cand = cand.strip()
        else:
            cand = None

        questions.append(q)
        candidates.append(cand if cand else None)
        y_true.append(GOLD_LABEL_MAPPING[gold_raw])

        rec = dict(item)
        rec["_eval_idx"] = idx
        rec["_eval_question"] = q
        eval_records.append(rec)

    print(f"有效样本数：{len(questions)}")
    print("开始并行执行：输入侧审核 -> (通过则)输出侧审核 ...")

    # 与你之前脚本一致：二分类评测仍以“输入侧审核结果”作为 y_pred
    y_pred = [None] * len(questions)
    detail_results = [None] * len(questions)

    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_REQUESTS) as executor:
        futures = {
            executor.submit(process_one, i, questions[i], candidates[i]): i
            for i in range(len(questions))
        }

        for future in tqdm(as_completed(futures), total=len(futures)):
            i = futures[future]
            try:
                r = future.result()
            except Exception as e:
                print(f"\n样本 {i} 处理失败: {e}")
                r = {
                    "index": i,
                    "question": questions[i],
                    "input_pred_binary": 0,
                    "input_raw_output": None,
                    "input_categories": None,
                    "candidate_answer": candidates[i] if candidates[i] else None,
                    "output_pred_binary": None,
                    "output_raw_output": None,
                    "output_categories": None,
                    "final_answer": None,
                }

            detail_results[i] = r
            y_pred[i] = int(r["input_pred_binary"] or 0)

    assert None not in y_pred, "有样本没有生成预测，请检查并发逻辑"
    assert len(y_pred) == len(y_true), "预测数量和真实标签数量不一致！"

    print("\n====== 输入侧(GA_Guard_Core) 二分类报告 (0: 不违规, 1: 违规) ======")
    print(classification_report(y_true, y_pred, digits=4))
    acc = accuracy_score(y_true, y_pred)
    print(f"整体准确率 ACC: {acc:.4f}")

    # 如需保存每条完整链路结果（含 input/output tag 与 final_answer），打开下面注释：
    new_stem = jsonl_path.stem + "_inout_guard_with_file_resp"  # 用下划线拼接，无点
    out_path = jsonl_path.with_stem(new_stem)  # 保留原后缀 .jsonl
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(detail_results):
            rec = dict(eval_records[i])
            rec.update({
                "_input_pred_binary": r["input_pred_binary"],
                "_input_raw_output": r["input_raw_output"],
                "_input_categories": r["input_categories"],
                "_candidate_answer": r["candidate_answer"],
                "_output_pred_binary": r["output_pred_binary"],
                "_output_raw_output": r["output_raw_output"],
                "_output_categories": r["output_categories"],
                "_final_answer": r["final_answer"],
            })
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[INFO] 完整链路结果已保存到: {out_path}")


if __name__ == "__main__":
    main()