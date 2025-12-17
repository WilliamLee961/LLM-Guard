import sys
import os
import json
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import random

# 添加 MY_GUARD 根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_client import llm_chat

INPUT_FILE = "/home/lijiahang/MY_GUARD/datasets/base_set_final.jsonl"
OUTPUT_FILE = "/home/lijiahang/MY_GUARD/results/results_base_set/simply_Qwen_answer.jsonl"

SYSTEM_PROMPT = "你是一个智能助手，请根据用户的问题生成合理的回答。"

# 根据显卡和任务调整，比如 20-50，如果太高可能内存爆或显存占满
MAX_WORKERS = 50  

def process_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        data = json.loads(line)
    except Exception as e:
        return None

    question = data.get("question", "").strip()
    if not question:
        return None

    qwen_answer = llm_chat(SYSTEM_PROMPT, question)

    ordered_data = OrderedDict()
    for k, v in data.items():
        ordered_data[k] = v
        if k == "label":
            ordered_data["response_with_qwen"] = qwen_answer

    return json.dumps(ordered_data, ensure_ascii=False)

# def main():
#     with open(INPUT_FILE, "r", encoding="utf-8") as fin:
#         lines = fin.readlines()

#     results = []
#     with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
#         futures = {executor.submit(process_line, line): line for line in lines}
#         for future in tqdm(as_completed(futures), total=len(lines), desc="处理进度", unit="条"):
#             result = future.result()
#             if result:
#                 results.append(result)

#     # 写结果
#     with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
#         for r in results:
#             fout.write(r + "\n")

#     print(f"处理完成，结果已保存到 {OUTPUT_FILE}")

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin:
        lines = fin.readlines()

    # # 测试阶段只处理前 500 条
    # random.shuffle(lines)
    # lines = lines[:500]
    # print(f"已随机抽取 500 条数据，开始处理...")

    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_line, line): line for line in lines}
        for future in tqdm(as_completed(futures), total=len(lines), desc="处理进度", unit="条"):
            result = future.result()
            if result:
                results.append(result)

    # 写结果
    output_dir = os.path.dirname(OUTPUT_FILE)
    os.makedirs(output_dir, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as fout:
        for r in results:
            fout.write(r + "\n")

    print(f"处理完成，结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
