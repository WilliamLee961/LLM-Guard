import sys
import os
import json
from collections import OrderedDict

# 添加 MY_GUARD 根目录到 Python 路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from llm_client import llm_chat

INPUT_FILE = "/home/lijiahang/MY_GUARD/datasets/typical_safety_scenarios_final.jsonl"
OUTPUT_FILE = "/home/lijiahang/MY_GUARD/results/simply_Qwen_answer.jsonl"

SYSTEM_PROMPT = "你是一个智能助手，请根据用户的问题生成合理的回答。"

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as fin, \
         open(OUTPUT_FILE, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except Exception as e:
                print(f"跳过无法解析的行: {line[:50]}... 原因: {e}")
                continue

            question = data.get("question", "").strip()
            if not question:
                print("没有找到 question，跳过")
                continue

            # 调用 Qwen 生成新回答
            qwen_answer = llm_chat(SYSTEM_PROMPT, question)

            # 构造一个有序字典，保证 "response_with_qwen" 在 "label" 后
            ordered_data = OrderedDict()
            for k, v in data.items():
                ordered_data[k] = v
                if k == "label":
                    ordered_data["response_with_qwen"] = qwen_answer

            fout.write(json.dumps(ordered_data, ensure_ascii=False) + "\n")

    print(f"处理完成，结果已保存到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
