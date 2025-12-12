import sys
import os
import json
import argparse
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

"""
职责：干活。它负责跑通 Question -> InputGuard -> Model -> OutputGuard -> Rewrite 这一整套流程。

产出：它生产出新的数据（即 _after_guard.jsonl 文件），在这个文件里，你既有原始回答，也有由于 Guard 介入而生成的安全回答。

定位：这是你的生产线。
"""
# 设置 CUDA 设备 (在 import torch 之前)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import datetime
# ==============================================================================
# 1. 导入复用模块 (REUSE)
# ==============================================================================

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    # [REUSE] 导入模块化的 SafetyGuard
    # 确保 safety_guard.py 已经存在且包含 SafetyGuard 类
    from safety_guard import SafetyGuard
    
except ImportError as e:
    print(f"【环境错误】无法导入 SafetyGuard: {e}")
    print("请确认 safety_guard.py, guard_graph.py, routing_hub.py, risk_hub.py, safe_qa_graph.py 在同一目录下。")
    sys.exit(1)

# ==============================================================================
# 2. 配置与业务逻辑
# ==============================================================================

# 默认配置
DEFAULT_INPUT_PATH = "/home/lijiahang/MY_GUARD/results/results_en/simply_Qwen_answer_strict_total.jsonl"
DEFAULT_DETECT_MODE = "normal"
DEFAULT_TARGET_FIELD = "response_with_qwen" 
MAX_WORKERS = 50

def parse_args():
    parser = argparse.ArgumentParser(description="Guard 治理模式 v3：全流程模块化防护 (Input -> Model -> Output)")
    
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_INPUT_PATH,
                        help="输入文件路径")
    
    parser.add_argument("--target_field", type=str, default=DEFAULT_TARGET_FIELD,
                        help="JSON 中包含模型回复的字段名 (用于模拟模型输出)")

    parser.add_argument("--output_path", type=str, default=None, 
                        help="输出路径。如果不填，自动生成。")
    
    parser.add_argument("--detect_mode", type=str, default=DEFAULT_DETECT_MODE, 
                        choices=["lenient", "normal", "strict"],
                        help="风险判定严格程度")
    
    # [新增] 开关控制
    parser.add_argument("--enable_input_guard", action="store_true", default=True,
                        help="是否开启输入端防护 (检查问题)")
    parser.add_argument("--disable_input_guard", action="store_false", dest="enable_input_guard",
                        help="关闭输入端防护")
                        
    parser.add_argument("--enable_output_guard", action="store_true", default=True,
                        help="是否开启输出端防护 (检查回答)")
    parser.add_argument("--disable_output_guard", action="store_false", dest="enable_output_guard",
                        help="关闭输出端防护")
    
    # RAG 开关 (建议默认关闭以提速，除非需要高质量重写)
    parser.add_argument("--use_rag", action="store_true", default=False,
                        help="在生成安全回复时是否使用 RAG (开启会变慢但质量更高)")

    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    
    return parser.parse_args()


def process_line_full_pipeline(line_data: dict, target_field: str, guard: SafetyGuard) -> dict:
    """
    单行处理逻辑：模拟 Question -> InputGuard -> Model -> OutputGuard -> FinalResponse
    """
    
    user_query = line_data.get("question", "")
    # 这里我们用文件中已有的模型回答来“模拟”模型刚刚生成的内容
    raw_model_response = line_data.get(target_field, "")
    
    final_response = raw_model_response
    guard_stage = "PASS"      # 最终在哪个阶段触发了拦截 (INPUT / OUTPUT / PASS)
    guard_action = "pass"     # 具体动作
    risk_info = "NONE"        # 风险等级
    error_msg = None

    try:
        # =================================================
        # Stage 1: Input Guard (前置防护)
        # =================================================
        # 即使这里是离线清洗，检查输入也有意义：如果问题本身就违规，
        # 我们应该直接替换成拒答，而不是让模型原来的回答（即使它可能也是拒答）留着。
        
        is_safe_in, content_in, meta_in = guard.check_input(user_query)
        
        if not is_safe_in:
            # 💥 输入违规 -> 直接拦截
            final_response = content_in  # 这里的 content_in 是安全代答
            guard_stage = "INPUT"
            guard_action = meta_in.get("action", "blocked_input")
            risk_info = str(meta_in.get("risk_profile", {}).get("overall_risk_level", "Unknown"))
            
        else:
            # ✅ 输入安全 -> 进入模型生成阶段
            # (模拟) model_response = raw_model_response 
            
            # =================================================
            # Stage 2: Output Guard (后置防护)
            # =================================================
            # 只有当输入安全（没有被前置拦截）时，才需要检查输出
            
            is_safe_out, content_out, meta_out = guard.check_output(user_query, raw_model_response)
            
            if not is_safe_out:
                # 💥 输出违规 -> 进行重写
                final_response = content_out # 这里的 content_out 是重写后的安全回复
                guard_stage = "OUTPUT"
                guard_action = meta_out.get("action", "rewrote_output")
                risk_info = str(meta_out.get("risk_profile", {}).get("overall_risk_level", "Unknown"))
            else:
                # ✅ 输出也安全 -> 放行
                final_response = raw_model_response
                guard_stage = "PASS"
                guard_action = "pass"
                risk_info = str(meta_out.get("risk_profile", {}).get("overall_risk_level", "L1"))

    except Exception as e:
        # 错误处理
        print(f"[Line Error] {e}")
        traceback.print_exc()
        guard_stage = "ERROR"
        error_msg = str(e)
        # 出错时保留原文，防止数据丢失
        final_response = raw_model_response

    # =================================================
    # 组装结果
    # =================================================
    out_data = OrderedDict()
    output_field_name = f"{target_field}_after_guard"
    inserted = False
    
    for k, v in line_data.items():
        out_data[k] = v
        if k == target_field:
            out_data[output_field_name] = final_response
            out_data["_guard_stage"] = guard_stage
            out_data["_guard_action"] = guard_action
            out_data["_guard_risk"] = risk_info
            if error_msg:
                out_data["_guard_error"] = error_msg
            inserted = True
            
    if not inserted:
        out_data[output_field_name] = final_response
        out_data["_guard_stage"] = guard_stage
        out_data["_guard_risk"] = risk_info

    return out_data


# ==============================================================================
# 3. 主流程
# ==============================================================================

def main():
    args = parse_args()
    
    if not os.path.exists(args.jsonl_path):
        print(f"Error: Input file not found: {args.jsonl_path}")
        return

    # --- 动态生成 output_path ---
    if args.output_path is None:
        base_dir = os.path.dirname(args.jsonl_path)
        if not base_dir: base_dir = "."

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"Guard_FullPipeline_{args.detect_mode}_{timestamp}.jsonl"
        args.output_path = os.path.join(base_dir, filename)
        
        print(f"[Info] 未指定输出路径，自动生成: {args.output_path}")

    # --- 初始化 SafetyGuard ---
    # 这里通过 SafetyGuard 的 enabled 参数，虽然 SafetyGuard 类本身可能没有区分 input/output enabled，
    # 但我们可以在 process_line 里逻辑控制，或者 SafetyGuard 类够强的话可以传进去。
    # 为了简单复用上一版的 SafetyGuard，我们实例化一个全开的 Guard，
    # 然后在 process_line 中利用 args.enable_input_guard 等参数来决定是否调用 check_input/check_output。
    
    # 注意：为了支持多线程，SafetyGuard 最好是线程安全的或者在主线程初始化后只读。
    # 由于 SafetyGuard 内部持有 graph 对象，通常是线程安全的 (LangGraph stateless invocation)。
    
    print(f"Initializing SafetyGuard (Mode: {args.detect_mode}, RAG: {args.use_rag})...")
    
    # 实例化全局 Guard 对象
    global_guard = SafetyGuard(
        detect_mode=args.detect_mode, 
        use_rag=args.use_rag, 
        enabled=True # 总开关开启，具体 Input/Output 开关我们在逻辑里控制
    )
    
    # 临时打补丁：如果 SafetyGuard 类支持动态开关最好，如果不支持，我们可以在这里 wrapper 一下
    # 或者简单点，我们把 args 传给 process 函数，那里做 if 判断。
    # 为了代码整洁，我们假设 SafetyGuard 是全功能的，我们在 process_line 里做 `if args.enable_input_guard: guard.check_input(...)` 的判断。
    
    # 但是！为了复用你之前要求的“在每个guard的位置保留是否应用guard的开关”，
    # 我将修改 SafetyGuard 的 check 方法，或者在 process_line 里写逻辑。
    # 让我们在 process_line 里写逻辑，这样最灵活。

    print(f"Pipeline Config: InputGuard={args.enable_input_guard}, OutputGuard={args.enable_output_guard}")
    print(f"Processing data from: {args.jsonl_path}")

    lines = []
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    
    results = [None] * len(lines)
    
    # 并发处理
    # 注意：我们需要一个 wrapper 函数把 args 和 guard 传进去
    def process_wrapper(line):
        # 这里的 guard 是闭包引用的 global_guard
        # 为了实现开关，我们需要在这里动态调整 guard 的行为，或者手动跳过
        
        # 技巧：我们创建一个临时的 proxy 或者直接在 process_line_full_pipeline 里判断
        # 这里我修改 process_line_full_pipeline 签名，传入开关参数是不太优雅的，
        # 不如直接在这里手动控制。
        
        # 更好的方式：SafetyGuard 类本身应该已经有了 check_input / check_output。
        # 如果 args.enable_input_guard 为 False，我们就不调 guard.check_input，直接当做 True (Safe)。
        
        # 让我们微调一下 process_line_full_pipeline 的逻辑，把 args 传进去
        return process_line_with_switches(line, args.target_field, global_guard, 
                                          args.enable_input_guard, args.enable_output_guard)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(process_wrapper, line): i 
            for i, line in enumerate(lines)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(lines), desc="Guard Pipeline"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Line {idx} failed: {e}")
                results[idx] = lines[idx] 

    # 保存结果
    print(f"Saving results to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for res in results:
            if res:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    print("Done.")


def process_line_with_switches(line_data: dict, target_field: str, guard: SafetyGuard, 
                               enable_input: bool, enable_output: bool) -> dict:
    """
    带有显式开关控制的行处理函数
    """
    user_query = line_data.get("question", "")
    raw_model_response = line_data.get(target_field, "")
    
    final_response = raw_model_response
    guard_stage = "PASS"
    guard_action = "pass"
    risk_info = "NONE"
    
    try:
        # --- Stage 1: Input Guard ---
        input_blocked = False
        if enable_input:
            is_safe_in, content_in, meta_in = guard.check_input(user_query)
            if not is_safe_in:
                # 拦截
                final_response = content_in
                guard_stage = "INPUT"
                guard_action = meta_in.get("action", "blocked_input")
                risk_info = str(meta_in.get("risk_profile", {}).get("overall_risk_level", "Unknown"))
                input_blocked = True
        
        # --- Stage 2: Output Guard ---
        # 只有在输入未被拦截的情况下，才检查输出
        if not input_blocked and enable_output:
            is_safe_out, content_out, meta_out = guard.check_output(user_query, raw_model_response)
            if not is_safe_out:
                # 拦截
                final_response = content_out
                guard_stage = "OUTPUT"
                guard_action = meta_out.get("action", "rewrote_output")
                risk_info = str(meta_out.get("risk_profile", {}).get("overall_risk_level", "Unknown"))
            elif guard_stage == "PASS": # 只有在没被Input拦截且Output也安全时，才是真正的PASS
                 risk_info = str(meta_out.get("risk_profile", {}).get("overall_risk_level", "L1"))


    except Exception as e:
        guard_stage = "ERROR"
        risk_info = str(e)
        # traceback.print_exc()

    # 组装结果 (同上)
    out_data = OrderedDict()
    output_field_name = f"{target_field}_after_guard"
    inserted = False
    for k, v in line_data.items():
        out_data[k] = v
        if k == target_field:
            out_data[output_field_name] = final_response
            out_data["_guard_stage"] = guard_stage
            out_data["_guard_action"] = guard_action
            out_data["_guard_risk"] = risk_info
            inserted = True
    if not inserted:
        out_data[output_field_name] = final_response
        out_data["_guard_stage"] = guard_stage
        out_data["_guard_risk"] = risk_info
        
    return out_data

def main():
    args = parse_args()
    
    if not os.path.exists(args.jsonl_path):
        print(f"Error: Input file not found: {args.jsonl_path}")
        return

    # --- 动态生成 output_path (带时间戳) ---
    if args.output_path is None:
        base_dir = os.path.dirname(args.jsonl_path)
        if not base_dir: base_dir = "."
        
        # [修改点] 获取当前时间，精确到秒
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 拼接文件名：Guard_FullPipeline_{模式}_{时间戳}.jsonl
        filename = f"Guard_FullPipeline_{args.detect_mode}_{timestamp}.jsonl"
        args.output_path = os.path.join(base_dir, filename)
        
        print(f"[Info] 未指定输出路径，自动生成: {args.output_path}")

    print(f"Initializing SafetyGuard (Mode: {args.detect_mode}, RAG: {args.use_rag})...")
    
    # 实例化全局 Guard 对象
    global_guard = SafetyGuard(
        detect_mode=args.detect_mode, 
        use_rag=args.use_rag, 
        enabled=True
    )
    
    print(f"Pipeline Config: InputGuard={args.enable_input_guard}, OutputGuard={args.enable_output_guard}")
    print(f"Processing data from: {args.jsonl_path}")

    lines = []
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    
    results = [None] * len(lines)
    
    # Wrapper 函数
    def process_wrapper(line):
        return process_line_full_pipeline(
            line, 
            args.target_field, 
            global_guard, 
            args.enable_input_guard, 
            args.enable_output_guard
        )

    # 并发处理
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_idx = {
            executor.submit(process_wrapper, line): i 
            for i, line in enumerate(lines)
        }
        
        for future in tqdm(as_completed(future_to_idx), total=len(lines), desc="Guard Pipeline"):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                print(f"Line {idx} failed: {e}")
                results[idx] = lines[idx] 

    # 保存结果
    print(f"Saving results to: {args.output_path}")
    with open(args.output_path, "w", encoding="utf-8") as f:
        for res in results:
            if res:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
    
    print("Done.")
    
if __name__ == "__main__":
    main()