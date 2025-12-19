import sys
import os
import json
import argparse
import traceback
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from datetime import datetime
import faiss

"""
调用方法：python apply_guard_to_pipeline.py   --input_detect_mode strict   --output_detect_mode normal   --enable_input_guard   --enable_output_guard
"""
"""
职责：干活。它负责跑通 Question -> InputGuard -> Model -> OutputGuard -> Rewrite 这一整套流程。
支持针对 Input 和 Output 分别设置不同的检测粒度（strict, normal, lenient）。
"""

# 设置 CUDA 设备
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# ==============================================================================
# 1. 导入复用模块
# ==============================================================================

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

try:
    from safety_guard import SafetyGuard
except ImportError as e:
    print(f"【环境错误】无法导入 SafetyGuard: {e}")
    sys.exit(1)

# ==============================================================================
# 2. 配置与业务逻辑
# ==============================================================================

DEFAULT_INPUT_PATH = "/home/lijiahang/MY_GUARD/results/results_en/Guard_FullPipeline_normal_20251215_123909.jsonl"
DEFAULT_DETECT_MODE = "normal"
DEFAULT_TARGET_FIELD = "response_with_qwen" 
MAX_WORKERS = 32

def parse_args():
    parser = argparse.ArgumentParser(description="Guard 治理模式 v3.1: 独立粒度全流程防护")
    
    parser.add_argument("--jsonl_path", type=str, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--target_field", type=str, default=DEFAULT_TARGET_FIELD,
                        help="包含模型回复的字段名")
    parser.add_argument("--output_path", type=str, default=None)
    
    # [核心修改] 基础模式 + 独立模式配置
    parser.add_argument("--detect_mode", type=str, default=DEFAULT_DETECT_MODE, 
                        choices=["lenient", "normal", "strict"],
                        help="默认风险判定严格程度 (作为基准)")
    
    parser.add_argument("--input_detect_mode", type=str, default=None,
                        choices=["lenient", "normal", "strict"],
                        help="Input Guard 的严格程度 (默认跟随 detect_mode)")
    
    parser.add_argument("--output_detect_mode", type=str, default=None,
                        choices=["lenient", "normal", "strict"],
                        help="Output Guard 的严格程度 (默认跟随 detect_mode)")
    
    # 开关控制
    parser.add_argument("--enable_input_guard", action="store_true", default=True)
    parser.add_argument("--disable_input_guard", action="store_false", dest="enable_input_guard")
                        
    parser.add_argument("--enable_output_guard", action="store_true", default=True)
    parser.add_argument("--disable_output_guard", action="store_false", dest="enable_output_guard")
    
    parser.add_argument("--use_rag", action="store_true", default=False)
    parser.add_argument("--max_workers", type=int, default=MAX_WORKERS)
    
    return parser.parse_args()


def process_line_full_pipeline(line_data: dict, target_field: str, 
                               input_guard: SafetyGuard, output_guard: SafetyGuard, 
                               enable_input: bool, enable_output: bool) -> dict:
    """
    单行处理逻辑：支持传入两个不同的 Guard 实例
    """
    
    user_query = line_data.get("question", "")
    raw_model_response = line_data.get(target_field, "")
    
    final_response = raw_model_response
    guard_stage = "PASS"
    guard_action = "pass"
    risk_info = "NONE"
    error_msg = None

    try:
        # =================================================
        # Stage 1: Input Guard (使用 input_guard 实例)
        # =================================================
        input_blocked = False
        
        if enable_input:
            # 使用 input_guard 检查
            is_safe_in, content_in, meta_in = input_guard.check_input(user_query)
            
            if not is_safe_in:
                # 💥 输入违规 -> 拦截
                final_response = content_in
                guard_stage = "INPUT"
                guard_action = meta_in.get("action", "blocked_input")
                risk_info = str(meta_in.get("risk_profile", {}).get("overall_risk_level", "Unknown"))
                input_blocked = True
        
        # =================================================
        # Stage 2: Output Guard (使用 output_guard 实例)
        # =================================================
        # 只有当输入未被拦截时，才检查输出
        if not input_blocked and enable_output:
            # 使用 output_guard 检查
            is_safe_out, content_out, meta_out = output_guard.check_output(user_query, raw_model_response)
            
            if not is_safe_out:
                # 💥 输出违规 -> 重写
                final_response = content_out
                guard_stage = "OUTPUT"
                guard_action = meta_out.get("action", "rewrote_output")
                risk_info = str(meta_out.get("risk_profile", {}).get("overall_risk_level", "Unknown"))
            elif guard_stage == "PASS":
                # 只有全程通过才记录为 PASS 的 risk
                risk_info = str(meta_out.get("risk_profile", {}).get("overall_risk_level", "L1"))

    except Exception as e:
        # [优化] 捕获具体的 NameError 错误并提供更清晰的指引
        err_str = str(e)
        if "name 'faiss' is not defined" in err_str:
            print(f"[Line Critical Error] 运行时找不到 'faiss'。这通常意味着 safety_guard.py 中导入失败或未导入。")
        else:
            print(f"[Line Error] {e}")
            
        guard_stage = "ERROR"
        error_msg = err_str
        final_response = raw_model_response # Fallback

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

    # [新增] 环境依赖预检查 (Fail Fast)
    if args.use_rag:
        print("-" * 40)
        print("[System] 正在检查 RAG 核心依赖 (faiss)...")
        try:
            import faiss
            # 打印路径以确认加载了正确的库
            print(f"[System] ✅ faiss 模块加载成功。")
            if hasattr(faiss, '__file__'):
                print(f"[System] 库路径: {os.path.dirname(faiss.__file__)}")
            if hasattr(faiss, 'get_num_gpus'):
                print(f"[System] GPU 支持: {faiss.get_num_gpus()} GPUs")
        except ImportError as e:
            print(f"【无法启动】检测到 --use_rag，但当前 Python 环境无法导入 'faiss'。")
            print(f"错误详情: {e}")
            print("解决建议: 请运行 `pip install faiss-gpu` (Linux/CUDA) 或 `pip install faiss-cpu` (Mac/No-GPU)。")
            print("注意: 不要只安装 `pip install faiss` (这通常是无效的空包)。")
            sys.exit(1) # 强制退出，不要带着病跑
        except Exception as e:
            print(f"【环境异常】检查 faiss 时发生未知错误: {e}")
            sys.exit(1)
        print("-" * 40)

    # --- 1. 确定 Input 和 Output 的模式 ---
    # 如果没单独指定 input/output mode，就用全局的 detect_mode
    in_mode = args.input_detect_mode if args.input_detect_mode else args.detect_mode
    out_mode = args.output_detect_mode if args.output_detect_mode else args.detect_mode

    # --- 2. 动态生成文件名 ---
    if args.output_path is None:
        base_dir = os.path.dirname(args.jsonl_path)
        if not base_dir: base_dir = "."
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # [修改点] 文件名体现双模式配置，且体现是否禁用
        # 如果对应的 Guard 被禁用，模式名替换为 "None"
        file_in_mode = in_mode if args.enable_input_guard else "None"
        file_out_mode = out_mode if args.enable_output_guard else "None"
        

        mode_str = f"in_{file_in_mode}_out_{file_out_mode}_rag_{args.use_rag}"
            
        filename = f"Guard_FullPipeline_{mode_str}_{timestamp}.jsonl"
        args.output_path = os.path.join(base_dir, filename)
        print(f"[Info] 自动生成输出路径: {args.output_path}")

    # --- 3. 初始化 Guard 实例 ---
    print(f"Initializing SafetyGuards (Input Mode: {in_mode}, Output Mode: {out_mode}, RAG: {args.use_rag})...")
    
    input_guard = None
    output_guard = None

    # 优化资源：如果模式相同，只实例化一个 Guard 对象复用
    # 注意：这里依然使用 in_mode/out_mode 来初始化对象，即使该 Guard 可能被禁用。
    # 禁用逻辑在 process_line 中通过 enable_input/output 开关控制。
    if in_mode == out_mode:
        print(f"Modes are identical ({in_mode}). Sharing single SafetyGuard instance.")
        shared_guard = SafetyGuard(detect_mode=in_mode, use_rag=args.use_rag, enabled=True)
        input_guard = shared_guard
        output_guard = shared_guard
    else:
        print(f"Modes differ. Creating separate SafetyGuard instances.")
        print(f"  -> Creating Input Guard ({in_mode})...")
        input_guard = SafetyGuard(detect_mode=in_mode, use_rag=args.use_rag, enabled=True)
        
        print(f"  -> Creating Output Guard ({out_mode})...")
        output_guard = SafetyGuard(detect_mode=out_mode, use_rag=args.use_rag, enabled=True)
    
    print(f"Pipeline Config: InputGuard={args.enable_input_guard} [{in_mode}], OutputGuard={args.enable_output_guard} [{out_mode}]")
    print(f"Processing data from: {args.jsonl_path}")

    lines = []
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                lines.append(json.loads(line))
    
    results = [None] * len(lines)
    
    # --- 4. 包装处理函数 ---
    def process_wrapper(line):
        return process_line_full_pipeline(
            line_data=line, 
            target_field=args.target_field, 
            input_guard=input_guard,   # 传入专门的 Input Guard
            output_guard=output_guard, # 传入专门的 Output Guard
            enable_input=args.enable_input_guard, 
            enable_output=args.enable_output_guard
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