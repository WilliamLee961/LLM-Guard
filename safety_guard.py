# safety_guard.py
import logging
from typing import Dict, Any, Optional, Tuple

# 导入你现有的核心组件
# 确保 guard_graph.py, safe_qa_graph.py, routing_hub.py, risk_hub.py 在同一目录下
from guard_graph import build_guard_graph, GuardState
from safe_qa_graph import safe_answer_node, QAState
from routing_hub import analyze_query
from risk_hub import detect_risk_discrete
from response_policy import (
    render_fallback_safe_response,
    rewrite_query_for_safe_mode,
    select_response_action,
)

class SafetyGuard:
    """
    安全围栏控制器 (Safety Guard Controller)
    
    功能：
    1. 统一封装输入(Input)和输出(Output)的风险检测逻辑。
    2. 提供开关机制 (enabled)，方便在评估时一键关闭。
    3. 集成安全代答 (SafeQA)，当检测到违规时自动重写内容。
    """

    def __init__(
        self,
        detect_mode: str = "normal",
        use_rag: bool = False,
        enabled: bool = True,
        runtime_mode: str = "accurate",
    ):
        """
        初始化安全围栏。

        参数:
            detect_mode (str): 检测严格程度 ('lenient', 'normal', 'strict')。
                               - strict: 宁可错杀，不可放过。
                               - normal: 平衡模式。
            use_rag (bool):    生成安全代答时，是否使用 RAG 检索安全知识库。
                               - True: 回答更专业，但消耗显存和时间。
                               - False: 纯 LLM 生成，速度快，适合批量处理。
            enabled (bool):    总开关。
                               - True: 开启拦截和重写。
                               - False: 仅仅是“透传”，不做任何处理（方便评估原始模型）。
        """
        self.detect_mode = detect_mode
        self.use_rag = use_rag
        self.enabled = enabled
        self.runtime_mode = runtime_mode

        # 初始化检测图（只负责判断是否违规，不负责重写）
        self.detect_graph = build_guard_graph(
            detection_difficulty=detect_mode,
            runtime_mode=runtime_mode,
        )

        print(
            f"[SafetyGuard] 初始化完成 | 检测: {detect_mode} | 运行: {runtime_mode} | "
            f"RAG: {use_rag} | 开关: {'开启' if enabled else '关闭'}"
        )

    def check_input(self, question: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        【前置围栏】检测用户提问 (Input Guard)

        返回:
            is_safe (bool):  True 表示安全（或Guard已关闭），False 表示被拦截。
            content (str):   如果安全，返回原问题；如果违规，返回拒答话术。
            metadata (dict): 调试信息（包含风险等级、路由信息等）。
        """
        # 1. 如果开关关闭，直接放行
        if not self.enabled:
            return True, question, {"guard_status": "disabled"}

        # 2. 调用 GuardGraph 进行检测
        state = {"text": question}
        try:
            result = self.detect_graph.invoke(state)
            violation = int(result.get("pred_violation", 0))
            risk_profile = result.get("risk_profile", {})
            router_ctx = result.get("router_ctx", {})
        except Exception as e:
            print(f"[Guard Error] 检测失败: {e}")
            # 出错时默认放行，或者你可以选择默认阻断
            return True, question, {"error": str(e)}

        # 3. 如果判定安全 (0)，放行
        if violation == 0:
            return True, question, {"risk_profile": risk_profile, "action": "pass"}

        strategy = select_response_action(
            stage="input",
            difficulty=self.detect_mode,
            input_risk_profile=risk_profile,
            output_risk_profile={},
            router_ctx_in=router_ctx,
            router_ctx_out={},
        )
        strategy_action = str(strategy.get("action", "SAFE_ANSWER")).upper()

        # 3.1 允许“输入改写后放行”：提升 strict/normal 的可用性
        if strategy_action in {"ALLOW", "REWRITE"}:
            rewritten_query = rewrite_query_for_safe_mode(question, strategy)
            return True, rewritten_query, {
                "risk_profile": risk_profile,
                "action": "rewrote_input_query" if strategy_action == "REWRITE" else "pass",
                "response_strategy": strategy,
                "rewritten_query": rewritten_query,
            }
        
        # 4. 如果判定违规 (1)，生成拒答/代答
        print(f"[Guard] 拦截到不安全提问，正在生成拒绝回复...")
        safe_response, response_strategy = self._generate_safe_response(
            question,
            source_type="input",
            input_risk_profile=risk_profile,
        )
        strategy_action = str(response_strategy.get("action", "SAFE_ANSWER")).lower()
        
        return False, safe_response, {
            "risk_profile": risk_profile, 
            "action": f"blocked_input_{strategy_action}",
            "response_strategy": response_strategy,
            "original_query": question
        }

    def check_output(self, question: str, response: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        【后置围栏】检测模型回答 (Output Guard)

        参数:
            question: 用户的原始问题（用于给 SafeQA 提供上下文）。
            response: 模型生成的原始回答。

        返回:
            is_safe (bool):  True 表示安全，False 表示被重写。
            content (str):   最终展示给用户的内容（原回答 或 安全代答）。
            metadata (dict): 调试信息。
        """
        # 1. 如果开关关闭，直接放行
        if not self.enabled:
            return True, response, {"guard_status": "disabled"}

        # 2. 检测回答的风险
        state = {"text": response}
        try:
            result = self.detect_graph.invoke(state)
            violation = int(result.get("pred_violation", 0))
            risk_profile = result.get("risk_profile", {})
        except Exception as e:
            print(f"[Guard Error] 检测失败: {e}")
            return True, response, {"error": str(e)}

        # 3. 如果安全，放行
        if violation == 0:
            return True, response, {"risk_profile": risk_profile, "action": "pass"}

        # 4. 如果违规，进行安全重写
        print(f"[Guard] 拦截到不安全回答，正在进行重写...")
        safe_rewrite, response_strategy = self._generate_safe_response(
            question, 
            source_type="output", 
            unsafe_response=response,
            output_risk_profile=risk_profile,
        )
        strategy_action = str(response_strategy.get("action", "REWRITE")).lower()
        output_action = "rewrote_output" if strategy_action == "rewrite" else f"replaced_output_{strategy_action}"
        
        return False, safe_rewrite, {
            "risk_profile": risk_profile, 
            "action": output_action,
            "response_strategy": response_strategy,
            "original_response": response
        }

    def _generate_safe_response(
        self,
        question: str,
        source_type: str,
        unsafe_response: str = "",
        input_risk_profile: Optional[Dict[str, Any]] = None,
        output_risk_profile: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """
        内部方法：调用 SafeQA Graph 生成高质量的安全回复。
        """
        try:
            # 1. 分析意图（SafeQA 需要知道用户想干嘛）
            router_ctx_in = analyze_query(question, mode=self.runtime_mode)
            
            # 如果是检查输出，最好也分析一下原来的回答（可选）
            router_ctx_out = {}
            if source_type == "output" and unsafe_response:
                router_ctx_out = analyze_query(unsafe_response, mode=self.runtime_mode)

            # 2. 构造 SafeQA 的输入状态
            # 这里复用了 safe_qa_graph 中的 QAState 定义
            state: QAState = {
                "user_query": question,
                "difficulty": self.detect_mode,
                "runtime_mode": self.runtime_mode,
                "model_answer": unsafe_response, # 如果是 Input Guard，这里是空的
                
                "router_ctx_in": router_ctx_in,
                "router_ctx_out": router_ctx_out,
                "guard_stage": source_type,
                "input_risk_profile": input_risk_profile or {},
                "output_risk_profile": output_risk_profile or {},
                "input_violation": 1 if source_type == "input" else 0,
                "output_violation": 1 if source_type == "output" else 0,
                
                # 传入 RAG 开关
                "use_rag": self.use_rag
            }

            # 3. 调用 safe_answer_node 生成回复
            # 注意：这里直接调用 node 函数，而不是 invoke 整个 graph，节省开销
            result_state = safe_answer_node(state)
            return (
                result_state.get("final_answer", "很抱歉，我无法回答该问题。"),
                result_state.get("response_strategy", {}),
            )

        except Exception as e:
            print(f"[Guard Error] 生成安全代答失败: {e}")
            strategy = select_response_action(
                stage=source_type,
                difficulty=self.detect_mode,
                input_risk_profile=input_risk_profile or {},
                output_risk_profile=output_risk_profile or {},
                router_ctx_in=analyze_query(question, mode=self.runtime_mode),
                router_ctx_out=(
                    analyze_query(unsafe_response, mode=self.runtime_mode)
                    if unsafe_response
                    else {}
                ),
            )
            risk_profile = output_risk_profile if source_type == "output" else input_risk_profile
            fallback = render_fallback_safe_response(
                query=question,
                strategy=strategy,
                risk_profile=risk_profile or {},
                stage=source_type,
                original_answer=unsafe_response,
            )
            return fallback, strategy
