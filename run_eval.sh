python eval_with_langgraph_arg.py \
  --jsonl_path /home/lijiahang/MY_GUARD/results/results_en/Guard_FullPipeline_in_strict_out_normal_20251215_111638.jsonl \
  --detect_mode strict
python eval_with_langgraph_arg.py \
  --jsonl_path /home/lijiahang/MY_GUARD/results/results_en/Guard_FullPipeline_in_strict_out_normal_20251215_111638.jsonl \
  --detect_mode normal
python eval_with_langgraph_arg.py \
  --jsonl_path /home/lijiahang/MY_GUARD/results/results_en/Guard_FullPipeline_in_strict_out_normal_20251215_111638.jsonl \
  --detect_mode lenient
# python eval_with_langgraph_arg.py \
#   --jsonl_path ./datasets/base_set_final.jsonl \
#   --detect_mode lenient
# python eval_with_langgraph_arg.py \
#   --jsonl_path ./datasets/base_set_final.jsonl \
#   --detect_mode strict
# python eval_with_langgraph_arg.py \
#   --jsonl_path ./datasets/safety_en_14k_final.jsonl \
#   --detect_mode normal
# python eval_with_langgraph_arg.py \
#   --jsonl_path ./datasets/safety_en_14k_final.jsonl \
#   --detect_mode lenient
# python eval_with_langgraph_arg.py \
#   --jsonl_path ./datasets/safety_en_14k_final.jsonl \
#   --detect_mode strict
