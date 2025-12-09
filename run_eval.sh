python eval_with_langgraph_arg.py \
  --jsonl_path ./datasets/typical_safety_scenarios_final.jsonl \
  --detect_mode strict
python eval_with_langgraph_arg.py \
  --jsonl_path ./datasets/base_set_final.jsonl \
  --detect_mode normal
python eval_with_langgraph_arg.py \
  --jsonl_path ./datasets/base_set_final.jsonl \
  --detect_mode lenient
python eval_with_langgraph_arg.py \
  --jsonl_path ./datasets/base_set_final.jsonl \
  --detect_mode strict
python eval_with_langgraph_arg.py \
  --jsonl_path ./datasets/safety_en_14k_final.jsonl \
  --detect_mode normal
python eval_with_langgraph_arg.py \
  --jsonl_path ./datasets/safety_en_14k_final.jsonl \
  --detect_mode lenient
python eval_with_langgraph_arg.py \
  --jsonl_path ./datasets/safety_en_14k_final.jsonl \
  --detect_mode strict
