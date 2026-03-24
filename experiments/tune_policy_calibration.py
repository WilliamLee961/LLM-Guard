from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid-search policy calibration for sidecar guard.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="experiments/datasets/sample_guard_eval.jsonl",
    )
    parser.add_argument("--output-dir", type=str, default="experiments/results/calibration")
    parser.add_argument("--difficulty", type=str, default="normal")
    parser.add_argument("--judge-difficulty", type=str, default="strict")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--w-security", type=float, default=0.5, help="Weight for 1-ASR")
    parser.add_argument("--w-overrefusal", type=float, default=0.2, help="Weight for 1-benign_refusal")
    parser.add_argument("--w-helpfulness", type=float, default=0.3, help="Weight for benign_helpful_rate")
    return parser.parse_args()


def build_calibration_grid() -> List[Dict[str, Any]]:
    grid: List[Dict[str, Any]] = []
    for in_safe_th in [2, 3, 4]:
        for in_refuse_th in [5, 6]:
            for out_rewrite_th in [3, 4, 5]:
                grid.append(
                    {
                        "input_uncertainty_safe_answer_threshold": in_safe_th,
                        "input_uncertainty_refuse_threshold": in_refuse_th,
                        "output_uncertainty_rewrite_threshold": out_rewrite_th,
                        "input_hard_refuse_min": {
                            "PHYSICAL_HARM": "MEDIUM",
                            "CRIMES_ILLEGAL": "MEDIUM",
                            "MALICIOUS_USE": "MEDIUM",
                        },
                    }
                )
    return grid


def latest_summary_file(output_dir: Path) -> Path:
    summaries = sorted(output_dir.glob("*_summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not summaries:
        raise FileNotFoundError(f"No summary file found under {output_dir}")
    return summaries[0]


def score_summary(summary: Dict[str, Any], w_security: float, w_overrefusal: float, w_helpfulness: float) -> float:
    metrics = summary.get("baselines", {}).get("full_sidecar", {})
    asr = float(metrics.get("asr_harmful_output_rate", 1.0))
    over = float(metrics.get("benign_refusal_rate", 1.0))
    helpful = float(metrics.get("benign_helpful_rate", 0.0))
    return w_security * (1.0 - asr) + w_overrefusal * (1.0 - over) + w_helpfulness * helpful


def run_one_config(
    cfg: Dict[str, Any],
    cfg_idx: int,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, Any]:
    calib_path = output_dir / f"policy_cfg_{cfg_idx:03d}.json"
    with calib_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    tag = f"calib_{cfg_idx:03d}"
    cmd = [
        sys.executable,
        str(ROOT_DIR / "experiments" / "run_guard_benchmark.py"),
        "--dataset",
        args.dataset,
        "--output-dir",
        str(output_dir),
        "--difficulty",
        args.difficulty,
        "--judge-difficulty",
        args.judge_difficulty,
        "--baselines",
        "full_sidecar",
        "--tag",
        tag,
    ]
    if args.max_samples > 0:
        cmd.extend(["--max-samples", str(args.max_samples)])

    env = os.environ.copy()
    env["GUARD_POLICY_CALIB_PATH"] = str(calib_path)

    subprocess.run(cmd, check=True, env=env, cwd=str(ROOT_DIR))
    summary_path = latest_summary_file(output_dir)
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    score = score_summary(summary, args.w_security, args.w_overrefusal, args.w_helpfulness)
    return {
        "idx": cfg_idx,
        "config_path": str(calib_path),
        "summary_path": str(summary_path),
        "score": round(score, 6),
        "summary": summary,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = build_calibration_grid()
    print(f"[calibration] grid_size={len(grid)}")

    results: List[Dict[str, Any]] = []
    best: Dict[str, Any] | None = None
    for i, cfg in enumerate(grid):
        t0 = time.perf_counter()
        record = run_one_config(cfg, i, args, output_dir)
        record["elapsed_sec"] = round(time.perf_counter() - t0, 2)
        results.append(record)
        if best is None or record["score"] > best["score"]:
            best = record
        print(f"[cfg={i:03d}] score={record['score']} elapsed={record['elapsed_sec']}s")

    ranking_path = output_dir / "calibration_ranking.json"
    results_sorted = sorted(results, key=lambda x: x["score"], reverse=True)
    with ranking_path.open("w", encoding="utf-8") as f:
        json.dump(results_sorted, f, ensure_ascii=False, indent=2)

    if best:
        best_path = output_dir / "best_calibration.json"
        cfg_path = Path(best["config_path"])
        with cfg_path.open("r", encoding="utf-8") as f:
            best_cfg = json.load(f)
        with best_path.open("w", encoding="utf-8") as f:
            json.dump(best_cfg, f, ensure_ascii=False, indent=2)
        print(f"[best] score={best['score']} cfg={cfg_path}")
        print(f"[best] exported -> {best_path}")

    print(f"[ranking] {ranking_path}")


if __name__ == "__main__":
    main()
