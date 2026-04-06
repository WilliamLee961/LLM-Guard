from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: str
    note: str


DEFAULT_DATASETS: List[DatasetSpec] = [
    DatasetSpec("jailbreakbench", "jbb_mini.jsonl", "JBB harmful+benign slice"),
    DatasetSpec("harmbench", "harmbench_mini.jsonl", "HarmBench harmful behavior slice"),
    DatasetSpec("xstest", "xstest_mini.jsonl", "Over-refusal focused safe/unsafe slice"),
    DatasetSpec("wildguard", "wildguard_mini.jsonl", "In-the-wild harmful/benign slice"),
    DatasetSpec("toxicchat", "toxicchat_mini.jsonl", "Toxic dialog prompt safety slice"),
    DatasetSpec("biz_benign_cn", "biz_benign_cn_mini.jsonl", "Project-specific benign Chinese workload"),
]

SUMMARY_METRICS = [
    "num_samples",
    "num_harmful",
    "num_benign",
    "asr_harmful_output_rate",
    "harmful_blocked_rate",
    "benign_refusal_rate",
    "benign_helpful_rate",
    "avg_latency_ms",
    "p95_latency_ms",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal 4x6 guard comparison matrix.")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="experiments/datasets",
        help="Directory that contains the normalized JSONL datasets.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/results/minimal_matrix",
        help="Directory for all benchmark outputs and merged summary csv.",
    )
    parser.add_argument(
        "--difficulty",
        type=str,
        default="normal",
        choices=["lenient", "normal", "strict", "any_risk", "overall_l2plus"],
    )
    parser.add_argument(
        "--judge-difficulty",
        type=str,
        default="strict",
        choices=["lenient", "normal", "strict", "any_risk", "overall_l2plus"],
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="no_guard,input_only,output_only,full_sidecar",
        help="Comma-separated baseline list passed to run_guard_benchmark.py",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Per-dataset sample cap. 0 means full file.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(spec.name for spec in DEFAULT_DATASETS),
        help="Comma-separated dataset keys from the default set.",
    )
    parser.add_argument(
        "--allow-sample-fallback",
        action="store_true",
        help="If a target dataset file does not exist, fallback to sample_guard_eval.jsonl.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when a dataset run fails.",
    )
    return parser.parse_args()


def resolve_datasets(selected: str) -> List[DatasetSpec]:
    selected_names = [x.strip() for x in selected.split(",") if x.strip()]
    spec_map: Dict[str, DatasetSpec] = {spec.name: spec for spec in DEFAULT_DATASETS}
    out: List[DatasetSpec] = []
    for name in selected_names:
        if name not in spec_map:
            available = ",".join(spec_map.keys())
            raise ValueError(f"Unknown dataset key: {name}. Available: {available}")
        out.append(spec_map[name])
    return out


def find_latest_summary(output_dir: Path, tag: str) -> Path:
    files = sorted(output_dir.glob(f"*_{tag}_summary.json"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"Summary file not found for tag={tag} in {output_dir}")
    return files[-1]


def run_one_dataset(
    run_script: Path,
    dataset_path: Path,
    output_dir: Path,
    baselines: str,
    difficulty: str,
    judge_difficulty: str,
    max_samples: int,
    tag: str,
) -> Path:
    cmd = [
        sys.executable,
        str(run_script),
        "--dataset",
        str(dataset_path),
        "--output-dir",
        str(output_dir),
        "--baselines",
        baselines,
        "--difficulty",
        difficulty,
        "--judge-difficulty",
        judge_difficulty,
        "--tag",
        tag,
    ]
    if max_samples > 0:
        cmd.extend(["--max-samples", str(max_samples)])

    print(f"[run] dataset={dataset_path.name} tag={tag}")
    subprocess.run(cmd, check=True)
    return find_latest_summary(output_dir, tag=tag)


def flatten_summary(dataset_name: str, summary: Dict[str, object]) -> List[Dict[str, object]]:
    baselines = summary.get("baselines", {})
    rows: List[Dict[str, object]] = []
    if not isinstance(baselines, dict):
        return rows

    for baseline, metrics in baselines.items():
        if not isinstance(metrics, dict):
            continue
        row: Dict[str, object] = {
            "dataset": dataset_name,
            "baseline": baseline,
            "difficulty": summary.get("difficulty", ""),
            "judge_difficulty": summary.get("judge_difficulty", ""),
        }
        for key in SUMMARY_METRICS:
            row[key] = metrics.get(key, "")
        rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    run_script = root_dir / "experiments" / "run_guard_benchmark.py"
    dataset_dir = (root_dir / args.dataset_dir).resolve()
    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_specs = resolve_datasets(args.datasets)
    fallback_path = dataset_dir / "sample_guard_eval.jsonl"
    all_rows: List[Dict[str, object]] = []
    failures: List[str] = []

    for spec in selected_specs:
        expected_path = dataset_dir / spec.path
        dataset_path = expected_path
        if not dataset_path.exists():
            if args.allow_sample_fallback and fallback_path.exists():
                print(f"[warn] missing {expected_path.name}, fallback to {fallback_path.name} ({spec.note})")
                dataset_path = fallback_path
            else:
                print(f"[skip] dataset file missing: {expected_path} ({spec.note})")
                continue

        tag = spec.name
        try:
            summary_path = run_one_dataset(
                run_script=run_script,
                dataset_path=dataset_path,
                output_dir=output_dir,
                baselines=args.baselines,
                difficulty=args.difficulty,
                judge_difficulty=args.judge_difficulty,
                max_samples=args.max_samples,
                tag=tag,
            )
            with summary_path.open("r", encoding="utf-8") as f:
                summary = json.load(f)
            all_rows.extend(flatten_summary(dataset_name=spec.name, summary=summary))
        except subprocess.CalledProcessError as e:
            msg = f"{spec.name}: benchmark failed ({e})"
            failures.append(msg)
            print(f"[error] {msg}")
            if args.fail_fast:
                raise
        except Exception as e:
            msg = f"{spec.name}: unexpected error ({e})"
            failures.append(msg)
            print(f"[error] {msg}")
            if args.fail_fast:
                raise

    merged_path = output_dir / "minimal_matrix_summary.csv"
    write_csv(merged_path, all_rows)
    print(f"[done] merged summary: {merged_path}")
    print(f"[done] rows: {len(all_rows)}")
    if failures:
        fail_path = output_dir / "minimal_matrix_failures.txt"
        fail_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"[done] failures: {len(failures)} -> {fail_path}")


if __name__ == "__main__":
    main()
