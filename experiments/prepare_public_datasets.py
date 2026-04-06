from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

TRUE_LABEL_TOKENS = {
    "1",
    "true",
    "yes",
    "harmful",
    "unsafe",
    "toxic",
    "jailbreak",
    "jailbreaking",
    "malicious",
}
FALSE_LABEL_TOKENS = {
    "0",
    "false",
    "no",
    "benign",
    "safe",
    "harmless",
    "nonharmful",
    "non_harmful",
}
SPLIT_TRUE_HINTS = ("harm", "unsafe", "toxic", "jailbreak", "attack", "malicious")
SPLIT_FALSE_HINTS = ("benign", "safe", "harmless", "normal", "helpful", "clean")


@dataclass(frozen=True)
class SourceSpec:
    key: str
    hf_dataset: str
    output_file: str
    config_candidates: Tuple[Optional[str], ...]
    query_fields: Tuple[str, ...]
    label_fields: Tuple[str, ...] = ()
    category_fields: Tuple[str, ...] = ("category", "Category")
    default_label: Optional[bool] = None
    note: str = ""


SPECS: Tuple[SourceSpec, ...] = (
    SourceSpec(
        key="jailbreakbench",
        hf_dataset="JailbreakBench/JBB-Behaviors",
        output_file="jbb_mini.jsonl",
        config_candidates=("behaviors", None),
        query_fields=("Goal", "goal", "prompt", "query", "Behavior", "behavior"),
        label_fields=("is_harmful", "harmful", "label"),
        note="Expect harmful/benign split names from JBB.",
    ),
    SourceSpec(
        key="harmbench",
        hf_dataset="walledai/HarmBench",
        output_file="harmbench_mini.jsonl",
        config_candidates=("standard", None),
        query_fields=("prompt", "query", "Goal", "goal"),
        label_fields=("is_harmful", "harmful", "label", "unsafe"),
        default_label=True,
        note="HarmBench is mostly harmful prompts; fallback default_label=True.",
    ),
    SourceSpec(
        key="xstest",
        hf_dataset="allenai/xstest-response",
        output_file="xstest_mini.jsonl",
        config_candidates=(None,),
        query_fields=("prompt", "query", "instruction", "Goal", "goal"),
        label_fields=("is_harmful", "harmful", "unsafe", "prompt_harmful", "prompt_type", "type", "label"),
    ),
    SourceSpec(
        key="wildguard",
        hf_dataset="AlignmentResearch/WildGuardTest",
        output_file="wildguard_mini.jsonl",
        config_candidates=(None,),
        query_fields=("prompt", "query", "user_input", "instruction", "text", "content"),
        label_fields=(
            "is_harmful",
            "harmful",
            "unsafe",
            "prompt_harm_label",
            "prompt_harmfulness",
            "harm_label",
            "clf_label",
            "label",
        ),
    ),
    SourceSpec(
        key="toxicchat",
        hf_dataset="lmsys/toxic-chat",
        output_file="toxicchat_mini.jsonl",
        config_candidates=("toxicchat0124", "toxicchat", None),
        query_fields=("user_input", "prompt", "query", "instruction", "text"),
        label_fields=("toxicity", "jailbreaking", "is_harmful", "label"),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare 5 public guard datasets into normalized *_mini.jsonl files.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiments/datasets",
        help="Output directory for normalized JSONL files.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(spec.key for spec in SPECS),
        help="Comma-separated dataset keys to prepare.",
    )
    parser.add_argument(
        "--max-samples-per-dataset",
        type=int,
        default=400,
        help="Maximum rows for each output mini dataset.",
    )
    parser.add_argument(
        "--harmful-ratio",
        type=float,
        default=0.5,
        help="Target harmful ratio when both harmful/benign are available.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately when one dataset preparation fails.",
    )
    return parser.parse_args()


def _norm_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.strip().lower())


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        if not value:
            return ""
        first = value[0]
        if isinstance(first, str):
            return first.strip()
        if isinstance(first, dict):
            for key in ("content", "text", "prompt", "query", "value"):
                if key in first and isinstance(first[key], str):
                    return first[key].strip()
        return ""
    if isinstance(value, dict):
        for key in ("content", "text", "prompt", "query", "value"):
            if key in value and isinstance(value[key], str):
                return value[key].strip()
        return ""
    return ""


def parse_label_value(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
        return None
    if isinstance(value, float):
        if value in (0.0, 1.0):
            return bool(int(value))
        return None
    if isinstance(value, str):
        token = re.sub(r"[^a-z0-9_]+", "", value.strip().lower())
        if not token:
            return None
        if token in TRUE_LABEL_TOKENS:
            return True
        if token in FALSE_LABEL_TOKENS:
            return False
        if "unsafe" in token or "toxic" in token or "jailbreak" in token:
            return True
        if "harmful" in token and "non" not in token:
            return True
        if "benign" in token or "harmless" in token or token.startswith("safe"):
            return False
    return None


def infer_label_from_split_name(split_name: str) -> Optional[bool]:
    name = split_name.strip().lower()
    if any(hint in name for hint in SPLIT_TRUE_HINTS):
        return True
    if any(hint in name for hint in SPLIT_FALSE_HINTS):
        return False
    return None


def pick_query_text(row: Dict[str, Any], query_fields: Sequence[str]) -> str:
    row_norm = {_norm_key(k): v for k, v in row.items()}

    for key in query_fields:
        norm = _norm_key(key)
        if norm in row_norm:
            txt = _as_text(row_norm[norm])
            if len(txt) >= 3:
                return txt

    excluded = {
        "id",
        "index",
        "label",
        "is_harmful",
        "harmful",
        "unsafe",
        "toxicity",
        "jailbreaking",
        "category",
        "source",
        "target",
        "behavior",
    }
    for k, v in row_norm.items():
        if k in excluded:
            continue
        txt = _as_text(v)
        if len(txt) >= 8:
            return txt
    return ""


def infer_label_from_row(row: Dict[str, Any], label_fields: Sequence[str]) -> Optional[bool]:
    row_norm = {_norm_key(k): v for k, v in row.items()}

    for key in label_fields:
        norm = _norm_key(key)
        if norm not in row_norm:
            continue
        parsed = parse_label_value(row_norm[norm])
        if parsed is not None:
            return parsed

    # ToxicChat-style two-signal fallback
    tox = parse_label_value(row_norm.get("toxicity"))
    jb = parse_label_value(row_norm.get("jailbreaking"))
    if tox is True or jb is True:
        return True
    if tox is False and jb is False:
        return False

    return None


def pick_category(row: Dict[str, Any], category_fields: Sequence[str], default_name: str) -> str:
    row_norm = {_norm_key(k): v for k, v in row.items()}
    for key in category_fields:
        norm = _norm_key(key)
        if norm in row_norm:
            txt = _as_text(row_norm[norm])
            if txt:
                return txt
    return default_name


def pick_row_id(row: Dict[str, Any], fallback: str) -> str:
    for key in ("id", "ID", "index", "Index"):
        if key in row:
            val = row[key]
            if isinstance(val, (str, int)):
                txt = str(val).strip()
                if txt:
                    return txt
    return fallback


def load_with_candidates(dataset_id: str, config_candidates: Sequence[Optional[str]]):
    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "Missing dependency `datasets`. Install it with:\n"
            "  pip install datasets>=2.20.0"
        ) from exc

    errors: List[str] = []
    for config in config_candidates:
        try:
            if config is None:
                data = load_dataset(dataset_id)
            else:
                data = load_dataset(dataset_id, config)
            return data, config
        except Exception as exc:  # pragma: no cover
            cfg = "default" if config is None else config
            errors.append(f"{cfg}: {type(exc).__name__}: {exc}")
    joined = "\n".join(errors)
    raise RuntimeError(f"Failed to load {dataset_id} with all configs:\n{joined}")


def deduplicate_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        key = row.get("query", "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def sample_rows(rows: List[Dict[str, Any]], max_samples: int, harmful_ratio: float, seed: int) -> List[Dict[str, Any]]:
    if max_samples <= 0 or len(rows) <= max_samples:
        return rows

    rng = random.Random(seed)
    harmful = [r for r in rows if bool(r.get("is_harmful", False))]
    benign = [r for r in rows if not bool(r.get("is_harmful", False))]

    if harmful and benign:
        harmful_target = max(1, int(round(max_samples * harmful_ratio)))
        harmful_target = min(harmful_target, len(harmful), max_samples)
        benign_target = min(max_samples - harmful_target, len(benign))

        if harmful_target + benign_target < max_samples:
            remaining = max_samples - (harmful_target + benign_target)
            if len(harmful) - harmful_target >= remaining:
                harmful_target += remaining
            else:
                benign_target += min(remaining, len(benign) - benign_target)

        rng.shuffle(harmful)
        rng.shuffle(benign)
        selected = harmful[:harmful_target] + benign[:benign_target]
        rng.shuffle(selected)
        return selected

    shuffled = list(rows)
    rng.shuffle(shuffled)
    return shuffled[:max_samples]


def normalize_source(spec: SourceSpec, max_samples: int, harmful_ratio: float, seed: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    ds_obj, used_config = load_with_candidates(spec.hf_dataset, spec.config_candidates)
    splits = ds_obj.items() if hasattr(ds_obj, "items") else [("train", ds_obj)]

    all_rows: List[Dict[str, Any]] = []
    total_scanned = 0
    total_missing_query = 0
    total_missing_label = 0

    for split_name, dataset_split in splits:
        split_label = infer_label_from_split_name(str(split_name))
        for idx, row in enumerate(dataset_split):
            total_scanned += 1
            if not isinstance(row, dict):
                continue

            query = pick_query_text(row, spec.query_fields)
            if not query:
                total_missing_query += 1
                continue

            label = infer_label_from_row(row, spec.label_fields)
            if label is None:
                label = split_label
            if label is None:
                label = spec.default_label
            if label is None:
                total_missing_label += 1
                continue

            category = pick_category(row, spec.category_fields, default_name=spec.key)
            row_id = pick_row_id(row, fallback=f"{spec.key}-{split_name}-{idx}")
            all_rows.append(
                {
                    "id": row_id,
                    "query": query,
                    "is_harmful": bool(label),
                    "category": category,
                }
            )

    deduped = deduplicate_rows(all_rows)
    sampled = sample_rows(deduped, max_samples=max_samples, harmful_ratio=harmful_ratio, seed=seed)
    harmful_count = sum(1 for row in sampled if bool(row["is_harmful"]))
    benign_count = len(sampled) - harmful_count

    report = {
        "source_key": spec.key,
        "hf_dataset": spec.hf_dataset,
        "used_config": used_config if used_config is not None else "default",
        "note": spec.note,
        "total_scanned": total_scanned,
        "kept_before_dedup": len(all_rows),
        "kept_after_dedup": len(deduped),
        "final_rows": len(sampled),
        "harmful_rows": harmful_count,
        "benign_rows": benign_count,
        "missing_query_rows": total_missing_query,
        "missing_label_rows": total_missing_label,
    }
    return sampled, report


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    output_dir = (root_dir / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_keys = {k.strip() for k in args.datasets.split(",") if k.strip()}
    spec_map = {spec.key: spec for spec in SPECS}
    unknown = sorted(selected_keys - set(spec_map.keys()))
    if unknown:
        raise ValueError(f"Unknown dataset keys: {unknown}. Available: {sorted(spec_map.keys())}")

    reports: List[Dict[str, Any]] = []
    failed: List[Dict[str, str]] = []

    for key in sorted(selected_keys):
        spec = spec_map[key]
        print(f"[start] {spec.key} <- {spec.hf_dataset}")
        try:
            rows, report = normalize_source(
                spec,
                max_samples=args.max_samples_per_dataset,
                harmful_ratio=args.harmful_ratio,
                seed=args.seed,
            )
            out_path = output_dir / spec.output_file
            write_jsonl(out_path, rows)
            report["output_file"] = str(out_path)
            reports.append(report)
            print(
                f"[done] {spec.key}: rows={report['final_rows']} "
                f"(harmful={report['harmful_rows']}, benign={report['benign_rows']}) -> {out_path.name}"
            )
        except Exception as exc:
            err = {"source_key": spec.key, "error": f"{type(exc).__name__}: {exc}"}
            failed.append(err)
            print(f"[error] {spec.key}: {err['error']}")
            if args.fail_fast:
                raise

    report_path = output_dir / "prepare_public_datasets_report.json"
    report_payload = {
        "selected_datasets": sorted(selected_keys),
        "max_samples_per_dataset": args.max_samples_per_dataset,
        "harmful_ratio": args.harmful_ratio,
        "seed": args.seed,
        "success": reports,
        "failed": failed,
    }
    report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report] {report_path}")
    print(f"[summary] success={len(reports)} failed={len(failed)}")

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
