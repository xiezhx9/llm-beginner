"""Prepare the model and deterministic data subsets required by Task 3."""

from __future__ import annotations

import argparse
import json
import random
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from huggingface_hub import hf_hub_download, snapshot_download


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"

QWEN_REPO = "Qwen/Qwen2.5-0.5B"
MOSS_REPO = "OpenMOSS-Team/moss-003-sft-data"
DPO_REPO = "hiyouga/DPO-En-Zh-20k"
CEVAL_REPO = "ceval/ceval-exam"

MOSS_SFT_ARCHIVE = "moss-003-sft-no-tools.jsonl.zip"
MOSS_PLUGIN_ARCHIVE = "moss-003-sft-with-tools-no-text2image.zip"

CEVAL_SUBJECTS = (
    "advanced_mathematics",
    "business_administration",
    "chinese_language_and_literature",
    "clinical_medicine",
    "college_programming",
    "computer_network",
    "law",
    "operating_system",
)


@dataclass(frozen=True)
class DataPrepConfig:
    """Sample budgets used to create the local learning-sized datasets."""

    sft_train_samples: int = 2_000
    sft_eval_samples: int = 200
    plugin_train_samples: int = 1_000
    plugin_eval_samples: int = 100
    dpo_train_samples_per_language: int = 1_000
    dpo_eval_samples_per_language: int = 100
    ceval_samples_per_subject: int = 10
    seed: int = 42


def _resolve_url(repo_id: str, filename: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"


def _read_exact(stream: Any, size: int) -> bytes:
    chunks: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("ZIP stream ended before its local header was complete")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def stream_zip_jsonl_subset(
    url: str,
    output_path: Path,
    max_records: int,
) -> int:
    """Stream the first JSONL member of a ZIP and stop after ``max_records``."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".part")

    with requests.get(url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        header = _read_exact(response.raw, 30)
        fields = struct.unpack("<IHHHHHIIIHH", header)
        signature, compression = fields[0], fields[3]
        filename_length, extra_length = fields[9], fields[10]
        if signature != 0x04034B50:
            raise ValueError("The remote file does not begin with a ZIP member")
        if compression != 8:
            raise ValueError(f"Unsupported ZIP compression method: {compression}")

        _read_exact(response.raw, filename_length + extra_length)
        decompressor = zlib.decompressobj(-zlib.MAX_WBITS)
        pending = b""
        written = 0

        with temporary_path.open("wb") as output:
            while written < max_records:
                compressed = response.raw.read(1024 * 1024)
                if not compressed:
                    break
                pending += decompressor.decompress(compressed)
                lines = pending.split(b"\n")
                pending = lines.pop()
                for line in lines:
                    if line.strip():
                        json.loads(line)
                        output.write(line + b"\n")
                        written += 1
                    if written == max_records:
                        break

            if written < max_records and pending.strip():
                json.loads(pending)
                output.write(pending + b"\n")
                written += 1

    if written < max_records:
        temporary_path.unlink(missing_ok=True)
        raise RuntimeError(f"Expected {max_records} records but received {written}")
    temporary_path.replace(output_path)
    return written


def split_jsonl(
    source_path: Path,
    train_path: Path,
    eval_path: Path,
    train_samples: int,
    eval_samples: int,
) -> None:
    """Split an already deterministic JSONL sample into train and eval files."""
    lines = source_path.read_bytes().splitlines(keepends=True)
    expected = train_samples + eval_samples
    if len(lines) != expected:
        raise ValueError(f"Expected {expected} records in {source_path}, got {len(lines)}")
    train_path.parent.mkdir(parents=True, exist_ok=True)
    train_path.write_bytes(b"".join(lines[:train_samples]))
    eval_path.write_bytes(b"".join(lines[train_samples:]))
    source_path.unlink()


def prepare_base_model() -> Path:
    """Download the fixed Qwen base model expected by the evaluator."""
    model_dir = MODELS_DIR / "Qwen2.5-0.5B"
    snapshot_download(QWEN_REPO, local_dir=model_dir)
    return model_dir


def _prepare_moss_subset(
    archive: str,
    destination: Path,
    train_count: int,
    eval_count: int,
) -> dict[str, int]:
    """Stream and split one fixed-size MOSS subset."""
    staging_path = destination / "sample.jsonl"
    total = train_count + eval_count
    stream_zip_jsonl_subset(_resolve_url(MOSS_REPO, archive), staging_path, total)
    split_jsonl(
        staging_path,
        destination / "train.jsonl",
        destination / "eval.jsonl",
        train_count,
        eval_count,
    )
    return {str(destination.relative_to(ROOT_DIR)): total}


def prepare_moss_sft_subset(config: DataPrepConfig) -> dict[str, int]:
    """Prepare only the no-tools MOSS subset used by S1 and S2."""
    return _prepare_moss_subset(
        MOSS_SFT_ARCHIVE,
        DATA_DIR / "moss-sft",
        config.sft_train_samples,
        config.sft_eval_samples,
    )


def prepare_moss_subsets(config: DataPrepConfig) -> dict[str, int]:
    """Prepare no-tools and with-tools MOSS train/eval subsets."""
    counts = prepare_moss_sft_subset(config)
    counts.update(
        _prepare_moss_subset(
            MOSS_PLUGIN_ARCHIVE,
            DATA_DIR / "moss-plugin",
            config.plugin_train_samples,
            config.plugin_eval_samples,
        )
    )
    return counts


def _load_dpo_language(language: str) -> list[dict[str, Any]]:
    filename = f"dpo_{language}.json"
    path = hf_hub_download(DPO_REPO, filename, repo_type="dataset")
    with Path(path).open(encoding="utf-8") as file:
        records: list[dict[str, Any]] = json.load(file)
    return records


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def prepare_dpo_subsets(config: DataPrepConfig) -> dict[str, int]:
    """Prepare balanced Chinese/English preference train and eval subsets."""
    rng = random.Random(config.seed)
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    required = config.dpo_train_samples_per_language + config.dpo_eval_samples_per_language

    for language in ("en", "zh"):
        records = _load_dpo_language(language)
        rng.shuffle(records)
        if len(records) < required:
            raise ValueError(f"DPO {language} has only {len(records)} records")
        train_records.extend(records[: config.dpo_train_samples_per_language])
        eval_records.extend(records[config.dpo_train_samples_per_language : required])

    rng.shuffle(train_records)
    rng.shuffle(eval_records)
    _write_jsonl(DATA_DIR / "dpo" / "train.jsonl", train_records)
    _write_jsonl(DATA_DIR / "dpo" / "eval.jsonl", eval_records)
    return {"data/dpo/train.jsonl": len(train_records), "data/dpo/eval.jsonl": len(eval_records)}


def prepare_ceval_subset(config: DataPrepConfig) -> dict[str, int]:
    """Prepare a labeled, cross-subject C-Eval validation subset."""
    records: list[dict[str, Any]] = []
    for index, subject in enumerate(CEVAL_SUBJECTS):
        filename = f"{subject}/val-00000-of-00001.parquet"
        path = hf_hub_download(CEVAL_REPO, filename, repo_type="dataset")
        frame = pd.read_parquet(path)
        sample_count = min(config.ceval_samples_per_subject, len(frame))
        sample = frame.sample(n=sample_count, random_state=config.seed + index)
        for row in sample.to_dict(orient="records"):
            row["subject"] = subject
            records.append(row)

    output_path = DATA_DIR / "ceval" / "validation.jsonl"
    _write_jsonl(output_path, records)
    return {str(output_path.relative_to(ROOT_DIR)): len(records)}


def write_manifest(config: DataPrepConfig, counts: dict[str, int]) -> Path:
    """Record data sources, sample budgets, and generated file counts."""
    manifest = {
        "config": asdict(config),
        "sources": {
            "base_model": QWEN_REPO,
            "sft_and_plugin": MOSS_REPO,
            "dpo": DPO_REPO,
            "ceval": CEVAL_REPO,
        },
        "generated_counts": counts,
    }
    path = DATA_DIR / "manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def prepare_all(config: DataPrepConfig, skip_model: bool = False) -> Path:
    """Prepare every mandatory and bonus resource for Task 3."""
    counts: dict[str, int] = {}
    if not skip_model:
        model_path = prepare_base_model()
        counts[str(model_path.relative_to(ROOT_DIR))] = 1
    counts.update(prepare_moss_subsets(config))
    counts.update(prepare_dpo_subsets(config))
    counts.update(prepare_ceval_subset(config))
    return write_manifest(config, counts)


def prepare_s1(config: DataPrepConfig, skip_model: bool = False) -> Path:
    """Prepare only the base model and deterministic MOSS data required by S1."""
    counts: dict[str, int] = {}
    if not skip_model:
        model_path = prepare_base_model()
        counts[str(model_path.relative_to(ROOT_DIR))] = 1
    counts.update(prepare_moss_sft_subset(config))
    return write_manifest(config, counts)


def parse_args() -> argparse.Namespace:
    """Parse data-preparation command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true", help="Do not download Qwen weights")
    parser.add_argument(
        "--s1-only",
        action="store_true",
        help="Prepare only the Qwen base model and MOSS SFT subset required by S1",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Prepare Task 3 resources using learning-sized defaults."""
    args = parse_args()
    config = DataPrepConfig(seed=args.seed)
    if args.s1_only:
        manifest_path = prepare_s1(config, skip_model=args.skip_model)
    else:
        manifest_path = prepare_all(config, skip_model=args.skip_model)
    print(f"Task 3 resources are ready. Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
