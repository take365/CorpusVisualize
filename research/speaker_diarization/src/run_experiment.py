from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path
from typing import List


def build_command(args: argparse.Namespace) -> List[str]:
    cmd = [
        sys.executable,
        "-m",
        "pipeline.run_pipeline",
        "--input-dir",
        str(args.input),
        "--output-dir",
        str(args.output),
    ]

    if args.config:
        cmd += ["--config", str(args.config)]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.asr:
        cmd += ["--asr", args.asr]
    if args.diarization:
        cmd += ["--diar", args.diarization]
    if args.pyannote_threshold is not None:
        cmd += ["--pyannote-threshold", str(args.pyannote_threshold)]
    if args.pyannote_min_cluster_size is not None:
        cmd += ["--pyannote-min-cluster-size", str(args.pyannote_min_cluster_size)]
    if args.pyannote_min_duration_off is not None:
        cmd += ["--pyannote-min-duration-off", str(args.pyannote_min_duration_off)]
    if args.no_llm:
        cmd += ["--no-llm"]
    if args.llm_base_url:
        cmd += ["--llm-base-url", args.llm_base_url]
    if args.llm_model:
        cmd += ["--llm-model", args.llm_model]
    if args.llm_max_tokens is not None:
        cmd += ["--llm-max-tokens", str(args.llm_max_tokens)]
    if args.llm_temperature is not None:
        cmd += ["--llm-temperature", str(args.llm_temperature)]
    return cmd


def ensure_output_path(path: Path, timestamp: bool) -> Path:
    if timestamp:
        now = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        path = path / now
    path.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run diarization experiment using the main pipeline")
    parser.add_argument("--input", required=True, type=Path, help="Directory containing input audio")
    parser.add_argument("--output", required=True, type=Path, help="Directory to store experiment outputs")
    parser.add_argument("--config", type=Path, help="Optional YAML config")
    parser.add_argument("--limit", type=int, help="Limit number of audio files")
    parser.add_argument("--asr", type=str, help="Override ASR backend")
    parser.add_argument("--diarization", type=str, help="Override diarization backend")
    parser.add_argument("--pyannote-threshold", type=float, help="pyannote clustering threshold")
    parser.add_argument("--pyannote-min-cluster-size", type=int, help="pyannote minimum cluster size")
    parser.add_argument("--pyannote-min-duration-off", type=float, help="pyannote minimum silence duration")
    parser.add_argument("--llm-base-url", type=str, help="Override LLM base URL")
    parser.add_argument("--llm-model", type=str, help="Override LLM model ID")
    parser.add_argument("--llm-max-tokens", type=int, help="LLM max tokens")
    parser.add_argument("--llm-temperature", type=float, help="LLM temperature")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM refinement")
    parser.add_argument("--timestamp", action="store_true", help="Append timestamp subdirectory under output")
    parser.add_argument("--dry-run", action="store_true", help="Show command without executing")

    args = parser.parse_args()

    output_dir = ensure_output_path(args.output, args.timestamp)
    args.output = output_dir

    command = build_command(args)

    print("[Experiment] command:")
    print(" ".join(json.dumps(part) if " " in part else part for part in command))

    if args.dry_run:
        print("[Experiment] dry-run: skipping execution")
        return

    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
