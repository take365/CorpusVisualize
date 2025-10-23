#!/usr/bin/env python3
"""
run_whisperx_batch.py

Batch runner for WhisperX (GPU by default) + char-level eval + HTML diff report.

Features
- Scans an input directory (or explicit file list) for audio files
- Runs WhisperX with diarization on GPU (CUDA) by default
- Output directories are versioned to avoid overwrite (timestamp + optional slug)
- Per-file subfolder layout, plus a summary CSV
- Optional evaluation if reference files are available (same stem)
- Optional HTML visual diff report per file

Assumptions
- You have `whisperx` CLI installed and visible in PATH.
- `evaluate_char.py` and `report_html.py` exist locally or are importable / runnable via Python.
  If they are scripts, point to them via --eval-script and --html-script or let defaults work
  if they are importable modules (not required).

Example
--------
python run_whisperx_batch.py \
  --inputs research/speaker_diarization/data \
  --pattern "*.wav" \
  --ref-dir research/speaker_diarization/data/reference \
  --output-root research/speaker_diarization/runs/whisperx \
  --exp-slug raw_large_v2_cuda \
  --model large-v2 --language ja --batch-size 16 --compute-type float16 \
  --diarize --make-html

This will create something like:
  runs/whisperx/2025-10-22_12-45-31_raw_large_v2_cuda/
    ├── <file-stem>/
    │    ├── whisperx/          # raw whisperx outputs
    │    ├── segments.jsonl     # normalized hypothesis if produced by your pipeline
    │    ├── metrics.json       # per-file metrics (if ref present)
    │    └── comparison.html    # per-file HTML diff (if --make-html)
    └── summary.csv             # all files aggregate

Notes
- If a reference with the same stem exists under --ref-dir (jsonl or txt), eval is run.
- If not found, eval is skipped gracefully for that file.
- To force overwrite into a specific dir name, use --exp-name exactly; otherwise timestamped.

"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
REF_EXTS = {".jsonl", ".txt", ".vtt", ".srt"}


@dataclass
class RunConfig:
    inputs: List[Path]
    pattern: str
    ref_dir: Optional[Path]
    output_root: Path
    exp_name: Optional[str]
    exp_slug: Optional[str]
    diarize: bool
    make_html: bool
    model: str
    language: str
    device: str
    compute_type: str
    batch_size: int
    hf_token: Optional[str]
    whisperx_extra: List[str]
    eval_script: Optional[Path]
    html_script: Optional[Path]


# ---------- Utils ----------

def log(msg: str):
    print(f"[run-whisperx-batch] {msg}")


def ensure_run_dir(output_root: Path, exp_name: Optional[str], exp_slug: Optional[str]) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if exp_name:  # use as-is, but avoid overwrite by suffixing a counter
        base = output_root / exp_name
        run_dir = base
        i = 1
        while run_dir.exists():
            run_dir = output_root / f"{exp_name}_{i}"
            i += 1
        run_dir.mkdir(parents=True, exist_ok=False)
        return run_dir

    # timestamped run name
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    name = ts if not exp_slug else f"{ts}_{exp_slug}"
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def find_audio_files(inputs: List[Path], pattern: str) -> List[Path]:
    files: List[Path] = []
    for p in inputs:
        p = p.resolve()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            files.append(p)
        elif p.is_dir():
            files.extend(sorted(p.rglob(pattern)))
        else:
            log(f"WARN: input not found or unsupported: {p}")
    # filter valid audio
    files = [f for f in files if f.suffix.lower() in AUDIO_EXTS]
    return sorted(set(files))


def find_reference(stem: str, ref_dir: Optional[Path]) -> Optional[Path]:
    if not ref_dir:
        return None
    for ext in REF_EXTS:
        cand = ref_dir / f"{stem}{ext}"
        if cand.exists():
            return cand.resolve()
    return None


def run_cmd(cmd: List[str], cwd: Optional[Path] = None) -> int:
    log("$ " + " ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    return proc.returncode


# ---------- Core per-file pipeline ----------

def run_whisperx_on_file(audio: Path, out_dir: Path, cfg: RunConfig) -> Path:
    """Run whisperx on a single audio file. Returns the directory containing whisperx outputs."""
    whisperx_out = out_dir / "whisperx"
    whisperx_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "whisperx",
        str(audio),
        "--model", cfg.model,
        "--language", cfg.language,
        "--device", cfg.device,
        "--compute_type", cfg.compute_type,
        "--batch_size", str(cfg.batch_size),
        "--output_dir", str(whisperx_out),
        "--output_format", "json",
    ]
    if cfg.diarize:
        cmd.append("--diarize")
    if cfg.hf_token:
        cmd.extend(["--hf_token", cfg.hf_token])
    if cfg.whisperx_extra:
        cmd.extend(cfg.whisperx_extra)

    code = run_cmd(cmd)
    if code != 0:
        raise RuntimeError(f"whisperx failed with exit code {code} for {audio}")

    return whisperx_out


def guess_hyp_segments_jsonl(whisperx_out: Path) -> Optional[Path]:
    # 1) TXT を最優先（評価器が確実に読める）
    for p in [*(whisperx_out.glob("*.txt"))]:
        return p

    # 2) 既定の jsonl があれば
    for name in ("segments.jsonl",):
        p = whisperx_out / name
        if p.exists():
            return p

    # 3) その他の jsonl
    cands = list(whisperx_out.glob("*.jsonl"))
    if cands:
        return cands[0]

    # 4) （非推奨）json は最後の手段
    cands = list(whisperx_out.glob("*.json"))
    return cands[0] if cands else None



def run_eval(hyp_path: Path, ref_path: Path, out_dir: Path, cfg: RunConfig) -> Optional[Path]:
    metrics_path = out_dir / "metrics.json"

    # Prefer calling as a script to avoid import issues
    eval_entry = str(cfg.eval_script) if cfg.eval_script else "evaluate_char.py"

    cmd = [
        sys.executable,
        eval_entry,
        "--hyp", str(hyp_path),
        "--ref", str(ref_path),
        "--json-out", str(metrics_path),
    ]
    code = run_cmd(cmd)
    if code != 0:
        log(f"WARN: eval failed for {hyp_path.name} (exit {code})")
        return None
    return metrics_path


def run_html(hyp_path: Path, ref_path: Path, out_dir: Path, cfg: RunConfig) -> Optional[Path]:
    html_path = out_dir / "comparison.html"

    html_entry = str(cfg.html_script) if cfg.html_script else "report_html.py"

    cmd = [
        sys.executable,
        html_entry,
        "--hyp", str(hyp_path),
        "--ref", str(ref_path),
        "--output", str(out_dir),
        "--title", out_dir.name,
    ]
    code = run_cmd(cmd)
    if code != 0:
        log(f"WARN: html report failed for {hyp_path.name} (exit {code})")
        return None
    return html_path

def convert_whisperx_json_to_jsonl(json_path: Path, out_jsonl: Path) -> bool:
    """WhisperXの *.json から segments.jsonl を生成（話者を初出順に A,B,C... へ正規化）"""
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
        segments = data.get("segments") or []
        spk_map = {}
        def map_spk(s):
            if not s: return ""
            if s not in spk_map:
                spk_map[s] = chr(ord('A') + len(spk_map))
            return spk_map[s]
        with out_jsonl.open("w", encoding="utf-8") as f:
            for s in segments:
                f.write(json.dumps({
                    "start": s.get("start"),
                    "end": s.get("end"),
                    "speaker": map_spk(s.get("speaker")),
                    "text": s.get("text") or "",
                }, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        log(f"WARN: convert json->jsonl failed for {json_path.name}: {e}")
        return False

# ---------- Main ----------

def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Batch WhisperX + eval + HTML report")

    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", nargs="+", type=Path, help="Input audio files and/or directories to scan")
    src.add_argument("--filelist", type=Path, help="Text file with one audio path per line")

    p.add_argument("--pattern", default="*.wav", help="Glob pattern when scanning directories (default: *.wav)")
    p.add_argument("--ref-dir", type=Path, help="Directory containing reference transcripts (same stem)")

    p.add_argument("--output-root", type=Path, default=Path("research/speaker_diarization/runs/whisperx"))
    p.add_argument("--exp-name", help="Exact run folder name under output-root (avoid overwrite with numeric suffix)")
    p.add_argument("--exp-slug", help="Slug to append after timestamp to form run folder name")

    p.add_argument("--model", default="large-v2")
    p.add_argument("--language", default="ja")
    p.add_argument("--device", default="cuda", help="cuda|cpu (default: cuda)")
    p.add_argument("--compute-type", default="float16")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--diarize", action="store_true", help="Enable diarization in WhisperX")

    p.add_argument("--hf-token", help="Hugging Face token (or set HUGGINGFACE_TOKEN env)")
    p.add_argument("--whisperx-extra", nargs=argparse.REMAINDER, help="Extra args passed to whisperx after --")

    p.add_argument("--eval-script", type=Path, help="Path to evaluate_char.py (script mode)")
    p.add_argument("--html-script", type=Path, help="Path to report_html.py (script mode)")
    p.add_argument("--make-html", action="store_true", help="Generate HTML visual diff if ref exists")

    args = p.parse_args(argv)

    inputs: List[Path]
    if args.filelist:
        with open(args.filelist, "r", encoding="utf-8") as f:
            inputs = [Path(line.strip()) for line in f if line.strip()]
    else:
        inputs = list(args.inputs)

    hf_token = args.hf_token or os.getenv("HUGGINGFACE_TOKEN")

    cfg = RunConfig(
        inputs=inputs,
        pattern=args.pattern,
        ref_dir=args.ref_dir.resolve() if args.ref_dir else None,
        output_root=args.output_root,
        exp_name=args.exp_name,
        exp_slug=args.exp_slug,
        diarize=bool(args.diarize),
        make_html=bool(args.make_html),
        model=args.model,
        language=args.language,
        device=args.device,
        compute_type=args.compute_type,
        batch_size=int(args.batch_size),
        hf_token=hf_token,
        whisperx_extra=args.whisperx_extra or [],
        eval_script=args.eval_script,
        html_script=args.html_script,
    )

    run_dir = ensure_run_dir(cfg.output_root, cfg.exp_name, cfg.exp_slug)
    log(f"Run dir: {run_dir}")

    audio_files = find_audio_files(cfg.inputs, cfg.pattern)
    if not audio_files:
        log("No audio files found. Check --inputs / --filelist / --pattern.")
        return 2
    log(f"Found {len(audio_files)} audio files.")

    summary_rows: List[Tuple] = []
    summary_header = [
        "file",
        "hyp_path",
        "ref_path",
        "char_acc",
        "char_err_rate",
        "speaker_acc",
        "notes",
    ]

    for audio in audio_files:
        stem = audio.stem
        file_dir = run_dir / stem
        file_dir.mkdir(parents=True, exist_ok=True)

        # 1) WhisperX
        try:
            wx_dir = run_whisperx_on_file(audio, file_dir, cfg)
        except Exception as e:
            log(f"ERROR: whisperx failed for {audio}: {e}")
            summary_rows.append((str(audio), "", "", "", "", "", f"whisperx failed: {e}"))
            continue

        # 2) Locate hyp
        hyp = guess_hyp_segments_jsonl(wx_dir)
        if hyp is None:
            note = "no hyp json/jsonl found in whisperx output"
            log(f"WARN: {note} for {audio}")
            summary_rows.append((str(audio), "", "", "", "", "", note))
            continue
        # 2.5) 必要なら json → jsonl に正規化
        if hyp.suffix.lower() == ".json":
            out_jsonl = hyp.with_name("segments.jsonl")
            if convert_whisperx_json_to_jsonl(hyp, out_jsonl):
                hyp = out_jsonl

        # 3) Reference
        ref = find_reference(stem, cfg.ref_dir) if cfg.ref_dir else None
        if not ref:
            note = "no reference found; eval skipped"
            log(f"INFO: {note} for {audio}")
            summary_rows.append((str(audio), str(hyp), "", "", "", "", note))
            # Even without eval, we can still produce HTML if user supplied --make-html and ref exists (it doesn't)
            continue

        # 4) Eval
        metrics_path: Optional[Path] = run_eval(hyp, ref, file_dir, cfg)
        char_acc = char_err = spk_acc = ""
        if metrics_path and metrics_path.exists():
            try:
                data = json.loads(metrics_path.read_text(encoding="utf-8"))
                char_acc = data.get("char_accuracy", "")
                char_err = data.get("char_error_rate", "")
                spk_acc = data.get("speaker_accuracy", "")
            except Exception as e:
                log(f"WARN: could not parse metrics.json for {audio}: {e}")

        # 5) HTML
        if cfg.make_html:
            _ = run_html(hyp, ref, file_dir, cfg)

        summary_rows.append((str(audio), str(hyp), str(ref), char_acc, char_err, spk_acc, ""))

    # Write summary
    summary_csv = run_dir / "summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(summary_header)
        for row in summary_rows:
            w.writerow(row)

    log(f"Summary written: {summary_csv}")
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
