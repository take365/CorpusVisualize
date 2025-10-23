from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Sequence

BASE_DIR = Path(__file__).resolve().parents[3]
AUDIO_DIR = BASE_DIR / "research" / "speaker_diarization" / "data"
REFERENCE_PATH = AUDIO_DIR / "LD01-Dialogue-01.txt"
RUNS_BASE = BASE_DIR / "research" / "speaker_diarization" / "runs"
REPORT_BASE = BASE_DIR / "research" / "speaker_diarization" / "report"
SUMMARY_PATH = REPORT_BASE / "pre_llm_summary.md"

PYTHON = sys.executable


@dataclass
class Experiment:
    group: str
    name: str
    params: Sequence[str]


ENERGY_EXPERIMENTS: List[Experiment] = [
    Experiment("energy", "energy_default", ["--diar", "energy_basic"]),
    Experiment("energy", "energy_min_1s", ["--diar", "energy_basic", "--min-seg-sec", "1.0", "--max-seg-sec", "25.0"]),
    Experiment("energy", "energy_min_0p5s", ["--diar", "energy_basic", "--min-seg-sec", "0.5", "--max-seg-sec", "15.0"]),
]

PYANNOTE_EXPERIMENTS: List[Experiment] = [
    Experiment("pyannote", "py_default", ["--diar", "pyannote"]),
    Experiment("pyannote", "py_th060", ["--diar", "pyannote", "--pyannote-threshold", "0.60"]),
    Experiment("pyannote", "py_th065", ["--diar", "pyannote", "--pyannote-threshold", "0.65"]),
    Experiment("pyannote", "py_th070", ["--diar", "pyannote", "--pyannote-threshold", "0.70"]),
    Experiment("pyannote", "py_th075", ["--diar", "pyannote", "--pyannote-threshold", "0.75"]),
    Experiment("pyannote", "py_th080", ["--diar", "pyannote", "--pyannote-threshold", "0.80"]),
    Experiment("pyannote", "py_th070_cluster6", ["--diar", "pyannote", "--pyannote-threshold", "0.70", "--pyannote-min-cluster-size", "6"]),
    Experiment("pyannote", "py_th070_cluster8", ["--diar", "pyannote", "--pyannote-threshold", "0.70", "--pyannote-min-cluster-size", "8"]),
    Experiment("pyannote", "py_th070_off02", ["--diar", "pyannote", "--pyannote-threshold", "0.70", "--pyannote-min-duration-off", "0.2"]),
    Experiment("pyannote", "py_th070_off04", ["--diar", "pyannote", "--pyannote-threshold", "0.70", "--pyannote-min-duration-off", "0.4"]),
]

EXPERIMENTS = ENERGY_EXPERIMENTS + PYANNOTE_EXPERIMENTS


def run_command(cmd: List[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("\n[RUN]", " ".join(cmd))
    return subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)


def find_latest_run(exp: Experiment) -> Path | None:
    base = RUNS_BASE / exp.group / exp.name
    if not base.exists():
        return None
    for run_dir in sorted([d for d in base.iterdir() if d.is_dir()], reverse=True):
        if (run_dir / "metrics.json").exists():
            return run_dir
    return None


def find_latest_html(exp: Experiment) -> str | None:
    base = REPORT_BASE / exp.group / exp.name
    if not base.exists():
        return None
    for folder in sorted([d for d in base.iterdir() if d.is_dir()], reverse=True):
        html_file = folder / "comparison.html"
        if html_file.exists():
            return str(html_file.relative_to(BASE_DIR))
    return None


def main() -> None:
    RUNS_BASE.mkdir(parents=True, exist_ok=True)
    REPORT_BASE.mkdir(parents=True, exist_ok=True)

    results = []

    for exp in EXPERIMENTS:
        run_dir = find_latest_run(exp)
        if run_dir is None:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            run_dir = RUNS_BASE / exp.group / exp.name / timestamp
            run_dir.mkdir(parents=True, exist_ok=True)

            pipeline_cmd = [
                PYTHON,
                "-m",
                "pipeline.run_pipeline",
                "--input-dir",
                str(AUDIO_DIR),
                "--output-dir",
                str(run_dir),
                "--limit",
                "1",
                "--no-llm",
            ] + list(exp.params)

            run_result = run_command(pipeline_cmd)
            print(run_result.stdout)
        else:
            print(f"[SKIP] {exp.group}/{exp.name} -> {run_dir.name} を利用 (再評価のみ)")

        conversation_dirs = [d for d in run_dir.iterdir() if d.is_dir()]
        if not conversation_dirs:
            raise RuntimeError(f"No conversation output found in {run_dir}")
        conversation_dir = conversation_dirs[0]
        hyp_path = conversation_dir / "segments.jsonl"

        metrics_path = run_dir / "metrics.json"
        eval_cmd = [
            PYTHON,
            "-m",
            "research.speaker_diarization.src.evaluate_char",
            "--hyp",
            str(hyp_path),
            "--ref",
            str(REFERENCE_PATH),
            "--json-out",
            str(metrics_path),
        ]
        eval_result = run_command(eval_cmd)
        print(eval_result.stdout)

        report_dir = REPORT_BASE / exp.group / exp.name
        report_cmd = [
            PYTHON,
            "-m",
            "research.speaker_diarization.src.report_html",
            "--hyp",
            str(hyp_path),
            "--ref",
            str(REFERENCE_PATH),
            "--output",
            str(report_dir),
            "--theme",
            "light",
            "--title",
            f"{exp.group} / {exp.name}",
            "--metrics",
            str(metrics_path),
        ]
        report_result = run_command(report_cmd)
        print(report_result.stdout)
        html_path = None
        for line in report_result.stdout.splitlines():
            if line.startswith("HTMLレポートを生成しました:"):
                html_path = line.split(":", 1)[1].strip()
                break

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        html_rel = None
        if html_path:
            try:
                html_rel = str(Path(html_path).resolve().relative_to(BASE_DIR))
            except ValueError:
                html_rel = html_path
        results.append(
            {
                "group": exp.group,
                "name": exp.name,
                "params": " ".join(exp.params),
                "run_dir": str(run_dir.relative_to(BASE_DIR)),
                "html": html_rel,
                "metrics": metrics,
            }
        )

    write_summary(results)


def write_summary(results: List[dict]) -> None:
    lines: List[str] = ["# LLM 導入前 話者分離まとめ", ""]

    best_energy = max(
        (r for r in results if r["group"] == "energy"),
        key=lambda x: x["metrics"]["accuracy"],
    )
    best_py = max(
        (r for r in results if r["group"] == "pyannote"),
        key=lambda x: x["metrics"]["accuracy"],
    )

    lines.append("## 概要")
    lines.append("- Energy 方式は正解率 55〜71% 程度。短区間にすると挿入が増えるものの正解率は向上。")
    lines.append("- pyannote.audio 方式はチューニングによって 35〜48% 程度。threshold=0.80 が最も良好。")
    lines.append(
        f"- 最高スコア: Energy={best_energy['metrics']['accuracy']:.1%} ({best_energy['name']}), pyannote={best_py['metrics']['accuracy']:.1%} ({best_py['name']})."
    )
    lines.append("- 話者正解率は Energy 系で 58〜77%、pyannote 系で 48〜72% 程度。一致文字のみ評価しても話者入れ替わりが残る。")
    lines.append("")

    lines.append("## 詳細結果")
    lines.append("| グループ | 実験名 | 主要パラメータ | 文字正解率 | 文字誤り率 | 話者正解率 | 一致/置換/挿入/削除 | 結果フォルダ | HTML レポート |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    for item in results:
        metrics = item["metrics"]
        matches = metrics["matches"]
        subs = metrics["substitutions"]
        ins = metrics["insertions"]
        dels = metrics["deletions"]
        acc = metrics["accuracy"]
        err = metrics["error_rate"]
        params_display = item["params"] or "(default)"
        run_link = item["run_dir"]
        spk_acc = metrics.get("speaker_accuracy", 0.0)
        html_link = item["html"] or "-"
        lines.append(
            f"| {item['group']} | {item['name']} | {params_display} | {acc:.2%} | {err:.2%} | {spk_acc:.2%} | {matches}/{subs}/{ins}/{dels} | {run_link} | {html_link} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nサマリーを出力しました: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
