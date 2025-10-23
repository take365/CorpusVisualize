from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

from collections import Counter
from difflib import SequenceMatcher


def _iter_chars(text: str):
    for ch in text:
        if ch.isspace():
            continue
        yield ch


def load_segments_sequence(path: Path):
    sequence = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            speaker = str(payload.get("speaker", "?")).strip() or "?"
            text = str(payload.get("text", ""))
            for ch in _iter_chars(text):
                sequence.append((speaker, ch))
    return sequence


def load_reference_sequence(path: Path):
    suffix = path.suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return load_segments_sequence(path)

    sequence = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                speaker, text = line.split(":", 1)
            else:
                speaker, text = "?", line
            speaker = speaker.strip() or "?"
            for ch in _iter_chars(text):
                sequence.append((speaker, ch))
    return sequence


def compute_char_stats(reference, hypothesis) -> Tuple[int, int, int, int]:
    matcher = SequenceMatcher(None, reference, hypothesis)
    matches = substitutions = insertions = deletions = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        len_ref = i2 - i1
        len_hyp = j2 - j1
        if tag == "equal":
            matches += len_ref
        elif tag == "replace":
            substitutions += max(len_ref, len_hyp)
        elif tag == "insert":
            insertions += len_hyp
        elif tag == "delete":
            deletions += len_ref
    return matches, substitutions, insertions, deletions


def summarize(reference, hypothesis):
    matches, subs, ins, dels = compute_char_stats(reference, hypothesis)
    ref_len = len(reference)
    hyp_len = len(hypothesis)
    accuracy = matches / ref_len if ref_len else 0.0
    ser = (subs + ins + dels) / ref_len if ref_len else 0.0
    matcher = SequenceMatcher(None, reference, hypothesis)
    speaker_matches = 0
    speaker_total = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for off in range(i2 - i1):
                speaker_total += 1
                if reference[i1 + off][0] == hypothesis[j1 + off][0]:
                    speaker_matches += 1
    speaker_accuracy = speaker_matches / speaker_total if speaker_total else 0.0
    speaker_counts = Counter(spk for spk, _ in reference)

    return {
        "reference_length": ref_len,
        "hypothesis_length": hyp_len,
        "matches": matches,
        "substitutions": subs,
        "insertions": ins,
        "deletions": dels,
        "accuracy": accuracy,
        "error_rate": ser,
        "speaker_accuracy": speaker_accuracy,
        "speaker_matches": speaker_matches,
        "speaker_total": speaker_total,
        "speaker_counts": dict(speaker_counts),
    }


def print_report(summary: dict) -> None:
    print("=== 文字レベル評価 ===")
    print(f"参照文字数 : {summary['reference_length']}")
    print(f"推定文字数 : {summary['hypothesis_length']}")
    print(f"一致        : {summary['matches']}")
    print(f"置換        : {summary['substitutions']}")
    print(f"挿入        : {summary['insertions']}")
    print(f"削除        : {summary['deletions']}")
    print(f"正解率      : {summary['accuracy']:.4f}")
    print(f"誤り率      : {summary['error_rate']:.4f}")
    if summary.get("speaker_total", 0):
        print(f"話者正解率  : {summary['speaker_accuracy']:.4f} ({summary['speaker_matches']}/{summary['speaker_total']})")

    print("\n参照側の話者別文字数:")
    for speaker, count in summary["speaker_counts"].items():
        print(f"  {speaker}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Character-level evaluation between hypothesis and reference")
    parser.add_argument("--hyp", required=True, type=Path, help="Hypothesis segments.jsonl or text file")
    parser.add_argument("--ref", required=True, type=Path, help="Reference JSONL or text file")
    parser.add_argument("--json-out", type=Path, help="評価結果を JSON 形式で保存するパス")

    args = parser.parse_args()

    hyp_sequence = load_segments_sequence(args.hyp) if args.hyp.suffix.lower() in {".jsonl", ".json"} else load_reference_sequence(args.hyp)
    ref_sequence = load_reference_sequence(args.ref)
    summary = summarize(ref_sequence, hyp_sequence)
    print_report(summary)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        data = summary.copy()
        args.json_out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
