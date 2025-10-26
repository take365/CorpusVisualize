from __future__ import annotations

import datetime as dt
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SpeakerChar = Tuple[str, str]
AlignmentItem = Tuple[Optional[SpeakerChar], Optional[SpeakerChar]]


@dataclass(frozen=True)
class ComparisonResult:
    html_path: Path
    metrics: Dict[str, float]


def _iter_chars(text: str) -> Iterable[str]:
    for ch in text:
        if ch.isspace():
            continue
        yield ch


def _load_hypothesis(path: Path) -> List[SpeakerChar]:
    utterances: List[SpeakerChar] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            payload = json.loads(line)
            speaker = str(payload.get("speaker", "?")) or "?"
            text = str(payload.get("text", ""))
            for ch in _iter_chars(text):
                utterances.append((speaker, ch))
    return utterances


def _load_reference(path: Path) -> List[SpeakerChar]:
    if path.suffix.lower() in {".jsonl", ".json"}:
        return _load_hypothesis(path)

    utterances: List[SpeakerChar] = []
    with path.open("r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        if ":" in line:
            speaker, text = line.split(":", 1)
        else:
            speaker, text = "?", line
        speaker = speaker.strip() or "?"
        for ch in _iter_chars(text):
            utterances.append((speaker, ch))
        if idx < len(lines) - 1:
            utterances.append((speaker, "\n"))
    return utterances


def _lcs_align(ref: List[SpeakerChar], hyp: List[SpeakerChar]) -> List[AlignmentItem]:
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1][1] == hyp[j - 1][1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    alignment: List[AlignmentItem] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1][1] == hyp[j - 1][1] and dp[i][j] == dp[i - 1][j - 1] + 1:
            alignment.append((ref[i - 1], hyp[j - 1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j - 1] >= dp[i - 1][j]):
            alignment.append((None, hyp[j - 1]))
            j -= 1
        else:
            alignment.append((ref[i - 1], None))
            i -= 1

    alignment.reverse()
    return alignment


def _compute_metrics(alignment: List[AlignmentItem]) -> Dict[str, float]:
    matches = substitutions = insertions = deletions = 0
    speaker_matches = speaker_total = 0

    for ref_item, hyp_item in alignment:
        if ref_item and hyp_item:
            ref_s, ref_c = ref_item
            hyp_s, hyp_c = hyp_item
            if ref_c == hyp_c:
                matches += 1
                speaker_total += 1
                if ref_s == hyp_s:
                    speaker_matches += 1
            else:
                substitutions += 1
                speaker_total += 1
        elif ref_item and not hyp_item:
            deletions += 1
        elif hyp_item and not ref_item:
            insertions += 1

    ref_len = matches + substitutions + deletions
    hyp_len = matches + substitutions + insertions
    accuracy = matches / ref_len if ref_len else 0.0
    error_rate = (substitutions + insertions + deletions) / ref_len if ref_len else 0.0
    speaker_accuracy = speaker_matches / speaker_total if speaker_total else 0.0

    return {
        "matches": matches,
        "substitutions": substitutions,
        "insertions": insertions,
        "deletions": deletions,
        "ref_len": ref_len,
        "hyp_len": hyp_len,
        "accuracy": accuracy,
        "error_rate": error_rate,
        "speaker_accuracy": speaker_accuracy,
    }


_COLOR_CLASSES = ["A", "B", "C", "D", "E", "F", "G"]


def _speaker_class_map(ref: List[SpeakerChar], hyp: List[SpeakerChar]) -> Dict[str, str]:
    speakers: List[str] = []
    for spk, _ in ref + hyp:
        if spk not in speakers:
            speakers.append(spk)
    mapping: Dict[str, str] = {}
    for idx, spk in enumerate(speakers):
        mapping[spk] = _COLOR_CLASSES[idx % len(_COLOR_CLASSES)]
    return mapping


def _render_alignment_rows(alignment: List[AlignmentItem], mapping: Dict[str, str]) -> Tuple[str, str]:
    ref_lines: List[str] = []
    hyp_lines: List[str] = []
    cur_ref: List[str] = []
    cur_hyp: List[str] = []

    def push_line() -> None:
        ref_lines.append("".join(cur_ref) if cur_ref else "<span class='char gap'>∅</span>")
        hyp_lines.append("".join(cur_hyp) if cur_hyp else "<span class='char gap mismatch'>∅</span>")
        cur_ref.clear()
        cur_hyp.clear()

    for ref_item, hyp_item in alignment:
        if ref_item:
            ref_spk, ref_char = ref_item
            ref_class = mapping.get(ref_spk, "")
            if ref_char == "\n":
                cur_ref.append(
                    f"<span class='char newline {ref_class}' data-spk='{html.escape(ref_spk)}'>↵</span>"
                )
                push_line()
                continue
            label = html.escape(ref_char) if ref_char else "&nbsp;"
            cur_ref.append(
                f"<span class='char {ref_class}' data-spk='{html.escape(ref_spk)}'>{label}</span>"
            )
        else:
            cur_ref.append("<span class='char gap'>∅</span>")

        if hyp_item:
            hyp_spk, hyp_char = hyp_item
            hyp_class = mapping.get(hyp_spk, "")
            if hyp_char == "\n":
                cur_hyp.append(
                    f"<span class='char newline {hyp_class} mismatch' data-spk='{html.escape(hyp_spk)}'>↵</span>"
                )
                push_line()
                continue
            label = html.escape(hyp_char) if hyp_char else "&nbsp;"
            status = "match" if ref_item and hyp_item and ref_item[1] == hyp_item[1] and ref_item[0] == hyp_item[0] else "mismatch"
            cur_hyp.append(
                f"<span class='char {hyp_class} {status}' data-spk='{html.escape(hyp_spk)}'>{label}</span>"
            )
        else:
            cur_hyp.append("<span class='char gap mismatch'>∅</span>")

    if cur_ref or cur_hyp:
        push_line()

    ref_html = "<br/>".join(ref_lines)
    hyp_html = "<br/>".join(hyp_lines)
    return ref_html, hyp_html


def _render_summary_html(metrics: Dict[str, float], mapping: Dict[str, str]) -> str:
    chips = [
        f"<span class='chip'><span class='dot {cls}'></span>{html.escape(spk)}</span>"
        for spk, cls in mapping.items()
    ]
    return """
    <div class='summary'>
      <div>参照文字数: {ref_len} / 推定文字数: {hyp_len}</div>
      <div>一致: {matches} / 置換: {subs} / 挿入: {ins} / 削除: {dels}</div>
      <div>文字正解率: {acc:.2%} / 文字誤り率: {err:.2%} / 話者正解率: {spk_acc:.2%}</div>
    </div>
    <div class='legend'>{chips}</div>
    """.format(
        ref_len=int(metrics["ref_len"]),
        hyp_len=int(metrics["hyp_len"]),
        matches=int(metrics["matches"]),
        subs=int(metrics["substitutions"]),
        ins=int(metrics["insertions"]),
        dels=int(metrics["deletions"]),
        acc=metrics["accuracy"],
        err=metrics["error_rate"],
        spk_acc=metrics["speaker_accuracy"],
        chips=" ".join(chips),
    )


def _resolve_colors(theme: str) -> Dict[str, str]:
    if theme == "light":
        return {
            "bg": "#f8fafc",
            "panel": "#ffffff",
            "text": "#1f2933",
            "border": "rgba(148,163,184,0.35)",
            "accent": "#0ea5e9",
            "muted": "#64748b",
            "seq_bg": "rgba(15,23,42,0.05)",
        }
    return {
        "bg": "#0b1220",
        "panel": "rgba(15,23,42,0.8)",
        "text": "#e2e8f0",
        "border": "rgba(148,163,184,0.2)",
        "accent": "#60a5fa",
        "muted": "#94a3b8",
        "seq_bg": "rgba(11,19,32,0.7)",
    }


def _build_html(
    alignment: List[AlignmentItem],
    metrics: Dict[str, float],
    mapping: Dict[str, str],
    title: str,
    theme: str,
) -> str:
    ref_html, hyp_html = _render_alignment_rows(alignment, mapping)
    summary_block = _render_summary_html(metrics, mapping)
    colors = _resolve_colors(theme)
    timestamp = dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""
<!DOCTYPE html>
<html lang='ja'>
<head>
  <meta charset='utf-8'>
  <title>{html.escape(title)}</title>
  <style>
    body {{ background: {colors['bg']}; color: {colors['text']}; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Meiryo, sans-serif; margin: 0; padding: 24px; }}
    h1 {{ font-size: 20px; margin-bottom: 8px; }}
    .meta {{ color: {colors['muted']}; margin-bottom: 16px; }}
    .panel {{ background: {colors['panel']}; border: 1px solid {colors['border']}; border-radius: 12px; padding: 16px; box-shadow: 0 8px 26px rgba(15,23,42,0.2); }}
    .summary {{ margin-bottom: 12px; line-height: 1.6; }}
    .legend {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }}
    .chip {{ display: inline-flex; align-items: center; gap: 6px; padding: 6px 10px; border-radius: 9999px; background: rgba(148,163,184,0.12); border: 1px solid {colors['border']}; font-size: 12px; }}
    .dot {{ width: 10px; height: 10px; border-radius: 9999px; display: inline-block; }}
    .columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-top: 16px; }}
    .column h2 {{ font-size: 14px; margin: 0 0 8px 0; color: #cbd5f5; }}
    .sequence {{ display: block; padding: 12px; border-radius: 10px; background: {colors['seq_bg']}; border: 1px solid {colors['border']}; min-height: 60px; overflow: auto; }}
    .char {{ display: inline-block; min-width: 18px; min-height: 24px; padding: 2px 4px; border-radius: 6px; border: 1px solid rgba(94,114,166,0.4); font-family: 'Fira Mono', monospace; text-align: center; margin: 1px; }}
    .match {{ box-shadow: inset 0 0 0 2px rgba(34,197,94,0.8); }}
    .mismatch {{ box-shadow: inset 0 0 0 2px rgba(239,68,68,0.7); }}
    .gap {{ color: #94a3b8; font-style: italic; }}
    .newline {{ background: rgba(148,163,184,0.2); border-style: dashed; color: #cbd5f5; }}
    .A {{ background: rgba(31,119,180,0.25); border-color: rgba(31,119,180,0.55); }}
    .B {{ background: rgba(44,160,44,0.25); border-color: rgba(44,160,44,0.55); }}
    .C {{ background: rgba(214,39,40,0.25); border-color: rgba(214,39,40,0.55); }}
    .D {{ background: rgba(148,103,189,0.25); border-color: rgba(148,103,189,0.55); }}
    .E {{ background: rgba(140,86,75,0.25); border-color: rgba(140,86,75,0.55); }}
    .F {{ background: rgba(227,119,194,0.25); border-color: rgba(227,119,194,0.55); }}
    .G {{ background: rgba(127,127,127,0.25); border-color: rgba(127,127,127,0.55); }}
  </style>
</head>
<body>
  <div class='panel'>
    <h1>{html.escape(title)}</h1>
    <div class='meta'>生成日時: {timestamp}</div>
    {summary_block}
    <div class='columns'>
      <div class='column'>
        <h2>正解</h2>
        <div class='sequence'>
          {ref_html}
        </div>
      </div>
      <div class='column'>
        <h2>推定</h2>
        <div class='sequence'>
          {hyp_html}
        </div>
      </div>
    </div>
  </div>
</body>
</html>
"""


def generate_comparison_html(
    hypothesis_path: Path,
    reference_path: Path,
    output_dir: Path,
    *,
    title: Optional[str] = None,
    theme: str = "light",
    metrics_override: Optional[Dict[str, float]] = None,
    timestamped: bool = True,
) -> ComparisonResult:
    hyp_seq = _load_hypothesis(hypothesis_path)
    ref_seq = _load_reference(reference_path)
    alignment = _lcs_align(ref_seq, hyp_seq)
    metrics = metrics_override or _compute_metrics(alignment)
    mapping = _speaker_class_map(ref_seq, hyp_seq)
    html_content = _build_html(
        alignment,
        metrics,
        mapping,
        title=title or hypothesis_path.stem,
        theme=theme,
    )

    if timestamped:
        timestamp_dir = dt.datetime.now(dt.UTC).strftime("%Y%m%d_%H%M%S")
        out_dir = output_dir / timestamp_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "comparison.html"
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "comparison.html"
    out_file.write_text(html_content, encoding="utf-8")
    return ComparisonResult(html_path=out_file, metrics=metrics)


__all__ = ["ComparisonResult", "generate_comparison_html"]
