#!/usr/bin/env python3
"""Run quick sentiment experiments on pipeline segments."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import plotly.express as px
import torch
from mlask import MLAsk
from plotly.offline import plot as plot_html
from transformers import AutoModelForMaskedLM, AutoTokenizer


def default_output_dir(conversation_id: str) -> Path:
    return Path("research/sentiment_trials/output") / conversation_id


def load_segments(path: Path) -> List[dict]:
    segments: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            segments.append(json.loads(line))
    return segments


def ensure_mecab_arg() -> Optional[str]:
    """Try to build a MeCab argument pointing to an installed dictionary."""

    try:
        import ipadic

        dicdir = Path(ipadic.DICDIR)
        mecabrc = dicdir / "mecabrc"
        if mecabrc.exists():
            return f"-d {dicdir} -r {mecabrc}"
    except ImportError:
        pass

    try:
        import unidic_lite

        dicdir = Path(unidic_lite.DICDIR)
        mecabrc = dicdir / "mecabrc"
        if mecabrc.exists():
            return f"-d {dicdir} -r {mecabrc}"
    except ImportError:
        pass

    return None


@dataclass
class BertMaskedSentiment:
    model_name: str = "tohoku-nlp/bert-base-japanese-whole-word-masking"

    def __post_init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
        self.model.eval()
        self.positive_vocab = self._collect_token_ids(
            ["楽しい", "幸せ", "最高", "面白い", "素晴らしい", "大好き", "感動", "好調", "快適"]
        )
        self.negative_vocab = self._collect_token_ids(
            ["嫌い", "最悪", "不満", "ひどい", "怒り", "苦しい"]
        )
        if not self.positive_vocab or not self.negative_vocab:
            raise RuntimeError("Failed to build sentiment vocab lists for BERT mask scoring.")

    def _collect_token_ids(self, words: Iterable[str]) -> List[int]:
        ids: List[int] = []
        for word in words:
            pieces = self.tokenizer.encode(word, add_special_tokens=False)
            if len(pieces) == 1:
                ids.append(pieces[0])
        return ids

    def score_text(self, text: str) -> Dict[str, float]:
        template = f"{text} 全体的な感情は{self.tokenizer.mask_token}だ。"
        inputs = self.tokenizer(
            template,
            return_tensors="pt",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
        )
        mask_idx = (inputs["input_ids"][0] == self.tokenizer.mask_token_id).nonzero(as_tuple=True)
        if not mask_idx[0].numel():
            return {
                "bert_pos_score": 0.0,
                "bert_neg_score": 0.0,
                "bert_score": 0.0,
                "bert_label": "neutral",
            }
        mask_position = mask_idx[0].item()
        with torch.no_grad():
            logits = self.model(**inputs).logits[0, mask_position]
            probs = torch.softmax(logits, dim=-1)
            pos_score = float(probs[self.positive_vocab].sum().item())
            neg_score = float(probs[self.negative_vocab].sum().item())
            score = pos_score - neg_score
        label = "positive" if score > 0 else "negative" if score < 0 else "neutral"
        return {
            "bert_pos_score": pos_score,
            "bert_neg_score": neg_score,
            "bert_score": score,
            "bert_label": label,
        }


def run_pymlask(analyzer: MLAsk, text: str) -> Dict[str, object]:
    result = analyzer.analyze(text)
    orientation = result.get("orientation") if result else None
    emotions_dict = result.get("emotion") if result else None
    if emotions_dict:
        # Convert defaultdict(list) to plain dict for JSON serialisation.
        emotions: Dict[str, List[str]] = {k: list(v) for k, v in emotions_dict.items()}
    else:
        emotions = {}
    representative = result.get("representative") if result else None
    rep_label, rep_words = (None, [])
    if isinstance(representative, tuple) and len(representative) == 2:
        rep_label, rep_words = representative
    return {
        "pymlask_orientation": orientation,
        "pymlask_activation": result.get("activation") if result else None,
        "pymlask_emotions": emotions,
        "pymlask_representative_label": rep_label,
        "pymlask_representative_words": rep_words,
    }


def build_html_outputs(df, out_dir: Path) -> None:
    plot_frame = df.copy()
    plot_frame["pymlask_orientation"] = plot_frame["pymlask_orientation"].fillna("NONE")
    scatter = px.scatter(
        plot_frame,
        x="start",
        y="bert_score",
        color="pymlask_orientation",
        hover_name="segment_id",
        hover_data={"speaker": True, "text": True, "bert_pos_score": ':.3f', "bert_neg_score": ':.3f'},
        title="BERT masked-LM sentiment vs time",
    )
    plot_html(scatter, filename=str(out_dir / "bert_scores.html"), auto_open=False)

    emotion_counts = Counter()
    for emotions in df["pymlask_emotions"]:
        for key in emotions.keys():
            emotion_counts[key] += 1
    if emotion_counts:
        rows = sorted(emotion_counts.items(), key=lambda kv: kv[1], reverse=True)
        emo_fig = px.bar(x=[k for k, _ in rows], y=[v for _, v in rows], title="ML-Ask emotion counts")
        emo_fig.update_layout(xaxis_title="Emotion label", yaxis_title="Count")
        plot_html(emo_fig, filename=str(out_dir / "pymlask_emotions.html"), auto_open=False)

    build_conversation_view(df, out_dir)


def build_conversation_view(df, out_dir: Path) -> None:
    palette = ["#6c5ce7", "#00b894", "#0984e3", "#e17055", "#fdcb6e", "#2d3436"]
    speaker_colors: Dict[str, str] = {}
    for idx, speaker in enumerate(df["speaker"].fillna("?").unique()):
        speaker_colors[speaker] = palette[idx % len(palette)]

    bert_colors = {
        "positive": ("#ffe3e3", "#c92a2a"),
        "negative": ("#e7f5ff", "#1c7ed6"),
        "neutral": ("#f1f3f5", "#495057"),
    }
    orientation_colors = {
        "POSITIVE": ("#ffe3e3", "#c92a2a"),
        "NEGATIVE": ("#e7f5ff", "#1c7ed6"),
        "NEUTRAL": ("#f1f3f5", "#495057"),
    }

    segments_html: List[str] = []
    sorted_df = df.sort_values("start")

    conversation_id = ""
    if "conversation_id" in df.columns:
        non_null = df["conversation_id"].dropna()
        if not non_null.empty:
            conversation_id = str(non_null.iloc[0])

    for _, row in sorted_df.iterrows():
        speaker = row.get("speaker") or "?"
        text = str(row.get("text") or "").strip()
        start = row.get("start")
        end = row.get("end")

        bert_label = row.get("bert_label")
        if not isinstance(bert_label, str) or not bert_label:
            bert_label = "neutral"
        bert_label = bert_label.lower()
        bert_bg, bert_fg = bert_colors.get(bert_label, ("#f1f3f5", "#495057"))
        bert_score = row.get("bert_score")
        if bert_score is None or (isinstance(bert_score, float) and math.isnan(bert_score)):
            bert_score = 0.0

        py_orientation = row.get("pymlask_orientation")
        if isinstance(py_orientation, float) and math.isnan(py_orientation):
            py_orientation = None
        py_display = py_orientation if py_orientation else "未検出"
        py_bg, py_fg = orientation_colors.get(py_orientation, ("#f1f3f5", "#495057"))

        emotions_obj = row.get("pymlask_emotions") or {}
        if isinstance(emotions_obj, float) and math.isnan(emotions_obj):
            emotions_obj = {}
        if isinstance(emotions_obj, str):
            emotions_summary = emotions_obj
        else:
            collected: List[str] = []
            if isinstance(emotions_obj, dict):
                for label, words in emotions_obj.items():
                    word_str = "、".join(words) if isinstance(words, (list, tuple)) else str(words)
                    collected.append(f"{label}: {word_str}")
            emotions_summary = " / ".join(collected)

        rep_words = row.get("pymlask_representative_words") or []
        if isinstance(rep_words, str):
            rep_words_display = rep_words
        else:
            rep_words_display = "、".join(str(w) for w in rep_words)

        speaker_color = speaker_colors.get(speaker, "#868e96")
        escaped_text = html.escape(text) if text else "(無音 / 空白)"
        time_window = "" if start is None or end is None else f"{start:.2f}s – {end:.2f}s"

        segment_block = f"""
        <div class=\"segment\" style=\"border-left:6px solid {speaker_color};\">
          <div class=\"segment__meta\">
            <span class=\"segment__speaker\">{html.escape(str(speaker))}</span>
            <span class=\"segment__time\">{time_window}</span>
          </div>
          <div class=\"segment__text\">{escaped_text}</div>
          <div class=\"segment__labels\">
            <span class=\"badge\" style=\"background:{bert_bg};color:{bert_fg};\">BERT: {bert_label}</span>
            <span class=\"badge\" style=\"background:{py_bg};color:{py_fg};\">ML-Ask: {py_display}</span>
            <span class=\"detail\">score={bert_score:.3f}</span>
          </div>
        """

        if emotions_summary:
            segment_block += f"<div class=\"segment__extra\">感情候補: {html.escape(emotions_summary)}</div>"
        if rep_words_display:
            segment_block += f"<div class=\"segment__extra\">代表語: {html.escape(rep_words_display)}</div>"

        segment_block += "</div>"
        segments_html.append(segment_block)

    page = f"""<!DOCTYPE html>
    <html lang=\"ja\">
      <head>
        <meta charset=\"utf-8\" />
        <title>Conversation Sentiment View</title>
        <style>
          body {{
            font-family: 'Segoe UI', 'Hiragino Sans', 'Helvetica Neue', Arial, sans-serif;
            background:#fafafa;
            color:#212529;
            margin:0;
            padding:24px;
          }}
          h1 {{
            font-size:1.6rem;
            margin-bottom:0.4rem;
          }}
          .subtitle {{
            color:#868e96;
            margin-bottom:24px;
          }}
          .segment {{
            background:#fff;
            border-radius:12px;
            box-shadow:0 2px 6px rgba(33,37,41,0.08);
            padding:16px 20px;
            margin-bottom:16px;
            border-left:6px solid #4dabf7;
          }}
          .segment__meta {{
            display:flex;
            gap:12px;
            font-size:0.9rem;
            color:#495057;
            margin-bottom:8px;
            font-weight:600;
          }}
          .segment__speaker {{
            text-transform:uppercase;
            letter-spacing:0.05em;
          }}
          .segment__text {{
            font-size:1.05rem;
            line-height:1.6;
            margin-bottom:12px;
          }}
          .segment__labels {{
            display:flex;
            gap:12px;
            flex-wrap:wrap;
            align-items:center;
            font-size:0.9rem;
            color:#495057;
          }}
          .segment__extra {{
            margin-top:6px;
            font-size:0.85rem;
            color:#495057;
            background:#f8f9fa;
            padding:6px 10px;
            border-radius:8px;
          }}
          .badge {{
            border-radius:999px;
            padding:2px 12px;
            font-weight:600;
            letter-spacing:0.02em;
          }}
          .detail {{
            color:#868e96;
          }}
        </style>
      </head>
      <body>
        <h1>会話セグメント感情ビュー</h1>
        <div class=\"subtitle\">セグメント数: {len(sorted_df)} / 会話ID: {html.escape(conversation_id) if conversation_id else "不明"}</div>
        {''.join(segments_html)}
      </body>
    </html>
    """

    output_path = out_dir / "conversation_view.html"
    output_path.write_text(page, encoding="utf-8")


def summarise(results: List[dict]) -> Dict[str, object]:
    orientation_counts = Counter(r.get("pymlask_orientation") for r in results if r.get("pymlask_orientation"))
    bert_scores = [r.get("bert_score", 0.0) for r in results]
    summary = {
        "pymlask_orientation_counts": dict(orientation_counts),
        "bert_score_min": min(bert_scores) if bert_scores else 0.0,
        "bert_score_max": max(bert_scores) if bert_scores else 0.0,
        "bert_score_mean": sum(bert_scores) / len(bert_scores) if bert_scores else 0.0,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentiment experiments on pipeline outputs")
    parser.add_argument(
        "segments",
        type=Path,
        nargs="?",
        help="Path to segments.jsonl. If omitted, --conversation-id is used.",
    )
    parser.add_argument(
        "--conversation-id",
        type=str,
        help="Conversation ID under output/pipeline to load segments from.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory to store experiment outputs (default: research/sentiment_trials/output/<conversation_id>)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optionally limit the number of segments processed.",
    )
    args = parser.parse_args()

    if args.segments:
        segments_path = args.segments
        conversation_id = args.conversation_id or segments_path.parent.name
    else:
        if not args.conversation_id:
            parser.error("Either segments path or --conversation-id must be supplied")
        conversation_id = args.conversation_id
        segments_path = Path("output/pipeline") / conversation_id / "segments.jsonl"

    if not segments_path.exists():
        raise FileNotFoundError(f"Could not find segments at {segments_path}")

    output_dir = args.output_dir or default_output_dir(conversation_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    mecab_arg = ensure_mecab_arg()
    if mecab_arg:
        # Ensure MeCab picks up the bundled rc file when Python is executed elsewhere.
        parts = mecab_arg.split()
        if "-r" in parts:
            rc_index = parts.index("-r")
            if rc_index + 1 < len(parts):
                os.environ.setdefault("MECABRC", parts[rc_index + 1])
    analyzer = MLAsk(mecab_arg=mecab_arg) if mecab_arg else MLAsk()
    bert = BertMaskedSentiment()

    segments = load_segments(segments_path)
    if args.limit:
        segments = segments[: args.limit]

    enriched: List[dict] = []
    for seg in segments:
        base = {
            "segment_id": seg.get("id"),
            "conversation_id": seg.get("conversation_id"),
            "speaker": seg.get("speaker"),
            "start": seg.get("start"),
            "end": seg.get("end"),
            "text": seg.get("text", ""),
        }
        text = base["text"].strip()
        if not text:
            base.update(
                {
                    "pymlask_orientation": None,
                    "pymlask_activation": None,
                    "pymlask_emotions": {},
                    "pymlask_representative_label": None,
                    "pymlask_representative_words": [],
                    "bert_pos_score": 0.0,
                    "bert_neg_score": 0.0,
                    "bert_score": 0.0,
                    "bert_label": "neutral",
                }
            )
            enriched.append(base)
            continue
        py_result = run_pymlask(analyzer, text)
        bert_result = bert.score_text(text)
        merged = {**base, **py_result, **bert_result}
        enriched.append(merged)

    summary = summarise(enriched)
    payload = {
        "conversation_id": conversation_id,
        "segments_path": str(segments_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "segments": enriched,
    }

    json_path = output_dir / "analysis_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # Build visualisations
    import pandas as pd  # local import to avoid hard dependency at module import time

    frame = pd.DataFrame(enriched)
    if not frame.empty and frame["text"].notna().any():
        build_html_outputs(frame, output_dir)

    print(f"Wrote analysis to {json_path}")


if __name__ == "__main__":
    main()
