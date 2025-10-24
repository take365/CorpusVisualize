#!/usr/bin/env python3
# HMD/src/transcribe_whisper_pure.py
# -------------------------------------------------------------
# OpenAI Whisper (純正) で文字起こしし、transcribe_whisper.py と
# ほぼ同じ入出力（plain.txt / segments.jsonl / words.jsonl）を作る。
# ※ 注意: 純正Whisperは単語タイムスタンプを提供しないため、
#   words.jsonl はセグメント時間をテキストの「見かけの語」に
#   均等割りした "approx"（近似）タイムスタンプを出力する。
#   精密な word-level が必要なら Faster-Whisper / WhisperX を使用。
# -------------------------------------------------------------

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import List


# 日本語の「話者名っぽい接頭辞」を剥がすための簡易正規表現
SPEAKER_PREFIX_RE = re.compile(
    r"^\s*[（(]?(?:[一-龥]{1,6}|[A-Za-zＡ-Ｚa-z]{1,20}|話者[Ａ-ＺA-Z]|司会|男性|女性)"
    r"[)）]?\s*(?:[:：]\s*|\s+)"
)

# 日本語の粗いトークナイズ（見かけの語の列）
# ・かな/カナ/漢字の連続、英数の連続、その他は1文字
WORDLIKE_RE = re.compile(r"[一-龥ぁ-んァ-ンー]+|[A-Za-z0-9]+|[^\s]")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "OpenAI Whisper (純正) で文字起こし。"
            "transcribe_whisper.py と同等のファイルを出力します"
        )
    )
    ap.add_argument("audio", help="入力音声（例: HMD/data/LD10-Dialogue-06.wav）")
    ap.add_argument("--language", default="ja", help="言語コード（既定: ja）")
    ap.add_argument(
        "--data-root", default="HMD/data", help="入出力のルート（既定: HMD/data）"
    )
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="推論デバイス（既定: auto）",
    )
    ap.add_argument(
        "--compute-type",
        default="auto",
        help="互換のためのダミー引数（純正Whisperでは未使用）",
    )
    ap.add_argument(
        "--model",
        default="large-v2",
        help="Whisper モデル名（既定: large-v2）",
    )
    ap.add_argument(
        "--initial-prompt",
        default=None,
        help="Whisper の initial_prompt（話者名の抑制などに活用）",
    )
    ap.add_argument(
        "--no-condition",
        action="store_true",
        help="condition_on_previous_text を無効化（文脈ドリフト抑制）",
    )
    ap.add_argument(
        "--strip-speaker-prefix",
        action="store_true",
        help="各セグメント先頭の話者名っぽい接頭辞を除去",
    )
    ap.add_argument(
        "--no-words",
        action="store_true",
        help="words.jsonl を出力しない（純正Whisperはword時刻がないため）",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        import whisper  # openai-whisper
    except Exception as e:
        print(
            "openai-whisper が見つかりません。`pip install -U openai-whisper` を実行してください。",
            file=sys.stderr,
        )
        raise

    # audio パス解決
    audio_path = Path(args.audio)
    if not audio_path.exists():
        maybe = Path(args.data_root) / args.audio
        if maybe.exists():
            audio_path = maybe
        else:
            raise FileNotFoundError(f"音声ファイルが見つかりません: {args.audio}")

    # 出力ディレクトリ: HMD/data/<basename>/
    basename = audio_path.stem
    out_dir = Path(args.data_root) / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    # デバイス
    device = args.device
    if device == "auto":
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    print(f"[info] device={device}, model={args.model} (whisper-pure)")

    # Whisper モデル
    model = whisper.load_model(args.model, device=device)

    # transcribe 実行
    # 純正Whisperは vad_filter / word_timestamps などは持たない。
    result = model.transcribe(
        str(audio_path),
        language=args.language,
        task="transcribe",
        initial_prompt=args.initial_prompt,
        condition_on_previous_text=not args.no_condition,
        fp16=(device == "cuda"),
        verbose=False,
    )

    segments: List[dict] = result.get("segments", [])

    # 出力パス
    plain_txt = out_dir / "plain.txt"
    seg_jsonl = out_dir / "segments.jsonl"
    words_jsonl = out_dir / "words.jsonl"

    # meta.json（ログ）
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "model": f"{args.model} (whisper-pure)",
                "language": args.language,
                "invoked_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # 1) segments.jsonl & 2) plain.txt
    texts: List[str] = []
    with seg_jsonl.open("w", encoding="utf-8") as fseg:
        for seg in segments:
            text = (seg.get("text") or "").strip()
            if args.strip_speaker_prefix:
                text = SPEAKER_PREFIX_RE.sub("", text)

            obj = {
                "start": float(seg.get("start", 0.0)),
                "end": float(seg.get("end", 0.0)),
                "text": text,
            }
            # 1行=1 JSON（ensure_ascii=False で参照文字にせず生の日本語に）
            fseg.write(json.dumps(obj, ensure_ascii=False) + "\n")
            texts.append(text)

    plain_txt.write_text("".join(texts), encoding="utf-8")

    # 3) words.jsonl（近似: セグメント時間を見かけの語に均等割り）
    if not args.no_words:
        with words_jsonl.open("w", encoding="utf-8") as fword:
            for seg in segments:
                text = (seg.get("text") or "").strip()
                if args.strip_speaker_prefix:
                    text = SPEAKER_PREFIX_RE.sub("", text)

                start = float(seg.get("start", 0.0))
                end = float(seg.get("end", 0.0))
                dur = max(0.0, end - start)

                tokens = WORDLIKE_RE.findall(text)
                n = len(tokens)
                if n == 0:
                    continue

                step = dur / n if n > 0 else 0.0
                for i, w in enumerate(tokens):
                    w_start = start + step * i
                    w_end = start + step * (i + 1)
                    w_obj = {
                        "start": float(w_start),
                        "end": float(w_end),
                        "word": w,
                        "segment_start": float(start),
                        "segment_end": float(end),
                        "approx": True,  # 純正Whisper由来ではない近似
                    }
                    fword.write(json.dumps(w_obj, ensure_ascii=False) + "\n")

    print(f"[done] plain:   {plain_txt}")
    print(f"[done] segments:{seg_jsonl}")
    if not args.no_words:
        print(f"[done] words:   {words_jsonl} (approx)")


if __name__ == "__main__":
    main()
