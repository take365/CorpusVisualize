#!/usr/bin/env python3
# HMD/src/transcribe_whisper.py

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

def main():
    ap = argparse.ArgumentParser(
        description="Whisper large-v2 で文字起こし（細かめのタイムスタンプ付き）"
    )
    ap.add_argument("audio", help="入力音声（例: HMD/data/LD01-Dialogue-01.wav）")
    ap.add_argument("--language", default="ja", help="言語コード（既定: ja）")
    ap.add_argument(
        "--data-root",
        default="HMD/data",
        help="入出力のルート（既定: HMD/data）",
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
        help="CTranslate2 の compute type（cuda: float16 / cpu: int8_float32 等。既定: auto）",
    )
    ap.add_argument(
        "--model",
        default="large-v3-turbo",
        help="Whisper モデル名（既定: large-v3-turbo）",
    )
    args = ap.parse_args()

    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        print(
            "faster-whisper が見つかりません。`pip install faster-whisper` を実行してください。",
            file=sys.stderr,
        )
        raise

    audio_path = Path(args.audio)
    if not audio_path.exists():
        # HMD/data 以下の相対指定かもしれないので補助
        maybe = Path(args.data_root) / args.audio
        if maybe.exists():
            audio_path = maybe
        else:
            raise FileNotFoundError(f"音声ファイルが見つかりません: {args.audio}")

    # 出力ディレクトリ: HMD/data/<basename>/
    basename = audio_path.stem
    out_dir = Path(args.data_root) / basename
    out_dir.mkdir(parents=True, exist_ok=True)

    # ログ的にタイムスタンプを残しておく（任意）
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "model": args.model,
                "language": args.language,
                "invoked_at": datetime.now().isoformat(),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # デバイス自動判定
    device = args.device
    if device == "auto":
        try:
            import torch  # なくてもOK、あれば判定に使う
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    # compute_type 自動化（経験則）
    compute_type = args.compute_type
    if compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8_float32"

    print(f"[info] device={device}, compute_type={compute_type}, model={args.model}")

    # WhisperModel 準備
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        # cpu_num_threads や num_workers を環境に応じて調整可
    )

    # なるべく細かいタイムスタンプ:
    #  - vad_filter で無音区間処理
    #  - word_timestamps=True で単語タイムスタンプ
    #  - beam_size などは精度/速度の好みで調整
    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        vad_filter=True,
        #vad_parameters={"min_silence_duration_ms": 200},
        vad_parameters={"min_silence_duration_ms": 150},
        beam_size=5,
        best_of=5,
        temperature=0.0,
        word_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt="発話の先頭に話者名を付けず、発話内容のみを出力してください。"
    )

    # 出力ファイル
    plain_txt = out_dir / "plain.txt"
    seg_jsonl = out_dir / "segments.jsonl"
    words_jsonl = out_dir / "words.jsonl"

    # 1) 純テキスト結合
    full_text_parts = []

    # 2) セグメント JSONL
    with seg_jsonl.open("w", encoding="utf-8") as fseg, \
         words_jsonl.open("w", encoding="utf-8") as fword:
        for seg in segments:
            # seg.text は先頭に空白が付くことがあるので strip
            text = (seg.text or "").strip()

            # セグメントを書き出し
            seg_obj = {
                "start": float(seg.start) if seg.start is not None else None,
                "end": float(seg.end) if seg.end is not None else None,
                "text": text,
            }
            fseg.write(json.dumps(seg_obj, ensure_ascii=False) + "\n")

            # 単語（サブワード）レベル
            if seg.words:
                for w in seg.words:
                    wtext = (w.word or "").strip()
                    if not wtext:
                        continue
                    w_obj = {
                        "start": float(w.start) if w.start is not None else None,
                        "end": float(w.end) if w.end is not None else None,
                        "word": wtext,
                        "segment_start": float(seg.start) if seg.start is not None else None,
                        "segment_end": float(seg.end) if seg.end is not None else None,
                    }
                    fword.write(json.dumps(w_obj, ensure_ascii=False) + "\n")

            full_text_parts.append(text)

    plain_txt.write_text("".join(full_text_parts), encoding="utf-8")

    print(f"[done] plain:   {plain_txt}")
    print(f"[done] segments:{seg_jsonl}")
    print(f"[done] words:   {words_jsonl}")

if __name__ == "__main__":
    main()
