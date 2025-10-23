#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch-run Google Cloud Speech-to-Text v2 (with diarization) and
produce hypothesis files that plug into our existing evaluation/HTML pipeline
(evaluate_char.py / report_html.py).

Now supports two modes:
  1) recognize (inline base64, for short/small audio)
  2) batchRecognize to GCS (for long/large audio)  <-- NEW

Usage (example):

python research/speaker_diarization/src/run_gcloud_stt_batch.py \
  --inputs research/speaker_diarization/data \
  --pattern "*.wav" \
  --ref-dir research/speaker_diarization/data \
  --output-root research/speaker_diarization/runs/gcloud \
  --exp-slug chirp_v2_diar \
  --project YOUR_GCP_PROJECT \
  --location asia-northeast1 \
  --recognizer ja-rec \
  --min-speakers 2 --max-speakers 2 \
  --language ja-JP --model long \
  --make-html \
  --eval-script research/speaker_diarization/src/evaluate_char.py \
  --html-script research/speaker_diarization/src/report_html.py

Batch mode (recommended for > ~9MB files):
  --use-batch --gcs-bucket your-bucket --gcs-prefix stt_in --gcs-output-prefix stt_out

Prereqs:
- `gcloud auth login` と `gcloud auth application-default login` 済み
- `pip install google-auth google-auth-httplib2 google-auth-oauthlib google-api-core google-cloud-speech google-cloud-core google-cloud-storage`
- Recognizer 作成済み（language/model は recognizer 側にセット推奨）

Notes:
- recognize は容量/長さ制限にかかりやすい。大きいファイルは batchRecognize を使う。
- 応答の `results` から最終 `alternatives[0].words` を取り、
  `speakerX` と時刻で隣接語をまとめて JSONL で保存。
- JSONL は {start, end, speaker, text} 行の連なり。report_html.py/evaluate_char.py で読めます。
"""
from __future__ import annotations
import argparse
import base64
import datetime as dt
import glob
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Dict, Iterable, List, Tuple
import wave
import audioop

# Google auth (ADC)
from google.auth import default as google_auth_default
from google.auth.transport.requests import AuthorizedSession, Request
from google.cloud import storage  # NEW for GCS upload/download

# -------------------------------
# Helpers
# -------------------------------

def now_slug() -> str:
    jst = dt.timezone(dt.timedelta(hours=9))
    return dt.datetime.now(jst).strftime("%Y-%m-%d_%H-%M-%S")


def seconds_from_duration_str(s: str) -> float:
    if s is None:
        return 0.0
    s = str(s)
    m = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)s", s)
    if m:
        return float(m.group(1))
    if ":" in s:
        h, m_, rest = s.split(":", 2)
        sec = float(rest)
        return int(h) * 3600 + int(m_) * 60 + sec
    try:
        return float(s)
    except Exception:
        return 0.0


def group_words_to_segments(words: List[Dict]) -> List[Dict]:
    norm = []
    for w in words:
        spk = (
            w.get("speakerTag")
            or w.get("speaker")
            or w.get("speakerLabel")
            or w.get("speaker_id")
            or w.get("speakerTagId")
        )
        st = w.get("startOffset") or w.get("startTime") or w.get("start_time")
        et = w.get("endOffset") or w.get("endTime") or w.get("end_time")
        norm.append({
            "speaker": spk,
            "start": seconds_from_duration_str(st),
            "end": seconds_from_duration_str(et),
            "word": w.get("word", ""),
        })

    segs: List[Dict] = []
    cur: Dict | None = None
    for item in norm:
        spk = item["speaker"]
        if spk is None:
            if cur is None:
                cur = {"speaker": "U", "start": item["start"], "end": item["end"], "text": item["word"]}
            else:
                cur["end"] = item["end"]
                if item["word"]:
                    cur["text"] += (" " + item["word"]) if cur["text"] else item["word"]
            continue
        if cur is None:
            cur = {"speaker": spk, "start": item["start"], "end": item["end"], "text": item["word"]}
            continue
        if cur["speaker"] == spk and item["start"] <= cur["end"] + 0.2:
            cur["end"] = max(cur["end"], item["end"])
            if item["word"]:
                cur["text"] += (" " + item["word"]) if cur["text"] else item["word"]
        else:
            segs.append(cur)
            cur = {"speaker": spk, "start": item["start"], "end": item["end"], "text": item["word"]}
    if cur is not None:
        segs.append(cur)

    spk_ids = []
    for s in segs:
        if s["speaker"] not in spk_ids:
            spk_ids.append(s["speaker"])
    spk_map = {k: chr(ord('A') + i) for i, k in enumerate(spk_ids)}
    for s in segs:
        s["speaker"] = spk_map.get(s["speaker"], "U")
    return segs


def save_jsonl_segments(path: Path, segments: List[Dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps({
                "start": seg["start"],
                "end": seg["end"],
                "speaker": seg["speaker"],
                "text": seg.get("text", "")
            }, ensure_ascii=False) + "\n")


# -------------------------------
# Google STT v2 REST call (recognize)
# -------------------------------

def stt_v2_recognize(session: AuthorizedSession, *, endpoint: str, audio_bytes: bytes,
                      language: str | None, model: str | None,
                      min_speakers: int, max_speakers: int,
                      enable_conf: bool = True) -> Dict:
    b64 = base64.b64encode(audio_bytes).decode("ascii")
    body = {
        "config": {
            **({"languageCodes": [language]} if language else {}),
            **({"model": model} if model else {}),
            "autoDecodingConfig": {},
            "features": {
                "enableWordTimeOffsets": True,
                "enableWordConfidence": enable_conf,
                "enableAutomaticPunctuation": True,
                "diarizationConfig": {
                    "minSpeakerCount": int(min_speakers),
                    "maxSpeakerCount": int(max_speakers)
                }
            }
        },
        "content": b64,
    }
    resp = session.post(endpoint, json=body, timeout=600)
    if not resp.ok:
        print("[run-gcloud-stt] ERROR DETAIL:", resp.text[:4000])
        resp.raise_for_status()
    return resp.json()


# -------------------------------
# Google STT v2 REST call (batchRecognize via GCS)  NEW
# -------------------------------

def stt_v2_batch_recognize(session: AuthorizedSession, *, endpoint_batch: str,
                            gcs_uri: str, gcs_out_prefix: str,
                            language: str | None, model: str | None,
                            min_speakers: int, max_speakers: int) -> str:
    body = {
        "recognizer": endpoint_batch.split(":")[0].replace("https://", "").split("/v2/")[-1],
        "config": {
            **({"languageCodes": [language]} if language else {}),
            **({"model": model} if model else {}),
            "features": {
                "enableWordTimeOffsets": True,
                "enableWordConfidence": True,
                "enableAutomaticPunctuation": True,
                "diarizationConfig": {
                    "minSpeakerCount": int(min_speakers),
                    "maxSpeakerCount": int(max_speakers)
                }
            }
        },
        "files": [{"uri": gcs_uri}],
        "recognitionOutputConfig": {"gcsOutputConfig": {"uri": gcs_out_prefix}},
    }
    resp = session.post(endpoint_batch, json=body, timeout=600)
    if not resp.ok:
        print("[run-gcloud-stt] ERROR DETAIL(batch):", resp.text[:4000])
        resp.raise_for_status()
    op = resp.json().get("name")
    return op


def ensure_linear16_mono(audio_path: Path, target_hz: int = 16000) -> Tuple[bytes, int]:
    """Convert WAV to 16-bit mono Linear PCM for v1p1beta1."""
    with wave.open(str(audio_path), 'rb') as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        frame_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())

    if sample_width != 2:
        frames = audioop.lin2lin(frames, sample_width, 2)
        sample_width = 2
    if channels > 1:
        frames = audioop.tomono(frames, sample_width, 0.5, 0.5)
        channels = 1
    if frame_rate != target_hz:
        frames, _ = audioop.ratecv(frames, sample_width, channels, frame_rate, target_hz, None)
        frame_rate = target_hz
    return frames, frame_rate


def stt_v1p1beta1_recognize(session: AuthorizedSession, *, audio_path: Path,
                             language: str | None, model: str | None,
                             min_speakers: int, max_speakers: int,
                             enable_conf: bool = True) -> Dict:
    frames, rate = ensure_linear16_mono(audio_path)
    b64 = base64.b64encode(frames).decode("ascii")
    config = {
        "languageCode": language or "en-US",
        "encoding": "LINEAR16",
        "sampleRateHertz": rate,
        "enableAutomaticPunctuation": True,
        "enableWordTimeOffsets": True,
        "enableSpeakerDiarization": True,
        "diarizationConfig": {
            "enableSpeakerDiarization": True,
            "minSpeakerCount": int(min_speakers),
            "maxSpeakerCount": int(max_speakers)
        }
    }
    if model:
        config["model"] = model
    # enhanced models require flag
    config["useEnhanced"] = True
    if enable_conf:
        config["enableWordConfidence"] = True

    endpoint = "https://speech.googleapis.com/v1p1beta1/speech:recognize"
    body = {"config": config, "audio": {"content": b64}}
    resp = session.post(endpoint, json=body, timeout=600)
    if not resp.ok:
        print("[run-gcloud-stt] ERROR DETAIL(v1p1beta1):", resp.text[:4000])
        resp.raise_for_status()
    return resp.json()


def poll_operation(session: AuthorizedSession, op_name: str) -> Dict:
    url = f"https://speech.googleapis.com/v2/{op_name}"
    while True:
        r = session.get(url)
        j = r.json()
        if j.get("done"):
            return j
        time.sleep(3)


def pick_result_json_from_gcs(storage_client: storage.Client, bucket: str, out_prefix: str) -> Dict:
    b = storage_client.bucket(bucket)
    # list blobs under prefix
    blobs = list(storage_client.list_blobs(bucket, prefix=out_prefix))
    # Pick first *.json (skip manifest if needed)
    for bl in blobs:
        name = bl.name
        if name.endswith(".json"):
            data = bl.download_as_bytes()
            try:
                return json.loads(data)
            except Exception:
                continue
    raise FileNotFoundError("No JSON result under gs://%s/%s" % (bucket, out_prefix))


# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Batch Google STT v2 + diarization + HTML compare")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--inputs", nargs="+", help="Input dirs or files")
    g.add_argument("--filelist", help="Text file listing input audio paths")

    ap.add_argument("--pattern", default="*.wav")
    ap.add_argument("--ref-dir", default=None, help="Directory containing *.txt references (same stem). Optional.")
    ap.add_argument("--output-root", default="research/speaker_diarization/runs/gcloud")
    ap.add_argument("--exp-name", default=None)
    ap.add_argument("--exp-slug", default="chirp_v2_diar")

    ap.add_argument("--project", required=True)
    ap.add_argument("--location", default="asia-northeast1")
    ap.add_argument("--recognizer", required=False, help="Recognizer ID or full name (v2 only). If only ID, it will be expanded.")
    ap.add_argument("--api-version", choices=["v2", "v1p1beta1"], default="v2",
                    help="Select Speech API version. v1p1beta1 uses enhanced models with inline recognize only.")

    ap.add_argument("--language", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--min-speakers", type=int, default=2)
    ap.add_argument("--max-speakers", type=int, default=2)

    ap.add_argument("--use-batch", action="store_true", help="Use batchRecognize to GCS (v2 only)")
    ap.add_argument("--gcs-bucket", default=None, help="GCS bucket for batchRecognize IO")
    ap.add_argument("--gcs-prefix", default="stt_in", help="input prefix in bucket")
    ap.add_argument("--gcs-output-prefix", default="stt_out", help="output prefix in bucket")

    ap.add_argument("--eval-script", default="research/speaker_diarization/src/evaluate_char.py")
    ap.add_argument("--html-script", default="research/speaker_diarization/src/report_html.py")
    ap.add_argument("--make-html", action="store_true")

    args = ap.parse_args()

    run_slug = args.exp_name or f"{now_slug()}_{args.exp_slug}"
    out_root = Path(args.output_root) / run_slug
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"[run-gcloud-stt] Run dir: {out_root}")

    files: List[str] = []
    if args.filelist:
        with open(args.filelist, "r", encoding="utf-8") as f:
            for line in f:
                p = line.strip()
                if p:
                    files.append(p)
    else:
        for p in args.inputs:
            pth = Path(p)
            if pth.is_dir():
                files += glob.glob(str(pth / args.pattern))
            else:
                files.append(str(pth))
    files = sorted({str(Path(f).resolve()) for f in files})
    print(f"[run-gcloud-stt] Found {len(files)} audio files.")

    creds, _ = google_auth_default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    if not creds.valid:
        creds.refresh(Request())
    session = AuthorizedSession(creds)

    if args.api_version == "v2":
        if not args.recognizer:
            ap.error("--recognizer is required for v2 usage")
        if "/" in args.recognizer:
            recog_name = args.recognizer
        else:
            recog_name = f"projects/{args.project}/locations/{args.location}/recognizers/{args.recognizer}"
        endpoint_recognize = f"https://{args.location}-speech.googleapis.com/v2/{recog_name}:recognize"
        endpoint_batch = f"https://{args.location}-speech.googleapis.com/v2/{recog_name}:batchRecognize"

        storage_client = None
        if args.use_batch:
            assert args.gcs_bucket, "--gcs-bucket is required when --use-batch"
            storage_client = storage.Client(project=args.project)
    else:
        if args.use_batch:
            ap.error("--use-batch is only supported for api-version=v2")
        endpoint_recognize = endpoint_batch = None
        storage_client = None

    summary_rows: List[Tuple[str, str, str]] = []

    for wav in files:
        stem = Path(wav).stem
        work_dir = out_root / stem
        hyp_dir = work_dir / "gcloud"
        hyp_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run-gcloud-stt] >> {wav}")

        try:
            if args.api_version == "v1p1beta1":
                result_json = stt_v1p1beta1_recognize(
                    session,
                    audio_path=Path(wav),
                    language=args.language,
                    model=args.model,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
            elif args.use_batch:
                # Upload to GCS
                in_key = f"{args.gcs_prefix}/{stem}{Path(wav).suffix}"
                out_prefix = f"{args.gcs_output_prefix}/{stem}/"
                bucket = storage_client.bucket(args.gcs_bucket)
                blob = bucket.blob(in_key)
                blob.upload_from_filename(wav)
                gcs_uri = f"gs://{args.gcs_bucket}/{in_key}"
                gcs_out_prefix = f"gs://{args.gcs_bucket}/{out_prefix}"
                print(f"[run-gcloud-stt] uploaded -> {gcs_uri}")

                op_name = stt_v2_batch_recognize(
                    session,
                    endpoint_batch=endpoint_batch,
                    gcs_uri=gcs_uri,
                    gcs_out_prefix=gcs_out_prefix,
                    language=args.language,
                    model=args.model,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
                print(f"[run-gcloud-stt] operation: {op_name}")
                done = poll_operation(session, op_name)
                if done.get("error"):
                    raise RuntimeError(done["error"])
                # read result JSON from GCS
                result_json = pick_result_json_from_gcs(storage_client, args.gcs_bucket, out_prefix)
            else:
                # recognize (inline)
                audio_bytes = Path(wav).read_bytes()
                # guardrail: if file too large, suggest batch mode
                if len(audio_bytes) > 9_000_000:
                    print("[run-gcloud-stt] WARN: file >9MB; inline recognize may fail. Consider --use-batch.")
                result_json = stt_v2_recognize(
                    session,
                    endpoint=endpoint_recognize,
                    audio_bytes=audio_bytes,
                    language=args.language,
                    model=args.model,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers,
                )
        except Exception as e:
            print(f"[run-gcloud-stt] ERROR: recognize failed for {wav}: {e}")
            continue

        # Extract words (final)
        results = result_json.get("results", [])
        words: List[Dict] = []
        for r in results:
            alts = r.get("alternatives") or []
            if not alts:
                continue
            w = alts[0].get("words") or []
            if w:
                words = w
        if not words and args.use_batch:
            # batchRecognize output format sometimes nests under "transcriptions"; try to locate
            # Fallback traversal
            def traverse(obj):
                if isinstance(obj, dict):
                    if "words" in obj and isinstance(obj["words"], list):
                        return obj["words"]
                    for v in obj.values():
                        got = traverse(v)
                        if got:
                            return got
                elif isinstance(obj, list):
                    for it in obj:
                        got = traverse(it)
                        if got:
                            return got
                return None
            maybe = traverse(result_json)
            words = maybe or []

        if not words:
            print(f"[run-gcloud-stt] WARN: no words for {wav}")
            continue

        segments = group_words_to_segments(words)
        hyp_jsonl = hyp_dir / f"{stem}.jsonl"
        save_jsonl_segments(hyp_jsonl, segments)
        print(f"[run-gcloud-stt] wrote {hyp_jsonl}")

        metrics_path = work_dir / "metrics.json"
        ref_txt = None
        if args.ref_dir:
            cand = Path(args.ref_dir) / f"{stem}.txt"
            if cand.exists():
                ref_txt = str(cand)
        if ref_txt:
            cmd = [sys.executable, args.eval_script, "--hyp", str(hyp_jsonl), "--ref", ref_txt, "--json-out", str(metrics_path)]
            print("[run-gcloud-stt] $", " ".join(cmd))
            rc = os.spawnve(os.P_WAIT, sys.executable, cmd, os.environ.copy())
            if rc != 0:
                print(f"[run-gcloud-stt] WARN: eval failed for {stem} (exit {rc})")
        else:
            print(f"[run-gcloud-stt] INFO: no reference found; eval skipped for {stem}")

        if args.make_html and ref_txt:
            out_dir = work_dir
            cmd = [sys.executable, args.html_script, "--hyp", str(hyp_jsonl), "--ref", ref_txt, "--output", str(out_dir), "--title", stem]
            print("[run-gcloud-stt] $", " ".join(cmd))
            rc = os.spawnve(os.P_WAIT, sys.executable, cmd, os.environ.copy())
            if rc != 0:
                print(f"[run-gcloud-stt] WARN: html report failed for {stem} (exit {rc})")

        metrics_str = str(metrics_path if metrics_path.exists() else '-')
        summary_rows.append((stem, str(hyp_jsonl), metrics_str))

    if summary_rows:
        import csv
        csv_path = out_root / "summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["stem", "hyp_jsonl", "metrics_json"])
            w.writerows(summary_rows)
        print(f"[run-gcloud-stt] Summary written: {csv_path}")
    print("[run-gcloud-stt] Done.")


if __name__ == "__main__":
    main()
