#!/usr/bin/env python3
"""
HMD Stage 2 (Discourse Classifier)
----------------------------------

Reads a transcript (plain text or WhisperX segments.jsonl), sends it to a local
LM Studio OpenAI-compatible endpoint, and writes a JSON file with discourse
probabilities and short speaker-attribute guesses.

Usage:
  python research/hmd/src/hmd_discourse_classifier.py \
    --input research/speaker_diarization/runs/whisperx/<run>/segments.jsonl \
    --output research/hmd/runs/<run>/stage2_discourse.json \
    --env HMD/.env

.env example (HMD/.env):
  LLM_BASE_URL=http://192.168.40.182:1234/v1
  LLM_MODEL=openai/gpt-oss-20b

Output JSON schema:
{
  "discourse_type": {"monologue": 35.0, "dialogue": 60.0, "multi_party": 5.0},
  "predicted_type": "dialogue",
  "speakers": [
    {"name": "A", "attribute": "先生"},
    {"name": "B", "attribute": "生徒"}
  ],
  "notes": "...optional"
}
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import List, Dict, Any

import requests

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None


def load_env(env_path: str | None) -> Dict[str, str]:
    env: Dict[str, str] = {}
    if env_path:
        if dotenv_values is None:
            raise RuntimeError("python-dotenv is not installed. pip install python-dotenv or omit --env")
        if not os.path.exists(env_path):
            raise FileNotFoundError(f".env not found: {env_path}")
        env.update({k: v for k, v in dotenv_values(env_path).items() if v is not None})
    # allow process env to override
    env.update({k: v for k, v in os.environ.items() if k.startswith("LLM_")})
    return env


def read_transcript(input_path: str) -> str:
    """Read transcript from segments.jsonl or plain .txt/.json.

    If jsonl: concatenates each line's `text` field in order, with newlines.
    If txt: reads as-is.
    If json: tries to read {segments:[{text:...}]}.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(input_path)

    lower = input_path.lower()
    if lower.endswith(".jsonl"):
        texts: List[str] = []
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                t = obj.get("text")
                if isinstance(t, str):
                    texts.append(t)
        return "\n".join(texts)

    if lower.endswith(".txt"):
        with open(input_path, "r", encoding="utf-8") as f:
            return f.read()

    if lower.endswith(".json"):
        with open(input_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("segments"), list):
            texts = [seg.get("text", "") for seg in obj["segments"]]
            return "\n".join([t for t in texts if t])
        # fallback: stringify
        return json.dumps(obj, ensure_ascii=False)

    # fallback: treat as text
    with open(input_path, "r", encoding="utf-8") as f:
        return f.read()


def build_prompt(transcript: str) -> List[Dict[str, str]]:
    system = (
        "You are an expert dialogue analyst for Japanese transcripts. "
        "Classify the overall discourse as one of: monologue (1 speaker), "
        "dialogue (2 speakers), or multi_party (>=3 speakers). "
        "Also infer brief one-word attributes for speakers (e.g., 先生, 生徒, 面接官, 候補者, 司会, ゲスト). "
        "Always respond as STRICT JSON only, no extra text."
    )

    # Keep transcript length reasonable for local models
    trimmed = transcript.strip()
    if len(trimmed) > 24000:
        trimmed = trimmed[:24000] + "\n..."

    user = (
        "日本語の書き起こし全文を示します。全体を読み、以下を出力してください。\n"
        "1) 話し方のタイプを3カテゴリで確率（合計100%）： monologue, dialogue, multi_party\n"
        "2) 最も妥当なカテゴリ（predicted_type）\n"
        "3) 出てくる話者の属性を一言で（可能なら A/B/C のように最大6名まで）\n\n"
        "必ず以下のJSONスキーマのみで返答：\n"
        "{\n  \"discourse_type\": {\"monologue\": <0-100>, \"dialogue\": <0-100>, \"multi_party\": <0-100>},\n"
        "  \"predicted_type\": \"monologue|dialogue|multi_party\",\n"
        "  \"speakers\": [ {\"name\": \"A\", \"attribute\": \"先生\"}, {\"name\": \"B\", \"attribute\": \"生徒\"} ]\n}"
        "\n\n--- transcript ---\n" + trimmed
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_llm_json(content: str) -> Dict[str, Any]:
    if not content:
        raise ValueError("Empty content from LLM response")

    content_str = content.strip()
    content_str = content_str.strip("`")

    idx = content_str.find("{")
    if idx != -1:
        content_str = content_str[idx:]

    try:
        return json.loads(content_str)
    except json.JSONDecodeError:
        start = content_str.find("{")
        end = content_str.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(content_str[start:end + 1])
        raise


def call_llm(
    *,
    base_url: str,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    api_key: str | None = None,
) -> Dict[str, Any]:
    if not messages:
        raise ValueError("messages must not be empty")

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    def _post(payload: Dict[str, Any]) -> Dict[str, Any]:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("LLM response missing choices")
        first = choices[0]
        message = first.get("message") or {}
        content = message.get("content") or first.get("text")
        return _parse_llm_json(content or "")

    payload_min = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": 512,
    }

    try:
        return _post(payload_min)
    except requests.HTTPError as e:
        payload_jsonfmt = dict(payload_min)
        payload_jsonfmt["response_format"] = {"type": "json_object"}
        try:
            return _post(payload_jsonfmt)
        except Exception:
            raise e


def normalize_output(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure probabilities sum to 100 (float)
    disc = obj.get("discourse_type", {})
    keys = ["monologue", "dialogue", "multi_party"]
    vals = [float(disc.get(k, 0.0)) for k in keys]
    s = sum(vals)
    if s <= 0:
        vals = [0.0, 100.0, 0.0]  # default to dialogue
        s = 100.0
    norm = [round(v * 100.0 / s, 2) for v in vals]
    obj["discourse_type"] = {k: v for k, v in zip(keys, norm)}

    # Clamp speaker attributes list
    speakers = obj.get("speakers", [])
    out_speakers = []
    for i, sp in enumerate(speakers[:6]):
        name = sp.get("name") or chr(ord('A') + i)
        attr = sp.get("attribute") or "話者"
        out_speakers.append({"name": str(name), "attribute": str(attr)})
    obj["speakers"] = out_speakers

    # predicted_type fallback
    pt = str(obj.get("predicted_type", "")).strip()
    if pt not in ("monologue", "dialogue", "multi_party"):
        # pick max
        m_idx = max(range(3), key=lambda i: norm[i])
        pt = keys[m_idx]
    obj["predicted_type"] = pt
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Transcript input: segments.jsonl or .txt/.json")
    ap.add_argument("--output", required=True, help="Output JSON path")
    ap.add_argument("--env", default="HMD/.env", help="Path to .env with LLM_BASE_URL and LLM_MODEL (default: HMD/.env)")
    ap.add_argument("--model", default=None, help="Override model id")
    ap.add_argument("--base-url", default=None, help="Override base URL, e.g., http://192.168.40.182:1234/v1")
    ap.add_argument("--temperature", type=float, default=0.2)
    args = ap.parse_args()

    env = load_env(args.env)
    base_url = args.base_url or env.get("LLM_BASE_URL")
    model = args.model or env.get("LLM_MODEL")

    if not base_url or not model:
        raise SystemExit("LLM_BASE_URL / LLM_MODEL are required (set in --env or via args)")

    transcript = read_transcript(args.input)
    messages = build_prompt(transcript)

    obj = call_llm(
        base_url=base_url,
        model=model,
        messages=messages,
        temperature=args.temperature,
        api_key=env.get("LLM_API_KEY"),
    )
    obj = normalize_output(obj)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"[HMD] Discourse JSON written -> {args.output}")


if __name__ == "__main__":
    main()
