from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional

from openai import OpenAI


@dataclass
class WordBoundary:
    text: str
    start: float
    end: float


@dataclass
class SegmentDraft:
    speaker: str
    start: float
    end: float
    text: str
    words: List[WordBoundary]


def _normalize(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def _merge_contiguous_segments(drafts: Iterable[SegmentDraft]) -> List[SegmentDraft]:
    merged: List[SegmentDraft] = []
    current: Optional[SegmentDraft] = None
    for seg in drafts:
        if not seg.words:
            if current is not None:
                merged.append(current)
                current = None
            merged.append(seg)
            continue

        if (
            current is not None
            and current.speaker == seg.speaker
            and seg.start <= current.end + 0.05
        ):
            current.end = max(current.end, seg.end)
            if seg.text:
                if current.text and not current.text.endswith(" "):
                    current.text += " "
                current.text += seg.text
            current.words.extend(seg.words)
        else:
            if current is not None:
                merged.append(current)
            current = SegmentDraft(
                speaker=seg.speaker,
                start=seg.start,
                end=seg.end,
                text=seg.text,
                words=list(seg.words),
            )
    if current is not None:
        merged.append(current)
    return merged


def _map_sentences_to_words(turn: SegmentDraft, sentences: List[str]) -> Optional[List[SegmentDraft]]:
    if not sentences:
        return None
    words = turn.words
    normalized_words = [_normalize(w.text) for w in words]
    normalized_sentences = [_normalize(s) for s in sentences if _normalize(s)]
    if not normalized_sentences:
        return None

    results: List[SegmentDraft] = []
    word_index = 0
    for sentence, norm_sentence in zip(sentences, normalized_sentences):
        if word_index >= len(words):
            return None
        start_index = word_index
        buffer = ""
        while word_index < len(words) and len(buffer) < len(norm_sentence):
            buffer += normalized_words[word_index]
            word_index += 1
        if buffer != norm_sentence:
            return None
        segment_words = words[start_index:word_index]
        start_time = segment_words[0].start if segment_words else turn.start
        end_time = segment_words[-1].end if segment_words else turn.end
        results.append(
            SegmentDraft(
                speaker=turn.speaker,
                start=start_time,
                end=end_time,
                text=sentence.strip(),
                words=list(segment_words),
            )
        )

    if word_index < len(words):
        remainder = words[word_index:]
        start_time = remainder[0].start if remainder else turn.start
        end_time = remainder[-1].end if remainder else turn.end
        text = "".join(word.text for word in remainder)
        results.append(
            SegmentDraft(
                speaker=turn.speaker,
                start=start_time,
                end=end_time,
                text=text,
                words=list(remainder),
            )
        )
    return results


def refine_segments_with_llm(
    drafts: List[SegmentDraft],
    client: Optional[OpenAI],
    model: str,
    max_tokens: int,
    temperature: float,
) -> List[SegmentDraft]:
    if client is None or not drafts:
        return drafts

    merged = _merge_contiguous_segments(drafts)
    refined: List[SegmentDraft] = []

    system_prompt = (
        "あなたは日本語の会話を文の自然な切れ目で分割するアシスタントです。"
        "句点（。）、疑問符、感嘆符、接続詞による転換などを考慮して、"
        "話者ごとに文を分割してください。出力は JSON 配列のみで、"
        "各要素は {\"sentences\": [\"文1\", \"文2\", ...]} の形にしてください。"
    )

    for turn in merged:
        if not turn.words or not turn.text:
            refined.append(turn)
            continue

        user_prompt = (
            "話者: {speaker}\n"
            "発話: {text}\n"
            "\nJSON だけを返してください。"
        ).format(speaker=turn.speaker, text=turn.text.strip())

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            refined.append(turn)
            continue

        message = response.choices[0].message.content if response.choices else ""
        if not message:
            refined.append(turn)
            continue
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            refined.append(turn)
            continue

        sentences: List[str] = []
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    value = item.get("sentences")
                    if isinstance(value, list):
                        sentences.extend(str(v) for v in value if isinstance(v, str))
        if not sentences:
            refined.append(turn)
            continue
        mapped = _map_sentences_to_words(turn, sentences)
        if mapped is None:
            refined.append(turn)
        else:
            refined.extend(mapped)

    # 保持されなかった draft を補う（安全策）
    if not refined:
        return drafts

    # 時刻順にソートし、極小区間を除去
    cleaned: List[SegmentDraft] = []
    for seg in sorted(refined, key=lambda s: (s.start, s.end)):
        duration = seg.end - seg.start
        if math.isfinite(duration) and duration >= 1e-3:
            cleaned.append(seg)
    return cleaned or drafts
