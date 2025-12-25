#!/usr/bin/env python3
"""Annotate conversation HTML with valence/arousal scores via embedding axes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
from bs4 import BeautifulSoup
from openai import OpenAI


VALENCE_POSITIVE = ["ありがとう", "うれしい", "幸せ", "楽しかった"]
VALENCE_NEGATIVE = ["つらい", "最悪", "死にたい", "むかつく"]

AROUSAL_ACTIVE = ["怒鳴った", "興奮した", "走り出した", "叫んだ"]
AROUSAL_PASSIVE = ["静かだった", "しんみりした", "落ち着いた", "眠たい"]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def build_axis(client: OpenAI, model: str, positives: Sequence[str], negatives: Sequence[str]) -> np.ndarray:
    pos_vec = average_embedding(client, model, positives)
    neg_vec = average_embedding(client, model, negatives)
    axis = pos_vec - neg_vec
    norm = np.linalg.norm(axis)
    if norm == 0:
        raise RuntimeError("Axis vector collapsed to zero; adjust anchor texts.")
    return axis / norm


def average_embedding(client: OpenAI, model: str, texts: Sequence[str]) -> np.ndarray:
    embeds = fetch_embeddings(client, model, texts)
    return np.mean(embeds, axis=0)


def fetch_embeddings(client: OpenAI, model: str, texts: Sequence[str]) -> np.ndarray:
    if not texts:
        raise ValueError("No texts provided for embedding.")
    response = client.embeddings.create(model=model, input=list(texts))
    vectors = [np.array(item.embedding, dtype=float) for item in response.data]
    return np.stack(vectors)


def extract_segments(soup: BeautifulSoup) -> List[Tuple[str, str, str]]:
    segments: List[Tuple[str, str, str]] = []
    for idx, container in enumerate(soup.select(".segment")):
        text_node = container.select_one(".segment__text") or container
        text = text_node.get_text(" ", strip=True)
        if not text:
            continue
        speaker_node = container.select_one(".segment__speaker")
        speaker = speaker_node.get_text(strip=True) if speaker_node else "?"
        seg_id = container.get("id") or f"segment-{idx+1}"
        container["id"] = seg_id  # ensure id exists for hover reference
        segments.append((seg_id, speaker, text))
    return segments


def assign_attributes(
    soup: BeautifulSoup,
    embeddings: List[np.ndarray],
    valence_axis: np.ndarray,
    arousal_axis: np.ndarray,
    segments_info: List[Tuple[str, str, str]],
) -> List[dict]:
    enriched: List[dict] = []
    for idx, (seg_id, speaker, text) in enumerate(segments_info):
        val = cosine_similarity(embeddings[idx], valence_axis)
        aro = cosine_similarity(embeddings[idx], arousal_axis)
        tags: List[str] = []
        if val > 0.4:
            tags.append("POSITIVE")
        elif val < -0.4:
            tags.append("NEGATIVE")
        if aro > 0.4:
            tags.append("EXCITED")
        elif aro < -0.4:
            tags.append("CALM")

        container = soup.find(id=seg_id)
        if container is None:
            continue
        container["data-valence"] = f"{val:.4f}"
        container["data-arousal"] = f"{aro:.4f}"
        if tags:
            container["data-tags"] = "|".join(tags)
        enriched.append(
            {
                "id": seg_id,
                "speaker": speaker,
                "text": text,
                "valence": val,
                "arousal": aro,
                "tags": tags,
            }
        )
    return enriched


def ensure_body(soup: BeautifulSoup):
    if soup.body is None:
        body = soup.new_tag("body")
        if soup.html is None:
            html = soup.new_tag("html")
            soup.append(html)
            html.append(body)
        else:
            soup.html.append(body)
    return soup.body


def append_scatter_plot(soup: BeautifulSoup, data: List[dict]) -> None:
    if not data:
        return
    body = ensure_body(soup)
    plot_container = soup.new_tag("div", attrs={"id": "valence-arousal-scatter", "style": "height:480px;"})
    body.append(plot_container)

    script = soup.new_tag("script", src="https://cdn.plot.ly/plotly-2.27.0.min.js")
    body.append(script)

    payload = json.dumps(data, ensure_ascii=False)
    inline_script = """
    const vaData = __PAYLOAD__;
    const points = vaData.map(item => ({
      x: item.valence,
      y: item.arousal,
      text: item.text,
      customdata: item.id
    }));
    const tags = vaData.map(item => item.tags.join(',') || 'NEUTRAL');
    Plotly.newPlot('valence-arousal-scatter', [{
      x: points.map(p => p.x),
      y: points.map(p => p.y),
      mode: 'markers',
      text: vaData.map(item => item.text),
      customdata: vaData.map(item => item.id),
      marker: {
        size: 11,
        color: tags.map(tag => tag.includes('NEGATIVE') ? '#1c7ed6' : tag.includes('POSITIVE') ? '#e03131' : tag.includes('EXCITED') ? '#fab005' : tag.includes('CALM') ? '#20c997' : '#868e96'),
        line: { width: 1, color: '#ffffff' }
      },
      hovertemplate: 'ID: %{customdata}<br>Valence: %{x:.3f}<br>Arousal: %{y:.3f}<br>%{text}<extra></extra>'
    }], {
      title: 'Valence / Arousal Scatter',
      xaxis: { title: 'Valence (Negative ⟵ 0 ⟶ Positive)', range: [-1, 1], zeroline: true, zerolinecolor: '#adb5bd' },
      yaxis: { title: 'Arousal (Calm ⟵ 0 ⟶ Excited)', range: [-1, 1], zeroline: true, zerolinecolor: '#adb5bd' },
      margin: { l: 60, r: 20, t: 60, b: 60 }
    });
    """.replace("__PAYLOAD__", payload)
    inline_tag = soup.new_tag("script")
    inline_tag.string = inline_script
    body.append(inline_tag)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate conversation HTML with valence/arousal scores")
    parser.add_argument("html", type=Path, help="Path to conversation HTML")
    parser.add_argument("--output", type=Path, help="Output HTML path (default: overwrite input)")
    parser.add_argument("--model", type=str, default="text-embedding-embeddinggemma-300m-qat")
    args = parser.parse_args()

    base_url = os.environ.get("CV_LLM_BASE_URL")
    if not base_url:
        raise EnvironmentError("CV_LLM_BASE_URL is not set in environment")
    api_key = os.environ.get("CV_LLM_API_KEY", "not-needed")
    client = OpenAI(base_url=base_url, api_key=api_key)

    html_path: Path = args.html
    if not html_path.exists():
        raise FileNotFoundError(html_path)
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

    segments = extract_segments(soup)
    if not segments:
        raise RuntimeError("No segments found (.segment elements).")

    # Build axes first
    valence_axis = build_axis(client, args.model, VALENCE_POSITIVE, VALENCE_NEGATIVE)
    arousal_axis = build_axis(client, args.model, AROUSAL_ACTIVE, AROUSAL_PASSIVE)

    utter_texts = [text for _, _, text in segments]
    utter_embeddings = fetch_embeddings(client, args.model, utter_texts)

    enriched = assign_attributes(soup, utter_embeddings, valence_axis, arousal_axis, segments)

    append_scatter_plot(soup, enriched)

    output_path = args.output or html_path
    output_path.write_text(soup.prettify(), encoding="utf-8")
    print(f"Annotated HTML written to {output_path}")


if __name__ == "__main__":
    main()
