# Sentiment Trials

音声パイプラインで生成された `segments.jsonl` を入力に、pymlask と BERT マスクド LM を使った簡易的な感情分析を試すスクリプトです。

## 依存関係
- pymlask（`pip install pymlask ipadic` で辞書付きの MeCab が利用可能）
- transformers, torch, plotly（既存環境に含まれている想定）

pymlask は内部で MeCab を使うため、辞書ディレクトリのパスが必要です。`ipadic` または `unidic_lite` が見つかれば自動的に設定されます。

## 実行例
```bash
python research/sentiment_trials/analyze_sentiment.py --conversation-id SD07-Dialogue-01 --limit 20
```

主なオプション:
- `segments`: `segments.jsonl` へのパスを直接指定したい場合に使用します。
- `--conversation-id`: `output/pipeline/<id>/segments.jsonl` を探して読み込みます。
- `--output-dir`: 結果保存先を明示的に変更します（未指定時は `research/sentiment_trials/output/<id>/`）。
- `--limit`: 試行段階でセグメント数を制限したいときに指定します。

## 出力
指定した出力ディレクトリに以下が生成されます。

- `analysis_results.json`: セグメントごとの pymlask 結果（orientation/emotion）と BERT マスクド LM スコア（ポジティブ/ネガティブ確率、差分ラベル）を含む JSON。
- `bert_scores.html`: 時間軸に沿った BERT スコアの散布図（pymlask orientation を色分け）。
- `pymlask_emotions.html`: 検出された感情ラベルの出現数をまとめたバー図。
- `conversation_view.html`: 会話順にセグメントを並べ、スコアとバッジを表示するカード型ビュー。

`analysis_results.json` にはサマリ統計（orientation カウント、BERT スコアの min/max/mean）と元セグメント ID・話者・テキストなどのメタデータも含めています。

## メモ
- ML-Ask の辞書は古い語彙が中心で、必ずしも全てのポジティブ/ネガティブ表現を検出できません。必要に応じて `pymlask_emotions` の辞書を拡張したり、形態素レベルで前処理する余地があります。
- BERT のマスクド LM を使ったスコアリングは簡易的なプロンプトベースの疑似分類です。ポジティブ／ネガティブ語彙リストを調整することで精度が変わるので、試行錯誤の余地があります。

## Valence/Arousal 埋め込み可視化
作成した HTML ビュー（例: `conversation_view.html`）に対して、埋め込みベースで Valence / Arousal を付与し、散布図も同時に出力するユーティリティ:

```bash
CV_LLM_BASE_URL=http://192.168.40.182:1234/v1 \
python research/sentiment_trials/embed_valence_arousal.py \
  research/sentiment_trials/output/LD01-Dialogue-01/conversation_view.html \
  --output research/sentiment_trials/output/LD01-Dialogue-01/conversation_view_annotated.html
```

- `text-embedding-embeddinggemma-300m-qat` を使って埋め込みを取得します（OpenAI 互換エンドポイント想定）。
- ポジ/ネガ（Valence）、動/静（Arousal）の代表文から軸ベクトルを構築し、各発話をコサイン類似度でスコアリングします。
- `data-valence`, `data-arousal`, `data-tags` 属性を `.segment` 要素に追加し、閾値（Valence ±0.4、Arousal ±0.4）に基づいて `POSITIVE` / `NEGATIVE` / `EXCITED` / `CALM` タグを付与します。
- ページ末尾に Plotly.js の散布図を追加し、発話 ID とラベルをホバーで確認できます。
