# sentiment_trials 作業メモ

## 2025-10-27 時点の状況
- `analyze_sentiment.py` を追加し、`segments.jsonl` から pymlask と BERT マスクド LM で感情指標を算出。`analysis_results.json` / `bert_scores.html` / `pymlask_emotions.html` / `conversation_view.html` を出力するフローを確認。
- `conversation_view.html` は発話順カードビュー。`--conversation-id LD01-Dialogue-01 --limit 5` で動作検証済み。その後 `--limit` を外して全セグメント出力。
- pymlask は `ipadic` を指定して動作。ただし現状の会話データでは検出率が低い（多くが `null`）。辞書拡張・前処理で改善余地あり。
- BERT マスクド LM（tohoku-nlp/bert-base-japanese-whole-word-masking）のスコア差は小さめ。語彙リスト拡張やスコアの再スケールを検討。

## Valence/Arousal 付与
- `embed_valence_arousal.py` を追加。HTML（`.segment` 構造）を読み込み、埋め込みエンドポイント（`CV_LLM_BASE_URL=http://192.168.40.182:1234/v1`）から `text-embedding-embeddinggemma-300m-qat` を呼び出し。
- 代表文セット：
  - Valence: ポジ側「ありがとう」「うれしい」「幸せ」「楽しかった」、ネガ側「つらい」「最悪」「死にたい」「むかつく」
  - Arousal: 動側「怒鳴った」「興奮した」「走り出した」「叫んだ」、静側「静かだった」「しんみりした」「落ち着いた」「眠たい」
- 軸ベクトルはポジ/動-ネガ/静の平均差。各発話に対してコサイン類似度を計算し `data-valence` / `data-arousal` / `data-tags` を付加。Plotly で散布図をページ下部に追加。
- 閾値は |0.4| でタグ付与（POSITIVE/NEGATIVE/CALM/EXCITED）。現在はスコアが ±0.3 程度に収まりタグが付かないケースが多い → 閾値調整や軸テキストの見直しが課題。

## 出力フォルダ状況
- `research/sentiment_trials/output/LD01-Dialogue-01`: 17 セグメント。BERT はポジ寄りとネガ寄りが拮抗、pymlask 判定は 1 件のみ。
- `research/sentiment_trials/output/LD10-Dialogue-06`: 19 セグメント。BERT 平均はややネガ側、pymlask ネガ 4 件。
- `research/sentiment_trials/output/SD07-Dialogue-01`: 4 セグメント。BERT は全てポジ寄り、pymlask 未検出。
- `research/sentiment_trials/output/yuru`: 453 セグメント。pymlask は 41 件程度のみラベル付与。BERT スコアは弱いネガ傾向。

## 次の検討ポイント
- pymlask 辞書の追加や前処理（句読点、平仮名化など）で検出率を上げる。
- BERT マスクド LM のポジ/ネガ語彙、動/静語彙の拡張や、スコアに対する標準化・閾値見直し。
- Valence/Arousal タグ用の閾値を ±0.15〜0.2 へ引き下げ、全体の広がりを確認。
- `conversation_view_annotated.html` への追加表示（タグバッジ、数値表示など）で観察しやすくする。
