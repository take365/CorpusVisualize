# Speaker Diarization Research Sandbox

このフォルダは、話者分離（diarization）を集中的に検証するためのサンドボックスです。本体パイプラインから切り離し、再現性のある実験・評価を行うことを目的としています。

## ディレクトリ構成

```
research/speaker_diarization/
├─ data/         # 実験用の音声・トランスクリプト・ラベルを配置
├─ src/          # 実験スクリプトと評価ツール
├─ report/       # 結果まとめ・ノート
└─ README.md
```

## 主なスクリプト

- `src/run_experiment.py` : 既存パイプライン部品を呼び出し、pyannote パラメータや LLM 設定を切り替えながら処理を実行。
- `src/evaluate_char.py` : 推定結果 (`segments.jsonl`) と参照テキストを突き合わせ、文字単位の正解率をレポート。

## 使い方の概要

1. `data/` に以下のファイルを配置してください。
   - `audio/` … WAV など音声ファイル
   - `reference/` … 話者ラベル付きの参照トランスクリプト (例: JSONL または TSV)
2. `src/run_experiment.py` を実行し、`--config` もしくは CLI オプションで pyannote・LLM パラメータを指定します。
3. 出力は `runs/<timestamp>/...` に保存されます。必要に応じて `report/` に結果をまとめてください。

詳細は `src/README.md` を参照してください。
