# src/ 概要

話者分離実験で利用する Python スクリプトを配置しています。主に以下の 3 つを提供します。

- `run_experiment.py` — 既存の `pipeline` モジュールを呼び出して話者分離を実行する CLI。設定 YAML と CLI オプションで pyannote・LLM のパラメータを切り替えられます。
- `evaluate_char.py` — 推定結果 (`segments.jsonl`) と参照テキストを突き合わせ、文字単位の一致率・混同行列を算出します。
- `report_html.py` — 正解と推定結果を並べた HTML を生成し、話者・テキストの差分を目視で確認できます。

## 依存

ルート直下の仮想環境（`.venv`）を利用します。未構築の場合は下記を実行してください。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 実験の流れ（例）

```bash
# 1. 実験を実行
python -m research.speaker_diarization.src.run_experiment \
  --input research/speaker_diarization/data/audio \
  --ref research/speaker_diarization/data/reference/LD01-Dialogue-01.jsonl \
  --output research/speaker_diarization/runs/baseline \
  --pyannote-threshold 0.70

# 2. 推定結果を評価
python -m research.speaker_diarization.src.evaluate_char \
  --hyp research/speaker_diarization/runs/baseline/LD01-Dialogue-01/segments.jsonl \
  --ref research.speaker_diarization/data/reference/LD01-Dialogue-01.jsonl

# 3. HTML レポートを生成
python -m research.speaker_diarization.src.report_html \
  --hyp research/speaker_diarization/runs/baseline/LD01-Dialogue-01/segments.jsonl \
  --ref research/speaker_diarization/data/LD01-Dialogue-01.txt \
  --output research/speaker_diarization/report
```

必要に応じて `runs/` 配下を `report/` フォルダに整理し、実験ログ・考察を Markdown で残してください。
