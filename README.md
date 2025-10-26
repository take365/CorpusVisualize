# CorpusVisualize Pipeline & UI

## 概要
- 音声会話から話者分離・文字起こし・感情/韻律/語彙などを抽出し、`segments.jsonl` を中心としたデータ群を生成する Python パイプライン。
- Quick diarization / Quick ASR 用アーティファクトを必要に応じて自動生成し、最終成果物は `output/<conversation_id>/` に集約。
- Vite + React 製 UI で `segments.jsonl` を読み込み、分析チャートやチャットビュー、比較 HTML を可視化できます。

## セットアップ手順
1. Python 3.12 以上で仮想環境を用意し、依存をインストール:
   ```bash
   pip install -r requirements.txt
   ```
2. `.env.sample` を `.env` にコピーし、必要な値を設定:
   ```bash
   cp .env.sample .env
   ```
   - LLM を使う場合は `CV_LLM_*`、pyannote を使う場合は `PYANNOTE_AUDIO_AUTH_TOKEN` を設定します。
3. UI を利用する場合は Node.js 環境で依存取得:
   ```bash
   cd ui
   npm install
   ```

## パイプライン構成
- エントリポイント: `python -m pipeline.run_pipeline`
- 設定ソース: `.env` → （必要なら）`--config` で指定した YAML → CLI 引数（優先）
- 主要モジュール
  - `pipeline/audio.py`: 音声読み込み & キャッシュ
  - `pipeline/diarization.py`: quick/pyannote/energy 話者分離
  - `pipeline/asr.py`: Quick 再利用 or Faster-Whisper
  - `pipeline/features/*`: 感情・ピッチ・語彙・語単位韻律抽出
  - `pipeline/report_html.py`: 正解テキストとの比較レポート生成
- Quick クラスタリング (`pipeline/hmd_quick_cluster.py`)
  - Whisper → ECAPA → PCA/KMeans → clusters/segments を生成
  - パイプライン実行時に該当アーティファクトが存在しない場合、自動で CLI を呼び出します。

## 使い方
```bash
# 単一ファイルを処理（Quick diarization / Quick ASR / LLM 無効が既定）
python -m pipeline.run_pipeline data/SD07-Dialogue-01.wav --no-llm

# ディレクトリ配下を一括処理
python -m pipeline.run_pipeline data/wav

# カスタム設定の例
python -m pipeline.run_pipeline data/LD10-Dialogue-06.wav \
  --diar pyannote --asr whisper-large-v3 --language ja --llm
```

### 主な CLI オプション
| オプション | 説明 |
| --- | --- |
| `inputs` 引数 | 音声ファイル or ディレクトリを列挙。省略時は `CV_INPUT_DIR` 配下を探索 |
| `--output-dir` | 出力ルート。Quick アーティファクトや比較 HTML も同じパスに生成 |
| `--diar`, `--asr` | 話者分離・ASR メソッド指定（`quick_cluster` / `pyannote` / `whisper-*` / `dummy`） |
| `--no-llm` / `--llm` | LLM による整形の有効化/無効化（`CV_LLM_*` と併用） |
| `--ref-dir` | 正解テキストディレクトリを明示（省略時は音声ファイルと同じディレクトリから自動探索） |
| `--comparison-html/--no-comparison-html` | 比較 HTML の生成 ON/OFF |

## 出力構造
```
output/<conversation_id>/
├─ segments.jsonl         # セグメント情報 (SegmentSchema)
├─ speakers.parquet       # 話者統計 (aggregate)
├─ <id>.raw.txt           # 生テキスト連結
└─ viz/
     ├─ quick_scatter.html   # Quick クラスタリング散布図
     └─ comparison.html      # 正解テキストとの文字単位比較 (リファレンスが存在する場合)
```
※ Quick diarization/ASR の中間成果物（segments/clusters/viz など）も同フォルダ内に生成されます。

## UI の利用方法
```bash
cd ui
npm install    # 初回のみ
npm run dev    # http://localhost:5173 で開発サーバー起動
```
- `segments.jsonl` もしくは JSON (配列/オブジェクト) を読み込むと以下を参照できます:
  - タイムライン、感情/ピッチ/音量/テンポ、方言スコア、語彙ハイライト
  - 話者別統計サマリ
  - チャットビュー: 音声同期または内部タイマーで語単位特徴を可視化（音声ファイルアップロード対応）
  - Web Speech API 録音（Chrome / Edge 推奨）→ 簡易解析 → JSON ダウンロード

## テスト
- 単体テスト: `pytest tests/test_schema.py`
- 動作確認例: `python -m pipeline.run_pipeline data/SD07-Dialogue-01.wav --no-llm`

## 補足
- SpeechBrain / pyannote / Faster-Whisper 等のモデルは初回実行時に自動ダウンロードされます（Git 管理対象外）。
- `.env` で LLM や pyannote の資格情報を設定しない場合、該当機能は自動的にフォールバックします。
- Quick diarization/ASR を使わずに動かしたい場合は `--diar pyannote --asr whisper-xxx` のように切り替えてください。
