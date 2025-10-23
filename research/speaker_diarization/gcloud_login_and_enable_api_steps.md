# Google Cloud Speech-to-Text v2 利用手順

Google Cloud Speech-to-Text v2（話者分離対応）をこのリポジトリから利用するための準備手順です。`src/run_gcloud_stt_batch.py` を実行する前に以下を完了してください。

## 1. 前提条件
- Google Cloud プロジェクト（課金有効化済み）
- `gcloud` CLI と Python 3.10+ がローカル環境にインストール済み
- （バッチ推論を使う場合）Cloud Storage バケットを作成できる権限
- リポジトリのルートで `pip install google-auth google-auth-httplib2 google-auth-oauthlib google-api-core google-cloud-speech google-cloud-core google-cloud-storage` を実行し、必要な Python パッケージをインストール

## 2. GCP 認証
対話的に CLI を利用する場合は下記 2 コマンドを順番に実行し、ブラウザ認証を済ませます。

```bash
gcloud auth login
gcloud auth application-default login
```

サービスアカウント鍵を利用する場合は、JSON 鍵をダウンロードし `export GOOGLE_APPLICATION_CREDENTIALS=/path/to/key.json` を設定してから進めてください。

## 3. プロジェクトとリージョンの設定
```bash
gcloud config set project YOUR_GCP_PROJECT
gcloud config set compute/region asia-northeast1
```

Speech v2 はリージョン単位のリソースです。実行時に指定する `--location` も同じリージョンに合わせてください。

## 4. API の有効化
Speech-to-Text v2 と、バッチ実行で利用する Cloud Storage API を有効化します。

```bash
gcloud services enable speech.googleapis.com storage.googleapis.com
```


> **注意**: 2025年10月時点では ja-JP (日本語) モデルに話者分離 (speaker diarization) 機能は提供されていません。英語など一部モデルのみ対応です。日本語で話者分離を利用したい場合は別製品（WhisperX など）の検討が必要です。
## 5. Recognizer の作成
音声モデルと言語、話者数などを指定して Recognizer リソースを準備します。`gcloud alpha` トラックを利用します。

```bash
gcloud alpha ml speech recognizers create ja-rec \
  --project=YOUR_GCP_PROJECT \
  --location=asia-northeast1 \
  --language-codes=ja-JP \
  --model=long \
  --enable-word-time-offsets \
  --enable-word-confidence \
  --min-speaker-count=2 \
  --max-speaker-count=2
```

- `ja-rec` は任意の Recognizer ID です。
- `--model` は用途に合わせて `long`, `short`, `chirp` などを指定してください。
- 既存の Recognizer を使う場合はこのステップは不要です。

## 6. バッチ推論用の GCS バケット（任意）
長時間音声を処理する際は `--use-batch` オプションを利用します。入力音声と出力 JSON を保存するバケットを作成し、適切な IAM を付与してください。

```bash
gcloud storage buckets create gs://YOUR_BUCKET --project=YOUR_GCP_PROJECT --location=asia-northeast1
```

実行アカウントに `Storage Object Admin` などの権限が必要です。

## 7. スクリプトの実行例
### インライン（認識時間が短い場合 / v2 利用）
```bash
python3 research/speaker_diarization/src/run_gcloud_stt_batch.py \
  --inputs research/speaker_diarization/data \
  --pattern "*.wav" \
  --ref-dir research/speaker_diarization/data \
  --output-root research/speaker_diarization/runs/gcloud \
  --exp-slug chirp_v2_diar \
  --project YOUR_GCP_PROJECT \
  --location asia-northeast1 \
  --recognizer ja-rec \
  --language ja-JP \
  --model long \
  --min-speakers 2 --max-speakers 2 \
  --make-html
```

### バッチ（長時間音声向け / v2 利用）
```bash
python3 research/speaker_diarization/src/run_gcloud_stt_batch.py \
  --inputs research/speaker_diarization/data \
  --pattern "*.wav" \
  --ref-dir research/speaker_diarization/data \
  --output-root research/speaker_diarization/runs/gcloud \
  --exp-slug chirp_v2_diar \
  --project YOUR_GCP_PROJECT \
  --location asia-northeast1 \
  --recognizer ja-rec \
  --language ja-JP \
  --model long \
  --min-speakers 2 --max-speakers 2 \
  --use-batch \
  --gcs-bucket YOUR_BUCKET \
  --gcs-prefix stt_in \
  --gcs-output-prefix stt_out \
  --make-html
```

- `--use-batch` をつけると音声ファイルを Cloud Storage にアップロードし、`batchRecognize` API で非同期処理を行います。
- 出力は `runs/gcloud/<日付>_<exp-slug>/<音声名>/gcloud/*.jsonl` に保存され、必要に応じて `metrics.json` や HTML レポートが生成されます。

### Enhanced モデル（v1p1beta1）で話者分離を試す場合
`--api-version v1p1beta1` を付けると、Speech-to-Text v1p1beta1 API の enhanced モデル（例: `model=video` や `medical_conversation`）を利用します。リクエストはインラインモードのみ対応で、音声は内部で 16kHz モノラル PCM に変換されます。

```bash
python3 research/speaker_diarization/src/run_gcloud_stt_batch.py \
  --inputs research/speaker_diarization/data/en_multi_mono.wav \
  --output-root research/speaker_diarization/runs/gcloud \
  --exp-slug en_us_enhanced \
  --project YOUR_GCP_PROJECT \
  --api-version v1p1beta1 \
  --language en-US \
  --model video \
  --min-speakers 2 --max-speakers 2
```

- Enhanced モデルの diarization は Google の公開情報に従います（Preview 扱いの場合はアクセス権が必要なことがあります）。
- v1p1beta1 では Recognizer リソースを事前作成する必要はありません。

## 8. トラブルシューティング
- `google.auth.exceptions.RefreshError` が出る場合：`gcloud auth application-default login` を再実行するか、サービスアカウント鍵を設定してください。
- `403 PERMISSION_DENIED`：Speech-to-Text や Storage の権限（`roles/speech.admin`, `roles/storage.objectAdmin` など）が付与されているか確認。
- バッチ出力が見つからない：`--gcs-output-prefix` に指定したパスに対して読み取り権限があるか、`pick_result_json_from_gcs` が参照する JSON ファイルが生成されているか確認してください。

以上で準備完了です。`src/run_gcloud_stt_batch.py --help` で詳細なオプションを確認できます。
