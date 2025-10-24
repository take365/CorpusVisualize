# Quick先行→Pipeline再利用 仕様（Codex対応タスク）

## ゴール
- **先に Quick を実行**して Whisper+ECAPA+KMeans で話者クラスタを確定し、
- **Pipeline 側は Quick の成果物を再利用**して「話者分離（=クラスタ割当）」と「ASR（文字起こし）」を読み込む方式に変更する。
- 環境変数で **オン/オフ切替**でき、既存フローを壊さない。

---

## 変更概要（最小）
1. **Diarization 置換**：`CV_DIARIZATION=quick_cluster`
   - `pipeline/diarization.py` に **QuickClusterDiarizer** を追加。
   - 入力音源と同一ベース名の **`HMD/data/<basename>/segments.jsonl`** と **`clusters/speaker_embedding/kmeans_pca2.json`** を読み、
     `seg_id -> cluster(int)` を `A/B/C...` に写像して **`List[DiarizationSegment]`** を返す。

2. **ASR 再利用**：`CV_ASR=quick`
   - `pipeline/asr.py` に **`method == "quick"`** 分岐を追加。
   - **`segments.jsonl`**（必要に応じて **`words.jsonl`**）を読み、`asr_results`（start/end/text/words）を構成。
   - 既存の `_align_transcripts(diar_segments, asr_results)` に渡して整列を流用。

3. **構成管理**：`.env` に Quick 系の切替とパスを追記（すでに Quick 用 `.env` あり）。

---

## 追加/修正ファイル
- 追加: `pipeline/hmd_quick_cluster.py`（配置済）
- 変更: `pipeline/diarization.py`（factory + 新クラス）
- 変更: `pipeline/asr.py`（`transcribe()` に quick 分岐）
- 参照のみ: `pipeline/run_pipeline.py`（既存のまま／ENV 経由で切替）

---

## インターフェース仕様
### Diarization 呼び出し
```python
__call__(audio: np.ndarray, sample_rate: int) -> List[DiarizationSegment]
# DiarizationSegment: {start: float, end: float, speaker: str}
```
- **speaker** は `"A"..` に正規化。
- Quick 側の `cluster`(0..k-1) → `chr(ord('A') + cluster)` に写像。

### ASR 再利用
```python
transcribe(audio, sample_rate, diar_segments, method='quick', language='ja', ...)
# quick: HMD/data/<basename>/segments.jsonl / words.jsonl を読み asr_results を構成
# 既存 _align_transcripts(diar_segments, asr_results) を使用
```

---

## 実装タスク一覧（Codex向け）
- [ ] **Diarization**: `QuickClusterDiarizer` を `pipeline/diarization.py` に実装
  - [ ] `__init__(self, data_root: str = 'HMD/data')`
  - [ ] `__call__(audio, sr)` 内で `basename` 推定、`segments.jsonl` / `kmeans_pca2.json` をロード
  - [ ] `seg_id` キー整合（`segments.jsonl`のIDとクラスタ対応）
  - [ ] `cluster -> 'A','B',...` 変換し `List[DiarizationSegment]` を返す
  - [ ] 例外: 成果物がない場合は明示的に `RuntimeError`（初期はフォールバック無し）

- [ ] **ASR**: `pipeline/asr.py` に `method == 'quick'` 分岐
  - [ ] `segments.jsonl` を読み `asr_results = [{start,end,text,words?}]` を構築
  - [ ] `words.jsonl` があれば `words` に格納（無ければ空）
  - [ ] 既存 `_align_transcripts(diar_segments, asr_results)` をそのまま利用

- [ ] **ENV 読み**: `CV_DIARIZATION` / `CV_ASR` 読み込みは既存を流用（変更不要）

- [ ] **ユーティリティ**: `basename` 推定ロジック（拡張子除去）と `data_root` 連結

- [ ] **ログ/メトリクス**: Quick モード時に参照ファイルパスとクラスタ k をINFO出力

---

## .env 例（Pipeline 側）
```dotenv
# Pipeline core
CV_DIARIZATION=quick_cluster
CV_ASR=quick
CV_LANGUAGE=ja
CV_SAMPLE_RATE=16000

# Quick 成果物のルート（Quick .env と整合させる）
HMD_DATA_ROOT=HMD/data
```
※ Quick 側は別 `.env` にて `WHISPER_MODEL=large-v3-turbo` ほか設定済。

---

## 実行手順（運用）
1. **Quick 実行**（音源: `X.wav`）
   - 出力: `HMD/data/X/{segments.jsonl, words.jsonl?, clusters/.../kmeans_pca2.json, ...}`
2. **Pipeline 実行**（同じ `X.wav` を入力）
   - `.env` で `CV_DIARIZATION=quick_cluster` / `CV_ASR=quick`
   - `run_pipeline.py` は Quick 成果物を読み込み、以降は通常通り（特徴量→出力）

---

## エラーハンドリング
- `segments.jsonl` / `kmeans_pca2.json` 不在 → 例外で停止（メッセージに想定パスを表示）
- `seg_id` 不一致 → 警告ログ + スキップ or 停止（初期は停止を推奨）
- `k=1` 等の単一クラスタ → そのまま `speaker='A'` で通す（情報として警告）

---

## テスト計画
1. **最小音源**（~30秒・2話者）で Quick → Pipeline を通し、話者ラベルと文字起こしが整合しているか確認。
2. **`words.jsonl` 有無**の2通りで、語単位整列が落ちないことを確認。
3. **k固定/自動**の2通り（Quick 側 `.env`: `HMD_K_MIN/HMD_K_MAX`）で A/B/C マッピングが安定すること。
4. **存在しない basename** で適切なエラーメッセージが出ること。

---

## 既知の注意点
- Quick Whisper の文単位境界と Pipeline の整列処理の切り方が微妙に違う場合、境界±数百 ms 程度のズレが起こり得る（許容）。
- 追加の正規化（例: 句読点付与や全角半角処理）を行う場合は、整列前に行うこと（テキスト同一性に依存する処理がある場合に注意）。

---

## 将来拡張（任意）
- Quick の **ASR 完全採用モード**：Pipeline 側で整列をスキップして、Quick の `segments.jsonl` をそのまま出力に反映する軽量パス。
- **フォールバック**：Quick 成果物が無い場合、`energy_split` へ自動切替（フラグで制御）。
- **クラスタ名の永続化**：A/B/C を `speakers.yaml` に保存し、再実行時のラベル揺れを抑制。

---

## 受け入れ基準（Acceptance Criteria）
- [ ] `.env` の切替のみで **Quick→Pipeline** が動作する
- [ ] `segments.jsonl` のテキストが最終 `segments.jsonl` に反映される（ASR 再利用）
- [ ] 話者ラベルが `A..` で連番付与され、セグメント数が Quick のセグメント数と一致
- [ ] 既存の特徴量抽出（emotion/pitch/loudness/tempo など）が動作
- [ ] 主要ログに Quick 参照パスとクラスタ k が出る

---

## 擬似コード（参考・最小差分）
### `pipeline/diarization.py`
```python
class QuickClusterDiarizer:
    def __init__(self, data_root: str = os.getenv('HMD_DATA_ROOT', 'HMD/data')):
        self.data_root = data_root
    def __call__(self, audio, sr) -> List[DiarizationSegment]:
        basename = guess_basename(audio)  # 入力パスから拡張子除去
        seg_path = f"{self.data_root}/{basename}/segments.jsonl"
        clu_path = f"{self.data_root}/{basename}/clusters/speaker_embedding/kmeans_pca2.json"
        segs = load_segments(seg_path)         # [{id,start,end,text}]
        clu = load_cluster_map(clu_path)       # {seg_id: cluster_int}
        out = []
        for s in segs:
            c = clu[s['id']]
            spk = chr(ord('A') + int(c))
            out.append(DiarizationSegment(start=s['start'], end=s['end'], speaker=spk))
        return out

# factory
if mode == 'quick_cluster':
    return QuickClusterDiarizer()
```

### `pipeline/asr.py`
```python
elif method == 'quick':
    segs = load_segments(seg_path)  # [{start,end,text, words?}]
    asr_results = [{
        'start': s['start'], 'end': s['end'], 'text': s.get('text',''),
        'words': s.get('words', []),
    } for s in segs]
    return _align_transcripts(diar_segments, asr_results)
```

---

## コマンド例
```bash
# 1) Quick を先行実行（例）
python pipeline/hmd_quick_cluster.py --audio path/to/X.wav --data-root HMD/data --reuse-segments 1

# 2) Pipeline を Quick 再利用モードで実行
export CV_DIARIZATION=quick_cluster
export CV_ASR=quick
python pipeline/run_pipeline.py --audio path/to/X.wav --output-root outputs/X
```

