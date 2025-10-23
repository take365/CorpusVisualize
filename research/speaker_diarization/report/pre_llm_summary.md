# LLM 導入前 話者分離まとめ

## 概要
- Energy 方式は正解率 55〜71% 程度。短区間にすると挿入が増えるものの正解率は向上。
- pyannote.audio 方式はチューニングによって 35〜48% 程度。threshold=0.80 が最も良好。
- 最高スコア: Energy=70.6% (energy_min_0p5s), pyannote=47.8% (py_th080).
- 話者正解率は Energy 系で 58〜77%、pyannote 系で 48〜72% 程度。一致文字のみ評価しても話者入れ替わりが残る。

## Google Cloud Speech-to-Text API 検証メモ
- v2 の Recognizer（ja-JP）に話者分離設定を付けて呼び出したが、レスポンスが `Recognizer does not support feature: speaker_diarization` となり、日本語モデルでは diarization がまだ提供されていないことを確認。
- 公開ドキュメントに従い enhanced モデル（v1p1beta1, `model=video`, `language=en-US`）で日本語音声を処理してみたが、出力は英語前提で 1 セグメントのみ検出、文字一致率も 1% 程度とほぼ失敗。
- 現時点で Google API の話者分離を日本語に適用するのは困難と判断。国内ミーティング用途では WhisperX や pyannote など既存手段の継続検討が必要。
