# radio-cm-cutter

運用者向け: MP3音源からCM区間を検出し、CM除去済みMP3を作成するツールです。

## 日常運用（最初に読む）

普段触る場所は基本的に次の3つです。
- **app起動**: `python -m radio_cm_cutter_app`
- **model（保存先）**: `model/model.pkl`
- **output（出力先）**: 実行時に指定した出力先（未指定時は `cut_output/YYYYMMDD_HHMMSS/`）

### GUI起動
```bash
python -m radio_cm_cutter_app
```

### GUIでの通常フロー
1. 入力フォルダを選択
2. モデル（通常は `model/model.pkl`）を選択
3. 出力フォルダを選択（空欄なら `cut_output/YYYYMMDD_HHMMSS/`）
4. 実行モードを選択（検出のみ / 検出+カット）
5. 実行

---

## トラブルシュート

### ffmpeg/ffprobe が見つからない
- 症状: `Required command not found`。
- 対応: FFmpegをインストールし、`ffmpeg` と `ffprobe` のPATHを通してください。
- GUIでは以下を案内します。
  - 自動検索結果（PATH、前回指定パス、`%LOCALAPPDATA%/radio-cm-cutter/ffmpeg/`）
  - `ffmpeg.exeを選択` / `導入手順を開く(README)` / `キャンセル`
  - 任意でFFmpegダウンロード・展開

### モデルが読み込めない
- 症状: `Model not found` / モデルロード例外。
- 対応: `model/model.pkl` の存在、指定パス、ファイル破損有無を確認してください。

### 出力で失敗する
- 症状: 権限エラーや書き込み失敗。
- 対応: 書き込み可能な場所を指定するか、別の出力先を選択してください。

---

## 学習 / 評価

GUIから実行できます（train / evaluate）。

CLIでも従来どおり実行できます。
```bash
python radio_cm_cutter.py train --labels data/labels/*.csv --out model/model.pkl
python radio_cm_cutter.py evaluate --labels data/labels/*.csv --model model/model.pkl
```

---

## 高度設定 / 開発者向け

### CLI互換エントリポイント
```bash
python radio_cm_cutter.py --help
```

### 主なCLI
- `detect`
- `detect-ml`
- `cut`
- `process-folder`
- `init-label-template`
- `train`
- `evaluate`

### 出力構成（process-folder）
- `.../cut_output/<timestamp>/cut`
- `.../cut_output/<timestamp>/segments`
- `.../cut_output/<timestamp>/reports`

### Windows向けEXEビルド
- `./.internal/scripts/build_windows.bat`
- `./.internal/scripts/build_windows.ps1`

### インストーラ作成（任意）
- `./.internal/scripts/build_installer.ps1`
- Inno Setup script: `./.internal/installer/RadioCmCutter.iss`

### GitHub Actions
- workflow: `.github/workflows/build-windows-exe.yml`
- `windows-latest` 上で `./.internal/scripts/build_windows.ps1` を実行

### 最低限の手動確認
1. `python -m radio_cm_cutter_app`
2. detect-only
3. detect+cut
4. train
5. evaluate

### 運用ポリシー
- `model/` と `cut_output/` はGit管理しない運用を推奨（`.gitignore`）。
- 検出ロジック・特徴量・しきい値の意味は変更せず、構造改善中心で保守。
