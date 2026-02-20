# radio-cm-cutter

運用者向け: MP3音源からCM区間を検出し、CM除去済みMP3を作成するCLIツールです。

## Quickstart

### 1) 前提
- Python 3.11 推奨
- FFmpeg (`ffmpeg` / `ffprobe` をPATHに設定)
- `pip install -r requirements.txt`

### 2) ヘルプ確認
```bash
python radio_cm_cutter.py --help
```

### 3) 基本運用（単体）
```bash
python radio_cm_cutter.py detect input.mp3
python radio_cm_cutter.py cut input.mp3 --segments input_segments.csv
```

### 4) 基本運用（フォルダ）
```bash
python radio_cm_cutter.py process-folder "C:/radio" --recursive
```
- デフォルト出力先は `cut_output/YYYYMMDD_HHMMSS/` です（上書き事故防止）。
- 同一出力先の既存ファイルを上書きしたい場合のみ `--overwrite` を使用してください。

### 5) 学習/評価
```bash
python radio_cm_cutter.py train --labels data/labels/*.csv --out model/model.pkl
python radio_cm_cutter.py evaluate --labels data/labels/*.csv --model model/model.pkl
```

### 6) GUI起動（MVP / Windows想定）
```bash
python -m radio_cm_cutter_app
```
- 入力フォルダ、`model.pkl`、出力フォルダを選択して「実行」を押すだけで `process-folder` 相当を実行します。
- 未指定時の出力先は `cut_output/YYYYMMDD_HHMMSS/` です。
- 設定値（入力/出力/モデル/ffmpegパス）は `%APPDATA%/radio-cm-cutter/gui_settings.json` に保存され、次回起動時に復元されます。

## CLI一覧
- `detect`
- `detect-ml`
- `cut`
- `process-folder`
- `init-label-template`
- `train`
- `evaluate`

## 出力
`process-folder` の出力:
- `.../cut_output/<timestamp>/cut` : CM除去後MP3
- `.../cut_output/<timestamp>/segments` : 検出CSV
- `.../cut_output/<timestamp>/reports` : HTMLレポート

## トラブルシュート

### ffmpeg/ffprobe が見つからない
- 症状: 実行時に `Required command not found`。
- 対応: FFmpegをインストールし、`ffmpeg` と `ffprobe` のPATHを通してください。

### モデルが読み込めない
- 症状: `Model not found` / モデルロード例外。
- 対応: `--model` のパス、`config.json` の `model_path`、モデルファイルの破損有無を確認してください。

### 入力ファイル/フォルダが見つからない
- 症状: `FileNotFoundError` / `Folder not found`。
- 対応: 絶対パスで指定し、存在とアクセス権限を確認してください。

### 出力で失敗する
- 症状: 権限エラーや書き込み失敗。
- 対応: 書き込み可能な場所を指定するか、別の `--out-dir` / `--out` を利用してください。

## 運用ポリシー
- `model/` と `cut_output/` はGit管理しない運用を推奨（`.gitignore`で除外）。
- 検出ロジック・特徴量・しきい値の意味は変更せず、構造改善中心で保守してください。

## GUI 手動テスト（最小）
1. `python -m radio_cm_cutter_app` を起動。
2. 入力フォルダに MP3 を1つ以上含むディレクトリを選択（例: `C:/radio_input`）。
3. モデルは既定の `model/model.pkl` を使うか、任意の `.pkl` を指定。
4. 出力フォルダは空欄なら `cut_output/<timestamp>/`、または任意ディレクトリを指定。
5. 「実行」を押し、ログ欄で進行を確認。
6. 完了後、`<出力>/reports/*_report.html` が生成されることを確認（`ml_report.html` がある場合はGUIボタンから開けます）。
