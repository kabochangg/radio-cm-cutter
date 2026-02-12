# radio-cm-cutter

MP3 音源から CM 区間を検出し、CM を除いた MP3 をまとめて出力するツールです。

## UI 改善ポイント

- `process-folder` 実行時に、見出し・ステップ・結果を見やすく表示します。
- ファイルごとの進捗表示を揃え、`SKIP / RUN / NG` が判別しやすくなっています。
- 最後に成功件数・失敗件数・合計 CM 秒数をまとめて表示します。

## 必要環境

- Python 3.x
- FFmpeg（`ffmpeg` / `ffprobe` が PATH に通っていること）
- 依存ライブラリ: `numpy`, `scipy`

> 追加ライブラリは増やしていません。

## 使い方（Windows かんたん実行）

### 1) `run_folder.bat` を実行

- ダブルクリック、または `run_folder.bat "対象フォルダ"` で実行できます。
- 引数なし実行時はフォルダ選択ダイアログが開きます。

### 2) 自動セットアップ

`run_folder.bat` は以下を自動で行います。

1. `.venv` が無ければ自動作成
2. `numpy`, `scipy` をインストール
3. `ffmpeg` / `ffprobe` の存在チェック
4. `radio_cm_cutter.py process-folder` を実行

### 3) 出力先

- `対象フォルダ/cut_output/cut` : CM 除去後 MP3
- `対象フォルダ/cut_output/segments` : 検出 CSV
- `対象フォルダ/cut_output/reports` : HTML レポート

## CLI 例

```bash
python radio_cm_cutter.py process-folder "C:/music/radio" --recursive
```

## トラブルシュート

- `ffmpeg が見つかりません` / `ffprobe が見つかりません`
  - FFmpeg をインストールし、PATH を設定してください。
- 処理途中でエラーが出る
  - 出力ログの `NG` 行にファイル名と原因が表示されます。
