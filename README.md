# radio-cm-cutter

MP3音源からCM区間を検出し、CMを除いたMP3を出力するツールです。  
従来の音量差ルール（ヒューリスティック）に加えて、ラベル付きデータから学習するML検出を使えるようになりました。

## 必要環境

- Python 3.11 推奨
- FFmpeg（`ffmpeg` / `ffprobe` がPATHに通っていること）
- 依存ライブラリ
  - `numpy`
  - `scipy`
  - `scikit-learn`

---

## できること

- `detect`: CM区間検出（**モデルがあればML優先**、無ければ従来ルール）
- `detect-ml`: MLモデル必須でCM区間検出
- `cut`: 検出CSVを使ってCM部分を除去
- `process-folder`: フォルダ内MP3を一括検出＆カット（モデルがあれば自動使用）
- `init-label-template`: 学習ラベルのテンプレCSV生成
- `train`: ラベルCSVからモデル学習
<<<<<<< ours
=======
- `evaluate`: 学習データ上で混同行列 / precision / recall / F1 を確認
>>>>>>> theirs

---

## 学習データの形式（`data/labels/*.csv`）

ラベルCSVは以下の列を持ちます。

- `audio_path`: 元音声ファイルのパス
- `start_sec`: 区間開始秒
- `end_sec`: 区間終了秒
- `label`: `cm` または `program`
- `note`: 任意メモ

例（`data/labels/template.csv`）:

```csv
audio_path,start_sec,end_sec,label,note
path/to/radio.mp3,0.0,15.0,program,番組本編の例
path/to/radio.mp3,120.0,150.0,cm,CM区間の例
```

> ラベルはあとからCSVに追記して継続的に増やせます。

---

## 学習の流れ（継続学習）
<<<<<<< ours

1. テンプレート作成

```bash
python radio_cm_cutter.py init-label-template --out data/labels/new_labels.csv
```

2. `data/labels/*.csv` に `cm/program` 区間を追記
3. 学習

```bash
python radio_cm_cutter.py train --labels data/labels/*.csv --out model/model.pkl
```

4. 推論（モデルがあれば自動利用）

```bash
python radio_cm_cutter.py process-folder "C:/radio"
```

新しいラベルを追加したら、再度 `train` を実行して `model/model.pkl` を更新してください。

---

## CLI例

### 単体ファイル検出（自動でML優先）

```bash
python radio_cm_cutter.py detect input.mp3
```

### 単体ファイル検出（MLのみ）

```bash
python radio_cm_cutter.py detect-ml input.mp3 --model model/model.pkl
```

### 学習

```bash
python radio_cm_cutter.py train --labels data/labels/*.csv --out model/model.pkl
```

=======

1. テンプレート作成

```bash
python radio_cm_cutter.py init-label-template --out data/labels/new_labels.csv
```

2. `data/labels/*.csv` に `cm/program` 区間を追記
3. 学習

```bash
python radio_cm_cutter.py train --labels data/labels/*.csv --out model/model.pkl
```

4. 推論（モデルがあれば自動利用）

```bash
python radio_cm_cutter.py process-folder "C:/radio"
```

新しいラベルを追加したら、再度 `train` を実行して `model/model.pkl` を更新してください。

---

## CLI例

### 単体ファイル検出（自動でML優先）

```bash
python radio_cm_cutter.py detect input.mp3
```

### 単体ファイル検出（MLのみ）

```bash
python radio_cm_cutter.py detect-ml input.mp3 --model model/model.pkl
```

### 学習

```bash
python radio_cm_cutter.py train --labels data/labels/*.csv --out model/model.pkl
```

>>>>>>> theirs
### フォルダ一括

```bash
python radio_cm_cutter.py process-folder "C:/music/radio" --recursive
```

---

<<<<<<< ours
=======
### 学習データ評価

```bash
python radio_cm_cutter.py evaluate --labels data/labels/*.csv --model model/model.pkl
```

フレーム単位で以下を表示します。

- 混同行列（TP/FN/FP/TN）
- Precision / Recall / F1


>>>>>>> theirs
## `run_folder.bat` のメニュー操作

`run_folder.bat` を起動すると次のメニューが出ます。

1. フォルダ一括カット（推論）
2. 学習用データ作成（テンプレ生成）
3. 学習（train）
4. レポート/確認（reportを開く）
<<<<<<< ours
=======
5. 学習データ評価（evaluate）
>>>>>>> theirs

日本語メッセージで案内されるので、順番に実行できます。

---

<<<<<<< ours
=======

レポート（`*_report.html`）には、検出セグメントに加えて **MLスコアタイムライン**（赤が濃いほどCM確率/スコア高）も表示されます。

>>>>>>> theirs
## モデルとフォールバック

- モデル保存先デフォルト: `model/model.pkl`
- `process-folder` / `detect` は、モデルが存在すればML推論を使います。
- モデルが無い、または読み込み失敗時は従来の音量差ルールにフォールバックします。

`config.json` で以下を調整可能です。

- `model_path`
- `ml_start_prob` / `ml_end_prob`
- 従来方式の `start_delta_db`, `end_delta_db`, `min_cm_sec`, `merge_gap_sec`, `pad_sec`

---

## うまくいかないときの調整

1. **誤検出が多い**
   - ラベルCSVに「program」の例を増やす
   - `ml_start_prob` を上げる（例: 0.60）
2. **取りこぼしが多い**
   - ラベルCSVに「cm」の例を増やす
   - `ml_start_prob` を下げる（例: 0.50）
3. **区間が細切れになる**
   - `merge_gap_sec` を増やす
   - `min_cm_sec` を増やす
4. **境界が厳しすぎる**
   - `pad_sec` を増やす

---

## 出力先

`process-folder` の出力:

- `対象フォルダ/cut_output/cut` : CM除去後MP3
- `対象フォルダ/cut_output/segments` : 検出CSV
- `対象フォルダ/cut_output/reports` : HTMLレポート
