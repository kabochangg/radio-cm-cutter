@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [INFO] .venv が見つからないため自動セットアップします...
  where py >nul 2>&1
  if %errorlevel%==0 (
    py -3 -m venv "%~dp0.venv"
  ) else (
    where python >nul 2>&1
    if %errorlevel%==0 (
      python -m venv "%~dp0.venv"
    ) else (
      echo [ERROR] Python が見つかりません。Python 3.11 をインストールしてください。
      pause
      exit /b 1
    )
  )

  if not exist "%PY%" (
    echo [ERROR] .venv の作成に失敗しました。
    pause
    exit /b 1
  )
)

echo [INFO] 依存ライブラリを確認/インストールしています...
"%PY%" -m pip install --upgrade pip
"%PY%" -m pip install numpy scipy scikit-learn

where ffmpeg >nul 2>&1
if not %errorlevel%==0 (
  echo [ERROR] ffmpeg が見つかりません。PATH を確認してください。
  pause
  exit /b 1
)

where ffprobe >nul 2>&1
if not %errorlevel%==0 (
  echo [ERROR] ffprobe が見つかりません。PATH を確認してください。
  pause
  exit /b 1
)

:MENU
echo.
echo ================================================
echo   Radio CM Cutter メニュー
echo ================================================
echo   1. フォルダ一括カット（推論）
echo   2. 学習用データ作成（テンプレ生成）
echo   3. 学習（train）
echo   4. レポート/確認（reportを開く）
echo   5. 学習データ評価（evaluate）
echo   0. 終了
echo ================================================
set /p CHOICE=番号を入力してください: 

if "%CHOICE%"=="1" goto RUN_INFER
if "%CHOICE%"=="2" goto MAKE_TEMPLATE
if "%CHOICE%"=="3" goto TRAIN
if "%CHOICE%"=="4" goto OPEN_REPORT
if "%CHOICE%"=="5" goto EVALUATE
if "%CHOICE%"=="0" goto END

echo [WARN] 無効な入力です。
goto MENU

:RUN_INFER
set "FOLDER=%~1"
if "%FOLDER%"=="" (
  for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description='MP3フォルダを選択'; if($d.ShowDialog() -eq 'OK'){ $d.SelectedPath }"`) do set "FOLDER=%%I"
)

if "%FOLDER%"=="" (
  echo [INFO] フォルダが選択されませんでした。
  goto MENU
)

echo [INFO] 対象フォルダ: "%FOLDER%"
"%PY%" "%~dp0radio_cm_cutter.py" process-folder "%FOLDER%" --recursive
if not %errorlevel%==0 (
  echo [ERROR] 推論処理でエラーが発生しました。
)
goto MENU

:MAKE_TEMPLATE
set /p TEMPLATE=出力先CSV（既定: data/labels/template_new.csv）: 
if "%TEMPLATE%"=="" set "TEMPLATE=data/labels/template_new.csv"
set /p AUDIO=サンプル音声パス（任意。空欄可）: 
if "%AUDIO%"=="" (
  "%PY%" "%~dp0radio_cm_cutter.py" init-label-template --out "%TEMPLATE%"
) else (
  "%PY%" "%~dp0radio_cm_cutter.py" init-label-template --out "%TEMPLATE%" --audio-path "%AUDIO%"
)
if not %errorlevel%==0 (
  echo [ERROR] テンプレ生成に失敗しました。
)
goto MENU

:TRAIN
set /p PATTERN=ラベルCSVパターン（既定: data/labels/*.csv）: 
if "%PATTERN%"=="" set "PATTERN=data/labels/*.csv"
set /p MODEL_OUT=モデル出力先（既定: model/model.pkl）: 
if "%MODEL_OUT%"=="" set "MODEL_OUT=model/model.pkl"
"%PY%" "%~dp0radio_cm_cutter.py" train --labels "%PATTERN%" --out "%MODEL_OUT%"
if not %errorlevel%==0 (
  echo [ERROR] 学習に失敗しました。
)
goto MENU

:OPEN_REPORT
set /p REP_FOLDER=処理済みフォルダ（cut_output がある場所）を入力: 
if "%REP_FOLDER%"=="" (
  echo [INFO] フォルダ未入力です。
  goto MENU
)
if exist "%REP_FOLDER%\cut_output" (
  start "" "%REP_FOLDER%\cut_output"
) else (
  echo [WARN] "%REP_FOLDER%\cut_output" が見つかりません。
)
goto MENU

:EVALUATE
set /p PATTERN=ラベルCSVパターン（既定: data/labels/*.csv）: 
if "%PATTERN%"=="" set "PATTERN=data/labels/*.csv"
set /p MODEL_PATH=評価モデルパス（既定: model/model.pkl）: 
if "%MODEL_PATH%"=="" set "MODEL_PATH=model/model.pkl"
"%PY%" "%~dp0radio_cm_cutter.py" evaluate --labels "%PATTERN%" --model "%MODEL_PATH%"
if not %errorlevel%==0 (
  echo [ERROR] evaluate に失敗しました。
)
goto MENU

:END
echo 終了します。
exit /b 0
