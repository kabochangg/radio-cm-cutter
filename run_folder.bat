@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [INFO] .venv が見つからないため自動セットアップします...
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3 -m venv "%~dp0.venv"
  ) else (
    where python >nul 2>&1
    if not errorlevel 1 (
      python -m venv "%~dp0.venv"
    ) else (
      echo [ERROR] Python が見つかりません。Python 3.x をインストールしてください。
      pause
      exit /b 1
    )
  )

  if not exist "%PY%" (
    echo [ERROR] .venv の作成に失敗しました。
    pause
    exit /b 1
  )

  echo [INFO] 依存ライブラリをインストールしています...
  "%PY%" -m pip install --upgrade pip
  "%PY%" -m pip install numpy scipy
)

set "FOLDER=%~1"

if "%FOLDER%"=="" (
  for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; $d=New-Object System.Windows.Forms.FolderBrowserDialog; $d.Description='MP3フォルダを選択'; if($d.ShowDialog() -eq 'OK'){ $d.SelectedPath }"`) do set "FOLDER=%%I"
)

if "%FOLDER%"=="" (
  echo フォルダが選択されませんでした。
  pause
  exit /b 0
)

echo 対象フォルダ: "%FOLDER%"
echo.

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

"%PY%" "%~dp0radio_cm_cutter.py" process-folder "%FOLDER%"

if not %errorlevel%==0 (
  echo.
  echo [ERROR] 処理中にエラーが発生しました。
  pause
  exit /b %errorlevel%
)

echo.
echo 完了。出力は対象フォルダ内の cut_output にあります。
pause
