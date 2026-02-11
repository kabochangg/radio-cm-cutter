@echo off
chcp 65001 >nul
setlocal

cd /d "%~dp0"

set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" (
  echo [ERROR] .venv が見つかりません。
  echo まず PowerShell で次を実行してください:
  echo   py -3.11 -m venv .venv
  echo   .\.venv\Scripts\Activate.ps1
  echo   pip install numpy scipy
  pause
  exit /b 1
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

"%PY%" "%~dp0radio_cm_cutter.py" process-folder "%FOLDER%"

echo.
echo 完了。出力は対象フォルダ内の cut_output にあります。
pause
