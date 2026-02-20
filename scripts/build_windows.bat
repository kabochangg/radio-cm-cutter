@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
pushd "%REPO_ROOT%"

set "VENV_DIR=%REPO_ROOT%\.venv-build"
set "DIST_DIR=%REPO_ROOT%\dist"
set "BUILD_DIR=%REPO_ROOT%\build"
set "SPEC_FILE=%SCRIPT_DIR%radio_cm_cutter_gui.spec"
set "EXE_NAME=radio-cm-cutter-gui.exe"
set "ZIP_NAME=radio-cm-cutter-gui-windows.zip"

echo [1/5] Creating virtual environment...
python -m venv "%VENV_DIR%" || goto :error

echo [2/5] Installing dependencies...
"%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip || goto :error
"%VENV_DIR%\Scripts\python.exe" -m pip install -r "%REPO_ROOT%\requirements-app.txt" || goto :error

echo [3/5] Building EXE with PyInstaller...
"%VENV_DIR%\Scripts\python.exe" -m PyInstaller --clean --noconfirm --distpath "%DIST_DIR%" --workpath "%BUILD_DIR%" "%SPEC_FILE%" || goto :error

if not exist "%DIST_DIR%\%EXE_NAME%" (
  echo EXE not found: %DIST_DIR%\%EXE_NAME%
  goto :error
)

echo [4/5] Copying runtime files...
copy /Y "%REPO_ROOT%\README.md" "%DIST_DIR%\README.md" >nul || goto :error

echo [5/5] Creating distributable zip...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Compress-Archive -Path '%DIST_DIR%\%EXE_NAME%','%DIST_DIR%\README.md' -DestinationPath '%DIST_DIR%\%ZIP_NAME%' -Force" || goto :error

echo Build complete: %DIST_DIR%\%ZIP_NAME%
popd
exit /b 0

:error
echo Build failed.
popd
exit /b 1
