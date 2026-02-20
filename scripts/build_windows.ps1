param(
  [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir "..")
$VenvDir = Join-Path $RepoRoot ".venv-build"
$DistDir = Join-Path $RepoRoot "dist"
$BuildDir = Join-Path $RepoRoot "build"
$SpecFile = Join-Path $ScriptDir "radio_cm_cutter_gui.spec"
$ExeName = "radio-cm-cutter-gui.exe"
$PackageName = "radio-cm-cutter-gui-windows.zip"

Write-Host "[1/5] Creating virtual environment..."
& $PythonExe -m venv $VenvDir

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[2/5] Installing dependencies..."
& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $RepoRoot "requirements-app.txt")

Write-Host "[3/5] Building EXE with PyInstaller..."
Push-Location $RepoRoot
try {
  & $VenvPython -m PyInstaller --clean --noconfirm --distpath $DistDir --workpath $BuildDir $SpecFile
} finally {
  Pop-Location
}

$ExePath = Join-Path $DistDir $ExeName
if (-not (Test-Path $ExePath)) {
  throw "EXE not found: $ExePath"
}

Write-Host "[4/5] Copying runtime files..."
Copy-Item (Join-Path $RepoRoot "README.md") (Join-Path $DistDir "README.md") -Force

Write-Host "[5/5] Creating distributable zip..."
$ZipPath = Join-Path $DistDir $PackageName
if (Test-Path $ZipPath) {
  Remove-Item $ZipPath -Force
}
Compress-Archive -Path $ExePath, (Join-Path $DistDir "README.md") -DestinationPath $ZipPath

Write-Host "Build complete: $ZipPath"
