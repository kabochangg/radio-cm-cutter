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
$IssFile = Join-Path $RepoRoot "installer\RadioCmCutter.iss"
$InstallerOutput = Join-Path $RepoRoot "installer_out\RadioCmCutterSetup.exe"

$VersionFile = Join-Path $RepoRoot "radio_cm_cutter_app\__init__.py"
$VersionMatch = Select-String -Path $VersionFile -Pattern '__version__\s*=\s*"(?<version>[0-9]+\.[0-9]+\.[0-9]+(?:[-+A-Za-z0-9\.]*)?)"' | Select-Object -First 1
if (-not $VersionMatch) {
  throw "アプリのバージョンが取得できませんでした: $VersionFile"
}
$AppVersion = $VersionMatch.Matches[0].Groups['version'].Value

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

$Iscc = Get-Command iscc.exe -ErrorAction SilentlyContinue
if (-not $Iscc) {
  $ProgramFilesX86 = ${env:ProgramFiles(x86)}
  if ($ProgramFilesX86) {
    $Candidate = Join-Path $ProgramFilesX86 "Inno Setup 6\ISCC.exe"
    if (Test-Path $Candidate) {
      $Iscc = @{ Source = $Candidate }
    }
  }
}

if (-not $Iscc) {
  throw @"
Inno Setup (ISCC.exe) が見つかりません。
- Inno Setup 6 をインストールしてください: https://jrsoftware.org/isdl.php
- インストール後に PowerShell を再起動して、再度 scripts/build_installer.ps1 を実行してください。
"@
}

Write-Host "[4/5] Building installer with Inno Setup..."
& $Iscc.Source "/DAppVersion=$AppVersion" $IssFile

if (-not (Test-Path $InstallerOutput)) {
  throw "インストーラが生成されませんでした: $InstallerOutput"
}

Write-Host "[5/5] Build complete: $InstallerOutput"
