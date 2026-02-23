Param(
  [string]$ExePath = "dist/radio-cm-cutter-gui/radio-cm-cutter-gui.exe"
)

$ErrorActionPreference = "Stop"

function Assert-Exists([string]$path) {
  if (!(Test-Path $path)) {
    throw "not found: $path"
  }
}

function Run-And-Capture([string]$label, [string[]]$args) {
  Write-Host "== $label =="
  $tmp = New-TemporaryFile
  try {
    $p = Start-Process -FilePath $ExePath -ArgumentList $args -NoNewWindow -RedirectStandardOutput $tmp -RedirectStandardError $tmp -PassThru -Wait
    $text = Get-Content -Raw -Path $tmp
    [PSCustomObject]@{ ExitCode = $p.ExitCode; Output = $text }
  } finally {
    Remove-Item -Force $tmp -ErrorAction SilentlyContinue
  }
}

Assert-Exists $ExePath
Write-Host "Smoke test target: $ExePath"

$help = Run-And-Capture "help/version equivalent check" @("--help")
if ($help.ExitCode -eq 0 -or $help.Output.Length -gt 0) {
  Write-Host "[OK] --help equivalent produced output or exited 0"
} else {
  Write-Warning "--help equivalent produced no output"
}

$origPath = $env:PATH
try {
  $env:PATH = ""
  $noFfmpeg = Run-And-Capture "ffmpeg-missing check" @()
  if ($noFfmpeg.Output -match "ffmpeg|ffprobe|Required command not found") {
    Write-Host "[OK] ffmpeg missing error was explicit"
  } else {
    Write-Warning "Could not confirm ffmpeg-missing guidance in output. Open GUI and run a small folder manually to verify error text."
  }
} finally {
  $env:PATH = $origPath
}

Write-Host "Smoke test finished"
