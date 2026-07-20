# fetch_substrate.ps1 — download the AUTO-FETCHABLE substrate rows (GGUFs) and print
# clear instructions for the manual rows (llama.cpp/whisper.cpp binaries, ComfyUI, node).
# QC 2026-07-13 (D-08/D-10): the substrate used to be pure tribal knowledge.
#   powershell -File bin\fetch_substrate.ps1                       # -> C:\models (or FERRYMAN_GGUF_DIR)
#   powershell -File bin\fetch_substrate.ps1 -GgufDir D:\models
param([string]$GgufDir = "")
$ErrorActionPreference = 'Stop'
$env:HF_HUB_DISABLE_XET = '1'

$HOME2 = if ($env:FERRYMAN_HOME) { $env:FERRYMAN_HOME } else { Split-Path $PSScriptRoot -Parent }
$manifest = Join-Path $HOME2 'manifests\substrate.json'
if (-not (Test-Path $manifest)) { Write-Error "substrate manifest not found: $manifest"; exit 1 }
$data = Get-Content $manifest -Raw | ConvertFrom-Json
$gguf = if ($GgufDir) { $GgufDir } elseif ($env:FERRYMAN_GGUF_DIR) { $env:FERRYMAN_GGUF_DIR } else { 'C:\models' }
New-Item -ItemType Directory -Force -Path $gguf | Out-Null
Write-Host "substrate: GGUF dir = $gguf"

$failed = @()
foreach ($g in $data.gguf) {
    $dest = Join-Path $gguf $g.file
    if ($g.manual) {
        if (Test-Path $dest) { Write-Host "  [have]   $($g.file)" }
        else { Write-Host "  [MANUAL] $($g.file) — $($g.role)" }
        continue
    }
    if (Test-Path $dest) { Write-Host "  [have]   $($g.file)"; continue }
    $url = "https://huggingface.co/$($g.hf)/resolve/main/$($g.src)?download=true"
    Write-Host "  [fetch]  $($g.file)  <-  $($g.hf)/$($g.src)"
    $part = "$dest.part"
    & curl.exe -L --fail --retry 3 --retry-delay 2 --retry-all-errors -o $part $url
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $part)) { Write-Warning "curl failed: $($g.file)"; $failed += $g.file; continue }
    Move-Item $part $dest -Force
}

Write-Host ""
Write-Host "substrate: manual items (binaries/apps — see manifests\substrate.json notes):"
foreach ($b in $data.binaries) { Write-Host "  - $($b.what)" }

if ($failed.Count -gt 0) { Write-Error "substrate INCOMPLETE: [$($failed -join ', ')]"; exit 1 }
Write-Host ""
Write-Host "substrate: auto-fetchable rows done. Set FERRYMAN_LLAMA_DIR / EARSHOT_WHISPER_DIR if not default."
exit 0
