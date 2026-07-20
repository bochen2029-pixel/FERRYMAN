# hydrate_weights.ps1 — RUNTIME weight self-download (the seed ships recipes, not bytes).
# QC 2026-07-13 v2 (D-06/D-07/D-08/D-14/D-01):
#   · atomic downloads (.part + rename)                 · size gate + optional sha256 (-Verify, LFS oid)
#   · disk-space preflight before anything downloads    · -ModelsDir to place weights on a big drive
#   · loud failure tally + exit 1 (never a false "done")· applies the MuseTalk layout junctions (D-01)
#   · optional repos skipped unless -IncludeOptional
# Uses the proven fallback: HF_HUB_DISABLE_XET=1 + curl.exe on resolve URLs (the hf CLI xet
# backend wedges on some boxes — see memory hf-download-wedge-fix).
# Usage:
#   powershell -File bin\hydrate_weights.ps1                         # required repos -> <HOME>\models
#   powershell -File bin\hydrate_weights.ps1 -Verify                 # + sha256 check on LFS files
#   powershell -File bin\hydrate_weights.ps1 -ModelsDir D:\FM_models # big-drive placement (then junction
#                                                                    #   <HOME>\models -> that dir)
param([string]$ModelsDir = "", [switch]$Verify, [switch]$IncludeOptional, [switch]$All)
$ErrorActionPreference = 'Stop'
$env:HF_HUB_DISABLE_XET = '1'
if ($All) { $IncludeOptional = $true }   # back-compat with the v1 flag

$HOME2 = if ($env:FERRYMAN_HOME) { $env:FERRYMAN_HOME } else { Split-Path $PSScriptRoot -Parent }
$manifest = Join-Path $HOME2 'manifests\weights.json'
if (-not (Test-Path $manifest)) { Write-Error "weights manifest not found: $manifest"; exit 1 }
$data = Get-Content $manifest -Raw | ConvertFrom-Json
$models = if ($ModelsDir) { $ModelsDir } else { Join-Path $HOME2 'models' }
New-Item -ItemType Directory -Force -Path $models | Out-Null

Write-Host "hydrate: FERRYMAN_HOME = $HOME2"
Write-Host "hydrate: models dir   = $models   (verify=$Verify, optional=$IncludeOptional)"

# ---- pass 1: enumerate every file + size; decide what's needed (size-gated skip) ----
$plan = @()
$failedRepos = @()
foreach ($r in $data.repos) {
    if ($r.optional -and -not $IncludeOptional) { Write-Host "  (optional, skipped: $($r.id))"; continue }
    try { $m = Invoke-RestMethod "https://huggingface.co/api/models/$($r.id)?blobs=true" -TimeoutSec 60 }
    catch { Write-Warning "HF API failed for $($r.id): $($_.Exception.Message)"; $failedRepos += $r.id; continue }
    foreach ($s in $m.siblings) {
        $name = $s.rfilename
        if ($r.filter -and ($name -notmatch $r.filter)) { continue }
        $size = if ($s.lfs) { [long]$s.lfs.size } elseif ($s.size) { [long]$s.size } else { 0 }
        $sha  = if ($s.lfs) { $s.lfs.oid } else { $null }
        $dest = Join-Path $models (Join-Path $r.dest ($name -replace '/', '\'))
        $have = (Test-Path $dest) -and ((Get-Item $dest).Length -eq $size)
        if ($have -and $Verify -and $sha) {
            $h = (Get-FileHash $dest -Algorithm SHA256).Hash.ToLower()
            if ($h -ne $sha) { Write-Warning "sha256 MISMATCH (will re-download): $dest"; $have = $false }
        }
        if (-not $have) {
            $plan += [pscustomobject]@{ repo = $r.id; name = $name; size = $size; sha = $sha; dest = $dest
                                        url = "https://huggingface.co/$($r.id)/resolve/main/$name`?download=true" }
        }
    }
}
$needBytes = ($plan | Measure-Object size -Sum).Sum
if (-not $needBytes) { $needBytes = 0 }
Write-Host ("hydrate: {0} file(s) to fetch, {1:N1} GB" -f $plan.Count, ($needBytes / 1GB))

# ---- disk preflight (QC D-14): refuse a download that cannot fit ----
if ($plan.Count -gt 0) {
    # REM-3.4 (R-19): measure the JUNCTION TARGET's volume, not the link's — models\ may be
    # a junction onto another drive (the recommended big-drive pattern).
    $mi = Get-Item $models
    $measurePath = if ($mi.LinkType -and $mi.Target) { @($mi.Target)[0] } else { $models }
    $driveName = (Get-Item $measurePath).PSDrive.Name
    $free = (Get-PSDrive $driveName).Free
    if ($free -lt ($needBytes * 1.1)) {
        Write-Error ("hydrate: only {0:N1} GB free on {1}: but {2:N1} GB needed (+10% headroom). " -f `
            ($free / 1GB), $driveName, ($needBytes / 1GB))
        Write-Host  "        Re-run with -ModelsDir <big-drive-path>, then junction <HOME>\models to it:"
        Write-Host  "        New-Item -ItemType Junction -Path <HOME>\models -Target <big-drive-path>"
        exit 1
    }
}

# ---- pass 2: download (atomic .part -> rename), tally failures loudly (QC D-06/D-07) ----
$failed = @()
$i = 0
foreach ($f in $plan) {
    $i++
    New-Item -ItemType Directory -Force -Path (Split-Path $f.dest) | Out-Null
    $part = "$($f.dest).part"
    if (Test-Path $part) { Remove-Item $part -Force -EA SilentlyContinue }
    Write-Host ("  [{0}/{1}] {2}  ({3:N0} MB)" -f $i, $plan.Count, $f.name, ($f.size / 1MB))
    & curl.exe -L --fail --retry 3 --retry-delay 2 --retry-all-errors -o $part $f.url
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $part)) {
        Write-Warning "curl failed: $($f.name)"; $failed += "$($f.repo)/$($f.name)"; continue
    }
    if ($f.size -gt 0 -and (Get-Item $part).Length -ne $f.size) {
        Write-Warning "size mismatch after download: $($f.name)"; $failed += "$($f.repo)/$($f.name)"
        Remove-Item $part -Force -EA SilentlyContinue; continue
    }
    if ($Verify -and $f.sha) {
        $h = (Get-FileHash $part -Algorithm SHA256).Hash.ToLower()
        if ($h -ne $f.sha) {
            Write-Warning "sha256 mismatch after download: $($f.name)"; $failed += "$($f.repo)/$($f.name)"
            Remove-Item $part -Force -EA SilentlyContinue; continue
        }
    }
    Move-Item $part $f.dest -Force
}

# ---- pass 3: layout junctions (QC D-01 — MuseTalk resolves weights via vendor/MuseTalk/models) ----
foreach ($j in $data.layout.junctions) {
    $link = Join-Path $HOME2 ($j.link -replace '/', '\')
    $target = if ($ModelsDir) { Join-Path $models ((($j.target -replace '/', '\')) -replace '^models\\', '') }
              else { Join-Path $HOME2 ($j.target -replace '/', '\') }
    if (-not (Test-Path (Split-Path $link))) {
        Write-Host "  (layout: vendor repo missing — clone vendor/ first (manifests\vendor.json), then re-run)"
        $failed += "layout:$($j.link)"
        continue
    }
    if (Test-Path $link) { continue }   # idempotent
    if (-not (Test-Path $target)) { Write-Warning "layout: junction target missing: $target"; $failed += "layout:$($j.link)"; continue }
    New-Item -ItemType Junction -Path $link -Target $target | Out-Null
    Write-Host "  layout: $($j.link) -> $target"
}

# ---- verdict ----
if ($failedRepos.Count -gt 0 -or $failed.Count -gt 0) {
    Write-Host ""
    Write-Error ("hydrate INCOMPLETE — repo listings failed: [{0}] · items failed: [{1}]" -f `
        ($failedRepos -join ', '), ($failed -join ', '))
    exit 1
}
Write-Host ""
Write-Host "hydrate: DONE (all files present at manifest sizes)."
Write-Host "hydrate: substrate (llama.cpp / GGUFs / whisper.cpp) is separate -> bin\fetch_substrate.ps1"
Write-Host "hydrate: then verify -> python src\ferryman.py doctor"
exit 0
