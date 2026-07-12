# make_manifest.ps1 — regenerate download_manifest.html at the repo root.
# Lists every required model file with a direct HF resolve URL + exact destination path.
# Files already complete on disk (size matches the API) are marked DONE and sorted last.
# Run any time: powershell -File <FERRYMAN_HOME>\bin\make_manifest.ps1

# Resolve the repo root from this script's own location (bin\ -> parent), or FERRYMAN_HOME.
$FerrymanHome = if ($env:FERRYMAN_HOME) { $env:FERRYMAN_HOME } else { Split-Path -Parent $PSScriptRoot }

$repos = @(
    @{ id = 'IndexTeam/IndexTTS-2';                  dest = 'indextts2' },
    @{ id = 'FunAudioLLM/Fun-CosyVoice3-0.5B-2512';  dest = 'cosyvoice3' },
    @{ id = 'FireRedTeam/FireRedASR2-AED';           dest = 'fireredasr2-aed' },
    @{ id = 'TMElyralab/MuseTalk';                   dest = 'musetalk' },
    @{ id = 'stabilityai/sd-vae-ft-mse';             dest = 'sd-vae-ft-mse' },
    @{ id = 'KwaiVGI/LivePortrait';                  dest = 'liveportrait' },
    @{ id = 'ByteDance/LatentSync-1.6';              dest = 'latentsync16' },
    @{ id = 'Systran/faster-whisper-large-v3';       dest = 'faster-whisper-large-v3' },
    @{ id = 'stabilityai/sd-turbo';                  dest = 'sd-turbo'; filter = '(fp16\.safetensors|\.json|\.txt)$' }
)

$rows = @()
$neededBytes = 0; $doneBytes = 0
foreach ($r in $repos) {
    try { $m = Invoke-RestMethod "https://huggingface.co/api/models/$($r.id)?blobs=true" -TimeoutSec 30 } catch { $rows += [pscustomobject]@{repo=$r.id; file="(API ERROR: $($_.Exception.Message))"; mb=0; status='ERROR'; url=''; dest=''}; continue }
    foreach ($s in $m.siblings) {
        $name = $s.rfilename
        if ($r.filter -and ($name -notmatch $r.filter)) { continue }
        $size = if ($s.lfs) { $s.lfs.size } else { $s.size }
        if (-not $size) { $size = 0 }
        $local = "$FerrymanHome\models\$($r.dest)\$($name -replace '/','\')"
        $have = (Test-Path $local) -and ((Get-Item $local -ErrorAction SilentlyContinue).Length -eq $size)
        if ($have) { $doneBytes += $size } else { $neededBytes += $size }
        $rows += [pscustomobject]@{
            repo = $r.id; file = $name; mb = [math]::Round($size/1MB,1)
            status = $(if ($have) {'DONE'} else {'NEEDED'})
            url = "https://huggingface.co/$($r.id)/resolve/main/$name`?download=true"
            dest = $local
        }
    }
}

$stamp = (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')
$html = @"
<!DOCTYPE html><html><head><meta charset="utf-8"><title>FERRYMAN weight manifest</title>
<style>body{font-family:Segoe UI,sans-serif;background:#111;color:#ddd;margin:2em}
h2{color:#8ecaff;margin-top:1.6em} a{color:#7fd77f} td,th{padding:4px 10px;text-align:left;font-size:14px}
tr.done{opacity:.35} .path{font-family:consolas;font-size:12px;color:#aaa}
.badge{padding:1px 8px;border-radius:8px;font-size:12px} .n{background:#5a2222} .d{background:#1f4d1f}</style></head><body>
<h1>FERRYMAN — model weight manifest</h1>
<p>Generated $stamp · <b>still needed: $([math]::Round($neededBytes/1GB,2)) GB</b> · already on disk: $([math]::Round($doneBytes/1GB,2)) GB</p>
<p>Manual procedure: click a NEEDED link (Chrome downloads it), then move the file to the exact path in the last column
(create subfolders if the filename contains a subfolder). Re-run <span class="path">$FerrymanHome\bin\make_manifest.ps1</span> to refresh statuses.</p>
"@
foreach ($g in ($rows | Group-Object repo)) {
    $html += "<h2>$($g.Name)</h2><table><tr><th>file</th><th>MB</th><th>status</th><th>destination</th></tr>"
    foreach ($row in ($g.Group | Sort-Object status -Descending)) {
        $cls = if ($row.status -eq 'DONE') {'done'} else {''}
        $badge = if ($row.status -eq 'DONE') {'d'} else {'n'}
        $html += "<tr class='$cls'><td><a href='$($row.url)'>$($row.file)</a></td><td>$($row.mb)</td><td><span class='badge $badge'>$($row.status)</span></td><td class='path'>$($row.dest)</td></tr>"
    }
    $html += "</table>"
}
$html += "</body></html>"
$manifestPath = Join-Path $FerrymanHome 'download_manifest.html'
Set-Content -Path $manifestPath -Value $html -Encoding utf8
$needed = ($rows | Where-Object status -eq 'NEEDED' | Measure-Object).Count
$done = ($rows | Where-Object status -eq 'DONE' | Measure-Object).Count
"manifest written: $manifestPath  |  NEEDED: $needed files ($([math]::Round($neededBytes/1GB,2)) GB)  |  DONE: $done files ($([math]::Round($doneBytes/1GB,2)) GB)"
