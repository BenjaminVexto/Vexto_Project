# ===== upload_vexto.ps1 (preflight + chunked + per-file fallback + legacy cleanup) =====
if ($env:SKIP_GIST -eq '1') { Write-Host "Skipping Gist sync (SKIP_GIST=1)"; exit 0 }

$ErrorActionPreference = 'Stop'
try { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 } catch {}

# Paths
$root      = (Resolve-Path .).Path
$share     = Join-Path $root "share"
$cachePath = Join-Path $root ".gist_hash_cache.json"

Write-Host "== upload_vexto.ps1 start =="

# --- Autoload .env if GH_TOKEN/GIST_ID are missing ---
$dotenv = Join-Path $root ".env"
if ((-not $env:GH_TOKEN -or -not $env:GIST_ID) -and (Test-Path $dotenv)) {
  Write-Host "Loading .env from project root..."
  Get-Content $dotenv | ForEach-Object {
    if ($_ -match '^\s*GH_TOKEN\s*=\s*"?([^"#]+)"?\s*$') { $env:GH_TOKEN = $Matches[1].Trim() }
    elseif ($_ -match '^\s*GIST_ID\s*=\s*"?([^"#]+)"?\s*$') { $env:GIST_ID = $Matches[1].Trim() }
  }
}
Write-Host ("GH_TOKEN set: " + [bool]$env:GH_TOKEN)
Write-Host ("GIST_ID  set: " + [bool]$env:GIST_ID)
if (-not $env:GH_TOKEN) { throw "GH_TOKEN missing. Put it in .env as GH_TOKEN=ghp_xxx" }
if (-not $env:GIST_ID)  { throw "GIST_ID missing. Put it in .env as GIST_ID=xxxxxxxx..." }

# Normalize GIST_ID if user pasted a full URL
if ($env:GIST_ID -match 'gist\.github\.com\/[^\/]+\/([0-9a-f]+)') { $env:GIST_ID = $Matches[1] }

# --- Whitelist ---
$whitelist = @(
  'main.py',
  'fetcher_diagnose_full.py',
  'test_contact_finder.py',
  'config\scoring_rules.yml',
  'src\vexto\scoring\analyzer.py',
  'src\vexto\scoring\authority_fetcher.py',
  'src\vexto\scoring\authority_score.py',
  'src\vexto\scoring\contact_fetchers.py',
  'src\vexto\scoring\content_fetcher.py',
  'src\vexto\scoring\crawler.py',
  'src\vexto\scoring\cta_fetcher.py',
  'src\vexto\scoring\form_fetcher.py',
  'src\vexto\scoring\gmb_fetcher.py',
  'src\vexto\scoring\http_client.py',
  'src\vexto\scoring\image_fetchers.py',
  'src\vexto\scoring\link_fetcher.py',
  'src\vexto\scoring\log_utils.py',
  'src\vexto\scoring\performance_fetcher.py',
  'src\vexto\scoring\privacy.py',
  'src\vexto\scoring\privacy_fetchers.py',
  'src\vexto\scoring\robots_sitemap_fetcher.py',
  'src\vexto\scoring\schemas.py',
  'src\vexto\scoring\scorer.py',
  'src\vexto\scoring\security_fetchers.py',
  'src\vexto\scoring\social_fetchers.py',
  'src\vexto\scoring\technical_fetchers.py',
  'src\vexto\scoring\tracking_fetchers.py',
  'src\vexto\scoring\trust_signal_fetcher.py',
  'src\vexto\scoring\url_finder.py',
  'src\vexto\scoring\website_fetchers.py',
  'src\vexto\enrichment\contact_finder.py',
  'src\vexto\data_processing\data_cleaner.py'
)

# --- Load old hash cache ---
$oldHashes = @{}
if (Test-Path $cachePath) {
  try {
    $cacheObj = Get-Content $cachePath -Raw | ConvertFrom-Json
    if ($cacheObj -and $cacheObj.hashes) {
      foreach ($prop in $cacheObj.hashes.PSObject.Properties) { $oldHashes[$prop.Name] = $prop.Value }
    }
  } catch { Write-Host "Cache unreadable. Rebuilding..." }
}

# --- Prepare share/ ---
Write-Host "Preparing share folder..."
if (Test-Path $share) { Remove-Item $share -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Path $share | Out-Null

# --- Compute deltas ---
$existingAbs  = @()
$currentNames = @()
$toUpload     = @()
$newHashes    = @{}
$script:LegacyList = @()

foreach ($rel in $whitelist) {
  $abs = Join-Path $root $rel
  if (-not (Test-Path $abs -PathType Leaf)) { Write-Host "(skip) Not found: $rel"; continue }

  $existingAbs += $abs
  $relKey  = ($rel -replace '\\','/')

  # Unik Gist-nøgle for ALLE filer (full path -> underscores)
  $gistKey = ($rel -replace '[\\/]', '_')                # fx src_vexto_scoring_image_fetchers.py

  # Legacy-navne der kan ligge på Gist
  $legacyKey  = ($rel -replace '[\\/]', '__')            # gammelt dobbelt-underscore-navn
  $legacyBase = [IO.Path]::GetFileName($rel)             # gammelt rent basenavn (fx image_fetchers.py)

  $currentNames += $gistKey
  $hash = (Get-FileHash -LiteralPath $abs -Algorithm SHA256).Hash
  $newHashes[$gistKey] = $hash

  if (-not $oldHashes.ContainsKey($gistKey) -or $oldHashes[$gistKey] -ne $hash) {
    $toUpload += @{ abs = $abs; relKey = $relKey; key = $gistKey }
  }

  # Saml legacy-navne til oprydning
  if ($legacyKey -and $script:LegacyList -notcontains $legacyKey) { $script:LegacyList += $legacyKey }
  if ($gistKey -ne $legacyBase -and $script:LegacyList -notcontains $legacyBase) { $script:LegacyList += $legacyBase }
}

# --- Determine deletions from cache ---
$toDelete = @()
foreach ($oldName in $oldHashes.Keys) {
  if ($currentNames -notcontains $oldName) { $toDelete += $oldName }
}
if ($script:LegacyList) {
  foreach ($legacy in $script:LegacyList) {
    if ($oldHashes.ContainsKey($legacy) -and $toDelete -notcontains $legacy) { $toDelete += $legacy }
  }
}

# --- HTTP setup (Bearer) ---
$headers = @{
  "Authorization" = ("Bearer " + $env:GH_TOKEN)
  "Accept"        = "application/vnd.github+json"
  "User-Agent"    = "vexto-sync/1.2 (PowerShell)"
}
$uri = ("https://api.github.com/gists/" + $env:GIST_ID)

# --- Preflight: harmless description patch ---
try {
  $diag = @{ description = "vexto-sync preflight $(Get-Date -Format o)" } | ConvertTo-Json -Depth 5 -Compress
  Invoke-WebRequest -Method PATCH -Uri $uri -Headers $headers -ContentType "application/json; charset=utf-8" -Body $diag -TimeoutSec 20 | Out-Null
  Write-Host "Preflight OK (write access confirmed)."
} catch {
  $msg = $_.Exception.Message; throw "Auth/preflight failed: $msg"
}

# --- Live Gist cleanup: mark legacy files present on remote for deletion ---
$gistMeta = $null
try { $resp = Invoke-WebRequest -Method GET -Uri $uri -Headers $headers -TimeoutSec 20; $gistMeta = $resp.Content | ConvertFrom-Json } catch {}
if ($gistMeta -and $script:LegacyList) {
  foreach ($legacy in $script:LegacyList) {
    if ($gistMeta.files.PSObject.Properties[$legacy]) {
      if ($toDelete -notcontains $legacy) { $toDelete += $legacy }
    }
  }
}

# --- Normalize deletions: don't delete keys we're updating (prevents 422) ---
$uploadKeys = @($toUpload | ForEach-Object { $_.key })
if ($uploadKeys.Count -gt 0 -and $toDelete.Count -gt 0) {
  $beforeDel = $toDelete.Count
  $toDelete  = $toDelete | Where-Object { $uploadKeys -notcontains $_ }
  if ($toDelete.Count -lt $beforeDel) {
    Write-Host ("Normalized deletions: removed {0} collisions with upserts." -f ($beforeDel - $toDelete.Count))
  }
}

Write-Host ("Files existing locally: " + $existingAbs.Count)
Write-Host ("Changed/new to upload: " + $toUpload.Count)
Write-Host ("To delete from Gist:   " + $toDelete.Count)
if ($existingAbs.Count -eq 0) { throw "No whitelisted files found - nothing to do." }
if ($toUpload.Count -eq 0 -and $toDelete.Count -eq 0) { Write-Host "No changes detected. Nothing to upload or delete."; Write-Host "== upload_vexto.ps1 done =="; exit 0 }

# -------------------- Helpers --------------------
function Get-BackoffSecondsFromResponse {
  param([System.Net.WebResponse]$Response)
  $fallback = 60
  if (-not $Response) { return $fallback }
  try {
    $retryAfter = $Response.Headers["Retry-After"]; if ($retryAfter) { return [int]$retryAfter }
    $remain = $Response.Headers["X-RateLimit-Remaining"]; $reset = $Response.Headers["X-RateLimit-Reset"]
    if ($remain -and [int]$remain -le 0 -and $reset) {
      $nowUtc = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds(); $delta  = [int]$reset - [int]$nowUtc
      if ($delta -gt 0 -and $delta -lt 3600) { return $delta + 2 }
    }
  } catch {}
  return $fallback
}

# return $true on 2xx; $false otherwise; writes error body if any
function Invoke-GistPatch($filesMap, $uri, $headers, $maxAttempts=4) {
  $payload = @{ files = $filesMap } | ConvertTo-Json -Depth 100 -Compress
  for ($attempt=1; $attempt -le $maxAttempts; $attempt++) {
    try {
      Invoke-WebRequest -Method PATCH -Uri $uri -Headers $headers -ContentType "application/json; charset=utf-8" -Body $payload -TimeoutSec 120 | Out-Null
      return $true
    } catch {
      $resp = $_.Exception.Response; $status = $null
      if ($resp) {
        try { $status = [int]$resp.StatusCode } catch {}
        $body = $null
        try { $reader = New-Object System.IO.StreamReader($resp.GetResponseStream()); $body = $reader.ReadToEnd() } catch {}
        if ($body) { Write-Host "GitHub error body (status=$status):"; Write-Host $body } else { Write-Host ("GitHub returned status {0} with empty body." -f $status) }
      }
      if ($status -eq 403 -and $attempt -lt $maxAttempts) {
        $sleepSec = Get-BackoffSecondsFromResponse -Response $resp
        Write-Host ("403 received. Backing off {0}s (attempt {1}/{2})..." -f $sleepSec,$attempt,$maxAttempts); Start-Sleep -Seconds $sleepSec; continue
      }
      Write-Host ("Bulk PATCH failed (attempt {0}/{1}): {2}" -f $attempt,$maxAttempts,$_.Exception.Message)
      return $false
    }
  }
  return $false
}

function Get-ApproxUtf8Size($s) { if ($null -eq $s) { return 0 } ([Text.Encoding]::UTF8.GetByteCount($s)) }

function Split-FilesMapBySize {
  param([hashtable]$FilesMap, [int]$maxBytes=700000, [int]$maxFiles=6)
  $chunks = @(); $current=@{}; $curSize=0; $curCount=0
  foreach ($k in $FilesMap.Keys) {
    $entry = $FilesMap[$k]
    $addSize = 100 + [Text.Encoding]::UTF8.GetByteCount($k)
    if ($null -ne $entry -and $entry.ContainsKey("content")) { $addSize += Get-ApproxUtf8Size($entry["content"]) }
    if (($curCount -ge $maxFiles) -or ($curSize + $addSize -gt $maxBytes)) {
      if ($current.Count -gt 0) { $chunks += ,$current; $current=@{}; $curSize=0; $curCount=0 }
    }
    $current[$k] = $entry; $curSize += $addSize; $curCount++
  }
  if ($current.Count -gt 0) { $chunks += ,$current }
  return ,$chunks
}

function Invoke-GistBulkPatch-Chunked {
  param([hashtable]$FilesMap, [string]$Uri, [hashtable]$Headers)
  $parts = Split-FilesMapBySize -FilesMap $FilesMap -maxBytes 700000 -maxFiles 6
  $idx=0
  foreach ($part in $parts) {
    $idx++
    Write-Host ("Uploading chunk {0}/{1} ({2} files)..." -f $idx, $parts.Count, $part.Keys.Count)
    Write-Host "Chunk $idx filer:"; $part.Keys | ForEach-Object { Write-Host " - $_" }
    if (-not (Invoke-GistPatch -filesMap $part -uri $Uri -headers $Headers -maxAttempts 4)) { throw "CHUNK_422" }
    Start-Sleep -Milliseconds 400
  }
}

# --- Helper: sanitize file content (strip NUL + control chars except CR/LF/TAB) ---
function Get-SafeUtf8([string]$path) {
  $bytes = [System.IO.File]::ReadAllBytes($path)
  $bytes = $bytes | Where-Object { $_ -ne 0 }
  $text  = [Text.Encoding]::UTF8.GetString($bytes)
  return ($text -replace '[\x00-\x08\x0B\x0C\x0E-\x1F]', '')
}

function Rename-And-Update-SingleFile {
  param(
    [string]$Uri, [hashtable]$Headers,
    [string]$TargetKey, [string]$LocalAbsPath
  )
  # TargetKey is the desired Gist filename (e.g., src_vexto_scoring_image_fetchers.py)
  $base = [IO.Path]::GetFileName($LocalAbsPath)
  $content = Get-SafeUtf8 $LocalAbsPath
  if ($null -eq $content) { $content = "" }

  # Inline metadata fetch (no external func dependency)
  $gist = $null
  try { $resp = Invoke-WebRequest -Method GET -Uri $Uri -Headers $Headers -TimeoutSec 20; $gist = $resp.Content | ConvertFrom-Json } catch { return $false }

  $hasBase = $false
  if ($gist.files.PSObject.Properties[$base]) { $hasBase = $true }

  if ($hasBase) {
    # 1-step: rename + content
    $payload = @{ files = @{ $base = @{ filename = $TargetKey; content = $content } } } | ConvertTo-Json -Depth 100 -Compress
    try { Invoke-WebRequest -Method PATCH -Uri $Uri -Headers $Headers -ContentType "application/json; charset=utf-8" -Body $payload -TimeoutSec 60 | Out-Null; return $true }
    catch {
      # 2-step: rename -> content
      try {
        $payloadRename  = @{ files = @{ $base      = @{ filename = $TargetKey } } } | ConvertTo-Json -Depth 10 -Compress
        Invoke-WebRequest -Method PATCH -Uri $Uri -Headers $Headers -ContentType "application/json; charset=utf-8" -Body $payloadRename  -TimeoutSec 60 | Out-Null
        $payloadContent = @{ files = @{ $TargetKey = @{ content  = $content } } } | ConvertTo-Json -Depth 10 -Compress
        Invoke-WebRequest -Method PATCH -Uri $Uri -Headers $Headers -ContentType "application/json; charset=utf-8" -Body $payloadContent -TimeoutSec 60 | Out-Null
        return $true
      } catch { return $false }
    }
  } else {
    # no root-base file present -> plain single upload
    $payload = @{ files = @{ $TargetKey = @{ content = $content } } } | ConvertTo-Json -Depth 100 -Compress
    try { Invoke-WebRequest -Method PATCH -Uri $Uri -Headers $Headers -ContentType "application/json; charset=utf-8" -Body $payload -TimeoutSec 60 | Out-Null; return $true }
    catch { return $false }
  }
}

# --- Build payload ---
# 1) dedup by key (scriptblock to avoid Hashtable 'Keys' collision)
$toUpload = $toUpload | Group-Object -Property { $_.key } | ForEach-Object { $_.Group[0] }

# 2) filesMap (updates) + deletions
$filesMap = @{}
foreach ($it in $toUpload) {
  $content = Get-SafeUtf8 $it.abs
  if ($null -eq $content) { $content = "" }
  $filesMap[$it.key] = @{ content = $content }

  # copy to share\... preserving folders locally
  $dest = Join-Path $share ($it.relKey -replace '/','\')
  $destDir = Split-Path $dest -Parent
  if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Path $destDir | Out-Null }
  Copy-Item $it.abs $dest -Force
}
foreach ($name in $toDelete) { $filesMap[$name] = $null }

Write-Host ("Uploading in chunks: upserts=" + $filesMap.Keys.Count + " deletes=" + $toDelete.Count)

# --- Upload: chunked -> per-file fallback (with rename) ---
$chunkOk = $true
try { Invoke-GistBulkPatch-Chunked -FilesMap $filesMap -Uri $uri -Headers $headers }
catch { if ($_.Exception.Message -eq 'CHUNK_422') { $chunkOk = $false } else { throw } }

if (-not $chunkOk) {
  Write-Host "Chunked upload failed with 422. Falling back to per-file mode..."
  foreach ($k in $filesMap.Keys) {
    $single = @{}; $single[$k] = $filesMap[$k]
    Write-Host ("[single] Patching {0} ..." -f $k)
    if (-not (Invoke-GistPatch -filesMap $single -uri $uri -headers $headers -maxAttempts 4)) {
      Write-Host "Single PATCH failed for $k - trying rename/update fallback..."
      $localAbs = $null
      foreach ($it in $toUpload) { if ($it.key -eq $k) { $localAbs = $it.abs; break } }
      if ($null -eq $localAbs -or -not (Rename-And-Update-SingleFile -Uri $uri -Headers $headers -TargetKey $k -LocalAbsPath $localAbs)) {
        throw "Single-file patch failed for: $k"
      } else {
        Write-Host "Fallback succeeded for $k"
      }
    }
    Start-Sleep -Milliseconds 250
  }
}

# --- Update cache on success ---
foreach ($del in $toDelete) { if ($newHashes.ContainsKey($del)) { $newHashes.Remove($del) | Out-Null } }
$cacheOut = @{ hashes = @{} }
foreach ($k in $newHashes.Keys) { $cacheOut.hashes[$k] = $newHashes[$k] }
($cacheOut | ConvertTo-Json -Depth 6 -Compress) | Set-Content -Encoding UTF8 $cachePath
Write-Host "Cache updated."
Write-Host "Summary: uploaded OK=$($toUpload.Count) | deleted OK=$($toDelete.Count)"
Write-Host "== upload_vexto.ps1 done =="
