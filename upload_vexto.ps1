# ===== upload_vexto.ps1 (single bulk PATCH + proper 403 backoff) =====
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

# --- Whitelist: code + config only ---
$whitelist = @(
  'main.py',
  'fetcher_diagnose_full.py',
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
  'src\vexto\scoring\website_fetchers.py'
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

# --- Prepare share/ for optional local inspection ---
Write-Host "Preparing share folder..."
if (Test-Path $share) { Remove-Item $share -Recurse -Force -ErrorAction SilentlyContinue }
New-Item -ItemType Directory -Path $share | Out-Null

# --- Compute current hashes and deltas ---
$existingAbs  = @()
$currentNames = @()
$toUpload     = @()
$newHashes    = @{}

foreach ($rel in $whitelist) {
  $abs = Join-Path $root $rel
  if (-not (Test-Path $abs -PathType Leaf)) {
    Write-Host "(skip) Not found: $rel"
    continue
  }
  $existingAbs += $abs
  $name = [IO.Path]::GetFileName($abs)
  $currentNames += $name
  $hash = (Get-FileHash -LiteralPath $abs -Algorithm SHA256).Hash
  $newHashes[$name] = $hash
  if (-not $oldHashes.ContainsKey($name) -or $oldHashes[$name] -ne $hash) {
    $toUpload += @{ abs = $abs; name = $name }
  }
}

# --- Determine deletions (in cache but not local) ---
$toDelete = @()
foreach ($oldName in $oldHashes.Keys) {
  if ($currentNames -notcontains $oldName) { $toDelete += $oldName }
}

Write-Host ("Files existing locally: " + $existingAbs.Count)
Write-Host ("Changed/new to upload: " + $toUpload.Count)
Write-Host ("To delete from Gist:   " + $toDelete.Count)

if ($existingAbs.Count -eq 0) { throw "No whitelisted files found - nothing to do." }
if ($toUpload.Count -eq 0 -and $toDelete.Count -eq 0) {
  Write-Host "No changes detected. Nothing to upload or delete."
  Write-Host "== upload_vexto.ps1 done =="
  exit 0
}

# --- HTTP setup ---
$headers = @{
  "Authorization" = ("token " + $env:GH_TOKEN)
  "Accept"        = "application/vnd.github+json"
  "User-Agent"    = "vexto-sync/1.1 (PowerShell)"
}
$uri = ("https://api.github.com/gists/" + $env:GIST_ID)

function Get-BackoffSecondsFromResponse {
  param([System.Net.WebResponse]$Response)
  $fallback = 60
  if (-not $Response) { return $fallback }
  try {
    $retryAfter = $Response.Headers["Retry-After"]
    if ($retryAfter) { return [int]$retryAfter }
    $remain = $Response.Headers["X-RateLimit-Remaining"]
    $reset  = $Response.Headers["X-RateLimit-Reset"]
    if ($remain -and [int]$remain -le 0 -and $reset) {
      $nowUtc = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds()
      $delta  = [int]$reset - [int]$nowUtc
      if ($delta -gt 0 -and $delta -lt 3600) { return $delta + 2 }
    }
  } catch {}
  return $fallback
}

function Invoke-GistBulkPatch-WithBackoff {
  param(
    [Parameter(Mandatory=$true)][hashtable]$FilesMap,  # name -> @{ content="..." } or $null
    [Parameter(Mandatory=$true)][string]$Uri,
    [Parameter(Mandatory=$true)][hashtable]$Headers,
    [int]$MaxAttempts = 4
  )
  $payload = @{ files = $FilesMap } | ConvertTo-Json -Depth 7 -Compress
  for ($attempt=1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Invoke-WebRequest -Method Patch -Uri $Uri -Headers $Headers -ContentType "application/json; charset=utf-8" -Body $payload -TimeoutSec 120 | Out-Null
      return $true
    } catch {
      $resp = $_.Exception.Response
      $status = $null; if ($resp) { $status = [int]$resp.StatusCode }
      if ($status -eq 403 -and $attempt -lt $MaxAttempts) {
        $sleepSec = Get-BackoffSecondsFromResponse -Response $resp
        Write-Host ("403 received. Backing off {0}s (attempt {1}/{2})..." -f $sleepSec,$attempt,$MaxAttempts)
        Start-Sleep -Seconds $sleepSec
        continue
      }
      Write-Host ("Bulk PATCH failed (attempt {0}/{1}): {2}" -f $attempt,$MaxAttempts,$_.Exception.Message)
      return $false
    }
  }
  return $false
}

# --- Build single bulk payload (changed + deletions) ---
$filesMap = @{}

# changed/new
foreach ($it in $toUpload) {
  $content = Get-Content -LiteralPath $it.abs -Raw -Encoding UTF8 -ErrorAction Stop
  if ($null -eq $content) { $content = "" }
  $filesMap[$it.name] = @{ content = $content }
  Copy-Item $it.abs $share -Force
}

# deletions
foreach ($name in $toDelete) {
  $filesMap[$name] = $null
}

Write-Host ("Bulk uploading: upserts=" + $toUpload.Count + " deletes=" + $toDelete.Count)

# --- Single bulk PATCH with backoff ---
if (Invoke-GistBulkPatch-WithBackoff -FilesMap $filesMap -Uri $uri -Headers $headers -MaxAttempts 4) {
  # Update cache on success
  foreach ($del in $toDelete) { if ($newHashes.ContainsKey($del)) { $newHashes.Remove($del) | Out-Null } }
  $cacheOut = @{ hashes = @{} }
  foreach ($k in $newHashes.Keys) { $cacheOut.hashes[$k] = $newHashes[$k] }
  ($cacheOut | ConvertTo-Json -Depth 6 -Compress) | Set-Content -Encoding UTF8 $cachePath
  Write-Host "Cache updated."
  Write-Host "Summary: uploaded OK=$($toUpload.Count) | deleted OK=$($toDelete.Count)"
  Write-Host "== upload_vexto.ps1 done =="
  exit 0
} else {
  Write-Host "Cache NOT updated due to failures."
  throw "Bulk upload failed. See messages above."
}
