# Activate the benchmark environment.
#   . d:\UAV_VisualSurv\scripts\setup\env.ps1
#
# Keeps every large artefact on D:. The C: drive has ~40 GB free and must not
# accumulate model weights or dataset archives, which is what the default
# %USERPROFILE%\.cache locations would do.

$env:UAV_ROOT      = 'd:\UAV_VisualSurv'
$env:HF_HOME       = "$env:UAV_ROOT\.cache\huggingface"
$env:TORCH_HOME    = "$env:UAV_ROOT\.cache\torch"
$env:PIP_CACHE_DIR = "$env:UAV_ROOT\.cache\pip"

# pip stages every wheel through TMP *before* it reaches the cache, and TMP is a
# separate setting from PIP_CACHE_DIR. Redirecting only the cache still puts a
# full copy of each wheel on C: -- the torch wheel alone is ~2.4 GB. Scoped to
# this process, not set user-wide, so other applications are unaffected.
$env:TMP  = "$env:UAV_ROOT\.cache\tmp"
$env:TEMP = "$env:UAV_ROOT\.cache\tmp"

# Fail loudly rather than silently falling back to a C: cache.
foreach ($d in $env:HF_HOME, $env:TORCH_HOME, $env:PIP_CACHE_DIR, $env:TMP) {
    if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
}

& "$env:UAV_ROOT\.venv\Scripts\Activate.ps1"

Write-Host "UAV_VisualSurv environment active" -ForegroundColor Green
Write-Host "  python  : $((Get-Command python).Source)"
Write-Host "  caches  : $env:UAV_ROOT\.cache  (D: drive)"
$free = (Get-PSDrive D).Free / 1GB
Write-Host ("  D: free : {0:N1} GB" -f $free)
