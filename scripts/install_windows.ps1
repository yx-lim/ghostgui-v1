$ErrorActionPreference = "Stop"

Set-Location (Join-Path $PSScriptRoot "..")

if (Get-Command py -ErrorAction SilentlyContinue) {
    $PythonExe = "py"
    $PythonArgs = @("-3")
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $PythonExe = "python"
    $PythonArgs = @()
} else {
    Write-Error "Python was not found. Install Python 3.10 or newer and re-run this script."
}

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    & $PythonExe @PythonArgs -m venv .venv
} else {
    Write-Host "Reusing existing .venv."
}

. ".\.venv\Scripts\Activate.ps1"

python scripts/check_qt_install.py --preflight
if ($LASTEXITCODE -ne 0) {
    Write-Error "Remove or move the dedicated .venv, then rerun this installer."
}

python -m pip install --no-cache-dir --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -e .
python scripts/check_qt_install.py
if ($LASTEXITCODE -ne 0) {
    Write-Error "GhostGUI did not install the lightweight Qt dependency set."
}

Write-Host ""
Write-Host "GhostGUI installed successfully."
Write-Host "Run it with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  ghostgui"
Write-Host ""
Write-Host "Or use:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1"
