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

$PythonVersion = & $PythonExe @PythonArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
& $PythonExe @PythonArgs -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Error "GhostGUI requires Python 3.10 or newer. Found Python $PythonVersion."
}
Write-Host "Using Python $PythonVersion."

if (-not (Test-Path ".venv")) {
    Write-Host "Creating .venv..."
    & $PythonExe @PythonArgs -m venv .venv
} else {
    Write-Host "Reusing existing .venv."
}

. ".\.venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .

Write-Host ""
Write-Host "GhostGUI installed successfully."
Write-Host "Run it with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  ghostgui"
Write-Host ""
Write-Host "Or use:"
Write-Host "  powershell -ExecutionPolicy Bypass -File scripts/run_windows.ps1"
