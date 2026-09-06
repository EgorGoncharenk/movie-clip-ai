$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Окружение .venv не найдено. Сначала запустите setup_windows.ps1."
}

& $python -m app.cli @args
exit $LASTEXITCODE
