$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$pythonw = Join-Path $PSScriptRoot ".venv\Scripts\pythonw.exe"
if (-not (Test-Path -LiteralPath $pythonw)) {
    throw "Окружение .venv не найдено. Сначала запустите setup_windows.ps1."
}

Start-Process `
    -FilePath $pythonw `
    -ArgumentList "-m", "app.gui" `
    -WorkingDirectory $PSScriptRoot
