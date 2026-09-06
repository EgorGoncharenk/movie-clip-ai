$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

$python = Get-Command py -ErrorAction SilentlyContinue
if ($python) {
    & py -3.11 -m venv .venv
} else {
    $python = Get-Command python -ErrorAction SilentlyContinue
    if (-not $python) {
        throw "Python 3.11 не найден. Установите его и повторите запуск."
    }
    & python -m venv .venv
}

if ($LASTEXITCODE -ne 0) {
    throw "Не удалось создать виртуальное окружение."
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    throw "Не удалось обновить pip."
}

& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Не удалось установить зависимости."
}

$nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if ($nvidia) {
    Write-Host "Обнаружена NVIDIA GPU. Устанавливаю CUDA runtime для Whisper..."
    & .\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "CUDA runtime не установился; будет доступен CPU fallback."
    }
}

& .\.venv\Scripts\python.exe -m app.cli doctor
exit $LASTEXITCODE
