@echo off
cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_gui.ps1"
if errorlevel 1 (
    echo Failed to start Movie Clip AI.
    pause
)
