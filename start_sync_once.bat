@echo off
setlocal
cd /d "%~dp0"
echo [TXT Auto Sync] Running one-time sync...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0sync_txt.ps1"
echo [TXT Auto Sync] Done.
endlocal
