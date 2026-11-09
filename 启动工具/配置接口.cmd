@echo off
setlocal
cd /d "%~dp0.."
where pythonw.exe >nul 2>nul
if errorlevel 1 goto console
start "" pythonw.exe "%~dp0..\image-eval-review\configure_api.py"
exit /b

:console
python "%~dp0..\image-eval-review\configure_api.py"
if errorlevel 1 pause
