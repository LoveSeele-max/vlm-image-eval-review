@echo off
setlocal
cd /d "%~dp0"
where pythonw.exe >nul 2>nul
if errorlevel 1 goto console
start "" pythonw.exe "%~dp0image-eval-review\launch_dev_v1.py"
exit /b

:console
python "%~dp0image-eval-review\launch_dev_v1.py"
if errorlevel 1 pause
