@echo off
setlocal
cd /d "%~dp0"
python "%~dp0image-eval-review\open_demo.py"
if errorlevel 1 pause
