@echo off
setlocal
cd /d "%~dp0"
python "%~dp0image-eval-review\launch_annotation.py"
if errorlevel 1 pause
