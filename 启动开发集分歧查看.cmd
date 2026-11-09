@echo off
setlocal
cd /d "%~dp0"
python "%~dp0image-eval-review\launch_review.py"
if errorlevel 1 pause
