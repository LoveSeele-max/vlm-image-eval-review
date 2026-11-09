@echo off
setlocal
cd /d "%~dp0.."
python "%~dp0..\image-eval-review\launch_holdout_review.py"
if errorlevel 1 pause
