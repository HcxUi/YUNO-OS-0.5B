@echo off
title YUNO OS - Interactive CLI Launcher
echo ===================================================
echo        YUNO OS - Starting Interactive CLI
echo ===================================================
echo.
python -c "import os; exit(0 if os.path.exists('extracted_model') else 1)"
if %errorlevel% neq 0 (
    echo [!] Extracted model not found. Running extract_and_run.py first...
    python extract_and_run.py
)
echo.
python cli.py
pause
