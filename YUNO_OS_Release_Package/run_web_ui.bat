@echo off
title YUNO OS - Web Application Launcher
echo ===================================================
echo        YUNO OS - Starting Web Interface
echo ===================================================
echo.
python -c "import os; exit(0 if os.path.exists('extracted_model') else 1)"
if %errorlevel% neq 0 (
    echo [!] Extracted model not found. Running extract_and_run.py first...
    python extract_and_run.py
)
echo.
echo Launching Streamlit App...
python -m streamlit run app.py
pause
