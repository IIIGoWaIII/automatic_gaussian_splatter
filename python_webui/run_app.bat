@echo off
cd /d "%~dp0"

echo Checking for virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate

echo Installing dependencies...
pip install -r requirements.txt

echo Starting WebUI...
echo Open http://127.0.0.1:8000 in your browser.
python server.py

pause
